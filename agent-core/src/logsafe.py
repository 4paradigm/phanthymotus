"""Atomic, control-character-safe stdout for containerised processes.

Docker's log drivers frame each write into a record: `json-file` wraps it in
JSON, `local` wraps it in a length-prefixed protobuf. Two things break that
framing, and both leave `docker logs` unable to render anything at all:

1. **Torn records.** A write to a pipe is only atomic up to `PIPE_BUF`
   (4096 bytes on Linux). Our containers write to stdout from many threads at
   once — a ROS `MultiThreadedExecutor`, a `ThreadingHTTPServer` handler per
   request, background registration loops — so two long lines can interleave
   mid-record. The reader then sees a fragment where a record header should be.

2. **Control bytes.** NUL and ESC inside a message are passed through to the
   framer. NUL breaks the JSON parser outright; ESC sequences render as the
   "garbage characters" a human sees when the stream is not a terminal.

This module installs a `sys.stdout` replacement that makes both impossible:
every completed line is emitted as exactly one `os.write()`, capped below
`PIPE_BUF`, with C0 control characters removed.

Install it as the *first* thing a process does, and again at the top of every
`multiprocessing` child entry point — a spawned child gets a fresh interpreter
and does not inherit the parent's `sys.stdout` object.

    import logsafe; logsafe.install()          # agent-core, drivers
    from utils import logsafe; logsafe.install()  # perception

`sys.stderr` is wrapped too, by default — see install(). Tracebacks survive:
each line becomes its own atomic record, which is how `docker logs` frames them
anyway. Pass `stderr=False` to opt out.

NOTE ON SCOPE: this is a safety net, not a root-cause fix. It prevents *Python*
writers from producing torn or control-laden records. It does not repair a log
file that was corrupted out from under the daemon — e.g. by truncating a live
container's log file, which resets the file size but not the daemon's write
offset and leaves a NUL hole. Don't do that.

DUPLICATION: three byte-identical copies exist, because the three Docker build
contexts cannot see a common directory (`agent-core/`, `perception/`, and the
driver repo are separate contexts):

    phanthymotus/agent-core/src/logsafe.py
    phanthymotus/perception/utils/logsafe.py
    phanthymotus-driver/common/logsafe.py

Keep them identical; `sha256sum` all three when changing one.
"""

from __future__ import annotations

import io
import os
import re
import select
import sys
import threading

# A write to a pipe is only atomic up to PIPE_BUF, and that value is
# platform-dependent: 4096 on Linux, 512 on macOS. Query it rather than
# hardcoding, so the guarantee holds wherever this runs.
_MARKER_HEADROOM = 48  # truncation marker + newline


def _detect_pipe_buf(fd: int = 1) -> int:
    try:
        n = os.fpathconf(fd, 'PC_PIPE_BUF')
        if n and int(n) > 0:
            return int(n)
    except (OSError, ValueError, AttributeError):
        pass
    return int(getattr(select, 'PIPE_BUF', 4096) or 4096)


PIPE_BUF = _detect_pipe_buf()
MAX_LINE_BYTES = max(256, PIPE_BUF - _MARKER_HEADROOM)

# Strip C0 controls except tab and newline, plus DEL. Carriage return goes too:
# it makes `docker logs` output overwrite itself.
_SCRUB = {c: None for c in range(0x20) if c not in (0x09, 0x0A)}
_SCRUB[0x7F] = None

# Remove whole ANSI escape sequences, not just the ESC byte — dropping ESC alone
# would leave the parameter bytes behind as literal "[31m" litter in the log.
# Covers CSI (ESC[...final) and OSC (ESC]...BEL/ST).
_ANSI = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])')

_write_lock = threading.Lock()


def scrub(text: str) -> str:
    """Remove control characters that would corrupt or garble a log record."""
    return _ANSI.sub('', text).translate(_SCRUB)


def _truncate(payload: bytes) -> bytes:
    """Cap payload at MAX_LINE_BYTES, cutting on a UTF-8 character boundary."""
    if len(payload) <= MAX_LINE_BYTES:
        return payload
    dropped = len(payload) - MAX_LINE_BYTES
    cut = payload[:MAX_LINE_BYTES]
    # Never split a multi-byte character: back off continuation bytes (0b10xxxxxx).
    while cut and (cut[-1] & 0xC0) == 0x80:
        cut = cut[:-1]
    if cut and (cut[-1] & 0xC0) == 0xC0:
        cut = cut[:-1]  # dropped a lead byte whose continuations we just cut
    return cut + f'…[truncated {dropped} bytes]'.encode('utf-8')


def _emit(fd: int, line: str) -> None:
    """Write one complete line to `fd` as a single atomic os.write."""
    payload = _truncate(scrub(line).encode('utf-8', errors='replace')) + b'\n'
    with _write_lock:
        offset = 0
        while offset < len(payload):
            try:
                offset += os.write(fd, payload[offset:])
            except (BrokenPipeError, OSError):
                return  # log consumer went away; never let logging kill the process


class LineAtomicStream(io.TextIOBase):
    """Line-buffered stream that emits each line in one atomic write.

    `print(x)` issues two writes — the text, then the newline — so buffering
    must happen here rather than relying on the caller. The buffer is
    thread-local: two threads mid-line must not splice into each other.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._local = threading.local()

    @property
    def _buf(self) -> list:
        buf = getattr(self._local, 'buf', None)
        if buf is None:
            buf = self._local.buf = []
        return buf

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            raise TypeError(f'write() argument must be str, not {type(s).__name__}')
        if not s:
            return 0
        buf = self._buf
        start = 0
        while True:
            nl = s.find('\n', start)
            if nl < 0:
                break
            buf.append(s[start:nl])
            _emit(self._fd, ''.join(buf))
            buf.clear()
            start = nl + 1
        if start < len(s):
            buf.append(s[start:])
        return len(s)

    def flush(self) -> None:
        """Emit any partial line. The fd itself is unbuffered."""
        buf = self._buf
        if buf:
            _emit(self._fd, ''.join(buf))
            buf.clear()

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._fd

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return 'utf-8'

    @property
    def errors(self) -> str:
        return 'replace'


_installed = False


def install(check_fd: bool = True, stderr: bool = True) -> None:
    """Replace sys.stdout (and by default sys.stderr) with line-atomic writers.

    Idempotent. `stderr` defaults to True because anything using the `logging`
    module writes there: measured on a robot, perception emitted 364 B on stdout
    against 39 KB on stderr, so protecting only stdout guards the wrong stream.

    When `check_fd` is set, warn if fd 1 is not what sys.stdout points at —
    a tripwire for code that shuffles file descriptors and silently detaches
    the process from the container log.
    """
    global _installed
    if _installed:
        return
    # Re-detect against the real fd 1: at import time it may not have been the
    # pipe we ultimately write to.
    global PIPE_BUF, MAX_LINE_BYTES
    PIPE_BUF = _detect_pipe_buf(1)
    MAX_LINE_BYTES = max(256, PIPE_BUF - _MARKER_HEADROOM)
    if check_fd:
        try:
            current = sys.stdout.fileno()
            if current != 1:
                sys.stderr.write(
                    f'[logsafe] warning: sys.stdout is on fd {current}, not fd 1 — '
                    f'output may bypass the container log\n'
                )
        except (AttributeError, OSError, io.UnsupportedOperation):
            pass  # already redirected to a non-fd object; nothing useful to check
    sys.stdout = LineAtomicStream(1)
    if stderr:
        sys.stderr = LineAtomicStream(2)
    _installed = True
