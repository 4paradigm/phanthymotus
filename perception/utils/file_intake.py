"""
utils/file_intake.py — Receive a file over HTTP into a service's own container.

## Why this exists

agent-core serves the browser and holds the canvas; the tools that need a file
(a photo to enrol a face, an audio clip to play) run in *other* containers. They
share no filesystem: agent-core mounts `/opt/phanthy-motus`, perception mounts
`/opt/embodied/models`, and each container's `/tmp` and `/work` are its own. So a
path produced on one side means nothing on the other — which is exactly how the
first face-enrolment attempt failed, with an `image_path` that really existed,
in agent-core.

The alternatives were worse:

* **base64 through the tool call.** Tried, and it failed in production: a 43 800
  character string does not survive being carried through an LLM's context, and
  what arrived was truncated.
* **A shared host mount.** Works, but needs every participating container
  recreated (Docker fixes mounts at creation), one host directory per pairing,
  and leaves the path ambiguous — `/uploads` in one container, something else in
  another.

Instead agent-core *proxies* the upload to the service that will read the file,
resolving the target's address from the MCP registry (every service reports
`url` when it registers, so the port is already known — including drivers, whose
ports differ). The service writes the file into a directory it can actually see
and returns **its own** absolute path, which the caller uses verbatim. One
viewpoint, no mounts, no recreation.

## The contract

    POST /file/upload
      multipart/form-data: file=<the file>, subdir=<optional>
      header: X-Access-Token: <ACCESS_TOKEN>   (when the service has one)
    → 200 {"ok": true, "path": "/models/uploads/2026-09-08/alice.jpg", "bytes": N}
    → 4xx {"ok": false, "error": "..."}

`path` is absolute *inside the receiving container*. That is the whole point.

This module is deliberately dependency-free (stdlib only) so the drivers — which
run a bare `ThreadingHTTPServer`, not FastAPI — can use the same implementation.
"""

from __future__ import annotations

import cgi
import hmac
import io
import json
import logging
import os
import re
import shutil
import time
import unicodedata

log = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_RETENTION_DAYS = 7
# Date-stamped subdirectories, so retention is a directory listing rather than a
# stat() per file, and a human reading the disk can see when something arrived.
_DATE_FORMAT = "%Y-%m-%d"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\u4e00-\u9fff-]+")


class IntakeError(Exception):
    """Rejected upload. `status` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def safe_filename(name: str, fallback: str = "upload.bin") -> str:
    """Reduce a client-supplied filename to something safe to join onto a path.

    Takes the basename (so `../../etc/passwd` and `C:\\x\\y.jpg` both collapse),
    strips anything outside a conservative allowlist, and keeps CJK because the
    people using this name their files in Chinese. NFC-normalises first so a
    macOS upload (NFD) and a Linux one produce the same name rather than two
    files that look identical.
    """
    name = unicodedata.normalize("NFC", str(name or ""))
    name = os.path.basename(name.replace("\\", "/")).strip()
    name = _SAFE_NAME.sub("_", name).lstrip(".")
    if not name or name in (".", ".."):
        return fallback
    # Bound the length: some filesystems cap a component at 255 bytes, and a CJK
    # name is 3 bytes a character.
    stem, extension = os.path.splitext(name)
    while len((stem + extension).encode("utf-8")) > 200 and stem:
        stem = stem[:-1]
    return (stem + extension) or fallback


def safe_subdir(subdir: str) -> str:
    """Validate an optional caller-chosen subdirectory: one plain component.

    Refuses anything containing a separator or a `..` rather than sanitising it.
    Sanitising turned `../escape` into `.._escape`, which is a *valid* name — so
    the write succeeded, just not where the caller asked, and nothing said so.
    For a path a caller chose, "no" is a better answer than "somewhere else".
    """
    if not subdir:
        return ""
    raw = str(subdir).strip()
    if "/" in raw or "\\" in raw or raw in (".", "..") or raw.startswith(".."):
        raise IntakeError(f"subdir must be a single plain name: {subdir!r}")
    cleaned = safe_filename(raw, fallback="")
    if not cleaned or cleaned != raw:
        raise IntakeError(f"invalid subdir: {subdir!r}")
    return cleaned


# ACCESS_TOKEN, when this service is given one. Read from the environment and
# from the shared .env if that happens to be mounted — but **absent is the
# normal case**, verified on Tianyi: the perception container has ROS_DOMAIN_ID
# and the DDS profile, no ACCESS_TOKEN, and it mounts `dds-local.xml` as a single
# file rather than the directory holding `.env`.
#
# So this endpoint is gated by *reachability*, not by a secret: it binds the same
# host-network port the MCP server already serves `tools/call` on, and anything
# that can reach it can already drive the plugin. The token check is here so that
# a deployment which does distribute one gets enforcement for free, and so the
# check is not something to remember to add later.
_ENV_CANDIDATES = ("/opt/phanthy-motus/.env", "/work/.env")


def access_token() -> str:
    """ACCESS_TOKEN for this service, or "" when it has none.

    Empty disables the check, which mirrors agent-core's own behaviour
    (`auth.init()` prints "authentication disabled" and lets everything
    through). A dev deployment therefore works untouched, and one that sets the
    variable on the perception container enforces it on both ends automatically.
    """
    from os import environ
    direct = environ.get("ACCESS_TOKEN", "").strip()
    if direct:
        return direct
    for candidate in _ENV_CANDIDATES:
        try:
            with open(candidate, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith("ACCESS_TOKEN=") and not line.startswith("#"):
                        return line.split("=", 1)[1].strip()
        except OSError:
            continue
    return ""


def check_token(provided: str | None, expected: str | None) -> None:
    """Compare an access token in constant time.

    A write-anything endpoint on a host-network container is worth gating even
    though `tools/call` next to it is not: MCP calls are bounded by the tool
    schema, an upload is bounded only by the disk. When the service has no token
    configured this is a no-op, so a development deployment still works.
    """
    if not expected:
        return
    if not provided or not hmac.compare_digest(str(provided), str(expected)):
        raise IntakeError("invalid or missing access token", status=401)


def store_stream(
    stream,
    filename: str,
    base_dir: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    subdir: str = "",
) -> dict:
    """Write `stream` under `base_dir/<date>[/subdir]/<filename>`.

    Streams in chunks and enforces `max_bytes` *while* writing, so an oversized
    body is refused without ever being held in memory or fully committed to
    disk. Writes to a temporary name and renames, so a reader that lists the
    directory never sees a partial file under its final name — the same rule
    `utils/model_downloader.py` follows for model downloads.
    """
    name = safe_filename(filename)
    day = time.strftime(_DATE_FORMAT)
    directory = os.path.join(base_dir, day, safe_subdir(subdir)) if subdir else \
        os.path.join(base_dir, day)
    os.makedirs(directory, exist_ok=True)

    final = os.path.join(directory, name)
    partial = os.path.join(directory, f".{name}.part")
    written = 0
    try:
        with open(partial, "wb") as handle:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes and written > max_bytes:
                    raise IntakeError(
                        f"file exceeds the {max_bytes} byte limit",
                        status=413,
                    )
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if written == 0:
            raise IntakeError("uploaded file is empty")
        os.replace(partial, final)
    except IntakeError:
        _unlink_quietly(partial)
        raise
    except OSError as error:
        _unlink_quietly(partial)
        # Disk-full is the failure an operator can actually act on, and the raw
        # errno message does not say which filesystem ran out.
        raise IntakeError(f"cannot write to {directory}: {error}", status=507)

    log.info("[file_intake] stored %s (%d bytes)", final, written)
    return {"ok": True, "path": final, "bytes": written, "name": name}


def parse_multipart(headers, body: bytes) -> tuple[str, io.BytesIO, dict]:
    """Pull `(filename, stream, fields)` out of a multipart/form-data body.

    `cgi.FieldStorage` rather than a hand-rolled boundary split: the drivers and
    perception have no web framework, and getting multipart parsing subtly wrong
    is a classic way to mangle binary payloads (a boundary appearing inside JPEG
    data, CRLF translation, a missing final `--`). A test in
    `tests/test_file_intake.py` pins the boundary-inside-payload case.

    The returned stream is a fresh `BytesIO`, **not** FieldStorage's own file
    object: for a body over ~1000 bytes FieldStorage spills to a
    `TemporaryFile`, and it closes that file when the FieldStorage is collected —
    which happens the moment this function returns. Handing the caller that
    object gave "ValueError: read of closed file" for every upload big enough to
    matter, and none small enough to notice.
    """
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type.lower():
        raise IntakeError(
            "expected multipart/form-data with a `file` part; "
            f"got {content_type!r}"
        )
    environ = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type,
               "CONTENT_LENGTH": str(len(body))}
    form = cgi.FieldStorage(fp=io.BytesIO(body), environ=environ,
                            headers={"content-type": content_type},
                            keep_blank_values=True)
    if "file" not in form:
        raise IntakeError("no `file` part in the request")
    item = form["file"]
    if isinstance(item, list):
        item = item[0]
    if not getattr(item, "filename", ""):
        raise IntakeError("the `file` part has no filename")
    fields = {
        key: form.getfirst(key, "")
        for key in form.keys() if key != "file"
    }
    data = item.file.read()
    return item.filename, io.BytesIO(data), fields


def prune_old(base_dir: str, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Delete date directories older than `retention_days`. Returns how many.

    Uploads are a transfer buffer, not storage: a face enrolled from a photo
    keeps its *embedding*, never the photo. Without this the directory grows
    forever on a 57 GB eMMC. Date-named directories mean this is a name
    comparison, so it cannot accidentally delete a file that is merely being
    read slowly.
    """
    if retention_days <= 0 or not os.path.isdir(base_dir):
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for entry in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, entry)
        if not os.path.isdir(path):
            continue
        try:
            stamp = time.mktime(time.strptime(entry, _DATE_FORMAT))
        except ValueError:
            continue          # not one of ours; leave it alone
        if stamp < cutoff:
            try:
                shutil.rmtree(path)
                removed += 1
            except OSError as error:
                log.warning("[file_intake] could not remove %s: %s", path, error)
    if removed:
        log.info("[file_intake] pruned %d upload director(ies) older than %d days",
                 removed, retention_days)
    return removed


def handle_upload(
    headers,
    body: bytes,
    base_dir: str,
    token: str | None = None,
    provided_token: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> tuple[int, dict]:
    """Whole endpoint: auth, parse, store, prune. Returns `(status, payload)`.

    Never raises for a client mistake — the caller can send the payload back as
    JSON directly. This keeps the per-service glue to about five lines, which is
    what makes it realistic to add the same endpoint to every driver.
    """
    try:
        check_token(provided_token, token)
        filename, stream, fields = parse_multipart(headers, body)
        result = store_stream(
            stream, filename, base_dir,
            max_bytes=max_bytes, subdir=fields.get("subdir", ""),
        )
    except IntakeError as error:
        return error.status, {"ok": False, "error": error.message}
    except Exception as error:  # noqa: BLE001 - never 500 with a bare traceback
        log.exception("[file_intake] unexpected failure")
        return 500, {"ok": False, "error": str(error)}
    # After the reply is composed, so a slow prune cannot delay the upload's
    # acknowledgement; failures here are logged, not surfaced.
    try:
        prune_old(base_dir, retention_days)
    except Exception:  # noqa: BLE001
        log.warning("[file_intake] prune failed", exc_info=True)
    return 200, result


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


__all__ = [
    "DEFAULT_MAX_BYTES",
    "access_token",
    "DEFAULT_RETENTION_DAYS",
    "IntakeError",
    "check_token",
    "handle_upload",
    "parse_multipart",
    "prune_old",
    "safe_filename",
    "safe_subdir",
    "store_stream",
]
