"""Recover a complete MPEG-4 audio file sent in chunks on a PCM topic."""

import struct
import subprocess
import tempfile


class Mpeg4AudioInput:
    """Pass PCM through; buffer self-describing MP4 boxes until seekable decode.

    M4A commonly puts its sample table after the media bytes. A pipe decoder
    cannot reliably seek back to those samples, so use a temporary input file.
    Padding after the final box belongs to the transport, not the container.
    """

    MAX_BYTES = 32 * 1024 * 1024

    def __init__(self):
        self._buffer = None
        self._offset = 0
        self._boxes = set()
        self._failed = False

    def feed(self, data):
        """Return (PCM16 bytes, converted); incomplete containers return no PCM."""
        if self._failed:
            return b"", False
        if self._buffer is None:
            if len(data) < 12 or data[4:8] != b"ftyp":
                return data, False
            self._buffer = bytearray()
        self._buffer.extend(data)
        try:
            if len(self._buffer) > self.MAX_BYTES:
                raise ValueError("MPEG-4 audio exceeds the 32 MiB input limit")
            while self._offset + 8 <= len(self._buffer):
                size, kind = struct.unpack_from(">I4s", self._buffer, self._offset)
                header = 8
                if size == 1:
                    if self._offset + 16 > len(self._buffer):
                        break
                    size = struct.unpack_from(">Q", self._buffer, self._offset + 8)[0]
                    header = 16
                if size < header or self._offset + size > self.MAX_BYTES:
                    raise ValueError("MPEG-4 audio needs valid, explicit box lengths")
                end = self._offset + size
                if end > len(self._buffer):
                    break
                self._boxes.add(kind)
                self._offset = end
                if {b"ftyp", b"moov", b"mdat"} <= self._boxes:
                    encoded = bytes(self._buffer[:end])
                    with tempfile.TemporaryFile(suffix=".mp4") as source:
                        source.write(encoded)
                        source.seek(0)
                        result = subprocess.run(
                            ["ffmpeg", "-nostdin", "-v", "error", "-i",
                             "/proc/self/fd/" + str(source.fileno()),
                             "-map", "0:a:0", "-f", "s16le", "-ac", "1",
                             "-ar", "16000", "pipe:1"],
                            pass_fds=(source.fileno(),), capture_output=True,
                            timeout=15, check=True,
                        )
                    if not result.stdout or len(result.stdout) % 2:
                        raise ValueError("MPEG-4 audio produced no valid PCM16")
                    self._buffer = None
                    self._offset = 0
                    self._boxes.clear()
                    return result.stdout, True
            return b"", False
        except Exception:
            self._buffer = None
            self._failed = True
            raise
