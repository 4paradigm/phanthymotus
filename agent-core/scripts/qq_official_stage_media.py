"""Privileged operator-only media staging for QQ passive replies.

Not an Agent tool: a QQ viewer must not be able to stage arbitrary local files.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

OUTBOX = Path("/work/resource/qq_outbox")
MAX_BYTES = 20 * 1024 * 1024
_ALLOWED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
    ".mp4", ".mov", ".webm", ".mp3", ".wav", ".ogg", ".m4a",
    ".txt", ".pdf", ".csv", ".zip",
}


def stage_media(source: Path, outbox: Path = OUTBOX) -> Path:
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Source is not a regular file")
    suffix = resolved.suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError("Unsupported media extension")
    length = resolved.stat().st_size
    if not 0 < length <= MAX_BYTES:
        raise ValueError("Media size outside allowed bounds")
    outbox.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not outbox.is_dir() or outbox.is_symlink():
        raise ValueError("Outbox must be a local directory")
    os.chmod(outbox, 0o700)
    staged = outbox / (uuid4().hex + suffix)
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            with resolved.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return staged


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path,
                        help="Trusted local media path, selected by an operator")
    parser.add_argument("--outbox", type=Path, default=OUTBOX,
                        help="Override only for isolated operator testing")
    args = parser.parse_args(argv)
    try:
        dst = stage_media(args.source, args.outbox)
    except (ValueError, OSError) as exc:
        print(f"Stage failed: {type(exc).__name__}")
        return 2
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
