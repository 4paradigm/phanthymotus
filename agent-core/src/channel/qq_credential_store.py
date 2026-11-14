"""Managed QQ credential files outside Channel SQLite.

The installation's DB_PATH parent must be on a persistent volume. The app
accepts only its own randomly named files in a private directory, and never
returns their content in any API response.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import secrets
import stat

_APP = re.compile(r"^[0-9]{5,32}$")
_FILE = re.compile(r"^qq-([0-9]{5,32})-[0-9a-f]{32}\.secret$")


def directory() -> Path:
    import config
    return Path(config.DB_PATH).resolve().parent / "qq-credentials"


def is_managed_reference(app_id: str, path: str) -> bool:
    if not isinstance(path, str) or not _APP.fullmatch(str(app_id)):
        return False
    candidate = Path(path)
    match = _FILE.fullmatch(candidate.name)
    return (bool(match) and match.group(1) == str(app_id)
            and candidate.parent == directory() and candidate.is_absolute())


def persist(app_id: str, credential: str) -> str:
    """Create a private, immutable credential version; return its path."""
    if not _APP.fullmatch(str(app_id)):
        raise ValueError("Invalid QQ AppID")
    if not isinstance(credential, str) or not (8 <= len(credential.strip()) <= 512):
        raise ValueError("Invalid QQ credential")
    root = directory()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("QQ secret directory is not a safe directory")
    os.chmod(root, 0o700)
    path = root / f"qq-{app_id}-{secrets.token_hex(16)}.secret"
    options = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        options |= os.O_NOFOLLOW
    fd = os.open(path, options, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            if hasattr(os, 'fchmod'):
                os.fchmod(out.fileno(), stat.S_IRUSR | stat.S_IWUSR)
            else:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            out.write(credential.strip())
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return str(path)


def read(app_id: str, path: str) -> str:
    if not is_managed_reference(app_id, path):
        raise ValueError("Invalid managed QQ credential reference")
    options = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        options |= os.O_NOFOLLOW
    try:
        fd = os.open(path, options)
        with os.fdopen(fd, "r", encoding="utf-8") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("QQ credential must be a regular file")
            value = source.read(513).strip()
    except (OSError, UnicodeError):
        raise ValueError("Managed QQ credential unavailable") from None
    if not (8 <= len(value) <= 512):
        raise ValueError("Managed QQ credential empty or invalid")
    return value


def delete(app_id: str, path: str) -> None:
    if is_managed_reference(app_id, path):
        Path(path).unlink(missing_ok=True)
