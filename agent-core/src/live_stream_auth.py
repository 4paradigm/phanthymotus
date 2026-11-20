"""Local, shared authorization key for Bumi's public live-stream actions.

Core and the Bumi Driver run on the same trusted host and mount the same data
directory. Only this module creates the key; the Driver reads it for validation.
The key is never sent to clients, MCP responses, or browser JavaScript.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
import time
from pathlib import Path

_KEY_DIR = Path(os.environ.get(
    "PHANTHY_LIVE_STREAM_AUTH_DIR", "/opt/phanthy-motus/data/live_stream"
))
_KEY_NAME = ".operator-approval-key"


def _read_key() -> bytes:
    directory = os.open(_KEY_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(directory)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise OSError("live-stream authorization directory is not private")
        fd = os.open(_KEY_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise OSError("live-stream authorization key is not private")
            key = os.read(fd, 33)
            if len(key) != 32:
                raise OSError("invalid live-stream authorization key")
            return key
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def _ensure_key() -> bytes:
    try:
        return _read_key()
    except FileNotFoundError:
        pass
    _KEY_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd_dir = os.open(_KEY_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd_dir)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise OSError("live-stream authorization directory is not private")
        try:
            fd = os.open(
                _KEY_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=fd_dir,
            )
        except FileExistsError:
            pass  # Another Core instance created the key.
        else:
            with os.fdopen(fd, "wb") as output:
                output.write(secrets.token_bytes(32))
                output.flush()
                os.fsync(output.fileno())
            os.fsync(fd_dir)
    finally:
        os.close(fd_dir)
    return _read_key()


def sign_live_stream_action(action: str, url: str) -> str:
    if action not in ("config", "go_live") or not isinstance(url, str) or not url:
        raise ValueError("invalid live-stream operation")
    expiry = str(int(time.time()) + 20)
    nonce = secrets.token_hex(16)
    payload = (action + "\x00" + url + "\x00" + expiry + "\x00" + nonce).encode("utf-8")
    mac = hmac.new(_ensure_key(), payload, hashlib.sha256).hexdigest()
    return f"{expiry}.{nonce}.{mac}"
