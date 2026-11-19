"""Store QQ credentials outside Channel configuration and API responses."""

import os
from pathlib import Path
import re
import tempfile

_ROOT = Path('resource/qq-credentials')
_APP_ID = re.compile(r'[0-9]{5,32}')


def validate_app_id(app_id):
    if not isinstance(app_id, str) or not _APP_ID.fullmatch(app_id):
        raise ValueError('Invalid QQ AppID')
    return app_id


def _path(app_id):
    return _ROOT / validate_app_id(app_id)


def write_secret(app_id, secret):
    path = _path(app_id)
    if not isinstance(secret, str) or not 8 <= len(secret) <= 512 or not secret.isascii() or not secret.isprintable():
        raise ValueError('Invalid QQ credential')
    _ROOT.mkdir(mode=0o700, exist_ok=True)
    if _ROOT.is_symlink():
        raise OSError('Invalid credential directory')
    _ROOT.chmod(0o700)
    fd, temp = tempfile.mkstemp(dir=_ROOT)
    try:
        with os.fdopen(fd, 'w', encoding='ascii') as stream:
            stream.write(secret)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def read_secret(app_id):
    path = _path(app_id)
    if _ROOT.is_symlink() or path.is_symlink():
        raise OSError('Invalid QQ credential path')
    return path.read_text(encoding='ascii')


def remove_secret(app_id):
    if not _ROOT.is_symlink():
        _path(app_id).unlink(missing_ok=True)
