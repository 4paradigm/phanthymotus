"""Fixed APK artifact verification. No arbitrary paths or remote download URLs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

APK_DIRECTORY = Path('/work/onboarding')
APK_FILENAME = 'pico.apk'
MIME_TYPE = 'application/vnd.android.package-archive'


def package_metadata(directory=APK_DIRECTORY):
    directory = Path(directory)
    try:
        manifest = directory / 'package.json'
        apk = directory / APK_FILENAME
        if manifest.is_symlink() or apk.is_symlink() or manifest.stat().st_size > 8192:
            raise ValueError('invalid_artifact')
        data = json.loads(manifest.read_text())
        if (data['filename'] != APK_FILENAME or data['application_id'] != 'com.phanthymotus.picocapture'
                or data['build_type'] not in ('debug', 'release')
                or not isinstance(data['version'], str) or not 1 <= len(data['version']) <= 128
                or type(data['version_code']) is not int or data['version_code'] < 1
                or type(data['size_bytes']) is not int or not 0 < data['size_bytes'] <= 256*1024*1024
                or data['size_bytes'] != apk.stat().st_size):
            raise ValueError('invalid_artifact')
        for key in ('sha256', 'signing_certificate_sha256'):
            if len(data[key]) != 64 or any(c not in '0123456789abcdef' for c in data[key]):
                raise ValueError('invalid_artifact')
        digest = hashlib.sha256()
        with apk.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != data['sha256']:
            raise ValueError('artifact_digest_mismatch')
        metadata = {k: data[k] for k in ('filename', 'application_id', 'build_type', 'version',
                'version_code', 'size_bytes', 'sha256', 'signing_certificate_sha256')}
        metadata.update(available=True, mime_type=MIME_TYPE)
        return metadata
    except (OSError, ValueError, KeyError, TypeError):
        return {'available': False, 'reason': 'apk_artifact_missing_or_invalid'}


if __name__ == '__main__':
    import sys
    result = package_metadata(sys.argv[1] if len(sys.argv) > 1 else APK_DIRECTORY)
    print(json.dumps(result))
    raise SystemExit(0 if result['available'] else 1)
