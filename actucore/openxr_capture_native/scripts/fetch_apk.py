#!/usr/bin/env python3
"""Build-time fetch of the checked-in, immutable PICO artifact manifest."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen


def verified(path, manifest):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != manifest['size_bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest() == manifest['sha256']


def fetch(manifest_path, destination):
    manifest = json.loads(Path(manifest_path).read_text())
    url = urlsplit(manifest['source_url'])
    if (url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment
            or manifest.get('filename') != 'pico.apk'
            or manifest.get('application_id') != 'com.phanthymotus.picocapture'
            or manifest.get('build_type') != 'release'
            or type(manifest.get('size_bytes')) is not int or not 0 < manifest['size_bytes'] <= 256*1024*1024):
        raise ValueError('a published HTTPS release artifact manifest is required')
    for field in ('sha256', 'signing_certificate_sha256'):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid artifact digest')
    compressed = manifest.get('source_encoding') == 'gzip'
    if manifest.get('source_encoding', 'identity') not in ('identity', 'gzip'):
        raise ValueError('unsupported source encoding')
    source_size = manifest.get('source_size_bytes') if compressed else manifest['size_bytes']
    source_digest = manifest.get('source_sha256') if compressed else manifest['sha256']
    if (type(source_size) is not int or not 0 < source_size <= 256*1024*1024
            or not isinstance(source_digest, str) or len(source_digest) != 64
            or any(c not in '0123456789abcdef' for c in source_digest)):
        raise ValueError('invalid source artifact metadata')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/'pico.apk'
    staging = None
    downloaded = None
    try:
        if not verified(target, manifest):
            with tempfile.NamedTemporaryFile(dir=destination, suffix='.download', delete=False) as output:
                downloaded = Path(output.name)
                with urlopen(manifest['source_url'], timeout=60) as response:
                    if urlsplit(response.geturl()).scheme != 'https':
                        raise ValueError('artifact redirect must remain HTTPS')
                    total = 0
                    digest = hashlib.sha256()
                    while True:
                        block = response.read(1024*1024)
                        if not block:
                            break
                        total += len(block)
                        if total > source_size:
                            raise ValueError('artifact size exceeded')
                        digest.update(block)
                        output.write(block)
                    if total != source_size or digest.hexdigest() != source_digest:
                        raise ValueError('source artifact size or SHA256 mismatch')
            if compressed:
                with tempfile.NamedTemporaryFile(dir=destination, suffix='.part', delete=False) as output:
                    staging = Path(output.name)
                    with gzip.open(downloaded, 'rb') as source:
                        total = 0
                        while True:
                            block = source.read(min(1024*1024, manifest['size_bytes']-total+1))
                            if not block:break
                            total += len(block)
                            if total > manifest['size_bytes']:
                                raise ValueError('uncompressed artifact size exceeded')
                            output.write(block)
            else:
                staging = downloaded
            if not verified(staging, manifest):
                raise ValueError('artifact size or SHA256 mismatch')
            staging.replace(target)
        metadata = {k:v for k,v in manifest.items() if not k.startswith('source_')}
        temporary = destination/'package.json.tmp'
        temporary.write_text(json.dumps(metadata, indent=2)+'\n')
        temporary.replace(destination/'package.json')
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)
    return manifest['sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print('PICO APK SHA256 verified: '+fetch(args.manifest, args.destination))
