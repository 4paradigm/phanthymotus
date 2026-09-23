#!/usr/bin/env python3
"""Verify a signed PICO APK and materialize the fixed Docker build input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def stage(apk, destination, tools):
    badge = run(str(tools/'aapt2'), 'dump', 'badging', str(apk))
    package = re.search(r"package: name='([^']+)' versionCode='(\d+)' versionName='([^']+)'", badge)
    if not package or package[1] != 'com.phanthymotus.picocapture':
        raise ValueError('Expected PICO capture application')
    signature = run(str(tools/'apksigner'), 'verify', '--verbose', '--print-certs', str(apk))
    signers = re.findall(r'Signer #\d+ certificate SHA-256 digest: ([0-9a-f]{64})', signature)
    if len(signers) != 1 or 'Verified using v2 scheme (APK Signature Scheme v2): true' not in signature:
        raise ValueError('One valid v2 APK signer required')
    run(str(tools/'zipalign'), '-c', '-P', '16', '4', str(apk))
    contents = apk.read_bytes()
    metadata = {'filename':'pico.apk', 'application_id':package[1], 'version_code':int(package[2]),
        'version':package[3], 'build_type':'debug' if 'application-debuggable' in badge else 'release',
        'size_bytes':len(contents), 'sha256':hashlib.sha256(contents).hexdigest(),
        'signing_certificate_sha256':signers[0]}
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination/'pico.apk.tmp'
    temporary.write_bytes(contents)
    temporary.replace(destination/'pico.apk')
    temporary = destination/'package.json.tmp'
    temporary.write_text(json.dumps(metadata, indent=2)+'\n')
    temporary.replace(destination/'package.json')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('apk', type=Path)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts')
    args = parser.parse_args()
    sdk = os.environ.get('ANDROID_SDK_ROOT')
    if not sdk:
        parser.error('ANDROID_SDK_ROOT required')
    print(json.dumps(stage(args.apk.resolve(), args.destination, Path(sdk)/'build-tools/35.0.1'), indent=2))
