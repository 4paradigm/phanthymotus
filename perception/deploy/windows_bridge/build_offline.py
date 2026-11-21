"""Run at image build time; never download dependencies on the front desk."""
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def download(url, target, expected_sha256=None):
    """Download with visible progress so CI does not mistake it for a hang."""
    digest = hashlib.sha256()
    written = 0
    last_report = time.monotonic()
    with urllib.request.urlopen(url, timeout=120) as response, target.open('wb') as output:
        total = int(response.headers.get('Content-Length') or 0)
        print(f'[offline] downloading {target.name} ({total or "unknown"} bytes)', flush=True)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            written += len(chunk)
            now = time.monotonic()
            if now - last_report >= 10:
                percent = f' {written * 100 // total}%' if total else ''
                print(f'[offline] {target.name}: {written // (1024 * 1024)} MiB{percent}', flush=True)
                last_report = now
    actual = digest.hexdigest()
    if expected_sha256 and actual != expected_sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError('Downloaded artifact hash mismatch: ' + target.name)
    print(f'[offline] finished {target.name}: {written} bytes', flush=True)
    return actual


def download_from_mirrors(urls, target, expected_sha256=None):
    """Try fast regional mirrors first, while accepting only the expected bytes."""
    errors = []
    for url in urls:
        try:
            return download(url, target, expected_sha256)
        except Exception as exc:
            target.unlink(missing_ok=True)
            errors.append(f'{url}: {exc}')
            print(f'[offline] mirror failed, trying the next source: {url}', flush=True)
    raise RuntimeError('All download sources failed for %s:\n%s' %
                       (target.name, '\n'.join(errors)))


def build(root):
    assets = root / 'offline'
    assets.mkdir(parents=True, exist_ok=True)
    installer = assets / 'python-3.12.10-amd64.exe'
    download_from_mirrors([
        'https://registry.npmmirror.com/-/binary/python/3.12.10/python-3.12.10-amd64.exe',
        'https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe',
        'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe',
    ], installer, '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb')
    # Explicit Windows artifacts avoid depending on the Jetson build host's
    # Python ABI or old pip cross-platform resolver.
    packages = [('playwright', '1.51.0', 'py3-none-win_amd64.whl'),
                ('greenlet', '3.1.1', 'cp312-cp312-win_amd64.whl'),
                ('pyee', '12.1.1', 'py3-none-any.whl'),
                ('typing_extensions', '4.12.2', 'py3-none-any.whl')]
    for name, version, suffix in packages:
        with urllib.request.urlopen('https://pypi.org/pypi/%s/%s/json' % (name, version), timeout=60) as response:
            metadata = json.load(response)
        matches = [f for f in metadata['urls'] if f['filename'].endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError('Missing or ambiguous Windows wheel: ' + name)
        artifact = matches[0]
        target = assets / artifact['filename']
        path = urllib.parse.urlparse(artifact['url']).path
        download_from_mirrors([
            'https://mirrors.cloud.tencent.com/pypi' + path,
            artifact['url'],
        ], target, artifact['digests']['sha256'])
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(assets.iterdir()) if p.suffix in ('.exe', '.whl')}
    if len(manifest) != 5:
        raise RuntimeError('Expected Python installer and four Windows wheels')
    (assets / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


if __name__ == '__main__':
    build(Path(__file__).resolve().parent)
