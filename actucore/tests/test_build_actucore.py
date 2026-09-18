"""Run the real build entry point with Docker replaced in an isolated tree."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('args,extra,dockerfile', [
    ([], {}, 'Dockerfile.jetson'),
    (['--jp-version', '6.1'], {}, 'Dockerfile.vla'),
    (['--base'], {}, 'Dockerfile.navigation-base'),
    ([], {'BUILD_JOBS': '0'}, None),
    ([], {'ACTUCORE_NAVIGATION_BASE_IMAGE': 'image:latest'}, None),
    (['--jp-version', '6.1'], {'ACTUCORE_NAVIGATION_BASE_IMAGE': 'image@sha256:' + 'a' * 64}, None),
])
def test_platform_build_selection(tmp_path, args, extra, dockerfile):
    deploy = tmp_path / 'deploy'
    deploy.mkdir()
    for name in ('build_actucore.sh', 'build_common.sh'):
        shutil.copyfile(ROOT / 'deploy' / name, deploy / name)
    locks = Path('actucore/plugins/navigation/runtime')
    (tmp_path / locks).mkdir(parents=True)
    for name in ('fast_livo2-source.lock', 'nav2-source.lock'):
        shutil.copyfile(ROOT / locks / name, tmp_path / locks / name)
    bins = tmp_path / 'bin'
    bins.mkdir()
    log = tmp_path / 'docker.jsonl'
    scripts = {
        'uname': '#!/bin/sh\necho aarch64\n',
        'git': '#!/bin/sh\necho abcdef0\n',
        'docker': f'#!{sys.executable}\nimport json,sys\nwith open({str(log)!r}, "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")\n',
    }
    for name, body in scripts.items():
        (bins / name).write_text(body)
        (bins / name).chmod(0o755)
    # No inherited registry credentials, .env, Docker socket or real Docker.
    env = {'PATH': f'{bins}:/usr/bin:/bin', 'HOME': str(tmp_path), **extra}
    result = subprocess.run(['bash', str(deploy / 'build_actucore.sh'), '--mirror', 'none', *args],
                            env=env, capture_output=True, text=True, timeout=10)
    if dockerfile is None:
        assert result.returncode != 0
        assert not log.exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1 and calls[0][0] == 'build'
    call = calls[0]
    assert Path(call[call.index('--file') + 1]).name == dockerfile
    if dockerfile == 'Dockerfile.vla':
        assert 'JP_VERSION=61' in call
        assert any(arg.endswith('/jetson-base-actucore:jp61-torch') for arg in call)
        assert not any(arg.startswith('ACTUCORE_RUNTIME_BASE_IMAGE=') for arg in call)
    elif dockerfile == 'Dockerfile.jetson':
        assert any(arg.startswith('ACTUCORE_NAVIGATION_BASE_IMAGE=') and '@sha256:' in arg for arg in call)
