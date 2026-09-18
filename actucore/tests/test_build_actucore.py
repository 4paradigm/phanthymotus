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


@pytest.mark.parametrize('name', ['Dockerfile.jetson', 'Dockerfile.vla'])
def test_final_image_contains_entry_point_dependencies(name):
    final = (ROOT / 'actucore' / name).read_text().rsplit('\nFROM ', 1)[1]
    copies = [line.split() for line in final.splitlines() if line.startswith('COPY ')]
    for source, destination in [('actucore/main.py', '/work/main.py'),
                                ('actucore/utils/', '/work/utils/')]:
        assert ['COPY', source, destination] in copies
    assert 'python3 -c "import main' in final


@pytest.mark.parametrize('output,status,success', [
    ('libpcl_common.so.1.10 => /usr/lib/libpcl_common.so.1.10', 0, True),
    ('libpcl_common.so.1.10 => not found', 0, False),
    ('not a dynamic executable', 1, False),
])
def test_final_native_check_fails_closed(tmp_path, output, status, success):
    final = (ROOT / 'actucore/Dockerfile.jetson').read_text().rsplit('\nFROM ', 1)[1]
    check = 'for binary in ' + final.split('for binary in ', 1)[1].split('done &&', 1)[0] + 'done'
    assert 'fastlivo_mapping' in check and 'controller_server' in check
    assert 'libsegmented_controller.so' in check
    assert 'ctypes.CDLL' in final
    fake = tmp_path / 'ldd'
    fake.write_text(f'#!{sys.executable}\nimport sys\nprint({output!r})\nsys.exit({status})\n')
    fake.chmod(0o755)
    result = subprocess.run(['bash', '-o', 'pipefail', '-c', check],
                            env={'PATH': f'{tmp_path}:/usr/bin:/bin'},
                            capture_output=True, text=True, timeout=5)
    assert (result.returncode == 0) is success, result.stdout + result.stderr
