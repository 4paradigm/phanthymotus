"""Execute every Docker ldconfig mask against native and usrmerge layouts."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


DOCKERFILE = Path(__file__).parents[1] / 'Dockerfile.jetson'


def mask_commands(index):
    recipe = DOCKERFILE.read_text().replace('\\\n', ' ')
    starts = list(re.finditer(r'(?:mv|cp -a) /usr/sbin/ldconfig ', recipe))
    assert len(starts) == 4
    commands = recipe[starts[index].start():]
    restore = re.search(r'\bmv(?: -f)? \S+ /usr/sbin/ldconfig(?=[ ;&])', commands)
    assert restore is not None
    return commands[:restore.end()]


@pytest.mark.parametrize('index', range(4), ids=['system', 'espeak', 'python-dev', 'rtabmap'])
@pytest.mark.parametrize('wrapped', [False, True], ids=['native-binary', 'usrmerge-wrapper'])
def test_mask_restores_executable_without_overwriting_real_binary(tmp_path, index, wrapped):
    sbin = tmp_path / 'usr/sbin'
    sbin.mkdir(parents=True)
    (tmp_path / 'sbin').symlink_to('usr/sbin', target_is_directory=True)
    (tmp_path / 'tmp').mkdir()
    ldconfig = sbin / 'ldconfig'
    binary = sbin / 'ldconfig.real' if wrapped else ldconfig
    # Use an actual executable (ELF on Linux), so restoration is checked by exec too.
    native_true = shutil.which('true')
    shutil.copyfile(native_true, binary)
    binary.chmod(0o755)
    original_binary = binary.read_bytes()
    if wrapped:
        ldconfig.write_text('#!/bin/sh\nexec "$(dirname "$0")/ldconfig.real" "$@"\n')
        ldconfig.chmod(0o755)
    original_entrypoint = ldconfig.read_bytes()
    commands = mask_commands(index).replace('/tmp/', str(tmp_path / 'tmp') + '/')
    commands = commands.replace('/usr/sbin/', str(sbin) + '/')
    commands = commands.replace('/bin/true', native_true)
    script = '''
set -e
apt-get() { "$TEST_ROOT/usr/sbin/ldconfig"; }
''' + commands
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, TEST_ROOT=str(tmp_path)), timeout=10)
    assert result.returncode == 0, result.stderr
    assert binary.exists(), 'mask removed the wrapper target'
    assert binary.read_bytes() == original_binary
    assert ldconfig.read_bytes() == original_entrypoint
    restored = subprocess.run([str(tmp_path / 'sbin/ldconfig')], capture_output=True,
                              text=True, timeout=10)
    assert restored.returncode == 0, restored.stderr
