"""Execute the Docker source fetcher without network access or extraction."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


DOCKERFILE = Path(__file__).parents[1] / 'Dockerfile.jetson'
PAYLOAD = b'complete pinned source archive\n'


def run_fetch(tmp_path, *, failures, tar_status=0):
    dockerfile = DOCKERFILE.read_text()
    function = re.search(r'    fetch_rtabmap_source\(\) \{.*?\n    \};',
                         dockerfile, re.S).group()
    function = function.replace('/tmp/', str(tmp_path) + '/')
    script = r'''
set -e
curl() {
    printf '%s\n' "$*" >> "$TEST_ROOT/curl-args"
    count=0
    if [ -f "$TEST_ROOT/attempts" ]; then count=$(cat "$TEST_ROOT/attempts"); fi
    count=$((count + 1))
    echo "$count" > "$TEST_ROOT/attempts"
    while [ "$1" != '-o' ]; do shift; done
    if [ "$count" -le "$FAILURES" ]; then
        printf 'incomplete transfer' > "$2"
        return 92
    fi
    printf 'complete pinned source archive\n' > "$2"
}
sleep() { echo "$*" >> "$TEST_ROOT/sleeps"; }
sha256sum() { return 97; }
tar() {
    if [ "$TAR_STATUS" -ne 0 ]; then return "$TAR_STATUS"; fi
    cat "$2" > "$TEST_ROOT/extracted"
}
'''
    script += function + '\nfetch_rtabmap_source fixture owner/repo pinned-commit\n'
    env = dict(os.environ, TEST_ROOT=str(tmp_path), FAILURES=str(failures),
               TAR_STATUS=str(tar_status))
    return subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True,
                          timeout=10)


def test_download_failure_retries_then_extracts_archive(tmp_path):
    result = run_fetch(tmp_path, failures=1)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'attempts').read_text().strip() == '2'
    assert (tmp_path / 'extracted').read_bytes() == PAYLOAD
    assert not (tmp_path / 'fixture.tar.gz').exists()
    calls = (tmp_path / 'curl-args').read_text().splitlines()
    assert all('--http1.1' in call.split() for call in calls)
    assert all(call.split()[call.split().index('--max-time') + 1] == '1800'
               for call in calls)
    assert all('https://codeload.github.com/owner/repo/tar.gz/pinned-commit' in call
               for call in calls)


def test_exhausted_download_fails_without_extracting_partial_archive(tmp_path):
    result = run_fetch(tmp_path, failures=10)
    assert result.returncode != 0
    assert (tmp_path / 'attempts').read_text().strip() == '2'
    assert len((tmp_path / 'sleeps').read_text().splitlines()) == 1
    assert not (tmp_path / 'extracted').exists()
    assert not (tmp_path / 'fixture.tar.gz').exists()


def test_tar_failure_propagates_without_retrying_download(tmp_path):
    result = run_fetch(tmp_path, failures=0, tar_status=9)
    assert result.returncode == 9
    assert (tmp_path / 'attempts').read_text().strip() == '1'
    assert not (tmp_path / 'extracted').exists()
    assert not (tmp_path / 'sleeps').exists()


def run_core_fetch(tmp_path, *, interrupt_checkout=False, fail_checkout=False):
    git = shutil.which('git')
    assert git
    seed = tmp_path / 'seed'
    seed.mkdir()
    files = {'LICENSE': 'upstream license', 'CMakeLists.txt': 'project(RTABMap)',
             'Version.h.in': 'pinned version', 'cmake_modules/FindPCL.cmake': 'pcl finder',
             'corelib/src/Registration.cpp': 'registration source',
             'utilite/src/UFile.cpp': 'file source', 'data/unneeded.bin': 'excluded sample data'}
    for name, contents in files.items():
        path = seed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    def command(*args):
        return subprocess.check_output([git, *args], text=True, stderr=subprocess.DEVNULL).strip()
    command('init', '-q', str(seed))
    command('-C', str(seed), 'add', '.')
    command('-C', str(seed), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
            '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Pinned source')
    commit = command('-C', str(seed), 'rev-parse', 'HEAD')
    origin = tmp_path / 'origin.git'
    command('clone', '--bare', '-q', str(seed), str(origin))
    command('-C', str(origin), 'config', 'uploadpack.allowFilter', 'true')
    config = tmp_path / 'gitconfig'
    config.write_text(f'[url "{origin.as_uri()}"]\n    insteadOf = https://github.com/introlab/rtabmap.git\n')
    binary = tmp_path / 'bin'
    binary.mkdir()
    wrapper = binary / 'git'
    wrapper.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
if 'checkout' in sys.argv:
    count_file=pathlib.Path(os.environ['TEST_ROOT'])/'checkouts'
    count=int(count_file.read_text())+1 if count_file.exists() else 1
    count_file.write_text(str(count))
    if os.environ['FAIL_CHECKOUT']=='1' or (os.environ['INTERRUPT_CHECKOUT']=='1' and count==1):
        sys.exit(128)
os.execv(os.environ['REAL_GIT'], [os.environ['REAL_GIT'], *sys.argv[1:]])
''')
    wrapper.chmod(0o755)
    sleep = binary / 'sleep'
    sleep.write_text('#!/bin/sh\nexit 0\n')
    sleep.chmod(0o755)
    recipe = DOCKERFILE.read_text().replace('\\\n', ' ')
    runs = re.findall(r'^RUN (.*)$', recipe, re.M)
    core_run = next((run for run in runs if 'sparse-checkout set cmake_modules corelib utilite' in run), None)
    assert core_run is not None, 'core source must use the bounded sparse Git fetch'
    script = core_run.split('. /opt/ros/humble/install/setup.bash;')[0]
    script = script.replace('/tmp/rtabmap-ws', str(tmp_path / 'workspace'))
    env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'], REAL_GIT=git,
               TEST_ROOT=str(tmp_path), GIT_CONFIG_GLOBAL=str(config), GIT_CONFIG_NOSYSTEM='1',
               GIT_ALLOW_PROTOCOL='file', RTABMAP_CORE_COMMIT=commit,
               INTERRUPT_CHECKOUT=str(int(interrupt_checkout)), FAIL_CHECKOUT=str(int(fail_checkout)))
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, timeout=15)
    return result, tmp_path / 'workspace/src/rtabmap', commit, files


@pytest.mark.parametrize('interrupt_checkout', [False, True])
def test_core_sparse_fetch_preserves_pinned_sources_and_recovers_interrupted_checkout(tmp_path, interrupt_checkout):
    result, source, commit, files = run_core_fetch(tmp_path, interrupt_checkout=interrupt_checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == commit
    for name, contents in files.items():
        if name.startswith('data/'):
            assert not (source / name).exists()
        else:
            assert (source / name).read_text() == contents
    assert (tmp_path / 'checkouts').read_text() == ('2' if interrupt_checkout else '1')


def test_exhausted_core_checkout_fails_before_building_partial_sources(tmp_path):
    result, source, _, _ = run_core_fetch(tmp_path, fail_checkout=True)
    assert result.returncode != 0
    assert (tmp_path / 'checkouts').read_text() == '2'
    assert not (source / 'corelib/src/Registration.cpp').exists()
