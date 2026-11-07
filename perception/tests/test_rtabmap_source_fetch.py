"""Execute the Docker source fetcher without network access or extraction."""
import os
from pathlib import Path
import re
import subprocess


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
