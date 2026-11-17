"""Compile and execute the same capture-hint helper installed into RTAB-Map."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def native_rotation_hint(tmp_path_factory):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('native hint policy requires a C++ compiler')
    binary = tmp_path_factory.mktemp('rotation-hint') / 'rotation_hint_test'
    result = subprocess.run([
        compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
        '-I', str(ROOT / 'native/rtabmap'),
        str(ROOT / 'tests/native/test_rotation_hint.cpp'), '-o', str(binary),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize('case', [
    'exact_stamp', 'missing_late', 'dropped_failed', 'clear_anchor',
    'invalid', 'stale', 'rotation', 'bounded', 'reset',
])
def test_capture_hint_policy(native_rotation_hint, case):
    result = subprocess.run([str(native_rotation_hint), case],
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stdout + result.stderr


def test_rotation_hint_capability_marker_requires_successful_native_build(tmp_path):
    # Execute the real Docker build shell; only unavailable network/compiler
    # dependencies use the existing fixture, not the marker creation itself.
    from test_rtabmap_cache_layers import layers, run_layer, shell_fixture
    _, _, fetch, build = layers()
    prelude = shell_fixture(tmp_path)
    fetched = run_layer(fetch, tmp_path, prelude)
    assert fetched.returncode == 0, fetched.stdout + fetched.stderr
    marker = tmp_path / 'install/share/rtabmap_odom/motus_rotation_hint_v1'
    failed = run_layer(build, tmp_path, prelude, build_status=37)
    assert failed.returncode == 37
    assert not marker.exists()
    succeeded = run_layer(build, tmp_path, prelude)
    assert succeeded.returncode == 0, succeeded.stdout + succeeded.stderr
    assert marker.is_file()
    assert marker.read_bytes() == b'motus_rotation_hint/1\n'


def test_native_odometry_patch_removes_the_wrapper_timestamp_admission_gate():
    patch = (ROOT / 'native/rtabmap/rotation_hint_ros.patch').read_text()
    retired_guard = '-\t\tif(previousStamp_ >= rtabmap_conversions::timestampFromROS(header.stamp))'
    assert retired_guard in patch, 'wrapper must delegate invalid dt to native odometry instead of silently dropping'
    hunk = next(part for part in patch.split('@@') if retired_guard in part)
    assert '-\t\t\treturn;' in hunk
    assert '-\t\telse if(maxUpdateRate_' in hunk
    assert '+\t\tif(maxUpdateRate_' in hunk
