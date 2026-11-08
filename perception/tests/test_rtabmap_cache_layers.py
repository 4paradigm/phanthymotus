"""Offline cache-boundary and shell-scope checks, not a native Docker build."""
import io
import os
from pathlib import Path
import re
import subprocess
import tarfile

import pytest


DOCKERFILE = Path(__file__).parents[1] / 'Dockerfile.jetson'
BASE_PACKAGES = ('tf2 tf2_ros tf2_py tf2_ros_py tf2_eigen tf2_geometry_msgs '
                 'message_filters rtabmap_msgs rtabmap_conversions rtabmap_sync '
                 'rtabmap_util rtabmap_slam').split()
OPTIONAL_PACKAGES = ('pcl_conversions pcl_ros diagnostic_updater cv_bridge '
                     'image_geometry image_transport laser_geometry').split()


def layers():
    recipe = DOCKERFILE.read_text().replace('\\\n', ' ')
    instructions = re.findall(r'^(RUN|ARG|COPY) (.*)$', recipe, re.M)
    def index(marker):
        return next(i for i, (_, value) in enumerate(instructions) if marker in value)
    indexes = [index(marker) for marker in (
        'sparse-checkout set cmake_modules corelib utilite',
        'cmake --build /tmp/rtabmap-core-build',
        'fetch_rtabmap_source() {', 'MAKEFLAGS=-j2 colcon build')]
    assert len(set(indexes)) == 4, 'source fetches and compiles need four distinct RUN layers'
    assert indexes == sorted(indexes)
    assert indexes[1] < index('RTABMAP_ROS_COMMIT=') < indexes[2]
    assert indexes[-1] < index('perception/main.py')
    return [instructions[i][1] for i in indexes]


def test_native_source_fetches_have_independent_cache_boundaries():
    core_fetch, core_build, ros_fetch, ros_build = layers()
    for fetch in (core_fetch, ros_fetch):
        assert 'cmake --build' not in fetch and 'colcon build' not in fetch
    for build in (core_build, ros_build):
        assert 'curl ' not in build and 'git ' not in build
        assert '. /opt/ros/humble/install/setup.bash;' in build
    assert '--parallel 2' in core_build
    assert '--executor sequential' in ros_build


def shell_fixture(tmp_path):
    """Use real shells/files/tar; replace only unavailable network/ROS/build tools."""
    ros = tmp_path / 'ros'
    (ros / 'lib').mkdir(parents=True)
    (tmp_path / 'lib').mkdir()
    (tmp_path / 'work').mkdir()
    for path in (ros / 'lib/librcutils.so',
                 ros / 'lib/libbuiltin_interfaces__rosidl_generator_c.so',
                 tmp_path / 'lib/libcrypto.so'):
        path.touch()
    (ros / 'setup.bash').write_text('''export TEST_ROS_READY=1
ros2() {
    case " ${MISSING_PACKAGES} " in *" $3 "*) return 1;; *) return 0;; esac
}
''')
    with tarfile.open(tmp_path / 'source.tar.gz', 'w:gz') as archive:
        content = b'project(source_fixture)\n'
        entry = tarfile.TarInfo('pinned-source/CMakeLists.txt')
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    return r'''
curl() {
    printf '%s\n' "$*" >> "$TEST_ROOT/downloads"
    while [ "$1" != '-o' ]; do shift; done
    cp "$TEST_ROOT/source.tar.gz" "$2"
}
git() { return 91; }
cmake() {
    test "$TEST_ROS_READY" = 1 || return 92
    printf '%s\n' "$*" >> "$TEST_ROOT/cmake-args"
    if [ "$1" = '-S' ]; then test -f "$2/CMakeLists.txt"; else return "$BUILD_STATUS"; fi
}
colcon() {
    test "$TEST_ROS_READY" = 1 || return 92
    printf '%s\n' "$@" > "$TEST_ROOT/colcon-args"
    printf '%s\n' "$LD_LIBRARY_PATH" > "$TEST_ROOT/build-library-path"
    return "$BUILD_STATUS"
}
'''


def run_layer(script, tmp_path, prelude, *, missing=(), build_status=0):
    for old, new in (
        ('/opt/ros/humble/install', tmp_path / 'ros'),
        ('/usr/lib/aarch64-linux-gnu', tmp_path / 'lib'),
        ('/usr/local/lib', tmp_path / 'local/lib'),
        ('/opt/rtabmap_ros', tmp_path / 'install'),
        ('/tmp/rtabmap-core-build', tmp_path / 'core-build'),
        ('/tmp/rtabmap-ws', tmp_path / 'workspace'),
        ('/tmp/${source_name}.tar.gz', str(tmp_path) + '/${source_name}.tar.gz'),
        ('/work;', str(tmp_path / 'work') + ';'),
    ):
        script = script.replace(old, str(new))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('rtabmap_') and key != 'TEST_ROS_READY'}
    env.update(TEST_ROOT=str(tmp_path), MISSING_PACKAGES=' '.join(missing),
               BUILD_STATUS=str(build_status), RTABMAP_ROS_COMMIT='fixture-commit',
               LD_LIBRARY_PATH='/fixture/underlay')
    return subprocess.run(['bash', '-c', prelude + '\n' + script], env=env,
                          capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize('missing', [(), ('cv_bridge', 'pcl_ros'), tuple(OPTIONAL_PACKAGES)])
def test_ros_package_selection_survives_a_fresh_build_shell(tmp_path, missing):
    _, _, fetch, build = layers()
    prelude = shell_fixture(tmp_path)
    fetched = run_layer(fetch, tmp_path, prelude, missing=missing)
    assert fetched.returncode == 0, fetched.stdout + fetched.stderr
    assert not (tmp_path / 'colcon-args').exists()
    download_log = (tmp_path / 'downloads').read_bytes()
    for name in ('rtabmap_ros', 'message_filters', 'geometry2'):
        assert (tmp_path / 'workspace/src' / name / 'CMakeLists.txt').is_file()
    result = run_layer(build, tmp_path, prelude)
    assert result.returncode == 0, result.stdout + result.stderr
    args = (tmp_path / 'colcon-args').read_text().splitlines()
    selected = args[args.index('--packages-select') + 1:args.index('--cmake-args')]
    assert selected == BASE_PACKAGES + [p for p in OPTIONAL_PACKAGES if p in missing]
    assert any(arg.startswith('-DPython3_EXECUTABLE=/') for arg in args)
    assert any(arg.startswith('-DPYTHON_EXECUTABLE=/') for arg in args)
    assert (tmp_path / 'build-library-path').read_text().strip() == (
        str(tmp_path / 'local/lib') + ':/fixture/underlay')
    assert (tmp_path / 'downloads').read_bytes() == download_log
    assert not (tmp_path / 'workspace').exists()


def test_ros_build_without_saved_package_selection_fails_before_colcon(tmp_path):
    _, _, _, build = layers()
    prelude = shell_fixture(tmp_path)
    result = run_layer(build, tmp_path, prelude)
    assert result.returncode != 0
    assert not (tmp_path / 'colcon-args').exists()


def test_ros_compile_failure_can_retry_without_downloading_sources(tmp_path):
    _, _, fetch, build = layers()
    prelude = shell_fixture(tmp_path)
    fetched = run_layer(fetch, tmp_path, prelude, missing=('pcl_ros',))
    assert fetched.returncode == 0, fetched.stderr
    download_log = (tmp_path / 'downloads').read_bytes()
    failed = run_layer(build, tmp_path, prelude, build_status=37)
    assert failed.returncode == 37
    assert (tmp_path / 'workspace/src/rtabmap_ros/CMakeLists.txt').is_file()
    retried = run_layer(build, tmp_path, prelude)
    assert retried.returncode == 0, retried.stderr
    assert (tmp_path / 'downloads').read_bytes() == download_log


@pytest.mark.parametrize('build_status', [0, 37])
def test_core_build_runs_from_prepared_sources_in_a_fresh_shell(tmp_path, build_status):
    _, build, _, _ = layers()
    prelude = shell_fixture(tmp_path)
    source = tmp_path / 'workspace/src/rtabmap'
    source.mkdir(parents=True)
    (source / 'CMakeLists.txt').write_text('project(source_fixture)\n')
    result = run_layer(build, tmp_path, prelude, build_status=build_status)
    assert result.returncode == build_status, result.stdout + result.stderr
    assert not (tmp_path / 'downloads').exists()
    assert source.exists() == bool(build_status)
