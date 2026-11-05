"""Run the real build entry point with Docker replaced in an isolated tree."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
NAVIGATION_BASE = re.search(
    r'actucore-navigation-base:([\w.-]+)',
    (ROOT / 'deploy/build_actucore.sh').read_text(),
).group(1)


@pytest.mark.parametrize('entry,args,extra,kind', [
    ('build_actucore.sh', [], {}, 'app'),
    ('build_actucore.sh', ['--jp-version', '6.1'], {}, 'app'),
    ('build_actucore.sh', [], {'BASE_REGISTRY': 'mirror.example', 'BASE_NAMESPACE': 'robot'}, 'app'),
    ('build_actucore.sh', ['--jp-version', '6.1'], {'BASE_REGISTRY': 'mirror.example', 'BASE_NAMESPACE': 'robot'}, 'app'),
    ('build_actucore.sh', [], {'ACTUCORE_NAVIGATION_BASE_IMAGE': 'example.com/image:custom-v2'}, 'app'),
    ('prepare_navigation_base.sh', [], {'NAVIGATION_BASE_TAG': 'jp511-v1'}, 'base'),
    ('prepare_navigation_base.sh', ['--jp-version', '6.1'], {'NAVIGATION_BASE_TAG': 'jp61-v1'}, 'base'),
    ('build_actucore.sh', ['--base'], {'NAVIGATION_BASE_TAG': 'jp511-v1'}, 'base'),
    ('build_actucore.sh', ['--base', '--jp-version', '6.1'], {'NAVIGATION_BASE_TAG': 'jp61-v1'}, 'base'),
    ('prepare_navigation_base.sh', [], {'NAVIGATION_BASE_TAG': 'jp511-v2', 'BASE_REGISTRY': 'mirror.example', 'BASE_NAMESPACE': 'robot', 'BUILD_JOBS': '2', 'GIT_MIRROR_PREFIX': 'https://proxy/', 'TEST_MIRROR': 'tuna'}, 'base'),
    ('build_actucore.sh', [], {'BUILD_JOBS': '0'}, None),
    ('build_actucore.sh', [], {'ACTUCORE_NAVIGATION_BASE_IMAGE': 'image:latest'}, None),
    ('build_actucore.sh', [], {'ACTUCORE_NAVIGATION_BASE_IMAGE': 'registry:5000/image'}, None),
    ('build_actucore.sh', ['--jp-version', '6.1'], {'ACTUCORE_NAVIGATION_BASE_IMAGE': 'image:jp511-v1'}, None),
    ('prepare_navigation_base.sh', [], {}, None),
    ('prepare_navigation_base.sh', [], {'NAVIGATION_BASE_TAG': 'latest'}, None),
    ('prepare_navigation_base.sh', [], {'NAVIGATION_BASE_TAG': 'jp61-v1'}, None),
    ('prepare_navigation_base.sh', [], {'NAVIGATION_BASE_TAG': 'jp511-v1', 'BUILD_JOBS': '0'}, None),
    ('prepare_navigation_base.sh', [], {'NAVIGATION_BASE_TAG': 'jp511-v1', 'TEST_ARCH': 'x86_64'}, None),
])
def test_platform_build_selection(tmp_path, entry, args, extra, kind):
    deploy = tmp_path / 'deploy'
    deploy.mkdir()
    for name in ('build_actucore.sh', 'prepare_navigation_base.sh', 'build_common.sh'):
        shutil.copyfile(ROOT / 'deploy' / name, deploy / name)
    locks = Path('actucore/plugins/controlled_semantic_spatial/runtime')
    (tmp_path / locks).mkdir(parents=True)
    lock_values = {}
    for name in ('fast_livo2-source.lock', 'nav2-source.lock'):
        content = (ROOT / locks / name).read_text()
        (tmp_path / locks / name).write_text(content)
        lock_values.update(line.split('=', 1) for line in content.splitlines()
                           if line and not line.startswith('#'))
    bins = tmp_path / 'bin'
    bins.mkdir()
    log = tmp_path / 'docker.jsonl'
    generated = tmp_path / 'generated.Dockerfile'
    scripts = {
        'uname': '#!/bin/sh\necho "${TEST_ARCH:-aarch64}"\n',
        'git': '#!/bin/sh\necho abcdef0\n',
        'docker': (f'#!{sys.executable}\nimport json,sys,pathlib\n'
                   f'with open({str(log)!r}, "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                   f'p=pathlib.Path(sys.argv[sys.argv.index("--file")+1])\n'
                   f'if p.exists(): pathlib.Path({str(generated)!r}).write_text(p.read_text())\n'),
    }
    for name, body in scripts.items():
        (bins / name).write_text(body)
        (bins / name).chmod(0o755)
    env = {'PATH': f'{bins}:/usr/bin:/bin', 'HOME': str(tmp_path), **extra}
    mirror = extra.get('TEST_MIRROR', 'none')
    result = subprocess.run(['bash', str(deploy / entry), '--mirror', mirror, *args],
                            env=env, capture_output=True, text=True, timeout=10)
    if kind is None:
        assert result.returncode != 0
        assert not log.exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1 and calls[0][0] == 'build'
    call = calls[0]
    build_args = dict(call[index + 1].split('=', 1) for index, item in enumerate(call)
                      if item == '--build-arg')
    assert build_args['BUILD_JOBS'] == extra.get('BUILD_JOBS', '4')
    assert build_args['PYPI_MIRROR'] == (
        'https://pypi.tuna.tsinghua.edu.cn/simple/' if mirror == 'tuna'
        else 'https://pypi.org/simple/'
    )
    assert build_args['APT_MIRROR'] == ('mirrors.tuna.tsinghua.edu.cn' if mirror == 'tuna' else '')
    assert call[-1] == str(tmp_path)
    jp61 = ['--jp-version', '6.1'] == args[-2:]
    if kind == 'app':
        assert Path(call[call.index('--file') + 1]).name == 'Dockerfile.jetson'
        assert build_args['JP_VERSION'] == ('61' if jp61 else '511')
        base = build_args['BASE_IMAGE']
        if jp61:
            assert base.endswith('/jetson-base-actucore:jp61-torch')
        elif 'ACTUCORE_NAVIGATION_BASE_IMAGE' in extra:
            assert base == extra['ACTUCORE_NAVIGATION_BASE_IMAGE']
        else:
            assert base.endswith('/actucore-navigation-base:' + NAVIGATION_BASE)
    else:
        base = build_args['ACTUCORE_NAVIGATION_PARENT_IMAGE']
        assert f"jetson-base:jp{'61' if jp61 else '511'}-torch@sha256:" in base
        registry = extra.get('BASE_REGISTRY', 'bj-warehouse.tencentcloudcr.com')
        namespace = extra.get('BASE_NAMESPACE', 'phanthy-motus')
        assert call[call.index('--tag') + 1] == (
            f"{registry}/{namespace}/actucore-navigation-base:{extra['NAVIGATION_BASE_TAG']}"
        )
        assert {key: build_args[key] for key in lock_values} == lock_values
        assert build_args['GIT_MIRROR_PREFIX'] == extra.get('GIT_MIRROR_PREFIX', '')
        assert 'FROM ${ACTUCORE_NAVIGATION_PARENT_IMAGE} AS navigation_builder' in generated.read_text()
    if 'BASE_REGISTRY' in extra:
        assert base.startswith('mirror.example/robot/')


def test_final_image_contains_entry_point_dependencies():
    final = (ROOT / 'actucore/Dockerfile.jetson').read_text().rsplit('\nFROM ', 1)[1]
    copies = [line.split() for line in final.splitlines() if line.startswith('COPY ')]
    assert ['COPY', 'actucore/main.py', '/work/main.py'] in copies
    assert 'python3 -c "import main' in final


def test_audio_msgs_declares_ament_build_type():
    manifest = ET.parse(ROOT / 'deploy/ros-base/audio_msgs/package.xml').getroot()
    assert [node.text for node in manifest.findall('./export/build_type')] == ['ament_cmake']


def test_platform_ros_build_and_jp61_environment():
    dockerfile = (ROOT / 'actucore/Dockerfile.jetson').read_text()
    assert 'COPY deploy/ros-base/audio_msgs/ /ros_ws/src/audio_msgs/' in dockerfile
    assert '--packages-select audio_msgs fast_livo2 segmented_controller nav2' in dockerfile
    assert '--packages-select audio_msgs --cmake-args -DCMAKE_BUILD_TYPE=Release' in dockerfile
    assert 'AUDIO_MSGS_INDEX=present' in dockerfile
    assert 'AUDIO_MSGS_PREFIX=present' in dockerfile
    assert '/ros_ws/install/audio_msgs/share/ament_index/resource_index/packages/audio_msgs' in dockerfile
    assert 'ros2 pkg prefix audio_msgs >/dev/null' in dockerfile
    assert 'from audio_msgs.msg import AudioChunk' in dockerfile
    assert 'ENV LD_LIBRARY_PATH="/ros_ws/install/audio_msgs/lib:/opt/ros/humble/install/lib:${LD_LIBRARY_PATH}"' in dockerfile
    assert 'ENV AMENT_PREFIX_PATH="/ros_ws/install/audio_msgs:/opt/ros/humble/install"' in dockerfile
    assert 'ACTUCORE_JP_VERSION=${JP_VERSION}' in dockerfile
    assert 'if [ \\"${ACTUCORE_JP_VERSION}\\" = 511 ]; then source /opt/ros_deps_ws/install/setup.bash' in dockerfile


@pytest.mark.parametrize('index_exists,prefix_present', [
    (False, False), (False, True), (True, False), (True, True),
])
def test_audio_index_diagnostic_checks_real_shell_environment(tmp_path, index_exists, prefix_present):
    final = (ROOT / 'actucore/Dockerfile.jetson').read_text().rsplit('\nFROM ', 1)[1]
    check = final.split('    if [ -f /ros_ws/install/audio_msgs/share/', 1)[1]
    check = ('if [ -f /ros_ws/install/audio_msgs/share/' +
             check.split('    ros2 pkg prefix audio_msgs', 1)[0])
    index = tmp_path / 'audio_msgs/share/ament_index/resource_index/packages/audio_msgs'
    if index_exists:
        index.parent.mkdir(parents=True)
        index.touch()
    check = check.replace('/ros_ws/install/audio_msgs', str(tmp_path / 'audio_msgs'))
    check = check.replace('\\\n', '\n') + 'true\n'
    prefix = str(tmp_path / 'audio_msgs') if prefix_present else str(tmp_path / 'other')
    result = subprocess.run(['bash', '-e', '-c', check],
                            env={'AMENT_PREFIX_PATH': prefix}, capture_output=True, text=True)
    assert (result.returncode == 0) is index_exists
    assert ('AUDIO_MSGS_INDEX=present' in result.stdout) is index_exists
    if index_exists:
        assert ('AUDIO_MSGS_PREFIX=present' in result.stdout) is prefix_present
        assert ('AUDIO_MSGS_PREFIX=missing' in result.stderr) is not prefix_present


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


@pytest.mark.parametrize('dockerfile', ['deploy/prepare_navigation_base.sh', 'actucore/Dockerfile.jetson'])
def test_card_runtime_copy_sources_exist(dockerfile):
    lines = (ROOT / dockerfile).read_text().splitlines()
    prefix = 'COPY actucore/plugins/controlled_semantic_spatial/runtime/'
    sources = [line.split()[1] for line in lines if line.startswith(prefix)]
    assert sources
    assert all((ROOT / source).exists() for source in sources)


def test_jp61_copied_plugins_can_import_without_source_tree(tmp_path):
    final = (ROOT / 'actucore/Dockerfile.jetson').read_text()
    for line in final.splitlines():
        if not line.startswith('COPY ') or '/work/' not in line:
            continue
        _, source, destination = line.split()
        target = tmp_path / Path(destination).relative_to('/work')
        target.parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / source).is_dir():
            shutil.copytree(ROOT / source, target)
        else:
            shutil.copyfile(ROOT / source, target)
    assert 'if [ "${JP_VERSION}" = 61 ]; then' in final
    config_command = next(line for line in final.splitlines()
                          if line.strip().startswith('python3 -c "import yaml;'))
    config_code = config_command.split('"', 1)[1].rsplit('"', 1)[0]
    subprocess.run([sys.executable, '-c', config_code.replace('/work/', f'{tmp_path}/')],
                   check=True, timeout=10)
    original = yaml.safe_load((ROOT / 'actucore/config.yaml').read_text())
    shipped = yaml.safe_load((tmp_path / 'config.yaml').read_text())
    original['plugins']['controlled_semantic_spatial']['enabled'] = False
    assert shipped == original
    assert (tmp_path / 'plugins/navi/plugin.py').is_file()
    assert (tmp_path / 'tools/measure_fov.py').is_file()
    assert 'COPY deploy/ros-base/audio_msgs/' in final
    entry_code = """
import sys, types
for name in ('rclpy', 'rclpy.executors', 'rclpy._rclpy_pybind11'):
    sys.modules[name] = types.ModuleType(name)
sys.modules['rclpy._rclpy_pybind11'].InvalidHandle = type('InvalidHandle', (Exception,), {})
import main
bundle = main.ActuCoreBundle(main._load_config(), None)
assert {tool['name'] for tool in bundle.get_all_tools()} == {'vla', 'navi'}
assert main._load_config()['plugins']['controlled_semantic_spatial']['enabled'] is False
"""
    result = subprocess.run(
        [sys.executable, '-c', entry_code],
        cwd=tmp_path, env={'PYTHONPATH': str(tmp_path)},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
