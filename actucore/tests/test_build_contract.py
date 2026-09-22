"""Run real build scripts with isolated command sinks, never a Docker daemon."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def build_env(tmp_path):
    root = tmp_path / 'candidate'
    (root / 'deploy').mkdir(parents=True)
    for name in ('build_common.sh', 'build_actucore.sh', 'build_tianyi_actucore.sh'):
        shutil.copy2(ROOT / 'deploy' / name, root / 'deploy' / name)
    bin_path = tmp_path / 'bin'
    bin_path.mkdir()
    log = tmp_path / 'commands.jsonl'
    sink = f'''#!{sys.executable}
import json, os, sys
with open(os.environ['BUILD_COMMAND_LOG'], 'a') as stream:
    stream.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]]) + '\\n')
if os.environ.get('FAIL_PACKAGE_INSTALL') and '-r' in sys.argv:
    sys.exit(19)
'''
    for name, text in {'docker': sink, 'python3': sink,
                       'uname': '#!/bin/sh\nprintf aarch64',
                       'git': '#!/bin/sh\nprintf abc1234'}.items():
        path = bin_path / name
        path.write_text(text)
        path.chmod(0o755)
    env = {**os.environ, 'PATH': str(bin_path) + os.pathsep + os.environ['PATH'],
           'BUILD_COMMAND_LOG': str(log), 'MIRROR': 'none', 'REGISTRY': '',
           'REGISTRY_USER': '', 'REGISTRY_PASSWORD': '', 'IMAGE_NAMESPACE': '',
           'RESOURCE_CENTER_API_KEY': ''}
    return root, env, log


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def run_build(build_env, *args, script='build_actucore.sh'):
    root, env, log = build_env
    result = subprocess.run(['bash', str(root / 'deploy' / script), *args], env=env,
                            capture_output=True, text=True, timeout=10)
    return result, calls(log)


@pytest.mark.parametrize('args,legacy_flag,jp', [([], False, '511'),
    (['--jp-version', '6.1'], False, '61'),
    (['--jp-version', '6.1', '--with-teleop'], True, '61')])
def test_existing_jetson_build_selects_version_without_new_flags(build_env, args, legacy_flag, jp):
    result, recorded = run_build(build_env, *args)
    assert result.returncode == 0, result.stderr
    assert len(recorded) == 1 and recorded[0][:2] == ['docker', 'build']
    command = recorded[0]
    assert f'WITH_TELEOP={int(legacy_flag)}' in command
    assert f'JP_VERSION={jp}' in command
    tag = command[command.index('--tag') + 1]
    assert tag.endswith('-teleop') is legacy_flag
    assert all(part not in command for part in ('push', 'login', 'run', 'compose'))


def test_unsupported_jetpack_fails_before_docker(build_env):
    result, recorded = run_build(build_env, '--with-teleop')
    assert result.returncode != 0 and 'requires JetPack 6.1' in result.stderr
    assert recorded == []


def test_existing_tianyi_cpu_helper_needs_no_extra_flags(build_env):
    result, recorded = run_build(build_env, 'local/actucore:test', script='build_tianyi_actucore.sh')
    assert result.returncode == 0, result.stderr
    assert len(recorded) == 1 and recorded[0][:2] == ['docker', 'build']
    assert '--build-arg' not in recorded[0]
    assert str(build_env[0] / 'actucore/Dockerfile.cpu') in recorded[0]


def instructions(path):
    return path.read_text().replace('\\\n', ' ').splitlines()


def dependency_run(dockerfile):
    return next(line[4:] for line in instructions(ROOT / 'actucore' / dockerfile)
                if line.startswith('RUN ') and '--require-hashes' in line)


@pytest.mark.parametrize('dockerfile,jp,legacy_flag,enabled', [
    ('Dockerfile.cpu', '61', '0', True),
    ('Dockerfile.jetson', '61', '0', True),
    ('Dockerfile.jetson', '61', '1', True),
    ('Dockerfile.jetson', '511', '0', False),
    ('Dockerfile.jetson', '511', '1', False),
])
def test_real_dependency_run_follows_image_version(build_env, dockerfile, jp, legacy_flag, enabled):
    _, env, log = build_env
    result = subprocess.run(['/bin/sh', '-c', dependency_run(dockerfile)],
                            env={**env, 'JP_VERSION': jp, 'WITH_TELEOP': legacy_flag,
                            'PYPI_MIRROR': 'https://invalid.example/simple'}, capture_output=True, text=True)
    recorded = calls(log)
    if not enabled:
        assert result.returncode == 0 and recorded == []
    else:
        assert result.returncode == 0
        assert '--require-hashes' in recorded[0]
        assert '-r' in recorded[0]
        assert any('import pinocchio' in argument for call in recorded for argument in call)


@pytest.mark.parametrize('dockerfile', ['Dockerfile.cpu', 'Dockerfile.jetson'])
def test_install_failure_does_not_produce_successful_import(build_env, dockerfile):
    _, env, log = build_env
    result = subprocess.run(['/bin/sh', '-c', dependency_run(dockerfile)], env={**env, 'JP_VERSION': '61',
                            'FAIL_PACKAGE_INSTALL': '1'}, capture_output=True, text=True)
    assert result.returncode == 19
    assert len(calls(log)) == 1


def test_cpu_service_identity_and_runtime_payload_are_complete():
    jetson = yaml.safe_load((ROOT / 'actucore/deploy/service.yml').read_text())['actucore']
    cpu = yaml.safe_load((ROOT / 'actucore/deploy/service.cpu.yml').read_text())['actucore']
    assert cpu == {key: value for key, value in jetson.items() if key != 'runtime'}
    copied = {}
    for line in instructions(ROOT / 'actucore/Dockerfile.cpu'):
        if line.startswith('COPY '):
            args = shlex.split(line)[1:]
            for src in args[:-1]:
                assert (ROOT / src).exists(), src
                dest = args[-1]
                if dest.endswith('/') and not (ROOT / src).is_dir():
                    dest += Path(src).name
                copied[dest] = src
    assert copied['/deploy/service.yml'] == 'actucore/deploy/service.cpu.yml'
    assert copied['/deploy/dds-local.xml'] == 'agent-core/deploy/dds-local.xml'
    for helper in ('logsafe', 'model_downloader', 'model_progress'):
        assert copied[f'/work/{helper}.py'] == f'perception/utils/{helper}.py'
    assert copied['/work/plugins/'] == 'actucore/plugins/'


@pytest.mark.parametrize('dockerfile,jp,enabled', [
    ('Dockerfile.cpu', '61', True), ('Dockerfile.jetson', '61', True),
    ('Dockerfile.jetson', '511', False),
])
def test_collision_assets_materialize_for_cpu_and_jp61(build_env, dockerfile, jp, enabled):
    _, env, log = build_env
    lines = instructions(ROOT / 'actucore' / dockerfile)
    fetch = next(line[4:] for line in lines if line.startswith('RUN ') and '/tmp/fetch_g1_collision.py' in line)
    result = subprocess.run(['/bin/sh', '-c', fetch], env={**env, 'JP_VERSION': jp, 'WITH_TELEOP': '0'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert calls(log) == ([['python3', '/tmp/fetch_g1_collision.py', '--manifest-dir',
                           '/work/plugins/teleop/models/g1_collision']] if enabled else [])


def test_jetpack_arg_is_available_after_from_and_bare_build_stays_jp511():
    lines = instructions(ROOT / 'actucore/Dockerfile.jetson')
    start = next(i for i, line in enumerate(lines) if line.startswith('FROM '))
    assert 'ARG JP_VERSION=511' in lines[:start]
    assert 'ARG JP_VERSION' in lines[start:]


@pytest.mark.parametrize('dockerfile', ['Dockerfile.cpu', 'Dockerfile.jetson'])
def test_site_readiness_module_and_vla_share_one_shipped_plugin_tree(dockerfile):
    copies = [shlex.split(line)[1:] for line in instructions(ROOT / 'actucore' / dockerfile)
              if line.startswith('COPY ')]
    assert ['actucore/plugins/', '/work/plugins/'] in copies
    assert (ROOT / 'actucore/plugins/teleop/site.py').is_file()
    assert (ROOT / 'actucore/plugins/vla/__init__.py').is_file()


def test_existing_bot_command_selects_unified_jp61_build(tmp_path, monkeypatch):
    import asyncio
    from agents.pr_review import builder, worker
    from agents.pr_review.config import Config
    from agents.pr_review.models import BuildTarget, ReviewJob, parse_trigger_command

    parsed = parse_trigger_command('/request_bot_review core actucore jp61')
    job = ReviewJob(repo_full_name='example/repo', pr_number=1, pr_head_sha='abc1234',
        pr_head_ref='test', pr_base_ref='main', comment_id=1, requester='tester',
        force_targets=parsed['force_targets'], perception_variants=parsed['perception_variants'])
    targets, paths = worker._parse_forced_targets(job.force_targets)
    assert worker._build_plan(job, targets, paths) == [
        (BuildTarget.CORE, None, ''), (BuildTarget.ACTUCORE, None, '6.1')]
    recorded = []
    async def sink(**kwargs): recorded.append(kwargs)
    monkeypatch.setattr(builder, '_build_with_script', sink)
    asyncio.run(builder.build_actucore(tmp_path, Config(), tmp_path / 'log', '6.1'))
    assert recorded[0]['args'] == ['--mirror', 'tencent', '--jp-version', '6.1']
    assert recorded[0]['script'] == tmp_path / 'deploy/build_actucore.sh'
