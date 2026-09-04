import fnmatch
import pathlib

import yaml

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _context_includes(path: str) -> bool:
    rules = []
    dockerignore = _REPOSITORY_ROOT / 'agent-core' / 'Dockerfile.dockerignore'
    for raw in dockerignore.read_text(encoding='utf-8').splitlines():
        rule = raw.strip()
        if not rule or rule.startswith('#'):
            continue
        include = rule.startswith('!')
        pattern = rule[1:] if include else rule
        rules.append((pattern, include))

    included = True
    for pattern, include in rules:
        if fnmatch.fnmatchcase(path, pattern):
            included = include
    return included


def test_root_docker_context_keeps_sources_but_excludes_runtime_memory_and_caches():
    included = [
        'agent-core/src/agent_memory.py',
        'agent-core/resource/memory/prompt_memory.md',
        'agent-core/deploy/.env.example',
        'memory-core/src/memory_core/repository.py',
    ]
    excluded = [
        'agent-core/resource/memory.db',
        'agent-core/resource/memory.db-wal',
        'agent-core/resource/memory.db-shm',
        'agent-core/resource/nested/private.sqlite3',
        'agent-core/resource/nested/private.sqlite3-wal',
        'agent-core/resource/nested/private.sqlite3-shm',
        'agent-core/resource/.memory.db.handoff.lock',
        'agent-core/resource/memory/.prompt_memory.md.state.json',
        'agent-core/resource/memory/.prompt_memory.md.abc.tmp',
        'agent-core/resource/llm_data/request.json',
        'agent-core/resource/llm_recent_request/latest.json',
        'agent-core/resource/log/agent.log',
        'agent-core/resource/llm_bench/report.json',
        'agent-core/resource/certs/server.crt',
        'agent-core/resource/channel_files/private.txt',
        'agent-core/resource/frame_rgb.jpg',
        'agent-core/tools/llm_bench/bench.yaml',
        'agent-core/deploy/.env',
        'agent-core/deploy/.env.local',
        'agent-core/deploy/server.key',
        'agent-core/deploy/server.pem',
        'agent-core/deploy/id_ed25519',
        'agent-core/src/__pycache__/agent_memory.cpython-310.pyc',
        'memory-core/src/memory_core/__pycache__/repository.cpython-314.pyc',
    ]

    assert all(_context_includes(path) for path in included)
    assert not any(_context_includes(path) for path in excluded)


def test_core_compose_exposes_persistent_memory_overrides():
    compose = yaml.safe_load(
        (_REPOSITORY_ROOT / 'agent-core' / 'deploy' / 'docker-compose.yml').read_text(
            encoding='utf-8'
        )
    )
    environment = compose['services']['agent-core']['environment']

    assert 'MEMORY_CORE_ENABLED=${MEMORY_CORE_ENABLED:-1}' in environment
    assert 'MEMORY_DB_PATH=${MEMORY_DB_PATH:-/work/resource/memory.db}' in environment


def test_all_cold_start_memory_templates_are_identical():
    memory_dir = _REPOSITORY_ROOT / 'agent-core' / 'resource' / 'memory'
    templates = [
        memory_dir / 'prompt_memory.md',
        memory_dir / 'prompt_memory_init.md',
        memory_dir / 'defaults' / 'prompt_memory_init.md',
    ]

    assert len({path.read_bytes() for path in templates}) == 1
