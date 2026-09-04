import asyncio
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import agent_memory
import config
import prompt
import start
from api import agent_definition, solutions
from api import config as config_api
from event.memory import Event as MemoryEvent


def _snapshot(
    text: str, revision: int, *, fallback_ready: bool = True
) -> agent_memory.MemorySnapshot:
    return agent_memory.MemorySnapshot(
        text=text,
        revision=revision,
        cache_key=f'core:test:{revision}',
        backend='core',
        fallback_ready=fallback_ready,
    )


def test_cold_start_seeds_from_packaged_source_not_shared_copy(
    tmp_path,
    monkeypatch,
):
    defaults = tmp_path / 'packaged-defaults'
    defaults.mkdir()
    (defaults / 'prompt_memory_init.md').write_text(
        'complete factory memory',
        encoding='utf-8',
    )
    shared_memory = tmp_path / 'resource' / 'memory'
    shared_memory.mkdir(parents=True)
    (shared_memory / 'prompt_memory_init.md').write_text(
        'partial copy',
        encoding='utf-8',
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(start, '_memory_defaults_dir', lambda: defaults)

    seed = start._init_resource_files()

    assert seed == 'complete factory memory'


def test_prompt_cache_changes_on_revision_but_frozen_turn_stays_unchanged(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / 'resource' / 'memory'
    memory_dir.mkdir(parents=True)
    system_path = memory_dir / 'prompt_system.md'
    identity_path = memory_dir / 'identity.md'
    system_path.write_text('system rules', encoding='utf-8')
    identity_path.write_text('agent identity', encoding='utf-8')

    state = {'snapshot': _snapshot('first durable memory', 1)}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config,
        'main',
        {'event': {'llm': {'prompt_system': str(system_path)}}},
    )
    monkeypatch.setattr(agent_memory, 'snapshot', lambda: state['snapshot'])
    monkeypatch.setattr(prompt, '_l1_cache', {'fingerprint': None, 'content': ''})
    monkeypatch.setattr(
        prompt, '_l2_static_cache', {'fingerprint': None, 'content': ''}
    )

    frozen_l1 = prompt.capture_l1()
    frozen = prompt.build_system({}, set(), frozen_l1=frozen_l1)
    state['snapshot'] = _snapshot('second durable memory', 2)
    same_turn_after_static_rebuild = prompt.build_system({}, set(), frozen_l1=frozen_l1)
    following_turn = prompt.build_system({}, set())

    assert 'first durable memory' in frozen['content']
    assert 'second durable memory' not in frozen['content']
    assert 'first durable memory' in same_turn_after_static_rebuild['content']
    assert 'second durable memory' not in same_turn_after_static_rebuild['content']
    assert 'second durable memory' in following_turn['content']


def test_llm_update_uses_fixed_server_side_actor(monkeypatch):
    calls = []

    async def replace(text, *, actor_key, reason):
        calls.append((text, actor_key, reason))
        return _snapshot(text.strip(), 3)

    monkeypatch.setattr(agent_memory, 'replace', replace)

    result = asyncio.run(MemoryEvent().update(' remember this '))

    assert calls == [(' remember this ', 'agent:main', 'llm_update')]
    assert 'revision=3' in result


def test_llm_update_reports_storage_failure_without_leaking_details(monkeypatch):
    async def replace(*args, **kwargs):
        raise agent_memory.AgentMemoryUnavailableError('secret path and sql')

    monkeypatch.setattr(agent_memory, 'replace', replace)

    result = asyncio.run(MemoryEvent().update('new memory'))

    assert '原有记忆已保留' in result
    assert 'secret' not in result


def test_llm_update_distinguishes_an_uncertain_file_commit(monkeypatch):
    async def replace(*args, **kwargs):
        raise agent_memory.AgentMemoryCommitUncertainError('secret path')

    monkeypatch.setattr(agent_memory, 'replace', replace)

    result = asyncio.run(MemoryEvent().update('new memory'))

    assert '无法确认' in result
    assert '原有记忆已保留' not in result
    assert 'secret' not in result


def test_agent_definition_reads_and_writes_memory_through_adapter(
    tmp_path, monkeypatch
):
    identity = tmp_path / 'identity.md'
    system = tmp_path / 'system.md'
    identity.write_text('old identity', encoding='utf-8')
    system.write_text('old system', encoding='utf-8')
    calls = []

    async def replace(text, *, actor_key, reason):
        calls.append((text, actor_key, reason))
        return _snapshot(text, 2)

    monkeypatch.setattr(agent_definition, '_IDENTITY_PATH', identity)
    monkeypatch.setattr(agent_definition, '_SYSTEM_PATH', system)
    monkeypatch.setattr(
        agent_memory, 'snapshot', lambda: _snapshot('database memory', 1)
    )
    monkeypatch.setattr(
        agent_memory,
        'status',
        lambda snapshot=None: {
            'backend': 'core',
            'revision': 1,
            'enabled': True,
            'fallback_ready': True,
            'warning': None,
        },
    )
    monkeypatch.setattr(agent_memory, 'replace', replace)

    before = asyncio.run(agent_definition.get_definition())
    saved = asyncio.run(
        agent_definition.save_definition(
            agent_definition.DefinitionSaveRequest(
                identity='new identity', system='new system', memory='new memory'
            )
        )
    )

    assert before['data']['memory'] == 'database memory'
    assert calls == [('new memory', 'api:agent_definition', 'api_edit')]
    assert identity.read_text(encoding='utf-8') == 'new identity'
    assert system.read_text(encoding='utf-8') == 'new system'
    assert saved['code'] == 200


def test_solution_prompt_apply_routes_memory_before_file_prompts(tmp_path, monkeypatch):
    identity = tmp_path / 'identity.md'
    system = tmp_path / 'system.md'
    mirror = tmp_path / 'prompt_memory.md'
    calls = []

    async def replace(text, *, actor_key, reason):
        calls.append((text, actor_key, reason, identity.exists(), system.exists()))
        return _snapshot(text, 4)

    monkeypatch.setattr(
        solutions,
        '_prompt_paths',
        lambda: {'identity': identity, 'system': system, 'memory': mirror},
    )
    monkeypatch.setattr(agent_memory, 'replace', replace)

    written, warning = asyncio.run(
        solutions._apply_prompt(
            {'identity': 'id', 'system': 'rules', 'memory': 'facts'},
            ['prompt.identity', 'prompt.system', 'prompt.memory'],
        )
    )

    assert calls == [('facts', 'api:solutions', 'solution_apply', False, False)]
    assert identity.read_text(encoding='utf-8') == 'id'
    assert system.read_text(encoding='utf-8') == 'rules'
    assert not mirror.exists()
    assert set(written) == {'prompt.identity', 'prompt.system', 'prompt.memory'}
    assert warning is None


def test_solution_apply_warns_when_compatibility_copy_is_not_ready(monkeypatch):
    async def replace(text, *, actor_key, reason):
        return _snapshot(text, 4, fallback_ready=False)

    monkeypatch.setattr(config, 'main', {})
    monkeypatch.setattr(solutions, '_canvas_editor_conflict', lambda session_id: None)
    monkeypatch.setattr(agent_memory, 'replace', replace)

    result = asyncio.run(
        solutions.apply(
            SimpleNamespace(headers={}),
            solutions.LoadRequest(
                payload={
                    'formatVersion': solutions.FORMAT_VERSION,
                    'devices': [],
                    'prompt': {'memory': 'new memory'},
                },
                includes=['prompt.memory'],
                confirm=True,
            ),
        )
    )

    assert result['code'] == 200
    assert result['data']['applied']['prompt'] == ['prompt.memory']
    assert result['data']['warning'] == agent_memory.COMPATIBILITY_WARNING


@pytest.mark.parametrize(
    ('failure', 'expected_code'),
    [
        (agent_memory.AgentMemoryValidationError('empty'), 422),
        (agent_memory.AgentMemoryUnavailableError('offline'), 503),
        (agent_memory.AgentMemoryCommitUncertainError('uncertain'), 503),
    ],
)
def test_solution_apply_memory_failure_prevents_other_side_effects(
    tmp_path, monkeypatch, failure, expected_code
):
    identity = tmp_path / 'identity.md'
    system = tmp_path / 'system.md'
    mirror = tmp_path / 'prompt_memory.md'
    side_effects = []

    async def replace(*args, **kwargs):
        raise failure

    async def apply_canvas(*args, **kwargs):
        side_effects.append('canvas')

    async def apply_skills(*args, **kwargs):
        side_effects.append('skills')

    def apply_tasks(*args, **kwargs):
        side_effects.append('tasks')

    monkeypatch.setattr(solutions, '_canvas_editor_conflict', lambda session_id: None)
    monkeypatch.setattr(
        solutions,
        '_prompt_paths',
        lambda: {'identity': identity, 'system': system, 'memory': mirror},
    )
    monkeypatch.setattr(solutions, '_apply_canvas', apply_canvas)
    monkeypatch.setattr(solutions, '_apply_skills', apply_skills)
    monkeypatch.setattr(solutions, '_apply_tasks', apply_tasks)
    monkeypatch.setattr(agent_memory, 'replace', replace)

    includes = [
        solutions.BLOCK_CANVAS,
        solutions.BLOCK_SKILLS,
        'prompt.identity',
        'prompt.system',
        'prompt.memory',
        solutions.BLOCK_TASKS,
    ]
    result = asyncio.run(
        solutions.apply(
            SimpleNamespace(headers={}),
            solutions.LoadRequest(
                payload={
                    'formatVersion': solutions.FORMAT_VERSION,
                    'devices': [],
                    'canvas': {},
                    'skills': [],
                    'prompt': {
                        'identity': 'new identity',
                        'system': 'new system',
                        'memory': 'new memory',
                    },
                    'tasks': [],
                },
                includes=includes,
                confirm=True,
            ),
        )
    )

    assert result['code'] == expected_code
    if isinstance(failure, agent_memory.AgentMemoryCommitUncertainError):
        assert '无法确认' in result['error']
    assert side_effects == []
    assert not identity.exists()
    assert not system.exists()
    assert not mirror.exists()


def test_memory_reset_uses_adapter_and_local_default_fallback(tmp_path, monkeypatch):
    defaults = tmp_path / 'defaults'
    defaults.mkdir()
    (defaults / 'prompt_memory_init.md').write_text('factory memory', encoding='utf-8')
    calls = []

    async def replace(text, *, actor_key, reason):
        calls.append((text, actor_key, reason))
        return _snapshot(text, 5)

    monkeypatch.setattr(config_api, '_memory_defaults_dir', lambda: defaults)
    monkeypatch.setattr(agent_memory, 'replace', replace)

    result = asyncio.run(config_api.reset_config(config_api.ResetRequest(memory=True)))

    assert calls == [('factory memory', 'system:reset', 'memory_reset')]
    assert result['ok'] is True
    assert result['reset'] == ['memory']


def test_memory_reset_reports_an_uncertain_commit(tmp_path, monkeypatch):
    defaults = tmp_path / 'defaults'
    defaults.mkdir()
    (defaults / 'prompt_memory_init.md').write_text('factory memory', encoding='utf-8')

    async def replace(*args, **kwargs):
        raise agent_memory.AgentMemoryCommitUncertainError('secret path')

    monkeypatch.setattr(config_api, '_memory_defaults_dir', lambda: defaults)
    monkeypatch.setattr(agent_memory, 'replace', replace)

    with pytest.raises(Exception) as raised:
        asyncio.run(config_api.reset_config(config_api.ResetRequest(memory=True)))

    assert getattr(raised.value, 'status_code', None) == 503
    assert '无法确认' in getattr(raised.value, 'detail', '')
    assert 'secret' not in getattr(raised.value, 'detail', '')


def test_human_write_apis_return_compatibility_warning(tmp_path, monkeypatch):
    identity = tmp_path / 'identity.md'
    system = tmp_path / 'system.md'
    defaults = tmp_path / 'defaults'
    defaults.mkdir()
    (defaults / 'prompt_memory_init.md').write_text('factory memory', encoding='utf-8')

    async def replace(text, *, actor_key, reason):
        return _snapshot(text, 7, fallback_ready=False)

    monkeypatch.setattr(agent_definition, '_IDENTITY_PATH', identity)
    monkeypatch.setattr(agent_definition, '_SYSTEM_PATH', system)
    monkeypatch.setattr(config_api, '_memory_defaults_dir', lambda: defaults)
    monkeypatch.setattr(agent_memory, 'replace', replace)

    saved = asyncio.run(
        agent_definition.save_definition(
            agent_definition.DefinitionSaveRequest(
                identity='identity',
                system='system',
                memory='memory',
            )
        )
    )
    reset = asyncio.run(config_api.reset_config(config_api.ResetRequest(memory=True)))

    assert saved['warning'] == agent_memory.COMPATIBILITY_WARNING
    assert saved['memoryStatus']['fallback_ready'] is False
    assert reset['warning'] == agent_memory.COMPATIBILITY_WARNING
    assert reset['memoryStatus']['fallback_ready'] is False


def test_empty_definition_memory_fails_before_other_files_change(tmp_path, monkeypatch):
    identity = tmp_path / 'identity.md'
    system = tmp_path / 'system.md'
    identity.write_text('old identity', encoding='utf-8')
    system.write_text('old system', encoding='utf-8')

    async def replace(*args, **kwargs):
        raise agent_memory.AgentMemoryValidationError('empty')

    monkeypatch.setattr(agent_definition, '_IDENTITY_PATH', identity)
    monkeypatch.setattr(agent_definition, '_SYSTEM_PATH', system)
    monkeypatch.setattr(agent_memory, 'replace', replace)

    with pytest.raises(Exception) as raised:
        asyncio.run(
            agent_definition.save_definition(
                agent_definition.DefinitionSaveRequest(
                    identity='new identity', system='new system', memory=''
                )
            )
        )

    assert getattr(raised.value, 'status_code', None) == 422
    assert identity.read_text(encoding='utf-8') == 'old identity'
    assert system.read_text(encoding='utf-8') == 'old system'


def test_human_write_api_reports_an_uncertain_memory_commit(tmp_path, monkeypatch):
    identity = tmp_path / 'identity.md'
    system = tmp_path / 'system.md'
    identity.write_text('old identity', encoding='utf-8')
    system.write_text('old system', encoding='utf-8')

    async def replace(*args, **kwargs):
        raise agent_memory.AgentMemoryCommitUncertainError('secret path')

    monkeypatch.setattr(agent_definition, '_IDENTITY_PATH', identity)
    monkeypatch.setattr(agent_definition, '_SYSTEM_PATH', system)
    monkeypatch.setattr(agent_memory, 'replace', replace)

    with pytest.raises(Exception) as raised:
        asyncio.run(
            agent_definition.save_definition(
                agent_definition.DefinitionSaveRequest(
                    identity='new identity',
                    system='new system',
                    memory='new memory',
                )
            )
        )

    assert getattr(raised.value, 'status_code', None) == 503
    assert '无法确认' in getattr(raised.value, 'detail', '')
    assert 'secret' not in getattr(raised.value, 'detail', '')
    assert identity.read_text(encoding='utf-8') == 'old identity'
    assert system.read_text(encoding='utf-8') == 'old system'
