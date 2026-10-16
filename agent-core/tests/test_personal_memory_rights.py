"""User-owned controls are enforced by SQLite scope, even while recording is paused."""
import asyncio
import json
import sys

import pytest
import personal_memory as pm
import event
from personal_memory_tools import Tools
from test_personal_memory import store


def test_pause_preserves_existing_data_but_blocks_new_records(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    saved = pm.save_memory('existing preference', 'preference', 'old')
    pm.set_collection(False)
    assert pm.recall('')[0]['id'] == saved['id']
    with pytest.raises(PermissionError):
        pm.save_memory('new information', 'fact', 'new')
    pm.save_turn('new', [{'role': 'user', 'content': 'not recorded'}])
    assert pm.load_turns() == []
    assert pm.delete_memory(saved['id']) is True


def test_export_is_owner_scoped_and_paginated_even_when_paused(store):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_context(a['id'])
    pm.save_memory('a-first', 'fact', 'a1')
    pm.save_memory('a-second', 'analysis', 'a2')
    pm.bind_identity(a['id'], 'face', 'face-a')
    pm.bind_context(b['id'])
    pm.save_memory('Bob secret', 'fact', 'b1')
    pm.bind_context(a['id'])
    pm.set_collection(False)
    page = pm.export_data('memories', limit=1)
    assert page['total'] == 2
    assert page['next_offset'] == 1
    other = pm.export_data('memories', limit=1, offset=1)
    assert other['next_offset'] is None
    assert {page['items'][0]['content'], other['items'][0]['content']} == {'a-first', 'a-second'}
    assert 'Bob' not in json.dumps(page)
    assert pm.export_data('bindings')['items'][0]['external_id'] == 'face-a'
    pm.bind_context(None)
    with pytest.raises(PermissionError):
        pm.export_data('memories')


def test_correction_and_deletion_clear_old_transcripts_without_touching_other_users(store):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_context(a['id'])
    old = pm.save_memory('incorrect secret', 'analysis', 'old')
    keep = pm.save_memory('unrelated preference', 'preference', 'other')
    pm.save_turn('old', [{'role': 'user', 'content': 'incorrect secret'}])
    pm.save_turn('echo', [{'role': 'assistant', 'content': 'repeated incorrect secret'}])
    pm.bind_context(b['id'])
    theirs = pm.save_memory('Bob secret', 'fact', 'b')
    with pytest.raises(PermissionError):
        pm.correct_memory(old['id'], 'tampered', 'fact')
    pm.bind_context(a['id'])
    corrected = pm.correct_memory(old['id'], 'corrected self-report', 'fact')
    assert corrected['id'] == old['id']
    assert pm.recall('incorrect') == []
    assert pm.load_turns() == []
    assert pm.recall('unrelated')[0]['id'] == keep['id']
    assert pm.delete_memory(old['id']) is True
    assert pm.recall('corrected') == []
    pm.bind_context(b['id'])
    assert pm.recall('Bob')[0]['id'] == theirs['id']


def test_user_can_withdraw_public_contribution_and_clear_own_profile(store):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_context(a['id'])
    pm.bind_identity(a['id'], 'voice', 'voice-a')
    contribution = pm.propose_public('generic contribution')
    pm.approve_public(contribution['id'])
    pm.bind_context(b['id'])
    assert pm.withdraw_public(contribution['id']) is False
    pm.bind_context(a['id'])
    assert pm.withdraw_public(contribution['id']) is True
    pm.clear_personal_data(delete_profile=True)
    assert pm.get_user(a['id']) is None
    assert pm.get_user(b['id']) is not None
    with pytest.raises(PermissionError):
        pm.export_data('bindings')


def test_clear_data_keeps_identity_but_disables_recording_and_public_items(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    item = pm.propose_public('contribution')
    pm.approve_public(item['id'])
    pm.clear_personal_data(delete_profile=False)
    assert pm.get_user(a['id'])['remember'] == 0
    assert pm.recall('') == []


def test_builtin_memory_skill_is_only_loaded_in_personal_mode(store):
    import config
    import importlib
    skills = importlib.import_module('event.skills')
    assert [s['slug'] for s in skills.get_active_skills()] == ['personal-memory']
    assert 'export_personal_data' in skills.get_active_skills()[0]['instruction']
    config.main['personal_memory'] = {'enabled': False}
    assert 'personal-memory' not in [s['slug'] for s in skills.get_active_skills()]


def test_tools_require_confirmation_before_clearing_history(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    item = pm.save_memory('old secret', 'fact', 'source')
    result = asyncio.run(Tools().forget_personal(item['id']))
    assert '确认' in result
    assert pm.recall('old secret')


def test_real_turn_forget_rebuilds_prompt_and_cannot_save_deleted_data_again(store, monkeypatch):
    from types import SimpleNamespace
    import time
    import client
    import collector
    llm = sys.modules['event.llm']
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    item = pm.save_memory('old secret', 'fact', 'source')
    pm.save_turn('source', [{'role': 'user', 'content': 'old secret'}])
    loop = llm.Event()
    loop._desktop_tools = SimpleNamespace(reset_python_namespace=lambda: None)
    tools = Tools()

    async def finish():
        return 'done'

    loop._sys_tools = llm._build_system_tools([('forget_personal', tools.forget_personal), ('finish', finish)])
    seen = []

    async def model(**kwargs):
        seen.append(json.dumps(kwargs['message_list'], ensure_ascii=False))
        if len(seen) == 1:
            return {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'forget', 'type': 'function', 'function': {'name': 'forget_personal',
                 'arguments': json.dumps({'memory_id': item['id'], 'confirmed': True})}},
                {'id': 'end-first', 'type': 'function', 'function': {'name': 'finish', 'arguments': '{}'}}]}
        return {'role': 'assistant', 'content': '已完成资料管理', 'tool_calls': [
            {'id': 'end', 'type': 'function', 'function': {'name': 'finish', 'arguments': '{}'}}]}

    monkeypatch.setattr(client, 'call', model)
    monkeypatch.setitem(sys.modules, 'ros2_bridge', SimpleNamespace(publish=lambda *args: None))
    monkeypatch.setattr(collector, '_steering_queue', asyncio.Queue())
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    asyncio.run(loop._one_turn({'source': 'collector', 'text': 'delete old secret', 'ts': time.time()}))
    loop._save_current_turn({})
    assert len(seen) == 2
    assert 'old secret' in seen[0]
    assert 'old secret' not in seen[1]
    assert pm.load_turns() == []
    assert pm.recall('old secret') == []


def test_destructive_preferences_require_confirmation(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    pm.save_memory('keep until confirmed', 'fact', 'origin')
    result = asyncio.run(Tools().set_memory_preferences(False, False))
    assert '确认' in result
    assert pm.recall('keep until confirmed')


def test_existing_four_column_user_database_is_upgraded_without_data_loss(store):
    import config
    with config._get_conn() as conn:
        conn.execute('CREATE TABLE pm_users (id TEXT PRIMARY KEY, name TEXT NOT NULL, remember INTEGER NOT NULL, onboarding INTEGER NOT NULL)')
        conn.execute("INSERT INTO pm_users VALUES ('old-user', 'Alice', 1, 0)")
        conn.commit()
    old = pm.get_user('old-user')
    assert old['name'] == 'Alice'
    assert old['paused'] == 0
    assert old['revision'] == 0


@pytest.mark.parametrize('during_barrier', [False, True])
def test_external_revocation_blocks_stale_output_and_turn_save(store, monkeypatch, during_barrier):
    from types import SimpleNamespace
    import time
    import client
    import collector
    import mcp_client
    llm = sys.modules['event.llm']
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    pm.save_memory('private fact', 'fact', 'origin')
    loop = llm.Event()
    loop._desktop_tools = SimpleNamespace(reset_python_namespace=lambda: None)

    async def finish():
        return 'done'

    loop._sys_tools = llm._build_system_tools([('finish', finish)])
    cfg = pm.configured_settings()
    cfg['output_tools'] = ['mcp__local__tts']
    import config
    config.main['personal_memory'] = cfg
    seen, dispatched = [], []

    async def model(**kwargs):
        seen.append(json.dumps(kwargs['message_list'], ensure_ascii=False))
        if len(seen) == 1:
            if not during_barrier:
                pm.update_user(a['id'], False, False)  # Revocation while the model request is in flight.
            return {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'speak', 'type': 'function', 'function': {
                    'name': 'mcp__local__tts', 'arguments': '{"text":"private fact"}'}}]}
        return {'role': 'assistant', 'content': '已撤回', 'tool_calls': [
            {'id': 'end', 'type': 'function', 'function': {'name': 'finish', 'arguments': '{}'}}]}

    async def call_tool(*args, **kwargs):
        dispatched.append(args)
        return 'done'

    monkeypatch.setattr(client, 'call', model)
    monkeypatch.setattr(mcp_client, 'call_tool', call_tool)
    if during_barrier:
        revoked = []

        async def barrier(*args, **kwargs):
            if not revoked:
                revoked.append(True)
                pm.update_user(a['id'], False, False)
            return None

        monkeypatch.setattr(llm, '_needs_barrier', lambda *args, **kwargs: (True, frozenset()))
        monkeypatch.setattr(llm, '_acp_barrier', barrier)
    monkeypatch.setitem(sys.modules, 'ros2_bridge', SimpleNamespace(publish=lambda *args: None))
    monkeypatch.setattr(collector, '_steering_queue', asyncio.Queue())
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    asyncio.run(loop._one_turn({'source': 'collector', 'text': 'recall', 'ts': time.time()}))
    loop._save_current_turn({})
    assert dispatched == []
    assert len(seen) == 2
    assert 'private fact' not in seen[1]
    assert pm.load_turns() == []


@pytest.mark.parametrize('operation', ['memory', 'public'])
def test_revocation_between_permission_read_and_insert_cannot_resurrect_data(store, monkeypatch, operation):
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    read_permission = pm._writing_user

    def revoked_after_read():
        previous = read_permission()
        pm.update_user(a['id'], False, False)
        return previous

    monkeypatch.setattr(pm, '_writing_user', revoked_after_read)
    with pytest.raises(PermissionError):
        if operation == 'memory':
            pm.save_memory('should never reappear', 'fact', 'racing-turn')
        else:
            pm.propose_public('should never reappear')
    assert pm.export_data('memories')['total'] == 0
    assert pm.export_data('public_contributions')['total'] == 0


def test_revocation_between_turn_permission_read_and_upsert_cannot_resurrect_history(store, monkeypatch):
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    original = pm.get_user
    revoked = []

    def user_then_revoke(user_id):
        previous = original(user_id)
        if previous and not revoked:
            revoked.append(True)
            pm.update_user(a['id'], False, False)
        return previous

    monkeypatch.setattr(pm, 'get_user', user_then_revoke)
    pm.save_turn('racing-turn', [{'role': 'user', 'content': 'must not return'}])
    assert pm.export_data('conversations')['total'] == 0


def test_idle_revocation_invalidates_opt_out_transient_cache_before_next_turn(store):
    llm = sys.modules['event.llm']
    a = pm.create_user('Alice', remember=False)
    loop = llm.Event()
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    loop._current_turn = [{'role': 'user', 'content': 'deleted transient secret'}]
    loop._save_current_turn({})
    assert 'deleted transient secret' in str(loop._transient_turns)
    pm.update_user(a['id'], False, False)
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    assert 'deleted transient secret' not in str(loop._build_history())


def test_revocation_after_model_finishes_cannot_cache_the_stale_turn(store):
    llm = sys.modules['event.llm']
    a = pm.create_user('Alice', remember=True)
    loop = llm.Event()
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    loop._current_turn = [{'role': 'assistant', 'content': 'deleted trailing secret'}]
    pm.update_user(a['id'], False, False)
    loop._save_current_turn({})
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    assert 'deleted trailing secret' not in str(loop._build_history())
    assert loop._transient_turns == []
