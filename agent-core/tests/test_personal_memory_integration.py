"""Exercise the real collector, prompts, tools and persistence without hardware."""
import asyncio
import json
import sys
import time

import pytest
import config
import personal_memory as pm
import collector
import prompt
from event.llm import Event, _memory_recall
from llm_logger import LLMLogger
from test_personal_memory import event, store


def test_collector_never_batches_two_personal_events(store):
    batch = [event('dds:/mic/asr', text='a'), event('dds:/mic/asr', text='b')]
    groups = collector._split_by_bot_trust(batch)
    assert len(groups) == 2
    assert groups[0][0]['text'] != groups[1][0]['text']


def test_collector_authenticates_raw_events_not_formatted_user_text(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_identity(a['id'], 'voice', 'v-1')
    pm.resolver = pm.IdentityResolver()
    trigger = collector._build_trigger([event('dds:/mic/asr', speaker_id='v-1', text='hi')], True)
    assert trigger['_personal_user_id'] == a['id']
    remote = collector._build_trigger([event('dds:/remote_control/message', speaker_id='v-1', text='hi')], True)
    assert remote['_personal_user_id'] is None


def test_prompt_does_not_read_legacy_shared_memory_or_other_users(store, tmp_path, monkeypatch):
    # A stale shared L1 cache must never be reused in personal mode.
    monkeypatch.setitem(prompt._l1_cache, 'content', 'legacy secret')
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_context(a['id'])
    pm.save_memory('Alice secret', 'fact', 'a')
    first = prompt.build_system({}, set())['content']
    assert 'Alice secret' in first
    assert 'legacy secret' not in first
    pm.bind_context(b['id'])
    second = prompt.build_system({}, set())['content']
    assert 'Alice secret' not in second
    assert 'legacy secret' not in second
    assert '心理' in second


def test_switch_resets_history_and_summaries_and_guest_cannot_resume(store):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    loop = Event()
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    loop._current_turn = [{'role': 'user', 'content': 'Alice transcript'}]
    loop._save_current_turn({})
    loop._summary = 'Alice summary'
    loop._prepare_personal_turn({'_personal_user_id': b['id']})
    assert 'Alice' not in json.dumps(loop._build_history())
    assert loop._summary is None
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    assert 'Alice transcript' in json.dumps(loop._build_history())
    # A new runtime starts empty until a fresh biometric match arrives.
    restarted = Event()
    restarted._prepare_personal_turn({})
    assert restarted._build_history() == []
    with config._get_conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM chat_messages').fetchone()[0] == 0


def test_recall_never_searches_legacy_history(store):
    import chat_history
    sid = chat_history.create_session()
    chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'legacysecret'}])
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    pm.save_memory('mysecret', 'analysis', 'a')
    result = asyncio.run(_memory_recall('secret'))
    assert 'mysecret' in result
    assert 'legacysecret' not in result


def test_personal_mode_disables_content_request_logs(store, tmp_path, monkeypatch):
    config.main['llm_logger'] = {'enabled': True, 'data_dir': str(tmp_path / 'logs'),
                                 'recent_dir': str(tmp_path / 'recent')}
    logger = LLMLogger()
    asyncio.run(logger.log_request('1', '1', {}, [{'role': 'user', 'content': 'secret'}], [], 'test'))
    asyncio.run(logger.log_response('1', '1', {}, {'content': 'secret'}))
    assert not list((tmp_path / 'logs').glob('*'))
    assert not list((tmp_path / 'recent').rglob('*.json'))


def test_private_tools_are_checked_even_when_model_invents_call(store):
    assert pm.tool_allowed('memory_recall')
    for name in ['Read', 'Bash', 'search_history', 'raw_input_info', 'update_memory',
                 'peer_delegate', 'subagent_spawn', 'task_list', 'mcp__perception__list_persons']:
        assert not pm.tool_allowed(name)
    config.main['personal_memory'] = {'enabled': False}
    assert pm.tool_allowed('Bash')
    assert pm.tool_allowed('subagent_spawn')


def test_dynamic_snapshot_does_not_expose_shared_tasks(store, monkeypatch):
    import task_store
    class Task:
        id = 'secret-task'
        goal = 'Alice private goal'
        progress = 'Alice progress'
        status = 'running'
        created_at = time.time()
    monkeypatch.setattr(task_store, 'active_tasks', lambda: [Task()])
    assert 'Alice' not in prompt._env_dynamic()


def test_scoped_tools_record_origin_and_distinguish_hypotheses(store):
    from personal_memory_tools import Tools
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    pm._turn.set('origin-turn')
    result = asyncio.run(Tools().remember_personal('可能与工作有关', 'analysis'))
    assert '可能与工作有关' in result
    item = pm.recall('')[0]
    assert item['kind'] == 'analysis'
    assert item['source_turn'] == 'origin-turn'
    asyncio.run(Tools().set_memory_preferences(False, False))
    assert pm.recall('') == []


def test_http_request_debug_hook_does_not_dump_private_payload(store, tmp_path, monkeypatch):
    import httpx
    llm = sys.modules['client.llm']
    monkeypatch.setattr(llm, 'LOG_PATH', tmp_path)
    request = httpx.Request('POST', 'http://model.invalid', json={'model': 'test', 'messages': ['secret']})
    asyncio.run(llm._log_request(request))
    assert list(tmp_path.glob('llm_request_*.json')) == []


def test_disabling_memory_removes_approved_derivatives_on_account_deletion(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_context(a['id'])
    item = pm.propose_public('approved')
    pm.approve_public(item['id'])
    pm.delete_user(a['id'])
    assert pm.recall('') == []


def test_real_turn_dispatch_cannot_call_unadvertised_desktop_tool(store, monkeypatch):
    """A hallucinated tool call must be denied at dispatch, not just hidden in schema."""
    from types import SimpleNamespace
    import client
    llm = sys.modules['event.llm']
    loop = Event()
    loop._desktop_tools = SimpleNamespace(reset_python_namespace=lambda: None)
    calls = []

    async def forbidden(path):
        calls.append(path)
        return 'other user secret'

    async def finish():
        return 'done'

    loop._sys_tools = llm._build_system_tools([('Read', forbidden), ('finish', finish)])
    seen = []

    async def model(**kwargs):
        seen.append(kwargs)
        if len(seen) == 1:
            return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'bad', 'type': 'function', 'function': {
                'name': 'Read', 'arguments': '{"path": "/other-user"}'}}]}
        return {'role': 'assistant', 'content': 'done', 'tool_calls': [{'id': 'end', 'type': 'function', 'function': {
            'name': 'finish', 'arguments': '{}'}}]}

    monkeypatch.setattr(client, 'call', model)
    monkeypatch.setitem(sys.modules, 'ros2_bridge', SimpleNamespace(publish=lambda *args: None))
    monkeypatch.setattr(collector, '_steering_queue', asyncio.Queue())
    loop._prepare_personal_turn({})
    asyncio.run(loop._one_turn({'source': 'collector', 'text': 'hello', 'ts': time.time()}))
    assert calls == []
    assert len(seen) == 2
    names = [t['function']['name'] for t in seen[0]['tool_list']]
    assert 'Read' not in names
    assert 'unavailable in personal memory mode' in json.dumps(seen[1]['message_list'])


def test_recall_paginates_without_revealing_another_user(store):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_context(a['id'])
    pm.save_memory('first', 'fact', 't1')
    pm.save_memory('second', 'fact', 't2')
    pm.bind_context(b['id'])
    pm.save_memory('Bob secret', 'fact', 't3')
    pm.bind_context(a['id'])
    result = json.loads(asyncio.run(_memory_recall('', limit=1, offset=1)))
    assert result['memories'][0]['content'] == 'first'
    assert 'Bob' not in json.dumps(result)


def test_queued_speech_keeps_identity_from_arrival_not_later_face(store, monkeypatch):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_identity(a['id'], 'face', 'alice-face')
    pm.bind_identity(b['id'], 'face', 'bob-face')
    pm.resolver.resolve(event('dds:/camera/face', faces=[{'person_id': 'alice-face', 'quality': 'ok'}]))
    speech = event('dds:/mic/asr', text='Alice disclosure')
    collector._capture_personal_identity(speech)
    pm.resolver.resolve(event('dds:/camera/face', faces=[{'person_id': 'bob-face', 'quality': 'ok'}]))
    monkeypatch.setattr(collector, '_priority_pending', [speech])
    monkeypatch.setattr(collector, '_steering_queue', asyncio.Queue())
    monkeypatch.setattr(collector, '_output', asyncio.Queue())
    collector._flush_all_pending()
    trigger = collector._output.get_nowait()
    assert trigger['_personal_user_id'] == a['id']
    assert trigger['_personal_user_id'] != b['id']


def test_private_decision_is_not_broadcast_to_shared_dds(store, monkeypatch):
    from types import SimpleNamespace
    import client
    llm = sys.modules['event.llm']
    loop = Event()
    loop._desktop_tools = SimpleNamespace(reset_python_namespace=lambda: None)

    async def finish():
        return 'done'

    loop._sys_tools = llm._build_system_tools([('finish', finish)])

    async def model(**kwargs):
        return {'role': 'assistant', 'content': 'Alice private assessment',
                'tool_calls': [{'id': 'end', 'type': 'function', 'function': {
                    'name': 'finish', 'arguments': '{}'}}]}

    published = []
    monkeypatch.setattr(client, 'call', model)
    monkeypatch.setitem(sys.modules, 'ros2_bridge', SimpleNamespace(publish=lambda *args: published.append(args)))
    monkeypatch.setattr(collector, '_steering_queue', asyncio.Queue())
    loop._prepare_personal_turn({})
    asyncio.run(loop._one_turn({'source': 'collector', 'text': 'hello', 'ts': time.time()}))
    assert not any(topic == '/decision_core' for topic, _ in published)


def test_opt_out_retains_only_current_users_short_term_conversation(store):
    a = pm.create_user('Alice', remember=False)
    b = pm.create_user('Bob', remember=False)
    loop = Event()
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    loop._current_turn = [{'role': 'user', 'content': 'just this conversation'}]
    loop._save_current_turn({})
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    assert 'just this conversation' in json.dumps(loop._build_history())
    assert pm.load_turns() == []
    loop._prepare_personal_turn({'_personal_user_id': b['id']})
    assert loop._build_history() == []
    loop._prepare_personal_turn({'_personal_user_id': a['id']})
    assert loop._build_history() == []


def test_busy_ingress_keeps_identical_words_from_different_people(store, monkeypatch):
    import event_bus
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_identity(a['id'], 'face', 'alice-face')
    pm.bind_identity(b['id'], 'face', 'bob-face')
    pending = []
    monkeypatch.setattr(collector, '_priority_pending', pending)
    monkeypatch.setattr(collector, '_steering_queue', asyncio.Queue())
    monkeypatch.setattr(collector, '_busy', True)
    monkeypatch.setattr(collector, '_cancel_event', asyncio.Event())
    config.main['core'] = {'project_running': True}
    events = iter([
        event('dds:/camera/face', faces=[{'person_id': 'alice-face', 'quality': 'ok'}]),
        event('dds:/mic/asr', text='我想聊聊', priority=2),
        event('dds:/camera/face', faces=[{'person_id': 'bob-face', 'quality': 'ok'}]),
        event('dds:/mic/asr', text='我想聊聊', priority=2),
    ])

    async def dequeue():
        try:
            return next(events)
        except StopIteration:
            raise asyncio.CancelledError

    monkeypatch.setattr(event_bus, 'dequeue', dequeue)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collector._drain_loop())
    assert len(pending) == 2
    assert [ev['_personal_identity'].user_id for ev in pending] == [a['id'], b['id']]
