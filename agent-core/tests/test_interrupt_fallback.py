"""The barge-in fallback must actually reach the device.

`_interrupt_active_outputs` runs when no `on_interrupt_all` hook binding exists --
the path taken on R1. It looked up tools from `registry[mcp_id]['tools']`, which
holds *bare* plugin names ('tts', 'loco'), and passed them to `call_tool()`.

`call_tool` parses its argument as a full `mcp__<id>__<tool>` name. Given 'loco' it
split into one segment, failed `len(parts) != 3`, and returned the *string*
'工具名格式错误: loco'. A string is not an exception, so the error check never
fired and the log still said "interrupted 2 active output(s)". Nothing had been
interrupted -- on any robot, for both TTS and locomotion.

Observed on R1 2026-09-02: `[decision] interrupted 3 active output(s) (fallback)`
during a barge-in, with no corresponding call reaching the driver.

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
     tests/test_interrupt_fallback.py -q
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402
import hooks  # noqa: E402
from event.llm import Event as DecisionLoop  # noqa: E402


@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(mcp_client.registry)
    mcp_client.registry.clear()
    yield
    mcp_client.registry.clear()
    mcp_client.registry.update(saved)


@pytest.fixture(autouse=True)
def no_hooks(monkeypatch):
    """Force the fallback path -- with a binding registered it is never reached."""
    async def _fire(hook_id, *a, **kw):
        return []
    monkeypatch.setattr(hooks, 'fire', _fire)


@pytest.fixture
def direct_calls(monkeypatch):
    """Record call_tool_direct instead of hitting a device."""
    seen = []

    async def _direct(mcp_id, tool_name, args):
        seen.append((mcp_id, tool_name, args))
        return {'ok': True}

    monkeypatch.setattr(mcp_client, 'call_tool_direct', _direct)
    return seen


@pytest.fixture
def forbid_call_tool(monkeypatch):
    """call_tool cannot be used here: it needs a full mcp__id__tool name, and the
    registry only has bare ones. Fail loudly if the old path comes back."""
    async def _call_tool(full_name, args):
        raise AssertionError(f'call_tool must not be used for interrupts: {full_name!r}')

    monkeypatch.setattr(mcp_client, 'call_tool', _call_tool)


def _register(mcp_id='mcp-1', tools=('tts', 'loco'), online=True):
    # Exactly the shape mcp_client._connect_one builds: bare plugin names.
    mcp_client.registry[mcp_id] = {
        'name': mcp_id, 'url': f'http://localhost/{mcp_id}', 'online': online,
        'tools': list(tools), 'schemas': {}, 'input_schemas': {}, 'tool_meta': {},
    }


def _run(loop=None):
    loop = loop or DecisionLoop.__new__(DecisionLoop)
    asyncio.run(loop._interrupt_active_outputs())


# ── the regression ───────────────────────────────────────────────────────────

def test_interrupt_reaches_tts_and_loco(direct_calls, forbid_call_tool):
    _register(tools=['mic', 'tts', 'speaker', 'loco', 'switch_mode', 'arm'])
    _run()
    assert sorted(direct_calls) == [
        ('mcp-1', 'loco', {'action': 'stop_move'}),
        ('mcp-1', 'tts', {'action': 'interrupt'}),
    ]


def test_bare_tool_names_are_not_mangled(direct_calls, forbid_call_tool):
    """'loco' must be passed as-is with its mcp_id, not parsed as a full name."""
    _register(tools=['loco'])
    _run()
    assert direct_calls == [('mcp-1', 'loco', {'action': 'stop_move'})]


def test_split_tools_do_not_hide_the_plugin(direct_calls, forbid_call_tool):
    """R1's loco is split by x-action-params into loco__move / loco__stop_move, but
    registry['tools'] still lists the bare 'loco' -- the split names live in
    'schemas'/'tool_groups'. The old `t.split('__')[-1]` reasoning was aimed at
    those; this asserts we key off the bare list."""
    mcp_client.registry['mcp-1'] = {
        'name': 'mcp-1', 'url': 'http://x', 'online': True,
        'tools': ['loco', 'tts'],
        'tool_groups': {'loco': ['mcp__mcp-1__loco__move',
                                 'mcp__mcp-1__loco__stop_move']},
        'schemas': {}, 'input_schemas': {}, 'tool_meta': {},
    }
    _run()
    assert ('mcp-1', 'loco', {'action': 'stop_move'}) in direct_calls


def test_every_online_device_is_interrupted(direct_calls, forbid_call_tool):
    _register('mcp-1', tools=['tts'])
    _register('mcp-2', tools=['loco'])
    _run()
    assert sorted(direct_calls) == [
        ('mcp-1', 'tts', {'action': 'interrupt'}),
        ('mcp-2', 'loco', {'action': 'stop_move'}),
    ]


def test_offline_devices_are_skipped(direct_calls, forbid_call_tool):
    _register('mcp-1', tools=['tts', 'loco'], online=False)
    _run()
    assert direct_calls == []


def test_switch_mode_is_not_interrupted(direct_calls, forbid_call_tool):
    """Aborting a posture change partway is how a controlled descent becomes a
    fall, so a running stand-up/lie-down is left to finish."""
    _register(tools=['switch_mode'])
    _run()
    assert direct_calls == []


# ── honest reporting ─────────────────────────────────────────────────────────

def test_a_failed_call_is_not_counted_as_interrupted(monkeypatch, capsys,
                                                     forbid_call_tool):
    """The whole point: an error must not be reported as a successful interrupt."""
    async def _direct(mcp_id, tool_name, args):
        return {'error': 'device offline'}

    monkeypatch.setattr(mcp_client, 'call_tool_direct', _direct)
    _register(tools=['tts', 'loco'])
    _run()
    out = capsys.readouterr().out
    assert 'interrupted 0/2 active output(s)' in out
    assert 'device offline' in out


def test_an_exception_is_reported_and_not_counted(monkeypatch, capsys,
                                                  forbid_call_tool):
    async def _direct(mcp_id, tool_name, args):
        if tool_name == 'loco':
            raise RuntimeError('boom')
        return {'ok': True}

    monkeypatch.setattr(mcp_client, 'call_tool_direct', _direct)
    _register(tools=['tts', 'loco'])
    _run()
    out = capsys.readouterr().out
    assert 'interrupted 1/2 active output(s)' in out
    assert 'boom' in out


def test_success_is_counted(direct_calls, capsys, forbid_call_tool):
    _register(tools=['tts', 'loco'])
    _run()
    assert 'interrupted 2/2 active output(s)' in capsys.readouterr().out


def test_no_interruptible_tool_says_so(direct_calls, capsys, forbid_call_tool):
    """Silence here used to look identical to a successful interrupt."""
    _register(tools=['mic', 'camera'])
    _run()
    out = capsys.readouterr().out
    assert 'no tts/loco tool registered' in out
    assert direct_calls == []


# ── the hook path still wins ─────────────────────────────────────────────────

def test_a_registered_hook_short_circuits_the_fallback(monkeypatch, direct_calls):
    async def _fire(hook_id, *a, **kw):
        return ['binding-1']

    monkeypatch.setattr(hooks, 'fire', _fire)
    _register(tools=['tts', 'loco'])
    _run()
    assert direct_calls == [], 'the hook handled it; the fallback must not also fire'


def test_the_hook_path_clears_pending_acp(monkeypatch):
    async def _fire(hook_id, *a, **kw):
        return ['binding-1']

    monkeypatch.setattr(hooks, 'fire', _fire)

    async def scenario():
        ev = asyncio.Event()
        mcp_client._pending_actions['speak-1'] = ev
        await DecisionLoop.__new__(DecisionLoop)._interrupt_active_outputs()
        return ev.is_set()

    try:
        assert asyncio.run(scenario()) is True
    finally:
        mcp_client._pending_actions.clear()
