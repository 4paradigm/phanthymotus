"""
test_on_notify_barrier.py — on_notify must narrate, not interrupt.

Regression found on Tianyi: `_round_already_notified` recognised a "the LLM
already spoke this round" tool_call only by string-parsing its name
(`name.split('__')[-1] == 'tts'`) and its args (`action == 'speak'`). That
convention only holds for devices whose `tts` tool takes a bare `action`
argument. A device using `x-action-params` (Tianyi's `tts` splits into
`tts__speak` / `tts__interrupt` / ...) exposes the LLM-facing name
`mcp__<id>__tts__speak` with no `action` arg at all — the check never
matched, so on_notify treated every content-bearing round as "not yet
notified" and fired a second, redundant speak call on top of the LLM's own.

That second call went through `hooks.fire`, which — correctly for true
interrupt hooks (on_interrupt_*, e-stop) but not for narration — bypasses the
barrier entirely (`call_tool_direct`): no check that the mouth is already
busy, and no pending registration for the call it makes. So even with the
dedup fixed, on_notify could still talk over whatever's currently playing,
and the very next barrier-respecting tool call could talk over on_notify's
own narration right back.

Fixed by:
- `mcp_client.resolve_tool_binding` — resolve an LLM tool_call name back to
  (mcp_id, tool, action) through the registry's split_map/schemas, the same
  lookup on_notify's own dispatch needs, instead of guessing from the name.
- `_round_already_notified` — use that resolver plus
  `hooks.get_hook_for_binding` to check "did this round already exercise
  whatever on_notify would also fire", which works for both tool-naming
  conventions.
- `mcp_client.call_tool_hook(..., barrier_aware=True)` — skip the call if the
  tool's resource is already held by a pending action, and register any
  action_id it gets back as pending, same as a normal LLM-issued dispatch.
- `hooks.fire(..., barrier_aware=True)` — on_notify's call site opts into the
  above; on_interrupt_* call sites are unchanged (default False → bypass).

Run: cd agent-core && python3 -m pytest tests/test_on_notify_barrier.py
"""
import asyncio
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402
import hooks  # noqa: E402
from event.llm import _round_already_notified  # noqa: E402


SPLIT_TOOL_META = {
    'type': 'actuator',
    'action_enum': None,
    'has_config_schema': False,
    'completion': {'actions': ['speak'], 'timeout': 180},
    'resource': frozenset({'mouth'}),
}


def _split_tts_registry():
    """A device whose `tts` tool is exposed pre-split, like Tianyi's."""
    return {
        'dev1': {
            'name': 'dev1', 'url': 'http://dev1', 'online': True,
            'tools': ['tts'],
            'schemas': {
                'mcp__dev1__tts__speak': {'name': 'mcp__dev1__tts__speak'},
                'mcp__dev1__tts__interrupt': {'name': 'mcp__dev1__tts__interrupt'},
            },
            'tool_meta': {
                'mcp__dev1__tts__speak': dict(SPLIT_TOOL_META),
                'mcp__dev1__tts__interrupt': {**SPLIT_TOOL_META, 'completion': None},
            },
            'split_map': {
                'mcp__dev1__tts__speak': {'tool': 'tts', 'action': 'speak'},
                'mcp__dev1__tts__interrupt': {'tool': 'tts', 'action': 'interrupt'},
            },
            'tool_groups': {'tts': ['mcp__dev1__tts__speak', 'mcp__dev1__tts__interrupt']},
            'input_schemas': {},
        },
    }


def _unsplit_tts_registry():
    """A device whose `tts` tool takes a bare `action` argument."""
    return {
        'dev2': {
            'name': 'dev2', 'url': 'http://dev2', 'online': True,
            'tools': ['tts'],
            'schemas': {'mcp__dev2__tts': {'name': 'mcp__dev2__tts'}},
            'tool_meta': {'mcp__dev2__tts': dict(SPLIT_TOOL_META)},
            'split_map': {},
            'tool_groups': {},
            'input_schemas': {},
        },
    }


class _RegistryFixture(unittest.TestCase):
    """Swap mcp_client.registry for a fixture, restore it after."""

    def setUp(self):
        self._saved_registry = dict(mcp_client.registry)
        self._saved_hooks = dict(hooks._registry)
        mcp_client.registry.clear()
        hooks._registry.clear()
        self._saved_pending = (
            dict(mcp_client._pending_actions), dict(mcp_client._pending_tools),
            dict(mcp_client._pending_resources), dict(mcp_client._pending_timeouts),
        )
        for d in (mcp_client._pending_actions, mcp_client._pending_tools,
                  mcp_client._pending_resources, mcp_client._pending_timeouts):
            d.clear()

    def tearDown(self):
        mcp_client.registry.clear()
        mcp_client.registry.update(self._saved_registry)
        hooks._registry.clear()
        hooks._registry.update(self._saved_hooks)
        targets = (mcp_client._pending_actions, mcp_client._pending_tools,
                   mcp_client._pending_resources, mcp_client._pending_timeouts)
        for d, saved in zip(targets, self._saved_pending):
            d.clear()
            d.update(saved)


class TestResolveToolBinding(_RegistryFixture):
    def test_split_tool_resolves_via_split_map(self):
        mcp_client.registry.update(_split_tts_registry())
        self.assertEqual(
            mcp_client.resolve_tool_binding('mcp__dev1__tts__speak', {'text': 'hi'}),
            ('dev1', 'tts', 'speak'))

    def test_unsplit_tool_resolves_via_args_action(self):
        mcp_client.registry.update(_unsplit_tts_registry())
        self.assertEqual(
            mcp_client.resolve_tool_binding('mcp__dev2__tts', {'action': 'speak', 'text': 'hi'}),
            ('dev2', 'tts', 'speak'))

    def test_unknown_name_is_none(self):
        mcp_client.registry.update(_unsplit_tts_registry())
        self.assertIsNone(mcp_client.resolve_tool_binding('mcp__dev2__nope', {}))


class TestRoundAlreadyNotified(_RegistryFixture):
    """The bug: split tool_calls never matched, so on_notify double-fired."""

    def test_split_speak_call_counts_as_notified(self):
        mcp_client.registry.update(_split_tts_registry())
        hooks.register('dev1', 'tts', {'on_notify': {'action': 'speak'}})
        tool_calls = [{'function': {
            'name': 'mcp__dev1__tts__speak',
            'arguments': json.dumps({'text': '马上就到'}),
        }}]
        self.assertTrue(_round_already_notified(tool_calls))

    def test_split_interrupt_call_does_not_count(self):
        """Calling a *different* action under the same base tool must not suppress notify."""
        mcp_client.registry.update(_split_tts_registry())
        hooks.register('dev1', 'tts', {'on_notify': {'action': 'speak'}})
        tool_calls = [{'function': {
            'name': 'mcp__dev1__tts__interrupt',
            'arguments': '{}',
        }}]
        self.assertFalse(_round_already_notified(tool_calls))

    def test_unsplit_speak_call_still_counts_as_notified(self):
        """The pre-fix convention must keep working, not just the fixed one."""
        mcp_client.registry.update(_unsplit_tts_registry())
        hooks.register('dev2', 'tts', {'on_notify': {'action': 'speak'}})
        tool_calls = [{'function': {
            'name': 'mcp__dev2__tts',
            'arguments': json.dumps({'action': 'speak', 'text': 'hi'}),
        }}]
        self.assertTrue(_round_already_notified(tool_calls))

    def test_no_tool_calls_is_not_notified(self):
        self.assertFalse(_round_already_notified([]))

    def test_unrelated_tool_call_is_not_notified(self):
        mcp_client.registry.update(_split_tts_registry())
        hooks.register('dev1', 'tts', {'on_notify': {'action': 'speak'}})
        tool_calls = [{'function': {'name': 'mcp__dev1__loco__navigate', 'arguments': '{}'}}]
        self.assertFalse(_round_already_notified(tool_calls))


class TestCallToolHookBarrierAware(_RegistryFixture):
    def test_skips_when_resource_busy(self):
        mcp_client.registry.update(_split_tts_registry())
        mcp_client._pending_actions['speak-1'] = asyncio.Event()
        mcp_client._pending_tools['speak-1'] = 'tts'
        mcp_client._pending_resources['speak-1'] = frozenset({'mouth'})

        called = []
        async def _direct(mcp_id, tool_name, args):
            called.append((mcp_id, tool_name, args))
            return {"state": "speaking", "action_id": "should-not-happen"}

        orig = mcp_client.call_tool_direct
        mcp_client.call_tool_direct = _direct
        try:
            out = asyncio.run(mcp_client.call_tool_hook(
                'dev1', 'tts', {'action': 'speak', 'text': 'hi'}, barrier_aware=True))
        finally:
            mcp_client.call_tool_direct = orig

        self.assertEqual(out, {"skipped": "resource busy"})
        self.assertEqual(called, [], 'must not have dispatched — mouth was busy')

    def test_registers_pending_when_it_dispatches(self):
        mcp_client.registry.update(_split_tts_registry())

        async def _direct(mcp_id, tool_name, args):
            return {"state": "speaking", "action_id": "notify-1"}

        orig = mcp_client.call_tool_direct
        mcp_client.call_tool_direct = _direct
        try:
            out = asyncio.run(mcp_client.call_tool_hook(
                'dev1', 'tts', {'action': 'speak', 'text': 'hi there'}, barrier_aware=True))
        finally:
            mcp_client.call_tool_direct = orig

        self.assertEqual(out['action_id'], 'notify-1')
        self.assertIn('notify-1', mcp_client._pending_actions)
        self.assertEqual(mcp_client._pending_tools['notify-1'], 'tts')
        self.assertEqual(mcp_client._pending_resources['notify-1'], frozenset({'mouth'}))
        # A subsequent barrier-respecting call for the same resource must now see it.
        self.assertEqual(mcp_client.conflicting_pending(frozenset({'mouth'})), ['notify-1'])

    def test_non_barrier_aware_bypasses_everything_unchanged(self):
        """Existing on_interrupt_* call sites must keep their old behaviour."""
        mcp_client.registry.update(_split_tts_registry())
        mcp_client._pending_actions['speak-1'] = asyncio.Event()
        mcp_client._pending_tools['speak-1'] = 'tts'
        mcp_client._pending_resources['speak-1'] = frozenset({'mouth'})

        called = []
        async def _direct(mcp_id, tool_name, args):
            called.append((mcp_id, tool_name, args))
            return {"state": "interrupted"}

        orig = mcp_client.call_tool_direct
        mcp_client.call_tool_direct = _direct
        try:
            out = asyncio.run(mcp_client.call_tool_hook(
                'dev1', 'tts', {'action': 'interrupt'}, barrier_aware=False))
        finally:
            mcp_client.call_tool_direct = orig

        self.assertEqual(out, {"state": "interrupted"})
        self.assertEqual(len(called), 1, 'must still dispatch immediately, busy or not')


class TestHooksFireBarrierAware(_RegistryFixture):
    def test_fire_barrier_aware_routes_through_call_tool_hook(self):
        mcp_client.registry.update(_split_tts_registry())
        hooks.register('dev1', 'tts', {'on_notify': {'action': 'speak'}})

        seen = {}
        async def _hook(mcp_id, tool_name, args, barrier_aware=False):
            seen['args'] = (mcp_id, tool_name, args, barrier_aware)
            return {"action_id": "notify-2"}

        orig = mcp_client.call_tool_hook
        mcp_client.call_tool_hook = _hook
        try:
            asyncio.run(hooks.fire('on_notify', {'text': '还在路上'}, barrier_aware=True))
        finally:
            mcp_client.call_tool_hook = orig

        self.assertEqual(seen['args'][0], 'dev1')
        self.assertEqual(seen['args'][1], 'tts')
        self.assertEqual(seen['args'][2]['action'], 'speak')
        self.assertEqual(seen['args'][2]['text'], '还在路上')
        self.assertTrue(seen['args'][3])

    def test_fire_default_still_bypasses(self):
        """on_interrupt_* callers that don't pass barrier_aware must be unaffected."""
        mcp_client.registry.update(_split_tts_registry())
        hooks.register('dev1', 'tts', {'on_interrupt_speak': {'action': 'interrupt'}})

        called = []
        async def _direct(mcp_id, tool_name, args):
            called.append((mcp_id, tool_name, args))
            return {"state": "interrupted"}

        orig = mcp_client.call_tool_direct
        mcp_client.call_tool_direct = _direct
        try:
            asyncio.run(hooks.fire('on_interrupt_speak'))
        finally:
            mcp_client.call_tool_direct = orig

        self.assertEqual(len(called), 1)


if __name__ == '__main__':
    unittest.main()
