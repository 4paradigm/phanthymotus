"""ACP receipts must survive response/registration ordering and retries.

Only fake devices and an in-memory event bus are used. Exercise the production
HTTP function and SSE parser as well as the pending/barrier implementation.
"""
import ast
import asyncio
import collections
from contextlib import ExitStack
import json
from pathlib import Path
import types
import unittest
from unittest.mock import AsyncMock, patch

import mcp_client


class CompletionReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in mcp_client._PENDING_TABLE_NAMES:
            self.stack.enter_context(patch.object(mcp_client, name, {}))
        for name in ('_early_completions', '_completion_receipts', '_action_outcomes'):
            self.stack.enter_context(patch.object(mcp_client, name, collections.OrderedDict()))
        self.stack.enter_context(patch.object(mcp_client, '_completion_notification_tasks', {}))
        self.stack.enter_context(patch.object(mcp_client, '_register_listeners', []))
        self.stack.enter_context(patch.object(mcp_client, '_settle_listeners', []))
        self.enqueue = self.stack.enter_context(patch.object(mcp_client.event_bus, 'enqueue', AsyncMock()))
        self.clock = self.stack.enter_context(patch.object(mcp_client, '_completion_clock', return_value=100.))

    async def asyncTearDown(self):
        tasks = list(mcp_client._completion_notification_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def register(self, action_id='a', *, silent=False, via_hook=False):
        mcp_client._register_pending(action_id, tool_name='arm_control',
            meta={'completion': {'timeout': 2, 'silent_completion': silent},
                  'resource': frozenset({'arm'})}, args={'action': 'move'}, result={}, via_hook=via_hook)

    async def receive(self, action_id='a', status='completed', **result):
        return await mcp_client.receive_action_completion(action_id,
            {'action_id': action_id, 'status': status, 'result': result})

    def http_function(self):
        source = Path(mcp_client.__file__).with_name('start.py')
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == 'acp_complete')
        node.decorator_list = []
        namespace = {'fastapi': types.SimpleNamespace(Request=object), 'mcp_client': mcp_client}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        return namespace['acp_complete']

    async def http(self, body):
        with patch.dict('sys.modules', {'benchmark_facts': types.SimpleNamespace(note_acp_post=lambda _: None)}):
            return await self.http_function()(types.SimpleNamespace(json=AsyncMock(return_value=body)))

    async def test_early_receipt_replays_after_registration_and_unblocks_sync(self):
        ack = await self.receive(result_value=42)
        self.assertEqual(ack, {'ok': True, 'action_id': 'a', 'state': 'cached'})
        self.assertEqual(mcp_client._pending_actions, {})
        self.enqueue.assert_not_awaited()
        self.register()
        result = await mcp_client.sync(['a'], timeout=.1)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['results']['a']['result'], {'result_value': 42})
        await asyncio.sleep(0)
        self.enqueue.assert_awaited_once()

    async def test_success_failure_and_cancelled_are_real_terminal_results(self):
        for index, status in enumerate(('completed', 'error', 'failed', 'cancelled')):
            action_id = str(index)
            await self.receive(action_id, status, reason='controller result')
            self.register(action_id)
            result = await mcp_client.sync([action_id], timeout=.1)
            self.assertEqual(result['results'][action_id]['status'], status)
            self.assertEqual(result['results'][action_id]['result']['reason'], 'controller result')

    async def test_known_completion_is_matched_without_changing_response_contract(self):
        self.register()
        ack = await self.receive()
        self.assertEqual(ack, {'ok': True, 'action_id': 'a', 'state': 'matched'})
        self.assertTrue(mcp_client._pending_actions['a'].is_set())
        self.enqueue.assert_awaited_once()

    async def test_duplicate_first_result_wins_before_and_after_cleanup(self):
        await self.receive(answer='first')
        self.assertEqual((await self.receive(answer='second'))['state'], 'duplicate')
        self.register()
        await asyncio.sleep(0)
        self.assertEqual(mcp_client.pending_result('a')['result']['answer'], 'first')
        self.assertEqual((await self.receive(answer='third'))['state'], 'duplicate')
        await mcp_client.sync(['a'], timeout=.1)
        self.assertEqual((await self.receive(answer='fourth'))['state'], 'duplicate')
        self.enqueue.assert_awaited_once()

    async def test_duplicate_registration_does_not_reset_terminal_event(self):
        self.register()
        await self.receive()
        event = mcp_client._pending_actions['a']
        self.register()
        self.assertIs(mcp_client._pending_actions['a'], event)
        self.assertTrue(event.is_set())

    async def test_register_precedes_settle_and_uses_real_resource_for_hooks_too(self):
        for via_hook in (False, True):
            action_id = str(via_hook)
            order = []
            with patch.object(mcp_client, '_register_listeners', [lambda aid, tool, args, resource:
                    order.append(('registered', aid, resource))]), patch.object(mcp_client, '_settle_listeners',
                    [lambda aid, resource: order.append(('settled', aid, resource))]):
                await self.receive(action_id)
                self.assertEqual(order, [])
                self.register(action_id, via_hook=via_hook)
                self.assertEqual(order, [('registered', action_id, frozenset({'arm'})),
                                         ('settled', action_id, frozenset({'arm'}))])

    async def test_silent_is_captured_after_registration_and_before_reaping(self):
        await self.receive()
        self.register(silent=True)
        await mcp_client.sync(['a'], timeout=.1)
        await asyncio.sleep(0)
        notification = self.enqueue.await_args.kwargs
        self.assertTrue(notification['payload']['silent'])
        self.assertTrue(json.loads(notification['text'])['silent'])

    async def test_unknown_receipt_never_wakes_llm_or_settles_another_action(self):
        settled = []
        mcp_client.on_action_settled(lambda aid, resource: settled.append(aid))
        self.register('real')
        await self.receive('unrelated')
        self.enqueue.assert_not_awaited()
        self.assertFalse(mcp_client._pending_actions['real'].is_set())
        self.assertEqual(settled, [])

    async def test_early_cache_is_expiring_and_does_not_evict_acknowledged_results(self):
        with patch.object(mcp_client, '_EARLY_COMPLETION_CAP', 1):
            self.assertEqual((await self.receive('first'))['state'], 'cached')
            ack = await self.receive('second')
            self.assertFalse(ack['ok'])
            self.assertEqual(ack['state'], 'rejected')
            self.assertIn('first', mcp_client._early_completions)
            self.clock.return_value += mcp_client._EARLY_COMPLETION_TTL_S
            self.assertEqual((await self.receive('second'))['state'], 'cached')
            self.register('first')
            self.assertFalse(mcp_client._pending_actions['first'].is_set())
        self.enqueue.assert_not_awaited()

    async def test_early_cache_limits_bytes_and_rejects_non_finite_payloads(self):
        with patch.object(mcp_client, '_EARLY_COMPLETION_MAX_ITEM_BYTES', 256), \
                patch.object(mcp_client, '_EARLY_COMPLETION_MAX_BYTES', 200):
            self.assertTrue((await self.receive('one', text='a' * 30))['ok'])
            self.assertFalse((await self.receive('two', text='b' * 120))['ok'])
            self.assertFalse((await self.receive('big', text='c' * 300))['ok'])
            self.assertFalse((await self.receive('nan', value=float('nan')))['ok'])

    async def test_dedup_tombstones_are_bounded_and_expire(self):
        with patch.object(mcp_client, '_COMPLETION_RECEIPT_CAP', 2):
            for action_id in ('one', 'two', 'three'):
                self.register(action_id)
                await self.receive(action_id)
                await mcp_client.sync([action_id], timeout=.1)
            self.assertEqual(list(mcp_client._completion_receipts), ['two', 'three'])
            self.clock.return_value += mcp_client._COMPLETION_RECEIPT_TTL_S
            mcp_client._prune_completion_receipts()
            self.assertEqual(mcp_client._completion_receipts, {})
            # Existing outcome history still prevents resurrecting a reaped task.
            self.assertEqual((await self.receive('three'))['state'], 'duplicate')

    async def test_notification_failure_can_retry_without_reapplying_robot_result(self):
        self.register()
        self.enqueue.side_effect = [RuntimeError('queue unavailable'), None]
        first = await self.receive(answer='actual')
        self.assertFalse(first['ok'])
        self.assertTrue(mcp_client._pending_actions['a'].is_set())
        await asyncio.sleep(0)
        second = await self.receive(answer='do not overwrite')
        self.assertTrue(second['ok'])
        self.assertEqual(second['state'], 'duplicate')
        self.assertEqual(mcp_client.pending_result('a')['result']['answer'], 'actual')
        self.assertEqual(self.enqueue.await_count, 2)

    async def test_concurrent_duplicate_waits_for_same_notification_task(self):
        self.register()
        release = asyncio.Event()
        async def enqueue(**kwargs):
            await release.wait()
        self.enqueue.side_effect = enqueue
        first = asyncio.create_task(self.receive())
        await asyncio.sleep(0)
        second = asyncio.create_task(self.receive())
        await asyncio.sleep(0)
        release.set()
        replies = await asyncio.gather(first, second)
        self.assertEqual([reply['state'] for reply in replies], ['matched', 'duplicate'])
        self.enqueue.assert_awaited_once()

    async def test_http_receipts_use_shared_cache_and_dedup_path(self):
        payload = {'action_id': 'http', 'status': 'completed', 'result': {'actual': 1}}
        self.assertEqual((await self.http(payload))['state'], 'cached')
        self.register('http')
        await asyncio.sleep(0)
        self.assertEqual((await self.http(payload))['state'], 'duplicate')
        self.assertEqual(self.enqueue.await_args.kwargs['source'], 'acp:http')
        self.enqueue.assert_awaited_once()

    async def test_http_invalid_bodies_are_rejected_without_notifications(self):
        for body in ([], {}, {'action_id': 12}, {'action_id': ['bad']}, {'action_id': 'x' * 257}):
            reply = await self.http(body)
            self.assertFalse(reply['ok'])
            self.assertEqual(reply['state'], 'rejected')
        self.enqueue.assert_not_awaited()

    async def test_sse_parser_uses_same_buffer_and_dedup_as_http(self):
        payload = {'type': 'action_complete', 'action_id': 'sse', 'status': 'cancelled',
                   'result': {'reason': 'stopped'}}
        async def lines():
            yield ('data: ' + json.dumps(payload)).encode()
            raise asyncio.CancelledError()
        response = types.SimpleNamespace(status=200, content=lines())
        class Context:
            def __init__(self, result): self.result = result
            async def __aenter__(self): return self.result
            async def __aexit__(self, *args): pass
        session = types.SimpleNamespace(get=lambda _: Context(response))
        with patch.object(mcp_client.aiohttp, 'ClientSession', return_value=Context(session)):
            await mcp_client._subscribe_sse('device', 'http://fake/mcp')
        self.enqueue.assert_not_awaited()
        self.register('sse', silent=True)
        await asyncio.sleep(0)
        self.assertEqual((await self.http(payload))['state'], 'duplicate')
        notice = self.enqueue.await_args.kwargs
        self.assertEqual(notice['source'], 'mcp:device')
        self.assertTrue(notice['payload']['silent'])
        self.assertEqual(notice['payload']['result']['reason'], 'stopped')
        self.enqueue.assert_awaited_once()

    async def test_rejected_sse_log_escapes_caps_and_rate_limits_untrusted_ids(self):
        action_id = 'bad\nforged\r\x1b[31m' + 'x' * 300
        payload = {'type': 'action_complete', 'action_id': action_id}
        async def lines():
            for _ in range(5):
                yield ('data: ' + json.dumps(payload)).encode()
            raise asyncio.CancelledError()
        response = types.SimpleNamespace(status=200, content=lines())
        class Context:
            def __init__(self, result): self.result = result
            async def __aenter__(self): return self.result
            async def __aexit__(self, *args): pass
        session = types.SimpleNamespace(get=lambda _: Context(response))
        with patch.object(mcp_client.aiohttp, 'ClientSession', return_value=Context(session)), \
                patch.object(mcp_client.time, 'monotonic', return_value=100), \
                patch('builtins.print') as log:
            await mcp_client._subscribe_sse('device', 'http://fake/mcp')
        log.assert_called_once()
        message = log.call_args.args[0]
        self.assertNotIn(action_id, message)
        self.assertNotIn('\n', message)
        self.assertNotIn('\r', message)
        self.assertNotIn('\x1b', message)
        self.assertIn(repr(action_id)[:200], message)
        self.assertLess(len(message), 350)
        self.enqueue.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
