"""Offline behavior and real canvas/MCP entry-point tests; no messages are sent."""
import ast
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
os.environ.setdefault('DB_PATH', str(Path(tempfile.mkdtemp()) / 'test.db'))

import config
import fanshi_card as card
import mcp_client
from api import mcp_manage

GOOD = (200, b'{"errcode":0,"data":{"fail":{}}}')
BAD = (200, b'{"errcode":0,"data":{"fail":{"123456":40000}}}')
PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jQioAAAAASUVORK5CYII=')


def register_core():
    # Importing start also starts ROS-dependent imports. Execute its real registration
    # function with real config/registry modules, as other registry tests do.
    tree = ast.parse((SRC / 'start.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_register_core_mcp')
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'start.py', 'exec'), ns)
    ns['_register_core_mcp'](silent=True)


class FanshiCardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = {'mode': 'direct', 'webhook_url': 'https://example.invalid/hook?key=TEST_ONLY',
                         'group_id': '123456'}
        self.cfg = {'tool_config:channel:fanshi_message_send': self.settings}
        cfg_patch = patch.object(config, 'main', self.cfg)
        cfg_patch.start()
        self.addCleanup(cfg_patch.stop)
        for value in (card._active, card._generation, card._busy, card._locks,
                      card._recent, card._jobs, card._job_relays):
            value.clear()
        post_patch = patch.object(card, '_post', AsyncMock(return_value=GOOD))
        self.post = post_patch.start()
        self.addCleanup(post_patch.stop)
        relay_patch = patch.object(card.RelayClient, 'load', return_value=None)
        self.relay_load = relay_patch.start()
        self.addCleanup(relay_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.photo = self.root / 'photo.png'
        self.photo.write_bytes(PNG)
        roots_patch = patch.object(card, '_IMAGE_ROOTS', (self.root,))
        roots_patch.start()
        self.addCleanup(roots_patch.stop)

    async def start(self, instance='A'):
        result = await card.handle({'action': 'start', 'instance_id': instance})
        self.assertTrue(result['ok'], result)

    async def send(self, **args):
        return await card.handle({'action': 'send', 'instance_id': 'A', 'text': '巡楼测试', **args})

    def types(self):
        return [[item['type'] for item in call.args[1]['message']['body']]
                for call in self.post.call_args_list]

    async def test_start_is_silent_and_stop_blocks(self):
        await self.start()
        self.post.assert_not_called()
        await card.handle({'action': 'stop', 'instance_id': 'A'})
        self.assertEqual((await self.send())['status'], 'failed')
        self.post.assert_not_called()

    async def test_text_only_and_target_cannot_be_overridden(self):
        await self.start()
        result = await self.send(group_id='999', webhook_url='https://other.invalid')
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(self.types(), [['TEXT']])
        self.assertEqual(self.post.call_args.args[0], self.settings['webhook_url'])
        self.assertEqual(self.post.call_args.args[1]['message']['header']['toid'], [123456])
        self.assertNotIn('TEST_ONLY', json.dumps(result))

    async def test_pair_is_split_and_mention_sent_once(self):
        await self.start()
        result = await self.send(image_path=str(self.photo), at_userid='manager')
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(self.types(), [['TEXT', 'AT'], ['IMAGE']])
        self.assertEqual(result['request_count'], 2)
        self.assertEqual(result['accepted_parts'], 2)
        self.assertEqual(base64.b64decode(self.post.call_args.args[1]['message']['body'][0]['content']), PNG)
        saved = await card.handle({'action': 'result', 'instance_id': 'A', 'action_id': result['action_id']})
        self.assertEqual(saved, result)
        saved['parts'].clear()
        self.assertEqual(len(card._jobs['A'][result['action_id']]['parts']), 2)

    async def test_slow_settings_reads_allow_other_coroutines_to_advance(self):
        await self.start()
        loop = asyncio.get_running_loop()
        pending_reads = asyncio.Queue()
        progressed = []
        read_settings = card._read_settings

        def slow_read():
            release = threading.Event()
            loop.call_soon_threadsafe(pending_reads.put_nowait, release)
            # Only another coroutine can release this simulated database wait.
            progressed.append(release.wait(2))
            return read_settings()

        async def other_work():
            while True:
                release = await pending_reads.get()
                await asyncio.sleep(0)
                release.set()

        worker = asyncio.create_task(other_work())
        try:
            with patch.object(card, '_read_settings', side_effect=slow_read):
                result = await self.send(image_path=str(self.photo))
        finally:
            worker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await worker

        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(self.post.await_count, 2)
        self.assertTrue(progressed)
        self.assertTrue(all(progressed), 'A settings read blocked the event loop')

    async def test_stop_during_settings_check_does_not_send_next_part(self):
        await self.start()
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        read_settings = card._read_settings

        def slow_read():
            settings = read_settings()
            if self.post.await_count == 1:
                loop.call_soon_threadsafe(entered.set)
                release.wait(2)
            return settings

        with patch.object(card, '_read_settings', side_effect=slow_read):
            task = asyncio.create_task(self.send(image_path=str(self.photo)))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                await card.handle({'action': 'stop', 'instance_id': 'A'})
            finally:
                release.set()
                result = await task

        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['parts'][1]['status'], 'cancelled')
        self.assertEqual(self.post.await_count, 1)

    async def test_stop_during_initial_settings_read_cancels_operation(self):
        for action, restart in (('start', False), ('send', True), ('config', True)):
            with self.subTest(action=action):
                await card.handle({'action': 'stop', 'instance_id': 'A'})
                if action != 'start':
                    await self.start()
                loop = asyncio.get_running_loop()
                entered, release = asyncio.Event(), threading.Event()
                read_settings = card._read_settings
                first_read = True

                def slow_read():
                    nonlocal first_read
                    settings = read_settings()
                    if first_read:
                        first_read = False
                        loop.call_soon_threadsafe(entered.set)
                        release.wait(2)
                    return settings

                with patch.object(card, '_read_settings', side_effect=slow_read):
                    task = asyncio.create_task(card.handle({
                        'action': action, 'instance_id': 'A', 'text': 'test'}))
                    try:
                        await asyncio.wait_for(entered.wait(), 2)
                        await card.handle({'action': 'stop', 'instance_id': 'A'})
                        if restart:
                            await self.start()
                    finally:
                        release.set()
                        result = await task

                self.assertEqual(result['status'], 'cancelled')
                self.assertEqual('A' in card._active, restart)
                self.post.assert_not_called()

    async def test_stale_shared_config_read_preserves_new_start(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                await self.start()
                loop = asyncio.get_running_loop()
                entered, release = asyncio.Event(), threading.Event()
                read_settings = card._read_settings
                first_read = True

                def slow_read():
                    nonlocal first_read
                    settings = read_settings()
                    if first_read:
                        first_read = False
                        loop.call_soon_threadsafe(entered.set)
                        release.wait(2)
                        if invalid:
                            raise ValueError('Stale invalid configuration')
                    return settings

                with patch.object(card, '_read_settings', side_effect=slow_read):
                    task = asyncio.create_task(card.handle({'action': 'config'}))
                    try:
                        await asyncio.wait_for(entered.wait(), 2)
                        self.settings['group_id'] = str(int(self.settings['group_id']) + 1)
                        await self.start()
                        current = card._active['A']
                    finally:
                        release.set()
                        result = await task

                self.assertEqual(result['status'], 'failed' if invalid else 'configured')
                self.assertEqual(card._active.get('A'), current)
                self.post.assert_not_called()

    async def test_image_only_retains_mention(self):
        await self.start()
        result = await self.send(text='', image_path=str(self.photo), at_userid='manager')
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(self.types(), [['IMAGE', 'AT']])

    async def test_invalid_arguments_never_post(self):
        await self.start()
        for args in ({'text': ' '}, {'text': False}, {'image_path': False}, {'at_userid': 7}):
            self.assertEqual((await self.send(**args))['status'], 'failed')
        self.photo.write_bytes(b'not an image')
        self.assertEqual((await self.send(image_path=str(self.photo)))['status'], 'failed')
        self.photo.write_bytes(b'\x89PNG\r\n\x1a\n' + b'x' * 750000)
        self.assertEqual((await self.send(image_path=str(self.photo)))['status'], 'failed')
        self.post.assert_not_called()

    async def test_symlink_cannot_escape_image_directory(self):
        allowed = self.root / 'allowed'
        allowed.mkdir()
        (allowed / 'outside.png').symlink_to(self.photo)
        with patch.object(card, '_IMAGE_ROOTS', (allowed,)):
            with self.assertRaises(ValueError):
                card._image_base64(str(allowed / 'outside.png'))

    async def test_config_rejects_unsafe_or_missing_destination(self):
        for cfg in ({}, {'webhook_url': 'http://example.invalid', 'group_id': '1'},
                    {'webhook_url': 'https://user:pass@example.invalid', 'group_id': '1'},
                    {'webhook_url': 'https://example.invalid', 'group_id': False},
                    {'webhook_url': 'https://[broken', 'group_id': '1'}):
            self.cfg['tool_config:channel:fanshi_message_send'] = {'mode': 'direct', **cfg}
            result = await card.handle({'action': 'start', 'instance_id': 'A'})
            self.assertEqual(result['status'], 'failed')
        self.post.assert_not_called()

    async def test_first_failure_stops_pair_without_retry(self):
        await self.start()
        self.post.return_value = BAD
        result = await self.send(image_path=str(self.photo))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['parts'][0]['failed_groups'], {'123456': 40000})
        self.assertEqual(result['parts'][1]['status'], 'not_sent')
        self.assertEqual(self.post.await_count, 1)

    async def test_second_failure_is_partial_and_keeps_success(self):
        await self.start()
        self.post.side_effect = [GOOD, BAD]
        result = await self.send(image_path=str(self.photo))
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(result['ok'])
        self.assertEqual(result['accepted_parts'], 1)
        self.assertEqual(self.post.await_count, 2)

    async def test_timeout_preserves_success_without_retry(self):
        await self.start()
        self.post.side_effect = [GOOD, TimeoutError('SECRET_URL')]
        result = await self.send(image_path=str(self.photo))
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['parts'][0]['status'], 'accepted')
        self.assertEqual(result['parts'][1]['status'], 'unknown')
        self.assertEqual(self.post.await_count, 2)
        self.assertNotIn('SECRET_URL', json.dumps(result))

    async def test_stop_between_parts_does_not_send_image(self):
        await self.start()
        async def stop_after_first(*args):
            await card.handle({'action': 'stop', 'instance_id': 'A'})
            return GOOD
        self.post.side_effect = stop_after_first
        result = await self.send(image_path=str(self.photo))
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['parts'][1]['status'], 'cancelled')
        self.assertEqual(self.post.await_count, 1)

    async def test_config_change_between_parts_does_not_change_destination(self):
        await self.start()
        async def change_after_first(*args):
            self.settings['group_id'] = '234567'
            return GOOD
        self.post.side_effect = change_after_first
        result = await self.send(image_path=str(self.photo))
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(self.post.await_count, 1)
        self.assertEqual((await self.send())['status'], 'failed')

    async def test_shared_config_apply_invalidates_active_instances(self):
        await self.start()
        self.settings['group_id'] = '234567'
        result = await card.handle({'action': 'config'})
        self.assertTrue(result['adapter_ok'])
        self.assertNotIn('A', card._active)
        self.post.assert_not_called()

    async def test_busy_prevents_concurrent_duplicate(self):
        await self.start()
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow_post(*args):
            entered.set()
            await release.wait()
            return GOOD
        self.post.side_effect = slow_post
        first = asyncio.create_task(self.send())
        await asyncio.wait_for(entered.wait(), 2)
        self.assertEqual((await self.send())['status'], 'failed')
        release.set()
        self.assertEqual((await first)['status'], 'accepted')
        self.assertEqual(self.post.await_count, 1)

    async def test_cancelling_during_image_preserves_text_success(self):
        await self.start()
        entered = asyncio.Event()
        calls = 0
        async def post(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return GOOD
            entered.set()
            await asyncio.Event().wait()
        self.post.side_effect = post
        task = asyncio.create_task(self.send(image_path=str(self.photo)))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        saved = next(iter(card._jobs['A'].values()))
        self.assertEqual(saved['status'], 'unknown')
        self.assertEqual(saved['parts'][0]['status'], 'accepted')
        self.assertNotIn('A', card._busy)

    async def test_confirmed_gateway_result_cannot_be_downgraded(self):
        action_id = 'fanshi_test'
        card._jobs['A'][action_id] = {'status': 'accepted', 'ok': True, 'action_id': action_id}
        stale = card._remember_result('A', action_id, {'status': 'unknown', 'ok': False})
        self.assertEqual(stale['status'], 'accepted')
        self.assertEqual(card._jobs['A'][action_id]['status'], 'accepted')

    async def test_rate_limit_reserves_two_slots_and_is_shared(self):
        await self.start()
        await self.start('B')
        for _ in range(29):
            self.assertEqual((await self.send())['status'], 'accepted')
        self.assertEqual((await self.send(instance_id='B', image_path=str(self.photo)))['status'], 'failed')
        self.assertEqual((await self.send(instance_id='B'))['status'], 'accepted')
        self.assertEqual((await self.send())['status'], 'failed')
        self.assertEqual(self.post.await_count, 30)

    async def test_result_records_are_scoped_and_bounded(self):
        await self.start()
        first = await self.send()
        self.assertEqual((await card.handle({'action': 'result', 'instance_id': 'B',
                                            'action_id': first['action_id']}))['status'], 'unknown')
        for _ in range(33):
            card._recent.clear()
            await self.send()
        self.assertEqual(len(card._jobs['A']), 32)
        self.assertNotIn(first['action_id'], card._jobs['A'])

    async def test_auto_uses_provisioned_relay_without_webhook(self):
        class FakeRelay:
            identity = 'test-relay'
            def run(self, action_id, payload, cancelled, progress):
                assert not cancelled()
                progress({'status': 'sending'})
                return {'ok': True, 'status': 'accepted', 'action_id': action_id}
        self.cfg['tool_config:channel:fanshi_message_send'] = {'mode': 'auto'}
        self.relay_load.return_value = FakeRelay()
        await self.start()
        result = await self.send()
        self.assertEqual(result['status'], 'accepted')
        self.post.assert_not_called()

    async def test_broken_relay_never_falls_back_to_direct(self):
        self.settings['mode'] = 'auto'
        self.relay_load.side_effect = card.RelayError('TEST_SECRET')
        result = await card.handle({'action': 'start', 'instance_id': 'A'})
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('TEST_SECRET', json.dumps(result))
        self.post.assert_not_called()

    async def test_relay_unknown_can_be_resolved_without_resubmission(self):
        class FakeRelay:
            identity = 'test-relay'
            def run(self, action_id, *args):
                return {'ok': False, 'status': 'unknown', 'action_id': action_id}
            def result(self, action_id):
                return {'ok': True, 'status': 'accepted', 'action_id': action_id}
        relay = FakeRelay()
        self.settings['mode'] = 'auto'
        self.relay_load.return_value = relay
        await self.start()
        with patch.object(relay, 'run', wraps=relay.run) as run:
            initial = await self.send()
            self.assertEqual(initial['status'], 'unknown')
            result = await card.handle({'action': 'result', 'instance_id': 'A',
                                        'action_id': initial['action_id']})
            self.assertEqual(result['status'], 'accepted')
            self.assertEqual(run.call_count, 1)
        self.post.assert_not_called()

    async def test_real_canvas_and_mcp_paths_with_any_driver(self):
        for driver in ('tianyi', 'q5', 'a3'):
            self.cfg['services'] = {'mcp': [{'id': driver, 'tools': []}]}
            with patch.object(mcp_client, 'registry', {}):
                register_core()
                register_core()
                mcp_client._register_internal_mcps()
                tools = next(m for m in self.cfg['services']['mcp'] if m['id'] == 'channel')['tools']
                self.assertEqual(sum(t['name'] == card.TOOL_NAME for t in tools), 1)
                self.assertTrue(any(m['id'] == driver for m in self.cfg['services']['mcp']))
                res = await mcp_manage.mcp_call_tool('channel', mcp_manage.MCPCallRequest(
                    tool=card.TOOL_NAME, arguments={'action': 'start', 'instance_id': 'A'}))
                self.assertEqual(res['data']['state'], 'running')
                result = json.loads(await mcp_client.call_tool('mcp__channel__fanshi_message_send',
                    {'action': 'send', 'instance_id': 'A', 'text': 'test'}))
                self.assertEqual(result['status'], 'accepted')
                self.assertNotIn(result['action_id'], mcp_client._pending_actions)
                # Failed platform details remain visible through the canvas API.
                self.post.return_value = BAD
                res = await mcp_manage.mcp_call_tool('channel', mcp_manage.MCPCallRequest(
                    tool=card.TOOL_NAME, arguments={'action': 'send', 'instance_id': 'A', 'text': 'test'}))
                self.assertEqual(res['data']['parts'][0]['failed_groups'], {'123456': 40000})
                self.post.return_value = GOOD

    async def test_credentials_not_in_model_schema_and_marked_sensitive(self):
        props = card.TOOL['inputSchema']['properties']
        self.assertNotIn('webhook_url', props)
        self.assertNotIn('group_id', props)
        for key in ('webhook_url', 'group_id'):
            self.assertTrue(card.TOOL['configSchema']['properties'][key]['x-sensitive'])

    async def test_response_variants(self):
        for code, raw, expected in [
            (500, b'', 'unknown'), (403, b'', 'failed'), (408, b'', 'unknown'),
            (200, b'{"errorcode":0}', 'accepted'), (200, b'{"errcode":0}', 'accepted'),
            (200, b'{"errcode":0,"errorcode":1}', 'unknown'),
            (200, b'{"errcode":false}', 'unknown'), (200, b'not json', 'unknown'),
            (200, b'x' * 65537, 'unknown'),
        ]:
            self.assertEqual(card._interpret(code, raw)['status'], expected)


if __name__ == '__main__':
    unittest.main()
