"""ACL role (viewer/operator/owner) must actually be enforced on the tool
dispatch path for human Channel messages — not just carried as text for the
LLM to optionally respect."""

import asyncio
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import collector  # noqa: E402
import mcp_client  # noqa: E402
from channel.manager import ChannelManager, viewer_reply_scope  # noqa: E402
from event.llm import (  # noqa: E402
    _channel_tool_restricted,
    _raw_input_info,
    _restricted_channel_tool_allowed,
    _viewer_channel_restricted,
    _viewer_channel_reply_allowed,
    _viewer_image_read_allowed,
)


def _human_payload(role: str, text: str = 'hi', *, user='u1', chat='g1', msgid='m1') -> dict:
    return {
        'source': 'dds:/channel/request/feishu_test',
        'text': json.dumps({'sender_type': 'user', 'user_role': role, 'text': text,
                            'channel_id': 'feishu_test', 'chat_id': chat,
                            'user_id': user, 'message_id': msgid}),
        'ts': 1,
    }


class ViewerRoleAuthorizationTest(unittest.TestCase):
    def test_viewer_only_batch_is_restricted(self):
        self.assertTrue(collector.has_viewer_only_channel_event([_human_payload('viewer')]))
        self.assertFalse(collector.has_viewer_only_channel_event([_human_payload('operator')]))
        self.assertFalse(collector.has_viewer_only_channel_event([_human_payload('owner')]))
        # No channel payload at all (e.g. a sensor trigger) — not a viewer restriction case.
        self.assertFalse(collector.has_viewer_only_channel_event([{
            'source': 'dds:/sensor/battery', 'text': '{}',
        }]))
        # Missing/unknown role defaults to restricted (fail closed, not fail open).
        self.assertTrue(collector.has_viewer_only_channel_event([_human_payload('')]))

    def test_mixed_batch_fails_closed_even_before_splitting(self):
        batch = [_human_payload('viewer'), _human_payload('operator')]
        self.assertTrue(collector.has_viewer_only_channel_event(batch))
        self.assertTrue(_viewer_channel_restricted(collector._build_trigger(batch, urgent=True)))

    def test_split_keeps_role_sender_and_conversation_separate(self):
        viewer = _human_payload('viewer', 'viewer instruction', user='alice')
        operator = _human_payload('operator', 'harmless', user='bob')
        other_chat = _human_payload('operator', user='bob', chat='g2')
        sensor = {'source': 'dds:/sensor/battery', 'text': '{}', 'ts': 1}
        batches = collector._split_by_authorization([viewer, operator, other_chat, sensor])
        self.assertEqual(batches, [[viewer], [operator], [other_chat], [sensor]])
        self.assertTrue(collector._build_trigger(batches[0], True)['_viewer_channel_event'])
        self.assertFalse(collector._build_trigger(batches[1], True)['_viewer_channel_event'])
        self.assertNotIn('viewer instruction', collector._build_trigger(batches[1], True)['text'])

    def test_unknown_channel_payload_is_read_only_and_isolated(self):
        malformed = {'source': 'dds:/channel/request/x', 'text': 'not-json', 'ts': 1}
        operator = _human_payload('operator')
        self.assertTrue(collector.has_viewer_only_channel_event([malformed]))
        self.assertEqual(collector._split_by_authorization([malformed, operator]),
                         [[malformed], [operator]])

    def test_bot_sender_does_not_count_as_a_human_viewer(self):
        bot_payload = {
            'source': 'dds:/channel/request/feishu_test',
            'text': json.dumps({'sender_type': 'bot', 'user_role': 'operator'}),
        }
        self.assertFalse(collector.has_viewer_only_channel_event([bot_payload]))

    def test_build_trigger_carries_viewer_flag_and_is_restricted_in_llm(self):
        viewer_trigger = collector._build_trigger([_human_payload('viewer')], urgent=True)
        operator_trigger = collector._build_trigger([_human_payload('operator')], urgent=True)

        self.assertTrue(viewer_trigger['_viewer_channel_event'])
        self.assertTrue(_viewer_channel_restricted(viewer_trigger))
        self.assertTrue(_channel_tool_restricted(viewer_trigger))

        self.assertFalse(operator_trigger['_viewer_channel_event'])
        self.assertFalse(_viewer_channel_restricted(operator_trigger))
        self.assertFalse(_channel_tool_restricted(operator_trigger))

    def test_viewer_restricted_turn_allows_only_sensor_resource_and_reply(self):
        registry = {
            'device': {
                'tool_meta': {
                    'mcp__device__camera': {'type': 'sensor'},
                    'mcp__device__navigate': {'type': 'actuator'},
                    'mcp__device__load_map': {'type': 'processor'},
                },
            },
        }
        with mock.patch.dict(mcp_client.registry, registry, clear=True):
            self.assertTrue(_restricted_channel_tool_allowed('finish', bot_restricted=False))
            self.assertTrue(_restricted_channel_tool_allowed('mcp__channel__channel_reply', bot_restricted=False))
            self.assertTrue(_restricted_channel_tool_allowed('mcp__device__camera', bot_restricted=False))
            self.assertFalse(_restricted_channel_tool_allowed('mcp__device__navigate', bot_restricted=False))
            self.assertFalse(_restricted_channel_tool_allowed('mcp__device__load_map', bot_restricted=False))

    def test_viewer_can_use_read_only_system_tools_but_bot_cannot(self):
        for name in ('WebSearch', 'search_history', 'memory_recall', 'raw_input_info'):
            self.assertTrue(_restricted_channel_tool_allowed(name, bot_restricted=False), name)
            self.assertFalse(_restricted_channel_tool_allowed(name, bot_restricted=True), name)
        # Still no mutating desktop tools for either — read-only means read-only.
        for name in ('Bash', 'Write', 'Edit', 'WebFetch', 'subagent_spawn'):
            self.assertFalse(_restricted_channel_tool_allowed(name, bot_restricted=False), name)

    def test_viewer_read_is_limited_to_current_channel_images(self):
        root = pathlib.Path(tempfile.mkdtemp())
        current = root / 'channel_a' / 'current.png'
        other = root / 'channel_a' / 'other.png'
        trigger = {'_channel_image_paths': [str(current)]}
        with mock.patch('channel.store._ROOT', root):
            self.assertTrue(_viewer_image_read_allowed(trigger, {'file_path': str(current)}))
            self.assertFalse(_viewer_image_read_allowed(trigger, {'file_path': str(other)}))
            self.assertFalse(_viewer_image_read_allowed(trigger, {'file_path': str(root.parent / 'secret.png')}))
        self.assertTrue(_restricted_channel_tool_allowed('Read', bot_restricted=False, viewer_has_images=True))
        self.assertFalse(_restricted_channel_tool_allowed('Read', bot_restricted=False))
        self.assertFalse(_restricted_channel_tool_allowed('Read', bot_restricted=True, viewer_has_images=True))

    def test_viewer_reply_cannot_contact_other_bots_or_send_unrelated_files(self):
        root = pathlib.Path(tempfile.mkdtemp())
        current = root / 'channel_a' / 'current.png'
        other = root / 'channel_a' / 'other.png'
        trigger = {'_channel_message_ids': ['m1'], '_channel_image_paths': [str(current)]}
        with mock.patch('channel.store._ROOT', root):
            self.assertTrue(_viewer_channel_reply_allowed(trigger, {'source_message_id': 'm1', 'text': 'ok'}))
            self.assertTrue(_viewer_channel_reply_allowed(trigger, {
                'source_message_id': 'm1', 'files': [{'path': str(current)}]}))
            self.assertFalse(_viewer_channel_reply_allowed(trigger, {
                'source_message_id': 'm1', 'files': [{'path': str(other)}]}))
            self.assertFalse(_viewer_channel_reply_allowed(trigger, {
                'source_message_id': 'm1', 'trusted_bot_id': 'operator-bot'}))
            self.assertFalse(_viewer_channel_reply_allowed(trigger, {
                'source_message_id': 'm1', 'mention_open_id': 'ou_operator', 'expect_reply': True}))
            self.assertFalse(_viewer_channel_reply_allowed(trigger, {'source_message_id': 'old'}))

    def test_collector_includes_only_human_image_attachments(self):
        event = _human_payload('viewer')
        payload = json.loads(event['text'])
        payload['files'] = [
            {'kind': 'image', 'path': '/work/resource/channel_files/current.png'},
            {'kind': 'file', 'path': '/work/resource/channel_files/other.txt'},
        ]
        event['text'] = json.dumps(payload)
        self.assertEqual(collector.channel_image_paths([event]),
                         ['/work/resource/channel_files/current.png'])
        payload['sender_type'] = 'bot'
        event['text'] = json.dumps(payload)
        self.assertEqual(collector.channel_image_paths([event]), [])


class ChannelQueueAuthorizationTest(unittest.IsolatedAsyncioTestCase):
    async def test_viewer_reply_scope_is_enforced_at_final_channel_routing(self):
        manager = ChannelManager()
        token = viewer_reply_scope.set((frozenset({'channel-a'}), frozenset({'msg-a'})))
        try:
            wrong_channel = await manager.send_to_channel('channel-b', text='x', source_message_id='msg-a')
            wrong_message = await manager.send_to_channel('channel-a', text='x', source_message_id='msg-b')
        finally:
            viewer_reply_scope.reset(token)
        self.assertIn('current channel and message', wrong_channel)
        self.assertIn('current channel and message', wrong_message)

    async def test_raw_input_tool_does_not_retrieve_other_channel_messages(self):
        with mock.patch.object(collector, 'get_source_detail') as get_detail:
            result = await _raw_input_info('dds:/channel/request/feishu_test')
        self.assertIn('only in their authorized turns', result)
        get_detail.assert_not_called()

    async def test_emit_and_pending_flush_split_roles(self):
        viewer = _human_payload('viewer', user='u1', msgid='v')
        operator = _human_payload('operator', user='u2', msgid='o')
        with mock.patch.object(collector, '_output', asyncio.Queue()), \
                mock.patch.object(collector, '_priority_pending', collector.deque()), \
                mock.patch.object(collector, '_steering_queue', asyncio.Queue()):
            await collector._emit_batch([viewer, operator])
            first = collector._output.get_nowait()
            second = collector._output.get_nowait()
            self.assertTrue(first['_viewer_channel_event'])
            self.assertFalse(second['_viewer_channel_event'])
            collector._priority_pending.extend([viewer, operator])
            collector._flush_all_pending()
            self.assertTrue(collector._output.get_nowait()['_viewer_channel_event'])
            self.assertFalse(collector._output.get_nowait()['_viewer_channel_event'])

    async def test_channel_steering_is_deferred_instead_of_injected(self):
        viewer = _human_payload('viewer', user='u1')
        sensor = {'source': 'dds:/sensor/battery', 'text': '{}', 'ts': 1}
        with mock.patch.object(collector, '_priority_pending', collector.deque()), \
                mock.patch.object(collector, '_steering_queue', asyncio.Queue()):
            collector._steering_queue.put_nowait(viewer)
            collector._steering_queue.put_nowait(sensor)
            self.assertEqual(await collector.drain_steering(), [sensor])
            self.assertEqual(list(collector._priority_pending), [viewer])


if __name__ == '__main__':
    unittest.main()
