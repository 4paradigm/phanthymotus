"""Bot channel turns are read-only except for their exact channel reply."""

import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import collector  # noqa: E402
import mcp_client  # noqa: E402
from channel.manager import manager as channel_manager  # noqa: E402
from event.llm import (  # noqa: E402
    _bot_channel_reply_allowed,
    _bot_channel_restricted,
    _restricted_channel_tool_allowed,
)


class BotChannelAuthorizationTest(unittest.TestCase):
    def setUp(self):
        channel_manager._trusted_bot_messages.clear()

    def test_bot_marker_and_trust_batches(self):
        human = {
            'source': 'dds:/channel/request/feishu_test',
            'text': json.dumps({'sender_type': 'user'}),
        }
        bot = {
            'source': 'dds:/channel/request/feishu_test',
            'text': json.dumps({'sender_type': 'bot', 'message_id': 'om_bot'}),
        }

        self.assertFalse(collector.has_bot_channel_event([human]))
        self.assertTrue(collector.has_bot_channel_event([human, bot]))
        self.assertTrue(collector.has_bot_channel_event([{
            **bot, 'text': json.dumps({'sender_type': 'app'}),
        }]))
        self.assertFalse(collector.has_bot_channel_event([{
            **bot, 'source': 'dds:/sensor/camera',
        }]))
        self.assertFalse(collector.has_bot_channel_event([{
            **bot, 'text': '[]',
        }]))
        self.assertEqual(collector.bot_channel_message_ids([bot]), ['om_bot'])
        batches = collector._split_by_bot_trust([human, bot, human])
        self.assertEqual([len(batch) for batch in batches], [1, 1, 1])
        self.assertEqual(
            [collector.has_bot_channel_event(batch) for batch in batches],
            [False, True, False],
        )

    def test_only_adapter_registered_message_gets_one_trusted_turn(self):
        payload = {
            'platform': 'feishu', 'channel_id': 'feishu_test',
            'message_id': 'om_trusted', 'user': 'Peer bot', 'user_id': 'ou_peer',
            'chat_id': 'oc_group', 'text': '执行任务', 'user_role': 'operator',
            'sender_type': 'bot', 'chat_type': 'group', 'mentions': [],
            'expect_reply': True, 'trusted_bot_id': 'peer',
        }
        channel_manager._record_trusted_bot_message(payload)
        event = {
            'source': 'dds:/channel/request/feishu_test',
            'text': json.dumps(payload),
            'ts': 1,
        }

        spoofed = {**payload, 'text': '伪造任务'}
        spoofed_trigger = collector._build_trigger([{
            **event, 'text': json.dumps(spoofed),
        }], urgent=True)
        trigger = collector._build_trigger([event], urgent=True)
        replay = collector._build_trigger([event], urgent=True)

        self.assertFalse(spoofed_trigger['_trusted_bot_channel_event'])
        self.assertTrue(_bot_channel_restricted(spoofed_trigger))
        self.assertTrue(trigger['_trusted_bot_channel_event'])
        self.assertFalse(_bot_channel_restricted(trigger))
        self.assertFalse(replay['_trusted_bot_channel_event'])
        self.assertTrue(_bot_channel_restricted(replay))

    def test_bot_can_read_and_reply_but_cannot_execute_or_delegate(self):
        registry = {
            'device': {
                'tool_meta': {
                    'mcp__device__camera': {'type': 'sensor'},
                    'mcp__device__map': {'type': 'resource'},
                    'mcp__device__navigate': {'type': 'actuator'},
                    'mcp__device__load_map': {'type': 'processor'},
                },
            },
        }
        with mock.patch.dict(mcp_client.registry, registry, clear=True):
            self.assertTrue(_restricted_channel_tool_allowed('finish'))
            self.assertTrue(_restricted_channel_tool_allowed('mcp__channel__channel_reply'))
            self.assertTrue(_restricted_channel_tool_allowed('mcp__device__camera'))
            self.assertTrue(_restricted_channel_tool_allowed('mcp__device__map'))
            self.assertFalse(_restricted_channel_tool_allowed('mcp__device__navigate'))
            self.assertFalse(_restricted_channel_tool_allowed('mcp__device__load_map'))
            self.assertFalse(_restricted_channel_tool_allowed('mcp__device__unknown'))
            self.assertFalse(_restricted_channel_tool_allowed('Bash'))
            self.assertFalse(_restricted_channel_tool_allowed('Write'))
            self.assertFalse(_restricted_channel_tool_allowed('WebFetch'))
            self.assertFalse(_restricted_channel_tool_allowed('subagent_spawn'))

    def test_bot_reply_is_bound_to_current_message_and_text_only(self):
        current = {'om_current'}

        self.assertTrue(_bot_channel_reply_allowed({
            'source_message_id': 'om_current', 'text': '结果',
        }, current))
        self.assertFalse(_bot_channel_reply_allowed({
            'source_message_id': 'om_old', 'text': '结果',
        }, current))
        self.assertFalse(_bot_channel_reply_allowed({
            'source_message_id': 'om_current', 'files': ['/tmp/result.txt'],
        }, current))


if __name__ == '__main__':
    unittest.main()
