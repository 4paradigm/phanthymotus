"""Channel turns get one retry when the model writes text without a tool call."""

import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import collector  # noqa: E402
from event.llm import _channel_tool_retry_message, _missed_channel_reply_warning  # noqa: E402


class ChannelReplyRetryTest(unittest.TestCase):
    def setUp(self):
        self.trigger = {
            'payload': {
                'sources': ['dds:/channel/request/wlcb_23'],
                'channel_ids': ['wlcb_23'],
            },
        }

    def test_retries_only_first_channel_round_with_text(self):
        retry = _channel_tool_retry_message(self.trigger, 0, '在的')

        self.assertIn('channel_reply', retry)
        self.assertIn('wlcb_23', retry)
        self.assertEqual(
            _channel_tool_retry_message(self.trigger, 0, '在的', retry_consumed=True),
            '',
        )
        self.assertEqual(_channel_tool_retry_message(self.trigger, 1, '在的'), '')
        self.assertEqual(_channel_tool_retry_message(self.trigger, 0, '  '), '')
        self.assertEqual(_channel_tool_retry_message({'payload': {'sources': []}}, 0, '在的'), '')

    def test_retry_consumption_survives_round_counter_reset(self):
        consumed = False
        retries = 0

        for round_idx in (0, 0, 0):  # max_rounds=1 resets the local counter each loop
            correction = _channel_tool_retry_message(
                self.trigger, round_idx, '在的', retry_consumed=consumed,
            )
            if correction:
                retries += 1
                consumed = True

        self.assertEqual(retries, 1)

    def test_retry_uses_original_non_ascii_channel_id(self):
        trigger = collector._build_trigger([{
            'source': 'dds:/channel/request/G1_0123456789',
            'text': json.dumps({
                'channel_id': '上海 G1',
                'sender_type': 'user',
            }, ensure_ascii=False),
            'ts': 1,
        }], urgent=True)

        retry = _channel_tool_retry_message(trigger, 0, '在的')

        self.assertEqual(trigger['payload']['channel_ids'], ['上海 G1'])
        self.assertIn('上海 G1', retry)
        self.assertNotIn('G1_0123456789', retry)


class MissedChannelReplyWarningTest(unittest.TestCase):
    def test_warns_when_one_of_two_messages_in_a_batch_is_unanswered(self):
        trigger = {'_channel_message_ids': ['om_a', 'om_b']}

        warning = _missed_channel_reply_warning(trigger, {'om_a'})

        self.assertIn('om_b', warning)
        self.assertNotIn('om_a', warning)

    def test_no_warning_when_every_message_got_a_reply(self):
        trigger = {'_channel_message_ids': ['om_a', 'om_b']}
        self.assertEqual(_missed_channel_reply_warning(trigger, {'om_a', 'om_b'}), '')

    def test_no_warning_without_channel_message_ids(self):
        self.assertEqual(_missed_channel_reply_warning({}, set()), '')
        self.assertEqual(
            _missed_channel_reply_warning({'_channel_message_ids': []}, set()), '')

    def test_build_trigger_exposes_all_channel_message_ids_human_and_bot(self):
        trigger = collector._build_trigger([
            {
                'source': 'dds:/channel/request/feishu_test',
                'text': json.dumps({'sender_type': 'user', 'message_id': 'om_human'}),
                'ts': 1,
            },
        ], urgent=True)

        self.assertEqual(trigger['_channel_message_ids'], ['om_human'])


if __name__ == '__main__':
    unittest.main()
