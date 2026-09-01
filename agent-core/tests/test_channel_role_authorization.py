"""ACL role (viewer/operator/owner) must actually be enforced on the tool
dispatch path for human Channel messages — not just carried as text for the
LLM to optionally respect."""

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
from event.llm import (  # noqa: E402
    _channel_tool_restricted,
    _restricted_channel_tool_allowed,
    _viewer_channel_restricted,
)


def _human_payload(role: str, text: str = 'hi') -> dict:
    return {
        'source': 'dds:/channel/request/feishu_test',
        'text': json.dumps({'sender_type': 'user', 'user_role': role, 'text': text}),
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

    def test_mixed_batch_with_an_operator_message_is_not_restricted(self):
        batch = [_human_payload('viewer'), _human_payload('operator')]
        self.assertFalse(collector.has_viewer_only_channel_event(batch))

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
            self.assertTrue(_restricted_channel_tool_allowed('finish'))
            self.assertTrue(_restricted_channel_tool_allowed('mcp__channel__channel_reply'))
            self.assertTrue(_restricted_channel_tool_allowed('mcp__device__camera'))
            self.assertFalse(_restricted_channel_tool_allowed('mcp__device__navigate'))
            self.assertFalse(_restricted_channel_tool_allowed('mcp__device__load_map'))


if __name__ == '__main__':
    unittest.main()
