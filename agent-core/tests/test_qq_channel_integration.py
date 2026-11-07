"""QQ Channel security, registration and Canvas integration tests."""
import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import collector  # noqa: E402
from channel.adapter import InboundMessage  # noqa: E402
from channel.manager import ChannelManager  # noqa: E402
from channel.tool_policy import allowed_for_principals  # noqa: E402


class QQSecurityTests(unittest.TestCase):
    def test_t05_viewer_cannot_act(self):
        with mock.patch('channel.acl.get_user', return_value={'role': 'viewer',
                                                               'tool_filter': '*'}):
            self.assertTrue(allowed_for_principals([('qq', '888')], 'mcp__channel__channel_reply'))
            self.assertFalse(allowed_for_principals([('qq', '888')], 'mcp__perception__tts'))
            self.assertFalse(allowed_for_principals([('qq', '888')], 'mcp__bumi__loco'))

    def test_operator_cannot_execute_even_audio_in_text_only_v1(self):
        with mock.patch('channel.acl.get_user', return_value={'role': 'operator',
                                                               'tool_filter': '*'}):
            self.assertTrue(allowed_for_principals(
                [('qq', '888')], 'mcp__channel__channel_reply'))
            for name in ('mcp__perception__tts__speak',
                         'mcp__perception__phanthy_music__play',
                         'mcp__bumi__speaker', 'mcp__bumi__loco',
                         'mcp__actucore__navigate', 'Bash', 'subagent_spawn'):
                self.assertFalse(allowed_for_principals([('qq', '888')], name), name)

    def test_acl_filter_is_enforced(self):
        with mock.patch('channel.acl.get_user', return_value={
            'role': 'owner', 'tool_filter': 'tts*'}):
            self.assertFalse(allowed_for_principals(
                [('qq', '888')], 'mcp__channel__channel_reply'))
            self.assertFalse(allowed_for_principals(
                [('qq', '888')], 'mcp__perception__tts__speak'))
        with mock.patch('channel.acl.get_user', return_value={
            'role': 'operator', 'tool_filter': 'channel_reply'}):
            self.assertTrue(allowed_for_principals(
                [('qq', '888')], 'mcp__channel__channel_reply'))

    def test_blocked_deleted_and_mixed(self):
        def user(platform, uid):
            return {'role': 'viewer' if uid == '999' else 'operator',
                    'tool_filter': '*'}
        with mock.patch('channel.acl.get_user', side_effect=user):
            self.assertFalse(allowed_for_principals(
                [('qq', '888'), ('qq', '999')], 'mcp__perception__tts'))
        with mock.patch('channel.acl.get_user', return_value=None):
            self.assertFalse(allowed_for_principals(
                [('qq', '888')], 'mcp__channel__channel_reply'))
        with mock.patch('channel.acl.get_user', return_value={'role': 'blocked',
                                                               'tool_filter': '*'}):
            self.assertFalse(allowed_for_principals([('qq', '888')], 'finish'))

    def test_collector_carries_principals(self):
        msg = {'source': 'dds:/channel/request/qq_main', 'ts': 1,
               'text': json.dumps({'platform': 'qq', 'user_id': '888',
                                   'user_role': 'operator', 'sender_type': 'user',
                                   'channel_id': 'qq_main', 'message_id': '19',
                                   'chat_id': '888', 'text': 'hi'})}
        trigger = collector._build_trigger([msg], urgent=True)
        self.assertEqual(trigger['_channel_principals'], [('qq', '888')])


class QQManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_user_defaults_to_viewer_even_if_global_operator(self):
        manager = ChannelManager()
        manager._active_input_channels.add('qq_main')
        manager.sync_from_canvas = mock.Mock()
        manager._get_channel_settings = mock.Mock(return_value={
            'default_role': 'operator', 'auto_approve': True})
        manager._set_last_context = mock.Mock()
        manager._record_chat_exchange = mock.Mock()
        message = InboundMessage(platform='qq', channel_id='qq_main',
            user_id='888', chat_id='888', display_name='Test', text='hello',
            message_id='11', chat_type='p2p')
        with (mock.patch('channel.acl.get_user', side_effect=[
                  None, {'role': 'viewer', 'tool_filter': '*'}]),
              mock.patch('channel.acl.upsert_user') as upsert,
              mock.patch('channel.acl.check_permission', return_value=(True, 'ok')),
              mock.patch('api.inspection.publish_to_topic', new_callable=mock.AsyncMock) as pub,
              mock.patch('api.motus_stream.push_event', new_callable=mock.AsyncMock)):
            await manager._on_inbound_message(message)
        self.assertEqual(upsert.call_args.kwargs['role'], 'viewer')
        self.assertEqual(json.loads(pub.call_args.args[1])['user_role'], 'viewer')


if __name__ == '__main__':
    unittest.main()
