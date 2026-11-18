"""QQ group auto-enable: authenticated @message, no duplicate verification."""
import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from channel import qq_groups
from channel.manager import ChannelManager
from channel import permissions
from channel.tool_policy import _qq_safe

APP = '1900000000'
CHANNEL = 'qq_official_' + APP
GROUP = 'groupABCDE123456789'
MEMBER = 'memberABCDEF1234'
IDENTITY = f'official:{APP}:group:{GROUP}:{MEMBER}'
CFG = {'id': CHANNEL, 'platform': 'qq', 'enabled': True,
       'config': {'app_id': APP, 'protocol': 'official',
                  'enabled_scopes': ['c2c', 'group'], 'private_only': False}}


def sdk_message(*, group=GROUP, identity=IDENTITY):
    return SimpleNamespace(
        platform='qq', channel_id=CHANNEL, chat_type='group',
        chat_id=group, user_id=identity, sender_type='user',
        display_name='Group member', text='@机器人 你好',
        message_id='group-msg-123', mentions=[], expect_reply=True,
        attachments=[])


def test_first_group_event_auto_joins_without_pairing():
    state = {}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CFG)):
        assert not qq_groups.approved(CHANNEL, GROUP)
        assert qq_groups.observe(sdk_message())
        assert qq_groups.approved(CHANNEL, GROUP)
        policy = qq_groups.get_group(CHANNEL, GROUP)
        assert policy['status'] == 'approved'
        assert policy['default_permission'] == 'chat'
        assert policy['mention_only'] is True
        assert qq_groups.observe(sdk_message())  # Idempotent, no second pairing
        assert len(qq_groups.list_groups(CHANNEL)) == 1
        with pytest.raises(ValueError, match='chat-only'):
            qq_groups.update(CHANNEL, GROUP, default_permission='daily',
                             mention_only=True, enabled=True)
        assert 'qq_group_challenge' not in state


@pytest.mark.parametrize('previous_status', ['pending', 'verified', 'unapproved'])
def test_legacy_unapproved_group_auto_enables_after_real_sdk_event(previous_status):
    state = {'qq_group_policies': {CHANNEL: {GROUP: {
        'status': previous_status, 'default_permission': 'daily'}}}}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CFG)):
        assert qq_groups.observe(sdk_message())
        assert qq_groups.approved(CHANNEL, GROUP)
        assert qq_groups.get_group(CHANNEL, GROUP)['default_permission'] == 'chat'


def test_legacy_approved_daily_group_migrates_to_chat_only():
    state = {'qq_group_policies': {CHANNEL: {GROUP: {
        'status': 'approved', 'default_permission': 'daily'}}}}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CFG),
          mock.patch('channel.manager._get_channel_configs', return_value=[CFG])):
        assert qq_groups.list_groups(CHANNEL)[0]['default_permission'] == 'chat'
        assert qq_groups.observe(sdk_message())
        assert qq_groups.get_group(CHANNEL, GROUP)['default_permission'] == 'chat'
        assert not _qq_safe('mcp__device__tts', role='operator', user_id=IDENTITY)
        assert not _qq_safe('Bash', role='owner', user_id=IDENTITY)


def test_disabling_discovered_group_is_sticky():
    state = {}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CFG)):
        assert qq_groups.observe(sdk_message())
        qq_groups.update(CHANNEL, GROUP, default_permission='chat',
                         mention_only=True, enabled=False)
        assert not qq_groups.observe(sdk_message())
        assert not qq_groups.approved(CHANNEL, GROUP)
        qq_groups.update(CHANNEL, GROUP, default_permission='chat',
                         mention_only=True, enabled=True)
        assert qq_groups.observe(sdk_message())


@pytest.mark.parametrize('bad', [
    {'platform':'feishu'},
    {'sender_type':'bot'},
    {'user_id':'official:99999:group:' + GROUP + ':' + MEMBER},
    {'chat_id':'another-group'},
    {'user_id':'official:' + APP + ':some-user'},
])
def test_untrusted_or_wrong_identity_does_not_auto_join(bad):
    state = {}
    msg = sdk_message()
    for key, value in bad.items():
        setattr(msg, key, value)
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CFG)):
        assert not qq_groups.observe(msg)
        assert state.get('qq_group_policies') is None


def test_group_scope_must_be_enabled_to_auto_join():
    state = {}
    no_groups = {**CFG, 'config': {**CFG['config'],
                 'enabled_scopes': ['c2c'], 'private_only': True}}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=no_groups)):
        assert not qq_groups.observe(sdk_message())


def test_auto_joined_group_has_chat_only_tool_ceiling():
    state = {}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CFG),
          mock.patch('channel.manager._get_channel_configs', return_value=[CFG])):
        assert qq_groups.observe(sdk_message())
        assert _qq_safe('mcp__channel__channel_reply', role='viewer', user_id=IDENTITY)
        assert not _qq_safe('Bash', role='owner', user_id=IDENTITY)
        assert not _qq_safe('mcp__robot__loco', role='owner', user_id=IDENTITY)
        assert not _qq_safe('mcp__device__tts', role='operator', user_id=IDENTITY)


def test_manager_first_group_message_reaches_canvas():
    state = {}
    mgr = ChannelManager()
    mgr._active_input_channels.add(CHANNEL)
    user = {'role': 'viewer', 'tool_filter': permissions.CHAT_FILTER}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CFG),
          mock.patch.object(mgr, 'sync_from_canvas'),
          mock.patch('channel.manager.acl.get_user', side_effect=[None, user]),
          mock.patch('channel.manager.acl.upsert_user') as register,
          mock.patch('channel.manager.acl.check_permission', return_value=(True, 'ok')),
          mock.patch.object(mgr, '_set_last_context'),
          mock.patch.object(mgr, '_record_chat_exchange'),
          mock.patch('api.inspection.publish_to_topic', new_callable=mock.AsyncMock) as publish,
          mock.patch('api.motus_stream.push_event', new_callable=mock.AsyncMock)):
        asyncio.run(mgr._on_inbound_message(sdk_message()))
        assert qq_groups.approved(CHANNEL, GROUP)
    assert publish.await_count == 1
    assert register.call_args.kwargs['role'] == 'viewer'


def test_manager_disabled_group_does_not_reach_canvas():
    state = {'qq_group_policies': {CHANNEL: {GROUP: {
        'status':'disabled', 'default_permission':'chat', 'mention_only':True}}}}
    mgr = ChannelManager()
    mgr._active_input_channels.add(CHANNEL)
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CFG),
          mock.patch.object(mgr, 'sync_from_canvas'),
          mock.patch('api.inspection.publish_to_topic', new_callable=mock.AsyncMock) as publish):
        asyncio.run(mgr._on_inbound_message(sdk_message()))
        assert publish.await_count == 0
        assert qq_groups.get_group(CHANNEL, GROUP)['status'] == 'disabled'
