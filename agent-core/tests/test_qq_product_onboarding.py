"""Local-only QQ first-run, access control and group enrollment acceptance tests."""
import asyncio
from types import SimpleNamespace
from unittest import mock

import fastapi
import pytest

from channel import qq_onboarding, qq_groups, qq_runtime_credentials
from channel import permissions
from channel.tool_policy import _qq_safe, _matches_filter
from channel.qq_default_flow import ensure_default_flow
from channel.adapters.qq_official import _load_credential, validate_qq_official_config
from api import channel_setup as setup


APP = '1900000000'
CHANNEL = 'qq_official_' + APP
PERSON = 'official:' + APP + ':ABCDEFGHIJKLMNOP940FBB'
GROUP = 'groupABCDE123456789'
GROUP_PERSON = 'official:' + APP + ':group:' + GROUP + ':memberABCDEF1234'
CONFIG = {'id': CHANNEL, 'platform': 'qq', 'enabled': True,
          'config': {'app_id': APP, 'protocol': 'official',
                     'enabled_scopes': ['c2c', 'group'], 'private_only': False}}


def qq_message(text, user=PERSON, chat_type='p2p', chat_id=None):
    return SimpleNamespace(
        platform='qq', channel_id=CHANNEL, user_id=user,
        chat_type=chat_type, chat_id=chat_id or user.split(':')[-1],
        sender_type='user', text=text)


def test_runtime_credential_never_in_channel_config():
    value = 'x' * 32
    ref = qq_runtime_credentials.put(value)
    cfg = {'protocol': 'official', 'app_id': APP, 'app_secret_runtime': ref}
    validate_qq_official_config(cfg)
    assert _load_credential(cfg) == value
    assert value not in repr(cfg)
    assert ref != value
    qq_runtime_credentials.revoke(ref)
    with pytest.raises(ValueError, match='not configured'):
        _load_credential(cfg)


def test_guided_setup_does_not_write_secret_to_disk_or_config():
    stored = {}
    password = 'x' * 32
    with (
        mock.patch('api.channel_setup.auth.is_enabled', return_value=True),
        mock.patch('api.channel_setup._get_channel_configs', return_value=[]),
        mock.patch('api.channel_setup.add_channel_config') as create,
        mock.patch('api.channel_setup.ensure_default_flow', return_value={'status':'created','ready':True}),
        mock.patch.object(setup.manager, 'restart_adapter', new_callable=mock.AsyncMock),
        mock.patch.object(setup.manager, '_health', {CHANNEL: (True, '')}),
    ):
        result = asyncio.run(setup.guided_setup(setup.QQSetup(app_id=APP, app_secret=password, source='runtime')))
        cfg = create.call_args.args[2]
        stored.update(cfg)
    assert result['connected']
    assert result['default_flow']['ready']
    assert 'app_secret' not in cfg
    assert 'app_secret_file' not in cfg
    assert 'app_secret_runtime' in cfg
    assert password not in repr(cfg)
    assert _load_credential(cfg) == password
    qq_runtime_credentials.revoke(cfg['app_secret_runtime'])


def test_setup_requires_dashboard_auth():
    with mock.patch('api.channel_setup.auth.is_enabled', return_value=False):
        with pytest.raises(fastapi.HTTPException) as ex:
            asyncio.run(setup.guided_setup(setup.QQSetup(app_id=APP, app_secret='x'*32)))
    assert ex.value.status_code == 403


def test_owner_binding_requires_real_private_message_and_dashboard_confirmation():
    state = {}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CONFIG),
          mock.patch('channel.acl.get_user', return_value=None),
          mock.patch('channel.acl.upsert_user') as grant):
        code = qq_onboarding.start(CHANNEL)
        assert not qq_onboarding.receive(qq_message('绑定 ' + code, chat_type='group'))
        assert not qq_onboarding.receive(qq_message('绑定 WRONG000'))
        assert qq_onboarding.status(CHANNEL)['status'] == 'waiting'
        assert qq_onboarding.receive(qq_message('绑定 ' + code))
        assert not qq_onboarding.receive(qq_message('绑定 ' + code))
        assert qq_onboarding.status(CHANNEL)['status'] == 'verified'
        result = qq_onboarding.confirm(CHANNEL)
        assert result['status'] == 'bound'
        assert state['qq_c2c_owner_grant']['principal'] == PERSON
        assert grant.call_args.kwargs['role'] == 'owner'
        assert grant.call_args.kwargs['tool_filter'] == permissions.OWNER_FILTER
        assert qq_onboarding.status(CHANNEL)['status'] == 'expired'
    assert not _qq_safe('Bash', role='owner', user_id='official:' + APP + ':other')


def test_verified_owner_full_access_restored_without_escalating_other_users():
    state = {'qq_c2c_owner_grant': {'enabled': True, 'principal': PERSON}}
    old = {'role': 'owner', 'display_name': 'QQ owner',
           'tool_filter': permissions.DAILY_FILTER}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CONFIG),
          mock.patch('channel.acl.get_user', return_value=old),
          mock.patch('channel.acl.upsert_user') as update):
        result = qq_onboarding.restore_owner_full_access(CHANNEL)
        assert result['status'] == 'full_access'
        assert update.call_args.kwargs['role'] == 'owner'
        assert update.call_args.kwargs['tool_filter'] == permissions.OWNER_FILTER
        assert _qq_safe('Bash', role='owner', user_id=PERSON)
        assert _matches_filter(permissions.OWNER_FILTER, 'Bash')
        assert _matches_filter(permissions.OWNER_FILTER, 'mcp__robot__loco')
        assert not _qq_safe('Bash', role='operator', user_id=PERSON)
        assert not _qq_safe('Bash', role='owner',
                            user_id='official:' + APP + ':OTHER_PERSON_12345')
        assert not _qq_safe('Bash', role='owner', user_id=GROUP_PERSON)

    with (mock.patch('config.main', {'qq_c2c_owner_grant': {
            'enabled': True, 'principal': 'official:99999:OTHER_PERSON_12345'}}),
          mock.patch('channel.manager.get_channel_config', return_value=CONFIG),
          mock.patch('channel.acl.get_user', return_value=old),
          mock.patch('channel.acl.upsert_user') as update):
        with pytest.raises(ValueError, match='No verified owner'):
            qq_onboarding.restore_owner_full_access(CHANNEL)
        update.assert_not_called()


def test_daily_preset_never_grants_shell():
    with mock.patch('config.main', {}):
        assert not _qq_safe('mcp__device__tts', role='operator', user_id=PERSON)
        assert _qq_safe('mcp__channel__channel_reply', role='viewer', user_id=PERSON)
        assert not _qq_safe('Bash', role='operator', user_id=PERSON)
        assert not _qq_safe('mcp__robot__loco', role='operator', user_id=PERSON)
        assert _matches_filter(permissions.DAILY_FILTER, 'mcp__perception__tts')


def test_qq_group_auto_enables_without_binding_and_has_independent_ceiling():
    state = {}
    cfg = dict(CONFIG, config=dict(CONFIG['config']))
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=cfg),
          mock.patch('channel.manager._get_channel_configs', return_value=[cfg])):
        assert not qq_groups.approved(CHANNEL, GROUP)
        assert not qq_groups.observe(qq_message(
            '你好', user='official:99999:group:' + GROUP + ':memberABCDEF1234',
            chat_type='group', chat_id=GROUP))
        assert qq_groups.observe(qq_message(
            '@机器人 你好', user=GROUP_PERSON, chat_type='group', chat_id=GROUP))
        assert qq_groups.approved(CHANNEL, GROUP)
        assert not _qq_safe('mcp__device__tts', role='operator', user_id=GROUP_PERSON)
        with pytest.raises(ValueError, match='chat-only'):
            qq_groups.update(CHANNEL, GROUP, default_permission='daily',
                             mention_only=True, enabled=True)
        assert not _qq_safe('mcp__device__tts', role='operator', user_id=GROUP_PERSON)
        assert not _qq_safe('Bash', role='owner', user_id=GROUP_PERSON)
        qq_groups.update(CHANNEL, GROUP, default_permission='chat',
                         mention_only=True, enabled=False)
        assert not qq_groups.observe(qq_message(
            '@机器人 仍然无法访问', user=GROUP_PERSON, chat_type='group', chat_id=GROUP))


def test_default_canvas_is_idempotent_and_does_not_overwrite_existing():
    state = {}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CONFIG)):
        first = ensure_default_flow(CHANNEL)
        second = ensure_default_flow(CHANNEL)
        assert first['status'] == 'created'
        assert second['status'] == 'existing'
        assert len(state['canvas_layout']['cards']) == 3
        assert len(state['canvas_layout']['connections']) == 1
        keys = [k for k in state if k.startswith('tool_config:channel:')]
        assert len(keys) == 2
        assert all(state[k]['channel_id'] == CHANNEL for k in keys)
        state['canvas_layout'] = {'cards': [{'id':'custom','mcpId':'x','toolName':'x'}],
                                  'connections':[]}
        result = ensure_default_flow(CHANNEL)
        assert result['status'] == 'manual_required'
        assert state['canvas_layout']['cards'][0]['id'] == 'custom'


def test_admin_preset_does_not_override_existing_binding():
    with (mock.patch('channel.acl.get_user', return_value={
            'role':'owner', 'display_name':'Owner'}),
          mock.patch('config.main', {'qq_c2c_owner_grant':{
              'enabled':True, 'principal':PERSON}})):
        with pytest.raises(ValueError, match='Revoke'):
            permissions.assign_preset('qq', PERSON, 'daily')




def test_manager_ignores_legacy_approval_setting_and_registers_qq_user():
    from channel.manager import ChannelManager
    mgr = ChannelManager()
    mgr._active_input_channels.add(CHANNEL)
    request = qq_message('hello')
    request.display_name = 'QQ User'
    request.message_id = 'legacy-approval-message'
    request.mentions = []
    request.expect_reply = True
    request.attachments = []
    user = {'role': 'viewer', 'tool_filter': permissions.CHAT_FILTER}
    with (mock.patch.object(mgr, 'sync_from_canvas'),
          mock.patch.object(mgr, '_get_channel_settings', return_value={}),
          mock.patch('channel.manager.get_channel_config',
                     return_value=dict(CONFIG, config=dict(CONFIG['config'],
                         access_policy={'new_user_policy':'approval',
                                        'default_permission':'chat'}))),
          mock.patch('channel.manager.acl.get_user', side_effect=[None, user]),
          mock.patch('channel.manager.acl.upsert_user') as register,
          mock.patch('channel.manager.acl.check_permission', return_value=(True,'ok')),
          mock.patch.object(mgr, '_set_last_context'),
          mock.patch.object(mgr, '_record_chat_exchange'),
          mock.patch('api.inspection.publish_to_topic', new_callable=mock.AsyncMock) as publish,
          mock.patch('api.motus_stream.push_event', new_callable=mock.AsyncMock)):
        asyncio.run(mgr._on_inbound_message(request))
    assert register.call_args.kwargs['role'] == 'viewer'
    assert register.call_args.kwargs['tool_filter'] == permissions.CHAT_FILTER
    publish.assert_awaited_once()


def test_manager_ignores_legacy_daily_policy_and_registers_qq_chat_only():
    from channel.manager import ChannelManager
    mgr = ChannelManager()
    mgr._active_input_channels.add(CHANNEL)
    message = qq_message('你好')
    message.display_name = 'QQ User'
    message.message_id = 'mid-1'
    message.mentions = []
    message.expect_reply = True
    message.attachments = []
    user = {'role':'operator', 'tool_filter':permissions.DAILY_FILTER}
    with (mock.patch.object(mgr, 'sync_from_canvas'),
          mock.patch.object(mgr, '_get_channel_settings', return_value={}),
          mock.patch('channel.manager.get_channel_config',
                     return_value=dict(CONFIG, config=dict(CONFIG['config'],
                         access_policy={'new_user_policy':'allow',
                                        'default_permission':'daily'}))),
          mock.patch('channel.manager.acl.get_user', side_effect=[None, user]),
          mock.patch('channel.manager.acl.upsert_user') as register,
          mock.patch('channel.manager.acl.check_permission', return_value=(True,'ok')),
          mock.patch.object(mgr, '_set_last_context'),
          mock.patch.object(mgr, '_record_chat_exchange'),
          mock.patch('api.inspection.publish_to_topic', new_callable=mock.AsyncMock) as published,
          mock.patch('api.motus_stream.push_event', new_callable=mock.AsyncMock)):
        asyncio.run(mgr._on_inbound_message(message))
    assert register.call_args.kwargs['role'] == 'viewer'
    assert register.call_args.kwargs['tool_filter'] == permissions.CHAT_FILTER
    assert published.await_count == 1


def test_manager_disabled_group_never_reaches_canvas():
    from channel.manager import ChannelManager
    mgr = ChannelManager()
    mgr._active_input_channels.add(CHANNEL)
    msg = qq_message('你好', user=GROUP_PERSON, chat_type='group', chat_id=GROUP)
    with (mock.patch('channel.qq_groups.observe', return_value=False),
          mock.patch.object(mgr, 'sync_from_canvas'),
          mock.patch('api.inspection.publish_to_topic',
                     new_callable=mock.AsyncMock) as published):
        asyncio.run(mgr._on_inbound_message(msg))
    published.assert_not_awaited()



def test_owner_revocation_removes_special_grant_and_demotes_owner():
    state = {'qq_c2c_owner_grant': {'enabled': True, 'principal': PERSON}}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=CONFIG),
          mock.patch('channel.acl.get_user',
                     return_value={'role':'owner','display_name':'Owner'}),
          mock.patch('channel.acl.upsert_user') as update):
        result = qq_onboarding.revoke(CHANNEL)
    assert result['status'] == 'revoked'
    assert state['qq_c2c_owner_grant'] == {}
    assert update.call_args.kwargs['role'] == 'viewer'
    assert update.call_args.kwargs['tool_filter'] == permissions.CHAT_FILTER


def test_default_policy_migrates_legacy_approval_without_touching_permissions():
    cfg = dict(CONFIG, config=dict(CONFIG['config'],
        access_policy={'new_user_policy': 'approval', 'default_permission': 'chat'}))
    with (mock.patch('api.channel_setup.auth.is_enabled', return_value=True),
          mock.patch('api.channel_setup.get_channel_config', return_value=cfg),
          mock.patch('api.channel_setup.update_channel_config') as save):
        assert setup.get_access_policy(CHANNEL) == {
            'policy': {'default_permission': 'chat'}}
        with pytest.raises(fastapi.HTTPException) as blocked:
            setup.set_access_policy(
                CHANNEL, setup.AccessPolicy(default_permission='daily'))
        assert blocked.value.status_code == 400
        result = setup.set_access_policy(CHANNEL, setup.AccessPolicy(default_permission='chat'))
    assert result['policy'] == {'default_permission': 'chat'}
    saved = save.call_args.kwargs['config']
    assert saved['access_policy'] == {'default_permission': 'chat'}
    assert saved['enabled_scopes'] == ['c2c', 'group']


def test_qq_guided_endpoints_are_registered_and_http_accessible():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(setup.router, prefix='/api')
    with (mock.patch('api.channel_setup.auth.is_enabled', return_value=True),
          mock.patch('api.channel_setup._get_channel_configs', return_value=[]),
          mock.patch.object(setup.manager, 'sync_from_canvas')):
        response = TestClient(app).get('/api/channel/qq/setup/status')
    assert response.status_code == 200
    assert response.json()['channels'] == []
    paths = {route.path for route in setup.router.routes}
    assert '/channel/qq/guided-setup' in paths
    assert '/channel/qq/pair/confirm' in paths
    assert '/channel/qq/groups/permission' in paths
    assert '/channel/users/preset' in paths
    assert '/channel/users/pending/resolve' not in paths


def test_secret_is_revoked_and_owner_grant_cleared_on_channel_deletion():
    from api import channel as channel_api
    ref = qq_runtime_credentials.put('x' * 32)
    state = {
        'qq_c2c_owner_grant': {'enabled':True, 'principal':PERSON},
        'qq_group_policies': {CHANNEL: {GROUP: {'status':'approved'}}},
        'channel_pending_users': {CHANNEL: {PERSON: {'display_name':'A'}}},
    }
    channel = dict(CONFIG, config=dict(CONFIG['config'], app_secret_runtime=ref))
    with (mock.patch('config.main', state),
          mock.patch('api.channel.get_channel_config', return_value=channel),
          mock.patch('api.channel.delete_channel_config', return_value=True),
          mock.patch.object(channel_api.manager, '_adapters', {})):
        result = asyncio.run(channel_api.delete_channel(CHANNEL))
    assert result['deleted'] == CHANNEL
    assert not qq_runtime_credentials.available(ref)
    assert state['qq_c2c_owner_grant'] == {}
    assert CHANNEL not in state['qq_group_policies']
    assert CHANNEL not in state['channel_pending_users']

def test_qq_guild_traffic_keeps_existing_sdk_route():
    """Ordinary QQ group discovery must not silently disable QQ Guild."""
    from channel.manager import ChannelManager
    manager = ChannelManager()
    guild_msg = qq_message(
        'hello',
        user='official:' + APP + ':guild:' + GROUP + ':memberABCDEF1234',
        chat_type='group', chat_id=GROUP)
    with mock.patch.object(manager, 'sync_from_canvas') as sync:
        asyncio.run(manager._on_inbound_message(guild_msg))
    sync.assert_called_once()


def test_group_scope_enable_requires_platform_confirmation_and_keeps_c2c():
    cfg = dict(CONFIG, config={
        'app_id': APP, 'protocol': 'official',
        'enabled_scopes': ['c2c'], 'private_only': True,
        'app_secret_runtime': 'A' * 43,
    })
    with (mock.patch('api.channel_setup.auth.is_enabled', return_value=True),
          mock.patch('api.channel_setup.get_channel_config', return_value=cfg),
          mock.patch('api.qq_settings._channel', return_value=cfg),
          mock.patch('api.qq_settings.update_channel_config') as update,
          mock.patch.object(setup.manager, '_health', {CHANNEL: (True, '')}),
          mock.patch.object(setup.manager, 'restart_adapter', new_callable=mock.AsyncMock)):
        with pytest.raises(fastapi.HTTPException) as exc:
            asyncio.run(setup.set_group_scope(
                setup.QQGroupScope(channel_id=CHANNEL, enabled=True)))
        assert exc.value.status_code == 400
        update.assert_not_called()
        result = asyncio.run(setup.set_group_scope(
            setup.QQGroupScope(channel_id=CHANNEL, enabled=True,
                               platform_permission_confirmed=True)))
    assert result['group_scope_enabled']
    saved = update.call_args.kwargs['config']
    assert saved['enabled_scopes'] == ['c2c', 'group']
    assert saved['private_only'] is False
    assert saved['app_secret_runtime'] == 'A' * 43


def test_deprecated_qq_approval_routes_are_removed():
    paths = {route.path for route in setup.router.routes}
    assert '/channel/qq/groups/pair/start' not in paths
    assert '/channel/qq/groups/pair/status' not in paths
    assert '/channel/qq/groups/pair/confirm' not in paths
    assert '/channel/users/pending' not in paths
    assert '/channel/users/pending/resolve' not in paths
