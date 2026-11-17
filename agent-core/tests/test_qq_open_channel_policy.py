"""QQ communication is open by default; device/system privilege is not."""
from types import SimpleNamespace
from unittest import mock

import fastapi
import pytest

from api.channel_setup import Pairing, pair_confirm, pair_start
from api.qq_settings import QQOwnerChange, update_owner
from channel import qq_diagnostics, qq_groups
from channel.qq_capabilities import SAFE_DEFAULTS, apply_safe_defaults, describe
from channel.tool_policy import _qq_safe, allowed_for_principals
import event.llm  # noqa: F401 - initialize before mocking config.main

APP = '1900000000'
CHANNEL = 'qq_official_' + APP
USER = 'official:' + APP + ':ABCDEFGHIJKLMNOPQRST'
GROUP = 'A_GROUP_VALID_OPENID'
GROUP_USER = f'official:{APP}:group:{GROUP}:ABCDEFGHIJKLMNOPQRST'


def test_safe_transport_defaults_preserve_explicit_opt_outs():
    cfg = apply_safe_defaults({
        'app_id': APP, 'protocol': 'official',
        'enabled_scopes': ['c2c', 'group'], 'enable_outbound_media': False,
    })
    assert cfg['enable_outbound_media'] is False
    for flag in SAFE_DEFAULTS:
        if flag != 'enable_outbound_media':
            assert cfg[flag] is True
    assert apply_safe_defaults(
        {'enable_outbound_media': False, 'enabled_scopes': ['c2c']},
        standard_mode=True)['enable_outbound_media'] is True
    assert 'enable_interactions' not in cfg
    assert 'enable_markdown' not in cfg
    assert 'guild' not in cfg['enabled_scopes']


@pytest.mark.parametrize('name', [
    'Bash', 'Read', 'Write', 'mcp__robot__move', 'mcp__watch7__watch_imu_stream',
    'mcp__sensors__camera', 'update_memory', 'search_history',
])
def test_qq_legacy_owner_and_normal_user_cannot_use_privileged_tools(name):
    for role in ('owner', 'viewer', 'operator'):
        assert not _qq_safe(name, role=role, user_id=USER)


def test_owner_wildcard_grant_cannot_bypass_runtime_tool_authorization():
    state = {'qq_c2c_owner_grant': {'enabled': True, 'principal': USER}}
    with (mock.patch('config.main', state),
          mock.patch('channel.acl.get_user', return_value={
              'role': 'owner', 'tool_filter': '*'})):
        assert not allowed_for_principals([('qq', USER)], 'Bash')
        assert not allowed_for_principals([('qq', USER)], 'mcp__robot__move')
        assert not allowed_for_principals([('qq', USER)], 'update_memory')
        assert allowed_for_principals([('qq', USER)], 'mcp__channel__channel_reply')
        assert allowed_for_principals([('qq', USER)], 'channel_memory_read')


def test_group_auto_discovers_without_admin_or_whitelist():
    channel = {'id': CHANNEL, 'platform': 'qq', 'enabled': True,
               'config': {'protocol': 'official', 'app_id': APP,
                          'enabled_scopes': ['c2c', 'group']}}
    msg = SimpleNamespace(platform='qq', chat_type='group',
                          sender_type='user', chat_id=GROUP,
                          user_id=GROUP_USER, channel_id=CHANNEL)
    state = {'qq_group_policies': {}, 'channel_configs': [channel]}
    with (mock.patch('config.main', state),
          mock.patch('channel.manager.get_channel_config', return_value=channel)):
        assert qq_groups.observe(msg)
        assert qq_groups.approved(CHANNEL, GROUP)
        assert qq_groups.observe(msg)
        assert _qq_safe('mcp__channel__channel_reply', role='viewer', user_id=GROUP_USER)
        assert not _qq_safe('mcp__robot__move', role='owner', user_id=GROUP_USER)
        qq_groups.update(CHANNEL, GROUP, default_permission='chat',
                         mention_only=True, enabled=False)
        assert not qq_groups.observe(msg)


def test_legacy_pair_and_owner_change_endpoints_retired():
    with (mock.patch('api.channel_setup.auth.is_enabled', return_value=True),
          mock.patch('api.channel_setup._qq', return_value={'platform': 'qq'})):
        with pytest.raises(fastapi.HTTPException) as exc:
            pair_start(Pairing(channel_id=CHANNEL))
        assert exc.value.status_code == 410
        with pytest.raises(fastapi.HTTPException) as exc:
            pair_confirm(Pairing(channel_id=CHANNEL))
        assert exc.value.status_code == 410
    with mock.patch('api.qq_settings._channel', return_value={'platform': 'qq'}):
        with pytest.raises(fastapi.HTTPException) as exc:
            update_owner(CHANNEL, QQOwnerChange(preset='full'))
        assert exc.value.status_code == 410


def test_capability_status_does_not_confuse_config_and_actual_delivery():
    cfg = apply_safe_defaults({'enabled_scopes': ['c2c', 'group']})
    empty = describe(cfg, gateway=True, flow_ready=True)
    assert empty['private_text']['state'] == 'awaiting_real_test'
    assert empty['outbound_media']['state'] == 'awaiting_real_test'
    assert empty['enable_interactions']['state'] == 'requires_platform_or_workflow'
    import time
    seen = describe(cfg, gateway=True, flow_ready=True, recent={
        'counts': {'private_inbound': 1, 'media_sent': 1},
        'recent': {'private_inbound': time.time(), 'media_sent': time.time()}})
    assert seen['private_text']['state'] == 'observed'
    assert seen['outbound_media']['state'] == 'observed'
    qq_diagnostics.record(CHANNEL, 'platform_accepted')
    assert qq_diagnostics.snapshot(CHANNEL)['counts']['platform_accepted'] >= 1
    qq_diagnostics.record(CHANNEL, 'untrusted_label')
    assert 'untrusted_label' not in qq_diagnostics.snapshot(CHANNEL)['counts']
