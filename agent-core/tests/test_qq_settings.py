"""QQ-only settings regression coverage: atomic apply, rollback, ACL isolation."""
import asyncio
from unittest import mock

import fastapi
import pytest

from api import qq_settings

APP = '1900000000'
CHANNEL = 'qq_official_' + APP
OWNER = 'official:' + APP + ':ABCDEFGHIJKLMNOP940FBB'
CREDENTIAL_KEY = 'app_' + 'secret_runtime'
BASE = {
    'protocol': 'official', 'app_id': APP,
    'enabled_scopes': ['c2c'], 'private_only': True,
    CREDENTIAL_KEY: 'A' * 43,
}
CONFIG = {'id': CHANNEL, 'platform': 'qq', 'enabled': True, 'config': BASE}


def test_standard_enables_safe_sdk_flags_without_upgrading_platform_scope():
    writes = []

    def update(channel_id, **kwargs):
        assert channel_id == CHANNEL
        writes.append(kwargs['config'])

    async def restart(channel_id, retries=1):
        qq_settings.manager._health[channel_id] = (True, '')

    with (mock.patch('api.qq_settings._channel', return_value=CONFIG),
          mock.patch('api.qq_settings._credential_available', return_value=True),
          mock.patch('api.qq_settings.update_channel_config', side_effect=update),
          mock.patch.object(qq_settings.manager, '_health', {}),
          mock.patch.object(qq_settings.manager, 'restart_adapter', side_effect=restart)):
        result = asyncio.run(qq_settings.save_settings(
            CHANNEL, qq_settings.QQSettingsChange(mode='standard')))

    assert result['saved'] and result['connected']
    saved = writes[0]
    assert saved[CREDENTIAL_KEY] == BASE[CREDENTIAL_KEY]
    assert saved['enabled_scopes'] == ['c2c']
    assert saved['private_only'] is True
    assert all(saved[flag] for flag in qq_settings.STANDARD_FLAGS)


def test_gateway_failure_restores_exact_old_config():
    changes = []

    def update(channel_id, **kwargs):
        changes.append(kwargs['config'])

    async def restart(channel_id, retries=1):
        qq_settings.manager._health[channel_id] = (
            (False, '4014 disallowed intents') if len(changes) == 1 else (True, ''))

    with (mock.patch('api.qq_settings._channel', return_value=CONFIG),
          mock.patch('api.qq_settings._credential_available', return_value=True),
          mock.patch('api.qq_settings.update_channel_config', side_effect=update),
          mock.patch.object(qq_settings.manager, '_health', {}),
          mock.patch.object(qq_settings.manager, 'restart_adapter', side_effect=restart)):
        with pytest.raises(fastapi.HTTPException) as error:
            asyncio.run(qq_settings.save_settings(
                CHANNEL, qq_settings.QQSettingsChange(
                    mode='advanced', enabled_scopes=['c2c', 'guild'])))

    assert error.value.status_code == 409
    assert error.value.detail['recovered']
    assert len(changes) == 2
    assert changes[0]['enabled_scopes'] == ['c2c', 'guild']
    assert changes[-1] == BASE
    assert all(item[CREDENTIAL_KEY] == BASE[CREDENTIAL_KEY] for item in changes)


def test_invalid_scopes_are_rejected_without_mutation():
    with (mock.patch('api.qq_settings._channel', return_value=CONFIG),
          mock.patch('api.qq_settings.update_channel_config') as update):
        with pytest.raises(fastapi.HTTPException) as error:
            asyncio.run(qq_settings.save_settings(
                CHANNEL, qq_settings.QQSettingsChange(
                    mode='advanced', enabled_scopes=['c2c', 'nope'])))

    assert error.value.status_code == 400
    update.assert_not_called()


def test_full_owner_permissions_preserve_verified_identity():
    record = {'role': 'owner', 'tool_filter': 'finish',
              'display_name': 'Owner'}
    with (mock.patch('api.qq_settings._channel', return_value=CONFIG),
          mock.patch('api.qq_settings._owner', return_value=(OWNER, record)),
          mock.patch('api.qq_settings.acl.upsert_user',
                     return_value={**record, 'tool_filter': '*'}) as update):
        result = qq_settings.update_owner(
            CHANNEL, qq_settings.QQOwnerChange(preset='full'))

    assert result['tool_filter'] == '*'
    assert update.call_args.args[:2] == ('qq', OWNER)
    assert update.call_args.kwargs['role'] == 'owner'
    assert update.call_args.kwargs['tool_filter'] == '*'


def test_unbound_owner_cannot_approve_full_tool_access():
    with (mock.patch('api.qq_settings._channel', return_value=CONFIG),
          mock.patch('api.qq_settings._owner', return_value=('', None)),
          mock.patch('api.qq_settings.acl.upsert_user') as update):
        with pytest.raises(fastapi.HTTPException) as error:
            qq_settings.update_owner(
                CHANNEL, qq_settings.QQOwnerChange(preset='full'))
    assert error.value.status_code == 409
    update.assert_not_called()


def test_qq_settings_refuse_non_qq_platform():
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config',
                     return_value={'platform': 'feishu'})):
        with pytest.raises(fastapi.HTTPException) as error:
            qq_settings.get_settings('feishu_main')
    assert error.value.status_code == 404




def test_recover_credential_preserves_identity_and_restores_safe_c2c():
    from channel import qq_runtime_credentials
    original = dict(BASE, enabled_scopes=['c2c', 'group', 'guild'],
                    private_only=False, enable_interactions=True)
    channel = dict(CONFIG, config=original)
    secret = 'x' * 32
    supplied_ref = qq_runtime_credentials.put(secret)
    qq_runtime_credentials.revoke(supplied_ref)
    # Use a deterministic non-sensitive ref to keep unit tests out of secrets.
    with (mock.patch('api.qq_settings._channel', return_value=channel),
          mock.patch('api.qq_settings.qq_runtime_credentials.put', return_value='R' * 43),
          mock.patch('api.qq_settings._save_config',
                     new_callable=mock.AsyncMock,
                     return_value={'saved':True,'changed':True,'connected':True}) as apply,
          mock.patch('api.qq_settings.qq_runtime_credentials.revoke') as revoke):
        result = asyncio.run(qq_settings.recover_qq_credential(
            CHANNEL, qq_settings.QQCredentialRecovery(app_secret=secret, persist=False)))
    new_cfg = apply.call_args.args[2]
    assert apply.call_args.args[1] == original
    assert new_cfg['app_id'] == APP
    assert new_cfg['enabled_scopes'] == ['c2c']
    assert new_cfg['private_only'] is True
    assert not new_cfg['enable_interactions']
    assert new_cfg['app_secret_runtime'] == 'R' * 43
    assert 'app_secret_file' not in new_cfg
    assert 'app_secret_env' not in new_cfg
    assert result['connected'] and result['credential_mode'] == 'runtime'
    assert result['restart_requires_reentry']
    revoke.assert_called_once_with(BASE[CREDENTIAL_KEY])


def test_recover_credential_failure_cleans_new_memory_reference():
    from channel import qq_runtime_credentials
    channel = dict(CONFIG, config=dict(BASE))
    with (mock.patch('api.qq_settings._channel', return_value=channel),
          mock.patch('api.qq_settings.qq_runtime_credentials.put', return_value='N' * 43),
          mock.patch('api.qq_settings._save_config', new_callable=mock.AsyncMock,
                     side_effect=fastapi.HTTPException(409, 'Gateway rejected')),
          mock.patch('api.qq_settings.qq_runtime_credentials.revoke') as revoke):
        with pytest.raises(fastapi.HTTPException) as error:
            asyncio.run(qq_settings.recover_qq_credential(
                CHANNEL, qq_settings.QQCredentialRecovery(app_secret='z'*32)))
    assert error.value.status_code == 409
    revoke.assert_called_once_with('N' * 43)

def test_group_subscription_reverts_on_unauthorized_gateway_intents():
    from api import channel_setup
    changes = []

    def update(channel_id, **kwargs):
        changes.append(kwargs['config'])

    async def restart(channel_id, retries=1):
        channel_setup.manager._health[channel_id] = (
            (False, '4014') if len(changes) == 1 else (True, ''))

    with (mock.patch('api.channel_setup._qq', return_value=CONFIG),
          mock.patch('api.qq_settings._channel', return_value=CONFIG),
          mock.patch('api.qq_settings.update_channel_config', side_effect=update),
          mock.patch('api.channel_setup.auth.is_enabled', return_value=True),
          mock.patch.object(channel_setup.manager, '_health', {}),
          mock.patch.object(channel_setup.manager, 'restart_adapter',
                            side_effect=restart)):
        with pytest.raises(fastapi.HTTPException) as error:
            asyncio.run(channel_setup.set_group_scope(
                channel_setup.QQGroupScope(
                    channel_id=CHANNEL, enabled=True,
                    platform_permission_confirmed=True)))

    assert error.value.status_code == 409
    assert changes[0]['enabled_scopes'] == ['c2c', 'group']
    assert changes[-1] == BASE
