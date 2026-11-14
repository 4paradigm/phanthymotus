"""QQ SDK standard/advanced updates must preserve credentials and restore failures."""
import asyncio
from unittest import mock

import fastapi
import pytest

from api import qq_settings as settings
from channel import permissions

APP = '1900000000'
CID = 'qq_official_' + APP
OWNER = 'official:' + APP + ':ABCDEFGHIJKLMNOP940FBB'


def channel_data():
    return {'id': CID, 'platform': 'qq', 'enabled': True,
            'config': {'protocol': 'official', 'app_id': APP,
                       'enabled_scopes': ['c2c'], 'private_only': True,
                       'enable_inbound_media': True,
                       'app_secret_runtime': 'A' * 43}}


def fixture(fake_online=True):
    """Simulate only the QQ connection, keeping real validation and API logic."""
    store = channel_data()
    reports = []
    def get(identifier):
        return store if identifier == CID else None
    def put(identifier, **updates):
        assert identifier == CID
        reports.append(updates)
        store.update(updates)
        return store
    async def restart(identifier, retries=3):
        assert identifier == CID and retries == 1
        settings.manager._health[identifier] = (fake_online, 'intent denied' if not fake_online else '')
    return store, reports, get, put, restart


def test_settings_get_no_inline_secret_leaked():
    data = channel_data()
    data['config']['app_secret_runtime'] = 'X' * 43
    owner = {'role':'owner', 'tool_filter':'*', 'display_name':'Owner'}
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config', return_value=data),
          mock.patch('api.qq_settings._owner', return_value=(OWNER, owner))):
        result = settings.get_settings(CID)
    assert result['credential_mode'] == 'runtime'
    assert result['owner']['filter'] == '*'
    assert 'X' * 43 not in repr(result)
    assert result['enabled_scopes'] == ['c2c']


def test_standard_mode_preserves_credentials_and_enables_safe_features():
    data, history, get, put, restart = fixture()
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config', side_effect=get),
          mock.patch('api.qq_settings.update_channel_config', side_effect=put),
          mock.patch.object(settings.manager, 'restart_adapter', side_effect=restart),
          mock.patch.object(settings.manager, '_health', {}),
          mock.patch('api.qq_settings._credential_available', return_value=True)):
        result = asyncio.run(settings.save_settings(CID, settings.QQSettingsChange(
            mode='standard', group_enabled=False)))
    assert result['saved'] and result['connected']
    assert data['config']['enabled_scopes'] == ['c2c']
    assert data['config']['app_secret_runtime'] == 'A' * 43
    for key in settings.STANDARD_FLAGS:
        assert data['config'][key] is True
    assert all('app_secret' not in update for update in history)
    assert len(history) == 1


def test_advanced_invalid_scope_does_not_save_or_restart():
    data, history, get, put, restart = fixture()
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config', side_effect=get),
          mock.patch('api.qq_settings.update_channel_config', side_effect=put),
          mock.patch.object(settings.manager, 'restart_adapter', side_effect=restart) as restart_mock):
        with pytest.raises(fastapi.HTTPException) as error:
            asyncio.run(settings.save_settings(CID, settings.QQSettingsChange(
                mode='advanced', enabled_scopes=['c2c','unsafe'])))
    assert error.value.status_code == 400
    assert not history
    restart_mock.assert_not_called()


def test_group_permission_denied_rolls_back_previous_c2c_config():
    data, history, get, put, _ = fixture()
    times = [0]
    async def restart(identifier, retries=1):
        times[0] += 1
        settings.manager._health[identifier] = (
            times[0] != 1, 'QQ Gateway 4014 disallowed intents' if times[0] == 1 else '')
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config', side_effect=get),
          mock.patch('api.qq_settings.update_channel_config', side_effect=put),
          mock.patch.object(settings.manager, 'restart_adapter', side_effect=restart),
          mock.patch.object(settings.manager, '_health', {}),
          mock.patch('api.qq_settings._credential_available', return_value=True)):
        with pytest.raises(fastapi.HTTPException) as error:
            asyncio.run(settings.save_settings(CID, settings.QQSettingsChange(
                mode='standard', group_enabled=True)))
    assert error.value.status_code == 409
    assert error.value.detail['recovered'] is True
    assert times[0] == 2
    assert data['config'] == channel_data()['config']
    assert history[-1]['config'] == channel_data()['config']


def test_owner_can_change_full_chat_and_custom_immediately_without_gateway_restart():
    owner = {'role':'owner','display_name':'Owner','tool_filter':'*'}
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config', return_value=channel_data()),
          mock.patch('api.qq_settings._owner', return_value=(OWNER,owner)),
          mock.patch('api.qq_settings.acl.upsert_user', return_value={
              'role':'owner','tool_filter':permissions.CHAT_FILTER}) as update,
          mock.patch.object(settings.manager, 'restart_adapter') as restart):
        result = settings.update_owner(CID, settings.QQOwnerChange(preset='chat'))
        assert result['role'] == 'owner'
        assert update.call_args.kwargs['tool_filter'] == permissions.CHAT_FILTER
        settings.update_owner(CID, settings.QQOwnerChange(
            preset='custom',capabilities=['speech','motion']))
        assert 'mcp__*__loco*' in update.call_args.kwargs['tool_filter']
        assert 'Bash' not in update.call_args.kwargs['tool_filter']
        restart.assert_not_called()


def test_owner_controls_reject_unverified_user():
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config', return_value=channel_data()),
          mock.patch('api.qq_settings._owner', return_value=('',None)),
          mock.patch('api.qq_settings.acl.upsert_user') as upsert):
        with pytest.raises(fastapi.HTTPException) as error:
            settings.update_owner(CID, settings.QQOwnerChange(preset='full'))
    assert error.value.status_code == 409
    upsert.assert_not_called()


def test_unavailable_credential_rejects_setting_changes_without_restart():
    data, history, get, put, restart = fixture()
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config', side_effect=get),
          mock.patch('api.qq_settings.update_channel_config', side_effect=put),
          mock.patch('api.qq_settings._credential_available', return_value=False),
          mock.patch.object(settings.manager, 'restart_adapter', side_effect=restart) as started):
        with pytest.raises(fastapi.HTTPException) as error:
            asyncio.run(settings.save_settings(CID, settings.QQSettingsChange(mode='standard')))
    assert error.value.status_code == 409
    assert not history
    started.assert_not_called()


def test_channel_other_than_qq_is_rejected():
    other = dict(channel_data(), platform='feishu')
    with (mock.patch('api.qq_settings.auth.is_enabled', return_value=True),
          mock.patch('api.qq_settings.get_channel_config', return_value=other)):
        with pytest.raises(fastapi.HTTPException) as error:
            settings.get_settings(CID)
    assert error.value.status_code == 404
