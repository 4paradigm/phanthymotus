"""Three distinct QQ AppIDs can share one Agent Core and one decision Core.

This is a LOCAL configuration/routing test, not real Tencent credentials or
simultaneous Gateway connections.
"""
from unittest import mock

from channel.manager import add_channel_config, get_channel_config, manager
from channel.qq_default_flow import ensure_default_flow
from channel.qq_ephemeral_store import put, get, revoke


def test_three_qq_bots_do_not_overwrite_channels_credentials_or_canvas():
    state = {}
    refs = []
    try:
        with mock.patch('config.main', state):
            for app in ('1900000001', '1900000002', '1900000003'):
                ref = put('secret_for_' + app)
                refs.append(ref)
                add_channel_config('qq_official_' + app, 'qq', {
                    'protocol': 'official',
                    'app_id': app,
                    'enabled_scopes': ['c2c', 'group'],
                    'private_only': False,
                    'app_secret_runtime': ref,
                }, enabled=True)
                result = ensure_default_flow('qq_official_' + app)
                assert result['ready'] is True
            assert len(state['channel_configs']) == 3
            assert len(state['canvas_layout']['cards']) == 7  # 1 core + 2 per bot
            assert len(state['canvas_layout']['connections']) == 3
            assert len(state['canvas_layout']['execConnections']) == 3
            assert manager.active_input_channels == {
                'qq_official_1900000001', 'qq_official_1900000002', 'qq_official_1900000003'}
            assert manager.active_output_channels == manager.active_input_channels
            assert get_channel_config('qq_official_1900000001')['config']['app_id'] == '1900000001'
            assert get_channel_config('qq_official_1900000002')['config']['app_id'] == '1900000002'
            assert get_channel_config('qq_official_1900000003')['config']['app_id'] == '1900000003'
            assert len({get(ref) for ref in refs}) == 3
            assert ensure_default_flow('qq_official_1900000001')['status'] == 'existing'
            assert ensure_default_flow('qq_official_1900000002')['status'] == 'existing'
            assert ensure_default_flow('qq_official_1900000003')['status'] == 'existing'
    finally:
        for ref in refs:
            revoke(ref)
        manager.sync_from_canvas()
