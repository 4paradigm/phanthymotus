"""The decision_core switch uses canvas wiring, without duplicate source settings."""
import asyncio

import pytest
import auth
import config
import personal_memory as pm
from api import canvas, mcp_manage
import personal_memory_config as card_config
from test_personal_memory import store


def test_card_save_is_pending_until_next_project_start(store, monkeypatch):
    monkeypatch.setattr(auth, '_auth_enabled', True)
    monkeypatch.setattr(pm, '_runtime_settings', None)
    config.main['personal_memory'] = {'enabled': False}
    pm.pin_settings()
    result = asyncio.run(canvas.save_tool_config('agentcore', 'decision_core', {'personal_memory_enabled': True}))
    assert result['code'] == 200
    assert result['personal_memory_pending_start'] is True
    assert result['personal_memory_restart_required'] is False
    assert config.main['personal_memory']['enabled'] is True
    assert pm.enabled() is False
    pm.pin_settings()
    assert pm.enabled() is True
    assert card_config.status()['personal_memory_restart_required'] is False


def test_card_rejects_enabled_without_auth_and_does_not_save(store, monkeypatch):
    monkeypatch.setattr(auth, '_auth_enabled', False)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        asyncio.run(canvas.save_tool_config('agentcore', 'decision_core', {'personal_memory_enabled': True}))
    assert error.value.status_code == 400
    assert config.main.get('tool_config:agentcore:decision_core') is None


def test_card_get_shows_configuration_created_outside_card(store):
    result = asyncio.run(canvas.get_tool_config('agentcore', 'decision_core'))
    assert result['data']['personal_memory_enabled'] is True


def test_unrelated_card_config_preserves_personal_memory(store):
    previous = dict(config.main['personal_memory'])
    asyncio.run(mcp_manage._handle_agentcore_call(mcp_manage.MCPCallRequest(
        tool='decision_core', arguments={'action': 'config', 'vision_input': True})))
    assert config.main['personal_memory'] == previous


def layout():
    return {'cards': [
        {'id': 'core', 'mcpId': 'agentcore', 'toolName': 'decision_core'},
        {'id': 'asr', 'mcpId': 'perception', 'toolName': 'asr'},
        {'id': 'face', 'mcpId': 'perception', 'toolName': 'face_recognition'},
        {'id': 'remote', 'mcpId': 'agentcore', 'toolName': 'remote_message'},
        {'id': 'tts', 'mcpId': 'perception', 'toolName': 'tts'},
    ], 'connections': [
        {'fromCardId': 'asr', 'toCardId': 'core', 'resolvedTopic': '/real/mic/asr'},
        {'fromCardId': 'face', 'toCardId': 'core', 'resolvedTopic': '/real/image/face'},
        {'fromCardId': 'remote', 'toCardId': 'core', 'resolvedTopic': '/remote_control/message'},
    ], 'execConnections': [
        {'fromCardId': 'core', 'toMcpId': 'perception', 'toToolName': 'tts', 'toCardId': 'tts'},
    ]}


def registry():
    return {'perception': {'online': True,
        'schemas': {'mcp__perception__tts_speak': {}, 'mcp__perception__tts_interrupt': {}},
        'tool_groups': {'tts': ['mcp__perception__tts_speak', 'mcp__perception__tts_interrupt']},
        'split_map': {'mcp__perception__tts_speak': {'tool': 'tts', 'action': 'speak'},
                      'mcp__perception__tts_interrupt': {'tool': 'tts', 'action': 'interrupt'}}}}


def test_routes_come_only_from_real_inbound_and_executor_connections(store):
    routes = card_config.canvas_routes(layout(), registry())
    assert routes == {'voice_sources': ['dds:/real/mic/asr'],
                      'face_sources': ['dds:/real/image/face'],
                      'output_tools': ['mcp__perception__tts_speak', 'mcp__perception__tts_interrupt']}
    broken = layout()
    broken['connections'] = []
    with pytest.raises(ValueError, match='ASR'):
        card_config.canvas_routes(broken, registry())
    broken = layout()
    broken['execConnections'] = []
    with pytest.raises(ValueError, match='TTS'):
        card_config.canvas_routes(broken, registry())


def test_pending_mode_requires_auth_before_start(store, monkeypatch):
    monkeypatch.setattr(pm, '_runtime_settings', {'enabled': False})
    monkeypatch.setattr(auth, '_auth_enabled', False)
    assert 'ACCESS_TOKEN' in card_config.start_error()


def test_enabled_switch_cannot_change_during_running_project(store, monkeypatch):
    monkeypatch.setattr(auth, '_auth_enabled', True)
    config.main['core'] = {'project_running': True}
    with pytest.raises(ValueError, match='停止'):
        card_config.save_enabled(False)


from test_start_project_resolution import driver, R1_LAYOUT, CORE, _start


def test_real_project_start_derives_live_routes_from_wires(store, driver, monkeypatch):
    import copy
    import mcp_client
    monkeypatch.setattr(auth, '_auth_enabled', True)
    graph = copy.deepcopy(R1_LAYOUT)
    next(c for c in graph['cards'] if c['id'] == CORE)['mcpId'] = 'agentcore'
    graph['cards'].append({'id': 'tts', 'mcpId': 'mcp-perception', 'toolName': 'tts'})
    graph['execConnections'] = [{'fromCardId': CORE, 'toCardId': 'tts',
                                'toMcpId': 'mcp-perception', 'toToolName': 'tts'}]
    # An old browser snapshot must lose to the actual info() result at startup.
    for connection in graph['connections']:
        if connection['fromCardId'] == 'card-asr':
            connection['fromTopic'] = '/stale/asr'
    monkeypatch.setattr(mcp_client, 'registry', {'mcp-perception': {
        'online': True, 'schemas': {'mcp__mcp-perception__tts': {}}}})
    desired = config.main['personal_memory']
    desired['use_canvas_routes'] = True
    config.main['personal_memory'] = desired
    pm.pin_settings()
    assert _start(graph) is True
    assert pm.settings()['voice_sources'] == ['dds:/ubuntu/mic/audio/asr']
    assert pm.settings()['face_sources'] == []
    assert pm.settings()['output_tools'] == ['mcp__mcp-perception__tts']
    assert '/stale/asr' not in str(pm.settings())


def test_pending_mode_without_auth_refuses_start_before_any_device(store, driver, monkeypatch):
    monkeypatch.setattr(pm, '_runtime_settings', {'enabled': False})
    monkeypatch.setattr(auth, '_auth_enabled', False)
    assert _start(R1_LAYOUT) is False
    assert driver.starts == {}
    assert any('ACCESS_TOKEN' in e.get('payload', {}).get('message', '') for e in driver.events)
