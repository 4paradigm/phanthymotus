"""Create a minimal QQ conversation Canvas only in an empty workspace.

Existing layouts are left untouched. This is intentionally idempotent and
does not start the project or replace an operator's custom graphs.
"""
import hashlib

import config
from channel.manager import channel_request_topic, manager


def ensure_default_flow(channel_id: str) -> dict:
    from channel.manager import get_channel_config
    channel = get_channel_config(channel_id)
    if channel is None or channel.get('platform') != 'qq':
        raise ValueError('QQ channel not found')
    layout = config.main.get('canvas_layout', {}) or {}
    cards = layout.get('cards', [])
    if cards:
        manager.sync_from_canvas()
        if (channel_id in manager.active_input_channels
                and channel_id in manager.active_output_channels):
            return {'status': 'existing', 'ready': True}
        return {'status': 'manual_required', 'ready': False}
    stem = hashlib.sha256(channel_id.encode()).hexdigest()[:8]
    inbound, controller, outbound = (f'qq-{stem}-in', f'qq-{stem}-core', f'qq-{stem}-out')
    topic = channel_request_topic(channel_id)
    layout = {
        'cards': [
            {'id': inbound, 'mcpId': 'channel', 'toolName': 'channel_request',
             'x': 120, 'y': 180, 'driverName': 'Channel',
             'topicIn': [], 'topicOut': [{'topic': topic, 'format': 'data/json'}]},
            {'id': controller, 'mcpId': 'agentcore', 'toolName': 'decision_core',
             'x': 430, 'y': 180, 'driverName': 'AgentCore',
             'topicIn': [{'format': 'data/json'}],
             'topicOut': [{'topic': '/decision_core', 'format': 'data/json'}]},
            {'id': outbound, 'mcpId': 'channel', 'toolName': 'channel_reply',
             'x': 750, 'y': 180, 'driverName': 'Channel', 'topicIn': [], 'topicOut': []},
        ],
        'connections': [
            {'id': f'qq-{stem}-conn', 'fromCardId': inbound, 'fromPortIdx': '0',
             'toCardId': controller, 'toPortIdx': '0', 'format': 'data/json',
             'fromTopic': topic},
        ],
        'execConnections': [],
    }
    config.main[f'tool_config:channel:channel_request:{inbound}'] = {'channel_id': channel_id}
    config.main[f'tool_config:channel:channel_reply:{outbound}'] = {'channel_id': channel_id}
    config.main['canvas_layout'] = layout
    manager.sync_from_canvas()
    return {'status': 'created', 'ready': True}
