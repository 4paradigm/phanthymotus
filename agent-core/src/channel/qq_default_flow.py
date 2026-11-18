"""Idempotently connect a QQ Channel to an empty or existing Canvas.

Never replace existing robot cards, paths, connections, or user tool settings.
Ambiguous existing decision cores require the operator to choose explicitly.
"""
import hashlib

import config
from channel.manager import channel_request_topic, manager


def ensure_default_flow(channel_id: str) -> dict:
    from channel.manager import get_channel_config

    channel = get_channel_config(channel_id)
    if channel is None or channel.get('platform') != 'qq':
        raise ValueError('QQ channel not found')

    layout = dict(config.main.get('canvas_layout', {}) or {})
    cards = list(layout.get('cards', []))
    created_empty = not cards
    cores = [card for card in cards if card.get('mcpId') == 'agentcore'
             and card.get('toolName') == 'decision_core']
    if not cores and created_empty:
        stem = hashlib.sha256(channel_id.encode()).hexdigest()[:8]
        core = {
            'id': f'qq-{stem}-core', 'mcpId': 'agentcore',
            'toolName': 'decision_core', 'x': 430, 'y': 180,
            'driverName': 'AgentCore',
            'topicIn': [{'format': 'data/json'}],
            'topicOut': [{'topic': '/decision_core', 'format': 'data/json'}],
        }
        cards.append(core)
        cores = [core]
    if len(cores) != 1:
        return {'status': 'manual_required', 'ready': False,
                'reason': 'Canvas requires exactly one decision_core; existing cards were preserved'}

    core = cores[0]
    stem = hashlib.sha256(channel_id.encode()).hexdigest()[:8]
    existing_ids = {card.get('id') for card in cards}
    instance_keys = ('channel', 'agentcore')

    def find_card(tool_name: str):
        for card in cards:
            if card.get('toolName') != tool_name or card.get('mcpId') not in instance_keys:
                continue
            key = f"tool_config:{card['mcpId']}:{tool_name}:{card['id']}"
            instance = config.main.get(key, {}) or {}
            if instance.get('channel_id') == channel_id:
                return card
        return None

    def new_id(suffix: str) -> str:
        base = f'qq-{stem}-{suffix}'
        candidate = base
        index = 2
        while candidate in existing_ids:
            candidate = f'{base}-{index}'
            index += 1
        existing_ids.add(candidate)
        return candidate

    changes = False
    inbound = find_card('channel_request')
    if inbound is None:
        inbound = {
            'id': new_id('in'), 'mcpId': 'channel', 'toolName': 'channel_request',
            'x': 120, 'y': 180 if created_empty else 540,
            'driverName': 'Channel',
            'topicIn': [], 'topicOut': [
                {'topic': channel_request_topic(channel_id), 'format': 'data/json'}],
        }
        cards.append(inbound)
        config.main[f"tool_config:channel:channel_request:{inbound['id']}"] = {
            'channel_id': channel_id}
        changes = True

    outbound = find_card('channel_reply')
    if outbound is None:
        outbound = {
            'id': new_id('out'), 'mcpId': 'channel', 'toolName': 'channel_reply',
            'x': 750, 'y': 180 if created_empty else 540,
            'driverName': 'Channel', 'topicIn': [], 'topicOut': [],
        }
        cards.append(outbound)
        config.main[f"tool_config:channel:channel_reply:{outbound['id']}"] = {
            'channel_id': channel_id}
        changes = True

    connections = list(layout.get('connections', []))
    if not any(link.get('fromCardId') == inbound['id']
               and link.get('toCardId') == core['id'] for link in connections):
        connections.append({
            'id': new_id('conn'), 'fromCardId': inbound['id'],
            'fromPortIdx': '0', 'toCardId': core['id'], 'toPortIdx': '0',
            'format': 'data/json', 'fromTopic': channel_request_topic(channel_id),
        })
        changes = True

    executions = list(layout.get('execConnections', []))
    if not any(link.get('fromCardId') == core['id']
               and link.get('toCardId') == outbound['id'] for link in executions):
        executions.append({
            'id': new_id('exec-reply'), 'fromCardId': core['id'],
            'toCardId': outbound['id'], 'toMcpId': 'channel',
            'toToolName': 'channel_reply',
        })
        changes = True

    if changes:
        layout.update(cards=cards, connections=connections, execConnections=executions)
        config.main['canvas_layout'] = layout
    manager.sync_from_canvas()
    return {'status': ('created' if created_empty else 'extended') if changes else 'existing',
            'ready': True}
