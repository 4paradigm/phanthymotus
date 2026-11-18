"""Install QQ cards on an unambiguous Canvas and observe its live input."""

from uuid import uuid4

import config
from channel.manager import channel_request_topic


def reply_card(channel_id, layout=None):
    layout = config.main.get('canvas_layout', {}) if layout is None else layout
    cores = {c['id'] for c in layout.get('cards', [])
             if c.get('mcpId') == 'agentcore' and c.get('toolName') == 'decision_core'}
    inputs = _selected_cards(layout, channel_id, 'channel_request')
    receiving_cores = {link.get('toCardId') for link in layout.get('connections', [])
                       if link.get('fromCardId') in inputs}
    outputs = _selected_cards(layout, channel_id, 'channel_reply')
    for link in layout.get('execConnections', []):
        if (link.get('fromCardId') in cores & receiving_cores and link.get('toMcpId') == 'channel'
                and link.get('toToolName') == 'channel_reply'):
            card_id = link.get('toCardId', '')
            if card_id in outputs:
                return card_id
    return ''


def _selected_cards(layout, channel_id, tool):
    return {c['id'] for c in layout.get('cards', [])
            if c.get('mcpId') == 'channel' and c.get('toolName') == tool
            and (config.main.get(f"tool_config:channel:{tool}:{c['id']}") or {})
            .get('channel_id') == channel_id}


def wired(channel_id, layout=None):
    return bool(reply_card(channel_id, layout))


def wire(channel_id, *, notify=True):
    from api.canvas import current_editor, notify_layout_changed
    from api import config as project_api

    layout = config.main.get('canvas_layout', {})
    if wired(channel_id, layout):
        return
    task = project_api._start_project_task
    if (current_editor() or (task is not None and not task.done())
            or config.main.get('core', {}).get('project_running')):
        return
    cards = list(layout.get('cards', []))
    cores = [c for c in cards if c.get('mcpId') == 'agentcore' and c.get('toolName') == 'decision_core']
    if len(cores) > 1 or (cards and not cores):
        return

    def add_card(mcp_id, tool, x, y, **ports):
        card_id = uuid4().hex
        cards.append(dict(id=card_id, mcpId=mcp_id, toolName=tool,
                          x=x, y=y, topicIn=[], topicOut=[]) | ports)
        return card_id

    core = cores[0]['id'] if cores else add_card(
        'agentcore', 'decision_core', 420, 180,
        topicIn=[{'topic': '', 'format': 'data/json'}],
        topicOut=[{'topic': '/decision_core', 'format': 'data/json'}],
    )
    y = max((c.get('y') or 0 for c in cards), default=0) + 240 if cores else 180
    topic = channel_request_topic(channel_id)
    request = next(iter(_selected_cards(layout, channel_id, 'channel_request')), '')
    reply = next(iter(_selected_cards(layout, channel_id, 'channel_reply')), '')
    if not request:
        request = add_card('channel', 'channel_request', 110, y,
                           topicOut=[{'topic': topic, 'format': 'data/json'}])
    if not reply:
        reply = add_card('channel', 'channel_reply', 760, y)
    config.main[f'tool_config:channel:channel_request:{request}'] = {'channel_id': channel_id}
    config.main[f'tool_config:channel:channel_reply:{reply}'] = {'channel_id': channel_id}

    links = list(layout.get('connections', []))
    if not any(c.get('fromCardId') == request and c.get('toCardId') == core for c in links):
        links.append(dict(id=uuid4().hex, fromCardId=request, toCardId=core,
                          fromPortIdx='0', toPortIdx='0', format='data/json', fromTopic=topic))
    executions = list(layout.get('execConnections', []))
    if not any(c.get('fromCardId') == core and c.get('toCardId') == reply for c in executions):
        executions.append(dict(id=uuid4().hex, fromCardId=core, toCardId=reply,
                               toMcpId='channel', toToolName='channel_reply'))
    config.main['canvas_layout'] = dict(layout, cards=cards, connections=links, execConnections=executions)
    if notify:
        notify_layout_changed()
    return True


def input_ready(channel_id):
    from collector import project_running
    import ros2_bridge
    import topic_subscriber

    topic = channel_request_topic(channel_id)
    if not project_running() or not ros2_bridge._running or not wired(channel_id):
        return False
    with topic_subscriber._lock:
        return topic in topic_subscriber._subscriptions
