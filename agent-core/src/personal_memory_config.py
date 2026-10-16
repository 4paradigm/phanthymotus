"""decision_core settings and automatic routes from the operator's canvas wiring."""
import auth
import config
import personal_memory as memory


def status() -> dict:
    desired = memory.configured_settings().get('enabled', False) is True
    return {'personal_memory_enabled': desired,
            'personal_memory_active': memory.enabled(),
            'personal_memory_restart_required': desired != memory.enabled()}


def save_enabled(value: bool) -> None:
    if type(value) is not bool:
        raise ValueError('个人记忆开关必须是布尔值。')
    if config.main.get('core', {}).get('project_running', False):
        raise ValueError('请先停止智能控制，再修改个人记忆设置。')
    if value and not auth.is_enabled():
        raise ValueError('请先配置 ACCESS_TOKEN 并重启 AgentCore，再开启个人记忆。')
    desired = memory.configured_settings()
    desired.update(enabled=value, use_canvas_routes=True)
    config.main['personal_memory'] = desired


def start_error() -> str:
    if status()['personal_memory_restart_required']:
        return '个人记忆模式已更改，请重启 AgentCore 后再开始智能控制。'
    return ''


def canvas_routes(layout: dict, registry: dict) -> dict:
    cards = {c['id']: c for c in layout.get('cards', [])}
    cores = {key for key, c in cards.items()
             if c.get('mcpId') == 'agentcore' and c.get('toolName') == 'decision_core'}
    routes = {'voice_sources': [], 'face_sources': [], 'output_tools': []}
    for connection in layout.get('connections', []):
        if connection.get('toCardId') not in cores:
            continue
        source = cards.get(connection.get('fromCardId'), {})
        if source.get('mcpId') == 'agentcore' or source.get('mcpId', '').startswith('peer:'):
            continue
        key = {'asr': 'voice_sources', 'face_recognition': 'face_sources'}.get(source.get('toolName'))
        topic = connection.get('resolvedTopic', '')
        if key and isinstance(topic, str) and topic.startswith('/'):
            value = f'dds:{topic}'
            if value not in routes[key]:
                routes[key].append(value)
    for connection in layout.get('execConnections', []):
        if connection.get('fromCardId') not in cores:
            continue
        target = cards.get(connection.get('toCardId'), {})
        mcp_id = connection.get('toMcpId', '')
        tool = connection.get('toToolName', '')
        if (target.get('toolName') != 'tts' or target.get('mcpId') != mcp_id
                or mcp_id.startswith('peer:')):
            continue
        entry = registry.get(mcp_id, {})
        if not entry.get('online'):
            continue
        full_name = f'mcp__{mcp_id}__{tool}'
        names = [full_name] if full_name in entry.get('schemas', {}) else entry.get('tool_groups', {}).get(tool, [])
        for name in names:
            split = entry.get('split_map', {}).get(name)
            if split and (split.get('tool') != 'tts' or split.get('action') not in ('speak', 'interrupt')):
                continue
            if name in entry.get('schemas', {}) and name not in routes['output_tools']:
                routes['output_tools'].append(name)
    if not routes['voice_sources']:
        raise ValueError('个人记忆需要将 ASR 输出连接到 decision_core 输入。')
    if not routes['output_tools']:
        raise ValueError('个人记忆需要从 decision_core 向本机 TTS 建立执行连线。')
    return routes


def apply_canvas_routes(layout: dict, registry: dict) -> None:
    desired = memory.configured_settings()
    if not memory.enabled() or not desired.get('use_canvas_routes', False):
        return
    routes = canvas_routes(layout, registry)
    desired.update(routes)
    config.main['personal_memory'] = desired
    memory.update_runtime_routes(routes)
