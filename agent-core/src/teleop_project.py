"""Canvas teleoperation bindings, with no network or robot side effects."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit


FORMAT = 'control/teleop'


class TeleopProjectError(ValueError):
    pass


def teleop_cards(layout):
    return [c for c in (layout or {}).get('cards', []) if c.get('toolName') == 'teleop']


def _registered(registry, card):
    matches = [m for m in registry if m.get('id') == card.get('mcpId')]
    if len(matches) != 1:
        raise TeleopProjectError('遥操连线的服务未注册或身份不唯一')
    service = matches[0]
    matches = [t for t in service.get('tools', [])
               if isinstance(t, dict) and t.get('name') == card.get('toolName')]
    if len(matches) != 1:
        raise TeleopProjectError('遥操连线的工具未注册或身份不唯一')
    return service, matches[0]


def _port(tool, direction, index):
    try:
        if isinstance(index, bool):
            raise ValueError()
        i = int(index)
        ports = tool.get(direction) or []
        if i < 0 or str(i) != str(index):
            raise ValueError()
        port = ports[i]
        if port.get('format') != FORMAT:
            raise ValueError()
        return port
    except (ValueError, TypeError, IndexError, AttributeError):
        raise TeleopProjectError('遥操必须连接 control/teleop 命令端口') from None


def resolve_bindings(layout, registry):
    """Resolve only saved graph identities and registered Driver declarations.

    Cached canvas topic paths and user-supplied URLs are never binding evidence.
    Live profile/configuration is checked separately before arming the card.
    """
    sources = teleop_cards(layout)
    if len(sources) > 1:
        raise TeleopProjectError('当前只支持一个遥操卡片实例')
    cards = layout.get('cards') or []
    connections = layout.get('connections') or []
    result = {}
    for source in sources:
        if not source.get('id'):
            raise TeleopProjectError('遥操卡片缺少实例身份')
        _, source_tool = _registered(registry, source)
        actions = source_tool.get('inputSchema', {}).get('properties', {}).get('action', {}).get('enum', [])
        if not {'project_start', 'project_stop'} <= set(actions):
            raise TeleopProjectError('当前遥操机型未提供智能控制启停与收臂能力')
        outgoing = [c for c in connections if c.get('fromCardId') == source['id']
                    and c.get('format') == FORMAT]
        if len(outgoing) != 1:
            raise TeleopProjectError('遥操命令端口必须且只能连接一个对应 Driver 的 teleop_executor')
        edge = outgoing[0]
        _port(source_tool, 'topic_out', edge.get('fromPortIdx'))
        targets = [c for c in cards if c.get('id') == edge.get('toCardId')]
        if len(targets) != 1 or targets[0].get('toolName') != 'teleop_executor':
            raise TeleopProjectError('遥操命令只能连接 Driver 的 teleop_executor 卡片')
        target = targets[0]
        incoming = [c for c in connections if c.get('toCardId') == target['id']
                    and str(c.get('toPortIdx')) == str(edge.get('toPortIdx'))]
        if len(incoming) != 1:
            raise TeleopProjectError('teleop_executor 命令端口不能同时连接其他输入')
        service, tool = _registered(registry, target)
        port = _port(tool, 'topic_in', edge.get('toPortIdx'))
        meta = tool.get('x-teleop-target')
        if not isinstance(meta, dict) or type(meta.get('protocol_version')) is not int or meta['protocol_version'] != 1:
            raise TeleopProjectError('Driver 未声明兼容的遥操协议，请更新对应 Driver')
        ns = meta.get('namespace')
        profile = meta.get('robot_profile')
        if (not isinstance(ns, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', ns)
                or profile not in ('tianyi2', 'g1_23')):
            raise TeleopProjectError('Driver 遥操机型或命名空间无效')
        command, feedback = f'/{ns}/motion/teleop/command', f'/{ns}/motion/teleop/feedback'
        if meta.get('command_topic') != command or meta.get('feedback_topic') != feedback or port.get('topic') != command:
            raise TeleopProjectError('Driver 遥操 topic 声明与命名空间不一致')
        if not any(p.get('topic') == feedback for p in tool.get('topic_out', [])):
            raise TeleopProjectError('Driver 未声明遥操执行反馈 topic')
        url = service.get('url', '')
        try:
            parsed = urlsplit(url)
            valid = (service.get('transport', 'http') == 'http' and parsed.scheme == 'http'
                     and not parsed.username and not parsed.password and parsed.path == '/mcp'
                     and not parsed.query and not parsed.fragment
                     and ipaddress.ip_address(parsed.hostname).is_loopback
                     and (parsed.port is None or 0 < parsed.port <= 65535))
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise TeleopProjectError('遥操 Driver 必须注册在本机回环 HTTP MCP 地址')
        result[source['id']] = {
            'mcp_id': target['mcpId'], 'tool': 'teleop_executor', 'url': url,
            'namespace': ns, 'command_topic': command, 'feedback_topic': feedback,
            'robot_profile': profile, 'protocol_version': 1,
        }
    return result


def validate_profile(binding, info):
    profile = (info.get('configuration') or {}).get('robot_profile') or info.get('robot_profile')
    if profile != binding['robot_profile']:
        raise TeleopProjectError('遥操机型与连线 Driver 不一致，请先配置对应机型')


def validate_target(binding, info):
    """Require the current Driver response to agree with registered metadata."""
    meta = info.get('x-teleop-target')
    keys = ('protocol_version', 'robot_profile', 'namespace', 'command_topic', 'feedback_topic')
    if (not isinstance(meta, dict) or type(meta.get('protocol_version')) is not int
            or any(meta.get(key) != binding[key] for key in keys)
            or not any(p.get('format') == FORMAT and p.get('topic') == binding['command_topic']
                       for p in info.get('topic_in', []) if isinstance(p, dict))
            or not any(p.get('topic') == binding['feedback_topic']
                       for p in info.get('topic_out', []) if isinstance(p, dict))):
        raise TeleopProjectError('Driver 当前遥操接口与已注册连线不一致，请刷新服务并检查机型和话题')


def has_project_lifecycle(registry, card):
    _, tool = _registered(registry, card)
    actions = tool.get('inputSchema', {}).get('properties', {}).get('action', {}).get('enum', [])
    return {'project_start', 'project_stop'} <= set(actions)


def shutdown_complete(payload):
    """A receipt/HTTP 200 is not a completed arm return or authority release."""
    return (isinstance(payload, dict) and not payload.get('error')
            and payload.get('state') not in ('accepted', 'error', 'fault', 'returning')
            and payload.get('armed') is False
            and payload.get('authority_released') is True
            and (payload.get('return_completed') is True or payload.get('return_required') is False))


def require_editable(core, *layouts):
    teleop_running = core.get('project_running') and any(teleop_cards(layout) for layout in layouts)
    if teleop_running or core.get('project_phase') in ('stopping', 'stop_failed'):
        raise TeleopProjectError('请先停止智能控制并等待收臂及释放控制权完成，再修改画布')
