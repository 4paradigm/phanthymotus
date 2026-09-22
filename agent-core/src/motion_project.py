"""Capability-based Canvas bindings for teleop -> motion control -> actuator.

This module does not contact hardware or grant a lease. Feedback edges are
derived from registered ports, never a user-supplied exemption from the DAG.
"""
from __future__ import annotations

import copy
import ipaddress
import re
import uuid
from urllib.parse import urlsplit

from teleop_project import TeleopProjectError, _port, _registered

FIELDS = ('protocol_version', 'robot_profile', 'namespace', 'command_topic', 'feedback_topic')


def _local_url(service):
    url = service.get('url', '')
    try:
        p = urlsplit(url)
        good = (service.get('transport', 'http') == 'http' and p.scheme == 'http'
                and (p.hostname == 'localhost' or ipaddress.ip_address(p.hostname).is_loopback)
                and p.path == '/mcp' and p.username is None and p.password is None
                and not (p.query or p.fragment)
                and (p.port is None or 0 < p.port < 65536))
    except (ValueError, TypeError):
        good = False
    if not good:
        raise TeleopProjectError('运动控制和执行卡必须来自已注册的同机 Driver')
    if p.hostname == 'localhost':
        # Driver registration uses localhost. Bind a literal loopback endpoint
        # so the resulting control session never depends on name resolution.
        return p._replace(netloc='127.0.0.1' + (f':{p.port}' if p.port is not None else '')).geturl()
    return url


def _metadata(tool, key):
    meta = tool.get(key)
    if not isinstance(meta, dict) or type(meta.get('protocol_version')) is not int or meta['protocol_version'] != 2:
        raise TeleopProjectError('连线目标未声明 motus.control/2 能力')
    if (not isinstance(meta.get('namespace'), str)
            or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', meta['namespace'])
            or not isinstance(meta.get('robot_profile'), str) or not meta['robot_profile']):
        raise TeleopProjectError('运动控制的机型或命名空间无效')
    for field in ('command_topic', 'feedback_topic'):
        topic = meta.get(field)
        if not isinstance(topic, str) or not re.fullmatch(r'/[A-Za-z0-9_/]+', topic) or '//' in topic:
            raise TeleopProjectError('运动控制的话题声明无效')
    return meta


def _only_target(layout, edge):
    cards = [c for c in layout.get('cards', []) if c.get('id') == edge.get('toCardId')]
    if len(cards) != 1:
        raise TeleopProjectError('控制连线目标不存在或不唯一')
    incoming = [c for c in layout.get('connections', [])
                if c.get('toCardId') == cards[0]['id']
                and str(c.get('toPortIdx')) == str(edge.get('toPortIdx'))
                and c.get('role') != 'feedback']
    if len(incoming) != 1:
        raise TeleopProjectError('会话控制端口必须只有一个命令来源')
    return cards[0]


def _has_port(tool, direction, fmt, topic):
    return any(p.get('format') == fmt and p.get('topic') == topic
               for p in tool.get(direction, []) if isinstance(p, dict))


def resolve_motion_binding(layout, registry, source, source_tool, edge):
    _port(source_tool, 'topic_out', edge.get('fromPortIdx'), 'control/eef')
    motion = _only_target(layout, edge)
    service, tool = _registered(registry, motion)
    meta = _metadata(tool, 'x-teleop-target')
    capability = tool.get('x-motion-control') or {}
    if (type(capability.get('protocol_version')) is not int or capability.get('protocol_version') != 2
            or any(capability.get(k) != meta[k] for k in ('namespace', 'robot_profile', 'feedback_topic'))):
        raise TeleopProjectError('中间卡缺少匹配的机器人运动控制能力')
    port = _port(tool, 'topic_in', edge.get('toPortIdx'), 'control/eef')
    if port.get('topic') != meta['command_topic'] or not _has_port(tool, 'topic_out', 'data/json', meta['feedback_topic']):
        raise TeleopProjectError('运动控制命令或反馈端口与能力声明不一致')
    downstream = [c for c in layout.get('connections', [])
                  if c.get('fromCardId') == motion['id'] and c.get('format') == 'control/joint'
                  and c.get('role') != 'feedback']
    if len(downstream) != 1:
        raise TeleopProjectError('本轮双臂运动控制必须连接一个完整双臂执行卡')
    execution_edge = downstream[0]
    actuator = _only_target(layout, execution_edge)
    execution_service, execution_tool = _registered(registry, actuator)
    execution = _metadata(execution_tool, 'x-control-target')
    if (actuator.get('mcpId') != motion.get('mcpId')
            or actuator.get('toolName') != capability.get('execution_tool')
            or any(execution[k] != meta[k] for k in ('robot_profile', 'namespace', 'feedback_topic'))
            or execution['command_topic'] != capability.get('execution_command_topic')
            or set(execution.get('resources', [])) != {'arm_l', 'arm_r'}):
        raise TeleopProjectError('运动控制与双臂执行卡不属于同一 Driver 或资源不匹配')
    output = _port(tool, 'topic_out', execution_edge.get('fromPortIdx'), 'control/joint')
    input_port = _port(execution_tool, 'topic_in', execution_edge.get('toPortIdx'), 'control/joint')
    if output.get('topic') != execution['command_topic'] or input_port.get('topic') != execution['command_topic']:
        raise TeleopProjectError('运动控制输出和执行卡输入的话题不一致')
    execution_binding = {k: copy.deepcopy(execution[k]) for k in (*FIELDS, 'resources')}
    execution_binding.update(mcp_id=actuator['mcpId'], tool=actuator['toolName'], url=_local_url(execution_service))
    result = {k: meta[k] for k in FIELDS}
    result.update(mcp_id=motion['mcpId'], tool=motion['toolName'], url=_local_url(service),
                  execution_binding=execution_binding)
    return result


def validate_motion_target(binding, info):
    meta = _metadata(info, 'x-teleop-target')
    cap = info.get('x-motion-control') or {}
    execution = binding['execution_binding']
    if (any(meta[k] != binding[k] for k in FIELDS)
            or type(cap.get('protocol_version')) is not int or cap.get('protocol_version') != 2
            or cap.get('execution_tool') != execution['tool']
            or cap.get('execution_command_topic') != execution['command_topic']
            or any(cap.get(k) != binding[k] for k in ('namespace', 'robot_profile', 'feedback_topic'))
            or not _has_port(info, 'topic_in', 'control/eef', binding['command_topic'])
            or not _has_port(info, 'topic_out', 'control/joint', execution['command_topic'])
            or not _has_port(info, 'topic_out', 'data/json', binding['feedback_topic'])):
        raise TeleopProjectError('Driver 当前运动控制接口与已保存连线不一致')


def validate_execution_target(binding, info):
    meta = _metadata(info, 'x-control-target')
    if (any(meta[k] != binding[k] for k in FIELDS)
            or set(meta.get('resources', [])) != set(binding['resources'])
            or not _has_port(info, 'topic_in', 'control/joint', binding['command_topic'])):
        raise TeleopProjectError('Driver 当前执行卡接口与已保存连线不一致')


def with_feedback_edges(layout, registry):
    """Return canonical display edges even while the forward graph is incomplete."""
    result = copy.deepcopy(layout)
    connections = [c for c in result.get('connections', []) if c.get('role') != 'feedback']
    for edge in list(connections):
        if edge.get('format') != 'control/eef':
            continue
        if sum(e.get('fromCardId') == edge.get('fromCardId') and e.get('format') == 'control/eef'
               for e in connections) != 1:
            continue
        try:
            source = next(c for c in result.get('cards', []) if c.get('id') == edge.get('fromCardId') and c.get('toolName') == 'teleop')
            target = next(c for c in result.get('cards', []) if c.get('id') == edge.get('toCardId'))
            _, source_tool = _registered(registry, source)
            _, target_tool = _registered(registry, target)
            _port(source_tool, 'topic_out', edge.get('fromPortIdx'), 'control/eef')
            target_port = _port(target_tool, 'topic_in', edge.get('toPortIdx'), 'control/eef')
            meta = _metadata(target_tool, 'x-teleop-target')
            if target_port.get('topic') != meta['command_topic'] or not target_tool.get('x-motion-control'):
                continue
            outgoing = next(i for i, p in enumerate(target_tool.get('topic_out', []))
                            if p.get('topic') == meta['feedback_topic'] and p.get('format') == 'data/json')
            incoming = next(i for i, p in enumerate(source_tool.get('topic_in', []))
                            if p.get('role') == 'feedback' and p.get('format') == 'data/json')
        except (StopIteration, TeleopProjectError, TypeError, KeyError):
            continue
        source['topicOut'] = copy.deepcopy(source_tool.get('topic_out', []))
        source['topicOut'][int(edge['fromPortIdx'])]['topic'] = meta['command_topic']
        source['topicIn'] = copy.deepcopy(source_tool.get('topic_in', []))
        source['topicIn'][incoming]['topic'] = meta['feedback_topic']
        connections.append({'id': f'feedback-{target["id"]}-{source["id"]}', 'role': 'feedback',
                            'automatic': True, 'fromCardId': target['id'], 'toCardId': source['id'],
                            'fromPortIdx': str(outgoing), 'toPortIdx': str(incoming),
                            'format': 'data/json', 'fromTopic': meta['feedback_topic']})
    result['connections'] = connections
    return result


def motion_targets(registry):
    """Only registered motion cards; a template never guesses a robot address."""
    targets = []
    for service in registry:
        for tool in service.get('tools', []):
            if not isinstance(tool, dict) or not tool.get('x-motion-control'):
                continue
            try:
                meta = _metadata(tool, 'x-teleop-target')
                _local_url(service)
            except TeleopProjectError:
                continue
            targets.append({'mcp_id': service['id'], 'tool': tool['name'],
                            'label': service.get('name') or service['id'],
                            'robot_profile': meta['robot_profile']})
    return targets


def build_motion_template(layout, registry, source_id, driver_id):
    """Prepare a reviewable graph; the existing editor/save endpoint applies it."""
    result = copy.deepcopy(layout)
    sources = [c for c in result.get('cards', []) if c.get('id') == source_id and c.get('toolName') == 'teleop']
    candidates = [x for x in motion_targets(registry) if x['mcp_id'] == driver_id]
    if len(sources) != 1 or len(candidates) != 1:
        raise TeleopProjectError('请选择一张遥操卡和一个已注册的运动控制 Driver')
    source, candidate = sources[0], candidates[0]
    _, source_tool = _registered(registry, source)
    motion_ref = {'mcpId': driver_id, 'toolName': candidate['tool']}
    _, motion_tool = _registered(registry, motion_ref)
    arm_ref = {'mcpId': driver_id, 'toolName': motion_tool['x-motion-control']['execution_tool']}
    _, arm_tool = _registered(registry, arm_ref)

    def card(ref, tool, offset):
        matches = [c for c in result['cards'] if c.get('mcpId') == driver_id and c.get('toolName') == ref['toolName']]
        if len(matches) > 1:
            raise TeleopProjectError('同一执行能力存在多张卡片，请先明确连接目标')
        if matches:
            return matches[0]
        value = {**ref, 'id': 'motion-' + uuid.uuid4().hex[:12],
                 'x': source.get('x', 0) + offset, 'y': source.get('y', 0),
                 'topicIn': copy.deepcopy(tool.get('topic_in', [])),
                 'topicOut': copy.deepcopy(tool.get('topic_out', []))}
        result['cards'].append(value)
        return value

    motion, arm = card(motion_ref, motion_tool, 460), card(arm_ref, arm_tool, 920)
    for src, dst, src_tool, dst_tool, fmt in ((source, motion, source_tool, motion_tool, 'control/eef'),
                                            (motion, arm, motion_tool, arm_tool, 'control/joint')):
        out = [i for i, p in enumerate(src_tool.get('topic_out', [])) if p.get('format') == fmt]
        incoming = [i for i, p in enumerate(dst_tool.get('topic_in', [])) if p.get('format') == fmt]
        if len(out) != 1 or len(incoming) != 1:
            raise TeleopProjectError('模板要求唯一且兼容的控制端口')
        edges = result.setdefault('connections', [])
        existing = [e for e in edges if e.get('fromCardId') == src['id'] and e.get('format') == fmt]
        if existing:
            if len(existing) != 1 or existing[0].get('toCardId') != dst['id']:
                raise TeleopProjectError('已有命令连线，请先移除或核对；模板不会覆盖它')
            continue
        edges.append({'id': 'conn-' + uuid.uuid4().hex[:12], 'fromCardId': src['id'], 'toCardId': dst['id'],
                      'fromPortIdx': str(out[0]), 'toPortIdx': str(incoming[0]), 'format': fmt})
    from teleop_project import resolve_bindings
    resolve_bindings(result, registry)
    return with_feedback_edges(result, registry)
