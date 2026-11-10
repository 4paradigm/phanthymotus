"""Route declared SLAM inputs by live source port, never by tool name.

This module is deliberately ROS-free. It selects only upstream info() claims
corresponding to actual canvas wires; Perception validates physical facts.
"""
from copy import deepcopy


POSE = 'state/odom'
CLOUD = 'sensor/pointcloud2'
SUPPORTED = {POSE, CLOUD}


class SlamRoutingError(ValueError):
    """A canvas wire cannot be matched to one unambiguous source port."""


def _port(info, index):
    ports = info.get('topic_out') if isinstance(info, dict) else None
    if not isinstance(ports, list) or index >= len(ports):
        return {}
    return ports[index] if isinstance(ports[index], dict) else {}


def _declaration(info, topic):
    declarations = info.get('input_metadata')
    if isinstance(declarations, dict):
        item = declarations.get(topic)
        return item if isinstance(item, dict) else None
    if isinstance(declarations, list):
        found = [item for item in declarations
                 if isinstance(item, dict) and item.get('topic') == topic]
        if len(found) > 1:
            raise SlamRoutingError(f'{topic}: duplicate input_metadata declarations')
        return found[0] if found else None
    return None


def collect_slam_inputs(connections, resolved_info, topic_of_connection):
    """Select concrete pose/cloud ports and associated producer evidence.

    connections are only the target consumer's inbound edges. A saved connection
    format is never evidence of current type: a live info() port takes precedence.
    Without live info, only the wiring hint is forwarded; missing producer
    metadata will be rejected by the strict SLAM consumer.
    """
    bindings, interfaces, metadata = {}, {}, {}
    for conn in connections:
        src_id = conn.get('fromCardId')
        try:
            index = int(conn.get('fromPortIdx', 0) or 0)
        except (TypeError, ValueError) as error:
            raise SlamRoutingError(f'{src_id}: invalid fromPortIdx') from error
        if index < 0:
            raise SlamRoutingError(f'{src_id}: negative fromPortIdx')
        info = resolved_info.get(src_id) or {}
        port = _port(info, index)
        topic = topic_of_connection(conn)
        if not topic:
            raise SlamRoutingError(f'{src_id}[{index}]: unresolved source topic')
        if port.get('topic') and port['topic'] != topic:
            raise SlamRoutingError(f'{topic}: live source port topic mismatch')
        fmt = port.get('format') or conn.get('format')
        if fmt not in SUPPORTED:
            raise SlamRoutingError(f'{topic}: expected {POSE} or {CLOUD}, got {fmt!r}')
        if port.get('format') and conn.get('format') and conn['format'] != fmt:
            raise SlamRoutingError(f'{topic}: saved wire format conflicts with live port ({fmt})')
        if topic in bindings:
            raise SlamRoutingError(f'{topic}: duplicate inbound SLAM topic')
        bindings[topic] = fmt

        claim = _declaration(info, topic)
        if claim is not None:
            if claim.get('topic') != topic or claim.get('format') != fmt:
                raise SlamRoutingError(f'{topic}: input_metadata topic/format mismatch')
            metadata[topic] = deepcopy(claim)

        if fmt == POSE:
            by_topic = info.get('odom_interfaces')
            interface = by_topic.get(topic) if isinstance(by_topic, dict) else None
            if interface is None:
                ports = info.get('topic_out') or []
                count = sum(isinstance(p, dict) and p.get('format') == POSE
                            for p in ports)
                if count == 1:
                    interface = info.get('odom_interface')
            if isinstance(interface, dict):
                interfaces[topic] = deepcopy(interface)
    return {'input_bindings': bindings, 'odom_interfaces': interfaces,
            'input_metadata': metadata}


def collect_connected_metadata(connections, resolved_info, topic_of_connection):
    """Select producer source declarations for generic metadata-aware cards.

    Unlike the SLAM route, older display consumers are allowed to have no
    declaration. Never use a persisted canvas snapshot as physical evidence.
    """
    out = {}
    for conn in connections:
        src_id = conn.get('fromCardId')
        info = resolved_info.get(src_id) or {}
        try:
            index = int(conn.get('fromPortIdx', 0) or 0)
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        port = _port(info, index)
        topic = topic_of_connection(conn)
        if not topic or port.get('topic') != topic or not port.get('format'):
            continue
        claim = _declaration(info, topic)
        if claim is None:
            continue
        if claim.get('topic') != topic or claim.get('format') != port.get('format'):
            raise SlamRoutingError(f'{topic}: producer metadata does not match connected port')
        if topic in out:
            raise SlamRoutingError(f'{topic}: duplicate connected source metadata')
        out[topic] = deepcopy(claim)
    return out
