"""Resolve a SLAM preflight request from live canvas sources without start.

A project may be stopped, or its SLAM tool may reject strict input evidence.
Preflight must therefore query each connected producer's existing info() once
rather than depend on previously successful project startup.
"""
from .slam_metadata import collect_slam_inputs, SlamRoutingError


class PreflightRoutingError(ValueError):
    """SLAM preflight cannot identify current connected source claims."""


async def resolve_canvas_preflight(layout, mcp_id, tool_name, instance_id, info_of_card):
    """Build strict start-shaped input arguments without mutating the project.

    info_of_card(card) is an asynchronous, bounded read-only info() callback.
    No stale saved topic/format is considered proof of a running output.
    """
    cards = [c for c in (layout or {}).get('cards', []) if isinstance(c, dict)]
    matches = [c for c in cards if c.get('mcpId') == mcp_id
               and c.get('toolName') == tool_name
               and (not instance_id or c.get('id') == instance_id)]
    if len(matches) != 1:
        raise PreflightRoutingError(
            'preflight requires one unambiguous connected SLAM canvas card')
    target = matches[0]
    inbound = [wire for wire in (layout or {}).get('connections', [])
               if isinstance(wire, dict) and wire.get('toCardId') == target.get('id')]
    if len(inbound) != 2:
        raise PreflightRoutingError('preflight needs exactly two SLAM input wires')
    by_id = {c.get('id'): c for c in cards}
    live = {}
    for wire in inbound:
        cid = wire.get('fromCardId')
        card = by_id.get(cid)
        if not card:
            raise PreflightRoutingError('SLAM source card no longer exists: ' + str(cid))
        if cid not in live:
            data = await info_of_card(card)
            if not isinstance(data, dict):
                raise PreflightRoutingError('SLAM source info() did not return an object: ' + str(cid))
            live[cid] = data

    def connected_topic(wire):
        cid = wire.get('fromCardId')
        ports = live[cid].get('topic_out') or []
        try:
            index = int(wire.get('fromPortIdx', 0) or 0)
            if not isinstance(ports, list) or not 0 <= index < len(ports):
                return ''
            port = ports[index]
            return port.get('topic', '') if isinstance(port, dict) else ''
        except (ValueError, TypeError):
            return ''

    routed = collect_slam_inputs(inbound, live, connected_topic)
    if set(routed['input_bindings'].values()) != {'state/odom', 'sensor/pointcloud2'}:
        raise PreflightRoutingError('two connected inputs must be odometry and native PointCloud2')
    return {'input_topics': list(routed['input_bindings']), **routed}
