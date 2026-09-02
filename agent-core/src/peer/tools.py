"""
peer/tools.py — tool proxy logic: filter and call MCP tools on behalf of peers.

A peer's tool access is governed by:
  * `role` — the coarse permission class (viewer vs operator)
  * `tool_filter` — a glob pattern or comma-separated list restricting which tools

`viewer` gets read-only sensors/state; `operator` gets actuators. The filter then
narrows further: `camera_*,status` lets an operator call camera tools and status,
but not `move` or `grasp`.

This is the trust boundary between robots. A peer calling a tool here is identical
to a human calling it through a channel — same ACL, same actuator double gate,
same audit trail.
"""

import fnmatch

import mcp_client
from peer import store


def filter_schemas(peer_id: str, all_schemas: list[dict]) -> list[dict]:
    """Return the subset of tools this peer may call.

    Enforces role (viewer ≠ operator) and tool_filter glob. Returns OpenAI
    function-calling schemas, so a peer can present them to its own LLM.
    """
    peer = store.get(peer_id)
    if peer is None or peer['role'] == 'blocked':
        return []

    role = peer['role']
    tool_filter = peer.get('tool_filter', '*')
    patterns = [p.strip() for p in tool_filter.split(',') if p.strip()]
    if not patterns:
        patterns = ['*']

    allowed = []
    for schema in all_schemas:
        name = schema.get('name', '')
        if not name:
            continue

        # Role enforcement: viewers get sensors/queries, operators get all
        # This is a policy decision; adjust if the split should be elsewhere.
        # For now: if the tool name contains 'move', 'grasp', 'speak', 'write',
        # or is an actuator keyword → operator-only.
        is_actuator = any(kw in name.lower() for kw in
                          ['move', 'grasp', 'speak', 'write', 'set_', 'execute', 'control'])
        if is_actuator and role == 'viewer':
            continue

        # Filter by glob
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            allowed.append(schema)

    return allowed


def check_tool_permission(peer_id: str, tool_name: str) -> tuple[bool, str]:
    """Pre-flight check before calling a tool on behalf of a peer.

    Returns (allowed, reason). The actuator double gate (role + canvas binding)
    is enforced in mcp_client or event/llm.py when the tool actually fires —
    this only checks the peer's static permission.
    """
    peer = store.get(peer_id)
    if peer is None:
        return False, 'unknown_peer'
    if peer['role'] == 'blocked':
        return False, 'blocked'

    tool_filter = peer.get('tool_filter', '*')
    patterns = [p.strip() for p in tool_filter.split(',') if p.strip()]
    if not patterns:
        patterns = ['*']

    if not any(fnmatch.fnmatch(tool_name, pat) for pat in patterns):
        return False, f'tool "{tool_name}" not in filter: {tool_filter}'

    # Role check
    role = peer['role']
    is_actuator = any(kw in tool_name.lower() for kw in
                      ['move', 'grasp', 'speak', 'write', 'set_', 'execute', 'control'])
    if is_actuator and role == 'viewer':
        return False, f'tool "{tool_name}" requires operator role, peer is {role}'

    return True, ''
