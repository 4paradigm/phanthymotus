"""
canvas_binding.py — which tools the operator has wired to the decision core.

The canvas is an authorization surface, not just a diagram: a tool reaches the
agent only if a human drew a connection from the decision-core card to it. That
rule lived solely inside event/llm.py, so it only ever constrained the LLM path.

The peer tool proxy (api/peer.py) reached mcp_client.call_tool directly and was
therefore not subject to it — an `operator` peer could invoke any tool the
tool_filter allowed, wired or not. Both paths now consult this module, so the
gate has one definition instead of two divergent ones.
"""

import config


def bound_tool_names() -> set[str]:
    """Full tool names (`mcp__<mcp_id>__<tool>`) wired to the decision core.

    Reads the same canvas layout event/llm.py does. Includes tools whose MCP is
    currently offline: absence should read as "not reachable right now", not as
    "not authorised", and conflating them would silently widen or narrow
    permission depending on device uptime.
    """
    layout = config.main.get('canvas_layout', {}) or {}
    cards = layout.get('cards', []) or []
    exec_conns = layout.get('execConnections', []) or []

    core_card_ids = {c['id'] for c in cards if c.get('mcpId') == 'agentcore'}

    names = set()
    for ec in exec_conns:
        if ec.get('fromCardId') not in core_card_ids:
            continue
        mcp_id = ec.get('toMcpId', '')
        tool_name = ec.get('toToolName', '')
        if not mcp_id or not tool_name:
            continue
        names.add(f'mcp__{mcp_id}__{tool_name}')
    return names


def is_peer_tool(full_tool_name: str) -> bool:
    """True for a tool that lives on a paired peer, not on this machine.

    `mcp__peer:<id>__<tool>` — see peer/mcp_bridge.py.
    """
    return full_tool_name.startswith('mcp__peer:')


def is_bound(full_tool_name: str) -> bool:
    """True if this tool is wired to the decision core on the local canvas.

    Split-out sub-tools (x-action-params) carry the parent's name plus a
    suffix, so a prefix match keeps them attached to the parent's binding
    rather than silently failing the gate.

    A **peer's** tool is exempt. This canvas is this operator's authority over what
    *this* machine exposes; a tool that runs on another robot is that operator's to
    gate, and they do — by the peer's role, its `tool_filter`, and their own canvas.
    Demanding a local card for a remote tool would mean inventing a peer card with
    nothing behind it, and a local omission would then look like a remote refusal.
    Inbound peer requests are unaffected: those name *local* tools, never `peer:`.
    """
    if is_peer_tool(full_tool_name):
        return True

    bound = bound_tool_names()
    if full_tool_name in bound:
        return True
    return any(full_tool_name.startswith(b + '__') for b in bound)
