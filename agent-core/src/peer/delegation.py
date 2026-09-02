"""
peer/delegation.py — task delegation logic: spawn subagents on behalf of peers.

A peer can delegate a task by sending a SubagentSpec. This agent spawns it
locally, streams progress, and returns the result. The delegated spec's
tool_filter is intersected with what the peer's role allows — a viewer peer
delegating a task cannot magically gain operator tools through the delegation.

hop_count prevents infinite chains: A → B → C → D is capped at 2 hops.
"""

from subagent.protocol import SubagentSpec, SubagentResult
from peer import store


MAX_HOP_COUNT = 2


def validate_delegation(peer_id: str, spec: SubagentSpec) -> tuple[bool, str]:
    """Pre-flight check before spawning a delegated task.

    Returns (allowed, reason). Enforces:
      * peer is paired and not blocked
      * hop_count ≤ MAX_HOP_COUNT
      * delegated tool_filter is subset of peer's allowed tools
    """
    peer = store.get(peer_id)
    if peer is None:
        return False, 'unknown_peer'
    if peer['role'] == 'blocked':
        return False, 'blocked'

    hop_count = getattr(spec, 'hop_count', 0)
    if hop_count > MAX_HOP_COUNT:
        return False, f'hop_count {hop_count} exceeds limit {MAX_HOP_COUNT}'

    # Tool filter intersection: if the peer has a filter, the delegated spec's
    # filter must be a subset. For simplicity, we enforce that the peer's
    # tool_filter string must match or be broader than the spec's — a full
    # glob intersection is complex, so this is a conservative check.
    peer_filter = peer.get('tool_filter', '*')
    if peer_filter != '*' and spec.tool_filter:
        # If peer has a restrictive filter, spec cannot widen it
        # For now: reject any delegation with tool_filter unless peer is '*'
        # (A real implementation would intersect globs; this is a safety gate)
        if peer_filter != '*':
            return False, f'peer tool_filter "{peer_filter}" does not allow arbitrary delegation filters'

    return True, ''


def prepare_delegated_spec(peer_id: str, spec: SubagentSpec) -> SubagentSpec:
    """Augment the delegated spec with local constraints.

    Increments hop_count and intersects tool_filter with the peer's permissions.
    Returns a new SubagentSpec ready for local spawn.
    """
    peer = store.get(peer_id)
    hop_count = getattr(spec, 'hop_count', 0) + 1

    # Tool filter: if peer has a filter, apply it
    peer_filter = peer.get('tool_filter', '*') if peer else '*'
    delegated_filter = spec.tool_filter or []
    if peer_filter != '*':
        # Simplistic: if peer has patterns, those become the filter
        # A real intersection would merge both; this ensures the peer's
        # restrictions are not bypassed
        delegated_filter = [p.strip() for p in peer_filter.split(',') if p.strip()]

    return SubagentSpec(
        goal=spec.goal,
        priority=spec.priority,
        model=spec.model,
        tool_filter=delegated_filter if delegated_filter else None,
        tool_deny=spec.tool_deny,
        max_rounds=spec.max_rounds,
        timeout_s=spec.timeout_s,
        hop_count=hop_count,
    )
