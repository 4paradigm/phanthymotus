"""
peer/delegation.py — task delegation logic: spawn subagents on behalf of peers.

A peer can delegate a task by sending a SubagentSpec. This agent spawns it
locally, streams progress, and returns the result. The delegated spec's
tool_filter is intersected with what the peer's role allows — a viewer peer
delegating a task cannot magically gain operator tools through the delegation.

hop_count prevents infinite chains: A → B → C → D is capped at 2 hops.
"""

import contextvars
from typing import Annotated

from subagent.protocol import SubagentSpec, SubagentResult
from peer import store


MAX_HOP_COUNT = 2

# How many peer hops the *currently executing* agent is already at.
#
# The peer_delegate tool is a single shared function reachable from both the
# main agent loop and from any subagent, and the dispatch path passes no caller
# context. Without an ambient value the tool would always report hop 0, and a
# chain A→B→C→D would look like a fresh delegation at every step — the limit
# would only ever constrain the first hop.
#
# The main loop leaves this at 0 (it is the origin). A subagent spawned from an
# inbound delegation sets it to its own spec.hop_count for the duration of the
# run; see subagent/agent.py.
current_hop_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    'peer_current_hop_count', default=0
)


async def peer_delegate(
    peer_id: Annotated[str, 'Paired peer to hand the task to — the name shown in the peers list, or its peer_id.'],
    goal: Annotated[str, 'What the remote agent should accomplish, stated as a complete instruction.'],
    timeout_s: Annotated[float, 'Seconds to wait for the result before giving up.'] = 120.0,
    max_rounds: Annotated[int, 'Maximum reasoning rounds the remote agent may use.'] = 10,
) -> str:
    """Ask a paired peer to carry out a task and wait for its result.

    The peer runs it under *its own* permissions, not ours: it re-clips the
    tool filter against the role it granted us, and its own actuator gates
    still apply. This asks; it does not command.
    """
    from peer import store as _store, transport as _transport
    from peer.registry import registry as _registry

    # Accept a name as well as a peer_id: the environment snapshot carries names,
    # because a 32-char hex fingerprint per peer costs more tokens than the rest
    # of the line, so a name is what the model has when it decides to delegate.
    # Names collide, so peer/naming.py renders and resolves them as one contract.
    from peer import naming as _naming
    peer, why = _naming.resolve(peer_id, _store.list_peers())
    if peer is None:
        return f'Error: {why}'
    peer_id = peer['peer_id']
    if peer['role'] == 'blocked':
        return f'Error: peer "{peer_id}" is blocked.'

    endpoints = _registry.endpoints_for(peer_id)
    if not endpoints:
        return (f'Error: no known endpoint for "{peer_id}" — it has not been seen by any '
                f'discovery provider since this agent started.')

    # Send our *current* depth; the receiver increments and enforces the limit.
    # Refuse locally too, so a doomed request never leaves the machine.
    hop = current_hop_count.get()
    if hop > MAX_HOP_COUNT:
        return (f'Error: refusing to delegate — already {hop} peer hops deep '
                f'(limit {MAX_HOP_COUNT}). This task has been passed along too many times.')

    result, err = await _transport.post_json(
        endpoints, '/api/peer/delegate',
        {'goal': goal, 'timeout_s': timeout_s, 'max_rounds': max_rounds, 'hop_count': hop},
        timeout=timeout_s + 10,
    )
    if result is None:
        return f'Error: delegation to "{peer_id}" failed: {err}'
    if result.get('status') != 'completed':
        return (f'Peer "{peer["display_name"] or peer_id[:12]}" did not complete the task '
                f'(status={result.get("status")}): {result.get("error") or "no detail"}')
    return result.get('output') or '(peer returned an empty result)'


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


async def peer_list(
    include_unpaired: Annotated[
        bool, 'Also list agents that have been discovered but not paired yet.'] = False,
) -> str:
    """List the other agents this one knows about, and whether they can be worked with.

    Answers "can you see other robots?" — which the agent had no way to answer
    before this existed: nothing exposed the peer registry, so it truthfully but
    misleadingly reported that it could not.

    Pairing is deliberately not offered as a tool. It needs a human to compare a
    6-digit code on both machines, so an agent that could pair on its own would
    defeat the check; this points at the UI instead.
    """
    from peer import store as _store
    from peer.registry import registry as _registry

    from peer import liveness as _liveness, naming as _naming

    lines = []
    paired = _store.list_peers()
    lab = _naming.labels(paired)
    for p in paired:
        live = _liveness.liveness(p)
        if live['online']:
            running = live.get('agent_running')
            if running is False:
                # Reachable but its agent loop is down: tools and state work,
                # peer_delegate answers 503. Saying only "online" here is how an
                # agent ends up promising work this peer cannot accept.
                # Measured, not assumed: with the loop off a peer still serves
                # tools/list and executes tools/call (the canvas gate reads the
                # saved layout, and its devices run independently of the loop).
                # Only delegation fails, and downstream cards on its canvas may
                # be stopped, so a call can dispatch and still have no effect.
                state = ('online, agent loop off — tools and state work, but it cannot take '
                         'delegated tasks and its downstream cards may be stopped')
            elif running is True:
                state = 'online, agent loop running'
            else:
                state = 'online (agent loop state unknown)'
        elif live['endpoints']:
            # An address is known but nothing has been heard: paired-and-switched-off
            # looks exactly like this, and it is not the same as never paired.
            state = f'offline, last contact {_liveness.describe_age(live["contact_age_s"])}'
        else:
            state = 'offline, no known address'
        lines.append(f'- {lab[p["peer_id"]]} (peer_id={p["peer_id"]}, role={p["role"]}, {state})')
    if not paired:
        lines.append('- (no paired agents)')

    if include_unpaired:
        known = {p['peer_id'] for p in paired}
        fresh = [a for a in _registry.discovered() if a['peer_id'] not in known]
        lines.append('')
        if fresh:
            lines.append('Discovered but not paired (a human must confirm the pairing code '
                         'on both machines, in the Peers page):')
            for a in fresh:
                nm = a.get('display_name') or a['peer_id'][:12]
                lines.append(f'- {nm} (peer_id={a["peer_id"]}, via {a.get("source", "?")})')
        else:
            lines.append('No unpaired agents discovered.')

    lines.append('')
    lines.append('peer_delegate takes either the name shown above or the peer_id.')
    lines.append('What can be done with a paired agent: send it a message, call the tools '
                 'its role allows, read the topics it shares, or hand it a task with '
                 'peer_delegate. A peer can never drive an actuator here directly — an '
                 'inbound request is input to this agent, not a command.')
    return '\n'.join(lines)


async def peer_state(
    peer: Annotated[str, 'Name or peer_id of one peer, or empty for all of them.'] = '',
) -> str:
    """What other agents can currently see: their ROS topics and whether they can act.

    This is the state each peer pushes here every few seconds over its signed link
    (`/api/peer/inbox/state`). It answers "is the other robot's camera up?" without
    calling anything on it, and it is the only way to know a peer's agent loop is
    running before handing it a task — reachable and able to accept work are
    different facts, and the second one is what a delegation needs.

    The data was previously reachable only through the API, so the agent could not
    see it at all — the same gap that had this agent answering "no, I cannot see
    other robots" while paired with one.
    """
    from peer import dds_state as _state, liveness as _liveness, naming as _naming
    from peer import store as _store

    peers = _store.list_peers()
    if not peers:
        return 'No paired agents, so there is no peer state to report.'

    if peer:
        found, why = _naming.resolve(peer, peers)
        if found is None:
            return f'Error: {why}'
        peers = [found]

    shared = _state.get_peer_topics()
    labels = _naming.labels(_store.list_peers())
    lines = []
    for p in peers:
        live = _liveness.liveness(p)
        info = shared.get(p['peer_id']) or {}
        topics = info.get('topics') or []
        name = labels[p['peer_id']]
        if not live['online']:
            lines.append(f'- {name}: offline, last contact '
                         f'{_liveness.describe_age(live["contact_age_s"])}. '
                         f'Topics below are the last it reported.'
                         if topics else f'- {name}: offline, nothing reported yet.')
        elif live['agent_running'] is False:
            lines.append(f'- {name}: online, agent loop off (tools and state work, '
                         f'it cannot take a delegated task)')
        else:
            lines.append(f'- {name}: online')
        if topics:
            lines.append(f'    topics ({len(topics)}): ' + ', '.join(topics))
    return '\n'.join(lines)
