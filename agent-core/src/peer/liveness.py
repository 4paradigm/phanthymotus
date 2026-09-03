"""
peer/liveness.py — is a paired peer reachable *right now*.

Pairing is durable; reachability is not. Conflating them is how an agent ends up
promising to hand a task to a robot that has been switched off: the peers table
still lists it, `peer_delegate` still accepts its id, and the failure only
surfaces as a timeout after the agent has already committed to the plan out loud.

Two independent signals, because each alone lies in a different direction:

* **last contact** — every inbound signed request touches the peer's row, and
  state sharing pushes every 5s, so a peer that is up and paired refreshes this
  continuously. It says nothing, though, about a peer that is up but has state
  sharing disabled.
* **a recent discovery advert** — mDNS re-announces on a timer, so a *recent*
  sighting means the peer answered on the network. The registry keeps adverts
  for `STALE_AFTER_S` (300s) before pruning, which is far too long to treat
  mere presence in that table as alive — a robot switched off two minutes ago is
  still in it. So the advert's own age is checked, not its existence.

Online means *either* holds. Both are reported so a caller can say which, rather
than reducing everything to one boolean the reader has to guess the meaning of.

**Reachable is not the same as able to take work.** State sharing pushes whether
or not the agent loop is running, so a peer with 智能控制 switched off is exactly
as reachable as one that can accept a delegation — and `/api/peer/delegate` then
answers 503, because delegation needs the subagent manager that only exists once
the loop is up. `agent_running` carries the peer's own answer, so the caller can
tell "cannot be reached" from "reached, but not accepting tasks".
"""

import time


# How long after the last inbound contact a peer still counts as online.
# State pushes are 5s apart, so 30s tolerates a few missed rounds without
# reporting a peer that has actually gone away as still present.
CONTACT_FRESH_S = 30.0

# How recent a discovery sighting has to be. mDNS refreshes every 60s
# (MdnsProvider.REFRESH_INTERVAL_S), so 90s allows one missed announcement
# without flapping, and is well inside the registry's 300s retention.
ADVERT_FRESH_S = 90.0


def liveness(peer: dict) -> dict:
    """Reachability for one peers-table row.

    Returns `{'online': bool, 'contact_age_s': float|None, 'endpoints': [...]}`.
    `contact_age_s` is None when the peer has never contacted us — different from
    "contacted us long ago", and worth keeping distinguishable.
    """
    from peer.registry import registry

    peer_id = peer.get('peer_id', '')
    endpoints = registry.endpoints_for(peer_id) if peer_id else []

    last_seen = peer.get('last_seen') or 0
    age = (time.time() - last_seen) if last_seen else None

    recently_heard = age is not None and age <= CONTACT_FRESH_S

    advert = registry.get(peer_id) if peer_id else None
    advert_age = (time.time() - advert.last_seen) if advert else None
    recently_seen = advert_age is not None and advert_age <= ADVERT_FRESH_S

    # None rather than False when the peer has never told us: an older peer that
    # does not send the field is not the same as one that says its loop is off.
    from peer import dds_state
    shared = dds_state.get_peer_topics().get(peer_id) or {}
    agent_running = shared.get('agent_running')

    return {
        'online': bool(recently_heard or recently_seen),
        'contact_age_s': age,
        'advert_age_s': advert_age,
        'endpoints': endpoints,
        'agent_running': agent_running if isinstance(agent_running, bool) else None,
    }


def describe_age(age_s: float | None) -> str:
    """Human-scale age, for a prompt or a tool result."""
    if age_s is None:
        return 'never'
    if age_s < 60:
        return f'{int(age_s)}s ago'
    if age_s < 3600:
        return f'{int(age_s / 60)}min ago'
    return f'{age_s / 3600:.1f}h ago'
