"""
peer/dds_state.py — share the local ROS2 topic list with paired peers.

What is shared is DDS state; **how** it travels is signed HTTPS, not DDS.

This module used to publish the local topic list on a ROS topic and subscribe to
every peer's equivalent. Two things ended that:

1. **DDS is now locked to loopback** (`deploy/dds-local.xml`), because one
   robot's `/remote_control/message` was reaching every robot on the LAN — a
   command bus with no authentication and no addressing. Cross-machine DDS is
   physically cut, so the subscriptions could never deliver anything again.
2. **Transport isolation in FastDDS is process-wide.** Measured on real
   hardware: a profile cannot let one participant stay local while another
   talks to the network, so "keep the internal bus local but give peers their
   own DDS domain" is not implementable, whatever the domain layout.

Moving to `transport.post_json` is not a workaround; it is what the data always
needed. The peer bus had no authentication, so any process on the same
ROS_DOMAIN_ID could forge another robot's topic list. Now every push carries an
Ed25519 signature and is accepted only from a paired peer.

Still **state only, never commands** — the rule outlives the transport change.

Optional throughout: with no ROS2 bridge running there are no local topics to
share, and this quietly does nothing. It is a visibility feature, and the
topology view degrades to empty rather than the app failing to start.
"""

import asyncio
import time


# peer_id → {'topics': [...], 'last_seen': float}
_peer_topics: dict[str, dict] = {}
_task: asyncio.Task | None = None

PUSH_INTERVAL_S = 5.0
STALE_AFTER_S = 60.0
STATE_PATH = '/api/peer/inbox/state'


def is_available() -> bool:
    """Whether there is any local DDS state worth sharing.

    Asked of the ROS2 bridge rather than of rclpy: rclpy imports fine in a
    container where the bridge never came up, and reporting "available" there
    produced a topology view that was permanently, inexplicably empty.
    """
    try:
        import ros2_bridge
        return ros2_bridge.get_dds_topics() is not None
    except Exception:
        return False


def local_topics() -> list[str]:
    """The ROS topic names visible on this machine."""
    try:
        import ros2_bridge
        return sorted(ros2_bridge.get_dds_topics() or [])
    except Exception:
        return []


def record_peer_topics(peer_id: str, topics: list[str]) -> None:
    """Store what a peer just told us it can see. Called by the inbox endpoint."""
    _peer_topics[peer_id] = {'topics': list(topics), 'last_seen': time.time()}


def get_peer_topics() -> dict[str, dict]:
    """Latest topic list per peer, stale entries dropped.

    Pruning on read, not on a timer: a peer that stops pushing should fade from
    the topology view on its own, and a timer would be one more thing that can
    silently stop running.
    """
    now = time.time()
    for pid in [p for p, info in _peer_topics.items()
                if now - info['last_seen'] > STALE_AFTER_S]:
        _peer_topics.pop(pid, None)
    return dict(_peer_topics)


def start() -> None:
    """Begin pushing our topic list to paired peers."""
    global _task
    if _task and not _task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        print('[peer] state sharing not started: no running event loop')
        return
    _task = loop.create_task(_push_loop(), name='peer_state_push')
    print('[peer] state sharing started (signed HTTPS)')


def stop() -> None:
    """Stop pushing. Safe to call when never started."""
    global _task
    if _task:
        _task.cancel()
        _task = None
    _peer_topics.clear()
    print('[peer] state sharing stopped')


async def _push_loop() -> None:
    while True:
        try:
            await push_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # One bad round must not end the loop: when the old DDS version let
            # an exception escape, sharing stayed dead for the rest of the
            # process with a single traceback in the log and no other symptom.
            print(f'[peer] state push failed: {type(e).__name__}: {e}')
        await asyncio.sleep(PUSH_INTERVAL_S)


async def push_once() -> int:
    """Push the local topic list to every paired peer. Returns how many took it.

    Failures are per-peer and silent by design — an unreachable peer is the
    normal state of a robot that has been switched off, and logging it every
    five seconds would bury everything else.
    """
    from peer import store, transport
    from peer.registry import registry

    topics = local_topics()
    if not topics:
        return 0

    payload = {'topics': topics, 'timestamp': time.time()}
    delivered = 0
    for peer in store.list_peers():
        endpoints = registry.endpoints_for(peer['peer_id'])
        if not endpoints:
            continue
        resp, _reason = await transport.post_json(endpoints, STATE_PATH, payload)
        if resp is not None:
            delivered += 1
    return delivered
