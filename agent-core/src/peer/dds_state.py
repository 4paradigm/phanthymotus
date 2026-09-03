"""
peer/dds_state.py — ROS2 DDS state sharing (optional).

Publishes local ROS topic list to `/motus/peer/{peer_id}/topics` and subscribes
to other peers' topics. This is **state broadcast only** — no commands, because
DDS has no authentication and anyone on ROS_DOMAIN_ID can write.

The topology view in the dashboard uses this to show which peers see which topics,
helping debug perception/actuation routing.

This module is optional: if `rclpy` is unavailable or ROS_DOMAIN_ID is unset, DDS
sharing silently disables. It's a debug/visibility feature, not a control path.
"""

import asyncio
import json
import os
import threading
import time


_node = None
_publisher = None
_subscribers: dict[str, object] = {}
_peer_topics: dict[str, dict] = {}  # peer_id → {topics: [...], last_seen: float}
_running = False
_thread: threading.Thread | None = None
# True only if this module called rclpy.init() itself. ros2_bridge.py normally
# owns the context; shutting down someone else's would tear the whole DDS bus
# out from under the inspection API.
_owns_rclpy = False


def is_available() -> bool:
    """Check if ROS2 is available and DDS sharing can run."""
    try:
        import rclpy  # noqa: F401
        return 'ROS_DOMAIN_ID' in os.environ
    except ImportError:
        return False


def start():
    """Start DDS state sharing in a background thread."""
    global _running, _thread
    if not is_available():
        print('[peer] DDS state sharing disabled: rclpy unavailable or ROS_DOMAIN_ID unset')
        return
    if _running:
        return
    _running = True
    _thread = threading.Thread(target=_run_loop, daemon=True, name='dds_state')
    _thread.start()
    print('[peer] DDS state sharing started')


def stop():
    """Stop DDS state sharing."""
    global _running
    _running = False
    if _thread:
        _thread.join(timeout=2.0)
    _cleanup()
    print('[peer] DDS state sharing stopped')


def get_peer_topics() -> dict[str, dict]:
    """Return the latest topic lists from all discovered peers.

    Returns {peer_id: {topics: [...], last_seen: float}}. Stale entries
    (>60s) are pruned.
    """
    now = time.time()
    stale = [pid for pid, info in _peer_topics.items() if now - info['last_seen'] > 60]
    for pid in stale:
        _peer_topics.pop(pid, None)
    return dict(_peer_topics)


def _run_loop():
    """Background thread: init ROS node, publish our topics, subscribe to peers."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    global _node, _publisher

    # rclpy's context is process-global and may only be initialised once.
    # ros2_bridge.py owns that lifecycle (it calls rclpy.init() at startup and
    # rclpy.shutdown() on exit), so this thread must attach to the existing
    # context rather than create one — calling init() again raises
    # "Context.init() must only be called once" and kills this thread.
    if not rclpy.ok():
        # No bridge running (rclpy present but never initialised). Own it here,
        # and remember that we did so cleanup knows whether it may shut down.
        rclpy.init()
        global _owns_rclpy
        _owns_rclpy = True

    _node = Node('motus_peer_dds_state')

    from peer import identity
    my_peer_id = identity.peer_id()

    # Publish our topic list every 5s
    topic_name = f'/motus/peer/{my_peer_id}/topics'
    _publisher = _node.create_publisher(String, topic_name, 10)

    # Subscribe to all peers we know about (discovered via mDNS/static)
    _subscribe_to_peers()

    last_publish = 0.0
    while _running:
        now = time.time()
        if now - last_publish > 5.0:
            _publish_our_topics()
            last_publish = now
            # Refresh subscriptions in case new peers appeared
            _subscribe_to_peers()
        rclpy.spin_once(_node, timeout_sec=1.0)

    _cleanup()
    # Only tear down the context if we created it. ros2_bridge.py's bus, the
    # inspection API and every /ws/bus/{topic} subscriber share this context —
    # shutting down one we merely borrowed would kill all of them.
    if _owns_rclpy:
        rclpy.shutdown()


def _publish_our_topics():
    """Publish our ROS topic list as JSON."""
    if not _node or not _publisher:
        return
    topic_list = [name for name, _ in _node.get_topic_names_and_types()]
    payload = json.dumps({'topics': topic_list, 'timestamp': time.time()})
    from std_msgs.msg import String
    msg = String()
    msg.data = payload
    _publisher.publish(msg)


def _subscribe_to_peers():
    """Subscribe to /motus/peer/{peer_id}/topics for all discovered peers."""
    from peer.registry import registry
    from std_msgs.msg import String

    discovered = registry.discovered(include_paired=True)
    for advert in discovered:
        peer_id = advert['peer_id']
        if peer_id in _subscribers:
            continue
        topic_name = f'/motus/peer/{peer_id}/topics'
        try:
            sub = _node.create_subscription(
                String, topic_name,
                lambda msg, pid=peer_id: _on_peer_topics(pid, msg),
                10
            )
            _subscribers[peer_id] = sub
        except Exception as e:
            print(f'[peer] DDS subscribe to {peer_id} failed: {e}')


def _on_peer_topics(peer_id: str, msg):
    """Callback when a peer publishes its topic list."""
    try:
        data = json.loads(msg.data)
        _peer_topics[peer_id] = {
            'topics': data.get('topics', []),
            'last_seen': time.time(),
        }
    except (json.JSONDecodeError, AttributeError):
        pass


def _cleanup():
    """Destroy ROS node and clear state."""
    global _node, _publisher, _subscribers
    if _node:
        _node.destroy_node()
        _node = None
    _publisher = None
    _subscribers.clear()
    _peer_topics.clear()
