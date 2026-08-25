"""
utils/ros_lifecycle.py — Safe teardown of per-instance ROS2 nodes.

Multi-instance plugins create one rclpy Node per input topic and register it
with the bundle's MultiThreadedExecutor. When an instance is retired (topic
change, stop-all, shutdown) the node must be removed from the executor *and*
destroyed, otherwise its subscriptions keep matching camera publishers and
its callbacks keep waking the executor for the rest of the process. The ASR
plugin established this order; OCR and obstacle reuse it here.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def dispose_node(executor, node, *, label: str = "") -> None:
    """Remove `node` from `executor` and destroy it, logging (not raising) errors.

    Callers stop the node's own workers/subscriptions first; this helper only
    handles the executor + rclpy handle lifecycle. Every step is attempted
    even if a previous one failed.
    """
    name = label or getattr(node, "get_name", lambda: type(node).__name__)()
    try:
        executor.remove_node(node)
    except Exception as error:
        log.warning("failed to remove ROS node %r from executor: %s", name, error)
    try:
        node.destroy_node()
    except Exception as error:
        log.warning("failed to destroy ROS node %r: %s", name, error)


__all__ = ["dispose_node"]
