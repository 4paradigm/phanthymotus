"""Optional ROS goal input; callbacks never block the ROS executor."""
from __future__ import annotations

from collections import OrderedDict
import json
import logging
import math
import re
import threading

from .contract import GOAL_ID_PATTERN

log = logging.getLogger(__name__)


def _goal(data: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result

    if not isinstance(data, str) or len(data) > 16384:
        raise ValueError("invalid message size")
    obj = json.loads(data, object_pairs_hook=pairs)
    required = {"schema", "goal_id", "x", "y", "yaw"}
    if not isinstance(obj, dict) or not required <= obj.keys() or obj.keys() - (required | {"speed", "mode"}):
        raise ValueError("invalid fields")
    if obj["schema"] != "phanthy.navigation.goal.v1" or not isinstance(obj["goal_id"], str) or not re.fullmatch(GOAL_ID_PATTERN, obj["goal_id"]):
        raise ValueError("invalid schema or goal_id")
    for key in ("x", "y", "yaw", "speed"):
        if key in obj:
            value = obj[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("invalid numeric field")
    if "speed" in obj and not 0.3 <= obj["speed"] <= 1.0:
        raise ValueError("invalid speed")
    if "mode" in obj and (isinstance(obj["mode"], bool) or not isinstance(obj["mode"], int) or obj["mode"] != 0):
        raise ValueError("invalid mode")
    return {"action": "navigate_to_pose", "_control_nav_id": obj["goal_id"], **{k: obj[k] for k in ("x", "y", "yaw", "speed", "mode") if k in obj}}


class GoalInputCleanupError(RuntimeError):
    """Construction failed and cleanup remains pending; retain input for retry."""

    def __init__(self, goal_input, construction_error):
        super().__init__("goal input construction failed and cleanup is pending")
        self.goal_input = goal_input
        self.construction_error = construction_error


class GoalTopicInput:
    def __init__(self, topic, executor, dispatch):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._condition = threading.Condition()
        self._cleanup_lock = threading.Lock()
        self._pending = None
        self._closed = False
        self._seen = OrderedDict()
        self._dispatch = dispatch
        self._executor = executor
        self._node = None
        self._remove_pending = False
        self._destroy_pending = False
        self._worker = None
        try:
            self._node = Node("nav2_goal_input")
            self._destroy_pending = True
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20,
                             reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self._subscription = self._node.create_subscription(String, topic, self._on_message, qos)
            # Even a failing add_node may have registered the node before raising.
            self._remove_pending = True
            if executor.add_node(self._node) is False:
                raise RuntimeError("goal node was not added to executor")
            self._worker = threading.Thread(target=self._run, name="nav2-goal-input", daemon=True)
            self._worker.start()
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_exc:
                raise GoalInputCleanupError(self, exc) from cleanup_exc
            raise

    def _on_message(self, message):
        try:
            goal = _goal(message.data)
        except (ValueError, TypeError, OverflowError, RecursionError):
            log.warning("[nav2] ignoring invalid goal topic message")
            return
        with self._condition:
            if self._closed:
                return
            goal_id = goal["_control_nav_id"]
            previous = self._seen.get(goal_id)
            if previous is None and self._pending is not None:
                if self._pending["_control_nav_id"] == goal_id:
                    previous = self._pending
            if previous is not None:
                if previous != goal:
                    log.warning("[nav2] conflicting goal topic payload for id %s", goal_id)
                return
            self._pending = goal
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                while not self._closed and self._pending is None:
                    self._condition.wait()
                if self._closed:
                    return
                goal, self._pending = self._pending, None
                goal_id = goal["_control_nav_id"]
                if goal_id in self._seen:
                    continue
                # Reserve even on failure: a timeout may have executed remotely.
                # This bounded, per-instance cache cannot deduplicate across restart.
                self._seen[goal_id] = goal
                if len(self._seen) > 1024:
                    self._seen.popitem(last=False)
            try:
                result = self._dispatch(goal)
                if (not isinstance(result, dict) or result.get("status") == "error"
                        or result.get("state") == "error" or result.get("ok") is False
                        or result.get("error")):
                    log.warning("[nav2] goal topic dispatch not confirmed: %s", result)
            except Exception:
                log.exception("[nav2] goal topic dispatch failed; outcome may be unknown")

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        # Serialize retries and remember each completed stage independently.
        # Never destroy a node whose executor removal has not been confirmed.
        with self._cleanup_lock:
            if self._remove_pending:
                self._executor.remove_node(self._node)
                self._remove_pending = False
            if self._destroy_pending:
                self._node.destroy_node()
                self._destroy_pending = False
