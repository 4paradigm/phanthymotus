"""ROS topic transport between the card and its in-container Nav2 runtime."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid

from .core import NavigationBackendError


_TERMINAL = {
    "arrived",
    "succeeded",
    "cancelled",
    "stopped",
    "timeout",
    "error",
    "aborted",
    "rejected",
}

_VELOCITY_LIMIT_FIELDS = (
    "min_x_mps",
    "max_x_mps",
    "min_y_mps",
    "max_y_mps",
    "min_yaw_rps",
    "max_yaw_rps",
)


def _topic_root(namespace: str) -> str:
    normalized = namespace.strip("/")
    return f"/{normalized}" if normalized else ""


def _build_command_payload(
    *,
    request_id: str,
    nav_id: str | None,
    action: str,
    args: dict,
    velocity_limits: dict,
) -> dict:
    return {
        "request_id": request_id,
        "nav_id": nav_id,
        "action": action,
        "args": args,
        "velocity_limits": dict(velocity_limits),
    }


class RosTopicNavigationBackend:
    """Request/reply bridge using only std_msgs/String in Perception."""

    def __init__(self, cfg: dict, namespace: str, executor):
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import String

        root = _topic_root(namespace)
        self._command_topic = cfg.get(
            "command_topic", f"{root}/navigation/nav2/command"
        )
        self._status_topic = cfg.get(
            "status_topic", f"{root}/navigation/nav2/status"
        )
        self._request_timeout = float(cfg.get("request_timeout_sec", 30.0))
        self._discovery_timeout = float(cfg.get("discovery_timeout_sec", 5.0))
        if self._request_timeout <= 0 or self._discovery_timeout <= 0:
            raise ValueError("navigation bridge timeouts must be positive")
        self._velocity_limits = {
            field: float(cfg[field]) for field in _VELOCITY_LIMIT_FIELDS
        }

        node_suffix = re.sub(r"[^a-zA-Z0-9_]", "_", namespace or "root")
        self._node = Node(f"nav2_{node_suffix}")
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._String = String
        self._command_pub = self._node.create_publisher(
            String, self._command_topic, command_qos
        )
        self._status_sub = self._node.create_subscription(
            String, self._status_topic, self._on_status, status_qos
        )
        self._executor = executor
        self._executor.add_node(self._node)

        self._condition = threading.Condition()
        self._responses: dict[str, dict] = {}
        self._navigation: dict[str, dict] = {}
        self._last_status: dict = {
            "state": "waiting_for_nav2_runtime",
            "shadow_only": True,
            "physical_execution": False,
        }
        self._closed = False

    def info(self) -> dict:
        with self._condition:
            result = dict(self._last_status)
            result.update(
                {
                    "backend": "nav2_ros_topic",
                    "command_topic": self._command_topic,
                    "status_topic": self._status_topic,
                    "bridge_subscribers": self._command_pub.get_subscription_count(),
                    "shadow_only": True,
                    "physical_execution": False,
                }
            )
            return result

    def execute(self, action: str, args: dict, *, nav_id: str | None) -> dict:
        if action == "wait_navigation_done":
            if nav_id is None:
                raise NavigationBackendError(
                    "no_active_navigation", "wait requires a navigation id"
                )
            return self._wait_navigation(nav_id, args["stall_timeout"])

        payload = self._request(action, args, nav_id=nav_id)
        if payload.get("status") == "error":
            raise NavigationBackendError(
                str(payload.get("error_code", "nav2_error")),
                str(payload.get("error", "Nav2 runtime rejected the request")),
            )
        return payload

    def stop(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._executor.remove_node(self._node)
        self._node.destroy_node()

    def _on_status(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self._condition:
            self._last_status = dict(payload)
            request_id = payload.get("request_id")
            if payload.get("event") == "response" and isinstance(request_id, str):
                self._responses[request_id] = payload
            nav_id = payload.get("nav_id")
            if isinstance(nav_id, str) and nav_id:
                previous = self._navigation.get(nav_id, {})
                progress_seq = payload.get("progress_seq")
                if progress_seq != previous.get("progress_seq"):
                    payload["progress_received_at"] = time.monotonic()
                else:
                    payload["progress_received_at"] = previous.get(
                        "progress_received_at", time.monotonic()
                    )
                self._navigation[nav_id] = dict(payload)
            self._condition.notify_all()

    def _wait_for_bridge(self) -> None:
        deadline = time.monotonic() + self._discovery_timeout
        while self._command_pub.get_subscription_count() == 0:
            if self._closed:
                raise NavigationBackendError("backend_closed", "backend is closed")
            if time.monotonic() >= deadline:
                raise NavigationBackendError(
                    "nav2_runtime_unavailable",
                    f"no subscriber on {self._command_topic}",
                )
            time.sleep(0.05)

    def _request(self, action: str, args: dict, *, nav_id: str | None) -> dict:
        self._wait_for_bridge()
        request_id = uuid.uuid4().hex
        payload = _build_command_payload(
            request_id=request_id,
            nav_id=nav_id,
            action=action,
            args=args,
            velocity_limits=self._velocity_limits,
        )
        message = self._String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._condition:
            self._command_pub.publish(message)
            deadline = time.monotonic() + self._request_timeout
            while request_id not in self._responses:
                if self._closed:
                    raise NavigationBackendError("backend_closed", "backend is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise NavigationBackendError(
                        "nav2_response_timeout",
                        f"Nav2 runtime did not answer {action} within "
                        f"{self._request_timeout:.1f}s",
                    )
                self._condition.wait(timeout=remaining)
            return self._responses.pop(request_id)

    def _wait_navigation(self, nav_id: str, stall_timeout: float) -> dict:
        with self._condition:
            progress_at = time.monotonic()
            progress_seq = None
            while True:
                if self._closed:
                    raise NavigationBackendError("backend_closed", "backend is closed")
                state = self._navigation.get(nav_id)
                if state:
                    status = state.get("status") or state.get("state")
                    if status in _TERMINAL:
                        return dict(state)
                    if status == "paused":
                        return dict(state)
                    if state.get("progress_seq") != progress_seq:
                        progress_seq = state.get("progress_seq")
                        progress_at = state.get("progress_received_at", time.monotonic())
                remaining = stall_timeout - (time.monotonic() - progress_at)
                if remaining <= 0:
                    break
                self._condition.wait(timeout=min(remaining, 0.5))

        stop_result = self._request("stop_nav", {}, nav_id=nav_id)
        stop_status = stop_result.get("status")
        terminal_confirmed = bool(stop_result.get("terminal_confirmed"))
        if stop_status not in {"stopped", "cancelled"} or not terminal_confirmed:
            raise NavigationBackendError(
                "navigation_stalled_stop_unconfirmed",
                "navigation stalled and Nav2 did not confirm a terminal stop",
            )
        return {
            "status": "timeout",
            "nav_id": nav_id,
            "error_code": "navigation_stalled",
            "error": f"no navigation progress for {stall_timeout:.1f}s",
            "stop_result": stop_status,
            "terminal_confirmed": True,
        }
