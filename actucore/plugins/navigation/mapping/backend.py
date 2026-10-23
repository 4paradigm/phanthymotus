"""ROS topic control plane for the in-container FAST-LIVO2 runtime."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid

from .core import FastLivo2BackendError


# The supervisor may wait for its 120 s controlled FAST-LIVO2 shutdown and
# then up to 130 s for the adapter to persist the confirmed static map, plus
# a bounded session-artifact snapshot and fsync.  This transport timeout must
# therefore not shrink with the user-facing ordinary request timeout setting.
_STOP_MAPPING_RESPONSE_TIMEOUT_SEC = 360.0
_UNLOAD_MAP_RESPONSE_TIMEOUT_SEC = 360.0
_LOAD_MAP_RESPONSE_TIMEOUT_SEC = 900.0
_RELOCALIZE_RESPONSE_TIMEOUT_SEC = 180.0


class RosTopicFastLivo2Backend:
    def __init__(self, cfg: dict, namespace: str, executor):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        root = f"/{namespace.strip('/')}"
        self._command_topic = f"{root}/navigation/fast_livo2/command"
        self._status_topic = f"{root}/navigation/fast_livo2/status"
        self._request_timeout = float(cfg["request_timeout_sec"])
        self._discovery_timeout = float(cfg["discovery_timeout_sec"])
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", namespace)
        self._node = Node(f"fast_livo2_{suffix}")
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
        self._publisher = self._node.create_publisher(String, self._command_topic, command_qos)
        self._subscription = self._node.create_subscription(
            String, self._status_topic, self._on_status, status_qos
        )
        self._executor = executor
        self._executor.add_node(self._node)
        self._condition = threading.Condition()
        self._responses: dict[str, dict] = {}
        self._pending_requests: set[str] = set()
        self._last_status = {"state": "waiting_for_fast_livo2_runtime"}
        self._closed = False

    def info(self) -> dict:
        with self._condition:
            result = dict(self._last_status)
        result.update(
            {
                "backend": "fast_livo2_ros_topic",
                "command_topic": self._command_topic,
                "status_topic": self._status_topic,
                "bridge_subscribers": self._publisher.get_subscription_count(),
                "physical_execution": False,
            }
        )
        return result

    def execute(self, action: str, args: dict) -> dict:
        self._wait_for_bridge()
        request_id = uuid.uuid4().hex
        message = self._String()
        message.data = json.dumps(
            {"request_id": request_id, "action": action, "args": args},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._condition:
            self._pending_requests.add(request_id)
            self._publisher.publish(message)
            timeout = self._response_timeout(action)
            deadline = time.monotonic() + timeout
            try:
                while request_id not in self._responses:
                    if self._closed:
                        raise FastLivo2BackendError("backend_closed", "backend is closed")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise FastLivo2BackendError(
                            "fast_livo2_response_timeout",
                            f"FAST-LIVO2 runtime did not answer {action}",
                            details={
                                "retryable": action
                                in {
                                    "stop_mapping",
                                    "load_map",
                                    "unload_map",
                                    "relocalize",
                                }
                            },
                        )
                    self._condition.wait(timeout=remaining)
                payload = self._responses.pop(request_id)
            finally:
                self._pending_requests.discard(request_id)
        if payload.get("status") == "error":
            raise FastLivo2BackendError(
                str(payload.get("error_code", "fast_livo2_error")),
                str(payload.get("error", "FAST-LIVO2 runtime rejected request")),
                details=payload,
            )
        return payload

    def _response_timeout(self, action: str) -> float:
        if action == "stop_mapping":
            return max(self._request_timeout, _STOP_MAPPING_RESPONSE_TIMEOUT_SEC)
        if action == "unload_map":
            return max(self._request_timeout, _UNLOAD_MAP_RESPONSE_TIMEOUT_SEC)
        if action == "load_map":
            return max(self._request_timeout, _LOAD_MAP_RESPONSE_TIMEOUT_SEC)
        if action == "relocalize":
            return max(self._request_timeout, _RELOCALIZE_RESPONSE_TIMEOUT_SEC)
        return self._request_timeout

    def stop(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._executor.remove_node(self._node)
        self._node.destroy_node()

    def _wait_for_bridge(self) -> None:
        deadline = time.monotonic() + self._discovery_timeout
        while self._publisher.get_subscription_count() == 0:
            if self._closed:
                raise FastLivo2BackendError("backend_closed", "backend is closed")
            if time.monotonic() >= deadline:
                raise FastLivo2BackendError(
                    "fast_livo2_runtime_unavailable",
                    f"no subscriber on {self._command_topic}",
                )
            time.sleep(0.05)

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
            if (
                payload.get("event") == "response"
                and isinstance(request_id, str)
                and request_id in self._pending_requests
            ):
                self._responses[request_id] = payload
            self._condition.notify_all()


__all__ = ["RosTopicFastLivo2Backend"]
