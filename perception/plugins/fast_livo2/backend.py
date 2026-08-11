"""ROS topic control plane for the FAST-LIVO2 companion."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid

from .core import FastLivo2BackendError


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
        self._last_status = {"state": "waiting_for_fast_livo2_companion"}
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
            self._publisher.publish(message)
            deadline = time.monotonic() + self._request_timeout
            while request_id not in self._responses:
                if self._closed:
                    raise FastLivo2BackendError("backend_closed", "backend is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FastLivo2BackendError(
                        "fast_livo2_response_timeout",
                        f"FAST-LIVO2 companion did not answer {action}",
                    )
                self._condition.wait(timeout=remaining)
            payload = self._responses.pop(request_id)
        if payload.get("status") == "error":
            raise FastLivo2BackendError(
                str(payload.get("error_code", "fast_livo2_error")),
                str(payload.get("error", "FAST-LIVO2 companion rejected request")),
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

    def _wait_for_bridge(self) -> None:
        deadline = time.monotonic() + self._discovery_timeout
        while self._publisher.get_subscription_count() == 0:
            if self._closed:
                raise FastLivo2BackendError("backend_closed", "backend is closed")
            if time.monotonic() >= deadline:
                raise FastLivo2BackendError(
                    "fast_livo2_companion_unavailable",
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
            if payload.get("event") == "response" and isinstance(request_id, str):
                self._responses[request_id] = payload
            self._condition.notify_all()


__all__ = ["RosTopicFastLivo2Backend"]
