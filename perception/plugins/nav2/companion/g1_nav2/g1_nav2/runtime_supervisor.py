"""Keep the Nav2 container alive while switching mapping/localization runtimes.

The supervisor owns only ROS process lifecycle.  It never publishes velocity and
does not talk to the robot Driver.  The command bridge requests a switch on a
private control topic after it has rejected active navigation or saved the map.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .runtime_process import VALID_MODES, build_launch_command, plain_map_name


class RuntimeSupervisor(Node):
    """Restart the child ROS launch when the command bridge requests a mode switch."""

    def __init__(self) -> None:
        super().__init__("g1_nav2_runtime_supervisor")
        self.declare_parameter(
            "switch_topic", "/ubuntu/navigation/nav2/runtime_switch"
        )
        self.declare_parameter("status_topic", "/ubuntu/navigation/nav2/status")
        self.declare_parameter("maps_root", "/maps")
        self._switch_topic = str(self.get_parameter("switch_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._maps_root = str(self.get_parameter("maps_root").value)
        self._mode = str(os.environ.get("NAV2_MODE", "mapping")).strip()
        self._map_name = str(os.environ.get("NAV2_MAP_NAME", "")).strip()
        if self._mode not in VALID_MODES:
            raise ValueError("NAV2_MODE must be mapping or localization")

        control_qos = QoSProfile(
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
        self._status_pub = self.create_publisher(String, self._status_topic, status_qos)
        self._switch_sub = self.create_subscription(
            String, self._switch_topic, self._on_switch, control_qos
        )
        self._lock = threading.Lock()
        self._pending: dict | None = None
        self._child: subprocess.Popen | None = None
        self._stopping = False
        self._worker = threading.Thread(
            target=self._run, name="nav2-runtime-supervisor", daemon=True
        )
        self._worker.start()

    def _on_switch(self, message: String) -> None:
        try:
            request = json.loads(message.data)
            if not isinstance(request, dict):
                raise ValueError("runtime switch request must be an object")
            target_mode = str(request.get("target_mode", ""))
            if target_mode not in VALID_MODES:
                raise ValueError("target_mode must be mapping or localization")
            request_id = str(request.get("request_id", "")).strip()
            if not request_id:
                raise ValueError("request_id is required")
            map_name = str(request.get("map_name", "")).strip()
            if target_mode == "localization":
                map_name = plain_map_name(map_name)
            with self._lock:
                if self._pending is not None:
                    raise ValueError("another runtime switch is already pending")
                self._pending = {
                    "request_id": request_id,
                    "target_mode": target_mode,
                    "map_name": map_name,
                }
        except Exception as exc:
            self._publish_status(
                "runtime_switch_error", error=f"{type(exc).__name__}: {exc}"
            )

    def _take_pending(self) -> dict | None:
        with self._lock:
            request = self._pending
            self._pending = None
            return request

    def _start_child(self, mode: str, map_name: str) -> None:
        command = build_launch_command(
            mode=mode, map_name=map_name, maps_root=self._maps_root
        )
        self.get_logger().info(f"starting Nav2 child in {mode} mode")
        self._child = subprocess.Popen(command, start_new_session=True)
        self._mode = mode
        self._map_name = map_name if mode == "localization" else ""

    def _stop_child(self) -> None:
        child = self._child
        self._child = None
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    return
                child.wait(timeout=5)

    def _switch(self, request: dict) -> None:
        previous_mode = self._mode
        previous_map = self._map_name
        target_mode = request["target_mode"]
        target_map = request["map_name"]
        request_id = request["request_id"]
        if target_mode == previous_mode and (
            target_mode != "localization" or target_map == previous_map
        ):
            return

        self._publish_status(
            "runtime_switching",
            request_id=request_id,
            previous_runtime_mode=previous_mode,
            target_runtime_mode=target_mode,
        )
        # Let the command response reach Perception before its publisher exits.
        time.sleep(0.5)
        self._stop_child()
        try:
            self._start_child(target_mode, target_map)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"runtime switch failed: {error}")
            try:
                self._start_child(previous_mode, previous_map)
            except Exception as rollback_exc:
                error += (
                    "; rollback failed: "
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
            self._publish_status(
                "runtime_switch_error",
                request_id=request_id,
                runtime_mode=self._mode,
                error=error,
            )

    def _run(self) -> None:
        try:
            self._start_child(self._mode, self._map_name)
            while not self._stopping:
                request = self._take_pending()
                if request is not None:
                    self._switch(request)
                child = self._child
                if child is not None and child.poll() is not None:
                    code = child.returncode
                    self._child = None
                    self._publish_status(
                        "runtime_child_exited",
                        runtime_mode=self._mode,
                        error=f"Nav2 child exited with code {code}",
                    )
                    if not self._stopping:
                        time.sleep(1.0)
                        self._start_child(self._mode, self._map_name)
                time.sleep(0.05)
        except Exception as exc:
            self._publish_status(
                "runtime_supervisor_error",
                runtime_mode=self._mode,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _publish_status(self, event: str, **fields) -> None:
        payload = {
            "event": event,
            "status": "error" if event.endswith("error") else "switching",
            "runtime_mode": self._mode,
            "shadow_only": True,
            "physical_execution": False,
            "timestamp": time.time(),
            **fields,
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._status_pub.publish(message)

    def close(self) -> None:
        self._stopping = True
        self._stop_child()
        self._worker.join(timeout=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RuntimeSupervisor()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.close()
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
