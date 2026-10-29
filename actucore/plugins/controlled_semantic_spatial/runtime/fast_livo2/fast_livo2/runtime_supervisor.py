"""Own the FAST-LIVO2 child process and its mapping session receipts."""

from __future__ import annotations

import logsafe

logsafe.install()

import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import threading
import time
import uuid

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .frame_adapter_core import normalize_obstacle_height_range
from .fault_diagnostics import (
    ADAPTER_DIAGNOSTICS_TOPIC,
    BRIDGE_STATUS_TOPIC,
    MAPPER_RUNTIME_TOPIC,
    SENSOR_REJECTION_TOPIC,
    write_fault_summary,
)
from .pipeline_diagnostics import PipelineDiagnostics
from .runtime_core import controlled_stop_succeeded


_STATUS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
_MAP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MAX_MAP_ARTIFACT_FILES = 64
_MAX_MAP_ARTIFACT_BYTES = 1_073_741_824
_MAX_MAP_ARTIFACT_TOTAL_BYTES = 536_870_912
_MAX_MAP_MANIFEST_BYTES = 65_536
_MAP_CONTROL_RESPONSE_GRACE_SEC = 5.0
_MAP_CONTROL_DISCOVERY_TIMEOUT_SEC = 5.0
_FAULT_CAPTURE_PRE_SEC = 5.0
_FAULT_CAPTURE_POST_SEC = 5.0
_FAULT_CAPTURE_CACHE_BYTES = 512 * 1024 * 1024
_FAULT_CAPTURE_ROOT = Path("/opt/phanthy-motus/data/navigation/fast_livo2/recordings/faults")
_FAULT_CAPTURE_QOS_PATH = Path(__file__).with_name("fault_capture_qos.yaml")
_DEFAULT_LIDAR_QOS_DEPTH = 2
_DEFAULT_IMU_QOS_DEPTH = 400
_RAW_IMU_PROPAGATED_ODOM_TOPIC = (
    "/ubuntu/navigation/fast_livo2/raw/imu_propagated_odom"
)


class FastLivo2Supervisor(Node):
    def __init__(self) -> None:
        super().__init__("fast_livo2_supervisor")
        self.declare_parameter("command_topic", "/ubuntu/navigation/fast_livo2/command")
        self.declare_parameter("status_topic", "/ubuntu/navigation/fast_livo2/status")
        self.declare_parameter("diagnostics_topic", "/ubuntu/navigation/fast_livo2/diagnostics")
        self.declare_parameter("bridge_status_topic", BRIDGE_STATUS_TOPIC)
        self.declare_parameter("mapper_runtime_topic", MAPPER_RUNTIME_TOPIC)
        self.declare_parameter("reset_topic", "/ubuntu/navigation/fast_livo2/reset_map")
        self.declare_parameter("map_control_topic", "/ubuntu/navigation/fast_livo2/map_control")
        self.declare_parameter("map_control_status_topic", "/ubuntu/navigation/fast_livo2/map_control_status")
        self.declare_parameter("lidar_topic", "/ubuntu/navigation/lidar")
        self.declare_parameter("imu_topic", "/ubuntu/navigation/imu")
        self.declare_parameter("odom_topic", "/ubuntu/navigation/odom")
        self.declare_parameter(
            "raw_odom_topic", "/ubuntu/navigation/fast_livo2/raw/odom"
        )
        self.declare_parameter(
            "raw_cloud_topic",
            "/ubuntu/navigation/fast_livo2/raw/cloud_registered",
        )
        self.declare_parameter("config_path", "/config/navigation_lio.yaml")
        self.declare_parameter("map_root", "/opt/fast_livo_ws/src/fast_livo/Log/pcd")
        self.declare_parameter("pcd_save_interval", 600)
        self.declare_parameter("stop_timeout_sec", 120.0)
        self.declare_parameter("map_control_timeout_sec", 130.0)

        self._lock = threading.RLock()
        self._runtime_lifecycle_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._active_map: str | None = None
        self._loaded_map: str | None = None
        self._pending_localization_stop_error: dict | None = None
        self._runtime_mode = "idle"
        self._started_unix_ms: int | None = None
        self._runtime_session_id: str | None = None
        self._lidar_qos_depth = _DEFAULT_LIDAR_QOS_DEPTH
        self._imu_qos_depth = _DEFAULT_IMU_QOS_DEPTH
        self._files_before: dict[str, tuple[int, int]] = {}
        self._pending_mapping_finalize: dict | None = None
        self._last_mapping_result: dict | None = None
        self._diagnostics: dict = {}
        self._diagnostics_monotonic: float | None = None
        self._pipeline_diagnostics = PipelineDiagnostics()
        self._map_control_responses: dict[str, dict] = {}
        self._pending_map_control_requests: set[str] = set()
        self._fault_capture_lock = threading.Lock()
        self._fault_capture_process: subprocess.Popen | None = None
        self._fault_capture_timer: threading.Timer | None = None
        self._fault_capture_directory: Path | None = None
        self._fault_capture_state = "starting"
        self._fault_capture_enabled = True
        self._fault_capture_error: str | None = None
        self._fault_capture_trigger: dict | None = None
        self._fault_capture_summary: str | None = None
        self._condition = threading.Condition(self._lock)
        self._map_root = Path(str(self.get_parameter("map_root").value))
        self._map_root.mkdir(parents=True, exist_ok=True)
        self._status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), _STATUS_QOS)
        self._reset_pub = self.create_publisher(String, str(self.get_parameter("reset_topic").value), 10)
        map_control_topic = str(self.get_parameter("map_control_topic").value)
        self._map_control_status_topic = str(
            self.get_parameter("map_control_status_topic").value
        )
        self._map_control_pub = self.create_publisher(
            String, map_control_topic, 10
        )
        callbacks = ReentrantCallbackGroup()
        self.create_subscription(
            String,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            10,
            callback_group=callbacks,
        )
        self.create_subscription(
            String,
            self._map_control_status_topic,
            self._on_map_control_status,
            10,
            callback_group=callbacks,
        )
        diagnostic_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("diagnostics_topic").value),
            self._on_diagnostics,
            diagnostic_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("bridge_status_topic").value),
            self._on_bridge_status,
            diagnostic_qos,
            callback_group=callbacks,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("mapper_runtime_topic").value),
            self._on_mapper_runtime,
            diagnostic_qos,
            callback_group=callbacks,
        )
        self.create_subscription(
            String,
            SENSOR_REJECTION_TOPIC,
            self._on_sensor_rejection,
            10,
            callback_group=callbacks,
        )
        self.create_timer(1.0, self._publish_heartbeat)
        self._arm_fault_capture()

    def _on_diagnostics(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(payload, dict):
            self._pipeline_diagnostics.observe("adapter", payload)
            with self._lock:
                self._diagnostics = payload
                self._diagnostics_monotonic = time.monotonic()

    def _observe_pipeline_json(self, layer: str, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        self._pipeline_diagnostics.observe(layer, payload)

    def _on_bridge_status(self, message: String) -> None:
        self._observe_pipeline_json("bridge", message)

    def _on_mapper_runtime(self, message: String) -> None:
        self._observe_pipeline_json("mapper", message)

    def _on_sensor_rejection(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if (
            isinstance(payload, dict)
            and payload.get("error_code") == "raw_odom_discontinuity"
        ):
            self._trigger_fault_capture(payload)

    def _fault_capture_command(self, output_directory: Path) -> list[str]:
        return [
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",
            "--output",
            str(output_directory),
            "--max-cache-size",
            str(_FAULT_CAPTURE_CACHE_BYTES),
            "--snapshot-mode",
            "--no-discovery",
            "--qos-profile-overrides-path",
            str(_FAULT_CAPTURE_QOS_PATH),
            str(self.get_parameter("lidar_topic").value),
            str(self.get_parameter("imu_topic").value),
            str(self.get_parameter("raw_odom_topic").value),
            _RAW_IMU_PROPAGATED_ODOM_TOPIC,
            SENSOR_REJECTION_TOPIC,
            str(self.get_parameter("bridge_status_topic").value),
            str(self.get_parameter("mapper_runtime_topic").value),
            str(self.get_parameter("diagnostics_topic").value),
        ]

    def _arm_fault_capture(self) -> None:
        if not getattr(self, "_fault_capture_enabled", True):
            with self._fault_capture_lock:
                self._fault_capture_state = "disabled"
            return
        try:
            _FAULT_CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
            session = (
                time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                + "-"
                + uuid.uuid4().hex[:8]
            )
            directory = _FAULT_CAPTURE_ROOT / session
            process = subprocess.Popen(
                self._fault_capture_command(directory),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                # Keep the recorder in the FAST-LIVO2 launch process group so
                # runtime stop/restart cannot leave a full-rate recorder behind.
                start_new_session=False,
            )
            time.sleep(0.25)
            if process.poll() is not None:
                raise RuntimeError(
                    f"ros2 bag snapshot recorder exited with {process.returncode}"
                )
        except (OSError, RuntimeError) as exc:
            with self._fault_capture_lock:
                self._fault_capture_state = "error"
                self._fault_capture_error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(
                f"cannot arm FAST-LIVO2 fault capture: {exc}"
            )
            return
        with self._fault_capture_lock:
            self._fault_capture_process = process
            self._fault_capture_directory = directory
            self._fault_capture_state = "armed"
            self._fault_capture_error = None
            self._fault_capture_summary = None

    def _trigger_fault_capture(self, event: dict) -> None:
        with self._fault_capture_lock:
            process = self._fault_capture_process
            if (
                self._fault_capture_state != "armed"
                or process is None
                or process.poll() is not None
            ):
                return
            self._fault_capture_state = "post_trigger"
            self._fault_capture_trigger = {
                "reason": "raw_odom_discontinuity",
                "detail": str(event.get("reason", event.get("detail", "")))[:500],
                "trigger_unix_ns": time.time_ns(),
            }
            timer = threading.Timer(
                _FAULT_CAPTURE_POST_SEC,
                self._save_fault_capture,
            )
            timer.daemon = True
            self._fault_capture_timer = timer
            timer.start()

    @staticmethod
    def _stop_fault_capture_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    def _save_fault_capture(self) -> None:
        with self._fault_capture_lock:
            if self._fault_capture_state != "post_trigger":
                return
            process = self._fault_capture_process
            directory = self._fault_capture_directory
            self._fault_capture_state = "saving"
        error = None
        summary_error = None
        summary_path = None
        try:
            if process is None or directory is None or process.poll() is not None:
                raise RuntimeError("snapshot recorder is not running")
            completed = subprocess.run(
                [
                    "ros2",
                    "service",
                    "call",
                    "/rosbag2_recorder/snapshot",
                    "rosbag2_interfaces/srv/Snapshot",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=10.0,
            )
            output = f"{completed.stdout}\n{completed.stderr}"
            if completed.returncode != 0 or not re.search(
                r"success\s*[=:]\s*true",
                output,
                re.IGNORECASE,
            ):
                raise RuntimeError(
                    "rosbag2 snapshot service failed: " + output.strip()[:500]
                )
            self._stop_fault_capture_process(process)
            if not (directory / "metadata.yaml").is_file() or not any(
                directory.glob("*.mcap")
            ):
                raise RuntimeError("snapshot did not produce an MCAP artifact")
            try:
                summary_path = write_fault_summary(
                    directory,
                    self._fault_capture_trigger,
                )
            except Exception as exc:  # The saved MCAP remains the primary receipt.
                summary_error = f"{type(exc).__name__}: {exc}"
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                try:
                    self._stop_fault_capture_process(process)
                except (OSError, subprocess.TimeoutExpired) as stop_exc:
                    error += f"; stop_failed:{stop_exc}"
        with self._fault_capture_lock:
            self._fault_capture_process = None
            self._fault_capture_timer = None
            self._fault_capture_state = "saved" if error is None else "error"
            self._fault_capture_error = error or summary_error
            self._fault_capture_summary = (
                None if summary_path is None else str(summary_path)
            )
        if error is None:
            self.get_logger().warning(
                f"saved first FAST-LIVO2 odom fault capture to {directory}"
            )
            if summary_error is not None:
                self.get_logger().error(
                    f"fault capture summary failed but MCAP was retained: {summary_error}"
                )
        else:
            self.get_logger().error(
                f"cannot save FAST-LIVO2 odom fault capture: {error}"
            )

    def _fault_capture_snapshot(self) -> dict:
        lock = getattr(self, "_fault_capture_lock", None)
        if lock is None:
            return {"state": "unavailable"}
        with lock:
            process = self._fault_capture_process
            if (
                self._fault_capture_state in {"armed", "post_trigger"}
                and process is not None
                and process.poll() is not None
            ):
                self._fault_capture_state = "error"
                self._fault_capture_error = (
                    "snapshot recorder exited with "
                    f"{process.returncode}"
                )
            return {
                "state": self._fault_capture_state,
                "directory": (
                    None
                    if self._fault_capture_directory is None
                    else str(self._fault_capture_directory)
                ),
                "pre_window_sec": _FAULT_CAPTURE_PRE_SEC,
                "post_window_sec": _FAULT_CAPTURE_POST_SEC,
                "cache_bytes": _FAULT_CAPTURE_CACHE_BYTES,
                "pid": (
                    process.pid
                    if process is not None and process.poll() is None
                    else None
                ),
                "trigger": (
                    None
                    if self._fault_capture_trigger is None
                    else dict(self._fault_capture_trigger)
                ),
                "error": self._fault_capture_error,
                "summary": self._fault_capture_summary,
            }

    def _discard_armed_fault_capture(self) -> None:
        with self._fault_capture_lock:
            timer = self._fault_capture_timer
            process = self._fault_capture_process
            directory = self._fault_capture_directory
            state = self._fault_capture_state
            if timer is not None:
                timer.cancel()
            self._fault_capture_timer = None
        if state == "saving" and timer is not None:
            timer.join(timeout=15.0)
            with self._fault_capture_lock:
                if self._fault_capture_state in {"saved", "error"}:
                    return
        if state == "post_trigger":
            self._save_fault_capture()
            return
        if process is not None:
            self._stop_fault_capture_process(process)
        if (
            state == "armed"
            and directory is not None
            and directory.parent.resolve() == _FAULT_CAPTURE_ROOT.resolve()
            and directory.exists()
        ):
            shutil.rmtree(directory)
        with self._fault_capture_lock:
            self._fault_capture_process = None
            if self._fault_capture_state == "armed":
                self._fault_capture_state = "discarded"

    def _on_map_control_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        request_id = payload.get("request_id")
        if payload.get("event") != "response" or not isinstance(request_id, str):
            return
        with self._condition:
            if request_id not in self._pending_map_control_requests:
                return
            self._map_control_responses[request_id] = payload
            self._condition.notify_all()

    def _on_command(self, message: String) -> None:
        try:
            request = json.loads(message.data)
            if not isinstance(request, dict):
                raise ValueError("command must be an object")
            request_id = str(request.get("request_id", ""))
            action = request.get("action")
            args = request.get("args") or {}
            if action in {
                "start_mapping",
                "stop_mapping",
                "load_map",
                "relocalize",
                "unload_map",
            }:
                if not self._runtime_lifecycle_lock.acquire(blocking=False):
                    result = {
                        "status": "error",
                        "error_code": "runtime_busy",
                        "error": "another FAST-LIVO2 lifecycle request is active",
                        "retryable": action == "stop_mapping",
                    }
                else:
                    try:
                        if action == "start_mapping":
                            result = self._start_mapping(str(args.get("map_name", "")))
                        elif action == "stop_mapping":
                            result = self._stop_mapping(
                                str(args.get("map_name", ""))
                            )
                        elif action == "load_map":
                            result = self._load_map(str(args.get("map_name", "")))
                        elif action == "relocalize":
                            result = self._relocalize(args)
                        else:
                            result = self._unload_map()
                    finally:
                        self._runtime_lifecycle_lock.release()
            elif action == "configure_obstacle_filter":
                if not self._runtime_lifecycle_lock.acquire(blocking=False):
                    result = {
                        "status": "error",
                        "error_code": "runtime_busy",
                        "error": "another FAST-LIVO2 lifecycle request is active",
                    }
                else:
                    try:
                        with self._lock:
                            runtime_mode = self._runtime_mode
                        if runtime_mode != "idle":
                            result = {
                                "status": "error",
                                "error_code": "runtime_active",
                                "error": (
                                    "obstacle height limits can change only while "
                                    "navigation mapping/localization is idle"
                                ),
                            }
                        else:
                            result = self._configure_runtime(args)
                    finally:
                        self._runtime_lifecycle_lock.release()
            else:
                result = {"status": "error", "error_code": "unsupported_action", "error": f"unsupported action {action}"}
        except Exception as exc:
            request_id = locals().get("request_id", "")
            action = locals().get("action", "")
            result = {"status": "error", "error_code": "supervisor_error", "error": f"{type(exc).__name__}: {exc}"}
        self._publish({"event": "response", "request_id": request_id, "action": action, **result})

    def _configure_runtime(self, args: dict) -> dict:
        depths = {}
        for key, default, maximum in (
            ("lidar_qos_depth", _DEFAULT_LIDAR_QOS_DEPTH, 32),
            ("imu_qos_depth", _DEFAULT_IMU_QOS_DEPTH, 4000),
        ):
            value = args.get(key, getattr(self, f"_{key}", default))
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                return {
                    "status": "error",
                    "error_code": "invalid_sensor_qos_config",
                    "error": f"{key} is outside the supported range",
                }
            depths[key] = value
        map_view_enabled = args.get("map_view_enabled", True)
        fault_capture_enabled = args.get(
            "fault_capture_enabled",
            getattr(self, "_fault_capture_enabled", True),
        )
        if not isinstance(map_view_enabled, bool) or not isinstance(
            fault_capture_enabled, bool
        ):
            return {
                "status": "error",
                "error_code": "invalid_runtime_feature_config",
                "error": "map_view_enabled and fault_capture_enabled must be booleans",
            }
        result = self._adapter_execute(
            "configure_obstacle_filter",
            {
                "min_height_m": args.get("min_height_m"),
                "max_height_m": args.get("max_height_m"),
                "map_view_enabled": map_view_enabled,
            },
        )
        if result.get("status") != "error":
            with self._lock:
                self._lidar_qos_depth = depths["lidar_qos_depth"]
                self._imu_qos_depth = depths["imu_qos_depth"]
            result["sensor_qos_depths"] = {
                "lidar": self._lidar_qos_depth,
                "imu": self._imu_qos_depth,
            }
            previous_fault_capture_enabled = getattr(
                self, "_fault_capture_enabled", True
            )
            self._fault_capture_enabled = fault_capture_enabled
            if previous_fault_capture_enabled and not fault_capture_enabled:
                self._discard_armed_fault_capture()
                with self._fault_capture_lock:
                    self._fault_capture_state = "disabled"
            elif not previous_fault_capture_enabled and fault_capture_enabled:
                self._arm_fault_capture()
            result["runtime_features"] = {
                "map_view_enabled": map_view_enabled,
                "fault_capture_enabled": fault_capture_enabled,
            }
        return result

    def _start_mapping(self, map_name: str) -> dict:
        with self._lock:
            if self._pending_mapping_finalize is not None:
                return {
                    "status": "error",
                    "error_code": "mapping_finalize_pending",
                    "error": "retry stop_mapping before starting another map",
                }
            if self._process is not None and self._process.poll() is None:
                return {"status": "error", "error_code": "mapping_active", "error": f"mapping {self._active_map} is already active"}
            if not _MAP_NAME_RE.fullmatch(map_name):
                return {
                    "status": "error",
                    "error_code": "invalid_argument",
                    "error": "map_name must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
                }
            self._files_before = self._pcd_files()
            self._diagnostics = {}
            self._diagnostics_monotonic = None
            self._runtime_session_id = None
            reset = String()
            reset.data = map_name
            self._reset_pub.publish(reset)
            self._process = subprocess.Popen(
                self._algorithm_command(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(0.25)
            return_code = self._process.poll()
            if return_code is not None:
                self._process = None
                return {"status": "error", "error_code": "algorithm_start_failed", "error": f"FAST-LIVO2 exited with {return_code}"}
            self._active_map = map_name
            self._loaded_map = None
            self._runtime_mode = "mapping"
            self._started_unix_ms = int(time.time() * 1000)
            self._runtime_session_id = uuid.uuid4().hex
            self._last_mapping_result = None
            return {"status": "mapping", "map_name": map_name, "pid": self._process.pid, "session_local": True}

    def _finish_mapping_runtime(
        self,
        process: subprocess.Popen,
        *,
        terminal_result: dict | None = None,
    ) -> None:
        with self._lock:
            if self._process is not process:
                return
            if terminal_result is not None and terminal_result.get("map_name"):
                terminal_result.setdefault("terminal_confirmed", True)
                self._last_mapping_result = dict(terminal_result)
            self._process = None
            self._active_map = None
            self._started_unix_ms = None
            self._runtime_session_id = None
            self._runtime_mode = "idle"
            self._pending_mapping_finalize = None

    def _stop_mapping(self, requested_map_name: str = "") -> dict:
        with self._lock:
            if self._runtime_mode == "localization":
                return {
                    "status": "error",
                    "error_code": "localization_active",
                    "error": f"map {self._loaded_map} is loaded for localization",
                }
            pending = self._pending_mapping_finalize
            process = self._process
            map_name = self._active_map
            owned_map_name = (
                str(pending.get("map_name", ""))
                if pending is not None
                else str(map_name or "")
            )
            if (
                requested_map_name
                and owned_map_name
                and requested_map_name != owned_map_name
            ):
                return {
                    "status": "error",
                    "error_code": "mapping_session_mismatch",
                    "error": (
                        f"stop request is for {requested_map_name}, but the "
                        f"active mapping session is {owned_map_name}"
                    ),
                    "map_name": owned_map_name,
                    "retryable": False,
                }
            started = self._started_unix_ms
            diagnostics = dict(self._diagnostics)
            diagnostics_monotonic = self._diagnostics_monotonic
            last_result = (
                None
                if self._last_mapping_result is None
                else dict(self._last_mapping_result)
            )
        if pending is None:
            if process is None:
                if (
                    last_result is not None
                    and requested_map_name
                    and last_result.get("map_name") == requested_map_name
                ):
                    return {**last_result, "already_finalized": True}
                return {
                    "status": "stopped",
                    "already_idle": True,
                    "map_name": map_name,
                }
            if process.poll() is not None:
                return_code = process.returncode
                files = self._changed_pcd_files()
                result = {
                    "status": "error",
                    "error_code": "algorithm_exited",
                    "error": f"FAST-LIVO2 exited unexpectedly with {return_code}",
                    "map_name": map_name,
                    "pcd_files": files,
                    "manifest": None,
                }
                self._finish_mapping_runtime(process, terminal_result=result)
                return result
            diagnostics_age = (
                None
                if diagnostics_monotonic is None
                else time.monotonic() - diagnostics_monotonic
            )
            point_count = diagnostics.get("map_point_count")
            diagnostics_current = (
                diagnostics_age is not None
                and diagnostics_age <= 2.0
                and diagnostics.get("session_name") == map_name
                and isinstance(point_count, int)
                and not isinstance(point_count, bool)
            )
            if (
                diagnostics_current
                and diagnostics.get("localization_state") == "mapping_error"
            ):
                stop_error = self._terminate_process(process)
                files = self._changed_pcd_files()
                result = {
                    "status": "error",
                    "error_code": "static_map_accumulation_failed",
                    "error": str(
                        diagnostics.get(
                            "static_map_error",
                            "static-map accumulation failed",
                        )
                    ),
                    "map_name": map_name,
                    "pcd_files": files,
                    "manifest": None,
                    "algorithm_stop_error": stop_error,
                }
                self._finish_mapping_runtime(process, terminal_result=result)
                return result
            diagnostics_ready = (
                diagnostics_current
                and diagnostics.get("localization_state") == "mapping"
            )
            if not diagnostics_ready or point_count < 40:
                warning_code = (
                    "static_map_not_ready"
                    if diagnostics_ready
                    else "static_map_status_unavailable"
                )
                warning = (
                    "confirmed static map has too few points to persist"
                    if diagnostics_ready
                    else "fresh static-map diagnostics were unavailable"
                )
                stop_error = self._terminate_process(process)
                result = {
                    "status": "stopped",
                    "map_name": map_name,
                    "map_saved": False,
                    "warning_code": warning_code,
                    "warning": warning,
                    "algorithm_stop_error": stop_error,
                }
                if diagnostics_ready:
                    result["static_point_count"] = point_count
                self._finish_mapping_runtime(process, terminal_result=result)
                return result
            stop_timeout_sec = float(
                self.get_parameter("stop_timeout_sec").value
            )
            stop_deadline = time.monotonic() + stop_timeout_sec
            flush_result = self._flush_raw_pcd(
                process,
                timeout_sec=max(0.1, stop_timeout_sec - 10.0),
            )
            if flush_result.get("status") != "flushed":
                result = {
                    **flush_result,
                    "map_name": map_name,
                    "pcd_files": self._changed_pcd_files(),
                    "manifest": None,
                }
                if process.poll() is not None:
                    self._finish_mapping_runtime(
                        process,
                        terminal_result=result,
                    )
                return result
            os.killpg(process.pid, signal.SIGINT)
            try:
                return_code = process.wait(
                    timeout=max(0.1, stop_deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                result = {
                    "status": "error",
                    "error_code": "algorithm_stop_timeout",
                    "error": "FAST-LIVO2 did not stop within timeout",
                    "map_name": map_name,
                }
                self._finish_mapping_runtime(process, terminal_result=result)
                return result
            files = self._changed_pcd_files()
            if not controlled_stop_succeeded(return_code):
                result = {
                    "status": "error",
                    "error_code": "algorithm_stop_failed",
                    "error": f"FAST-LIVO2 exited with {return_code}",
                    "map_name": map_name,
                    "pcd_files": files,
                }
                self._finish_mapping_runtime(process, terminal_result=result)
                return result
            if not files:
                result = {
                    "status": "error",
                    "error_code": "map_artifact_missing",
                    "error": "FAST-LIVO2 stopped without a new raw PCD artifact",
                    "map_name": map_name,
                    "pcd_files": [],
                    "manifest": None,
                }
                self._finish_mapping_runtime(process, terminal_result=result)
                return result
            pending = {
                "process": process,
                "map_name": map_name,
                "started_unix_ms": started,
                "return_code": return_code,
                "raw_pcd_files": list(files),
                "pcd_files": None,
                "static_result": None,
            }
            with self._lock:
                if self._process is not process:
                    return {
                        "status": "error",
                        "error_code": "mapping_session_changed",
                        "error": "mapping session changed while stopping",
                    }
                self._pending_mapping_finalize = pending
                self._runtime_mode = "finalizing"
        process = pending["process"]
        map_name = pending["map_name"]
        started = pending["started_unix_ms"]
        return_code = pending["return_code"]
        snapshotted = pending.get("pcd_files")
        if snapshotted is None:
            try:
                snapshotted = self._snapshot_session_pcd_files(
                    map_name,
                    started,
                    list(pending["raw_pcd_files"]),
                )
            except OSError as exc:
                return {
                    "status": "error",
                    "error_code": "map_artifact_snapshot_failed",
                    "error": str(exc),
                    "map_name": map_name,
                    "pcd_files": [],
                    "manifest": None,
                    "retryable": True,
                }
            except ValueError as exc:
                result = {
                    "status": "error",
                    "error_code": "map_artifact_snapshot_failed",
                    "error": str(exc),
                    "map_name": map_name,
                    "pcd_files": [],
                    "manifest": None,
                    "retryable": False,
                }
                self._finish_mapping_runtime(process, terminal_result=result)
                return result
            with self._lock:
                if self._pending_mapping_finalize is pending:
                    pending["pcd_files"] = list(snapshotted)
        files = list(snapshotted)
        # save_static_map changes the adapter to finalizing while holding its
        # map lock, so the snapshot cannot race a later registered-cloud callback.
        static_result = pending.get("static_result")
        if static_result is None:
            static_result = self._adapter_execute(
                "save_static_map", {"map_name": map_name}
            )
            if static_result.get("status") == "error":
                retryable = (
                    static_result.get("retryable") is True
                    or static_result.get("error_code") == "map_control_timeout"
                )
                result = {
                    "status": "error",
                    "error_code": "static_map_save_failed",
                    "error": static_result.get(
                        "error", "adapter did not persist the confirmed static map"
                    ),
                    "map_name": map_name,
                    "pcd_files": files,
                    "manifest": None,
                    "retryable": retryable,
                }
                if not retryable:
                    self._cleanup_session_artifacts(files, None)
                    self._finish_mapping_runtime(
                        process,
                        terminal_result=result,
                    )
                return result
            with self._lock:
                if self._pending_mapping_finalize is pending:
                    pending["static_result"] = dict(static_result)
        try:
            manifest = self._write_manifest(
                map_name, started, return_code, files, static_result
            )
        except OSError as exc:
            return {
                "status": "error",
                "error_code": "manifest_write_failed",
                "error": str(exc),
                "map_name": map_name,
                "pcd_files": files,
                "static_map_pcd": static_result.get("static_map_pcd"),
                "manifest": None,
                "retryable": True,
            }
        except (KeyError, TypeError, ValueError) as exc:
            result = {
                "status": "error",
                "error_code": "manifest_write_failed",
                "error": str(exc),
                "map_name": map_name,
                "pcd_files": files,
                "static_map_pcd": static_result.get("static_map_pcd"),
                "manifest": None,
                "retryable": False,
            }
            self._cleanup_session_artifacts(files, static_result)
            self._finish_mapping_runtime(process, terminal_result=result)
            return result
        receipt = {
            "status": "saved",
            "map_name": map_name,
            "pcd_files": files,
            "manifest": manifest,
            "static_map_pcd": static_result.get("static_map_pcd"),
            "static_point_count": static_result.get("static_point_count"),
            "algorithm_return_code": return_code,
            "controlled_stop": True,
            "global_relocalization_supported": False,
            "bounded_relocalization_supported": True,
        }
        self._finish_mapping_runtime(process, terminal_result=receipt)
        return receipt

    def _load_map(self, map_name: str) -> dict:
        with self._lock:
            if self._pending_mapping_finalize is not None:
                return {
                    "status": "error",
                    "error_code": "mapping_finalize_pending",
                    "error": "retry stop_mapping before loading a map",
                }
            process_running = (
                self._process is not None and self._process.poll() is None
            )
            runtime_mode = self._runtime_mode
            previous_map = self._loaded_map
            if process_running and runtime_mode != "localization":
                return {
                    "status": "error",
                    "error_code": "runtime_active",
                    "error": f"FAST-LIVO2 {runtime_mode} runtime is already active",
                }
        try:
            artifacts = self._map_artifacts_from_manifest(map_name)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "error",
                "error_code": "map_artifact_invalid",
                "error": str(exc),
                "loaded_map": previous_map,
                "runtime_mode": runtime_mode,
            }
        validation_files, validation_static, validation_height = artifacts
        validation_args = {
            "map_name": map_name,
            "pcd_files": [str(path) for path in validation_files],
        }
        if validation_static is not None:
            validation_args["static_map_pcd"] = str(validation_static)
            validation_args["obstacle_height_range_m"] = list(
                validation_height or ()
            )
        validation = self._adapter_execute("validate_map", validation_args)
        if validation.get("status") == "error":
            return {
                **validation,
                "error_code": "map_artifact_invalid",
                "loaded_map": previous_map,
                "runtime_mode": runtime_mode,
            }
        previous_artifacts = None
        if runtime_mode == "localization" and previous_map:
            try:
                previous_artifacts = self._map_artifacts_from_manifest(previous_map)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return {
                    "status": "error",
                    "error_code": "active_map_artifact_invalid",
                    "error": f"cannot replace active map {previous_map}: {exc}",
                    "loaded_map": previous_map,
                    "runtime_mode": runtime_mode,
                }
            previous_files, previous_static, previous_height = previous_artifacts
            previous_args = {
                "map_name": previous_map,
                "pcd_files": [str(path) for path in previous_files],
            }
            if previous_static is not None:
                previous_args["static_map_pcd"] = str(previous_static)
                previous_args["obstacle_height_range_m"] = list(
                    previous_height or ()
                )
            previous_validation = self._adapter_execute(
                "validate_map",
                previous_args,
            )
            if previous_validation.get("status") == "error":
                return {
                    **previous_validation,
                    "error_code": "active_map_artifact_invalid",
                    "error": (
                        f"cannot replace active map {previous_map}: "
                        f"{previous_validation.get('error', 'validation failed')}"
                    ),
                    "loaded_map": previous_map,
                    "runtime_mode": runtime_mode,
                }
        if runtime_mode == "localization":
            unloaded = self._unload_map()
            if unloaded.get("status") == "error":
                rollback = (
                    self._activate_localization(previous_map, previous_artifacts)
                    if previous_map and previous_artifacts
                    else None
                )
                with self._lock:
                    actual_map = self._loaded_map
                    actual_mode = self._runtime_mode
                return {
                    **unloaded,
                    "loaded_map": actual_map,
                    "runtime_mode": actual_mode,
                    "replaced_map": previous_map,
                    "rollback_status": (
                        "restored"
                        if rollback and rollback.get("status") != "error"
                        else "failed"
                    ),
                    "rollback_error": (
                        rollback.get("error")
                        if rollback and rollback.get("status") == "error"
                        else None
                    ),
                }
        loaded = self._activate_localization(map_name, artifacts)
        if loaded.get("status") == "error":
            rollback = None
            if previous_map and previous_artifacts:
                rollback = self._activate_localization(previous_map, previous_artifacts)
            with self._lock:
                actual_map = self._loaded_map
                actual_mode = self._runtime_mode
            return {
                **loaded,
                "loaded_map": actual_map,
                "runtime_mode": actual_mode,
                "replaced_map": previous_map,
                "rollback_status": (
                    "not_needed"
                    if previous_map is None
                    else "restored"
                    if rollback and rollback.get("status") != "error"
                    else "failed"
                ),
                "rollback_error": (
                    rollback.get("error") if rollback and rollback.get("status") == "error" else None
                ),
            }
        loaded["replaced_map"] = previous_map
        return loaded

    def _activate_localization(
        self,
        map_name: str,
        artifacts: tuple[
            tuple[Path, ...],
            Path | None,
            tuple[float, float] | None,
        ],
    ) -> dict:
        files, static_map, obstacle_height_range = artifacts
        args = {"map_name": map_name, "pcd_files": [str(path) for path in files]}
        if static_map is not None:
            args["static_map_pcd"] = str(static_map)
            args["obstacle_height_range_m"] = list(obstacle_height_range or ())
        loaded = self._adapter_execute(
            "load_map", args
        )
        if loaded.get("status") == "error":
            return loaded
        try:
            process = subprocess.Popen(
                self._algorithm_command(save_pcd=False),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            cleanup = self._adapter_execute("unload_map", {})
            if cleanup.get("status") == "error":
                with self._lock:
                    self._process = None
                    self._active_map = None
                    self._loaded_map = map_name
                    self._runtime_mode = "localization"
                    self._started_unix_ms = None
                    self._runtime_session_id = None
                return {
                    **cleanup,
                    "error_code": "algorithm_start_cleanup_pending",
                    "error": (
                        f"cannot start FAST-LIVO2: {exc}; adapter cleanup is "
                        f"pending: {cleanup.get('error', 'unknown cleanup error')}"
                    ),
                    "loaded_map": map_name,
                    "runtime_mode": "localization",
                    "retryable": True,
                }
            return {
                "status": "error",
                "error_code": "algorithm_start_failed",
                "error": f"cannot start FAST-LIVO2: {exc}",
            }
        time.sleep(0.25)
        return_code = process.poll()
        if return_code is not None:
            cleanup = self._adapter_execute("unload_map", {})
            if cleanup.get("status") == "error":
                with self._lock:
                    self._process = None
                    self._active_map = None
                    self._loaded_map = map_name
                    self._runtime_mode = "localization"
                    self._started_unix_ms = None
                    self._runtime_session_id = None
                return {
                    **cleanup,
                    "error_code": "algorithm_start_cleanup_pending",
                    "error": (
                        f"FAST-LIVO2 exited with {return_code}; adapter cleanup "
                        f"is pending: {cleanup.get('error', 'unknown cleanup error')}"
                    ),
                    "loaded_map": map_name,
                    "runtime_mode": "localization",
                    "retryable": True,
                }
            return {
                "status": "error",
                "error_code": "algorithm_start_failed",
                "error": f"FAST-LIVO2 exited with {return_code}",
            }
        with self._lock:
            self._process = process
            self._active_map = None
            self._loaded_map = map_name
            self._runtime_mode = "localization"
            self._started_unix_ms = int(time.time() * 1000)
            self._runtime_session_id = uuid.uuid4().hex
        return {
            **loaded,
            "pid": process.pid,
            "runtime_mode": "localization",
            "loaded_map": map_name,
            "bounded_relocalization_supported": True,
        }

    def _relocalize(self, args: dict) -> dict:
        with self._lock:
            process = self._process
            map_name = self._loaded_map
            mode = self._runtime_mode
        if mode != "localization" or process is None or process.poll() is not None:
            return {
                "status": "error",
                "error_code": "localization_not_active",
                "error": "load a saved map before relocalizing",
            }
        request = dict(args)
        request["map_name"] = map_name
        return self._adapter_execute("relocalize", request)

    def _unload_map(self) -> dict:
        with self._lock:
            process = self._process
            map_name = self._loaded_map
            mode = self._runtime_mode
        if mode != "localization":
            return {"status": "idle", "already_idle": True, "map_name": map_name}
        stop_error = self._terminate_process(process)
        with self._lock:
            if stop_error is not None:
                self._pending_localization_stop_error = dict(stop_error)
            stop_error = self._pending_localization_stop_error
        adapter_result = self._adapter_execute("unload_map", {})
        if adapter_result.get("status") == "error":
            return {
                **adapter_result,
                "loaded_map": map_name,
                "runtime_mode": "localization",
                "algorithm_running": False,
                "algorithm_stop_error": stop_error,
            }
        with self._lock:
            self._process = None
            self._loaded_map = None
            self._runtime_mode = "idle"
            self._started_unix_ms = None
            self._runtime_session_id = None
            self._pending_localization_stop_error = None
        if stop_error is not None:
            return {
                **stop_error,
                "loaded_map": None,
                "runtime_mode": "idle",
                "retryable": False,
                "terminal_confirmed": True,
            }
        return {
            **adapter_result,
            "loaded_map": None,
            "runtime_mode": "idle",
        }

    def _terminate_process(self, process: subprocess.Popen | None) -> dict | None:
        if process is None or process.poll() is not None:
            return None
        os.killpg(process.pid, signal.SIGINT)
        try:
            return_code = process.wait(
                timeout=float(self.get_parameter("stop_timeout_sec").value)
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            return {
                "status": "error",
                "error_code": "algorithm_stop_timeout",
                "error": "FAST-LIVO2 did not stop within timeout",
            }
        if not controlled_stop_succeeded(return_code):
            return {
                "status": "error",
                "error_code": "algorithm_stop_failed",
                "error": f"FAST-LIVO2 exited with {return_code}",
            }
        return None

    def _map_artifacts_from_manifest(
        self, map_name: str
    ) -> tuple[
        tuple[Path, ...],
        Path | None,
        tuple[float, float] | None,
    ]:
        if not _MAP_NAME_RE.fullmatch(map_name):
            raise ValueError("map_name must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
        manifest_path = self._map_root / "sessions" / f"{map_name}.json"
        manifest_size = manifest_path.stat().st_size
        if manifest_size > _MAX_MAP_MANIFEST_BYTES:
            raise ValueError("map manifest exceeds byte safety limit")
        with manifest_path.open("rb") as stream:
            manifest_bytes = stream.read(_MAX_MAP_MANIFEST_BYTES + 1)
        if len(manifest_bytes) > _MAX_MAP_MANIFEST_BYTES:
            raise ValueError("map manifest exceeds byte safety limit")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if manifest.get("schema") != "phanthy.navigation.fast_livo2_map_session.v1":
            raise ValueError("map manifest schema is unsupported")
        if manifest.get("map_name") != map_name:
            raise ValueError("map manifest name does not match request")
        static_format_version = manifest.get("static_map_format_version")
        if static_format_version is not None and static_format_version != 2:
            raise ValueError("map manifest static-map format is unsupported")
        names = manifest.get("pcd_files")
        if not isinstance(names, list) or not names:
            raise ValueError("map manifest has no PCD files")
        if len(names) > _MAX_MAP_ARTIFACT_FILES:
            raise ValueError("map manifest has too many PCD files")
        root = self._map_root.resolve()
        files = []
        total_bytes = 0
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pcd"):
                raise ValueError("map manifest contains an unsafe PCD path")
            path = (root / name).resolve()
            if path.parent != root or not path.is_file():
                raise ValueError(f"map PCD is missing: {name}")
            file_size = path.stat().st_size
            if file_size > _MAX_MAP_ARTIFACT_BYTES:
                raise ValueError(f"map PCD exceeds byte safety limit: {name}")
            total_bytes += file_size
            if total_bytes > _MAX_MAP_ARTIFACT_TOTAL_BYTES:
                raise ValueError("map PCD files exceed aggregate byte safety limit")
            files.append(path)
        static_name = manifest.get("static_map_pcd")
        if static_format_version == 2 and static_name is None:
            raise ValueError("v2 map manifest has no confirmed static PCD")
        static_path = None
        obstacle_height_range = None
        if static_name is not None:
            if not isinstance(static_name, str):
                raise ValueError("map manifest static_map_pcd must be a path")
            relative = Path(static_name)
            if (
                len(relative.parts) != 2
                or relative.parts[0] != "static"
                or Path(relative.parts[1]).name != relative.parts[1]
                or not relative.parts[1].endswith(".static.pcd")
            ):
                raise ValueError("map manifest contains an unsafe static PCD path")
            static_path = (root / relative).resolve()
            static_root = (root / "static").resolve()
            if static_path.parent != static_root or not static_path.is_file():
                raise ValueError(f"map static PCD is missing: {static_name}")
            static_size = static_path.stat().st_size
            if static_size > _MAX_MAP_ARTIFACT_BYTES:
                raise ValueError(
                    f"map static PCD exceeds byte safety limit: {static_name}"
                )
            total_bytes += static_size
            if total_bytes > _MAX_MAP_ARTIFACT_TOTAL_BYTES:
                raise ValueError(
                    "map PCD files exceed aggregate byte safety limit"
                )
            obstacle_height_range = normalize_obstacle_height_range(
                manifest.get("obstacle_height_range_m"),
                field_name="map manifest obstacle_height_range_m",
            )
        return tuple(files), static_path, obstacle_height_range

    def _adapter_execute(self, action: str, args: dict) -> dict:
        if not self._wait_for_adapter_bridge(
            _MAP_CONTROL_DISCOVERY_TIMEOUT_SEC
        ):
            return {
                "status": "error",
                "error_code": "fast_livo2_adapter_unavailable",
                "error": (
                    "FAST-LIVO2 adapter command/response topics were not "
                    "discovered"
                ),
                "retryable": True,
            }
        request_id = f"map-{time.time_ns()}"
        timeout = float(self.get_parameter("map_control_timeout_sec").value)
        operation_deadline = time.monotonic() + max(
            0.1,
            timeout - _MAP_CONTROL_RESPONSE_GRACE_SEC,
        )
        message = String()
        message.data = json.dumps(
            {
                "request_id": request_id,
                "action": action,
                "args": args,
                "operation_deadline_monotonic": operation_deadline,
            },
            separators=(",", ":"),
        )
        with self._condition:
            self._pending_map_control_requests.add(request_id)
            self._map_control_pub.publish(message)
            deadline = time.monotonic() + timeout
            try:
                while request_id not in self._map_control_responses:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {
                            "status": "error",
                            "error_code": "map_control_timeout",
                            "error": f"FAST-LIVO2 adapter did not answer {action}",
                            "retryable": True,
                        }
                    self._condition.wait(timeout=remaining)
                response = self._map_control_responses.pop(request_id)
            finally:
                self._pending_map_control_requests.discard(request_id)
        return {key: value for key, value in response.items() if key not in {"event", "request_id", "action"}}

    def _wait_for_adapter_bridge(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            try:
                command_ready = self._map_control_pub.get_subscription_count() >= 1
                response_ready = (
                    self.count_publishers(self._map_control_status_topic) >= 1
                )
            except Exception:
                command_ready = False
                response_ready = False
            if command_ready and response_ready:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _algorithm_command(self, *, save_pcd: bool = True) -> list[str]:
        interval = int(self.get_parameter("pcd_save_interval").value)
        config = str(self.get_parameter("config_path").value)
        return [
            "/opt/fast_livo_ws/install/fast_livo/lib/fast_livo/fastlivo_mapping",
            "--ros-args",
            "--params-file", config,
            "-p", f"lidar_qos_depth:={self._lidar_qos_depth}",
            "-p", f"imu_qos_depth:={self._imu_qos_depth}",
            "-p", (
                "runtime_diagnostics_topic:="
                + str(self.get_parameter("mapper_runtime_topic").value)
            ),
            "-p", f"pcd_save.pcd_save_en:={'true' if save_pcd else 'false'}",
            "-p", f"pcd_save.interval:={interval}",
            "-p", "pcd_save.type:=0",
            "--log-level", "warn",
            "-r", f"/livox/lidar:={self.get_parameter('lidar_topic').value}",
            "-r", f"/livox/imu:={self.get_parameter('imu_topic').value}",
            "-r", "/left_camera/image:=/ubuntu/navigation/camera_disabled",
            "-r", f"/cloud_registered:={self.get_parameter('raw_cloud_topic').value}",
            "-r", f"/aft_mapped_to_init:={self.get_parameter('raw_odom_topic').value}",
            "-r", "/path:=/ubuntu/navigation/fast_livo2/raw/path",
            "-r", "/Laser_map:=/ubuntu/navigation/fast_livo2/raw/laser_map",
            "-r", "/LIVO2/imu_propagate:=/ubuntu/navigation/fast_livo2/raw/imu_propagated_odom",
            "-r", "/cloud_visual_sub_map_before:=/ubuntu/navigation/fast_livo2/raw/debug/cloud_visual_sub_map",
            "-r", "/cloud_effected:=/ubuntu/navigation/fast_livo2/raw/debug/cloud_effected",
            "-r", "/visualization_marker:=/ubuntu/navigation/fast_livo2/raw/debug/visualization_marker",
            "-r", "/planner_normal:=/ubuntu/navigation/fast_livo2/raw/debug/planner_normal",
            "-r", "/voxels:=/ubuntu/navigation/fast_livo2/raw/debug/voxels",
            "-r", "/planes:=/ubuntu/navigation/fast_livo2/raw/debug/planes",
            "-r", "/dyn_obj:=/ubuntu/navigation/fast_livo2/raw/debug/dyn_obj",
            "-r", "/dyn_obj_removed:=/ubuntu/navigation/fast_livo2/raw/debug/dyn_obj_removed",
            "-r", "/dyn_obj_dbg_hist:=/ubuntu/navigation/fast_livo2/raw/debug/dyn_obj_history",
            "-r", "/mavros/vision_pose/pose:=/ubuntu/navigation/fast_livo2/raw/debug/vision_pose",
            "-r", "/rgb_img:=/ubuntu/navigation/fast_livo2/raw/debug/rgb_disabled",
            "-r", "/tf:=/ubuntu/navigation/fast_livo2/raw/tf",
            "-r", "/tf_static:=/ubuntu/navigation/fast_livo2/raw/tf_static",
        ]

    def _flush_raw_pcd(
        self,
        process: subprocess.Popen,
        *,
        timeout_sec: float,
    ) -> dict:
        sequence = time.monotonic_ns() & ((1 << 63) - 1)
        try:
            completed = subprocess.run(
                [
                    "ros2",
                    "param",
                    "set",
                    "--no-daemon",
                    "--spin-time",
                    "5.0",
                    "/laserMapping",
                    "pcd_save.flush_sequence",
                    str(sequence),
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=max(0.1, float(timeout_sec)),
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error_code": "map_artifact_flush_timeout",
                "error": "timed out while flushing FAST-LIVO2 raw PCD",
                "retryable": process.poll() is None,
            }
        except OSError as exc:
            return {
                "status": "error",
                "error_code": "map_artifact_flush_failed",
                "error": f"cannot request FAST-LIVO2 raw PCD flush: {exc}",
                "retryable": process.poll() is None,
            }
        files = self._changed_pcd_files()
        if completed.returncode == 0 and files:
            return {"status": "flushed", "pcd_files": files}
        detail = (completed.stderr or completed.stdout or "").strip()
        return {
            "status": "error",
            "error_code": "map_artifact_flush_failed",
            "error": (
                "FAST-LIVO2 raw PCD flush failed"
                + (f": {detail[:512]}" if detail else "")
            ),
            "retryable": process.poll() is None,
        }

    def _pcd_files(self) -> dict[str, tuple[int, int]]:
        files = {}
        for path in self._map_root.glob("*.pcd"):
            if path.is_file():
                stat = path.stat()
                files[path.name] = (stat.st_size, stat.st_mtime_ns)
        return files

    def _changed_pcd_files(self) -> list[str]:
        current = self._pcd_files()
        return sorted(
            name for name, signature in current.items()
            if self._files_before.get(name) != signature
        )

    def _snapshot_session_pcd_files(
        self,
        map_name: str,
        started_unix_ms: int | None,
        files: list[str],
    ) -> list[str]:
        if not _MAP_NAME_RE.fullmatch(map_name):
            raise ValueError("cannot snapshot artifacts for an unsafe map name")
        if not 1 <= len(files) <= _MAX_MAP_ARTIFACT_FILES:
            raise ValueError(
                "raw map artifact count must be between 1 and "
                f"{_MAX_MAP_ARTIFACT_FILES}"
            )
        root = self._map_root.resolve()
        session_id = f"{int(started_unix_ms or 0)}-{uuid.uuid4().hex[:12]}"
        sources: list[Path] = []
        total_bytes = 0
        for name in files:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("raw map artifact contains an unsafe path")
            source = (root / name).resolve()
            if source.parent != root or not source.is_file():
                raise ValueError(f"raw map artifact is missing: {name}")
            file_size = source.stat().st_size
            if file_size > _MAX_MAP_ARTIFACT_BYTES:
                raise ValueError(
                    f"raw map artifact exceeds byte safety limit: {name}"
                )
            total_bytes += file_size
            if total_bytes > _MAX_MAP_ARTIFACT_TOTAL_BYTES:
                raise ValueError(
                    "raw map artifacts exceed aggregate byte safety limit"
                )
            sources.append(source)
        snapshots: list[str] = []
        created: list[Path] = []
        temporaries: list[Path] = []
        try:
            for index, source in enumerate(sources):
                snapshot_name = (
                    f"{map_name}-{session_id}-{index:02d}.raw.pcd"
                )
                destination = (root / snapshot_name).resolve()
                temporary = destination.with_name(destination.name + ".tmp")
                if destination.parent != root or destination.exists():
                    raise ValueError("session-owned raw map path is not available")
                temporaries.append(temporary)
                with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                os.replace(temporary, destination)
                temporaries.remove(temporary)
                created.append(destination)
                snapshots.append(snapshot_name)
        except (OSError, ValueError):
            for artifact in [*temporaries, *created]:
                try:
                    artifact.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return snapshots

    def _cleanup_session_artifacts(
        self,
        files: list[str],
        static_result: dict | None,
    ) -> None:
        """Best-effort cleanup for artifacts that can never receive a manifest."""

        root = self._map_root.resolve()
        candidates: list[Path] = []
        for name in files:
            if isinstance(name, str) and Path(name).name == name:
                path = (root / name).resolve()
                if path.parent == root:
                    candidates.append(path)
        if static_result is not None:
            static_name = static_result.get("static_map_file")
            if isinstance(static_name, str) and Path(static_name).name == static_name:
                static_root = (root / "static").resolve()
                static_path = (static_root / static_name).resolve()
                if static_path.parent == static_root:
                    candidates.append(static_path)
        for artifact in candidates:
            try:
                artifact.unlink(missing_ok=True)
            except OSError as exc:
                self.get_logger().warning(
                    f"cannot remove uncommitted map artifact {artifact.name}: {exc}"
                )

    def _write_manifest(
        self, map_name, started, return_code, files, static_result
    ) -> str | None:
        if not map_name:
            return None
        if not 1 <= len(files) <= _MAX_MAP_ARTIFACT_FILES:
            raise ValueError(
                "map manifest PCD count must be between 1 and "
                f"{_MAX_MAP_ARTIFACT_FILES}"
            )
        root = self._map_root.resolve()
        total_bytes = 0
        for name in files:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("map manifest contains an unsafe PCD path")
            artifact = (root / name).resolve()
            if artifact.parent != root or not artifact.is_file():
                raise ValueError(f"map PCD is missing: {name}")
            file_size = artifact.stat().st_size
            if file_size > _MAX_MAP_ARTIFACT_BYTES:
                raise ValueError(f"map PCD exceeds byte safety limit: {name}")
            total_bytes += file_size
            if total_bytes > _MAX_MAP_ARTIFACT_TOTAL_BYTES:
                raise ValueError("map PCD files exceed aggregate byte safety limit")
        directory = self._map_root / "sessions"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{map_name}.json"
        temporary = path.with_suffix(".json.tmp")
        static_file = str(static_result["static_map_file"])
        static_path = (root / "static" / static_file).resolve()
        static_root = (root / "static").resolve()
        if static_path.parent != static_root or not static_path.is_file():
            raise ValueError(f"map static PCD is missing: static/{static_file}")
        static_size = static_path.stat().st_size
        if static_size > _MAX_MAP_ARTIFACT_BYTES:
            raise ValueError(
                f"map static PCD exceeds byte safety limit: static/{static_file}"
            )
        total_bytes += static_size
        if total_bytes > _MAX_MAP_ARTIFACT_TOTAL_BYTES:
            raise ValueError("map PCD files exceed aggregate byte safety limit")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "phanthy.navigation.fast_livo2_map_session.v1",
                    "static_map_format_version": 2,
                    "map_name": map_name,
                    "started_unix_ms": started,
                    "stopped_unix_ms": int(time.time() * 1000),
                    "return_code": return_code,
                    "pcd_files": files,
                    "static_map_pcd": f"static/{static_file}",
                    "static_point_count": int(static_result["static_point_count"]),
                    "static_map_filter": str(static_result["temporal_filter"]),
                    "obstacle_height_range_m": list(
                        static_result["obstacle_height_range_m"]
                    ),
                    "frame": "session-local map",
                    "global_relocalization_supported": False,
                    "bounded_relocalization_supported": True,
                    "relocalization_requires_initial_guess": True,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return str(path)

    def _publish_heartbeat(self) -> None:
        fault_capture = self._fault_capture_snapshot()
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            runtime_state = (
                str(self._diagnostics.get("localization_state", "localization"))
                if running and self._runtime_mode == "localization"
                else "mapping"
                if running
                else "finalizing"
                if self._runtime_mode == "finalizing"
                else "idle"
            )
            diagnostics = dict(self._diagnostics)
            pipeline = getattr(self, "_pipeline_diagnostics", None)
            if pipeline is None:
                pipeline = self._pipeline_diagnostics = PipelineDiagnostics()
            pipeline_diagnostics = pipeline.snapshot(
                algorithm_running=running
            )
            payload = {
                "event": "heartbeat",
                "schema": "phanthy.navigation.fast_livo2_status.v1",
                "state": runtime_state,
                "status": runtime_state,
                "active_map": self._active_map,
                "loaded_map": self._loaded_map,
                "map_session_id": self._runtime_session_id,
                "runtime_mode": self._runtime_mode,
                "sensor_qos_depths": {
                    "lidar": self._lidar_qos_depth,
                    "imu": self._imu_qos_depth,
                },
                "algorithm_running": running,
                "companion_ready": bool(diagnostics.get("ready", False)),
                "sensor_frame": diagnostics.get("sensor_frame"),
                "sensor_contract_ready": bool(
                    diagnostics.get("sensor_contract_ready", False)
                ),
                "sensor_contract_issues": list(
                    diagnostics.get("sensor_contract_issues", [])
                ),
                "base_to_sensor_tf_ready": bool(
                    diagnostics.get("base_to_sensor_tf_ready", False)
                ),
                "point_time_span_ms": diagnostics.get("point_time_span_ms"),
                "odom_health": diagnostics.get("odom_health", {}),
                "readiness_blockers": list(
                    diagnostics.get("readiness_blockers", [])
                ),
                "session_local": True,
                "global_relocalization_supported": False,
                "bounded_relocalization_supported": True,
                "diagnostics": diagnostics,
                "pipeline_diagnostics": pipeline_diagnostics,
                "fault_capture": fault_capture,
            }
        self._publish(payload)

    def _publish(self, payload: dict) -> None:
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._status_pub.publish(message)

    def destroy_node(self):
        try:
            with self._runtime_lifecycle_lock:
                with self._lock:
                    process = self._process
                    pending = self._pending_mapping_finalize
                    active_map = self._active_map
                stop_error = self._terminate_process(process)
                if stop_error is not None:
                    self.get_logger().error(stop_error["error"])
                if pending is not None and pending.get("static_result") is not None:
                    try:
                        self._write_manifest(
                            pending["map_name"],
                            pending["started_unix_ms"],
                            pending["return_code"],
                            pending["pcd_files"],
                            pending["static_result"],
                        )
                    except (OSError, KeyError, TypeError, ValueError) as exc:
                        self.get_logger().error(
                            f"cannot finish pending map manifest during shutdown: {exc}"
                        )
                elif active_map is not None:
                    self.get_logger().error(
                        "mapping stopped by process shutdown without a confirmed "
                        "static-map manifest; use stop_mapping before stopping Canvas"
                    )
                with self._lock:
                    self._process = None
                    self._active_map = None
                    self._loaded_map = None
                    self._started_unix_ms = None
                    self._runtime_session_id = None
                    self._runtime_mode = "idle"
                    self._pending_mapping_finalize = None
            self._discard_armed_fault_capture()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FastLivo2Supervisor()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
