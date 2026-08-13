"""Own the FAST-LIVO2 child process and its mapping session receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import uuid

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointCloud2
from std_msgs.msg import String

from .collection_core import (
    COLLECTION_SOURCES,
    CollectionHealth,
    finalize_collection_session,
    normalize_collection_directory,
    rosbag_record_command,
)


_STATUS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
_MAP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class FastLivo2Supervisor(Node):
    def __init__(self) -> None:
        super().__init__("g1_fast_livo2_supervisor")
        self.declare_parameter("command_topic", "/ubuntu/navigation/fast_livo2/command")
        self.declare_parameter("status_topic", "/ubuntu/navigation/fast_livo2/status")
        self.declare_parameter(
            "collection_status_topic",
            "/ubuntu/navigation/fast_livo2/collection_status",
        )
        self.declare_parameter("diagnostics_topic", "/ubuntu/navigation/fast_livo2/diagnostics")
        self.declare_parameter("reset_topic", "/ubuntu/navigation/fast_livo2/reset_map")
        self.declare_parameter("map_control_topic", "/ubuntu/navigation/fast_livo2/map_control")
        self.declare_parameter("map_control_status_topic", "/ubuntu/navigation/fast_livo2/map_control_status")
        self.declare_parameter("config_path", "/config/g1_lio.yaml")
        self.declare_parameter("map_root", "/opt/fast_livo_ws/src/fast_livo/Log/pcd")
        self.declare_parameter("pcd_save_interval", 600)
        self.declare_parameter("stop_timeout_sec", 120.0)
        self.declare_parameter("map_control_timeout_sec", 130.0)
        self.declare_parameter("collection_stop_timeout_sec", 30.0)

        self._lock = threading.RLock()
        self._runtime_lifecycle_lock = threading.Lock()
        self._collection_lifecycle_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._active_map: str | None = None
        self._loaded_map: str | None = None
        self._runtime_mode = "idle"
        self._started_unix_ms: int | None = None
        self._files_before: dict[str, tuple[int, int]] = {}
        self._diagnostics: dict = {}
        self._map_control_responses: dict[str, dict] = {}
        self._collection_process: subprocess.Popen | None = None
        self._collection_partial_dir: Path | None = None
        self._collection_final_dir: Path | None = None
        self._collection_directory: str | None = None
        self._collection_error: str | None = None
        self._collection_last_receipt: dict | None = None
        self._collection_health = CollectionHealth()
        self._condition = threading.Condition(self._lock)
        self._map_root = Path(str(self.get_parameter("map_root").value))
        self._map_root.mkdir(parents=True, exist_ok=True)
        self._status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), _STATUS_QOS)
        self._collection_status_pub = self.create_publisher(
            String,
            str(self.get_parameter("collection_status_topic").value),
            _STATUS_QOS,
        )
        self._reset_pub = self.create_publisher(String, str(self.get_parameter("reset_topic").value), 10)
        self._map_control_pub = self.create_publisher(
            String, str(self.get_parameter("map_control_topic").value), 10
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
            str(self.get_parameter("map_control_status_topic").value),
            self._on_map_control_status,
            10,
            callback_group=callbacks,
        )
        self.create_subscription(String, str(self.get_parameter("diagnostics_topic").value), self._on_diagnostics, 10)
        self._collection_subscriptions = self._create_collection_subscriptions()
        self.create_timer(1.0, self._publish_heartbeat)

    def _create_collection_subscriptions(self) -> list:
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        message_types = {
            "lidar": PointCloud2,
            "imu": Imu,
            "rgb": CompressedImage,
            "depth": Image,
            "camera_info": CameraInfo,
        }
        subscriptions = []
        for item in COLLECTION_SOURCES:
            port = item["port"]
            qos = reliable_qos if port in {"lidar", "imu"} else sensor_qos
            subscriptions.append(
                self.create_subscription(
                    message_types[port],
                    item["topic"],
                    lambda message, source_port=port: self._on_collection_sample(
                        source_port, message
                    ),
                    qos,
                )
            )
        return subscriptions

    def _on_collection_sample(self, port: str, message) -> None:
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        source_stamp_ns = None
        if stamp is not None:
            candidate = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            if candidate > 0:
                source_stamp_ns = candidate
        with self._lock:
            self._collection_health.observe(
                port,
                source_stamp_ns=source_stamp_ns,
            )

    def _on_diagnostics(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(payload, dict):
            with self._lock:
                self._diagnostics = payload

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
                with self._runtime_lifecycle_lock:
                    if action == "start_mapping":
                        result = self._start_mapping(str(args.get("map_name", "")))
                    elif action == "stop_mapping":
                        result = self._stop_mapping()
                    elif action == "load_map":
                        result = self._load_map(str(args.get("map_name", "")))
                    elif action == "relocalize":
                        result = self._relocalize(args)
                    else:
                        result = self._unload_map()
            elif action == "configure_collection":
                with self._collection_lifecycle_lock:
                    result = self._configure_collection(args)
            else:
                result = {"status": "error", "error_code": "unsupported_action", "error": f"unsupported action {action}"}
        except Exception as exc:
            request_id = locals().get("request_id", "")
            action = locals().get("action", "")
            result = {"status": "error", "error_code": "supervisor_error", "error": f"{type(exc).__name__}: {exc}"}
        self._publish({"event": "response", "request_id": request_id, "action": action, **result})

    def _configure_collection(self, args: dict) -> dict:
        enabled = args.get("enabled")
        if not isinstance(enabled, bool):
            return {
                "status": "error",
                "error_code": "invalid_collection_config",
                "error": "enabled must be a boolean",
            }
        if not enabled:
            return self._stop_collection()
        if args.get("namespace") != "ubuntu":
            return {
                "status": "error",
                "error_code": "invalid_collection_config",
                "error": "first-release collection requires namespace=ubuntu",
            }
        try:
            directory = normalize_collection_directory(args.get("directory"))
        except ValueError as exc:
            return {
                "status": "error",
                "error_code": "invalid_collection_config",
                "error": str(exc),
            }

        with self._lock:
            process = self._collection_process
            current_directory = self._collection_directory
        if process is not None and process.poll() is None:
            if current_directory == directory:
                return {
                    "status": "recording",
                    "already_started": True,
                    "collection": self._collection_snapshot(),
                }
            stopped = self._stop_collection()
            if stopped.get("status") == "error":
                return stopped

        root = Path(directory)
        process = None
        partial_dir = None
        final_dir = None
        session_id = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            now = time.gmtime()
            session_id = time.strftime("%Y%m%dT%H%M%SZ", now) + "-" + uuid.uuid4().hex[:8]
            parent = root / "ubuntu" / time.strftime("%Y-%m-%d", now)
            parent.mkdir(parents=True, exist_ok=True)
            partial_dir = parent / f"{session_id}.partial"
            final_dir = parent / session_id
            if partial_dir.exists() or final_dir.exists():
                raise FileExistsError(f"collection session already exists: {session_id}")
            process = subprocess.Popen(
                rosbag_record_command(str(partial_dir)),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(0.25)
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(f"ros2 bag record exited with {return_code}")
        except (OSError, RuntimeError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            receipt = None
            if partial_dir is not None and final_dir is not None and session_id is not None:
                try:
                    partial_dir.mkdir(parents=True, exist_ok=True)
                    failed_health = CollectionHealth()
                    failed_health.start(session_id, str(final_dir))
                    failed_status = failed_health.snapshot(
                        process_running=False,
                        process_return_code=(
                            None if process is None else process.poll()
                        ),
                        process_error=failure,
                    )
                    receipt = finalize_collection_session(
                        str(partial_dir),
                        str(final_dir),
                        {
                            "schema": "phanthy.navigation.fast_livo2_collection_receipt.v1",
                            "state": "failed",
                            "stopped_unix_ms": int(time.time() * 1000),
                            "storage_complete": False,
                            "return_code": None if process is None else process.poll(),
                            "failure_reason": failure,
                            "sources": failed_status["sources"],
                        },
                        storage_complete=False,
                    )
                except OSError as receipt_exc:
                    failure += f"; receipt_failed:{receipt_exc}"
            with self._lock:
                self._collection_error = failure
                self._collection_last_receipt = (
                    None if receipt is None else dict(receipt)
                )
            return {
                "status": "error",
                "error_code": "collection_start_failed",
                "error": self._collection_error,
                "receipt": receipt,
            }

        with self._lock:
            self._collection_process = process
            self._collection_partial_dir = partial_dir
            self._collection_final_dir = final_dir
            self._collection_directory = directory
            self._collection_error = None
            self._collection_health.start(session_id, str(final_dir))
        return {"status": "recording", "collection": self._collection_snapshot()}

    def _stop_collection(self) -> dict:
        with self._lock:
            process = self._collection_process
            partial_dir = self._collection_partial_dir
            final_dir = self._collection_final_dir
            source_status = self._collection_health.snapshot(
                process_running=process is not None and process.poll() is None,
                process_return_code=None if process is None else process.poll(),
                process_error=self._collection_error,
            )
            was_enabled = source_status["enabled"]
        if process is None and not was_enabled:
            return {
                "status": "disabled",
                "already_stopped": True,
                "collection": self._collection_snapshot(),
            }

        return_code = None if process is None else process.poll()
        stop_error = None
        if process is not None and return_code is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                return_code = process.wait(
                    timeout=float(
                        self.get_parameter("collection_stop_timeout_sec").value
                    )
                )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait(timeout=10)
                stop_error = "rosbag_stop_timeout"
            except OSError as exc:
                stop_error = f"rosbag_stop_failed:{exc}"
                if process.poll() is None:
                    with self._lock:
                        self._collection_error = stop_error
                    return {
                        "status": "error",
                        "error_code": "collection_stop_failed",
                        "error": stop_error,
                        "collection": self._collection_snapshot(),
                    }

        storage_ok = stop_error is None and return_code == 0 and partial_dir is not None
        receipt = {
            "schema": "phanthy.navigation.fast_livo2_collection_receipt.v1",
            "state": (
                "complete"
                if storage_ok and source_status["healthy"]
                else "degraded" if storage_ok else "failed"
            ),
            "stopped_unix_ms": int(time.time() * 1000),
            "storage_complete": storage_ok,
            "return_code": return_code,
            "failure_reason": (
                stop_error
                or source_status.get("failure_reason")
                or (
                    "missing_sources:" + ",".join(source_status["missing_sources"])
                    if source_status["missing_sources"]
                    else None
                )
            ),
            "sources": source_status["sources"],
        }
        if partial_dir is not None and partial_dir.is_dir():
            try:
                receipt = finalize_collection_session(
                    str(partial_dir),
                    str(final_dir),
                    receipt,
                    storage_complete=storage_ok,
                )
            except OSError as exc:
                storage_ok = False
                stop_error = f"collection_finalize_failed:{exc}"
                receipt.update(
                    {
                        "state": "failed",
                        "storage_complete": False,
                        "failure_reason": stop_error,
                    }
                )

        with self._lock:
            self._collection_process = None
            self._collection_partial_dir = None
            self._collection_final_dir = None
            self._collection_directory = None
            self._collection_error = stop_error
            self._collection_last_receipt = dict(receipt)
            self._collection_health.stop()
        if not storage_ok:
            return {
                "status": "error",
                "error_code": "collection_stop_failed",
                "error": stop_error or "rosbag did not finish cleanly",
                "receipt": receipt,
            }
        return {"status": "collection_saved", "receipt": receipt}

    def _collection_snapshot(self) -> dict:
        with self._lock:
            process = self._collection_process
            return_code = None if process is None else process.poll()
            result = self._collection_health.snapshot(
                process_running=process is not None and return_code is None,
                process_return_code=return_code,
                process_error=self._collection_error,
            )
            result["pid"] = (
                process.pid if process is not None and return_code is None else None
            )
            result["last_receipt"] = (
                None
                if self._collection_last_receipt is None
                else dict(self._collection_last_receipt)
            )
        for item in COLLECTION_SOURCES:
            result["sources"][item["port"]]["publisher_count"] = (
                self.count_publishers(item["topic"])
            )
        return result

    def _start_mapping(self, map_name: str) -> dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return {"status": "error", "error_code": "mapping_active", "error": f"mapping {self._active_map} is already active"}
            if not _MAP_NAME_RE.fullmatch(map_name):
                return {
                    "status": "error",
                    "error_code": "invalid_argument",
                    "error": "map_name must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
                }
            self._files_before = self._pcd_files()
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
            return {"status": "mapping", "map_name": map_name, "pid": self._process.pid, "session_local": True}

    def _stop_mapping(self) -> dict:
        with self._lock:
            if self._runtime_mode == "localization":
                return {
                    "status": "error",
                    "error_code": "localization_active",
                    "error": f"map {self._loaded_map} is loaded for localization",
                }
            process = self._process
            map_name = self._active_map
            started = self._started_unix_ms
        if process is None:
            return {"status": "stopped", "already_idle": True, "map_name": map_name}
        if process.poll() is not None:
            return_code = process.returncode
            files = self._changed_pcd_files()
            manifest = self._write_manifest(map_name, started, return_code, files)
            with self._lock:
                self._process = None
                self._active_map = None
                self._started_unix_ms = None
                self._runtime_mode = "idle"
            return {
                "status": "error",
                "error_code": "algorithm_exited",
                "error": f"FAST-LIVO2 exited unexpectedly with {return_code}",
                "map_name": map_name,
                "pcd_files": files,
                "manifest": manifest,
            }
        os.killpg(process.pid, signal.SIGINT)
        try:
            return_code = process.wait(timeout=float(self.get_parameter("stop_timeout_sec").value))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            return {"status": "error", "error_code": "algorithm_stop_timeout", "error": "FAST-LIVO2 did not stop within timeout"}
        files = self._changed_pcd_files()
        manifest = self._write_manifest(map_name, started, return_code, files)
        with self._lock:
            self._process = None
            self._active_map = None
            self._started_unix_ms = None
            self._runtime_mode = "idle"
        if return_code != 0:
            return {"status": "error", "error_code": "algorithm_stop_failed", "error": f"FAST-LIVO2 exited with {return_code}", "map_name": map_name, "pcd_files": files}
        return {
            "status": "saved" if files else "stopped",
            "map_name": map_name,
            "pcd_files": files,
            "manifest": manifest,
            "global_relocalization_supported": False,
            "bounded_relocalization_supported": True,
        }

    def _load_map(self, map_name: str) -> dict:
        with self._lock:
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
            files = self._map_files_from_manifest(map_name)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "error",
                "error_code": "map_artifact_invalid",
                "error": str(exc),
                "loaded_map": previous_map,
                "runtime_mode": runtime_mode,
            }
        previous_files = None
        if runtime_mode == "localization" and previous_map:
            try:
                previous_files = self._map_files_from_manifest(previous_map)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return {
                    "status": "error",
                    "error_code": "active_map_artifact_invalid",
                    "error": f"cannot replace active map {previous_map}: {exc}",
                    "loaded_map": previous_map,
                    "runtime_mode": runtime_mode,
                }
        if runtime_mode == "localization":
            unloaded = self._unload_map()
            if unloaded.get("status") == "error":
                rollback = (
                    self._activate_localization(previous_map, previous_files)
                    if previous_map and previous_files
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
        loaded = self._activate_localization(map_name, files)
        if loaded.get("status") == "error":
            rollback = None
            if previous_map and previous_files:
                rollback = self._activate_localization(previous_map, previous_files)
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

    def _activate_localization(self, map_name: str, files: tuple[Path, ...]) -> dict:
        loaded = self._adapter_execute(
            "load_map", {"map_name": map_name, "pcd_files": [str(path) for path in files]}
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
            self._adapter_execute("unload_map", {})
            return {
                "status": "error",
                "error_code": "algorithm_start_failed",
                "error": f"cannot start FAST-LIVO2: {exc}",
            }
        time.sleep(0.25)
        return_code = process.poll()
        if return_code is not None:
            self._adapter_execute("unload_map", {})
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
        adapter_result = self._adapter_execute("unload_map", {})
        with self._lock:
            self._process = None
            self._loaded_map = None
            self._runtime_mode = "idle"
            self._started_unix_ms = None
        if stop_error is not None:
            return {
                **stop_error,
                "loaded_map": None,
                "runtime_mode": "idle",
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
            process.wait(timeout=float(self.get_parameter("stop_timeout_sec").value))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            return {
                "status": "error",
                "error_code": "algorithm_stop_timeout",
                "error": "FAST-LIVO2 did not stop within timeout",
            }
        return None

    def _map_files_from_manifest(self, map_name: str) -> tuple[Path, ...]:
        if not _MAP_NAME_RE.fullmatch(map_name):
            raise ValueError("map_name must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
        manifest_path = self._map_root / "sessions" / f"{map_name}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "phanthy.navigation.fast_livo2_map_session.v1":
            raise ValueError("map manifest schema is unsupported")
        if manifest.get("map_name") != map_name:
            raise ValueError("map manifest name does not match request")
        names = manifest.get("pcd_files")
        if not isinstance(names, list) or not names:
            raise ValueError("map manifest has no PCD files")
        root = self._map_root.resolve()
        files = []
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pcd"):
                raise ValueError("map manifest contains an unsafe PCD path")
            path = (root / name).resolve()
            if path.parent != root or not path.is_file():
                raise ValueError(f"map PCD is missing: {name}")
            files.append(path)
        return tuple(files)

    def _adapter_execute(self, action: str, args: dict) -> dict:
        request_id = f"map-{time.time_ns()}"
        message = String()
        message.data = json.dumps(
            {"request_id": request_id, "action": action, "args": args},
            separators=(",", ":"),
        )
        with self._condition:
            self._map_control_pub.publish(message)
            deadline = time.monotonic() + float(
                self.get_parameter("map_control_timeout_sec").value
            )
            while request_id not in self._map_control_responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "status": "error",
                        "error_code": "map_control_timeout",
                        "error": f"FAST-LIVO2 adapter did not answer {action}",
                    }
                self._condition.wait(timeout=remaining)
            response = self._map_control_responses.pop(request_id)
        return {key: value for key, value in response.items() if key not in {"event", "request_id", "action"}}

    def _algorithm_command(self, *, save_pcd: bool = True) -> list[str]:
        interval = int(self.get_parameter("pcd_save_interval").value)
        config = str(self.get_parameter("config_path").value)
        return [
            "/opt/fast_livo_ws/install/fast_livo/lib/fast_livo/fastlivo_mapping",
            "--ros-args",
            "--params-file", config,
            "-p", f"pcd_save.pcd_save_en:={'true' if save_pcd else 'false'}",
            "-p", f"pcd_save.interval:={interval}",
            "-p", "pcd_save.type:=0",
            "--log-level", "warn",
            "-r", "/livox/lidar:=/ubuntu/navigation/lidar_fast_livo",
            "-r", "/livox/imu:=/ubuntu/navigation/imu",
            "-r", "/left_camera/image:=/ubuntu/navigation/camera_disabled",
            "-r", "/cloud_registered:=/ubuntu/navigation/fast_livo2/raw/cloud_registered",
            "-r", "/aft_mapped_to_init:=/ubuntu/navigation/fast_livo2/raw/odom",
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

    def _write_manifest(self, map_name, started, return_code, files) -> str | None:
        if not map_name:
            return None
        directory = self._map_root / "sessions"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{map_name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "phanthy.navigation.fast_livo2_map_session.v1",
                    "map_name": map_name,
                    "started_unix_ms": started,
                    "stopped_unix_ms": int(time.time() * 1000),
                    "return_code": return_code,
                    "pcd_files": files,
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
        collection = self._collection_snapshot()
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            payload = {
                "event": "heartbeat",
                "schema": "phanthy.navigation.fast_livo2_status.v1",
                "state": (
                    str(self._diagnostics.get("localization_state", "localization"))
                    if running and self._runtime_mode == "localization"
                    else "mapping" if running else "idle"
                ),
                "status": (
                    str(self._diagnostics.get("localization_state", "localization"))
                    if running and self._runtime_mode == "localization"
                    else "mapping" if running else "idle"
                ),
                "active_map": self._active_map,
                "loaded_map": self._loaded_map,
                "runtime_mode": self._runtime_mode,
                "algorithm_running": running,
                "companion_ready": True,
                "session_local": True,
                "global_relocalization_supported": False,
                "bounded_relocalization_supported": True,
                "diagnostics": dict(self._diagnostics),
                "collection": collection,
            }
        self._publish(payload)
        message = String()
        message.data = json.dumps(
            collection,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._collection_status_pub.publish(message)

    def _publish(self, payload: dict) -> None:
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._status_pub.publish(message)

    def destroy_node(self):
        try:
            with self._runtime_lifecycle_lock:
                if self._runtime_mode == "localization":
                    self._terminate_process(self._process)
                    with self._lock:
                        self._process = None
                        self._loaded_map = None
                        self._runtime_mode = "idle"
                else:
                    self._stop_mapping()
            with self._collection_lifecycle_lock:
                self._stop_collection()
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
