"""Fail-closed JSON command bridge from Perception to Nav2 actions."""

from __future__ import annotations

import json
import math
import os
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import SpeedLimit
from nav2_msgs.srv import LoadMap
from sensor_msgs.msg import LaserScan
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from slam_toolbox.srv import Pause, SaveMap as SlamSaveMap, SerializePoseGraph
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .execution_protocol import (
    ProtocolError,
    Velocity,
    build_velocity_proposal,
)
from .map_store import MapStore, MapStoreError, MappingSession
from .readiness import evaluate_readiness, navigation_motion_blocker


_TERMINAL_STATES = {
    "arrived",
    "cancelled",
    "stopped",
    "error",
    "aborted",
    "rejected",
}
_IDLE_OR_TERMINAL_STATES = _TERMINAL_STATES | {"paused"}


class CommandError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class NavigationCommandNode(Node):
    """Own the Nav2 action type so the Perception image only needs std_msgs."""

    def __init__(self) -> None:
        super().__init__("g1_nav2_navigation_command")
        self.declare_parameter(
            "command_topic", "/ubuntu/navigation/nav2/command"
        )
        self.declare_parameter(
            "status_topic", "/ubuntu/navigation/nav2/status"
        )
        self.declare_parameter(
            "runtime_switch_topic", "/ubuntu/navigation/nav2/runtime_switch"
        )
        self.declare_parameter("action_name", "/navigate_to_pose")
        self.declare_parameter(
            "shadow_topic", "/ubuntu/navigation/nav2/cmd_vel_shadow"
        )
        self.declare_parameter(
            "proposal_topic", "/ubuntu/navigation/nav2/velocity_proposal"
        )
        self.declare_parameter(
            "controller_speed_limit_topic",
            "/ubuntu/navigation/nav2/speed_limit",
        )
        self.declare_parameter("speed_limit_timeout", 3.0)
        self.declare_parameter("behavior_tree_path", "")
        self.declare_parameter("proposal_ttl_ms", 250)
        self.declare_parameter("enforce_shadow_isolation", True)
        self.declare_parameter("max_shadow_speed", 0.15)
        self.declare_parameter("supported_mode", 0)
        self.declare_parameter("goal_response_timeout", 8.0)
        self.declare_parameter("runtime_mode", "mapping")
        self.declare_parameter("maps_root", "/maps")
        self.declare_parameter("startup_map_name", "")
        self.declare_parameter("service_timeout", 20.0)
        self.declare_parameter("pose_lookup_timeout", 2.0)
        self.declare_parameter(
            "slam_save_map_service", "/slam_toolbox/save_map"
        )
        self.declare_parameter(
            "slam_serialize_service", "/slam_toolbox/serialize_map"
        )
        self.declare_parameter(
            "slam_pause_service", "/slam_toolbox/pause_new_measurements"
        )
        self.declare_parameter("load_map_service", "/map_server/load_map")
        self.declare_parameter("initial_pose_topic", "/initialpose")
        self.declare_parameter(
            "odom_status_topic", "/ubuntu/navigation/nav2/odom_status"
        )
        self.declare_parameter("scan_topic", "/ubuntu/navigation/nav2/scan")
        self.declare_parameter("sensor_max_age_sec", 0.5)
        self.declare_parameter("sensor_max_stamp_skew_sec", 0.2)
        self.declare_parameter(
            "required_lifecycle_nodes",
            ["controller_server", "velocity_smoother", "planner_server", "bt_navigator"],
        )

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._runtime_switch_topic = str(
            self.get_parameter("runtime_switch_topic").value
        )
        self._action_name = str(self.get_parameter("action_name").value)
        self._shadow_topic = str(self.get_parameter("shadow_topic").value)
        self._proposal_topic = str(self.get_parameter("proposal_topic").value)
        self._controller_speed_limit_topic = str(
            self.get_parameter("controller_speed_limit_topic").value
        )
        self._speed_limit_timeout = float(
            self.get_parameter("speed_limit_timeout").value
        )
        self._behavior_tree_path = str(
            self.get_parameter("behavior_tree_path").value
        )
        self._proposal_ttl_ms = int(
            self.get_parameter("proposal_ttl_ms").value
        )
        self._enforce_shadow_isolation = bool(
            self.get_parameter("enforce_shadow_isolation").value
        )
        self._max_shadow_speed = float(
            self.get_parameter("max_shadow_speed").value
        )
        self._supported_mode = int(self.get_parameter("supported_mode").value)
        self._goal_response_timeout = float(
            self.get_parameter("goal_response_timeout").value
        )
        self._runtime_mode = str(self.get_parameter("runtime_mode").value)
        self._maps_root = str(self.get_parameter("maps_root").value)
        self._startup_map_name = str(
            self.get_parameter("startup_map_name").value
        ).strip()
        self._service_timeout = float(
            self.get_parameter("service_timeout").value
        )
        self._pose_lookup_timeout = float(
            self.get_parameter("pose_lookup_timeout").value
        )
        self._slam_save_map_service = str(
            self.get_parameter("slam_save_map_service").value
        )
        self._slam_serialize_service = str(
            self.get_parameter("slam_serialize_service").value
        )
        self._slam_pause_service = str(
            self.get_parameter("slam_pause_service").value
        )
        self._load_map_service = str(
            self.get_parameter("load_map_service").value
        )
        self._initial_pose_topic = str(
            self.get_parameter("initial_pose_topic").value
        )
        self._odom_status_topic = str(
            self.get_parameter("odom_status_topic").value
        )
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._sensor_max_age_sec = float(
            self.get_parameter("sensor_max_age_sec").value
        )
        self._sensor_max_stamp_skew_sec = float(
            self.get_parameter("sensor_max_stamp_skew_sec").value
        )
        self._required_lifecycle_nodes = [
            str(item).strip("/")
            for item in self.get_parameter("required_lifecycle_nodes").value
        ]
        if not 0.0 < self._max_shadow_speed <= 0.2:
            raise ValueError("max_shadow_speed must be within (0, 0.2]")
        if not 50 <= self._proposal_ttl_ms <= 250:
            raise ValueError("proposal_ttl_ms must be within [50, 250]")
        if self._supported_mode != 0:
            raise ValueError("supported_mode must be 0 until another mode is implemented")
        if self._goal_response_timeout <= 0:
            raise ValueError("goal_response_timeout must be positive")
        if self._speed_limit_timeout <= 0:
            raise ValueError("speed_limit_timeout must be positive")
        if not os.path.isfile(self._behavior_tree_path):
            raise ValueError(
                "behavior_tree_path must point to the installed G1 Nav2 tree"
            )
        if self._runtime_mode not in {"mapping", "localization"}:
            raise ValueError("runtime_mode must be mapping or localization")
        if self._service_timeout <= 0 or self._pose_lookup_timeout <= 0:
            raise ValueError("service and pose lookup timeouts must be positive")
        if self._sensor_max_age_sec <= 0:
            raise ValueError("sensor_max_age_sec must be positive")
        if not 0.0 < self._sensor_max_stamp_skew_sec <= self._sensor_max_age_sec:
            raise ValueError(
                "sensor_max_stamp_skew_sec must be positive and no greater "
                "than sensor_max_age_sec"
            )
        if not self._required_lifecycle_nodes or any(
            not item for item in self._required_lifecycle_nodes
        ):
            raise ValueError("required_lifecycle_nodes must not be empty")

        self._callbacks = ReentrantCallbackGroup()
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
        self._status_pub = self.create_publisher(
            String, self._status_topic, status_qos
        )
        self._runtime_switch_pub = self.create_publisher(
            String, self._runtime_switch_topic, command_qos
        )
        self._proposal_pub = self.create_publisher(
            String, self._proposal_topic, command_qos
        )
        self._speed_limit_pub = self.create_publisher(
            SpeedLimit, self._controller_speed_limit_topic, command_qos
        )
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self._initial_pose_topic,
            status_qos,
        )
        self._command_sub = self.create_subscription(
            String,
            self._command_topic,
            self._on_command,
            command_qos,
            callback_group=self._callbacks,
        )
        self._shadow_sub = self.create_subscription(
            Twist,
            self._shadow_topic,
            self._on_shadow_velocity,
            command_qos,
            callback_group=self._callbacks,
        )
        self._odom_status_sub = self.create_subscription(
            String,
            self._odom_status_topic,
            self._on_odom_status,
            command_qos,
            callback_group=self._callbacks,
        )
        self._scan_sub = self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
            callback_group=self._callbacks,
        )
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            self._action_name,
            callback_group=self._callbacks,
        )
        self._slam_save_client = self.create_client(
            SlamSaveMap,
            self._slam_save_map_service,
            callback_group=self._callbacks,
        )
        self._slam_serialize_client = self.create_client(
            SerializePoseGraph,
            self._slam_serialize_service,
            callback_group=self._callbacks,
        )
        self._slam_pause_client = self.create_client(
            Pause,
            self._slam_pause_service,
            callback_group=self._callbacks,
        )
        self._load_map_client = self.create_client(
            LoadMap,
            self._load_map_service,
            callback_group=self._callbacks,
        )
        self._lifecycle_clients = {
            name: self.create_client(
                GetState,
                f"/{name}/get_state",
                callback_group=self._callbacks,
            )
            for name in self._required_lifecycle_nodes
        }
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(
            self._tf_buffer, self, spin_thread=False
        )
        self._heartbeat = self.create_timer(
            1.0, self._publish_heartbeat, callback_group=self._callbacks
        )
        self._lifecycle_timer = self.create_timer(
            1.0, self._refresh_lifecycle_states, callback_group=self._callbacks
        )

        self._lock = threading.RLock()
        self._state_changed = threading.Condition(self._lock)
        self._command_lock = threading.Lock()
        self._active: dict | None = None
        self._proposal_sequence = 0
        self._map_store = MapStore(self._maps_root)
        self._mapping_session: MappingSession | None = None
        self._mapping_closed = False
        self._active_map_name: str | None = None
        self._last_odom_status: dict = {}
        self._last_odom_status_monotonic: float | None = None
        self._last_scan_monotonic: float | None = None
        self._last_scan_source_age_sec: float | None = None
        self._last_scan_source_stamp_ns: int | None = None
        self._lifecycle_states = {
            name: 0 for name in self._required_lifecycle_nodes
        }
        if self._runtime_mode == "localization" and self._startup_map_name:
            self._map_store.map_summary(self._startup_map_name)
            self._active_map_name = self._startup_map_name
            self._map_store.set_active_map(self._startup_map_name)
        elif self._runtime_mode == "localization":
            self._active_map_name = self._map_store.active_map()
            if self._active_map_name is None:
                raise ValueError(
                    "localization runtime requires startup_map_name or an active map"
                )
        self._publish_heartbeat()

    def _on_odom_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._last_odom_status = dict(payload)
            self._last_odom_status_monotonic = time.monotonic()

    def _on_scan(self, message: LaserScan) -> None:
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        now_ns = self.get_clock().now().nanoseconds
        source_age = (
            (now_ns - stamp_ns) / 1_000_000_000.0 if stamp_ns > 0 else None
        )
        with self._lock:
            self._last_scan_monotonic = time.monotonic()
            self._last_scan_source_age_sec = source_age
            self._last_scan_source_stamp_ns = stamp_ns if stamp_ns > 0 else None

    def _refresh_lifecycle_states(self) -> None:
        for name, client in self._lifecycle_clients.items():
            if not client.service_is_ready():
                with self._lock:
                    self._lifecycle_states[name] = 0
                continue
            future = client.call_async(GetState.Request())

            def _done(completed, node_name=name) -> None:
                try:
                    response = completed.result()
                    state_id = int(response.current_state.id)
                except Exception:
                    state_id = 0
                with self._lock:
                    self._lifecycle_states[node_name] = state_id

            future.add_done_callback(_done)

    def _readiness(self) -> dict:
        with self._lock:
            odom_status = dict(self._last_odom_status)
            odom_received_at = self._last_odom_status_monotonic
            scan_received_at = self._last_scan_monotonic
            scan_source_age = self._last_scan_source_age_sec
            scan_source_stamp_ns = self._last_scan_source_stamp_ns
            lifecycle_states = dict(self._lifecycle_states)
            map_ready = bool(self._mapping_session or self._active_map_name)
        odom_source_stamp_ns = odom_status.get("source_stamp_ns")
        sensor_stamp_skew = None
        if (
            isinstance(odom_source_stamp_ns, int)
            and not isinstance(odom_source_stamp_ns, bool)
            and odom_source_stamp_ns > 0
            and scan_source_stamp_ns is not None
        ):
            sensor_stamp_skew = abs(
                odom_source_stamp_ns - scan_source_stamp_ns
            ) / 1_000_000_000.0
        map_to_base_ready = self._tf_buffer.can_transform(
            "map", "base_link", Time(), timeout=Duration(seconds=0.0)
        )
        return evaluate_readiness(
            now_monotonic=time.monotonic(),
            max_age_sec=self._sensor_max_age_sec,
            odom_status=odom_status,
            odom_status_received_at=odom_received_at,
            scan_received_at=scan_received_at,
            scan_source_age_sec=scan_source_age,
            sensor_stamp_skew_sec=sensor_stamp_skew,
            max_sensor_stamp_skew_sec=self._sensor_max_stamp_skew_sec,
            lifecycle_states=lifecycle_states,
            action_server_ready=self._action_client.server_is_ready(),
            map_ready=map_ready,
            map_to_base_ready=map_to_base_ready,
        )

    def _require_runtime_ready(self, action: str) -> None:
        receipt = self._readiness()
        if not receipt["n3_ready"]:
            raise CommandError(
                "navigation_runtime_not_ready",
                f"{action} blocked: " + ",".join(receipt["readiness_blockers"]),
            )

    def _require_navigation_ready(self, action: str) -> None:
        receipt = self._readiness()
        if not receipt["navigation_ready"]:
            raise CommandError(
                "navigation_not_ready",
                f"{action} blocked: "
                + ",".join(receipt["navigation_blockers"]),
            )

    def _on_shadow_velocity(self, message: Twist) -> None:
        with self._lock:
            if self._active is None:
                return
            nav_id = self._active.get("nav_id")
            status = self._active.get("status", "error")
        if not isinstance(nav_id, str) or not nav_id:
            return

        velocity = Velocity(
            x=float(message.linear.x),
            y=float(message.linear.y),
            yaw=float(message.angular.z),
        )
        reason = None
        if status not in {"starting", "navigating"}:
            velocity = Velocity.zero()
            reason = f"navigation_{status}"
        else:
            reason = navigation_motion_blocker(self._readiness())
            if reason is not None:
                velocity = Velocity.zero()
        try:
            self._publish_velocity_proposal(
                nav_id=nav_id,
                navigation_status=status,
                velocity=velocity,
                reason=reason,
            )
        except ProtocolError as exc:
            self.get_logger().error(
                f"unsafe Nav2 shadow velocity rejected: {exc.code}: {exc}"
            )
            with self._lock:
                if self._active and self._active.get("nav_id") == nav_id:
                    self._active["status"] = "error"
                    self._active["error_code"] = "unsafe_shadow_velocity"
                    self._active["error"] = f"{exc.code}: {exc}"
            self._publish_velocity_proposal(
                nav_id=nav_id,
                navigation_status="error",
                velocity=Velocity.zero(),
                reason=f"unsafe_shadow_velocity:{exc.code}",
            )

    def _on_command(self, message: String) -> None:
        request_id = ""
        nav_id = None
        action = ""
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise CommandError("invalid_request", "command must be a JSON object")
            request_id = payload.get("request_id", "")
            nav_id = payload.get("nav_id")
            action = payload.get("action", "")
            args = payload.get("args") or {}
            if not isinstance(request_id, str) or not request_id:
                raise CommandError("invalid_request", "request_id is required")
            if not isinstance(action, str) or not action:
                raise CommandError("invalid_request", "action is required")
            if not isinstance(args, dict):
                raise CommandError("invalid_request", "args must be an object")
            with self._command_lock:
                result = self._dispatch(
                    action, args, nav_id, request_id=request_id
                )
            self._respond(request_id, action, nav_id, result)
        except CommandError as exc:
            self._respond(
                request_id,
                action,
                nav_id,
                {
                    "status": "error",
                    "error_code": exc.code,
                    "error": str(exc),
                },
            )
        except MapStoreError as exc:
            self._respond(
                request_id,
                action,
                nav_id,
                {
                    "status": "error",
                    "error_code": exc.code,
                    "error": str(exc),
                },
            )
        except Exception as exc:
            self.get_logger().error(
                f"navigation command failed: {type(exc).__name__}: {exc}"
            )
            self._respond(
                request_id,
                action,
                nav_id,
                {
                    "status": "error",
                    "error_code": "internal_error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    def _dispatch(
        self, action: str, args: dict, nav_id, *, request_id: str = ""
    ) -> dict:
        if action == "start_mapping":
            return self._start_mapping(args, request_id=request_id)
        if action == "stop_mapping":
            return self._stop_mapping(request_id=request_id)
        if action == "switch_runtime_mode":
            return self._switch_runtime_mode(args, request_id=request_id)
        if action == "tag_place":
            return self._tag_place(args)
        if action == "untag_place":
            return self._untag_place(args)
        if action == "list_tags":
            return self._list_tags()
        if action == "list_maps":
            return self._list_maps()
        if action == "delete_map":
            return self._delete_map(args)
        if action == "load_map":
            return self._load_map(args)
        if action == "navigate_to_tag":
            if not isinstance(nav_id, str) or not nav_id:
                raise CommandError("invalid_request", "nav_id is required")
            return self._navigate_to_tag(nav_id, args)
        if action == "navigate_to_pose":
            if not isinstance(nav_id, str) or not nav_id:
                raise CommandError("invalid_request", "nav_id is required")
            return self._navigate_to_pose(nav_id, args)
        if action == "pause_nav":
            return self._pause(nav_id)
        if action == "resume_nav":
            return self._resume(nav_id)
        if action == "stop_nav":
            return self._stop(nav_id)
        if action == "wait_navigation_done":
            raise CommandError(
                "invalid_request",
                "wait_navigation_done is handled by the Perception adapter",
            )
        raise CommandError("unsupported_action", f"unsupported action: {action}")

    def _start_mapping(self, args: dict, *, request_id: str = "") -> dict:
        if self._runtime_mode == "localization":
            self._assert_no_active_navigation("start_mapping")
            map_name = self._map_store.validate_map_name(args.get("map_name"))
            if any(
                item.get("map_name") == map_name
                for item in self._map_store.list_maps()
            ):
                raise CommandError("map_exists", f"map already exists: {map_name}")
            self._request_runtime_switch(
                request_id=request_id,
                target_mode="mapping",
                map_name=map_name,
            )
            return {
                "status": "switching",
                "map_name": map_name,
                "runtime_mode": self._runtime_mode,
                "mode_switch_required": True,
                "next_runtime_mode": "mapping",
                "retry_action_after_switch": True,
            }
        self._require_runtime_mode("mapping", "start_mapping")
        self._require_runtime_ready("start_mapping")
        self._assert_no_active_navigation("start_mapping")
        with self._lock:
            if self._mapping_session is not None:
                raise CommandError(
                    "mapping_active",
                    f"mapping {self._mapping_session.map_name} is already active",
                )
            if self._mapping_closed:
                raise CommandError(
                    "mapping_restart_required",
                    "this SLAM runtime already saved a map; restart the Nav2 "
                    "companion in mapping mode before starting another map",
                )
            session = self._map_store.begin_mapping(args.get("map_name"))
            self._mapping_session = session
            self._active_map_name = None
        self._publish_state()
        return {
            "status": "mapping",
            "map_name": session.map_name,
            "runtime_mode": self._runtime_mode,
            "mapping_session": session.path.name,
        }

    def _stop_mapping(self, *, request_id: str = "") -> dict:
        self._require_runtime_mode("mapping", "stop_mapping")
        self._assert_no_active_navigation("stop_mapping")
        with self._lock:
            session = self._mapping_session
        if session is None:
            raise CommandError("no_active_mapping", "start_mapping is required first")

        paused = False
        try:
            prefix = str(session.path / "map")
            save_request = SlamSaveMap.Request()
            save_request.name.data = prefix
            save_response = None
            for attempt in range(3):
                save_response = self._call_service(
                    self._slam_save_client,
                    save_request,
                    self._slam_save_map_service,
                )
                if int(save_response.result) == 0:
                    break
                if attempt < 2:
                    self.get_logger().warning(
                        "SLAM occupancy save returned result="
                        f"{save_response.result}; retrying"
                    )
                    time.sleep(0.5)
            assert save_response is not None
            if int(save_response.result) != 0:
                raise CommandError(
                    "map_save_failed",
                    f"SLAM occupancy save returned result={save_response.result}",
                )

            pause_response = self._call_service(
                self._slam_pause_client,
                Pause.Request(),
                self._slam_pause_service,
            )
            if not bool(pause_response.status):
                raise CommandError(
                    "slam_pause_failed",
                    f"{self._slam_pause_service} rejected the pause request",
                )
            paused = True

            serialize_request = SerializePoseGraph.Request()
            serialize_request.filename = prefix
            serialize_response = self._call_service(
                self._slam_serialize_client,
                serialize_request,
                self._slam_serialize_service,
            )
            if int(serialize_response.result) != 0:
                raise CommandError(
                    "posegraph_save_failed",
                    "SLAM pose graph serialization returned "
                    f"result={serialize_response.result}",
                )

            summary = self._map_store.finalize_mapping(session)
            self._map_store.set_active_map(session.map_name)
        except Exception:
            if paused:
                self._resume_mapping_after_failed_save()
            raise

        with self._lock:
            self._mapping_session = None
            self._mapping_closed = True
            self._active_map_name = session.map_name
        self._request_runtime_switch(
            request_id=request_id,
            target_mode="localization",
            map_name=session.map_name,
        )
        self._publish_state()
        return {
            "status": "saved",
            "map_name": session.map_name,
            "map": summary,
            "runtime_mode": self._runtime_mode,
            "mode_switch_required": True,
            "next_runtime_mode": "localization",
            "next_map_yaml": summary["map_yaml"],
            "note": "Nav2 runtime is switching to localization automatically",
        }

    def _switch_runtime_mode(
        self, args: dict, *, request_id: str = ""
    ) -> dict:
        self._assert_no_active_navigation("switch_runtime_mode")
        target_mode = str(args.get("runtime_mode", "")).strip()
        if target_mode not in {"mapping", "localization"}:
            raise CommandError(
                "invalid_argument",
                "runtime_mode must be mapping or localization",
            )
        with self._lock:
            mapping_session = self._mapping_session
        if mapping_session is not None:
            raise CommandError(
                "mapping_active",
                "stop_mapping is required before switching runtime mode so the "
                "active map is saved",
            )

        map_name = ""
        summary = None
        if target_mode == "localization":
            map_name = self._map_store.validate_map_name(args.get("map_name"))
            summary = self._map_store.map_summary(map_name)

        self._request_runtime_switch(
            request_id=request_id,
            target_mode=target_mode,
            map_name=map_name,
        )
        result = {
            "status": "switching",
            "runtime_mode": self._runtime_mode,
            "mode_switch_required": True,
            "next_runtime_mode": target_mode,
            "map_name": map_name,
            "requested_mode_switch": True,
        }
        if summary is not None:
            result["next_map_yaml"] = summary["map_yaml"]
        return result

    def _request_runtime_switch(
        self, *, request_id: str, target_mode: str, map_name: str
    ) -> None:
        if not request_id:
            raise CommandError(
                "runtime_switch_unavailable",
                "request_id is required for mode switch",
            )
        message = String()
        message.data = json.dumps(
            {
                "request_id": request_id,
                "target_mode": target_mode,
                "map_name": map_name,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._runtime_switch_pub.publish(message)

    def _tag_place(self, args: dict) -> dict:
        map_name, directory = self._active_map_context()
        pose = self._current_map_pose()
        tag = self._map_store.put_tag(
            directory,
            map_name,
            args.get("name"),
            args.get("description", ""),
            pose,
        )
        return {
            "status": "tagged",
            "map_name": map_name,
            "tag": tag,
        }

    def _untag_place(self, args: dict) -> dict:
        map_name, directory = self._active_map_context()
        result = self._map_store.remove_tag(
            directory, map_name, args.get("name")
        )
        return {"map_name": map_name, **result}

    def _list_tags(self) -> dict:
        map_name, directory = self._active_map_context()
        tags = self._map_store.list_tags(directory, map_name)
        return {
            "status": "ready",
            "map_name": map_name,
            "tag_count": len(tags),
            "tags": tags,
        }

    def _list_maps(self) -> dict:
        maps = self._map_store.list_maps()
        return {
            "status": "ready",
            "runtime_mode": self._runtime_mode,
            "active_map": self._active_map_name,
            "mapping_map": self._mapping_session.map_name
            if self._mapping_session
            else None,
            "map_count": len(maps),
            "maps": maps,
        }

    def _delete_map(self, args: dict) -> dict:
        self._assert_no_active_navigation("delete_map")
        map_name = self._map_store.validate_map_name(args.get("map_name"))
        with self._lock:
            if self._mapping_session and self._mapping_session.map_name == map_name:
                raise CommandError(
                    "map_in_use", f"map {map_name} has an active mapping session"
                )
            if self._active_map_name == map_name:
                raise CommandError(
                    "map_in_use",
                    f"map {map_name} is active; load another map before deleting it",
                )
        return self._map_store.delete_map(map_name)

    def _load_map(self, args: dict) -> dict:
        self._require_runtime_mode("localization", "load_map")
        self._assert_no_active_navigation("load_map")
        map_name = self._map_store.validate_map_name(args.get("map_name"))
        summary = self._map_store.map_summary(map_name)
        request = LoadMap.Request()
        request.map_url = summary["map_yaml"]
        response = self._call_service(
            self._load_map_client,
            request,
            self._load_map_service,
        )
        if int(response.result) != 0:
            raise CommandError(
                "map_load_failed",
                f"Nav2 map server returned result={response.result} for {map_name}",
            )
        self._map_store.set_active_map(map_name)
        with self._lock:
            self._active_map_name = map_name
        self._publish_origin_initial_pose()
        self._publish_state()
        return {
            "status": "localized",
            "map": summary,
            "runtime_mode": self._runtime_mode,
            "initial_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "initial_pose_policy": "robot_must_be_at_mapping_origin",
        }

    def _navigate_to_tag(self, nav_id: str, args: dict) -> dict:
        map_name, directory = self._active_map_context()
        tag = self._map_store.get_tag(
            directory, map_name, args.get("tag_name")
        )
        result = self._navigate_to_pose(
            nav_id,
            {
                "x": tag["x"],
                "y": tag["y"],
                "yaw": tag["yaw"],
                "speed": args["speed"],
                "mode": args["mode"],
            },
        )
        result["map_name"] = map_name
        result["tag_name"] = tag["name"]
        return result

    def _active_map_context(self) -> tuple[str, object]:
        with self._lock:
            if self._mapping_session is not None:
                return (
                    self._mapping_session.map_name,
                    self._mapping_session.path,
                )
            map_name = self._active_map_name
        if not map_name:
            raise CommandError(
                "no_active_map",
                "start a mapping session or load a saved map first",
            )
        return map_name, self._map_store.directory_for_map(map_name)

    def _current_map_pose(self) -> dict:
        try:
            transform = self._tf_buffer.lookup_transform(
                "map",
                "base_link",
                Time(),
                timeout=Duration(seconds=self._pose_lookup_timeout),
            )
        except Exception as exc:
            raise CommandError(
                "map_pose_unavailable",
                f"map -> base_link transform is unavailable: {type(exc).__name__}: {exc}",
            ) from exc
        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )
        return {
            "x": float(translation.x),
            "y": float(translation.y),
            "yaw": float(yaw),
        }

    def _publish_origin_initial_pose(self) -> None:
        message = PoseWithCovarianceStamped()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.pose.orientation.w = 1.0
        message.pose.covariance[0] = 0.25
        message.pose.covariance[7] = 0.25
        message.pose.covariance[35] = 0.06853891945200942
        self._initial_pose_pub.publish(message)

    def _call_service(self, client, request, service_name: str):
        if not client.wait_for_service(timeout_sec=3.0):
            raise CommandError(
                "service_unavailable", f"ROS service is unavailable: {service_name}"
            )
        completed = threading.Event()
        outcome: dict = {}
        future = client.call_async(request)

        def _done(service_future) -> None:
            try:
                outcome["response"] = service_future.result()
            except Exception as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"
            completed.set()

        future.add_done_callback(_done)
        if not completed.wait(timeout=self._service_timeout):
            raise CommandError(
                "service_timeout",
                f"ROS service {service_name} exceeded {self._service_timeout:.1f}s",
            )
        if "error" in outcome or outcome.get("response") is None:
            raise CommandError(
                "service_call_failed",
                f"ROS service {service_name} failed: "
                f"{outcome.get('error', 'empty response')}",
            )
        return outcome["response"]

    def _resume_mapping_after_failed_save(self) -> None:
        try:
            response = self._call_service(
                self._slam_pause_client,
                Pause.Request(),
                self._slam_pause_service,
            )
            if not bool(response.status):
                self.get_logger().error(
                    "SLAM mapping could not resume after a failed save"
                )
        except Exception as exc:
            self.get_logger().error(
                "SLAM mapping resume failed after save error: "
                f"{type(exc).__name__}: {exc}"
            )

    def _require_runtime_mode(self, required: str, action: str) -> None:
        if self._runtime_mode != required:
            raise CommandError(
                "runtime_mode_mismatch",
                f"{action} requires {required} mode; current runtime is "
                f"{self._runtime_mode}. Restart the Nav2 companion manually.",
            )

    def _assert_no_active_navigation(self, action: str) -> None:
        with self._lock:
            if self._active and self._active.get("status") not in _TERMINAL_STATES:
                raise CommandError(
                    "navigation_active",
                    f"cannot run {action} while navigation "
                    f"{self._active.get('nav_id')} is active",
                )

    def _navigate_to_pose(self, nav_id: str, args: dict) -> dict:
        self._require_navigation_ready("navigate_to_pose")
        mode = int(args["mode"])
        if mode != self._supported_mode:
            semantic = "detour" if self._supported_mode == 0 else "stop-on-obstacle"
            raise CommandError(
                "mode_not_supported",
                f"current Nav2 shadow profile only supports mode="
                f"{self._supported_mode} ({semantic}); requested mode={mode}",
            )
        self._assert_shadow_isolated()
        with self._lock:
            if self._active and self._active.get("status") not in _TERMINAL_STATES:
                raise CommandError(
                    "navigation_active",
                    f"navigation {self._active['nav_id']} is already active",
                )
            self._active = {
                "nav_id": nav_id,
                "status": "starting",
                "target_pose": {
                    "x": float(args["x"]),
                    "y": float(args["y"]),
                    "yaw": float(args["yaw"]),
                },
                "requested_speed": float(args["speed"]),
                "effective_speed_limit": min(
                    float(args["speed"]), self._max_shadow_speed
                ),
                "mode": mode,
                "attempt": 0,
                "goal_handle": None,
                "cancel_intent": None,
                "progress_seq": 0,
                "last_distance": None,
                "last_pose": None,
                "last_feedback_publish": 0.0,
            }
        try:
            self._send_active_goal()
        except Exception:
            with self._lock:
                if self._active and self._active.get("nav_id") == nav_id:
                    self._active["status"] = "error"
            self._publish_state()
            raise
        with self._lock:
            active = self._require_active(nav_id)
            return {
                "status": "navigating",
                "nav_id": nav_id,
                "target_pose": dict(active["target_pose"]),
                "requested_speed": active["requested_speed"],
                "effective_speed_limit": active["effective_speed_limit"],
                "speed_policy": "nav2_absolute_controller_limit",
                "speed_limit_topic": self._controller_speed_limit_topic,
                "mode": active["mode"],
                "shadow_only": True,
            }

    def _send_active_goal(self) -> None:
        if not self._action_client.wait_for_server(timeout_sec=3.0):
            raise CommandError(
                "nav2_action_unavailable",
                f"action server {self._action_name} is unavailable",
            )
        with self._lock:
            if self._active is None:
                raise CommandError("no_active_navigation", "navigation disappeared")
            self._active["attempt"] += 1
            attempt = self._active["attempt"]
            nav_id = self._active["nav_id"]
            target = dict(self._active["target_pose"])
            speed_limit = float(self._active["effective_speed_limit"])
            self._active["status"] = "starting"
            self._active["cancel_intent"] = None
            self._active["goal_handle"] = None

        self._apply_controller_speed_limit(speed_limit)

        goal = NavigateToPose.Goal()
        goal.behavior_tree = self._behavior_tree_path
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = target["x"]
        goal.pose.pose.position.y = target["y"]
        goal.pose.pose.orientation.z = math.sin(target["yaw"] / 2.0)
        goal.pose.pose.orientation.w = math.cos(target["yaw"] / 2.0)

        accepted = threading.Event()
        outcome: dict = {}
        future = self._action_client.send_goal_async(
            goal,
            feedback_callback=lambda feedback: self._on_feedback(
                nav_id, attempt, feedback
            ),
        )
        future.add_done_callback(
            lambda completed: self._on_goal_response(
                nav_id, attempt, completed, accepted, outcome
            )
        )
        if not accepted.wait(timeout=self._goal_response_timeout):
            with self._lock:
                if self._active and self._active.get("nav_id") == nav_id:
                    self._active["attempt"] += 1
                    self._active["status"] = "error"
                    self._active["error_code"] = "goal_response_timeout"
            self._publish_state()
            raise CommandError(
                "goal_response_timeout",
                f"Nav2 did not answer goal request within "
                f"{self._goal_response_timeout:.1f}s",
            )
        if not outcome.get("accepted"):
            raise CommandError(
                str(outcome.get("error_code", "goal_rejected")),
                str(outcome.get("error", "Nav2 rejected the goal")),
            )

    def _apply_controller_speed_limit(self, speed_limit: float) -> None:
        deadline = time.monotonic() + self._speed_limit_timeout
        while self._speed_limit_pub.get_subscription_count() == 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CommandError(
                    "speed_limit_unavailable",
                    "Nav2 controller has no subscriber on "
                    f"{self._controller_speed_limit_topic}",
                )
            time.sleep(min(0.05, remaining))

        message = SpeedLimit()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.percentage = False
        message.speed_limit = speed_limit
        self._speed_limit_pub.publish(message)
        remaining = max(0.0, deadline - time.monotonic())
        if not self._speed_limit_pub.wait_for_all_acked(
            Duration(seconds=remaining)
        ):
            raise CommandError(
                "speed_limit_unconfirmed",
                "Nav2 controller did not acknowledge speed limit "
                f"{speed_limit:.3f} m/s",
            )

    def _on_goal_response(
        self,
        nav_id: str,
        attempt: int,
        future,
        accepted: threading.Event,
        outcome: dict,
    ) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            with self._lock:
                if (
                    self._active is not None
                    and self._active.get("nav_id") == nav_id
                    and self._active.get("attempt") == attempt
                ):
                    self._active["status"] = "error"
                    self._active["error"] = f"{type(exc).__name__}: {exc}"
            outcome.update(
                {
                    "accepted": False,
                    "error_code": "goal_request_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            accepted.set()
            self._publish_state()
            return

        with self._lock:
            stale = (
                self._active is None
                or self._active.get("nav_id") != nav_id
                or self._active.get("attempt") != attempt
            )
            if stale:
                if goal_handle.accepted:
                    goal_handle.cancel_goal_async()
                outcome.update(
                    {
                        "accepted": False,
                        "error_code": "stale_goal_response",
                        "error": "late goal response was cancelled",
                    }
                )
            elif not goal_handle.accepted:
                self._active["status"] = "rejected"
                outcome.update(
                    {
                        "accepted": False,
                        "error_code": "goal_rejected",
                        "error": "Nav2 rejected the goal",
                    }
                )
            else:
                self._active["goal_handle"] = goal_handle
                self._active["status"] = "navigating"
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(
                    lambda completed: self._on_result(
                        nav_id, attempt, completed
                    )
                )
                outcome["accepted"] = True
        accepted.set()
        self._publish_state()

    def _on_feedback(self, nav_id: str, attempt: int, wrapper) -> None:
        feedback = wrapper.feedback
        distance = float(feedback.distance_remaining)
        pose = feedback.current_pose.pose.position
        now = time.monotonic()
        publish = False
        with self._state_changed:
            if (
                self._active is None
                or self._active.get("nav_id") != nav_id
                or self._active.get("attempt") != attempt
            ):
                return
            last_distance = self._active.get("last_distance")
            last_pose = self._active.get("last_pose")
            progressed = False
            if math.isfinite(distance):
                if last_distance is None or distance < last_distance - 0.02:
                    progressed = True
                self._active["last_distance"] = distance
            current_pose = (float(pose.x), float(pose.y))
            if last_pose is None or math.hypot(
                current_pose[0] - last_pose[0], current_pose[1] - last_pose[1]
            ) > 0.03:
                progressed = True
            self._active["last_pose"] = current_pose
            if progressed:
                self._active["progress_seq"] += 1
            if now - self._active["last_feedback_publish"] >= 0.5:
                self._active["last_feedback_publish"] = now
                publish = True
        if publish:
            self._publish_state()

    def _on_result(self, nav_id: str, attempt: int, future) -> None:
        try:
            wrapped = future.result()
            status_code = wrapped.status
        except Exception as exc:
            status_code = GoalStatus.STATUS_UNKNOWN
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = ""

        with self._lock:
            if (
                self._active is None
                or self._active.get("nav_id") != nav_id
                or self._active.get("attempt") != attempt
            ):
                return
            intent = self._active.get("cancel_intent")
            if intent == "pause":
                self._active["status"] = "paused"
            elif intent == "stop":
                self._active["status"] = "stopped"
            elif status_code == GoalStatus.STATUS_SUCCEEDED:
                self._active["status"] = "arrived"
            elif status_code == GoalStatus.STATUS_CANCELED:
                self._active["status"] = "cancelled"
            elif status_code == GoalStatus.STATUS_ABORTED:
                self._active["status"] = "aborted"
                self._active["error"] = "Nav2 aborted the goal"
            else:
                self._active["status"] = "error"
                self._active["error"] = error or f"unexpected goal status {status_code}"
            self._active["goal_handle"] = None
            self._state_changed.notify_all()
        self._publish_state()

    def _pause(self, nav_id) -> dict:
        active = self._require_matching_navigation(nav_id)
        if active["status"] == "paused":
            return {"status": "paused", "nav_id": active["nav_id"], "already_paused": True}
        if active["status"] != "navigating":
            raise CommandError(
                "invalid_navigation_state",
                f"cannot pause navigation in state {active['status']}",
            )
        self._cancel_active("pause")
        return {"status": "paused", "nav_id": nav_id}

    def _resume(self, nav_id) -> dict:
        active = self._require_matching_navigation(nav_id)
        if active["status"] != "paused":
            raise CommandError(
                "invalid_navigation_state",
                f"cannot resume navigation in state {active['status']}",
            )
        self._require_navigation_ready("resume_nav")
        self._assert_shadow_isolated()
        try:
            self._send_active_goal()
        except Exception:
            with self._lock:
                if self._active and self._active.get("nav_id") == nav_id:
                    self._active["status"] = "error"
            self._publish_state()
            raise
        return {"status": "navigating", "nav_id": nav_id, "resumed": True}

    def _stop(self, nav_id) -> dict:
        with self._lock:
            if self._active is None:
                return {"status": "stopped", "nav_id": nav_id, "already_idle": True}
            active = self._require_active(nav_id)
            if active["status"] in _TERMINAL_STATES:
                return {
                    "status": active["status"],
                    "nav_id": nav_id,
                    "already_terminal": active["status"],
                }
            has_goal = active.get("goal_handle") is not None
            if not has_goal:
                active["attempt"] += 1
                active["status"] = "stopping"
        if not has_goal:
            self._publish_state()
            return {
                "status": "stopping",
                "nav_id": nav_id,
                "terminal_confirmed": False,
            }
        self._cancel_active("stop")
        return {
            "status": "stopped",
            "nav_id": nav_id,
            "terminal_confirmed": True,
        }

    def _cancel_active(self, intent: str) -> None:
        with self._lock:
            if self._active is None or self._active.get("goal_handle") is None:
                raise CommandError(
                    "invalid_navigation_state", "Nav2 goal handle is unavailable"
                )
            goal_handle = self._active["goal_handle"]
            nav_id = self._active["nav_id"]
            self._active["cancel_intent"] = intent
        completed = threading.Event()
        outcome: dict = {}
        future = goal_handle.cancel_goal_async()

        def _done(cancel_future) -> None:
            try:
                response = cancel_future.result()
                outcome["accepted"] = bool(response.goals_canceling)
            except Exception as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"
            completed.set()

        future.add_done_callback(_done)
        if not completed.wait(timeout=5.0) or not outcome.get("accepted"):
            with self._lock:
                if self._active is not None:
                    self._active["cancel_intent"] = None
            raise CommandError(
                "cancel_failed", outcome.get("error", "Nav2 did not acknowledge cancel")
            )
        expected = "paused" if intent == "pause" else "stopped"
        deadline = time.monotonic() + 5.0
        with self._state_changed:
            while True:
                if self._active is None or self._active.get("nav_id") != nav_id:
                    raise CommandError(
                        "cancel_terminal_unconfirmed",
                        "navigation disappeared before terminal receipt",
                    )
                if self._active.get("status") == expected:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CommandError(
                        "cancel_terminal_unconfirmed",
                        f"Nav2 accepted cancel but did not report {expected}",
                    )
                self._state_changed.wait(timeout=remaining)

    def _assert_shadow_isolated(self) -> None:
        if not self._enforce_shadow_isolation:
            raise CommandError(
                "unsafe_configuration",
                "enforce_shadow_isolation=false is forbidden by the Nav2 card contract",
            )
        root_publishers = self.get_publishers_info_by_topic("/cmd_vel")
        if root_publishers:
            raise CommandError(
                "shadow_isolation_failed",
                "root /cmd_vel has publishers; refusing a shadow goal",
            )
        shadow_subscribers = self.get_subscriptions_info_by_topic(self._shadow_topic)
        own_namespace = self.get_namespace().rstrip("/") or "/"
        foreign_subscribers = []
        for endpoint in shadow_subscribers:
            endpoint_topic_type = getattr(endpoint, "topic_type", "")
            if endpoint_topic_type and endpoint_topic_type != "geometry_msgs/msg/Twist":
                continue
            endpoint_namespace = endpoint.node_namespace.rstrip("/") or "/"
            if (
                endpoint.node_name == self.get_name()
                and endpoint_namespace == own_namespace
            ):
                continue
            foreign_subscribers.append(endpoint)
        if foreign_subscribers:
            names = sorted(
                f"{endpoint.node_namespace}/{endpoint.node_name}"
                f"[{getattr(endpoint, 'topic_type', 'unknown')}]"
                for endpoint in foreign_subscribers
            )
            raise CommandError(
                "shadow_isolation_failed",
                "raw shadow output has foreign subscribers: " + ",".join(names),
            )

    def _publish_velocity_proposal(
        self,
        *,
        nav_id: str,
        navigation_status: str,
        velocity: Velocity,
        reason: str | None = None,
    ) -> None:
        with self._lock:
            self._proposal_sequence += 1
            sequence = self._proposal_sequence
        wire_status = (
            "planning" if navigation_status == "starting" else navigation_status
        )
        payload = build_velocity_proposal(
            nav_id=nav_id,
            sequence=sequence,
            ttl_ms=self._proposal_ttl_ms,
            navigation_status=wire_status,
            velocity=velocity,
            reason=reason,
        )
        message = String()
        message.data = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        self._proposal_pub.publish(message)

    def _require_matching_navigation(self, nav_id) -> dict:
        with self._lock:
            return dict(self._require_active(nav_id))

    def _require_active(self, nav_id) -> dict:
        if self._active is None:
            raise CommandError("no_active_navigation", "no navigation is active")
        if not isinstance(nav_id, str) or nav_id != self._active.get("nav_id"):
            raise CommandError(
                "navigation_id_mismatch",
                f"active navigation is {self._active.get('nav_id')}",
            )
        return self._active

    def _respond(
        self, request_id: str, action: str, nav_id, result: dict
    ) -> None:
        payload = {
            "event": "response",
            "request_id": request_id,
            "action": action,
            "nav_id": nav_id,
            "shadow_only": True,
            "physical_execution": False,
            **result,
        }
        self._emit(payload)

    def _publish_state(self) -> None:
        stop_proposal = None
        with self._lock:
            if self._active is None:
                if self._mapping_session is not None:
                    status = "mapping"
                elif self._runtime_mode == "localization" and self._active_map_name:
                    status = "localized"
                elif self._mapping_closed:
                    status = "map_saved"
                else:
                    status = "idle"
                payload = {"event": "navigation_status", "status": status}
            else:
                payload = {
                    key: value
                    for key, value in self._active.items()
                    if key
                    not in {
                        "goal_handle",
                        "last_feedback_publish",
                        "last_pose",
                        "cancel_intent",
                    }
                }
                payload["event"] = "navigation_status"
                if payload.get("status") in _IDLE_OR_TERMINAL_STATES:
                    stop_proposal = (
                        str(payload["nav_id"]),
                        str(payload["status"]),
                    )
            payload.update(
                {
                    "runtime_mode": self._runtime_mode,
                    "active_map": self._active_map_name,
                    "mapping_map": self._mapping_session.map_name
                    if self._mapping_session
                    else None,
                }
            )
        payload.update(self._readiness())
        self._emit(payload)
        if stop_proposal is not None:
            nav_id, status = stop_proposal
            self._publish_velocity_proposal(
                nav_id=nav_id,
                navigation_status=status,
                velocity=Velocity.zero(),
                reason=f"navigation_{status}",
            )

    def _publish_heartbeat(self) -> None:
        with self._lock:
            payload = {
                "event": "heartbeat",
                "status": self._active.get("status", "idle")
                if self._active
                else "idle",
                "nav_id": self._active.get("nav_id") if self._active else None,
                "progress_seq": self._active.get("progress_seq", 0)
                if self._active
                else 0,
                "supported_modes": [self._supported_mode],
                "max_shadow_speed": self._max_shadow_speed,
                "runtime_mode": self._runtime_mode,
                "active_map": self._active_map_name,
                "mapping_map": self._mapping_session.map_name
                if self._mapping_session
                else None,
                "n5_protocol_ready": True,
                "velocity_proposal_topic": self._proposal_topic,
                "proposal_ttl_ms": self._proposal_ttl_ms,
                "proposal_subscribers": self._proposal_pub.get_subscription_count(),
                "controller_speed_limit_topic": self._controller_speed_limit_topic,
                "controller_speed_limit_subscribers": (
                    self._speed_limit_pub.get_subscription_count()
                ),
            }
        payload.update(self._readiness())
        self._emit(payload)

    def _emit(self, payload: dict) -> None:
        payload = {
            **payload,
            "timestamp": time.time(),
            "shadow_only": True,
            "physical_execution": False,
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationCommandNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
