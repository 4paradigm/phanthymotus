"""Fail-closed planner/controller bridge from the ActuCore card to Nav2."""

from __future__ import annotations

from utils import logsafe

logsafe.install()

import json
import math
import os
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
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
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .execution_protocol import (
    MotionLimits,
    ProtocolError,
    Velocity,
    apply_g1_motion_limits,
    build_velocity_proposal,
    proposal_context_is_publishable,
)
from .costmap_validation import (
    CostmapError,
    CostmapSnapshot,
    GoalCellRejected,
    validated_goal_cell_receipt,
)
from .readiness import control_odom_motion_blocker, evaluate_readiness


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


def _stamp_ns(message) -> int | None:
    stamp = message.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


class PlannerCommandNode(Node):
    """Own navigation lifecycle while FAST-LIVO2 owns localization and mapping."""

    def __init__(self) -> None:
        super().__init__("g1_nav2_planner_command")
        self.declare_parameter("command_topic", "/ubuntu/navigation/nav2/command")
        self.declare_parameter("status_topic", "/ubuntu/navigation/nav2/status")
        self.declare_parameter(
            "segment_status_topic", "/ubuntu/navigation/nav2/segment_status"
        )
        self.declare_parameter("action_name", "/navigate_to_pose")
        self.declare_parameter(
            "shadow_topic", "/ubuntu/navigation/nav2/cmd_vel_shadow"
        )
        self.declare_parameter(
            "proposal_topic", "/ubuntu/navigation/nav2/velocity_proposal"
        )
        self.declare_parameter(
            "controller_speed_limit_topic", "/ubuntu/navigation/nav2/speed_limit"
        )
        self.declare_parameter("speed_limit_timeout", 3.0)
        self.declare_parameter("behavior_tree_path", "")
        self.declare_parameter("proposal_ttl_ms", 250)
        self.declare_parameter("proposal_frequency_hz", 5.0)
        self.declare_parameter("enforce_shadow_isolation", True)
        self.declare_parameter("max_shadow_speed", 1.0)
        self.declare_parameter("supported_mode", 0)
        self.declare_parameter("goal_response_timeout", 8.0)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_topic", "/ubuntu/navigation/odom")
        self.declare_parameter(
            "obstacle_cloud_topic", "/ubuntu/navigation/cloud_registered"
        )
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("goal_costmap_max_age_sec", 2.0)
        self.declare_parameter("sensor_max_age_sec", 0.8)
        self.declare_parameter("sensor_source_max_age_sec", 1.0)
        self.declare_parameter("control_odom_max_age_sec", 0.60)
        self.declare_parameter("control_odom_source_max_age_sec", 0.80)
        self.declare_parameter(
            "required_lifecycle_nodes",
            [
                "controller_server",
                "planner_server",
                "bt_navigator",
            ],
        )

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._segment_status_topic = str(
            self.get_parameter("segment_status_topic").value
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
        self._proposal_ttl_ms = int(self.get_parameter("proposal_ttl_ms").value)
        self._proposal_frequency_hz = float(
            self.get_parameter("proposal_frequency_hz").value
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
        self._global_frame = str(self.get_parameter("global_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._obstacle_cloud_topic = str(
            self.get_parameter("obstacle_cloud_topic").value
        )
        self._global_costmap_topic = str(
            self.get_parameter("global_costmap_topic").value
        )
        self._goal_costmap_max_age_sec = float(
            self.get_parameter("goal_costmap_max_age_sec").value
        )
        self._sensor_max_age_sec = float(
            self.get_parameter("sensor_max_age_sec").value
        )
        self._sensor_source_max_age_sec = float(
            self.get_parameter("sensor_source_max_age_sec").value
        )
        self._control_odom_max_age_sec = float(
            self.get_parameter("control_odom_max_age_sec").value
        )
        self._control_odom_source_max_age_sec = float(
            self.get_parameter("control_odom_source_max_age_sec").value
        )
        self._required_lifecycle_nodes = [
            str(item).strip("/")
            for item in self.get_parameter("required_lifecycle_nodes").value
        ]

        if not 0.0 < self._max_shadow_speed <= 1.0:
            raise ValueError("max_shadow_speed must be within (0, 1.0]")
        if not 50 <= self._proposal_ttl_ms <= 250:
            raise ValueError("proposal_ttl_ms must be within [50, 250]")
        if not math.isfinite(self._proposal_frequency_hz) or not (
            1.0 <= self._proposal_frequency_hz <= 20.0
        ):
            raise ValueError("proposal_frequency_hz must be within [1, 20]")
        if self._supported_mode != 0:
            raise ValueError("supported_mode must be 0 until another mode is implemented")
        if self._goal_response_timeout <= 0 or self._speed_limit_timeout <= 0:
            raise ValueError("goal and speed-limit timeouts must be positive")
        if not os.path.isfile(self._behavior_tree_path):
            raise ValueError(
                "behavior_tree_path must point to the installed G1 Nav2 tree"
            )
        if not self._global_frame or not self._base_frame:
            raise ValueError("global_frame and base_frame must not be empty")
        if self._sensor_max_age_sec <= 0:
            raise ValueError("sensor_max_age_sec must be positive")
        if not (
            self._sensor_max_age_sec
            <= self._sensor_source_max_age_sec
            <= 2.0
        ):
            raise ValueError(
                "sensor_source_max_age_sec must be within "
                "[sensor_max_age_sec, 2.0]"
            )
        if self._goal_costmap_max_age_sec <= 0:
            raise ValueError("goal_costmap_max_age_sec must be positive")
        if not 0.0 < self._control_odom_max_age_sec <= self._sensor_max_age_sec:
            raise ValueError(
                "control_odom_max_age_sec must be within (0, sensor_max_age_sec]"
            )
        if not (
            self._control_odom_max_age_sec
            <= self._control_odom_source_max_age_sec
            <= self._sensor_source_max_age_sec
        ):
            raise ValueError(
                "control_odom_source_max_age_sec must be within "
                "[control_odom_max_age_sec, sensor_source_max_age_sec]"
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
        latest_command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
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
        self._proposal_pub = self.create_publisher(
            String, self._proposal_topic, latest_command_qos
        )
        self._speed_limit_pub = self.create_publisher(
            SpeedLimit, self._controller_speed_limit_topic, command_qos
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
            latest_command_qos,
            callback_group=self._callbacks,
        )
        self._segment_status_sub = self.create_subscription(
            String,
            self._segment_status_topic,
            self._on_segment_status,
            qos_profile_sensor_data,
            callback_group=self._callbacks,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            self._odom_topic,
            self._on_odom,
            qos_profile_sensor_data,
            callback_group=self._callbacks,
        )
        self._obstacle_cloud_sub = self.create_subscription(
            PointCloud2,
            self._obstacle_cloud_topic,
            self._on_obstacle_cloud,
            qos_profile_sensor_data,
            callback_group=self._callbacks,
        )
        costmap_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._global_costmap_sub = self.create_subscription(
            OccupancyGrid,
            self._global_costmap_topic,
            self._on_global_costmap,
            costmap_qos,
            callback_group=self._callbacks,
        )
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            self._action_name,
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
        self._proposal_timer = self.create_timer(
            1.0 / self._proposal_frequency_hz,
            self._publish_latest_velocity_proposal,
            callback_group=self._callbacks,
        )

        self._lock = threading.RLock()
        self._state_changed = threading.Condition(self._lock)
        self._command_lock = threading.Lock()
        self._active: dict | None = None
        self._proposal_sequence = 0
        self._latest_proposal_candidate: dict | None = None
        self._last_published_proposal: dict | None = None
        self._last_odom_monotonic: float | None = None
        self._last_odom_source_stamp_ns: int | None = None
        self._last_odom_frame_ready = False
        self._segment_status: dict = {}
        self._last_obstacle_monotonic: float | None = None
        self._last_obstacle_source_stamp_ns: int | None = None
        self._last_obstacle_frame_ready = False
        self._global_costmap: CostmapSnapshot | None = None
        self._global_costmap_error: str | None = None
        self._lifecycle_states = {
            name: 0 for name in self._required_lifecycle_nodes
        }
        self._publish_heartbeat()

    def _source_age(self, stamp_ns: int | None) -> float | None:
        if stamp_ns is None:
            return None
        return (self.get_clock().now().nanoseconds - stamp_ns) / 1_000_000_000.0

    def _on_odom(self, message: Odometry) -> None:
        stamp_ns = _stamp_ns(message)
        with self._lock:
            self._last_odom_monotonic = time.monotonic()
            self._last_odom_source_stamp_ns = stamp_ns
            self._last_odom_frame_ready = (
                message.header.frame_id.strip("/") == self._global_frame.strip("/")
                and message.child_frame_id.strip("/") == self._base_frame.strip("/")
            )

    def _on_segment_status(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(status, dict) or not isinstance(status.get("phase"), str):
            return
        with self._lock:
            self._segment_status = status

    def _on_obstacle_cloud(self, message: PointCloud2) -> None:
        stamp_ns = _stamp_ns(message)
        with self._lock:
            self._last_obstacle_monotonic = time.monotonic()
            self._last_obstacle_source_stamp_ns = stamp_ns
            self._last_obstacle_frame_ready = (
                message.header.frame_id.strip("/") == self._global_frame.strip("/")
            )

    def _on_global_costmap(self, message: OccupancyGrid) -> None:
        orientation = message.info.origin.orientation
        origin_yaw = math.atan2(
            2.0
            * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0
            - 2.0
            * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        try:
            snapshot = CostmapSnapshot.from_values(
                frame_id=message.header.frame_id,
                stamp_ns=_stamp_ns(message) or 0,
                resolution=float(message.info.resolution),
                width=int(message.info.width),
                height=int(message.info.height),
                origin_x=float(message.info.origin.position.x),
                origin_y=float(message.info.origin.position.y),
                origin_yaw=origin_yaw,
                data=message.data,
            )
        except CostmapError as exc:
            with self._lock:
                self._global_costmap_error = str(exc)
            return
        with self._lock:
            self._global_costmap = snapshot
            self._global_costmap_error = None

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
            odom_received_at = self._last_odom_monotonic
            odom_source_stamp_ns = self._last_odom_source_stamp_ns
            odom_frame_ready = self._last_odom_frame_ready
            obstacle_received_at = self._last_obstacle_monotonic
            obstacle_source_stamp_ns = self._last_obstacle_source_stamp_ns
            obstacle_frame_ready = self._last_obstacle_frame_ready
            lifecycle_states = dict(self._lifecycle_states)
        odom_source_age = self._source_age(odom_source_stamp_ns)
        obstacle_source_age = self._source_age(obstacle_source_stamp_ns)
        source_skew = None
        if odom_source_stamp_ns is not None and obstacle_source_stamp_ns is not None:
            source_skew = abs(
                odom_source_stamp_ns - obstacle_source_stamp_ns
            ) / 1_000_000_000.0
        source_transform_ready = False
        if obstacle_source_stamp_ns is not None:
            source_transform_ready = self._tf_buffer.can_transform(
                self._global_frame,
                self._base_frame,
                Time(nanoseconds=obstacle_source_stamp_ns),
                timeout=Duration(seconds=0.0),
            )
        global_to_base_ready = self._tf_buffer.can_transform(
            self._global_frame,
            self._base_frame,
            Time(),
            timeout=Duration(seconds=0.0),
        )
        return evaluate_readiness(
            now_monotonic=time.monotonic(),
            max_age_sec=self._sensor_max_age_sec,
            source_max_age_sec=self._sensor_source_max_age_sec,
            odom_received_at=odom_received_at,
            odom_source_age_sec=odom_source_age,
            odom_frame_ready=odom_frame_ready,
            obstacle_received_at=obstacle_received_at,
            obstacle_source_age_sec=obstacle_source_age,
            obstacle_frame_ready=obstacle_frame_ready,
            source_transform_ready=source_transform_ready,
            source_stamp_skew_sec=source_skew,
            lifecycle_states=lifecycle_states,
            action_server_ready=self._action_client.server_is_ready(),
            global_to_base_ready=global_to_base_ready,
        )

    def _global_costmap_diagnostics(self) -> dict:
        with self._lock:
            snapshot = self._global_costmap
            error = self._global_costmap_error
        if snapshot is None:
            return {
                "state": "error" if error else "waiting",
                "topic": self._global_costmap_topic,
                "error": error,
            }
        return {
            "topic": self._global_costmap_topic,
            **snapshot.diagnostics(),
        }

    def _validate_goal_cell(self, target: dict) -> dict:
        with self._lock:
            snapshot = self._global_costmap
        try:
            return validated_goal_cell_receipt(
                snapshot,
                x=float(target["x"]),
                y=float(target["y"]),
                expected_frame=self._global_frame,
                max_receive_age_sec=self._goal_costmap_max_age_sec,
            )
        except GoalCellRejected as exc:
            raise CommandError(exc.code, str(exc)) from exc

    def _require_navigation_ready(self, action: str) -> None:
        receipt = self._readiness()
        if not receipt["navigation_ready"]:
            raise CommandError(
                "navigation_not_ready",
                f"{action} blocked: " + ",".join(receipt["navigation_blockers"]),
            )

    def _on_shadow_velocity(self, message: Twist) -> None:
        with self._lock:
            if self._active is None:
                return
            nav_id = self._active.get("nav_id")
            attempt = self._active.get("attempt")
            status = self._active.get("status", "error")
            forward_speed_limit = self._active.get("effective_speed_limit")
            motion_limits = self._active.get("motion_limits")
        if not isinstance(nav_id, str) or not nav_id:
            return
        if not isinstance(motion_limits, MotionLimits):
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
            reason = control_odom_motion_blocker(
                self._readiness(),
                receive_max_age_sec=self._control_odom_max_age_sec,
                source_max_age_sec=self._control_odom_source_max_age_sec,
            )
            if reason is not None:
                velocity = Velocity.zero()
        try:
            if reason is None:
                velocity = apply_g1_motion_limits(
                    velocity,
                    limits=motion_limits,
                    max_forward_mps=forward_speed_limit,
                )
            with self._lock:
                if not proposal_context_is_publishable(
                    self._active,
                    nav_id=nav_id,
                    attempt=attempt,
                    status=status,
                ):
                    return
                self._latest_proposal_candidate = {
                    "nav_id": nav_id,
                    "attempt": attempt,
                    "navigation_status": status,
                    "velocity": velocity,
                    "reason": reason,
                    "received_monotonic": time.monotonic(),
                }
                publish_safety_zero = velocity == Velocity.zero() and (
                    self._last_published_proposal is None
                    or self._last_published_proposal.get("nav_id") != nav_id
                    or self._last_published_proposal.get("velocity")
                    != Velocity.zero()
                    or self._last_published_proposal.get("reason") != reason
                )
            if publish_safety_zero:
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
                if proposal_context_is_publishable(
                    self._active,
                    nav_id=nav_id,
                    attempt=attempt,
                    status=status,
                ):
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
            try:
                motion_limits = MotionLimits.from_payload(
                    payload.get("velocity_limits")
                )
            except ProtocolError as exc:
                raise CommandError(exc.code, str(exc)) from exc
            with self._command_lock:
                result = self._dispatch(action, args, nav_id, motion_limits)
            self._respond(request_id, action, nav_id, result)
        except CommandError as exc:
            self._respond(
                request_id,
                action,
                nav_id,
                {"status": "error", "error_code": exc.code, "error": str(exc)},
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
        self,
        action: str,
        args: dict,
        nav_id,
        motion_limits: MotionLimits,
    ) -> dict:
        if action == "navigate_to_pose":
            if not isinstance(nav_id, str) or not nav_id:
                raise CommandError("invalid_request", "nav_id is required")
            return self._navigate_to_pose(nav_id, args, motion_limits)
        if action == "pause_nav":
            return self._pause(nav_id)
        if action == "resume_nav":
            return self._resume(nav_id)
        if action == "stop_nav":
            return self._stop(nav_id)
        if action == "wait_navigation_done":
            raise CommandError(
                "invalid_request",
                "wait_navigation_done is handled by the ActuCore adapter",
            )
        raise CommandError("unsupported_action", f"unsupported action: {action}")

    def _navigate_to_pose(
        self,
        nav_id: str,
        args: dict,
        motion_limits: MotionLimits,
    ) -> dict:
        self._require_navigation_ready("navigate_to_pose")
        try:
            mode = int(args["mode"])
            target = {
                "x": float(args["x"]),
                "y": float(args["y"]),
                "yaw": float(args["yaw"]),
            }
            requested_speed = float(args["speed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CommandError(
                "invalid_argument", "x, y, yaw, speed and mode are required"
            ) from exc
        if mode != self._supported_mode:
            raise CommandError(
                "mode_not_supported",
                f"current planner profile only supports mode={self._supported_mode}; "
                f"requested mode={mode}",
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
                "target_pose": target,
                "target_frame": self._global_frame,
                "requested_speed": requested_speed,
                "effective_speed_limit": min(
                    requested_speed, self._max_shadow_speed
                ),
                "motion_limits": motion_limits,
                "mode": mode,
                "goal_cell": None,
                "attempt": 0,
                "goal_handle": None,
                "cancel_intent": None,
                "progress_seq": 0,
                "last_distance": None,
                "last_pose": None,
                "last_feedback_publish": 0.0,
            }
            self._latest_proposal_candidate = None
            self._last_published_proposal = None
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
                "target_frame": active["target_frame"],
                "requested_speed": active["requested_speed"],
                "effective_speed_limit": active["effective_speed_limit"],
                "speed_policy": "proposal_enforced_with_controller_advisory",
                "speed_limit_topic": self._controller_speed_limit_topic,
                "velocity_limits": active["motion_limits"].as_dict(),
                "mode": active["mode"],
                "goal_cell": dict(active["goal_cell"]),
                "shadow_only": True,
            }

    def _send_active_goal(self) -> None:
        with self._lock:
            if self._active is None:
                raise CommandError("no_active_navigation", "navigation disappeared")
            nav_id = self._active["nav_id"]
            target = dict(self._active["target_pose"])
        goal_cell = self._validate_goal_cell(target)
        if not self._action_client.wait_for_server(timeout_sec=3.0):
            raise CommandError(
                "nav2_action_unavailable",
                f"action server {self._action_name} is unavailable",
            )
        with self._lock:
            if self._active is None or self._active.get("nav_id") != nav_id:
                raise CommandError("no_active_navigation", "navigation disappeared")
            self._active["attempt"] += 1
            attempt = self._active["attempt"]
            speed_limit = float(self._active["effective_speed_limit"])
            self._active["goal_cell"] = goal_cell
            self._active["status"] = "starting"
            self._active["cancel_intent"] = None
            self._active["goal_handle"] = None

        self._publish_controller_speed_limit(speed_limit)
        goal = NavigateToPose.Goal()
        goal.behavior_tree = self._behavior_tree_path
        goal.pose.header.frame_id = self._global_frame
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
                f"Nav2 did not answer within {self._goal_response_timeout:.1f}s",
            )
        if not outcome.get("accepted"):
            raise CommandError(
                str(outcome.get("error_code", "goal_rejected")),
                str(outcome.get("error", "Nav2 rejected the goal")),
            )

    def _publish_controller_speed_limit(self, speed_limit: float) -> None:
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
        message.header.frame_id = self._base_frame
        message.percentage = False
        message.speed_limit = speed_limit
        self._speed_limit_pub.publish(message)

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
                goal_handle.get_result_async().add_done_callback(
                    lambda completed: self._on_result(nav_id, attempt, completed)
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
        with self._state_changed:
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
                active["status"] = "stopped"
        if has_goal:
            self._cancel_active("stop")
        else:
            self._publish_state()
        return {"status": "stopped", "nav_id": nav_id, "terminal_confirmed": True}

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
        if self.get_publishers_info_by_topic("/cmd_vel"):
            raise CommandError(
                "shadow_isolation_failed",
                "root /cmd_vel has publishers; refusing a shadow goal",
            )
        own_namespace = self.get_namespace().rstrip("/") or "/"
        foreign_subscribers = []
        for endpoint in self.get_subscriptions_info_by_topic(self._shadow_topic):
            endpoint_topic_type = getattr(endpoint, "topic_type", "")
            if endpoint_topic_type and endpoint_topic_type != "geometry_msgs/msg/Twist":
                continue
            endpoint_namespace = endpoint.node_namespace.rstrip("/") or "/"
            if endpoint.node_name == self.get_name() and endpoint_namespace == own_namespace:
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
        wire_status = "planning" if navigation_status == "starting" else navigation_status
        payload = build_velocity_proposal(
            nav_id=nav_id,
            sequence=sequence,
            ttl_ms=self._proposal_ttl_ms,
            navigation_status=wire_status,
            velocity=velocity,
            reason=reason,
        )
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._proposal_pub.publish(message)
        with self._lock:
            self._last_published_proposal = {
                "nav_id": nav_id,
                "navigation_status": navigation_status,
                "velocity": velocity,
                "reason": reason,
            }

    def _publish_latest_velocity_proposal(self) -> None:
        """Publish only the newest Nav2 sample at the G1 execution cadence."""
        with self._lock:
            candidate = (
                dict(self._latest_proposal_candidate)
                if self._latest_proposal_candidate is not None
                else None
            )
            active = dict(self._active) if self._active is not None else None
        if candidate is None or active is None:
            return

        nav_id = candidate.get("nav_id")
        attempt = candidate.get("attempt")
        status = candidate.get("navigation_status")
        if not proposal_context_is_publishable(
            active,
            nav_id=nav_id,
            attempt=attempt,
            status=status,
        ):
            return

        velocity = candidate.get("velocity")
        reason = candidate.get("reason")
        if not isinstance(velocity, Velocity):
            return
        candidate_age = time.monotonic() - float(
            candidate.get("received_monotonic", 0.0)
        )
        if candidate_age > self._proposal_ttl_ms / 1000.0:
            velocity = Velocity.zero()
            reason = "shadow_velocity_stale"
        else:
            blocker = control_odom_motion_blocker(
                self._readiness(),
                receive_max_age_sec=self._control_odom_max_age_sec,
                source_max_age_sec=self._control_odom_source_max_age_sec,
            )
            if blocker is not None:
                velocity = Velocity.zero()
                reason = blocker

        with self._lock:
            if not proposal_context_is_publishable(
                self._active,
                nav_id=nav_id,
                attempt=attempt,
                status=status,
            ):
                return
        self._publish_velocity_proposal(
            nav_id=str(nav_id),
            navigation_status=str(status),
            velocity=velocity,
            reason=reason,
        )

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

    def _respond(self, request_id: str, action: str, nav_id, result: dict) -> None:
        self._emit(
            {
                "event": "response",
                "request_id": request_id,
                "action": action,
                "nav_id": nav_id,
                "shadow_only": True,
                "physical_execution": False,
                **result,
            }
        )

    def _publish_state(self) -> None:
        stop_proposal = None
        with self._lock:
            if self._active is None:
                payload = {"event": "navigation_status", "status": "idle"}
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
                        "motion_limits",
                    }
                }
                payload["velocity_limits"] = self._active[
                    "motion_limits"
                ].as_dict()
                payload["event"] = "navigation_status"
                if payload.get("status") in _IDLE_OR_TERMINAL_STATES:
                    stop_proposal = (str(payload["nav_id"]), str(payload["status"]))
            payload.update(
                {
                    "runtime_mode": "planning",
                    "localization_backend": "fast_livo2",
                    "global_frame": self._global_frame,
                }
            )
        readiness = self._readiness()
        payload.update(readiness)
        payload["execution"] = self._execution_status(readiness)
        payload["global_costmap"] = self._global_costmap_diagnostics()
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
                "runtime_mode": "planning",
                "localization_backend": "fast_livo2",
                "global_frame": self._global_frame,
                "n5_protocol_ready": True,
                "velocity_proposal_topic": self._proposal_topic,
                "proposal_ttl_ms": self._proposal_ttl_ms,
                "proposal_frequency_hz": self._proposal_frequency_hz,
                "proposal_subscribers": self._proposal_pub.get_subscription_count(),
                "controller_speed_limit_topic": self._controller_speed_limit_topic,
                "controller_speed_limit_subscribers": (
                    self._speed_limit_pub.get_subscription_count()
                ),
                "velocity_limits": (
                    self._active["motion_limits"].as_dict()
                    if self._active
                    else None
                ),
            }
        readiness = self._readiness()
        payload.update(readiness)
        payload["execution"] = self._execution_status(readiness)
        payload["global_costmap"] = self._global_costmap_diagnostics()
        self._emit(payload)

    def _execution_status(self, readiness: dict) -> dict:
        with self._lock:
            execution = dict(self._segment_status)
        execution.update(
            {
                "odom_receive_age_sec": readiness.get("odom_status_age_sec"),
                "odom_source_age_sec": readiness.get("odom_source_age_sec"),
                "odom_receive_max_age_sec": self._control_odom_max_age_sec,
                "odom_source_max_age_sec": self._control_odom_source_max_age_sec,
            }
        )
        return execution

    def _emit(self, payload: dict) -> None:
        message = String()
        message.data = json.dumps(
            {
                **payload,
                "timestamp": time.time(),
                "shadow_only": True,
                "physical_execution": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerCommandNode()
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
