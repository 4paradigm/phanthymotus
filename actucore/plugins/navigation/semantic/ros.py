"""ROS2 bridge for synchronized camera and FAST-LIVO2 snapshots.

ROS imports are deliberately lazy. This keeps the vln logic unit-testable
on a development machine without ROS while still using explicit message types on
the robot (rather than guessing types from a briefly populated ROS graph).
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pose:
    """A validated FAST-LIVO2 pose safe to store outside ROS."""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    yaw: float
    frame_id: str
    child_frame_id: str
    source_timestamp: float | None
    received_at: float
    source_topic: str

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "orientation": {
                "x": self.qx,
                "y": self.qy,
                "z": self.qz,
                "w": self.qw,
            },
            "yaw": self.yaw,
            "frame_id": self.frame_id,
            "child_frame_id": self.child_frame_id,
            "source_timestamp": self.source_timestamp,
            "received_at": self.received_at,
            "source_topic": self.source_topic,
        }


@dataclass(frozen=True)
class Snapshot:
    image: bytes
    image_format: str
    image_mime_type: str
    image_source_timestamp: float | None
    image_received_at: float
    pose: Pose
    receive_skew_sec: float
    source_skew_sec: float | None = None
    synchronization_basis: str = "receive_time"
    map_session_id: str | None = None
    map_session_token: str = ""


@dataclass(frozen=True)
class _ImageSample:
    data: bytes
    image_format: str
    mime_type: str
    source_timestamp: float | None
    received_at: float
    received_monotonic: float


@dataclass(frozen=True)
class _PoseSample:
    pose: Pose
    received_monotonic: float


class MapSessionChangedError(RuntimeError):
    """Raised when a map-local goal would cross a FAST-LIVO2 session boundary."""


def _stamp_to_seconds(stamp) -> float | None:
    """Convert a builtin_interfaces/Time-like value to seconds."""

    if stamp is None:
        return None
    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None
    if sec == 0 and nanosec == 0:
        return None
    if nanosec < 0 or nanosec >= 1_000_000_000:
        return None
    return sec + nanosec / 1_000_000_000.0


def pose_from_odometry(message, source_topic: str, received_at: float) -> Pose:
    """Validate and detach the useful fields from nav_msgs/msg/Odometry."""

    try:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        values = [
            float(position.x),
            float(position.y),
            float(position.z),
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("odometry is missing pose fields") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("odometry pose contains a non-finite value")

    x, y, z, qx, qy, qz, qw = values
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        raise ValueError("odometry quaternion has zero length")
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    header = getattr(message, "header", None)
    frame_id = str(getattr(header, "frame_id", "") or "").strip()
    child_frame_id = str(getattr(message, "child_frame_id", "") or "").strip()
    source_timestamp = _stamp_to_seconds(getattr(header, "stamp", None))
    return Pose(
        x=x,
        y=y,
        z=z,
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
        yaw=yaw,
        frame_id=frame_id,
        child_frame_id=child_frame_id,
        source_timestamp=source_timestamp,
        received_at=received_at,
        source_topic=source_topic,
    )


class RosBridge:
    """Receive camera and FAST-LIVO2 data with low-latency sensor QoS."""

    def __init__(
        self,
        *,
        camera_topic: str,
        odometry_topic: str,
        goal_topic: str,
        status_topic: str = "",
        executor,
        required_frame_id: str = "map",
        required_child_frame_id: str = "base_link",
        status_restart_gap_sec: float = 5.0,
        status_stale_after_sec: float = 3.5,
        synchronization_mode: str = "receive_time",
    ):
        from g1_fast_livo2.camera_rgb_frame import decode as decode_camera_rgb_frame
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import String, UInt8MultiArray

        suffix = re.sub(
            r"[^a-zA-Z0-9_]",
            "_",
            f"{camera_topic}_{odometry_topic}",
        ).strip("_")
        self.camera_topic = camera_topic
        self.odometry_topic = odometry_topic
        self.goal_topic = goal_topic
        self.status_topic = status_topic
        self.required_frame_id = required_frame_id.strip()
        self.required_child_frame_id = required_child_frame_id.strip()
        self._status_restart_gap_sec = max(1.0, float(status_restart_gap_sec))
        self._status_stale_after_sec = max(1.0, float(status_stale_after_sec))
        self.synchronization_mode = str(synchronization_mode).strip()
        if self.synchronization_mode not in {"receive_time", "source_timestamp"}:
            raise ValueError(
                "synchronization_mode must be receive_time or source_timestamp"
            )
        self._executor = executor
        self._node = Node(f"vln_snapshot_{suffix[-40:] or 'bridge'}")
        self._condition = threading.Condition()
        self._latest_image: _ImageSample | None = None
        self._latest_pose: _PoseSample | None = None
        self._closed = False
        self._String = String
        self._decode_camera_rgb_frame = decode_camera_rgb_frame
        self._status_state = ""
        self._status_map_name = ""
        self._status_generation = 0
        self._status_received_monotonic: float | None = None
        self._status_companion_ready = False
        self._status_algorithm_running = False
        self._status_session_ready = False
        # Recorded points can outlive a bridge node, so every bridge needs its
        # own epoch even before status arrives.
        self._bridge_instance_id = uuid.uuid4().hex

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        odometry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        goal_qos = QoSProfile(
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

        try:
            self._image_subscription = self._node.create_subscription(
                UInt8MultiArray,
                camera_topic,
                self._on_image,
                image_qos,
            )
            self._odometry_subscription = self._node.create_subscription(
                Odometry,
                odometry_topic,
                self._on_odometry,
                odometry_qos,
            )
            self._goal_publisher = self._node.create_publisher(
                String,
                goal_topic,
                goal_qos,
            )
            self._status_subscription = None
            if status_topic:
                self._status_subscription = self._node.create_subscription(
                    String,
                    status_topic,
                    self._on_status,
                    status_qos,
                )
            self._executor.add_node(self._node)
        except Exception:
            self._node.destroy_node()
            raise

    def _on_image(self, message) -> None:
        try:
            metadata, jpeg = self._decode_camera_rgb_frame(bytes(message.data))
        except (AttributeError, TypeError, ValueError) as exc:
            log.warning("[vln] ignored invalid camera RGB frame frame: %s", exc)
            return
        received_at = time.time()
        received_monotonic = time.monotonic()
        sample = _ImageSample(
            data=jpeg,
            image_format="jpeg",
            mime_type="image/jpeg",
            source_timestamp=int(metadata["source_stamp_ns"]) / 1_000_000_000.0,
            received_at=received_at,
            received_monotonic=received_monotonic,
        )
        with self._condition:
            if self._closed:
                return
            self._latest_image = sample
            self._condition.notify_all()

    def _on_odometry(self, message) -> None:
        received_at = time.time()
        try:
            pose = pose_from_odometry(message, self.odometry_topic, received_at)
        except ValueError as exc:
            log.warning("[vln] ignored invalid FAST-LIVO2 odometry: %s", exc)
            return
        if self.required_frame_id and pose.frame_id != self.required_frame_id:
            log.warning(
                "[vln] ignored odometry in frame %r; expected %r",
                pose.frame_id,
                self.required_frame_id,
            )
            return
        if (
            self.required_child_frame_id
            and pose.child_frame_id != self.required_child_frame_id
        ):
            log.warning(
                "[vln] ignored odometry child frame %r; expected %r",
                pose.child_frame_id,
                self.required_child_frame_id,
            )
            return
        sample = _PoseSample(pose=pose, received_monotonic=time.monotonic())
        with self._condition:
            if self._closed:
                return
            self._latest_pose = sample
            self._condition.notify_all()

    def _on_status(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        schema = str(payload.get("schema") or "").strip()
        if schema and schema != "phanthy.navigation.fast_livo2_status.v1":
            return
        state = str(payload.get("state") or payload.get("status") or "").strip()
        companion_ready = payload.get("companion_ready") is True
        algorithm_running = payload.get("algorithm_running") is True
        diagnostics = payload.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        loaded_map = str(payload.get("loaded_map") or "").strip()
        map_name = str(
            payload.get("active_map")
            or loaded_map
            or diagnostics.get("session_name")
            or "unnamed"
        ).strip()
        session_ready = state == "mapping" or (
            state == "relocalized"
            and bool(loaded_map)
            and diagnostics.get("map_alignment_confirmed") is True
        )
        now = time.monotonic()
        with self._condition:
            if self._closed:
                return
            previous_health = self._map_session_health_locked(now=now)
            gap = (
                now - self._status_received_monotonic
                if self._status_received_monotonic is not None
                else None
            )
            began_session = session_ready and not self._status_session_ready
            changed_map = session_ready and map_name != self._status_map_name
            restarted_after_gap = (
                session_ready
                and gap is not None
                and gap > self._status_restart_gap_sec
            )
            incoming_ready = (
                session_ready and companion_ready and algorithm_running
            )
            recovered_to_ready = incoming_ready and previous_health != "ready"
            if incoming_ready and (
                self._status_generation == 0
                or began_session
                or changed_map
                or restarted_after_gap
                or recovered_to_ready
            ):
                self._status_generation += 1
            self._status_state = state
            self._status_map_name = map_name
            self._status_companion_ready = companion_ready
            self._status_algorithm_running = algorithm_running
            self._status_session_ready = session_ready
            self._status_received_monotonic = now
            self._condition.notify_all()

    def wait_for_snapshot(
        self,
        *,
        timeout: float,
        max_age: float,
        max_skew: float,
        after_monotonic: float | None = None,
    ) -> Snapshot | None:
        """Wait for a fresh, approximately synchronized image/pose pair."""

        deadline = time.monotonic() + timeout
        after = after_monotonic if after_monotonic is not None else float("-inf")
        with self._condition:
            while not self._closed:
                now = time.monotonic()
                image = self._latest_image
                pose_sample = self._latest_pose
                if image is not None and pose_sample is not None:
                    receive_skew = abs(
                        image.received_monotonic - pose_sample.received_monotonic
                    )
                    source_skew = None
                    if (
                        image.source_timestamp is not None
                        and pose_sample.pose.source_timestamp is not None
                    ):
                        source_skew = abs(
                            image.source_timestamp
                            - pose_sample.pose.source_timestamp
                        )
                    synchronization_basis = self.synchronization_mode
                    if self.synchronization_mode == "source_timestamp":
                        # Source-time mode is opt-in because the two upstream
                        # cards do not currently guarantee the same clock domain.
                        pair_skew = (
                            source_skew if source_skew is not None else float("inf")
                        )
                    else:
                        pair_skew = receive_skew
                    fresh = (
                        now - image.received_monotonic <= max_age
                        and now - pose_sample.received_monotonic <= max_age
                    )
                    new_enough = (
                        image.received_monotonic > after
                        and pose_sample.received_monotonic > after
                    )
                    if fresh and new_enough and pair_skew <= max_skew:
                        return Snapshot(
                            image=image.data,
                            image_format=image.image_format,
                            image_mime_type=image.mime_type,
                            image_source_timestamp=image.source_timestamp,
                            image_received_at=image.received_at,
                            pose=pose_sample.pose,
                            receive_skew_sec=receive_skew,
                            source_skew_sec=source_skew,
                            synchronization_basis=synchronization_basis,
                            map_session_id=self._map_session_id_locked(),
                            map_session_token=self._map_session_token_locked(),
                        )
                remaining = deadline - now
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=min(0.1, remaining))
        return None

    @property
    def current_map_session_id(self) -> str | None:
        """Best-effort process-local identity for a session-local FAST-LIVO2 map.

        The current FAST-LIVO2 status contract does not expose a true session UUID.
        This detects observed map changes, mapping restarts, and heartbeat gaps. A
        future upstream session_id should replace this heuristic.
        """

        with self._condition:
            return self._map_session_id_locked()

    @property
    def current_map_session_token(self) -> str:
        """A process-local fail-closed token, available even without status."""

        with self._condition:
            return self._map_session_token_locked()

    @property
    def map_session_ready(self) -> bool:
        with self._condition:
            return self._map_session_health_locked() == "ready"

    @property
    def map_session_issue(self) -> str:
        with self._condition:
            return self._map_session_health_locked()

    def wait_for_map_session(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._closed:
                if self._map_session_health_locked() == "ready":
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=min(0.1, remaining))
        return False

    def _map_session_id_locked(self) -> str | None:
        if self._map_session_health_locked() != "ready":
            return None
        if self._status_generation <= 0:
            return None
        return f"{self._status_map_name}#local-{self._status_generation}"

    def _map_session_token_locked(self) -> str:
        # This is deliberately a process-local identity. Its purpose is to
        # prevent points from crossing a bridge restart or an observed map
        # state transition while recorded points remain available.
        return "|".join(
            (
                self._bridge_instance_id,
                str(self._status_generation),
                self._status_state or "unknown",
                self._status_map_name or "unknown",
                self._map_session_health_locked(),
            )
        )

    def _map_session_health_locked(self, now: float | None = None) -> str:
        if not self.status_topic:
            return "status_unconfigured"
        if self._status_received_monotonic is None:
            return "status_missing"
        checked_at = time.monotonic() if now is None else now
        if checked_at - self._status_received_monotonic > self._status_stale_after_sec:
            return "status_stale"
        if not self._status_companion_ready:
            return "companion_not_ready"
        if not self._status_algorithm_running:
            return "algorithm_not_running"
        if not self._status_session_ready or self._status_generation <= 0:
            return "not_mapping"
        return "ready"

    @property
    def goal_subscribers(self) -> int:
        return self._goal_publisher.get_subscription_count()

    def wait_for_goal_subscriber(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self._closed:
            if self.goal_subscribers > 0:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
        return False

    def publish_goal(
        self,
        payload: dict,
        *,
        expected_map_session_token: str | None = None,
    ) -> None:
        message = self._String()
        message.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        # Status callbacks use the same condition lock. Checking the token and
        # publishing inside one critical section closes the last A->B race.
        with self._condition:
            if self._closed:
                raise RuntimeError("ROS bridge is closed")
            if (
                expected_map_session_token is not None
                and self._map_session_token_locked() != expected_map_session_token
            ):
                raise MapSessionChangedError(
                    "FAST-LIVO2 map session changed before goal publication"
                )
            self._goal_publisher.publish(message)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        try:
            self._executor.remove_node(self._node)
        finally:
            self._node.destroy_node()
