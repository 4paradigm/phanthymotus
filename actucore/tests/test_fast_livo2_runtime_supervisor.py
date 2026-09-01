from __future__ import annotations

import json
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "navigation"
    / "runtime"
    / "g1_fast_livo2"
)
sys.path.insert(0, str(PACKAGE_ROOT))


def _import_ros_runtime_modules():
    inserted: list[str] = []

    def module(name: str):
        value = types.ModuleType(name)
        sys.modules[name] = value
        inserted.append(name)
        return value

    class Message:
        def __init__(self) -> None:
            self.data = ""

    rclpy = module("rclpy")
    callbacks = module("rclpy.callback_groups")
    callbacks.MutuallyExclusiveCallbackGroup = type(
        "MutuallyExclusiveCallbackGroup", (), {}
    )
    callbacks.ReentrantCallbackGroup = type("ReentrantCallbackGroup", (), {})
    executors = module("rclpy.executors")
    executors.MultiThreadedExecutor = type("MultiThreadedExecutor", (), {})
    node = module("rclpy.node")
    node.Node = type("Node", (), {})
    rclpy_time = module("rclpy.time")
    rclpy_time.Time = type("Time", (), {})
    qos = module("rclpy.qos")
    policy = type(
        "Policy",
        (),
        {
            "KEEP_LAST": 1,
            "RELIABLE": 1,
            "TRANSIENT_LOCAL": 1,
            "VOLATILE": 1,
            "BEST_EFFORT": 1,
        },
    )
    qos.DurabilityPolicy = policy
    qos.HistoryPolicy = policy
    qos.ReliabilityPolicy = policy
    qos.QoSProfile = lambda **kwargs: kwargs
    qos.qos_profile_sensor_data = {}

    for parent in ("geometry_msgs", "nav_msgs", "sensor_msgs", "std_msgs"):
        module(parent)
    message_modules = {
        "geometry_msgs.msg": ("TransformStamped",),
        "nav_msgs.msg": ("OccupancyGrid", "Odometry"),
        "sensor_msgs.msg": (
            "CameraInfo",
            "CompressedImage",
            "Image",
            "Imu",
            "PointCloud2",
            "PointField",
        ),
        "std_msgs.msg": ("String", "UInt8MultiArray"),
        "tf2_ros": (
            "Buffer",
            "TransformBroadcaster",
            "TransformException",
            "TransformListener",
        ),
    }
    for module_name, names in message_modules.items():
        target = module(module_name)
        for name in names:
            if name == "TransformException":
                value = type(name, (Exception,), {})
            else:
                value = Message if name == "String" else type(name, (), {})
            setattr(target, name, value)

    try:
        from g1_fast_livo2 import adapter_node, runtime_supervisor
    finally:
        for name in reversed(inserted):
            sys.modules.pop(name, None)
    return adapter_node, runtime_supervisor


ADAPTER_MODULE, SUPERVISOR_MODULE = _import_ros_runtime_modules()
FastLivo2Adapter = ADAPTER_MODULE.FastLivo2Adapter
FastLivo2Supervisor = SUPERVISOR_MODULE.FastLivo2Supervisor
TemporalOccupancyMap = ADAPTER_MODULE.TemporalOccupancyMap
VoxelMap = ADAPTER_MODULE.VoxelMap
write_pcd_xyz_atomic = ADAPTER_MODULE.write_pcd_xyz_atomic
Pose3 = ADAPTER_MODULE.Pose3
Quaternion = ADAPTER_MODULE.Quaternion


class _CapturePublisher:
    def __init__(self, events=None, label="response") -> None:
        self.messages = []
        self.events = events
        self.label = label

    def publish(self, message) -> None:
        self.messages.append(message)
        if self.events is not None:
            self.events.append(self.label)


def _prepare_adapter_concurrency(adapter) -> None:
    adapter._static_lock = threading.Lock()
    adapter._mapping_work_condition = threading.Condition()
    adapter._mapping_work = None
    adapter._mapping_generation = 0
    adapter._mapping_work_generation = 0
    adapter._mapping_work_latest_monotonic = None
    adapter._mapping_work_dropped = 0


class FastLivo2RuntimeSupervisorTest(unittest.TestCase):
    def test_adapter_rejection_logs_are_sampled_and_recovery_is_reported(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        adapter._rejection_counts = {}
        adapter._last_lidar_source_stamp_ns = 1_000_000_000
        adapter._last_imu_source_stamp_ns = 1_350_000_000
        adapter._last_sensor_pair_monotonic = time.monotonic() - 0.6
        adapter._sensor_pair_max_age = 3.0
        adapter._sensor_rejection_pub = _CapturePublisher()
        warnings = []
        infos = []
        adapter.get_logger = lambda: SimpleNamespace(
            warning=warnings.append,
            info=infos.append,
        )

        rejected = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="camera_init",
                stamp=SimpleNamespace(sec=2, nanosec=50_000_000),
            )
        )
        for _ in range(101):
            adapter._warn_rejected(
                "FAST-LIVO2 cloud",
                "bad\nframe",
                rejected,
            )

        self.assertEqual(len(warnings), 2)
        self.assertIn("count=1", warnings[0])
        self.assertIn("count=100", warnings[1])
        self.assertNotIn("\n", warnings[0])
        self.assertEqual(len(adapter._sensor_rejection_pub.messages), 101)
        evidence = json.loads(adapter._sensor_rejection_pub.messages[0].data)
        self.assertEqual(evidence["event"], "rejected")
        self.assertEqual(evidence["source_stamp_ns"], 2_050_000_000)
        self.assertEqual(evidence["lidar_imu_skew_ms"], 350.0)
        self.assertEqual(evidence["lidar_imu_pair_result"], "skew_exceeded")
        self.assertGreaterEqual(evidence["last_valid_pair_age_sec"], 0.5)

        adapter._mark_valid("FAST-LIVO2 cloud")
        self.assertEqual(
            infos,
            ["FAST-LIVO2 cloud recovered after 101 rejected samples"],
        )
        self.assertEqual(adapter._rejection_counts, {})

    def test_collection_subscriptions_drop_backlog_instead_of_replaying_it(self) -> None:
        source = (
            PACKAGE_ROOT
            / "g1_fast_livo2"
            / "runtime_supervisor.py"
        ).read_text(encoding="utf-8")
        method = source.split(
            "    def _create_collection_subscriptions", 1
        )[1].split("    def _on_collection_sample", 1)[0]

        self.assertIn("depth=4", method)
        self.assertIn("ReliabilityPolicy.BEST_EFFORT", method)
        self.assertNotIn("ReliabilityPolicy.RELIABLE", method)
        self.assertNotIn("depth=200", method)

    def test_heartbeat_keeps_runtime_session_id_across_readiness_changes(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        supervisor._lock = threading.RLock()
        supervisor._process = mock.Mock()
        supervisor._process.poll.return_value = None
        supervisor._active_map = None
        supervisor._loaded_map = "office"
        supervisor._runtime_mode = "localization"
        supervisor._runtime_session_id = "runtime-a"
        supervisor._diagnostics = {"ready": True}
        supervisor._collection_snapshot = lambda: {}
        supervisor._collection_status_pub = _CapturePublisher()
        payloads = []
        supervisor._publish = payloads.append

        supervisor._publish_heartbeat()
        supervisor._diagnostics = {"ready": False}
        supervisor._publish_heartbeat()

        self.assertEqual(
            [payload["map_session_id"] for payload in payloads],
            ["runtime-a", "runtime-a"],
        )
        self.assertEqual(
            [payload["companion_ready"] for payload in payloads],
            [True, False],
        )

    def test_cached_map_view_refreshes_pose_without_reencoding_points(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        adapter._latency_ms = {}
        adapter._latency_max_ms = {}
        identity = Pose3(
            0.0,
            0.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        adapter._map_view_cache = ADAPTER_MODULE.encode_map_view_points(
            ((1.0, 2.0, 0.1),),
            identity,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
            max_points=80_000,
        )
        adapter._latest_pose = Pose3(
            3.0,
            -4.0,
            0.0,
            ADAPTER_MODULE.quaternion_from_rpy(0.0, 0.0, 1.0),
        )
        adapter._map_view_pub = _CapturePublisher()
        adapter.get_logger = lambda: SimpleNamespace(warning=lambda _msg: None)

        adapter._publish_map_view()

        self.assertEqual(len(adapter._map_view_pub.messages), 1)
        published = bytes(adapter._map_view_pub.messages[0].data)
        self.assertEqual(published[12:], adapter._map_view_cache[12:])
        x, y, yaw = struct.unpack_from("<fff", published, 0)
        self.assertAlmostEqual(x, 3.0)
        self.assertAlmostEqual(y, -4.0)
        self.assertAlmostEqual(yaw, 1.0)
        self.assertIn("map_view_pose_publish", adapter._latency_ms)

    def test_segment_latency_diagnostics_keep_last_and_maximum(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)

        adapter._record_latency_locked("cloud_decode", 0.012)
        adapter._record_latency_locked("cloud_decode", 0.004)

        self.assertAlmostEqual(adapter._latency_ms["cloud_decode"], 4.0)
        self.assertAlmostEqual(adapter._latency_max_ms["cloud_decode"], 12.0)

    def test_sensor_contract_accepts_arbitrary_shared_frame_and_static_tf(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        adapter._lidar_frame = "front_navigation_sensor"
        adapter._imu_frame = "front_navigation_sensor"
        adapter._last_lidar_source_stamp_ns = 1_000_000_000
        adapter._last_imu_source_stamp_ns = 1_050_000_000
        adapter._last_sensor_pair_monotonic = None
        adapter._source_max_age = 0.5
        adapter._sensor_pair_max_age = 3.0
        adapter._point_time_ready = True
        adapter._imu_time_ready = True
        adapter._point_time_span_ms = 80.0
        adapter._base_to_sensor = None
        adapter._base_to_sensor_tf_ready = False
        adapter._sensor_frame = None
        adapter._sensor_tf_error = None
        adapter._odom_health = ADAPTER_MODULE.OdomHealthMonitor()
        adapter._tf_buffer = mock.Mock()
        adapter._tf_buffer.lookup_transform.return_value = SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.2, y=-0.1, z=0.8),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )

        adapter._refresh_sensor_contract()
        adapter._refresh_sensor_contract()

        self.assertTrue(adapter._sensor_contract_ready_locked())
        self.assertEqual(adapter._sensor_frame, "front_navigation_sensor")
        self.assertAlmostEqual(adapter._base_to_sensor.z, 0.8)
        self.assertEqual(adapter._readiness_blockers_locked(), [])
        adapter._tf_buffer.lookup_transform.assert_called_once_with(
            "base_link",
            "front_navigation_sensor",
            mock.ANY,
        )

    def test_sensor_contract_tolerates_brief_pair_gap_but_rejects_three_seconds(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        adapter._rejection_counts = {}
        adapter._lidar_frame = "livox_frame"
        adapter._imu_frame = "livox_frame"
        adapter._last_lidar_source_stamp_ns = 1_000_000_000
        adapter._last_imu_source_stamp_ns = 1_050_000_000
        adapter._last_sensor_pair_monotonic = None
        adapter._source_max_age = 0.5
        adapter._sensor_pair_max_age = 3.0
        adapter._point_time_ready = True
        adapter._imu_time_ready = True
        adapter._point_time_span_ms = 100.0
        adapter._base_to_sensor = None
        adapter._base_to_sensor_tf_ready = False
        adapter._sensor_frame = None
        adapter._sensor_tf_error = None
        adapter._odom_health = ADAPTER_MODULE.OdomHealthMonitor()
        adapter._tf_buffer = mock.Mock()
        adapter._tf_buffer.lookup_transform.return_value = SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.0, y=0.0, z=0.46),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        adapter.get_logger = lambda: SimpleNamespace(
            warning=lambda _message: None,
            info=lambda _message: None,
        )

        adapter._refresh_sensor_contract()
        duplicate = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="livox_frame",
                stamp=SimpleNamespace(sec=1, nanosec=50_000_000),
            )
        )
        adapter._on_imu_contract(duplicate)

        self.assertTrue(adapter._sensor_contract_ready_locked())
        self.assertEqual(adapter._readiness_blockers_locked(), [])
        self.assertIsNotNone(adapter._base_to_sensor)

        adapter._last_sensor_pair_monotonic = time.monotonic() - 0.6
        self.assertTrue(adapter._sensor_contract_ready_locked())
        self.assertEqual(adapter._readiness_blockers_locked(), [])

        adapter._last_sensor_pair_monotonic = time.monotonic() - 3.1
        self.assertFalse(adapter._sensor_contract_ready_locked())
        self.assertTrue(adapter._sensor_geometry_ready_locked())
        self.assertEqual(adapter._readiness_blockers_locked(), [])
        self.assertEqual(
            adapter._sensor_contract_issues_locked(),
            ["point_time_invalid"],
        )
        self.assertIsNotNone(adapter._base_to_sensor)

    def test_stale_raw_sensor_pair_does_not_reject_fresh_fast_livo2_outputs(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        adapter._source_age = lambda _stamp: 0.0
        adapter._source_max_age = 0.5
        adapter._source_age_tolerance = 0.05
        adapter._sensor_pair_max_age = 3.0
        adapter._map_load_max_points = 200_000
        adapter._live_cloud_max_bytes = 64 * 1024 * 1024
        adapter._invalid_odom = 0
        adapter._invalid_cloud = 0
        adapter._rejection_counts = {}
        adapter._latency_ms = {}
        adapter._latency_max_ms = {}
        adapter._sensor_frame = "sensor_frame"
        adapter._lidar_frame = "sensor_frame"
        adapter._imu_frame = "sensor_frame"
        adapter._point_time_ready = True
        adapter._imu_time_ready = True
        adapter._last_sensor_pair_monotonic = time.monotonic() - 3.1
        adapter._base_to_sensor_tf_ready = True
        adapter._base_to_sensor = Pose3(
            0.0,
            0.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        adapter._odom_health = ADAPTER_MODULE.OdomHealthMonitor()
        adapter._latest_session_pose = None
        adapter._last_odom_monotonic = None
        adapter._last_odom_source_age = None
        adapter._latest_session_points = ()
        adapter._last_cloud_monotonic = None
        adapter._last_cloud_source_age = None
        adapter._mode = "localization"
        adapter._map_from_session = None
        adapter.get_logger = lambda: SimpleNamespace(
            warning=lambda _message: None,
            info=lambda _message: None,
        )
        odom = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="camera_init",
                stamp=SimpleNamespace(sec=1, nanosec=0),
            ),
            child_frame_id="aft_mapped",
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            ),
        )
        message = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="camera_init",
                stamp=SimpleNamespace(sec=1, nanosec=0),
            ),
            width=1,
            height=1,
            point_step=24,
            row_step=24,
            is_bigendian=False,
            fields=[
                SimpleNamespace(name="x", offset=0, datatype=8, count=1),
                SimpleNamespace(name="y", offset=8, datatype=8, count=1),
                SimpleNamespace(name="z", offset=16, datatype=8, count=1),
            ],
            data=struct.pack("<ddd", 1.0, 2.0, 0.0),
        )

        adapter._on_odom(odom)
        adapter._on_cloud(message)

        self.assertEqual(adapter._invalid_odom, 0)
        self.assertIsNotNone(adapter._latest_session_pose)
        self.assertIsNotNone(adapter._last_odom_monotonic)
        self.assertEqual(adapter._invalid_cloud, 0)
        self.assertEqual(len(adapter._latest_session_points), 1)
        self.assertIsNotNone(adapter._last_cloud_monotonic)

    def test_sensor_contract_reports_frame_mismatch_and_missing_tf(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        adapter._lidar_frame = "lidar_frame"
        adapter._imu_frame = "imu_frame"
        adapter._last_lidar_source_stamp_ns = 1_000_000_000
        adapter._last_imu_source_stamp_ns = 1_050_000_000
        adapter._last_sensor_pair_monotonic = None
        adapter._source_max_age = 0.5
        adapter._sensor_pair_max_age = 3.0
        adapter._point_time_ready = True
        adapter._imu_time_ready = True
        adapter._point_time_span_ms = 50.0
        adapter._base_to_sensor = None
        adapter._base_to_sensor_tf_ready = False
        adapter._sensor_frame = None
        adapter._sensor_tf_error = None
        adapter._odom_health = ADAPTER_MODULE.OdomHealthMonitor()
        adapter._tf_buffer = mock.Mock()

        adapter._refresh_sensor_contract()

        self.assertIn("sensor_frame_mismatch", adapter._readiness_blockers_locked())
        adapter._tf_buffer.lookup_transform.assert_not_called()

        adapter._imu_frame = "lidar_frame"
        adapter._tf_buffer.lookup_transform.side_effect = (
            ADAPTER_MODULE.TransformException("no static transform")
        )
        adapter._refresh_sensor_contract()
        self.assertIn("sensor_tf_unavailable", adapter._readiness_blockers_locked())
        self.assertFalse(adapter._sensor_contract_ready_locked())

    def test_relocalize_discards_live_points_encoded_under_old_alignment(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        adapter._mode = "relocalized"
        adapter._session_name = "office"
        adapter._latest_session_pose = Pose3(
            0.0,
            0.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        adapter._latest_session_points = ((0.0, 0.0, 0.0),) * 40
        adapter._relocalization_cloud_history = [
            (time.monotonic(), adapter._latest_session_points),
            (time.monotonic(), ((1.0, 0.0, 0.0),) * 40),
        ]
        adapter._relocalization_preview_pose = object()
        adapter._relocalization_preview_points = ((8.0, 8.0, 0.0),)
        adapter._last_odom_monotonic = time.monotonic()
        adapter._last_cloud_monotonic = time.monotonic()
        adapter._source_max_age = 0.5
        adapter._reference_points = ((0.0, 0.0, 0.0),) * 40
        adapter._obstacle_min_height = -0.3
        adapter._obstacle_max_height = 0.3
        adapter._latest_mapped_points = ((9.0, 9.0, 0.0),)
        adapter._pending_cloud = object()
        adapter._last_cloud_pose_skew_sec = 0.01
        adapter._last_navigation_cloud_monotonic = time.monotonic()
        adapter._map_view_cache = b"old-cache"
        adapter._map_view_cache_monotonic = time.monotonic()
        adapter._odom_health = ADAPTER_MODULE.OdomHealthMonitor()
        map_from_session = Pose3(
            1.0,
            2.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        map_base_pose = Pose3(
            3.0,
            4.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )

        with mock.patch.object(
            ADAPTER_MODULE,
            "estimate_planar_relocalization",
            return_value=SimpleNamespace(
                map_from_session=map_from_session,
                map_base_pose=map_base_pose,
                match_ratio=0.8,
                matched_points=40,
                evaluated_points=40,
            ),
        ) as estimate:
            result = adapter._relocalize(
                {
                    "initial_x": 3.0,
                    "initial_y": 4.0,
                    "initial_yaw": 0.0,
                    "search_xy_m": 1.0,
                    "search_yaw_rad": 0.35,
                    "_operation_deadline_monotonic": time.monotonic() + 1.0,
                }
            )

        self.assertEqual(result["status"], "relocalized")
        self.assertEqual(adapter._latest_mapped_points, ())
        self.assertIsNone(adapter._pending_cloud)
        self.assertIsNone(adapter._last_navigation_cloud_monotonic)
        self.assertIsNone(adapter._map_view_cache)
        self.assertIsNone(adapter._map_view_cache_monotonic)
        self.assertEqual(adapter._map_from_session, map_from_session)
        self.assertIsNone(adapter._relocalization_preview_pose)
        self.assertEqual(adapter._relocalization_preview_points, ())
        estimate.assert_called_once()
        self.assertEqual(
            estimate.call_args.kwargs["min_match_ratio"],
            ADAPTER_MODULE._RELOCALIZATION_MIN_MATCH_RATIO,
        )
        self.assertEqual(
            len(estimate.call_args.kwargs["session_points"]),
            80,
        )
        self.assertEqual(result["input_frame_count"], 2)

    def test_relocalization_rejection_returns_and_previews_best_candidate(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        adapter._mode = "awaiting_relocalization"
        adapter._session_name = "office"
        adapter._latest_session_pose = Pose3(
            0.0,
            0.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        points = tuple((index * 0.1, 0.0, 0.0) for index in range(40))
        adapter._latest_session_points = points
        adapter._relocalization_cloud_history = [(time.monotonic(), points)]
        adapter._relocalization_preview_pose = None
        adapter._relocalization_preview_points = ()
        adapter._last_odom_monotonic = time.monotonic()
        adapter._last_cloud_monotonic = time.monotonic()
        adapter._source_max_age = 0.5
        adapter._reference_points = points
        adapter._obstacle_min_height = -0.3
        adapter._obstacle_max_height = 0.3
        adapter._map_view_cache = b"old-cache"
        adapter._map_view_cache_monotonic = time.monotonic()
        adapter._last_match = None
        adapter._map_control_status_pub = _CapturePublisher()
        candidate = Pose3(
            1.0,
            2.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        rejected = ADAPTER_MODULE.RelocalizationRejected(
            "match too low",
            result=SimpleNamespace(
                map_from_session=candidate,
                map_base_pose=candidate,
                match_ratio=0.31,
                matched_points=35,
                evaluated_points=40,
            ),
            reason="match_ratio_below_threshold",
        )
        request = SimpleNamespace(
            data=json.dumps(
                {
                    "request_id": "relocalize-1",
                    "action": "relocalize",
                    "args": {
                        "initial_x": 1.0,
                        "initial_y": 2.0,
                        "initial_yaw": 0.0,
                    },
                    "operation_deadline_monotonic": time.monotonic() + 1.0,
                }
            )
        )

        with mock.patch.object(
            ADAPTER_MODULE,
            "estimate_planar_relocalization",
            side_effect=rejected,
        ):
            adapter._on_map_control(request)

        response = json.loads(adapter._map_control_status_pub.messages[-1].data)
        self.assertTrue(response["retryable"])
        self.assertEqual(response["candidate_pose"]["x"], 1.0)
        self.assertTrue(response["preview_available"])
        self.assertEqual(adapter._relocalization_preview_pose, candidate)
        self.assertEqual(adapter._last_match["input_frame_count"], 1)
        self.assertIsNone(adapter._map_view_cache)

    def test_adapter_execute_waits_for_bidirectional_discovery(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        supervisor._lock = threading.RLock()
        supervisor._condition = threading.Condition(supervisor._lock)
        supervisor._pending_map_control_requests = set()
        supervisor._map_control_responses = {}
        supervisor._map_control_status_topic = (
            "/ubuntu/navigation/fast_livo2/map_control_status"
        )
        publisher = mock.Mock()
        publisher.get_subscription_count.side_effect = [0, 1, 1]
        supervisor._map_control_pub = publisher
        supervisor.count_publishers = mock.Mock(side_effect=[0, 0, 1])
        supervisor.get_parameter = lambda _name: SimpleNamespace(value=1.0)

        def answer(message) -> None:
            request = json.loads(message.data)
            supervisor._on_map_control_status(
                SimpleNamespace(
                    data=json.dumps(
                        {
                            "event": "response",
                            "request_id": request["request_id"],
                            "action": request["action"],
                            "status": "configured",
                        }
                    )
                )
            )

        publisher.publish.side_effect = answer
        with mock.patch.object(SUPERVISOR_MODULE.time, "sleep") as sleep:
            result = supervisor._adapter_execute(
                "configure_obstacle_filter",
                {"min_height_m": -0.3, "max_height_m": 0.3},
            )

        self.assertEqual(result["status"], "configured")
        self.assertEqual(sleep.call_count, 2)
        publisher.publish.assert_called_once()

    def test_adapter_execute_fails_fast_when_adapter_is_undiscovered(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        supervisor._map_control_status_topic = (
            "/ubuntu/navigation/fast_livo2/map_control_status"
        )
        supervisor._map_control_pub = mock.Mock()
        supervisor._map_control_pub.get_subscription_count.return_value = 0
        supervisor.count_publishers = mock.Mock(return_value=0)

        with mock.patch.object(
            SUPERVISOR_MODULE,
            "_MAP_CONTROL_DISCOVERY_TIMEOUT_SEC",
            0.001,
        ):
            result = supervisor._adapter_execute(
                "configure_obstacle_filter",
                {"min_height_m": -0.3, "max_height_m": 0.3},
            )

        self.assertEqual(
            result["error_code"],
            "fast_livo2_adapter_unavailable",
        )
        self.assertTrue(result["retryable"])
        supervisor._map_control_pub.publish.assert_not_called()

    def test_extreme_float64_live_cloud_is_rejected_without_callback_escape(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._source_age = lambda _stamp: 0.0
        adapter._source_max_age = 0.5
        adapter._sensor_pair_max_age = 3.0
        adapter._source_age_tolerance = 0.05
        adapter._map_load_max_points = 200_000
        adapter._live_cloud_max_bytes = 64 * 1024 * 1024
        adapter._invalid_cloud = 0
        adapter._lock = threading.RLock()
        adapter._latest_session_points = ()
        adapter._last_cloud_monotonic = None
        adapter._last_cloud_source_age = None
        adapter._map_from_session = Pose3(
            0.0,
            0.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        adapter._mode = "localization"
        adapter._pose_history = []
        adapter._obstacle_min_height = -0.30
        adapter._obstacle_max_height = 0.30
        adapter._sensor_frame = "sensor_frame"
        adapter._lidar_frame = "sensor_frame"
        adapter._imu_frame = "sensor_frame"
        adapter._point_time_ready = True
        adapter._imu_time_ready = True
        adapter._last_sensor_pair_monotonic = time.monotonic()
        adapter._base_to_sensor_tf_ready = True
        adapter._base_to_sensor = Pose3(
            0.0,
            0.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        adapter._odom_health = ADAPTER_MODULE.OdomHealthMonitor()
        adapter._odom_health.observe(
            1_000_000_000,
            Pose3(0.0, 0.0, 0.0, Quaternion(0.0, 0.0, 0.0, 1.0)),
        )
        adapter.get_logger = lambda: SimpleNamespace(warning=lambda _msg: None)
        message = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="camera_init",
                stamp=SimpleNamespace(sec=1, nanosec=0),
            ),
            width=1,
            height=1,
            point_step=24,
            is_bigendian=False,
            fields=[
                SimpleNamespace(name="x", offset=0, datatype=8, count=1),
                SimpleNamespace(name="y", offset=8, datatype=8, count=1),
                SimpleNamespace(name="z", offset=16, datatype=8, count=1),
            ],
            data=struct.pack("<ddd", 1e308, 0.0, 0.0),
        )

        adapter._on_cloud(message)

        self.assertEqual(adapter._invalid_cloud, 1)
        self.assertEqual(adapter._latest_session_points, ())

    def test_navigation_cloud_waits_until_tf_history_brackets_its_stamp(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._lock = threading.RLock()
        _prepare_adapter_concurrency(adapter)
        identity = Pose3(
            0.0,
            0.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        stamp = SimpleNamespace(sec=1, nanosec=20_000_000)
        message = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
        adapter._pending_cloud = (
            message,
            ((1.0, 2.0, 0.0),),
            10.0,
            0.01,
        )
        adapter._map_from_session = identity
        adapter._mode = "relocalized"
        adapter._pose_history = [(1_000_000_000, identity)]
        adapter._static_pose_match_tolerance_ns = 50_000_000
        adapter._obstacle_min_height = -0.30
        adapter._obstacle_max_height = 0.30
        adapter._unmatched_navigation_cloud = 0
        adapter._unmatched_static_cloud = 0
        adapter._last_cloud_pose_skew_sec = None
        adapter._latest_mapped_points = ()
        adapter._invalid_cloud = 0
        adapter._cloud_pub = _CapturePublisher()
        adapter._xyz_cloud = lambda points, source_stamp: (points, source_stamp)
        adapter.get_logger = lambda: SimpleNamespace(warning=lambda _msg: None)

        adapter._drain_pending_cloud()

        self.assertIsNotNone(adapter._pending_cloud)
        self.assertEqual(adapter._cloud_pub.messages, [])

        adapter._pose_history.append((1_100_000_000, identity))
        adapter._drain_pending_cloud()

        self.assertIsNone(adapter._pending_cloud)
        self.assertEqual(len(adapter._cloud_pub.messages), 1)
        self.assertAlmostEqual(adapter._last_cloud_pose_skew_sec, 0.02)

    def test_static_map_disk_failure_remains_retryable_through_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = object.__new__(FastLivo2Adapter)
            adapter._lock = threading.RLock()
            _prepare_adapter_concurrency(adapter)
            adapter._session_name = "office"
            adapter._static_map_error = None
            adapter._static_save_result = None
            adapter._mode = "mapping"
            adapter._static_map = SimpleNamespace(
                confirmed_points=tuple(
                    (float(index), 0.0, 0.0) for index in range(40)
                )
            )
            adapter._static_map_load_max_points = 200_000
            adapter._obstacle_min_height = -0.30
            adapter._obstacle_max_height = 0.30
            adapter._pose_history = []
            adapter._map_root = Path(directory).resolve()
            adapter._map_control_status_pub = _CapturePublisher()
            request = SimpleNamespace(
                data=json.dumps(
                    {
                        "request_id": "save-1",
                        "action": "save_static_map",
                        "args": {"map_name": "office"},
                        "operation_deadline_monotonic": time.monotonic() + 10.0,
                    }
                )
            )

            with mock.patch.object(
                Path,
                "open",
                side_effect=OSError("disk full"),
            ):
                adapter._on_map_control(request)

            payload = json.loads(adapter._map_control_status_pub.messages[-1].data)
            self.assertEqual(payload["error_code"], "map_control_io_failed")
            self.assertTrue(payload["retryable"])
            self.assertEqual(adapter._mode, "finalizing")

    def test_expired_map_control_cannot_mutate_adapter_state(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        adapter._mode = "awaiting_relocalization"
        adapter._map_control_status_pub = _CapturePublisher()
        request = SimpleNamespace(
            data=json.dumps(
                {
                    "request_id": "late-unload",
                    "action": "unload_map",
                    "args": {},
                    "operation_deadline_monotonic": time.monotonic() - 1.0,
                }
            )
        )

        adapter._on_map_control(request)

        payload = json.loads(adapter._map_control_status_pub.messages[-1].data)
        self.assertEqual(payload["error_code"], "map_control_timeout")
        self.assertTrue(payload["retryable"])
        self.assertEqual(adapter._mode, "awaiting_relocalization")

    def test_map_control_replies_before_deferred_map_publication(self) -> None:
        events = []
        adapter = object.__new__(FastLivo2Adapter)
        adapter._map_control_status_pub = _CapturePublisher(events, "response")
        adapter._load_saved_map = lambda _args: {
            "status": "map_loaded",
            "_post_response": lambda: events.append("map"),
        }
        request = SimpleNamespace(
            data=json.dumps(
                {
                    "request_id": "load-1",
                    "action": "load_map",
                    "args": {},
                    "operation_deadline_monotonic": time.monotonic() + 10.0,
                }
            )
        )

        adapter._on_map_control(request)

        self.assertEqual(events, ["response", "map"])
        payload = json.loads(adapter._map_control_status_pub.messages[-1].data)
        self.assertNotIn("_post_response", payload)

    def test_load_timeout_before_atomic_commit_preserves_active_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            points = tuple((index * 0.2, 0.0, 0.0) for index in range(40))
            old_points = tuple((index * 0.2, 2.0, 0.0) for index in range(40))
            pcd = root / "office.pcd"
            write_pcd_xyz_atomic(pcd, points)
            adapter = object.__new__(FastLivo2Adapter)
            adapter._lock = threading.RLock()
            _prepare_adapter_concurrency(adapter)
            adapter._map_root = root
            adapter._map_load_max_points = 200_000
            adapter._static_map_load_max_points = 200_000
            adapter._obstacle_min_height = -0.30
            adapter._obstacle_max_height = 0.30
            adapter._map_view_voxel_size = 0.20
            adapter._map_view_context = VoxelMap(0.20)
            adapter._static_map = TemporalOccupancyMap(0.1)
            adapter._static_map.load_confirmed(old_points)
            adapter._mode = "idle"
            adapter.get_parameter = lambda _name: SimpleNamespace(value=0.1)
            args = {
                "map_name": "office",
                "pcd_files": [str(pcd)],
                "_operation_deadline_monotonic": time.monotonic() + 30.0,
            }
            original_require = FastLivo2Adapter._require_map_control_deadline

            def require_deadline(request_args, *, stage):
                if stage == "map activation commit":
                    raise TimeoutError("forced deadline before commit")
                return original_require(request_args, stage=stage)

            with mock.patch.object(
                FastLivo2Adapter,
                "_require_map_control_deadline",
                side_effect=require_deadline,
            ):
                with self.assertRaisesRegex(TimeoutError, "before commit"):
                    adapter._load_saved_map(args)

            self.assertEqual(adapter._mode, "idle")
            self.assertEqual(adapter._static_map.confirmed_points, old_points)

    def test_map_validation_recovers_ground_for_canvas_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            points = tuple(
                [(index * 0.25, 0.0, -1.20) for index in range(20)]
                + [(index * 0.25, 1.0, 0.0) for index in range(20)]
            )
            pcd = root / "office.pcd"
            write_pcd_xyz_atomic(pcd, points)
            adapter = object.__new__(FastLivo2Adapter)
            adapter._lock = threading.RLock()
            _prepare_adapter_concurrency(adapter)
            adapter._map_root = root
            adapter._map_load_max_points = 200_000
            adapter._static_map_load_max_points = 200_000
            adapter._obstacle_min_height = -0.30
            adapter._obstacle_max_height = 0.30
            adapter._map_view_voxel_size = 0.20
            adapter._map_view_context = VoxelMap(0.20)
            adapter._static_map = TemporalOccupancyMap(0.1)
            adapter.get_parameter = lambda _name: SimpleNamespace(value=0.1)

            result = adapter._load_saved_map(
                {
                    "map_name": "office",
                    "pcd_files": [str(pcd)],
                    "_operation_deadline_monotonic": time.monotonic() + 30.0,
                },
                validate_only=True,
            )

            self.assertEqual(result["status"], "map_validated")
            self.assertEqual(result["map_view_context_point_count"], 20)
            self.assertEqual(adapter._static_map.point_count, 0)

    def test_mapping_work_queue_keeps_only_the_latest_scan(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        _prepare_adapter_concurrency(adapter)

        adapter._queue_mapping_scan(
            {"generation": 0, "receive_monotonic": 10.0, "sample": "old"}
        )
        adapter._queue_mapping_scan(
            {"generation": 0, "receive_monotonic": 11.0, "sample": "latest"}
        )

        self.assertEqual(adapter._mapping_work["sample"], "latest")
        self.assertEqual(adapter._mapping_work_dropped, 1)

    def test_mapping_work_queue_drops_late_completion_from_older_scan(self) -> None:
        adapter = object.__new__(FastLivo2Adapter)
        _prepare_adapter_concurrency(adapter)

        adapter._queue_mapping_scan(
            {"generation": 0, "receive_monotonic": 11.0, "sample": "newer"}
        )
        adapter._mapping_work = None
        adapter._queue_mapping_scan(
            {"generation": 0, "receive_monotonic": 10.0, "sample": "late-old"}
        )

        self.assertIsNone(adapter._mapping_work)
        self.assertEqual(adapter._mapping_work_dropped, 1)

        adapter._invalidate_mapping_work_locked()
        adapter._queue_mapping_scan(
            {"generation": 0, "receive_monotonic": 12.0, "sample": "old-generation"}
        )
        self.assertIsNone(adapter._mapping_work)
        self.assertEqual(adapter._mapping_work_dropped, 2)

    def test_late_terminal_mapping_failure_is_replayed_on_retry(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        process = object()
        supervisor._lock = threading.RLock()
        supervisor._process = process
        supervisor._active_map = "office"
        supervisor._loaded_map = None
        supervisor._runtime_mode = "finalizing"
        supervisor._started_unix_ms = 123
        supervisor._pending_mapping_finalize = {"process": process}
        supervisor._last_mapping_result = None
        supervisor._diagnostics = {}
        supervisor._diagnostics_monotonic = None
        terminal = {
            "status": "error",
            "error_code": "manifest_write_failed",
            "error": "invalid manifest",
            "map_name": "office",
            "retryable": False,
        }

        # The outer backend may already have timed out when this late result is
        # produced.  The next stop request must not be turned into already_idle.
        supervisor._finish_mapping_runtime(
            process,
            terminal_result=terminal,
        )
        replay = supervisor._stop_mapping("office")

        self.assertEqual(replay["status"], "error")
        self.assertEqual(replay["error_code"], "manifest_write_failed")
        self.assertFalse(replay["retryable"])
        self.assertTrue(replay["already_finalized"])

    def test_stale_stop_request_cannot_terminate_new_mapping_session(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        process = mock.Mock()
        process.poll.return_value = None
        supervisor._lock = threading.RLock()
        supervisor._process = process
        supervisor._active_map = "map-b"
        supervisor._loaded_map = None
        supervisor._runtime_mode = "mapping"
        supervisor._started_unix_ms = 123
        supervisor._pending_mapping_finalize = None
        supervisor._last_mapping_result = None
        supervisor._diagnostics = {}
        supervisor._diagnostics_monotonic = None

        with mock.patch.object(SUPERVISOR_MODULE.os, "killpg") as killpg:
            result = supervisor._stop_mapping("map-a")

        self.assertEqual(result["error_code"], "mapping_session_mismatch")
        self.assertEqual(result["map_name"], "map-b")
        self.assertIs(supervisor._process, process)
        killpg.assert_not_called()

    def test_stop_mapping_without_persistable_map_releases_session(self) -> None:
        for diagnostics, diagnostics_monotonic, warning_code in (
            ({}, None, "static_map_status_unavailable"),
            (
                {
                    "session_name": "map-a",
                    "localization_state": "mapping",
                    "map_point_count": 0,
                },
                time.monotonic(),
                "static_map_not_ready",
            ),
        ):
            with self.subTest(warning_code=warning_code):
                supervisor = object.__new__(FastLivo2Supervisor)
                process = mock.Mock()
                process.poll.return_value = None
                supervisor._lock = threading.RLock()
                supervisor._process = process
                supervisor._active_map = "map-a"
                supervisor._loaded_map = None
                supervisor._runtime_mode = "mapping"
                supervisor._started_unix_ms = 123
                supervisor._pending_mapping_finalize = None
                supervisor._last_mapping_result = None
                supervisor._diagnostics = diagnostics
                supervisor._diagnostics_monotonic = diagnostics_monotonic
                supervisor._terminate_process = mock.Mock(return_value=None)

                result = supervisor._stop_mapping("map-a")

                self.assertEqual(result["status"], "stopped")
                self.assertFalse(result["map_saved"])
                self.assertEqual(result["warning_code"], warning_code)
                self.assertIsNone(supervisor._process)
                self.assertIsNone(supervisor._active_map)
                self.assertEqual(supervisor._runtime_mode, "idle")
                supervisor._terminate_process.assert_called_once_with(process)

    def test_raw_pcd_flush_uses_algorithm_parameter_service(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        supervisor._changed_pcd_files = mock.Mock(
            return_value=["tail_raw_points.pcd"]
        )
        process = mock.Mock()
        process.poll.return_value = None

        with mock.patch.object(
            SUPERVISOR_MODULE.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ) as run:
            result = supervisor._flush_raw_pcd(process, timeout_sec=12.0)

        self.assertEqual(result["status"], "flushed")
        self.assertEqual(result["pcd_files"], ["tail_raw_points.pcd"])
        command = run.call_args.args[0]
        self.assertEqual(command[:5], ["ros2", "param", "set", "/laserMapping", "pcd_save.flush_sequence"])
        self.assertEqual(run.call_args.kwargs["timeout"], 12.0)

    def test_stop_mapping_keeps_algorithm_running_when_raw_flush_fails(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        process = mock.Mock()
        process.poll.return_value = None
        supervisor._lock = threading.RLock()
        supervisor._process = process
        supervisor._active_map = "map-a"
        supervisor._loaded_map = None
        supervisor._runtime_mode = "mapping"
        supervisor._started_unix_ms = 123
        supervisor._pending_mapping_finalize = None
        supervisor._last_mapping_result = None
        supervisor._diagnostics = {
            "session_name": "map-a",
            "localization_state": "mapping",
            "map_point_count": 100,
        }
        supervisor._diagnostics_monotonic = time.monotonic()
        supervisor.get_parameter = lambda _name: SimpleNamespace(value=120.0)
        supervisor._changed_pcd_files = mock.Mock(return_value=[])
        supervisor._flush_raw_pcd = mock.Mock(
            return_value={
                "status": "error",
                "error_code": "map_artifact_flush_failed",
                "error": "flush rejected",
                "retryable": True,
            }
        )

        with mock.patch.object(SUPERVISOR_MODULE.os, "killpg") as killpg:
            result = supervisor._stop_mapping("map-a")

        self.assertEqual(result["error_code"], "map_artifact_flush_failed")
        self.assertTrue(result["retryable"])
        self.assertIs(supervisor._process, process)
        self.assertEqual(supervisor._runtime_mode, "mapping")
        killpg.assert_not_called()

    def test_stale_stop_request_cannot_finalize_other_pending_session(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        process = mock.Mock()
        supervisor._lock = threading.RLock()
        supervisor._process = process
        supervisor._active_map = "map-b"
        supervisor._loaded_map = None
        supervisor._runtime_mode = "finalizing"
        supervisor._started_unix_ms = 123
        pending = {"process": process, "map_name": "map-b"}
        supervisor._pending_mapping_finalize = pending
        supervisor._last_mapping_result = None
        supervisor._diagnostics = {}
        supervisor._diagnostics_monotonic = None
        supervisor._snapshot_session_pcd_files = mock.Mock()
        supervisor._adapter_execute = mock.Mock()

        result = supervisor._stop_mapping("map-a")

        self.assertEqual(result["error_code"], "mapping_session_mismatch")
        self.assertIs(supervisor._pending_mapping_finalize, pending)
        supervisor._snapshot_session_pcd_files.assert_not_called()
        supervisor._adapter_execute.assert_not_called()

    def test_manifest_aggregate_limit_counts_raw_and_static_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "raw.pcd").write_bytes(b"raw")
            (root / "static").mkdir()
            (root / "static" / "office.static.pcd").write_bytes(b"grid")
            (root / "sessions").mkdir()
            (root / "sessions" / "office.json").write_text(
                json.dumps(
                    {
                        "schema": "phanthy.navigation.fast_livo2_map_session.v1",
                        "static_map_format_version": 2,
                        "map_name": "office",
                        "pcd_files": ["raw.pcd"],
                        "static_map_pcd": "static/office.static.pcd",
                        "obstacle_height_range_m": [-0.30, 0.30],
                    }
                ),
                encoding="utf-8",
            )
            supervisor = object.__new__(FastLivo2Supervisor)
            supervisor._map_root = root

            with mock.patch.object(
                SUPERVISOR_MODULE,
                "_MAX_MAP_ARTIFACT_TOTAL_BYTES",
                6,
            ):
                with self.assertRaisesRegex(ValueError, "aggregate byte"):
                    supervisor._map_artifacts_from_manifest("office")

    def test_oversized_manifest_is_rejected_before_json_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "sessions").mkdir()
            (root / "sessions" / "office.json").write_bytes(
                b"{" + b"x" * SUPERVISOR_MODULE._MAX_MAP_MANIFEST_BYTES
            )
            supervisor = object.__new__(FastLivo2Supervisor)
            supervisor._map_root = root

            with mock.patch.object(
                SUPERVISOR_MODULE.json,
                "loads",
                side_effect=AssertionError("oversized manifest was decoded"),
            ) as loads:
                with self.assertRaisesRegex(ValueError, "manifest exceeds byte"):
                    supervisor._map_artifacts_from_manifest("office")

            loads.assert_not_called()

    def test_permanent_finalize_failure_cleans_uncommitted_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = root / "office-session.raw.pcd"
            static_root = root / "static"
            static_root.mkdir()
            static = static_root / "office.static.pcd"
            raw.write_bytes(b"raw")
            static.write_bytes(b"static")
            supervisor = object.__new__(FastLivo2Supervisor)
            supervisor._map_root = root
            supervisor.get_logger = lambda: SimpleNamespace(warning=lambda _msg: None)

            supervisor._cleanup_session_artifacts(
                [raw.name],
                {"static_map_file": static.name},
            )

            self.assertFalse(raw.exists())
            self.assertFalse(static.exists())

    def test_algorithm_start_cleanup_timeout_keeps_adapter_map_owned(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        supervisor._lock = threading.RLock()
        supervisor._process = None
        supervisor._active_map = None
        supervisor._loaded_map = None
        supervisor._runtime_mode = "idle"
        supervisor._started_unix_ms = None
        responses = iter(
            (
                {"status": "map_loaded", "loaded_map": "office"},
                {
                    "status": "error",
                    "error_code": "map_control_timeout",
                    "error": "adapter cleanup timed out",
                    "retryable": True,
                },
            )
        )
        supervisor._adapter_execute = lambda _action, _args: next(responses)
        supervisor._algorithm_command = lambda **_kwargs: ["fast-livo2"]

        with mock.patch.object(
            SUPERVISOR_MODULE.subprocess,
            "Popen",
            side_effect=OSError("cannot fork"),
        ):
            result = supervisor._activate_localization(
                "office",
                ((Path("raw.pcd"),), None, None),
            )

        self.assertEqual(result["error_code"], "algorithm_start_cleanup_pending")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["loaded_map"], "office")
        self.assertEqual(supervisor._loaded_map, "office")
        self.assertEqual(supervisor._runtime_mode, "localization")

    def test_replace_map_validates_active_artifacts_before_unload(self) -> None:
        supervisor = object.__new__(FastLivo2Supervisor)
        supervisor._lock = threading.RLock()
        supervisor._pending_mapping_finalize = None
        supervisor._process = mock.Mock()
        supervisor._process.poll.return_value = None
        supervisor._runtime_mode = "localization"
        supervisor._loaded_map = "map-a"
        target = ((Path("map-b.pcd"),), None, None)
        active = ((Path("map-a.pcd"),), None, None)
        supervisor._map_artifacts_from_manifest = mock.Mock(
            side_effect=[target, active]
        )
        supervisor._adapter_execute = mock.Mock(
            side_effect=[
                {"status": "map_validated"},
                {
                    "status": "error",
                    "error_code": "map_control_failed",
                    "error": "active PCD is corrupt",
                },
            ]
        )
        supervisor._unload_map = mock.Mock()

        result = supervisor._load_map("map-b")

        self.assertEqual(result["error_code"], "active_map_artifact_invalid")
        self.assertEqual(result["loaded_map"], "map-a")
        supervisor._unload_map.assert_not_called()


if __name__ == "__main__":
    unittest.main()
