from __future__ import annotations

import io
import json
import math
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError


ACTUCORE_DIR = Path(__file__).resolve().parents[1]
if str(ACTUCORE_DIR) not in sys.path:
    sys.path.insert(0, str(ACTUCORE_DIR))

from plugins.navigation.semantic.manifest import MANIFEST  # noqa: E402
from plugins.navigation.semantic.plugin import VisionAndLanguageNavigationPlugin  # noqa: E402
from plugins.navigation.semantic.ros import (  # noqa: E402
    MapSessionChangedError,
    Pose,
    RosBridge,
    Snapshot,
    _ImageSample,
    _PoseSample,
    pose_from_odometry,
)
from plugins.navigation.semantic.vlm import Client, _parse_json, validate_configuration  # noqa: E402
from utils.security import REDACTED, redact_sensitive  # noqa: E402


def _pose(x=1.25, y=-0.5, yaw=0.75) -> Pose:
    return Pose(
        x=x,
        y=y,
        z=0.0,
        qx=0.0,
        qy=0.0,
        qz=math.sin(yaw / 2.0),
        qw=math.cos(yaw / 2.0),
        yaw=yaw,
        frame_id="map",
        child_frame_id="base_link",
        source_timestamp=100.25,
        received_at=101.0,
        source_topic="/ubuntu/navigation/odom",
    )


def _snapshot(pose=None) -> Snapshot:
    return Snapshot(
        image=b"\xff\xd8fake-jpeg\xff\xd9",
        image_format="jpeg",
        image_mime_type="image/jpeg",
        image_source_timestamp=100.2,
        image_received_at=101.0,
        pose=pose or _pose(),
        receive_skew_sec=0.02,
    )


class FakeClient:
    configured = True
    model = "fake-vlm"

    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def complete_json(self, messages):
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("unexpected VLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(messages)
        return response

    @staticmethod
    def image_url(image, mime_type="image/jpeg"):
        return f"data:{mime_type};base64,ZmFrZQ=="


class FakeBridge:
    def __init__(
        self,
        *,
        snapshot=None,
        subscribers=1,
        session_id="demo#local-1",
        on_wait_for_subscriber=None,
        on_wait_for_snapshot=None,
        map_session_ready=True,
        map_session_issue="ready",
        **kwargs,
    ):
        self.camera_topic = kwargs["camera_topic"]
        self.odometry_topic = kwargs["odometry_topic"]
        self.goal_topic = kwargs["goal_topic"]
        self.status_topic = kwargs.get("status_topic", "")
        self.snapshot = snapshot or _snapshot()
        self.goal_subscribers = subscribers
        self.current_map_session_id = session_id
        self._instance_token = f"fake-bridge-{id(self)}"
        self.map_session_ready = map_session_ready
        self.map_session_issue = map_session_issue
        self.published = []
        self.closed = False
        self.wait_args = None
        self.on_wait_for_subscriber = on_wait_for_subscriber
        self.on_wait_for_snapshot = on_wait_for_snapshot

    def wait_for_snapshot(self, **kwargs):
        self.wait_args = kwargs
        if self.on_wait_for_snapshot:
            self.on_wait_for_snapshot(self)
        return self.snapshot

    def wait_for_goal_subscriber(self, timeout):
        if self.on_wait_for_subscriber:
            self.on_wait_for_subscriber(self)
        return self.goal_subscribers > 0

    def wait_for_map_session(self, timeout):
        return self.map_session_ready

    @property
    def current_map_session_token(self):
        return f"{self._instance_token}:{self.current_map_session_id or 'unknown'}"

    def publish_goal(self, payload, *, expected_map_session_token=None):
        if (
            expected_map_session_token is not None
            and expected_map_session_token != self.current_map_session_token
        ):
            raise MapSessionChangedError("fake session changed")
        self.published.append(json.loads(json.dumps(payload)))

    def close(self):
        self.closed = True


class FakeBridgeFactory:
    def __init__(self, **bridge_options):
        self.bridge_options = bridge_options
        self.instances = []
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        bridge = FakeBridge(**self.bridge_options, **kwargs)
        self.instances.append(bridge)
        return bridge


def _plugin(client, factory, *, goal_handler=None, **config):
    cfg = {
        "sensor_timeout_sec": 0.1,
        "subscriber_timeout_sec": 0.1,
        **config,
    }
    return VisionAndLanguageNavigationPlugin(
        cfg,
        "ubuntu",
        executor=object(),
        client=client,
        bridge_factory=factory,
        goal_handler=goal_handler,
    )


class ManifestTests(unittest.TestCase):
    def test_capture_is_zero_arg_and_navigate_requires_query(self):
        schema = MANIFEST["inputSchema"]
        self.assertIn("info", schema["properties"]["action"]["enum"])
        self.assertEqual(schema["x-action-params"]["capture"]["params"], [])
        self.assertEqual(schema["x-action-params"]["navigate"]["params"], ["query"])
        self.assertEqual(schema["required"], ["action"])

    def test_vlm_gear_config_schema_uses_a_password_field(self):
        schema = MANIFEST["inputSchema"]
        config_schema = MANIFEST["configSchema"]
        properties = config_schema["properties"]

        self.assertIn("config", schema["properties"]["action"]["enum"])
        self.assertNotIn("config", schema["x-action-params"])
        self.assertEqual(properties["api_key"]["format"], "password")
        self.assertTrue(properties["api_key"]["x-sensitive"])
        self.assertEqual(properties["api_key"]["scope"], "shared")
        self.assertEqual(config_schema["required"], [])
        self.assertNotIn("default", properties["api_key"])

    def test_ros_contract_is_explicit(self):
        self.assertEqual(MANIFEST["topic_in"][0]["ros_type"], "std_msgs/msg/UInt8MultiArray")
        self.assertEqual(MANIFEST["topic_in"][0]["schema"], "phanthy.sensor.camera_rgb_frame.v1")
        self.assertEqual(MANIFEST["topic_in"][1]["format"], "sensor/odometry")
        self.assertEqual(MANIFEST["topic_in"][1]["ros_type"], "nav_msgs/msg/Odometry")
        self.assertEqual(
            MANIFEST["topic_in"][2]["topic"],
            "/ubuntu/navigation/fast_livo2/status",
        )
        self.assertFalse(MANIFEST["topic_in"][2]["required"])
        self.assertEqual(MANIFEST["topic_out"][0]["topic"], "/ubuntu/navigation/goal_pose")
        self.assertEqual(MANIFEST["topic_out"][0]["schema"], "phanthy.navigation.goal.v1")


class ProcessorTests(unittest.TestCase):
    def test_valid_gear_config_atomically_replaces_vlm_without_echoing_key(self):
        secret = "unit-test-vlm-credential"
        plugin = _plugin(FakeClient([]), FakeBridgeFactory())
        result = plugin.dispatch(
            "vln",
            {
                "action": "config",
                "base_url": "https://vlm.example.test/v1/",
                "api_key": secret,
                "model": "vision-model",
                "timeout_sec": 12,
            },
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["adapter_ok"])
        self.assertTrue(result["vlm_api_key_configured"])
        self.assertNotIn(secret, json.dumps(result))
        info = plugin.dispatch("vln", {"action": "info"})
        self.assertEqual(info["vlm_model"], "vision-model")
        self.assertEqual(info["vlm_base_url"], "https://vlm.example.test/v1")
        self.assertEqual(info["vlm_timeout_sec"], 12.0)
        self.assertNotIn(secret, json.dumps(info))

    def test_invalid_gear_config_preserves_previous_vlm_client(self):
        invalid_configs = [
            {
                "base_url": "https://vlm.example.test/v1",
                "api_key": "",
                "model": "vision-model",
                "timeout_sec": 18,
            },
            {
                "base_url": "file:///tmp/model",
                "api_key": "test-key",
                "model": "vision-model",
                "timeout_sec": 18,
            },
            {
                "base_url": "https://user:pass@vlm.example.test/v1",
                "api_key": "test-key",
                "model": "vision-model",
                "timeout_sec": 18,
            },
            {
                "base_url": "https://vlm.example.test/v1",
                "api_key": "test-key",
                "model": "",
                "timeout_sec": 18,
            },
            {
                "base_url": "https://vlm.example.test/v1",
                "api_key": "test-key",
                "model": "vision-model",
                "timeout_sec": True,
            },
            {
                "base_url": "https://vlm.example.test/v1",
                "api_key": "test-key",
                "model": "vision-model",
                "timeout_sec": float("nan"),
            },
            {
                "base_url": "https://vlm.example.test/v1",
                "api_key": "test-key",
                "model": "vision-model",
                "timeout_sec": 121,
            },
        ]
        for invalid in invalid_configs:
            with self.subTest(config=invalid):
                plugin = _plugin(FakeClient([]), FakeBridgeFactory())
                result = plugin.dispatch("vln", {"action": "config", **invalid})
                self.assertFalse(result["ok"])
                self.assertFalse(result["adapter_ok"])
                self.assertEqual(result["error_code"], "invalid_vlm_config")
                self.assertEqual(
                    plugin.dispatch("vln", {"action": "info"})["vlm_model"],
                    "fake-vlm",
                )

    def test_capture_fails_closed_before_start_when_vlm_is_not_configured(self):
        plugin = VisionAndLanguageNavigationPlugin(
            {},
            "ubuntu",
            executor=object(),
            bridge_factory=FakeBridgeFactory(),
        )
        result = plugin.dispatch("vln", {"action": "capture"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "vlm_not_configured")

    def test_startup_yaml_values_override_environment_fallbacks(self):
        environment = {
            "VISION_AND_LANGUAGE_NAVIGATION_VLM_BASE_URL": "https://env.example.test/v1",
            "VISION_AND_LANGUAGE_NAVIGATION_VLM_API_KEY": "environment-key",
            "VISION_AND_LANGUAGE_NAVIGATION_VLM_MODEL": "environment-model",
            "VISION_AND_LANGUAGE_NAVIGATION_VLM_TIMEOUT_SEC": "27",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            plugin = VisionAndLanguageNavigationPlugin(
                {
                    "vlm": {
                        "base_url": "https://yaml.example.test/v1",
                        "api_key": "yaml-key",
                        "model": "yaml-model",
                        "timeout_sec": 11,
                    }
                },
                "ubuntu",
                executor=object(),
                bridge_factory=FakeBridgeFactory(),
            )
        info = plugin.dispatch("vln", {"action": "info"})
        self.assertTrue(info["vlm_configured"])
        self.assertEqual(info["vlm_base_url"], "https://yaml.example.test/v1")
        self.assertEqual(info["vlm_model"], "yaml-model")
        self.assertEqual(info["vlm_timeout_sec"], 11.0)

    def test_gear_config_waits_for_capture_and_preserves_recorded_points(self):
        entered_vlm = threading.Event()
        release_vlm = threading.Event()

        def slow_description(_messages):
            entered_vlm.set()
            if not release_vlm.wait(timeout=1.0):
                raise AssertionError("test did not release VLM")
            return {
                "scene": "办公室",
                "objects": ["白板"],
                "description": "白板办公室",
            }

        plugin = _plugin(FakeClient([slow_description]), FakeBridgeFactory())
        capture_results = []
        config_results = []
        capture_thread = threading.Thread(
            target=lambda: capture_results.append(
                plugin.dispatch("vln", {"action": "capture"})
            )
        )
        capture_thread.start()
        self.assertTrue(entered_vlm.wait(timeout=1.0))

        config_thread = threading.Thread(
            target=lambda: config_results.append(
                plugin.dispatch(
                    "vln",
                    {
                        "action": "config",
                        "base_url": "https://vlm.example.test/v1",
                        "api_key": "unit-test-key",
                        "model": "new-model",
                        "timeout_sec": 18,
                    },
                )
            )
        )
        config_thread.start()
        time.sleep(0.02)
        self.assertTrue(config_thread.is_alive())

        release_vlm.set()
        capture_thread.join(timeout=1.0)
        config_thread.join(timeout=1.0)
        self.assertTrue(capture_results[0]["ok"])
        self.assertTrue(config_results[0]["ok"])
        info = plugin.dispatch("vln", {"action": "info"})
        self.assertEqual(info["recorded_points"], 1)
        self.assertEqual(info["vlm_model"], "new-model")

    def test_start_classifies_unordered_flat_input_topics(self):
        factory = FakeBridgeFactory()
        plugin = _plugin(FakeClient([]), factory)
        result = plugin.dispatch(
            "vln",
            {
                "action": "start",
                "input_topics": [
                    "/ubuntu/navigation/odom",
                    "/ubuntu/camera/rgb_frame",
                ],
            },
        )
        self.assertEqual(result["state"], "running")
        self.assertEqual(
            factory.calls[0]["camera_topic"],
            "/ubuntu/camera/rgb_frame",
        )
        self.assertEqual(factory.calls[0]["odometry_topic"], "/ubuntu/navigation/odom")
        self.assertEqual(
            factory.calls[0]["status_topic"],
            "/ubuntu/navigation/fast_livo2/status",
        )
        self.assertEqual(factory.calls[0]["goal_topic"], "/ubuntu/navigation/goal_pose")

    def test_start_supports_port_aware_input_bindings(self):
        factory = FakeBridgeFactory()
        plugin = _plugin(FakeClient([]), factory)
        result = plugin.dispatch(
            "vln",
            {
                "action": "start",
                "input_bindings": {
                    "livo_odom": {"topic": "/robot/navigation/odom"},
                    "rgb": {"topic": "/robot/camera/rgb_frame"},
                    "livo_status": {"topic": "/robot/fast_livo2/status"},
                },
            },
        )
        self.assertEqual(result["state"], "running")
        self.assertEqual(
            factory.calls[0]["camera_topic"],
            "/robot/camera/rgb_frame",
        )
        self.assertEqual(factory.calls[0]["odometry_topic"], "/robot/navigation/odom")
        self.assertEqual(factory.calls[0]["status_topic"], "/robot/fast_livo2/status")

    def test_start_classifies_optional_status_in_flat_topics(self):
        factory = FakeBridgeFactory()
        plugin = _plugin(FakeClient([]), factory)
        result = plugin.dispatch(
            "vln",
            {
                "action": "start",
                "input_topics": [
                    "/robot/fast_livo2/status",
                    "/ubuntu/navigation/odom",
                    "/ubuntu/camera/rgb_frame",
                ],
            },
        )
        self.assertEqual(result["state"], "running")
        self.assertEqual(factory.calls[0]["status_topic"], "/robot/fast_livo2/status")

    def test_explicit_invalid_wiring_never_falls_back_to_defaults(self):
        invalid_arguments = [
            {"input_bindings": "bad"},
            {"input_bindings": {"unexpected": "/bad/topic"}},
            {"input_bindings": {}},
            {"input_topics": ["/ubuntu/camera/rgb_frame"]},
            {
                "input_topics": [
                    "/ubuntu/camera/rgb_frame",
                    "/ubuntu/navigation/odom",
                    "/unexpected/topic",
                ]
            },
        ]
        for wiring in invalid_arguments:
            with self.subTest(wiring=wiring):
                factory = FakeBridgeFactory()
                plugin = _plugin(FakeClient([]), factory)
                result = plugin.dispatch("vln", {"action": "start", **wiring})
                self.assertFalse(result["ok"])
                self.assertEqual(result["error_code"], "invalid_canvas_wiring")
                self.assertEqual(factory.calls, [])

    def test_capture_waits_for_new_pair_and_records_vlm_description(self):
        client = FakeClient(
            [
                {
                    "scene": "办公室",
                    "objects": ["白板", "蓝色沙发", "白板"],
                    "description": "有白板和蓝色沙发的办公室",
                }
            ]
        )
        factory = FakeBridgeFactory()
        plugin = _plugin(client, factory)
        result = plugin.dispatch("vln", {"action": "capture"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "captured")
        self.assertEqual(result["waypoint"]["point_id"], "vln_point_0001")
        self.assertEqual(result["waypoint"]["objects"], ["白板", "蓝色沙发"])
        self.assertEqual(result["waypoint"]["pose"]["frame_id"], "map")
        self.assertIsNotNone(factory.instances[0].wait_args["after_monotonic"])
        self.assertEqual(len(factory.instances[0].published), 0)

    def test_capture_timeout_does_not_create_partial_point(self):
        factory = FakeBridgeFactory(snapshot=None)
        plugin = _plugin(FakeClient([]), factory)
        # Override the default FakeBridge snapshot after construction.
        plugin.dispatch("vln", {"action": "start"})
        factory.instances[0].snapshot = None
        result = plugin.dispatch("vln", {"action": "capture"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "sensor_timeout")
        self.assertEqual(plugin.dispatch("vln", {"action": "info"})["recorded_points"], 0)

    def test_navigate_no_points_never_publishes(self):
        factory = FakeBridgeFactory()
        plugin = _plugin(FakeClient([]), factory)
        result = plugin.dispatch("vln", {"action": "navigate", "query": "白板旁边"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "not_found")
        self.assertFalse(result["navigation_requested"])
        self.assertEqual(factory.instances[0].published, [])

    def test_low_confidence_match_never_publishes(self):
        client = FakeClient(
            [
                {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"},
                {"point_id": "vln_point_0001", "confidence": 0.3, "reason": "不确定"},
            ]
        )
        factory = FakeBridgeFactory()
        plugin = _plugin(client, factory, match_threshold=0.55)
        plugin.dispatch("vln", {"action": "capture"})
        result = plugin.dispatch("vln", {"action": "navigate", "query": "去白板"})
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(factory.instances[0].published, [])

    def test_hallucinated_point_id_never_publishes(self):
        client = FakeClient(
            [
                {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"},
                {"point_id": "invented", "confidence": 1.0, "reason": "错误 ID"},
            ]
        )
        factory = FakeBridgeFactory()
        plugin = _plugin(client, factory)
        plugin.dispatch("vln", {"action": "capture"})
        result = plugin.dispatch("vln", {"action": "navigate", "query": "去白板"})
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(factory.instances[0].published, [])

    def test_boolean_confidence_never_publishes(self):
        client = FakeClient(
            [
                {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"},
                {"point_id": "vln_point_0001", "confidence": True, "reason": "非法类型"},
            ]
        )
        factory = FakeBridgeFactory()
        plugin = _plugin(client, factory)
        plugin.dispatch("vln", {"action": "capture"})
        result = plugin.dispatch("vln", {"action": "navigate", "query": "去白板"})
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(factory.instances[0].published, [])

    def test_invalid_confidence_never_publishes_at_zero_threshold(self):
        for invalid_confidence in (True, None, "0.9", float("nan"), -0.1, 0.0, 1.1):
            with self.subTest(confidence=invalid_confidence):
                client = FakeClient(
                    [
                        {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"},
                        {
                            "point_id": "vln_point_0001",
                            "confidence": invalid_confidence,
                            "reason": "非法置信度",
                        },
                    ]
                )
                factory = FakeBridgeFactory()
                plugin = _plugin(client, factory, match_threshold=0)
                plugin.dispatch("vln", {"action": "capture"})
                result = plugin.dispatch(
                    "vln", {"action": "navigate", "query": "去白板"}
                )
                self.assertEqual(result["status"], "not_found")
                self.assertEqual(factory.instances[0].published, [])

    def test_match_publishes_exact_ros_goal_contract_once(self):
        client = FakeClient(
            [
                {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"},
                {"point_id": "vln_point_0001", "confidence": 0.91, "reason": "白板明确匹配"},
            ]
        )
        factory = FakeBridgeFactory(subscribers=1)
        plugin = _plugin(client, factory, navigation_speed=0.5)
        plugin.dispatch("vln", {"action": "capture"})
        result = plugin.dispatch("vln", {"action": "navigate", "query": "去白板办公室"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "navigation_requested")
        self.assertTrue(result["goal_published"])
        self.assertEqual(len(factory.instances[0].published), 1)
        goal = factory.instances[0].published[0]
        self.assertEqual(set(goal), {"schema", "goal_id", "x", "y", "yaw", "speed"})
        self.assertEqual(goal["schema"], "phanthy.navigation.goal.v1")
        self.assertTrue(goal["goal_id"].startswith("vln-"))
        self.assertEqual((goal["x"], goal["y"], goal["yaw"]), (1.25, -0.5, 0.75))
        self.assertEqual(goal["speed"], 0.5)

    def test_unified_card_delivers_match_directly_to_planner_with_same_lease(self):
        client = FakeClient(
            [
                {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"},
                {"point_id": "vln_point_0001", "confidence": 0.91, "reason": "匹配"},
            ]
        )
        factory = FakeBridgeFactory(subscribers=0)
        calls = []

        def handle(goal, *, control_nav_id=None):
            calls.append((dict(goal), control_nav_id))
            return {"status": "navigating", "nav_id": control_nav_id}

        plugin = _plugin(client, factory, goal_handler=handle, navigation_speed=0.5)
        plugin.dispatch("vln", {"action": "capture"})
        result = plugin.dispatch(
            "vln",
            {
                "action": "navigate",
                "query": "去白板办公室",
                "_control_nav_id": "lease-unified-1",
            },
        )

        self.assertTrue(result["navigation_requested"])
        self.assertFalse(result["goal_published"])
        self.assertEqual(result["goal_delivery"], "in_process_planner")
        self.assertEqual(calls[0][1], "lease-unified-1")
        self.assertEqual(calls[0][0]["speed"], 0.5)
        self.assertEqual(factory.instances[0].published, [])

    def test_no_subscriber_still_publishes_and_reports_delivery_warning(self):
        client = FakeClient(
            [
                {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"},
                {"point_id": "vln_point_0001", "confidence": 0.9, "reason": "匹配"},
            ]
        )
        factory = FakeBridgeFactory(subscribers=0)
        plugin = _plugin(client, factory)
        plugin.dispatch("vln", {"action": "capture"})
        result = plugin.dispatch("vln", {"action": "navigate", "query": "办公室"})
        self.assertTrue(result["ok"])
        self.assertTrue(result["goal_published"])
        self.assertFalse(result["downstream_subscriber_ready"])
        self.assertEqual(len(factory.instances[0].published), 1)

    def test_session_change_blocks_old_coordinate(self):
        client = FakeClient(
            [{"scene": "办公室", "objects": ["白板"], "description": "白板办公室"}]
        )
        factory = FakeBridgeFactory(session_id="map-a#local-1")
        plugin = _plugin(client, factory)
        plugin.dispatch("vln", {"action": "capture"})
        factory.instances[0].current_map_session_id = "map-b#local-2"
        result = plugin.dispatch("vln", {"action": "navigate", "query": "办公室"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "map_session_mismatch")
        self.assertEqual(factory.instances[0].published, [])

    def test_capture_rejects_session_change_during_vlm(self):
        factory = FakeBridgeFactory(session_id="map-a#local-1")

        def change_session(_messages):
            factory.instances[0].current_map_session_id = "map-b#local-2"
            return {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"}

        plugin = _plugin(FakeClient([change_session]), factory)
        result = plugin.dispatch("vln", {"action": "capture"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "map_session_changed")
        self.assertEqual(plugin.dispatch("vln", {"action": "info"})["recorded_points"], 0)

    def test_capture_rejects_map_becoming_unready_during_sensor_wait(self):
        def lose_mapping(bridge):
            bridge.map_session_ready = False
            bridge.map_session_issue = "not_mapping"

        factory = FakeBridgeFactory(on_wait_for_snapshot=lose_mapping)
        plugin = _plugin(FakeClient([]), factory)
        result = plugin.dispatch("vln", {"action": "capture"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "map_session_changed")
        self.assertEqual(plugin.dispatch("vln", {"action": "info"})["recorded_points"], 0)

    def test_session_change_during_subscriber_wait_never_publishes(self):
        def change_session(bridge):
            bridge.current_map_session_id = "map-b#local-2"

        client = FakeClient(
            [
                {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"},
                {"point_id": "vln_point_0001", "confidence": 0.9, "reason": "匹配"},
            ]
        )
        factory = FakeBridgeFactory(
            session_id="map-a#local-1",
            on_wait_for_subscriber=change_session,
        )
        plugin = _plugin(client, factory)
        plugin.dispatch("vln", {"action": "capture"})
        result = plugin.dispatch("vln", {"action": "navigate", "query": "办公室"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "map_session_changed")
        self.assertEqual(factory.instances[0].published, [])

    def test_bridge_restart_invalidates_previous_bridge_coordinates(self):
        client = FakeClient(
            [{"scene": "办公室", "objects": ["白板"], "description": "白板办公室"}]
        )
        factory = FakeBridgeFactory(session_id="same-map#local-1")
        plugin = _plugin(client, factory)
        plugin.dispatch("vln", {"action": "capture"})
        plugin.dispatch("vln", {"action": "stop"})
        plugin.dispatch("vln", {"action": "start"})
        result = plugin.dispatch("vln", {"action": "navigate", "query": "办公室"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "map_session_mismatch")
        self.assertEqual(factory.instances[1].published, [])

    def test_missing_fast_livo_status_blocks_capture(self):
        factory = FakeBridgeFactory(
            map_session_ready=False,
            map_session_issue="status_missing",
        )
        plugin = _plugin(FakeClient([]), factory)
        result = plugin.dispatch("vln", {"action": "capture"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "map_session_unavailable")
        self.assertEqual(result["map_session_issue"], "status_missing")

    def test_stop_is_idempotent_and_closes_bridge(self):
        factory = FakeBridgeFactory()
        plugin = _plugin(FakeClient([]), factory)
        plugin.dispatch("vln", {"action": "start"})
        first = plugin.dispatch("vln", {"action": "stop"})
        second = plugin.dispatch("vln", {"action": "stop"})
        self.assertEqual(first["state"], "idle")
        self.assertEqual(second["state"], "idle")
        self.assertFalse(first["map_session_ready"])
        self.assertEqual(first["map_session_issue"], "no_bridge")
        self.assertTrue(factory.instances[0].closed)

    def test_stop_waits_for_inflight_capture_before_closing_bridge(self):
        entered_vlm = threading.Event()
        release_vlm = threading.Event()

        def slow_description(_messages):
            entered_vlm.set()
            if not release_vlm.wait(timeout=1.0):
                raise AssertionError("test did not release VLM")
            return {"scene": "办公室", "objects": ["白板"], "description": "白板办公室"}

        factory = FakeBridgeFactory()
        plugin = _plugin(FakeClient([slow_description]), factory)
        capture_results = []
        stop_results = []
        capture_thread = threading.Thread(
            target=lambda: capture_results.append(
                plugin.dispatch("vln", {"action": "capture"})
            )
        )
        capture_thread.start()
        self.assertTrue(entered_vlm.wait(timeout=1.0))

        stop_thread = threading.Thread(
            target=lambda: stop_results.append(
                plugin.dispatch("vln", {"action": "stop"})
            )
        )
        stop_thread.start()
        time.sleep(0.02)
        self.assertTrue(stop_thread.is_alive())
        self.assertFalse(factory.instances[0].closed)

        release_vlm.set()
        capture_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)
        self.assertTrue(capture_results[0]["ok"])
        self.assertEqual(stop_results[0]["state"], "idle")
        self.assertTrue(factory.instances[0].closed)

    def test_unknown_action_is_structured_error(self):
        plugin = _plugin(FakeClient([]), FakeBridgeFactory())
        result = plugin.dispatch("vln", {"action": "explode"})
        self.assertEqual(result["error_code"], "unsupported_action")


class PureHelperTests(unittest.TestCase):
    @staticmethod
    def _bare_ros_bridge(synchronization_mode="receive_time"):
        now = time.monotonic()
        bridge = object.__new__(RosBridge)
        bridge._condition = threading.Condition()
        bridge._closed = False
        bridge.synchronization_mode = synchronization_mode
        bridge.status_topic = ""
        bridge._status_state = ""
        bridge._status_map_name = ""
        bridge._status_generation = 0
        bridge._status_received_monotonic = None
        bridge._status_companion_ready = False
        bridge._status_algorithm_running = False
        bridge._status_session_ready = False
        bridge._status_stale_after_sec = 3.5
        bridge._status_restart_gap_sec = 5.0
        bridge._bridge_instance_id = "test-bridge"
        bridge._latest_image = _ImageSample(
            data=b"jpeg",
            image_format="jpeg",
            mime_type="image/jpeg",
            source_timestamp=1.0,
            received_at=time.time(),
            received_monotonic=now - 0.01,
        )
        bridge._latest_pose = _PoseSample(
            pose=_pose(),
            received_monotonic=now,
        )
        return bridge

    def test_odometry_conversion_normalizes_quaternion_and_computes_yaw(self):
        message = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="map",
                stamp=SimpleNamespace(sec=12, nanosec=500_000_000),
            ),
            child_frame_id="base_link",
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=2, y=3, z=0),
                    orientation=SimpleNamespace(x=0, y=0, z=2, w=2),
                )
            ),
        )
        pose = pose_from_odometry(message, "/odom", 20.0)
        self.assertAlmostEqual(pose.qz, math.sqrt(0.5))
        self.assertAlmostEqual(pose.qw, math.sqrt(0.5))
        self.assertAlmostEqual(pose.yaw, math.pi / 2)
        self.assertEqual(pose.source_timestamp, 12.5)

    def test_rgb_frame_is_decoded_for_semantic_snapshot(self):
        bridge = self._bare_ros_bridge()
        bridge._decode_camera_rgb_frame = mock.Mock(
            return_value=({"source_stamp_ns": 12_500_000_000}, b"jpeg-frame")
        )
        bridge._on_image(SimpleNamespace(data=b"PSE1-frame"))

        self.assertEqual(bridge._latest_image.data, b"jpeg-frame")
        self.assertEqual(bridge._latest_image.source_timestamp, 12.5)

    def test_vlm_json_parser_handles_markdown_and_rejects_non_json(self):
        self.assertEqual(_parse_json("```json\n{\"ok\": true}\n```"), {"ok": True})
        with self.assertRaises(RuntimeError):
            _parse_json("no object here")
        with self.assertRaises(RuntimeError):
            _parse_json('{"point_id":"example"} then {"point_id":null}')

    def test_vlm_client_with_empty_base_url_is_not_configured(self):
        client = Client(base_url="", api_key="demo", model="demo")
        self.assertFalse(client.configured)

    def test_vlm_client_has_no_default_api_key(self):
        client = Client()
        self.assertFalse(client.configured)
        self.assertFalse(client.api_key_configured)

    def test_invalid_startup_timeout_falls_back_to_finite_default(self):
        client = Client(
            base_url="https://vlm.example.test/v1",
            api_key="test-key",
            model="vision-model",
            timeout=float("nan"),
        )
        self.assertTrue(client.configured)
        self.assertEqual(client.timeout_sec, 18.0)
        json.dumps({"timeout_sec": client.timeout_sec}, allow_nan=False)

    def test_vlm_config_validation_normalizes_safe_values(self):
        result = validate_configuration(
            " https://vlm.example.test/v1/ ",
            " test-key ",
            " vision-model ",
            18,
        )
        self.assertEqual(
            result,
            (
                "https://vlm.example.test/v1",
                "test-key",
                "vision-model",
                18.0,
            ),
        )

    def test_vlm_provider_error_body_is_not_forwarded(self):
        secret = "provider-echoed-secret"
        client = Client(
            base_url="https://vlm.example.test/v1",
            api_key="test-key",
            model="vision-model",
        )
        upstream_error = HTTPError(
            client.base_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(secret.encode()),
        )
        with mock.patch(
            "plugins.navigation.semantic.vlm.urlopen",
            side_effect=upstream_error,
        ):
            with self.assertRaises(RuntimeError) as raised:
                client.complete_json([])
        self.assertEqual(str(raised.exception), "VLM request failed with HTTP 401")
        self.assertNotIn(secret, str(raised.exception))

    def test_recursive_log_redaction_does_not_mutate_arguments(self):
        arguments = {
            "action": "config",
            "api_key": "top-secret",
            "nested": [
                {"Authorization": "Bearer another-secret"},
                {"client_secret": "third-secret", "keyframe": "keep-this"},
            ],
        }
        safe = redact_sensitive(arguments)
        serialized = repr(safe)

        self.assertEqual(safe["api_key"], REDACTED)
        self.assertEqual(safe["nested"][0]["Authorization"], REDACTED)
        self.assertEqual(safe["nested"][1]["client_secret"], REDACTED)
        self.assertEqual(safe["nested"][1]["keyframe"], "keep-this")
        self.assertNotIn("top-secret", serialized)
        self.assertEqual(arguments["api_key"], "top-secret")

    def test_receive_time_sync_tolerates_unproven_source_clock_offset(self):
        bridge = self._bare_ros_bridge("receive_time")
        snapshot = bridge.wait_for_snapshot(
            timeout=0.01,
            max_age=1.0,
            max_skew=0.35,
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.synchronization_basis, "receive_time")
        self.assertAlmostEqual(snapshot.source_skew_sec, 99.25)

    def test_source_time_sync_is_opt_in_and_rejects_large_offset(self):
        bridge = self._bare_ros_bridge("source_timestamp")
        snapshot = bridge.wait_for_snapshot(
            timeout=0.01,
            max_age=1.0,
            max_skew=0.35,
        )
        self.assertIsNone(snapshot)

    def test_snapshot_pair_must_arrive_after_capture_invocation(self):
        bridge = self._bare_ros_bridge("receive_time")
        snapshot = bridge.wait_for_snapshot(
            timeout=0.01,
            max_age=1.0,
            max_skew=0.35,
            after_monotonic=time.monotonic(),
        )
        self.assertIsNone(snapshot)

    def test_fast_livo_status_controls_session_readiness_and_epoch(self):
        bridge = self._bare_ros_bridge()
        bridge.status_topic = "/ubuntu/navigation/fast_livo2/status"
        self.assertFalse(bridge.map_session_ready)
        self.assertEqual(bridge.map_session_issue, "status_missing")

        heartbeat = {
            "event": "heartbeat",
            "schema": "phanthy.navigation.fast_livo2_status.v1",
            "state": "mapping",
            "active_map": "demo",
            "algorithm_running": True,
            "companion_ready": True,
        }
        bridge._on_status(SimpleNamespace(data=json.dumps(heartbeat)))
        self.assertTrue(bridge.map_session_ready)
        self.assertEqual(bridge.current_map_session_id, "demo#local-1")
        ready_token = bridge.current_map_session_token

        bridge._status_received_monotonic = time.monotonic() - 4.0
        self.assertFalse(bridge.map_session_ready)
        self.assertEqual(bridge.map_session_issue, "status_stale")
        self.assertNotEqual(bridge.current_map_session_token, ready_token)
        bridge._on_status(SimpleNamespace(data=json.dumps(heartbeat)))
        self.assertTrue(bridge.map_session_ready)
        self.assertEqual(bridge.current_map_session_id, "demo#local-2")
        self.assertNotEqual(bridge.current_map_session_token, ready_token)

    def test_algorithm_and_companion_recovery_roll_session_epoch(self):
        bridge = self._bare_ros_bridge()
        bridge.status_topic = "/ubuntu/navigation/fast_livo2/status"
        heartbeat = {
            "schema": "phanthy.navigation.fast_livo2_status.v1",
            "state": "mapping",
            "active_map": "demo",
            "algorithm_running": True,
            "companion_ready": True,
        }
        bridge._on_status(SimpleNamespace(data=json.dumps(heartbeat)))
        first_token = bridge.current_map_session_token

        algorithm_down = {**heartbeat, "algorithm_running": False}
        bridge._on_status(SimpleNamespace(data=json.dumps(algorithm_down)))
        self.assertFalse(bridge.map_session_ready)
        bridge._on_status(SimpleNamespace(data=json.dumps(heartbeat)))
        self.assertEqual(bridge.current_map_session_id, "demo#local-2")
        second_token = bridge.current_map_session_token
        self.assertNotEqual(second_token, first_token)

        companion_down = {**heartbeat, "companion_ready": False}
        bridge._on_status(SimpleNamespace(data=json.dumps(companion_down)))
        self.assertFalse(bridge.map_session_ready)
        bridge._on_status(SimpleNamespace(data=json.dumps(heartbeat)))
        self.assertEqual(bridge.current_map_session_id, "demo#local-3")
        self.assertNotEqual(bridge.current_map_session_token, second_token)

    def test_confirmed_relocalization_is_a_ready_map_session(self):
        bridge = self._bare_ros_bridge()
        bridge.status_topic = "/ubuntu/navigation/fast_livo2/status"
        heartbeat = {
            "schema": "phanthy.navigation.fast_livo2_status.v1",
            "state": "relocalized",
            "loaded_map": "office",
            "algorithm_running": True,
            "companion_ready": True,
            "diagnostics": {"map_alignment_confirmed": True},
        }

        bridge._on_status(SimpleNamespace(data=json.dumps(heartbeat)))

        self.assertTrue(bridge.map_session_ready)
        self.assertEqual(bridge.current_map_session_id, "office#local-1")

        heartbeat["diagnostics"] = {"map_alignment_confirmed": False}
        bridge._on_status(SimpleNamespace(data=json.dumps(heartbeat)))
        self.assertFalse(bridge.map_session_ready)

    def test_ros_goal_publish_checks_session_and_serializes_exact_json(self):
        bridge = self._bare_ros_bridge()

        class FakeString:
            data = ""

        class FakePublisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message.data)

        bridge._String = FakeString
        bridge._goal_publisher = FakePublisher()
        token = bridge.current_map_session_token
        goal = {
            "schema": "phanthy.navigation.goal.v1",
            "goal_id": "vln-test",
            "x": 1.0,
            "y": 2.0,
            "yaw": 0.5,
            "speed": 0.3,
        }
        bridge.publish_goal(goal, expected_map_session_token=token)
        self.assertEqual(json.loads(bridge._goal_publisher.messages[0]), goal)

        bridge._status_state = "changed"
        with self.assertRaises(MapSessionChangedError):
            bridge.publish_goal(goal, expected_map_session_token=token)
        self.assertEqual(len(bridge._goal_publisher.messages), 1)


if __name__ == "__main__":
    unittest.main()
