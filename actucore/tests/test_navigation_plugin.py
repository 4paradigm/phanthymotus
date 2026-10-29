from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import jsonschema
except ImportError:
    jsonschema = None


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))

from plugins.navigation.contract import NAVIGATION_ACTIONS, navigation_tool_definition  # noqa: E402
from plugins.navigation.mapping.backend import RosTopicFastLivo2Backend  # noqa: E402
from plugins.navigation.mapping.core import FastLivo2BackendError  # noqa: E402
from plugins.navigation.mapping.plugin import FastLivo2Plugin  # noqa: E402
from plugins.navigation.plugin import NavigationPlugin  # noqa: E402
from plugins.navigation.runtime import (  # noqa: E402
    NavigationRuntime,
    _OwnedProcessGroup,
)


class FakeRuntime:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.started = False
        self.stop_calls = 0
        self.start_kwargs = None

    def start(self, **kwargs):
        if self.fail:
            raise RuntimeError("runtime boom")
        self.start_kwargs = dict(kwargs)
        self.started = True
        return self.info()

    def stop(self):
        self.stop_calls += 1
        self.started = False
        return self.info()

    def info(self):
        return {
            "state": "running" if self.started else "idle",
            "running": self.started,
            "container_model": "single_actucore_container",
            "docker_runtime_dependency": False,
        }


class FakeComponent:
    def __init__(self, name, *, fail_start=False, stop_result=None):
        self.name = name
        self.fail_start = fail_start
        self.stop_result = stop_result
        self.calls = []
        self.started = False

    def dispatch(self, prefix, args):
        self.calls.append((prefix, dict(args)))
        action = args.get("action")
        if action == "start":
            if self.fail_start:
                return {"state": "error", "status": "error", "error": "boom"}
            self.started = True
            return {"state": "ready", "status": "ready"}
        if action == "stop":
            self.started = False
            if self.stop_result is not None:
                return dict(self.stop_result)
            return {"state": "idle", "status": "idle"}
        if action == "info":
            return {"state": "ready" if self.started else "idle"}
        return {"status": action, "component": self.name}


class FailingConfigComponent(FakeComponent):
    def dispatch(self, prefix, args):
        if args.get("action") == "config":
            self.calls.append((prefix, dict(args)))
            return {
                "state": "error",
                "status": "invalid_config",
                "error": "rejected test config",
            }
        return super().dispatch(prefix, args)


class CollectionMappingComponent(FakeComponent):
    def dispatch(self, prefix, args):
        if args.get("action") == "info":
            self.calls.append((prefix, dict(args)))
            return {
                "state": "ready" if self.started else "idle",
                "config": {"collection_enabled": True},
            }
        return super().dispatch(prefix, args)


class TransientPlanningComponent(FakeComponent):
    def __init__(self):
        super().__init__("planning")
        self.start_attempts = 0

    def dispatch(self, prefix, args):
        if args.get("action") == "start":
            self.calls.append((prefix, dict(args)))
            self.start_attempts += 1
            if self.start_attempts == 1:
                return {
                    "state": "error",
                    "status": "error",
                    "error_code": "nav2_runtime_unavailable",
                    "error": (
                        "in-container Nav2 runtime is not subscribed to the "
                        "command topic"
                    ),
                }
            self.started = True
            return {"state": "ready", "status": "ready"}
        return super().dispatch(prefix, args)


class BlockingStartComponent(FakeComponent):
    def __init__(self):
        super().__init__("mapping")
        self.start_entered = threading.Event()
        self.release_start = threading.Event()
        self.stop_entered = threading.Event()

    def dispatch(self, prefix, args):
        if args.get("action") == "start":
            self.calls.append((prefix, dict(args)))
            self.start_entered.set()
            if not self.release_start.wait(timeout=2.0):
                raise RuntimeError("test did not release mapping start")
            self.started = True
            return {"state": "ready", "status": "ready"}
        if args.get("action") == "stop":
            self.stop_entered.set()
        return super().dispatch(prefix, args)


class RetryableMappingBackend:
    def __init__(self):
        self.stop_calls = 0

    def info(self):
        return {
            "state": "ready",
            "status": "ready",
            "backend": "fast_livo2_ros_topic",
            "bridge_subscribers": 1,
        }

    def execute(self, action, args):
        if action == "configure_obstacle_filter":
            return {"status": "configured"}
        if action == "configure_collection":
            return {"status": "recording" if args["enabled"] else "disabled"}
        if action == "start_mapping":
            return {"status": "mapping", "map_name": args["map_name"]}
        if action == "stop_mapping":
            raise FastLivo2BackendError(
                "manifest_write_failed",
                "temporary persistence failure",
                details={"retryable": True},
            )
        raise AssertionError(f"unexpected mapping action: {action}")

    def stop(self):
        self.stop_calls += 1


def _control_descriptor(**overrides):
    descriptor = {
        "control_interface": "motus.control/1",
        "mode": "twist",
        "dof": 6,
        "joint_names": ["vx", "vy", "vz", "wx", "wy", "wz"],
        "limits": {"lower": [-1.0] * 6, "upper": [1.0] * 6},
        "rate": {"max_hz": 20, "expected_hz": 8, "watchdog_ms": 300},
        "vendor": {"note": "round trip"},
    }
    descriptor.update(overrides)
    return descriptor


def _external_bindings():
    return [
        {"port": item["port"], "topic": item["topic"]}
        for item in navigation_tool_definition("ubuntu")["topic_in"]
        if item["required"]
    ]


class NavigationContractTest(unittest.TestCase):
    @unittest.skipIf(jsonschema is None, "Agent Core jsonschema dependency is unavailable")
    def test_mainline_lifecycle_payloads_pass_strict_schema(self):
        schema = navigation_tool_definition("ubuntu")["inputSchema"]
        topics = [
            "/ubuntu/navigation/lidar", "/ubuntu/navigation/imu",
            "/ubuntu/camera/rgb_frame", "/ubuntu/camera/depth_frame",
        ]
        # Upstream _upstream_camera_info copies Driver declarations by topic.
        rgb = {
            "schema": "motus.camera/1", "topic": topics[2],
            "id": "vendor/model/rgb", "format": "sensor/image",
            "width": 640, "height": 480, "distortion_model": "pinhole",
            "D": None, "K": None, "half_fov_rad": None,
            "half_fov_v_rad": None, "source": "unknown", "measured_on": "",
            "pipeline": [], "vendor": {},
        }
        depth = {**rgb, "topic": topics[3], "id": "vendor/model/depth"}
        start = {
            "action": "start", "instance_id": "card-1", "input_topics": topics,
            "input_topic": topics[0],
            "camera_info": {topics[2]: rgb, topics[3]: depth},
        }
        info = {
            "action": "info", "instance_id": "card-1", "input_topics": topics,
            "input_topic": topics[0],
        }
        for args in (start, info, {"action": "info", "instance_id": "card-1"},
                     {"action": "start", "instance_id": "card-1",
                      "input_topic": topics[0]},
                     {"action": "start", "control_interface": {}},
                     {"action": "start", "control_interface": _control_descriptor()}):
            with self.subTest(args=args):
                jsonschema.validate(instance=args, schema=schema)
        for value in (None, [], "descriptor"):
            with self.subTest(control_interface=value):
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(
                        instance={"action": "start", "control_interface": value},
                        schema=schema,
                    )
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance={"action": "info", "unexpected": True},
                                schema=schema)

    def test_one_public_card_hides_internal_ros_edges(self):
        tool = navigation_tool_definition("ubuntu")
        self.assertEqual(tool["name"], "controlled_semantic_spatial")
        self.assertEqual(tool["displayName"], "controlled_semantic_spatial")
        self.assertEqual(
            {item["port"] for item in tool["topic_in"]},
            {"lidar", "imu", "rgb", "depth_frame", "goal_pose"},
        )
        optional_inputs = {
            item["port"]
            for item in tool["topic_in"]
            if not item.get("required", True)
        }
        self.assertEqual(
            optional_inputs,
            {"rgb", "depth_frame", "goal_pose"},
        )
        self.assertEqual(
            {
                item["port"]
                for item in tool["topic_in"]
                if item["required"]
            },
            {"lidar", "imu"},
        )
        inputs = {item["port"]: item for item in tool["topic_in"]}
        self.assertEqual(
            inputs["rgb"]["topic"],
            "/ubuntu/camera/rgb_frame",
        )
        self.assertEqual(
            inputs["rgb"]["format"],
            "application/vnd.phanthy.sensor-envelope.v1",
        )
        self.assertEqual(inputs["rgb"]["schema"], "phanthy.sensor.camera_rgb_frame.v1")
        self.assertEqual(
            inputs["depth_frame"]["topic"],
            "/ubuntu/camera/depth_frame",
        )
        self.assertEqual(
            inputs["depth_frame"]["schema"],
            "phanthy.sensor.camera_depth_frame.v1",
        )
        self.assertEqual(inputs["goal_pose"]["topic"], "/ubuntu/navigation/goal_pose")
        self.assertEqual(inputs["goal_pose"]["schema"], "phanthy.navigation.goal.v1")
        self.assertFalse(inputs["goal_pose"]["monitor"])
        self.assertNotIn("livo_odom", {item["port"] for item in tool["topic_in"]})
        self.assertEqual(
            [item["port"] for item in tool["topic_out"]],
            [
                "map_view",
                "motion_sequence",
                "control_velocity",
            ],
        )
        control = tool["topic_out"][2]
        self.assertEqual(control["topic"], "/ubuntu/navigation/control_velocity")
        self.assertEqual(control["format"], "control/velocity")
        self.assertEqual(control["ros_type"], "std_msgs/msg/String")
        self.assertEqual(control["schema"], "motus.control/1")
        other = navigation_tool_definition("/robot_alpha/")
        other_control = next(item for item in other["topic_out"]
                             if item["port"] == "control_velocity")
        self.assertEqual(other_control["topic"], "/robot_alpha/navigation/control_velocity")
        self.assertEqual(tool["topic_out"][1]["topic"], "/ubuntu/navigation/motion_sequence")
        auxiliary = {item["port"]: item for item in tool["topic_aux"]}
        self.assertEqual(set(auxiliary), {"plan", "livo_odom"})
        self.assertEqual(auxiliary["plan"]["topic"], "/plan")
        self.assertEqual(auxiliary["plan"]["format"], "sensor/path")
        self.assertEqual(auxiliary["livo_odom"]["topic"], "/ubuntu/navigation/odom")
        self.assertEqual(auxiliary["livo_odom"]["format"], "sensor/odometry")
        self.assertNotIn("x-topic-actions", tool)
        self.assertEqual(tool["inputSchema"].get("x-topic-actions", []), [])
        self.assertEqual(inputs["goal_pose"]["ros_type"], "std_msgs/msg/String")
        self.assertFalse(inputs["goal_pose"]["required"])
        self.assertTrue(
            {
                "livo_odom",
                "registered_cloud",
                "obstacle_map",
                "static_map",
            }.isdisjoint({item["port"] for item in tool["topic_out"]})
        )
        actions = tool["inputSchema"]["properties"]["action"]["enum"]
        for action in (
            "start_mapping",
            "load_map",
            "navigate_to_pose",
            "capture",
            "navigate",
        ):
            self.assertIn(action, actions)
        self.assertNotIn("x-execution-control", tool)

    def test_motion_sequence_uses_consistent_public_and_runtime_contract(self):
        from plugins.navigation.planning.contract import nav2_tool_definition

        for namespace in ("ubuntu", "robot"):
            with self.subTest(namespace=namespace):
                tool = navigation_tool_definition(namespace)
                motion = tool["topic_out"][1]
                proposal = next(
                    item for item in nav2_tool_definition(namespace)["topic_out"]
                    if item["port"] == "motion_sequence"
                )
                self.assertEqual(motion, proposal)
                self.assertEqual(
                    motion["topic"], f"/{namespace}/navigation/motion_sequence"
                )
                self.assertEqual(motion["schema"], "phanthy.navigation.motion_sequence.v1")

    def test_unified_action_schema_only_exposes_supported_fields(self):
        tool = navigation_tool_definition("ubuntu")
        properties = tool["inputSchema"]["properties"]
        action_params = tool["inputSchema"]["x-action-params"]
        config_fields = list(tool["configSchema"]["properties"])

        self.assertEqual(action_params["config"]["params"], config_fields)
        self.assertEqual(tool["configSchema"]["required"], [])
        self.assertIn("vlm_api_key", properties)
        self.assertEqual(properties["vlm_api_key"]["format"], "password")
        self.assertTrue(properties["vlm_api_key"]["x-sensitive"])
        self.assertIn("planning_request_timeout_sec", properties)
        self.assertIn("obstacle_min_height_m", properties)
        self.assertIn("obstacle_max_height_m", properties)
        self.assertTrue(properties["map_view_enabled"]["default"])
        self.assertTrue(properties["fault_capture_enabled"]["default"])
        self.assertEqual(properties["lidar_qos_depth"]["default"], 2)
        self.assertEqual(properties["imu_qos_depth"]["default"], 400)
        self.assertIn("rotate_speed_rps", properties)
        self.assertEqual(
            tool["configSchema"]["properties"]["rotate_speed_rps"],
            {
                "type": "number",
                "minimum": 0.3,
                "maximum": 2.0,
                "default": 0.3,
                "description": "Fixed magnitude of every nonzero yaw proposal",
            },
        )
        self.assertNotIn("min_yaw_rps", properties)
        self.assertNotIn("max_yaw_rps", properties)
        self.assertNotIn("backend", properties)
        self.assertNotIn("request_timeout_sec", properties)
        self.assertNotIn("base_url", properties)
        for action in action_params.values():
            self.assertTrue(set(action["params"]) <= set(properties))

    def test_long_running_navigation_declares_scoped_acp_completion(self):
        tool = navigation_tool_definition("ubuntu")
        completion = tool["inputSchema"]["x-completion"]
        self.assertEqual(completion["actions"], ["navigate_to_pose", "navigate"])
        self.assertEqual(
            set(completion["passthrough_actions"]),
            {"wait_navigation_done", "pause_nav", "resume_nav", "stop_nav"},
        )
        self.assertEqual(
            tool["inputSchema"]["x-hooks"]["on_interrupt_navigation"],
            {"action": "stop_nav"},
        )
        actions = tool["inputSchema"]["properties"]["action"]["enum"]
        self.assertIn("wait_navigation_done", actions)
        self.assertIn("stop_nav", actions)

    def test_formal_service_has_exactly_one_container(self):
        service = (ACTUCORE_ROOT / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        main = (ACTUCORE_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertEqual(service.count("container_name:"), 1)
        self.assertNotIn("depends_on:", service)
        self.assertNotIn("embodied-perception-fast-livo2", service)
        self.assertNotIn("embodied-perception-nav2", service)
        self.assertIn('plugins_cfg.get("controlled_semantic_spatial"', main)
        self.assertNotIn('plugins_cfg.get("fast_livo2"', main)
        self.assertNotIn('plugins_cfg.get("nav2"', main)
        self.assertNotIn('plugins_cfg.get("vln"', main)
        self.assertIn("plugins.controlled_semantic_spatial.namespace must be a string", main)
        self.assertIn("requires namespace=ubuntu", main)
        self.assertNotIn("socket.gethostname()", main)

    def test_navigation_runtime_uses_host_dds_isolation(self):
        service = (ACTUCORE_ROOT / "deploy" / "service.yml").read_text()
        self.assertIn("FASTRTPS_DEFAULT_PROFILES_FILE=/opt/phanthy-motus/dds-local.xml", service)
        self.assertIn("/opt/phanthy-motus/dds-local.xml:/opt/phanthy-motus/dds-local.xml:ro", service)
        for manifest in (ACTUCORE_ROOT / "Dockerfile.jetson", ACTUCORE_ROOT / "deploy" / "service.yml"):
            self.assertNotIn("FASTDDS_BUILTIN_TRANSPORTS=", manifest.read_text())

        for source_lock in (
            "nav2-source.lock",
            "fast_livo2-source.lock",
        ):
            content = (
                ACTUCORE_ROOT
                / "plugins"
                / "navigation"
                / "runtime"
                / source_lock
            ).read_text(encoding="utf-8")
            self.assertNotIn("FASTDDS_BUILTIN_TRANSPORTS", content)

    def test_actucore_image_stays_free_of_perception_model_dependencies(self):
        """卡片只用标准库 + ROS 消息包，镜像不该被拖成第二个 perception。

        导航栈的语义航点是走 HTTP 调远端 VLM 的，本地既不跑 CLIP 也不跑 YOLO；
        torch / opencv-python / ultralytics 属于 perception 的 VOP 卡片。
        以前这两张卡片挤在同一个镜像里，所以旧断言反过来要求"别丢 VOP 依赖"。
        """
        dockerfile = (ACTUCORE_ROOT / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )
        for perception_only in (
            "ultralytics",
            "CLIP.git",
            "ViT-B-32.pt",
            "YOLO_CONFIG_DIR",
            "opencv-python",
            "sherpa-onnx",
            "silero-vad",
            "torchaudio",
        ):
            self.assertNotIn(perception_only, dockerfile)
        self.assertNotIn("python3-pip", dockerfile)
        self.assertIn('if [ "${JP_VERSION}" = 61 ]; then', dockerfile)
        self.assertIn('pip3 install --no-cache-dir -i "${PYPI_MIRROR}" pyyaml requests', dockerfile)
        self.assertIn("import numpy, rosbag2_py, yaml, em", dockerfile)
        self.assertIn("COPY deploy/ros-base/audio_msgs/", dockerfile)
        self.assertIn("COPY actucore/main.py", dockerfile)
        self.assertIn(
            "COPY perception/utils/logsafe.py /work/logsafe.py",
            dockerfile,
        )
        self.assertIn("COPY actucore/deploy/     /deploy/", dockerfile)
        self.assertIn("echo /work", dockerfile)
        self.assertIn('python3 -c "import logsafe"', dockerfile)

    def test_every_python_process_installs_the_atomic_log_writer(self):
        sources = [
            ACTUCORE_ROOT / "main.py",
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "runtime"
            / "fast_livo2"
            / "fast_livo2"
            / "adapter_node.py",
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "runtime"
            / "fast_livo2"
            / "fast_livo2"
            / "runtime_supervisor.py",
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "runtime"
            / "nav2"
            / "nav2"
            / "planner_command_node.py",
        ]
        for source in sources:
            content = source.read_text(encoding="utf-8")
            self.assertIn("import logsafe", content)
            self.assertIn("logsafe.install()", content)
            self.assertLess(
                content.index("logsafe.install()"), content.index("import rclpy")
            )
        self.assertFalse((ACTUCORE_ROOT / "utils" / "logsafe.py").exists())

    def test_navigation_base_owns_locked_third_party_builds(self):
        base_dockerfile = (
            ACTUCORE_ROOT / "Dockerfile.base"
        ).read_text(
            encoding="utf-8"
        )
        dockerfile = (ACTUCORE_ROOT / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )
        build_script = (
            ACTUCORE_ROOT.parent / "deploy" / "build_actucore.sh"
        ).read_text(encoding="utf-8")
        source_locks = [
            (
                ACTUCORE_ROOT
                / "plugins"
                / "navigation"
                / "runtime"
                / name
            ).read_text(encoding="utf-8")
            for name in ("fast_livo2-source.lock", "nav2-source.lock")
        ]

        self.assertRegex(
            base_dockerfile,
            r"ARG ACTUCORE_NAVIGATION_PARENT_IMAGE="
            r"bj-warehouse\.tencentcloudcr\.com/phanthy-motus/"
            r"jetson-base:jp511-torch@sha256:[0-9a-f]{64}",
        )
        self.assertIn("FROM ${ACTUCORE_NAVIGATION_PARENT_IMAGE}", base_dockerfile)
        self.assertIn(
            "FROM ${ACTUCORE_NAVIGATION_PARENT_IMAGE} AS navigation_builder",
            base_dockerfile,
        )
        runtime_stage = base_dockerfile.split(
            "FROM ${ACTUCORE_NAVIGATION_PARENT_IMAGE} AS navigation_runtime", 1
        )[1]
        for workspace in ("ros_deps_ws", "fast_livo_ws", "nav2_ws"):
            self.assertIn(
                f"COPY --from=navigation_builder /opt/{workspace}/install/",
                runtime_stage,
            )
            for directory in ("src", "build", "log"):
                self.assertNotIn(f"/opt/{workspace}/{directory}/", runtime_stage)
        self.assertIn("command -v colcon", base_dockerfile)
        self.assertIn("int(em.__version__.split('.')[0]) < 4", base_dockerfile)
        self.assertNotIn("AllowInsecureRepositories", base_dockerfile)
        self.assertNotIn("--allow-unauthenticated", base_dockerfile)
        self.assertNotIn(
            "rm -f /etc/apt/sources.list.d/*.list", base_dockerfile
        )
        self.assertIn("unauthenticated APT source is not allowed", base_dockerfile)
        self.assertIn("ARG GIT_MIRROR_PREFIX=", base_dockerfile)
        self.assertNotIn(
            "ARG GIT_MIRROR_PREFIX=https://ghfast.top/", base_dockerfile
        )
        self.assertIn(
            'GIT_MIRROR_PREFIX="${GIT_MIRROR_PREFIX:-}"', build_script
        )
        self.assertLess(
            base_dockerfile.index("git config --global"),
            base_dockerfile.index("WORKDIR /opt/ros_deps_ws/src"),
        )
        for dependency in ("MCAP", "LZ4"):
            self.assertIn(
                f"-DFETCHCONTENT_SOURCE_DIR_{dependency}=", base_dockerfile
            )
            self.assertIn(
                f"/opt/ros_deps_ws/vendor/{dependency.lower()}",
                base_dockerfile,
            )
        self.assertIn("GIT_LFS_SKIP_SMUDGE=1 git checkout", base_dockerfile)
        self.assertGreater(
            base_dockerfile.rindex("--remove-section"),
            base_dockerfile.index("WORKDIR /opt/nav2_ws"),
        )
        for source_lock in source_locks:
            self.assertNotIn("GIT_MIRROR_PREFIX=", source_lock)
            self.assertNotIn("APT_UBUNTU_MIRROR=", source_lock)
            for assignment in source_lock.splitlines():
                if not assignment or assignment.startswith("#"):
                    continue
                variable, separator, _ = assignment.partition("=")
                self.assertEqual(separator, "=")
                self.assertIn(f'"{variable}=${{{variable}}}"', build_script)
        self.assertIn("jetson-base:jp61-torch@sha256:", build_script)
        self.assertNotIn("FROM ${FAST_LIVO2_BASE_IMAGE}", base_dockerfile)
        self.assertNotIn("FAST_LIVO2_BASE_IMAGE=", build_script)
        self.assertIn("FROM ${BASE_IMAGE}", dockerfile)
        self.assertEqual(sum(line.startswith("FROM ") for line in dockerfile.splitlines()), 1)
        self.assertNotIn("--symlink-install", dockerfile)
        for workspace in ("ros_deps_ws", "fast_livo_ws", "nav2_ws"):
            self.assertIn(f"source /opt/{workspace}/install/setup.bash", dockerfile)
        for directory in ("src", "build", "log"):
            self.assertIn(f"test ! -e /ros_ws/{directory}", dockerfile)
        self.assertIn("rm -rf /work/plugins/navigation/runtime", dockerfile)
        self.assertIn("test ! -e /work/plugins/navigation/runtime", dockerfile)
        self.assertIn("COPY actucore/plugins/    /work/plugins/", dockerfile)
        self.assertIn("actucore-navigation-base@sha256:", dockerfile)
        self.assertNotIn("missing-digest", dockerfile)
        self.assertIn("--base", build_script)
        self.assertIn("ACTUCORE_NAVIGATION_BASE_IMAGE override", build_script)
        self.assertIn('"BASE_IMAGE=${ACTUCORE_NAVIGATION_BASE_IMAGE}"', build_script)
        self.assertIn(
            "actucore-navigation-base@sha256:14550b74bfce5c0ede5908e6081da375148b67695a134f102f025c535f702e4b",
            build_script,
        )
        self.assertIn(
            "build the navigation base on native ARM64, not through QEMU",
            build_script,
        )
        self.assertNotIn("git fetch", dockerfile)
        self.assertNotIn("FAST_LIVO2_REPO", dockerfile)
        self.assertNotIn("NAVIGATION2_REPO", dockerfile)
        self.assertIn("fast_livo2 segmented_controller nav2", dockerfile)
        self.assertIn("ros2 pkg prefix nav2", dockerfile)
        self.assertIn("plugin_lib_names", dockerfile)
        self.assertIn("ctypes.CDLL", dockerfile)
        self.assertIn("NAV2_BT_PLUGIN_LIBS=PASS", dockerfile)
        # base 是 Focal，humble 没有对应的 Debian 包 → Nav2 也只能源码编
        self.assertNotIn("ros-humble-navigation2", base_dockerfile)
        self.assertIn("--packages-select", base_dockerfile)
        # 找到 nav2 那次 colcon build 的选择列表
        select_blocks = [
            block.split("--cmake-args")[0]
            for block in base_dockerfile.split("--packages-select")[1:]
        ]
        nav2_select = next(b for b in select_blocks if "nav2_bringup" in b)
        self.assertIn("navigation2", nav2_select)
        for unused in (
            "nav2_dwb_controller",
            "dwb_core",
            "dwb_critics",
            "dwb_msgs",
            "dwb_plugins",
            "nav2_rotation_shim_controller",
            "nav2_map_server",
        ):
            self.assertNotIn(unused, nav2_select)
        for ignored_dwb in (
            "nav2_dwb_controller/dwb_core",
            "nav2_dwb_controller/dwb_critics",
            "nav2_dwb_controller/dwb_msgs",
            "nav2_dwb_controller/dwb_plugins",
            "nav2_dwb_controller/nav2_dwb_controller",
        ):
            self.assertIn(ignored_dwb, base_dockerfile)
        self.assertIn('test -d "${excluded}"', base_dockerfile)
        # 刻意不编的那批：会把 ompl / ceres / xtensor / Qt5 拖进来。
        # 只靠 --packages-select 挡不住 colcon 顺 test_depend 去要它们，
        # 所以镜像里还会给这些目录放 COLCON_IGNORE。
        for excluded in (
            "nav2_smac_planner",
            "nav2_mppi_controller",
            "nav2_constrained_smoother",
            "nav2_rviz_plugins",
            "nav2_regulated_pure_pursuit_controller",
        ):
            self.assertNotIn(excluded, nav2_select)
            self.assertIn(excluded, base_dockerfile)
        self.assertIn('touch "${excluded}/COLCON_IGNORE"', base_dockerfile)
        for variable in (
            "GIT_MIRROR_PREFIX",
            "FAST_LIVO2_REPO",
            "FAST_LIVO2_RUNTIME_PATCH_SHA256",
            "FAST_LIVO2_PCD_SAVE_PATCH_SHA256",
            "FAST_LIVO2_PCD_FLUSH_PATCH_SHA256",
            "RPG_VIKIT_REPO",
            "SOPHUS_REPO",
            "NAVIGATION2_REPO",
            "NAVIGATION2_COMMIT",
            "NAVIGATION2_RUNTIME_PATCH_SHA256",
            "BEHAVIORTREE_CPP_COMMIT",
        ):
            self.assertIn(f'"{variable}=${{{variable}}}"', build_script)
        for removed in (
            "LIVOX_ROS_DRIVER2_REPO",
            "LIVOX_ROS_DRIVER2_COMMIT",
            "LIVOX_SDK2_REPO",
            "LIVOX_SDK2_COMMIT",
        ):
            self.assertNotIn(removed, build_script)
        self.assertIn("ACTUCORE_BUILD_DURATION_SEC=", build_script)
        self.assertIn('"name":"controlled_semantic_spatial"', build_script)
        self.assertIn('"name":"vla"', build_script)
        self.assertIn('CARDS_JSON', build_script)


class NavigationPluginTest(unittest.TestCase):
    def make_plugin(self, *, runtime=None, mapping=None, planning=None, semantic=None):
        return NavigationPlugin(
            {"namespace": "ubuntu"},
            "ubuntu",
            executor=None,
            runtime=runtime or FakeRuntime(),
            mapping_plugin=mapping or FakeComponent("mapping"),
            planning_plugin=planning or FakeComponent("planning"),
            semantic_plugin=semantic or FakeComponent("semantic"),
        )

    def test_mainline_start_and_info_with_camera_metadata(self):
        plugin = self.make_plugin()
        topics = [
            "/ubuntu/navigation/lidar", "/ubuntu/navigation/imu",
            "/ubuntu/camera/rgb_frame", "/ubuntu/camera/depth_frame",
        ]
        start = {
            "action": "start", "instance_id": "card-1",
            "input_topics": topics, "input_topic": topics[0],
            "camera_info": {
                topics[2]: {"schema": "motus.camera/1", "topic": topics[2],
                            "id": "vendor/model/rgb", "source": "unknown"},
                topics[3]: {"schema": "motus.camera/1", "topic": topics[3],
                            "id": "vendor/model/depth", "source": "unknown"},
            },
        }
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        info = plugin.dispatch(plugin.PREFIX, {
            "action": "info", "instance_id": "card-1",
            "input_topics": topics, "input_topic": topics[0],
        })
        self.assertEqual(info["state"], "ready")

    def test_start_negotiates_descriptor_and_preserves_it_for_runtime(self):
        runtime = FakeRuntime()
        plugin = self.make_plugin(runtime=runtime)
        descriptor = _control_descriptor()
        request = {"action": "start", "input_bindings": _external_bindings(),
                   "control_interface": descriptor}
        started = plugin.dispatch(plugin.PREFIX, request)
        self.assertEqual(started["state"], "ready")
        self.assertFalse(started["shadow_only"])
        self.assertTrue(started["control_output_can_drive_hardware"])
        self.assertEqual(runtime.start_kwargs["control_interface"], descriptor)
        self.assertIsNot(runtime.start_kwargs["control_interface"], descriptor)
        info = plugin.dispatch(plugin.PREFIX, {"action": "info"})
        self.assertEqual(info["state"], "ready")
        output = next(item for item in info["topic_out"] if item["port"] == "control_velocity")
        self.assertEqual(output["topic"], "/ubuntu/navigation/control_velocity")
        self.assertEqual(output["format"], "control/velocity")
        self.assertEqual(output["schema"], "motus.control/1")
        self.assertTrue(plugin.dispatch(plugin.PREFIX, request)["already_started"])
        changed = _control_descriptor(rate={"max_hz": 20, "expected_hz": 5})
        result = plugin.dispatch(plugin.PREFIX, {**request, "control_interface": changed})
        self.assertEqual(result["error_code"], "navigation_control_interface_conflict")
        result = plugin.dispatch(plugin.PREFIX, {**request, "control_interface": {}})
        self.assertEqual(result["error_code"], "navigation_control_interface_conflict")
        self.assertEqual(runtime.start_kwargs["control_interface"], descriptor)
        self.assertEqual(plugin.dispatch(plugin.PREFIX, {"action": "stop"})["state"], "idle")
        self.assertEqual(plugin.dispatch(plugin.PREFIX, {
            **request, "control_interface": changed,
        })["state"], "ready")
        self.assertEqual(runtime.start_kwargs["control_interface"], changed)

    def test_start_accepts_navi_minimal_descriptor_and_short_watchdog(self):
        for descriptor in (
            {"control_interface": "motus.control/1", "mode": "twist"},
            {"control_interface": "motus.control/1", "mode": "twist", "dof": 6,
             "rate": {"expected_hz": 8, "watchdog_ms": 50}},
        ):
            with self.subTest(descriptor=descriptor):
                runtime = FakeRuntime()
                plugin = self.make_plugin(runtime=runtime)
                result = plugin.dispatch(plugin.PREFIX, {
                    "action": "start", "input_bindings": _external_bindings(),
                    "control_interface": descriptor,
                })
                self.assertEqual(result["state"], "ready")
                self.assertEqual(runtime.start_kwargs["control_interface"], descriptor)

    def test_start_rejects_incompatible_control_before_runtime(self):
        runtime = FakeRuntime()
        mapping = FakeComponent("mapping")
        plugin = self.make_plugin(runtime=runtime, mapping=mapping)
        cases = [
            ({"schema": "motus.control/1"}, "incompatible_control_interface"),
            (_control_descriptor(mode="joint_position"), "incompatible_control_interface"),
            (_control_descriptor(mode=""), "incompatible_control_interface"),
            (_control_descriptor(dof=4), "incompatible_control_interface"),
            (_control_descriptor(rate={"max_hz": 5}), "incompatible_control_interface"),
            (None, "invalid_control_interface"), ([], "invalid_control_interface"),
            ("descriptor", "invalid_control_interface"),
            (_control_descriptor(dof="invalid"), "invalid_control_interface"),
            (_control_descriptor(rate="invalid"), "invalid_control_interface"),
        ]
        for descriptor, code in cases:
            with self.subTest(descriptor=descriptor):
                result = plugin.dispatch(plugin.PREFIX, {
                    "action": "start", "input_bindings": _external_bindings(),
                    "control_interface": descriptor,
                })
                self.assertEqual(result["error_code"], code)
                self.assertFalse(runtime.started)
                self.assertEqual(mapping.calls, [])
        result = plugin.dispatch(plugin.PREFIX, {
            "action": "start", "input_bindings": _external_bindings(),
            "control_interface": {},
        })
        self.assertEqual(result["state"], "ready")
        self.assertEqual(runtime.start_kwargs["control_interface"], {})

    def test_all_operational_actions_reject_idle_before_component_dispatch(self):
        runtime = FakeRuntime()
        components = [FakeComponent(name) for name in ("mapping", "planning", "semantic")]
        plugin = self.make_plugin(runtime=runtime, mapping=components[0],
                                  planning=components[1], semantic=components[2])
        before = [list(component.calls) for component in components]
        for action in set(NAVIGATION_ACTIONS) - {"info", "config", "start", "stop"}:
            with self.subTest(action=action):
                result = plugin.dispatch("controlled_semantic_spatial", {"action": action})
                self.assertEqual(result["error_code"], "canvas_not_started")
                self.assertEqual([component.calls for component in components], before)
                self.assertFalse(runtime.started)
                self.assertIsNone(runtime.start_kwargs)

    def test_old_goal_ticket_rejected_after_stop_and_restart(self):
        """A request already received here cannot use the restarted planner."""
        class PausingRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()

            def info(self):
                if threading.current_thread().name == "old-navigation-request":
                    self.entered.set()
                    if not self.release.wait(3):
                        raise RuntimeError("test did not release old request")
                return super().info()

        runtime = PausingRuntime()
        planning = FakeComponent("planning")
        plugin = self.make_plugin(runtime=runtime, planning=planning)
        start = {
            "action": "start",
            "instance_id": "same-card",
            "input_bindings": _external_bindings(),
        }
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        result = {}
        thread = threading.Thread(
            name="old-navigation-request",
            target=lambda: result.update(plugin.dispatch(plugin.PREFIX, {
                "action": "navigate_to_pose", "instance_id": "same-card",
                "x": 1, "y": 2, "yaw": 0,
            })),
        )
        thread.start()
        try:
            self.assertTrue(runtime.entered.wait(2))
            self.assertEqual(plugin.dispatch(plugin.PREFIX, {"action": "stop"})["state"], "idle")
            self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        finally:
            runtime.release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result.get("error_code"), "navigation_stopping", result)
        self.assertFalse(any(args.get("action") == "navigate_to_pose"
                             for _, args in planning.calls))

    def test_mapping_inflight_blocks_teardown_and_stale_dispatch(self):
        class PausingRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()
            def info(self):
                if threading.current_thread().name == "old-mapping-request":
                    self.entered.set()
                    if not self.release.wait(3):
                        raise RuntimeError("test did not release mapping")
                return super().info()

        runtime = PausingRuntime()
        mapping = FakeComponent("mapping")
        planning = FakeComponent("planning")
        plugin = self.make_plugin(runtime=runtime, mapping=mapping, planning=planning)
        start = {"action": "start", "input_bindings": _external_bindings()}
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        result = {}
        thread = threading.Thread(name="old-mapping-request", target=lambda: result.update(
            plugin.dispatch(plugin.PREFIX, {"action": "start_mapping", "map_name": "room"})
        ))
        thread.start()
        try:
            self.assertTrue(runtime.entered.wait(2))
            self.assertEqual(plugin.stop()["error_code"], "navigation_stop_pending")
            self.assertTrue(runtime.started)
            self.assertFalse(any(args.get("action") == "stop" for _, args in mapping.calls))
            self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["error_code"],
                             "navigation_start_cancelled")
        finally:
            runtime.release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result.get("error_code"), "navigation_stopping")
        self.assertFalse(any(args.get("action") == "start_mapping" for _, args in mapping.calls))
        self.assertEqual(plugin.stop()["state"], "idle")
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")

    def test_inflight_planner_requires_cancel_then_later_confirmed_stop(self):
        entered = threading.Event()
        release = threading.Event()

        class Planning(FakeComponent):
            def dispatch(self, prefix, args):
                if args.get("action") == "navigate_to_pose":
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("test did not release planner")
                return super().dispatch(prefix, args)

            def stop_active_only(self):
                self.calls.append(("nav2", {"action": "stop_active_only"}))
                return {"status": "stopped", "terminal_confirmed": True}

        planning = Planning("planning")
        runtime = FakeRuntime()
        plugin = self.make_plugin(planning=planning, runtime=runtime)
        start = {"action": "start", "input_bindings": _external_bindings()}
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        result = {}
        thread = threading.Thread(target=lambda: result.update(plugin.dispatch(
            plugin.PREFIX, {"action": "navigate_to_pose", "x": 1, "y": 2, "yaw": 0}
        )))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            stop = plugin.dispatch(plugin.PREFIX, {"action": "stop"})
            self.assertEqual(stop["error_code"], "navigation_stop_pending")
            self.assertTrue(planning.started)
            self.assertTrue(runtime.started)
            self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["error_code"],
                             "navigation_start_cancelled")
            self.assertFalse(any(args.get("action") == "stop" for _, args in planning.calls))
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result.get("status"), "navigate_to_pose")
        self.assertEqual(plugin.dispatch(plugin.PREFIX, {"action": "stop"})["state"], "idle")
        self.assertEqual([args["action"] for _, args in planning.calls
                          if args.get("action") in {"stop_active_only", "navigate_to_pose", "stop"}],
                         ["stop_active_only", "navigate_to_pose", "stop"])
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")

    def test_concurrent_stops_keep_admission_closed_until_last_result(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        release_second = threading.Event()
        second_registered = threading.Event()

        class ObservedLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.pending_at_first_release = None
            def __enter__(self):
                if threading.current_thread().name == "second-stop":
                    second_registered.set()
                self.lock.acquire()
                return self
            def __exit__(self, *_):
                if threading.current_thread().name == "first-stop":
                    self.pending_at_first_release = plugin._pending_stops
                self.lock.release()

        class Planning(FakeComponent):
            def __init__(self):
                super().__init__("planning")
                self.stops = 0

            def dispatch(self, prefix, args):
                if args.get("action") == "stop":
                    self.stops += 1
                    if self.stops == 1:
                        first_entered.set()
                        if not release_first.wait(3):
                            raise RuntimeError("test did not release first stop")
                    else:
                        second_entered.set()
                        if not release_second.wait(3):
                            raise RuntimeError("test did not release second stop")
                        return {"status": "error", "error_code": "pending"}
                return super().dispatch(prefix, args)

        planning = Planning()
        plugin = self.make_plugin(planning=planning)
        start = {"action": "start", "input_bindings": _external_bindings()}
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        plugin._transition_lock = ObservedLock()
        results = {}
        first = threading.Thread(name="first-stop", target=lambda: results.update(first=plugin.stop()))
        second = threading.Thread(name="second-stop", target=lambda: results.update(second=plugin.stop()))
        first.start()
        try:
            self.assertTrue(first_entered.wait(2))
            second.start()
            self.assertTrue(second_registered.wait(2))
            with plugin._lock:
                self.assertEqual(plugin._pending_stops, 2)
            release_first.set()
            self.assertTrue(second_entered.wait(2))
            self.assertEqual(plugin._transition_lock.pending_at_first_release, 1)
            self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["error_code"],
                             "navigation_start_cancelled")
        finally:
            release_first.set()
            release_second.set()
            first.join(3)
            if second.ident is not None:
                second.join(3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(results["second"]["error_code"], "navigation_stop_pending")
        self.assertTrue(plugin._stop_requested.is_set())
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["error_code"],
                         "navigation_start_cancelled")

    def test_inflight_control_exception_remains_pending_until_retry(self):
        entered = threading.Event()
        release = threading.Event()

        class Planning(FakeComponent):
            def dispatch(self, prefix, args):
                if args.get("action") == "resume_nav":
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("test did not release control")
                    raise RuntimeError("control failed")
                return super().dispatch(prefix, args)

            def stop_active_only(self):
                raise RuntimeError("cancel failed")

        planning = Planning("planning")
        plugin = self.make_plugin(planning=planning)
        start = {"action": "start", "input_bindings": _external_bindings()}
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        errors = []
        def control():
            try:
                plugin.dispatch(plugin.PREFIX, {"action": "resume_nav"})
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=control)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            result = plugin.dispatch(plugin.PREFIX, {"action": "stop"})
            self.assertEqual(result["error_code"], "navigation_stop_pending")
            self.assertEqual(result["component_results"]["planning"]["error_code"],
                             "component_stop_failed")
            self.assertTrue(planning.started)
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["control failed"])
        self.assertEqual(plugin.dispatch(plugin.PREFIX, {"action": "stop"})["state"], "idle")

    def test_vlm_inflight_does_not_delay_planning_cancel_or_submit_old_goal(self):
        """Uses the real semantic processor with fake VLM, bridge and planner."""
        from tests.test_vln import FakeBridgeFactory, FakeClient, _plugin

        entered = threading.Event()
        release = threading.Event()
        planning_stopped = threading.Event()

        class Planning(FakeComponent):
            def stop_active_only(self):
                planning_stopped.set()
                self.calls.append(("nav2", {"action": "stop_active_only"}))
                return {"status": "stopped", "terminal_confirmed": True}

        client = FakeClient([
            {"scene": "office", "objects": ["desk"], "description": "desk"},
            None,
        ])
        planning = Planning("planning")
        semantic = _plugin(client, FakeBridgeFactory())
        plugin = self.make_plugin(planning=planning, semantic=semantic)
        semantic._processor._goal_handler = plugin._handle_semantic_goal
        start = {
            "action": "start",
            "input_bindings": _external_bindings() + [{"port": "rgb", "topic": "/camera/rgb"}],
        }
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        captured = plugin.dispatch(plugin.PREFIX, {"action": "capture"})
        self.assertTrue(captured["ok"])
        point_id = captured["waypoint"]["point_id"]

        def slow_match(_):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test did not release VLM")
            return {"point_id": point_id, "confidence": 0.9, "reason": "match"}

        client.responses[0] = slow_match
        results = {}
        navigate = threading.Thread(target=lambda: results.update(navigate=plugin.dispatch(
            plugin.PREFIX, {"action": "navigate", "query": "desk"}
        )))
        stop = threading.Thread(target=lambda: results.update(stop=plugin.dispatch(
            plugin.PREFIX, {"action": "stop"}
        )))
        navigate.start()
        try:
            self.assertTrue(entered.wait(2))
            stop.start()
            self.assertTrue(plugin._stop_requested.wait(2))
            planning_stop_before_release = planning_stopped.wait(0.1)
        finally:
            release.set()
            navigate.join(3)
            if stop.ident is not None:
                stop.join(3)
        self.assertFalse(navigate.is_alive())
        self.assertFalse(stop.is_alive())
        self.assertEqual(results.get("navigate", {}).get("error_code"), "navigation_stopping", results)
        self.assertEqual(results.get("stop", {}).get("error_code"), "navigation_stop_pending", results)
        self.assertTrue(planning_stop_before_release, results)
        self.assertFalse(any(args.get("action") == "navigate_to_pose"
                             for _, args in planning.calls))
        self.assertEqual(plugin.dispatch(plugin.PREFIX, {"action": "stop"})["state"], "idle")

    def test_lidar_imu_only_cannot_auto_start_real_semantic_processor(self):
        from plugins.navigation.semantic.plugin import VisionAndLanguageNavigationPlugin

        for action in ("capture", "navigate"):
            with self.subTest(action=action):
                client = mock.Mock(configured=True, timeout_sec=1.0)
                bridge_factory = mock.Mock(side_effect=AssertionError("unexpected RGB subscription"))
                semantic = VisionAndLanguageNavigationPlugin(
                    {"auto_start_on_action": True}, "ubuntu", None,
                    client=client, bridge_factory=bridge_factory,
                )
                planning = FakeComponent("planning")
                plugin = self.make_plugin(semantic=semantic, planning=planning)
                started = plugin.dispatch("controlled_semantic_spatial", {
                    "action": "start", "input_bindings": _external_bindings(),
                })
                self.assertEqual(started["state"], "ready")
                result = plugin.dispatch("controlled_semantic_spatial", {
                    "action": action, "query": "desk",
                })
                self.assertEqual(result["error_code"], "rgb_not_connected")
                bridge_factory.assert_not_called()
                client.complete_json.assert_not_called()
                coordinate = plugin.dispatch("controlled_semantic_spatial", {
                    "action": "navigate_to_pose", "x": 1, "y": 2, "yaw": 0,
                })
                self.assertEqual(coordinate["component"], "planning")

    def test_only_controlled_semantic_spatial_is_a_public_dispatch_name(self):
        plugin = self.make_plugin()

        self.assertEqual(plugin.PREFIX, "controlled_semantic_spatial")
        self.assertIsNone(plugin.dispatch("navigation", {"action": "info"}))
        self.assertEqual(
            plugin.dispatch("controlled_semantic_spatial", {"action": "info"})[
                "name"
            ],
            "controlled_semantic_spatial",
        )

    def test_single_start_owns_runtime_and_internal_wiring(self):
        runtime = FakeRuntime()
        mapping = FakeComponent("mapping")
        planning = FakeComponent("planning")
        semantic = FakeComponent("semantic")
        plugin = self.make_plugin(
            runtime=runtime,
            mapping=mapping,
            planning=planning,
            semantic=semantic,
        )

        result = plugin.dispatch(
            "controlled_semantic_spatial",
            {
                "action": "start",
                "instance_id": "canvas-navigation",
                "input_bindings": _external_bindings(),
            },
        )

        self.assertEqual(result["state"], "ready")
        self.assertTrue(runtime.started)
        self.assertEqual(runtime.start_kwargs["namespace"], "ubuntu")
        self.assertEqual(
            runtime.start_kwargs["input_topics"]["lidar"],
            "/ubuntu/navigation/lidar",
        )
        mapping_start = next(args for _, args in mapping.calls if args["action"] == "start")
        self.assertEqual(
            {item["port"] for item in mapping_start["input_bindings"]},
            {"lidar", "imu"},
        )
        planning_start = next(args for _, args in planning.calls if args["action"] == "start")
        planning_topics = {
            item["port"]: item["topic"] for item in planning_start["input_bindings"]
        }
        self.assertEqual(planning_topics["livo_odom"], "/ubuntu/navigation/odom")
        self.assertEqual(
            planning_topics["registered_cloud"],
            "/ubuntu/navigation/cloud_registered",
        )
        self.assertEqual(
            planning_topics["static_map"],
            "/ubuntu/navigation/static_map",
        )
        self.assertFalse(any(args["action"] == "start" for _, args in semantic.calls))
        stopped = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})
        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(runtime.stop_calls, 1)

    def test_custom_external_topics_reach_child_runtime(self):
        runtime = FakeRuntime()
        semantic = FakeComponent("semantic")
        plugin = self.make_plugin(runtime=runtime, semantic=semantic)
        bindings = [
            {"port": item["port"], "topic": item["topic"]}
            for item in navigation_tool_definition("ubuntu")["topic_in"]
            if item["port"] in {"lidar", "imu", "rgb"}
        ]
        custom = {
            "lidar": "/robot/sensors/lidar",
            "imu": "/robot/sensors/imu",
            "rgb": "/robot/camera/rgb",
        }
        for binding in bindings:
            binding["topic"] = custom[binding["port"]]

        result = plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": bindings},
        )

        self.assertEqual(result["state"], "ready")
        self.assertEqual(runtime.start_kwargs["input_topics"], custom)
        semantic_start = next(
            args for _, args in semantic.calls if args["action"] == "start"
        )
        self.assertEqual(
            {item["port"] for item in semantic_start["input_bindings"]},
            {"rgb", "livo_odom", "livo_status"},
        )

    def test_start_rejects_invalid_external_topic_names(self):
        for topic in (
            "   ",
            "relative/lidar",
            "/robot//lidar",
            "/robot/9lidar",
            "/robot/lidar?",
        ):
            with self.subTest(topic=topic):
                runtime = FakeRuntime()
                plugin = self.make_plugin(runtime=runtime)
                bindings = _external_bindings()
                next(item for item in bindings if item["port"] == "lidar")[
                    "topic"
                ] = topic

                result = plugin.dispatch(
                    "controlled_semantic_spatial",
                    {"action": "start", "input_bindings": bindings},
                )

                self.assertEqual(result["error_code"], "invalid_canvas_wiring")
                self.assertFalse(runtime.started)

    def test_collection_reuses_rgb_and_requires_depth_binding(self):
        mapping = CollectionMappingComponent("mapping")
        plugin = self.make_plugin(mapping=mapping)

        missing = plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )
        self.assertEqual(missing["error_code"], "invalid_canvas_wiring")
        self.assertIn("rgb", missing["error"])
        self.assertIn("depth_frame", missing["error"])

        all_bindings = [
            {"port": item["port"], "topic": item["topic"]}
            for item in navigation_tool_definition("ubuntu")["topic_in"]
        ]
        with mock.patch("plugins.navigation.plugin.GoalTopicInput") as goal_input:
            started = plugin.dispatch(
                "controlled_semantic_spatial",
                {"action": "start", "input_bindings": all_bindings},
            )
        self.assertEqual(started["state"], "ready")
        goal_input.assert_called_once()

    def test_runtime_start_failure_rolls_back_before_retry(self):
        runtime = FakeRuntime(fail=True)
        mapping = FakeComponent("mapping")
        plugin = self.make_plugin(runtime=runtime, mapping=mapping)
        request = {"action": "start", "input_bindings": _external_bindings(),
                   "control_interface": _control_descriptor()}
        failed = plugin.dispatch(plugin.PREFIX, request)
        self.assertEqual(failed["error_code"], "navigation_start_failed")
        self.assertEqual(runtime.stop_calls, 1)
        self.assertFalse(runtime.started)
        self.assertFalse(any(args["action"] == "start" for _, args in mapping.calls))
        runtime.fail = False
        minimal = {"control_interface": "motus.control/1", "mode": "twist"}
        self.assertEqual(plugin.dispatch(plugin.PREFIX, {
            **request, "control_interface": minimal,
        })["state"], "ready")
        self.assertEqual(runtime.start_kwargs["control_interface"], minimal)

    def test_start_failure_releases_every_acquired_resource(self):
        runtime = FakeRuntime()
        mapping = FakeComponent("mapping")
        planning = FakeComponent("planning", fail_start=True)
        plugin = self.make_plugin(
            runtime=runtime,
            mapping=mapping,
            planning=planning,
        )

        result = plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings(),
             "control_interface": _control_descriptor()},
        )

        self.assertEqual(result["error_code"], "navigation_start_failed")
        self.assertFalse(runtime.started)
        self.assertEqual(runtime.stop_calls, 1)
        self.assertTrue(any(args["action"] == "stop" for _, args in mapping.calls))
        planning.fail_start = False
        minimal = {"control_interface": "motus.control/1", "mode": "twist"}
        retry = plugin.dispatch(plugin.PREFIX, {
            "action": "start", "input_bindings": _external_bindings(),
            "control_interface": minimal,
        })
        self.assertEqual(retry["state"], "ready")
        self.assertEqual(runtime.start_kwargs["control_interface"], minimal)

    def test_transient_planning_discovery_retries_without_runtime_rollback(self):
        runtime = FakeRuntime()
        mapping = FakeComponent("mapping")
        planning = TransientPlanningComponent()
        plugin = self.make_plugin(
            runtime=runtime,
            mapping=mapping,
            planning=planning,
        )

        result = plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        self.assertEqual(result["state"], "ready")
        self.assertEqual(planning.start_attempts, 2)
        self.assertEqual(runtime.stop_calls, 0)
        self.assertEqual(
            sum(args["action"] == "start" for _, args in mapping.calls),
            1,
        )

    def test_stop_cancels_start_before_transient_planner_retry(self):
        class BlockingTransientPlanner(TransientPlanningComponent):
            def __init__(self):
                super().__init__()
                self.first_entered = threading.Event()
                self.release_first = threading.Event()

            def dispatch(self, prefix, args):
                if args.get("action") == "start" and self.start_attempts == 0:
                    self.first_entered.set()
                    if not self.release_first.wait(timeout=2.0):
                        raise RuntimeError("test did not release planner start")
                return super().dispatch(prefix, args)

        planner = BlockingTransientPlanner()
        runtime = FakeRuntime()
        plugin = self.make_plugin(runtime=runtime, planning=planner)
        start_result, stop_result = {}, {}
        start_thread = threading.Thread(target=lambda: start_result.update(
            plugin.dispatch(plugin.PREFIX, {
                "action": "start", "input_bindings": _external_bindings(),
            })
        ))
        stop_thread = threading.Thread(target=lambda: stop_result.update(
            plugin.dispatch(plugin.PREFIX, {"action": "stop"})
        ))
        start_thread.start()
        try:
            self.assertTrue(planner.first_entered.wait(timeout=1.0))
            stop_thread.start()
            self.assertTrue(plugin._stop_requested.wait(timeout=1.0))
        finally:
            planner.release_first.set()
            start_thread.join(timeout=3.0)
            if stop_thread.ident is not None:
                stop_thread.join(timeout=3.0)
        self.assertFalse(start_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(start_result["error_code"], "navigation_start_cancelled")
        self.assertEqual(planner.start_attempts, 1)
        self.assertEqual(stop_result["state"], "idle")
        self.assertFalse(runtime.started)

    def test_stop_wakes_transient_planner_retry_wait(self):
        planning = TransientPlanningComponent()
        plugin = self.make_plugin(planning=planning)
        entered = threading.Event()
        original_wait = plugin._stop_requested.wait
        start_result, stop_result = {}, {}

        def observed_wait(timeout=None):
            entered.set()
            return original_wait(timeout=2.0)

        start_thread = threading.Thread(target=lambda: start_result.update(
            plugin.dispatch(plugin.PREFIX, {
                "action": "start", "input_bindings": _external_bindings(),
            })
        ))
        stop_thread = threading.Thread(target=lambda: stop_result.update(
            plugin.dispatch(plugin.PREFIX, {"action": "stop"})
        ))
        with mock.patch.object(plugin._stop_requested, "wait", side_effect=observed_wait):
            start_thread.start()
            try:
                self.assertTrue(entered.wait(timeout=1.0))
                stop_thread.start()
                start_thread.join(timeout=1.0)
                stop_thread.join(timeout=1.0)
                self.assertFalse(start_thread.is_alive())
                self.assertFalse(stop_thread.is_alive())
            finally:
                plugin._stop_requested.set()
                start_thread.join(timeout=3.0)
                if stop_thread.ident is not None:
                    stop_thread.join(timeout=3.0)
        self.assertEqual(planning.start_attempts, 1)
        self.assertEqual(start_result["error_code"], "navigation_start_cancelled")
        self.assertEqual(stop_result["state"], "idle")

    def test_stop_cancels_inflight_start_without_closing_its_backend(self):
        mapping = BlockingStartComponent()
        plugin = self.make_plugin(mapping=mapping)
        start_result = {}
        stop_result = {}

        start_thread = threading.Thread(
            target=lambda: start_result.update(
                plugin.dispatch(
                    "controlled_semantic_spatial",
                    {"action": "start", "input_bindings": _external_bindings()},
                )
            )
        )
        start_thread.start()
        self.assertTrue(mapping.start_entered.wait(timeout=1.0))

        stop_thread = threading.Thread(
            target=lambda: stop_result.update(
                plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})
            )
        )
        stop_thread.start()
        self.assertFalse(mapping.stop_entered.wait(timeout=0.05))

        mapping.release_start.set()
        start_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)

        self.assertFalse(start_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(start_result["error_code"], "navigation_start_cancelled")
        self.assertFalse(plugin._started)
        self.assertFalse(any(args["action"] == "start" for _, args in plugin._planning.calls))
        self.assertEqual(stop_result["state"], "idle")
        self.assertTrue(mapping.stop_entered.is_set())

    def test_business_actions_route_to_owned_components(self):
        mapping = FakeComponent("mapping")
        planning = FakeComponent("planning")
        semantic = FakeComponent("semantic")
        plugin = self.make_plugin(
            mapping=mapping,
            planning=planning,
            semantic=semantic,
        )
        plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings() + [
                {"port": "rgb", "topic": "/robot/camera/rgb"}
            ]},
        )
        self.assertEqual(
            plugin.dispatch(
                "controlled_semantic_spatial",
                {"action": "start_mapping", "map_name": "room"},
            )["component"],
            "mapping",
        )
        self.assertEqual(
            plugin.dispatch(
                "controlled_semantic_spatial",
                {"action": "navigate_to_pose", "x": 1, "y": 2, "yaw": 0},
            )["component"],
            "planning",
        )
        self.assertEqual(
            plugin.dispatch("controlled_semantic_spatial", {"action": "capture"})[
                "component"
            ],
            "semantic",
        )

    def test_semantic_goal_without_control_id_lets_planner_create_task_id(self):
        planning = FakeComponent("planning")
        plugin = self.make_plugin(planning=planning)

        plugin.dispatch(plugin.PREFIX, {"action": "start", "input_bindings": _external_bindings()})
        plugin._handle_semantic_goal(
            {"x": 1.0, "y": -0.5, "yaw": 0.25, "speed": 0.5},
            admission_ticket=plugin._generation,
        )

        request = planning.calls[-1][1]
        self.assertEqual(request["action"], "navigate_to_pose")
        self.assertNotIn("_control_nav_id", request)

    def test_contradictory_component_stop_never_releases_runtime(self):
        responses = [
            {"status": "error", "terminal_confirmed": True},
            {"status": "idle", "ok": False, "terminal_confirmed": True},
            {"status": "idle", "state": "error", "terminal_confirmed": True},
            {"terminal_confirmed": True},
            {"status": "idle", "error_code": "failed", "terminal_confirmed": True},
        ]
        for response in responses:
            with self.subTest(response=response):
                runtime = FakeRuntime()
                planning = FakeComponent("planning", stop_result=response)
                plugin = self.make_plugin(runtime=runtime, planning=planning)
                self.assertEqual(plugin.dispatch(plugin.PREFIX, {
                    "action": "start", "input_bindings": _external_bindings(),
                })["state"], "ready")
                self.assertEqual(plugin.stop()["error_code"], "navigation_stop_pending")
                self.assertTrue(runtime.started)
                self.assertTrue(plugin._stop_requested.is_set())

    def test_failed_start_cleanup_pending_preserves_dependencies_until_retry(self):
        runtime = FakeRuntime()
        mapping = FakeComponent("mapping")
        planning = FakeComponent("planning", fail_start=True, stop_result={
            "state": "error", "status": "error", "error_code": "unconfirmed",
        })
        plugin = self.make_plugin(runtime=runtime, mapping=mapping, planning=planning)
        start = {"action": "start", "input_bindings": _external_bindings()}
        result = plugin.dispatch(plugin.PREFIX, start)
        self.assertEqual(result["error_code"], "navigation_start_failed")
        self.assertFalse(result["cleanup_confirmed"])
        self.assertEqual(list(result["cleanup"]), ["planning"])
        self.assertTrue(runtime.started)
        self.assertTrue(mapping.started)
        self.assertTrue(plugin._stop_requested.is_set())
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["error_code"],
                         "navigation_start_cancelled")
        planning.stop_result = None
        planning.fail_start = False
        self.assertEqual(plugin.stop()["state"], "idle")
        self.assertFalse(runtime.started)
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")

    def test_start_rollback_retains_terminal_mapping_failure_in_info(self):
        runtime = FakeRuntime()
        mapping = FakeComponent("mapping", stop_result={
            "state": "error", "status": "error", "error_code": "map_write_failed",
            "terminal_confirmed": True, "canvas_wired": False, "retryable": False,
            "collection_stop_result": {"receipt": {"storage_complete": False}},
        })
        planning = FakeComponent("planning", fail_start=True)
        plugin = self.make_plugin(runtime=runtime, mapping=mapping, planning=planning)
        result = plugin.dispatch(plugin.PREFIX, {
            "action": "start", "input_bindings": _external_bindings(),
        })
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(result["error_code"], "navigation_start_failed")
        self.assertFalse(runtime.started)
        retained = plugin.dispatch(plugin.PREFIX, {"action": "info"})["last_terminal_stop_error"]
        self.assertEqual(retained["error_code"], "map_write_failed")
        self.assertFalse(retained["collection_stop_result"]["receipt"]["storage_complete"])

    def test_retryable_mapping_stop_keeps_runtime_and_canvas_lifecycle(self):
        runtime = FakeRuntime()
        mapping = FakeComponent(
            "mapping",
            stop_result={
                "state": "error",
                "status": "error",
                "error_code": "manifest_write_failed",
                "error": "temporary persistence failure",
                "retryable": True,
            },
        )
        planning = FakeComponent("planning")
        semantic = FakeComponent("semantic")
        plugin = self.make_plugin(
            runtime=runtime,
            mapping=mapping,
            planning=planning,
            semantic=semantic,
        )
        plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        result = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_pending")
        self.assertTrue(result["retryable"])
        self.assertTrue(runtime.started)
        self.assertEqual(runtime.stop_calls, 0)
        self.assertTrue(plugin._started)
        self.assertTrue(any(args["action"] == "stop" for _, args in planning.calls))
        self.assertTrue(any(args["action"] == "stop" for _, args in semantic.calls))

    def test_nonretryable_mapping_stop_still_keeps_runtime_for_retry(self):
        runtime = FakeRuntime()
        mapping = FakeComponent(
            "mapping",
            stop_result={
                "state": "error",
                "status": "error",
                "error_code": "map_write_failed",
                "error": "map transaction was not persisted",
            },
        )
        plugin = self.make_plugin(runtime=runtime, mapping=mapping)
        plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        result = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_pending")
        self.assertTrue(result["retryable"])
        self.assertTrue(runtime.started)
        self.assertEqual(runtime.stop_calls, 0)
        self.assertTrue(plugin._started)

    def test_confirmed_mapping_failure_releases_owned_runtime(self):
        runtime = FakeRuntime()
        mapping = FakeComponent(
            "mapping",
            stop_result={
                "state": "error",
                "status": "error",
                "error_code": "map_write_failed",
                "error": "map transaction failed after runtime shutdown",
                "terminal_confirmed": True,
            },
        )
        plugin = self.make_plugin(runtime=runtime, mapping=mapping)
        plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        result = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_failed")
        self.assertTrue(result["terminal_confirmed"])
        self.assertEqual(runtime.stop_calls, 1)
        self.assertFalse(plugin._started)
        mapping.stop_result = None
        self.assertEqual(plugin.stop()["state"], "idle")
        self.assertEqual(plugin.dispatch(plugin.PREFIX, {"action": "info"})[
            "last_terminal_stop_error"]["error_code"], "map_write_failed")

    def test_mapping_terminal_error_survives_runtime_pending_retry(self):
        class PendingOnceRuntime(FakeRuntime):
            def stop(self):
                if self.stop_calls == 0:
                    self.stop_calls += 1
                    return {"state": "running", "running": True}
                return super().stop()

        class ErrorOnceMapping(FakeComponent):
            def dispatch(self, prefix, args):
                result = super().dispatch(prefix, args)
                if args.get("action") == "stop":
                    self.stop_result = None
                return result

        runtime = PendingOnceRuntime()
        mapping = ErrorOnceMapping("mapping", stop_result={
            "state": "error", "status": "error", "error_code": "map_write_failed",
            "error": "map save failed", "terminal_confirmed": True,
            "canvas_wired": False, "retryable": False,
        })
        plugin = self.make_plugin(runtime=runtime, mapping=mapping)
        start = {"action": "start", "input_bindings": _external_bindings()}
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        self.assertEqual(plugin.stop()["error_code"], "navigation_stop_pending")
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["error_code"],
                         "navigation_start_cancelled")
        final = plugin.stop()
        self.assertEqual(final["error_code"], "navigation_stop_failed")
        self.assertTrue(final["terminal_confirmed"])
        self.assertEqual(final["component_results"]["mapping"]["error_code"], "map_write_failed")
        self.assertFalse(plugin._stop_requested.is_set())
        self.assertEqual(plugin.stop()["status"], "idle")
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        self.assertEqual(plugin.stop()["status"], "idle")
        self.assertEqual(plugin.dispatch(plugin.PREFIX, {"action": "info"})[
            "last_terminal_stop_error"]["error_code"], "map_write_failed")

    def test_runtime_idle_with_live_child_is_not_confirmed(self):
        class MixedRuntime(FakeRuntime):
            def stop(self):
                self.stop_calls += 1
                return {"state": "idle", "running": False, "children": [
                    {"name": "first", "running": False},
                    {"name": "second", "running": True},
                ]}

        runtime = MixedRuntime()
        plugin = self.make_plugin(runtime=runtime)
        start = {"action": "start", "input_bindings": _external_bindings()}
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["state"], "ready")
        self.assertEqual(plugin.stop()["error_code"], "navigation_stop_pending")
        self.assertTrue(plugin._started)
        self.assertEqual(plugin.dispatch(plugin.PREFIX, start)["error_code"],
                         "navigation_start_cancelled")

    def test_runtime_stop_exception_keeps_lifecycle_for_retry(self):
        class FailingStopRuntime(FakeRuntime):
            def stop(self):
                self.stop_calls += 1
                raise RuntimeError("runtime stop failed")

        runtime = FailingStopRuntime()
        plugin = self.make_plugin(runtime=runtime)
        plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        result = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_pending")
        self.assertTrue(result["retryable"])
        self.assertTrue(plugin._started)
        self.assertEqual(runtime.stop_calls, 1)

    def test_retryable_planning_stop_keeps_every_owned_runtime_for_retry(self):
        runtime = FakeRuntime()
        mapping = FakeComponent("mapping")
        planning = FakeComponent(
            "planning",
            stop_result={
                "state": "error",
                "status": "error",
                "error_code": "navigation_stop_pending",
                "error": "stop was not confirmed",
                "retryable": True,
            },
        )
        plugin = self.make_plugin(
            runtime=runtime,
            mapping=mapping,
            planning=planning,
        )
        plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        result = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_pending")
        self.assertTrue(result["retryable"])
        self.assertTrue(runtime.started)
        self.assertEqual(runtime.stop_calls, 0)
        self.assertTrue(plugin._started)
        self.assertFalse(any(args["action"] == "stop" for _, args in mapping.calls))

    def test_nonretryable_planning_stop_error_still_keeps_owned_runtime(self):
        runtime = FakeRuntime()
        mapping = FakeComponent("mapping")
        planning = FakeComponent(
            "planning",
            stop_result={
                "state": "error",
                "status": "error",
                "error_code": "stop_failed",
                "error": "stop was not confirmed",
            },
        )
        plugin = self.make_plugin(
            runtime=runtime,
            mapping=mapping,
            planning=planning,
        )
        plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        result = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_pending")
        self.assertTrue(result["retryable"])
        self.assertEqual(runtime.stop_calls, 0)
        self.assertTrue(plugin._started)
        self.assertFalse(any(args["action"] == "stop" for _, args in mapping.calls))

    def test_real_mapping_retryability_reaches_unified_lifecycle(self):
        runtime = FakeRuntime()
        backend = RetryableMappingBackend()
        mapping = FastLivo2Plugin({}, None, backend=backend)
        planning = FakeComponent("planning")
        semantic = FakeComponent("semantic")
        plugin = self.make_plugin(
            runtime=runtime,
            mapping=mapping,
            planning=planning,
            semantic=semantic,
        )
        started = plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )
        self.assertEqual(started["state"], "ready")
        mapping_started = plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "start_mapping", "map_name": "office"},
        )
        self.assertEqual(mapping_started["status"], "mapping")

        result = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_pending")
        self.assertTrue(result["retryable"])
        self.assertTrue(runtime.started)
        self.assertEqual(runtime.stop_calls, 0)
        self.assertTrue(plugin._started)
        self.assertEqual(backend.stop_calls, 0)
        self.assertTrue(
            mapping.dispatch("fast_livo2", {"action": "info"})["canvas_wired"]
        )
        self.assertTrue(any(args["action"] == "stop" for _, args in planning.calls))
        self.assertTrue(any(args["action"] == "stop" for _, args in semantic.calls))

    def test_unified_config_rejects_unknown_and_forwards_partial_vlm_fields(self):
        plugin = self.make_plugin()

        unknown = plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "config", "legacy_companion": True},
        )
        self.assertEqual(unknown["error_code"], "invalid_config")
        self.assertIn("legacy_companion", unknown["error"])

        partial = plugin.dispatch(
            "controlled_semantic_spatial",
            {"action": "config", "vlm_base_url": "https://vlm.example.test"},
        )
        self.assertEqual(partial["state"], "configured")
        self.assertIn(
            (
                "vln",
                {"action": "config", "base_url": "https://vlm.example.test"},
            ),
            plugin._semantic.calls,
        )

        configured = plugin.dispatch(
            "controlled_semantic_spatial",
            {
                "action": "config",
                "obstacle_min_height_m": -0.8,
                "obstacle_max_height_m": 0.4,
                "lidar_qos_depth": 3,
                "imu_qos_depth": 600,
                "map_view_enabled": False,
                "fault_capture_enabled": False,
                "rotate_speed_rps": 0.3,
            },
        )
        self.assertEqual(configured["state"], "configured")
        self.assertIn(
            (
                "fast_livo2",
                {
                    "action": "config",
                    "obstacle_min_height_m": -0.8,
                    "obstacle_max_height_m": 0.4,
                    "lidar_qos_depth": 3,
                    "imu_qos_depth": 600,
                    "map_view_enabled": False,
                    "fault_capture_enabled": False,
                },
            ),
            plugin._mapping.calls,
        )
        self.assertIn(
            (
                "nav2",
                {
                    "action": "config",
                    "min_yaw_rps": 0.3,
                    "max_yaw_rps": 0.3,
                },
            ),
            plugin._planning.calls,
        )

    def test_unified_config_validates_all_components_before_mutating_any(self):
        plugin = NavigationPlugin(
            {"namespace": "ubuntu"},
            "ubuntu",
            executor=None,
            runtime=FakeRuntime(),
            semantic_plugin=FakeComponent("semantic"),
        )
        before = plugin._mapping.dispatch("fast_livo2", {"action": "info"})[
            "config"
        ]

        result = plugin.dispatch(
            "controlled_semantic_spatial",
            {
                "action": "config",
                "mapping_request_timeout_sec": 77.0,
                "rotate_speed_rps": 99.0,
            },
        )

        self.assertEqual(result["error_code"], "invalid_config")
        after = plugin._mapping.dispatch("fast_livo2", {"action": "info"})[
            "config"
        ]
        self.assertEqual(after, before)

    def test_unified_config_rolls_back_if_last_component_rejects(self):
        plugin = NavigationPlugin(
            {"namespace": "ubuntu"},
            "ubuntu",
            executor=None,
            runtime=FakeRuntime(),
            semantic_plugin=FailingConfigComponent("semantic"),
        )
        mapping_before = plugin._mapping.dispatch(
            "fast_livo2", {"action": "info"}
        )["config"]
        planning_before = plugin._planning.dispatch("nav2", {"action": "info"})[
            "config"
        ]

        result = plugin.dispatch(
            "controlled_semantic_spatial",
            {
                "action": "config",
                "mapping_request_timeout_sec": 77.0,
                "rotate_speed_rps": 0.3,
                "vlm_base_url": "https://vlm.example.test/v1",
                "vlm_api_key": "test-secret",
                "vlm_model": "test-model",
            },
        )

        self.assertEqual(result["error_code"], "invalid_config")
        self.assertEqual(
            plugin._mapping.dispatch("fast_livo2", {"action": "info"})["config"],
            mapping_before,
        )
        self.assertEqual(
            plugin._planning.dispatch("nav2", {"action": "info"})["config"],
            planning_before,
        )
        self.assertEqual(
            set(result["rollback_component_results"]), {"mapping", "planning"}
        )


class NavigationRuntimeTest(unittest.TestCase):
    class Process:
        next_pid = 100

        def __init__(self, command, *, exit_immediately=False):
            self.command = command
            self.return_code = 1 if exit_immediately else None
            self.pid = self.__class__.next_pid
            self.__class__.next_pid += 1

        def poll(self):
            return self.return_code

        def wait(self, timeout=None):
            del timeout
            self.return_code = 0
            return 0

    def test_launch_arguments_without_descriptor_preserve_legacy_and_default_control(self):
        arguments = dict(value.split(":=", 1) for value in
                         NavigationRuntime._launch_arguments("/other/", {})["nav2"])
        self.assertEqual(arguments["motion_sequence_topic"], "/other/navigation/motion_sequence")
        self.assertEqual(arguments["control_velocity_topic"], "/other/navigation/control_velocity")
        self.assertEqual(json.loads(arguments["control_interface_json"]), {})

    def test_minimal_descriptor_round_trips_into_bridge_configuration(self):
        # Load the ROS-free helper through its independent package path, just as
        # the subprocess does, without relying on a ROS installation.
        import importlib
        from plugins.control_stream import negotiate

        nav2_path = str(ACTUCORE_ROOT / "plugins/navigation/runtime/nav2")
        descriptors = [
            {},
            {"control_interface": "motus.control/1", "mode": "twist"},
            {"control_interface": "motus.control/1", "mode": "twist", "dof": 6,
             "rate": {"expected_hz": 8, "watchdog_ms": 50}},
            _control_descriptor(),
        ]
        with mock.patch.object(sys, "path", [nav2_path, *sys.path]):
            helper = importlib.import_module("nav2.control_velocity")
            for descriptor in descriptors:
                with self.subTest(descriptor=descriptor):
                    args = dict(value.split(":=", 1) for value in
                                NavigationRuntime._launch_arguments(
                                    "robot", {}, descriptor)["nav2"])
                    received, hz, ttl, *_ = helper.configure(args["control_interface_json"])
                    self.assertEqual(received, descriptor)
                    self.assertEqual(hz, negotiate.effective_rate(
                        {"control_hz": 10}, descriptor, 10))
                    self.assertEqual(ttl, negotiate.ttl_ms(hz, descriptor))

    def test_runtime_launches_ros_process_groups_without_docker(self):
        commands = []

        def factory(command, **kwargs):
            commands.append((list(command), dict(kwargs)))
            return self.Process(command)

        runtime = NavigationRuntime(popen_factory=factory, startup_grace_sec=0)
        started = runtime.start(
            namespace="robot",
            input_topics={
                "lidar": "/sensors/lidar",
                "imu": "/sensors/imu",
                "rgb": "/camera/rgb",
            },
            control_interface=_control_descriptor(),
        )
        self.assertEqual(started["state"], "running")
        self.assertEqual([item[0][2] for item in commands], ["fast_livo2", "nav2"])
        self.assertTrue(all(item[1]["start_new_session"] for item in commands))
        self.assertFalse(started["docker_runtime_dependency"])
        self.assertNotIn("docker", str([item[0] for item in commands]))
        fast_command = commands[0][0]
        nav2_command = commands[1][0]
        self.assertIn("lidar_topic:=/sensors/lidar", fast_command)
        self.assertIn("imu_topic:=/sensors/imu", fast_command)
        self.assertIn("rgb_topic:=/camera/rgb", fast_command)
        self.assertIn("rgb_enabled:=true", fast_command)
        self.assertIn("depth_enabled:=false", fast_command)
        self.assertIn("depth_topic:=/robot/camera/depth_frame", fast_command)
        self.assertIn("odom_topic:=/robot/navigation/odom", fast_command)
        self.assertIn("odom_topic:=/robot/navigation/odom", nav2_command)
        self.assertIn(
            "motion_sequence_topic:=/robot/navigation/motion_sequence",
            nav2_command,
        )
        self.assertIn("control_velocity_topic:=/robot/navigation/control_velocity", nav2_command)
        arguments = dict(arg.split(":=", 1) for arg in nav2_command[4:])
        self.assertEqual(json.loads(arguments["control_interface_json"]), _control_descriptor())
        self.assertEqual(arguments["control_interface_json"], json.dumps(
            _control_descriptor(), ensure_ascii=False, separators=(",", ":")
        ))
        with mock.patch("plugins.navigation.runtime.os.killpg") as killpg:
            stopped = runtime.stop()
        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(killpg.call_count, 2)

    def test_optional_camera_launch_flags_preserve_legacy_defaults(self):
        cases = (
            (None, ("true", "true")),
            ({}, ("false", "false")),
            ({"lidar": "/lidar", "imu": "/imu"}, ("false", "false")),
            ({"rgb": "/custom/rgb"}, ("true", "false")),
            ({"depth_frame": "/custom/depth"}, ("false", "true")),
            ({"rgb": "/custom/rgb", "depth_frame": "/custom/depth"}, ("true", "true")),
            ({"rgb": "  ", "depth_frame": ""}, ("false", "false")),
        )
        for mapping, flags in cases:
            with self.subTest(mapping=mapping):
                args = NavigationRuntime._launch_arguments("robot", mapping)["fast_livo2"]
                values = dict(arg.split(":=", 1) for arg in args)
                self.assertEqual((values["rgb_enabled"], values["depth_enabled"]), flags)
                self.assertTrue(values["rgb_topic"].startswith("/"))
                self.assertTrue(values["depth_topic"].startswith("/"))

    def test_runtime_signals_owned_nested_session_groups_before_launch_group(self):
        def factory(command, **kwargs):
            del kwargs
            return self.Process(command)

        runtime = NavigationRuntime(popen_factory=factory, startup_grace_sec=0)
        started = runtime.start()
        root_groups = {
            child["pid"]
            for child in started["children"]
            if child["pid"] is not None
        }
        nested_groups = {
            pid: _OwnedProcessGroup(pid + 1000, f"start-{pid}")
            for pid in root_groups
        }
        with mock.patch.object(
            NavigationRuntime,
            "_independent_descendant_process_groups",
            side_effect=lambda pid: (nested_groups[pid],),
        ), mock.patch.object(
            NavigationRuntime,
            "_remaining_process_groups",
            side_effect=lambda groups: groups,
        ), mock.patch.object(
            NavigationRuntime,
            "_wait_for_process_groups",
            return_value=(),
        ), mock.patch("plugins.navigation.runtime.os.killpg") as killpg:
            stopped = runtime.stop()

        self.assertEqual(stopped["state"], "idle")
        calls = [(call.args[0], call.args[1]) for call in killpg.call_args_list]
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(sig == signal.SIGINT for _, sig in calls))
        self.assertEqual(
            {group for group, _ in calls},
            root_groups | {pid + 1000 for pid in root_groups},
        )

    def test_runtime_escalates_nested_group_when_launch_root_exits_first(self):
        def factory(command, **kwargs):
            del kwargs
            return self.Process(command)

        runtime = NavigationRuntime(popen_factory=factory, startup_grace_sec=0)
        runtime._children = runtime._children[:1]
        started = runtime.start()
        root_pid = started["children"][0]["pid"]
        nested_group = _OwnedProcessGroup(root_pid + 1000, "owned-start")

        with mock.patch.object(
            NavigationRuntime,
            "_independent_descendant_process_groups",
            return_value=(nested_group,),
        ), mock.patch.object(
            NavigationRuntime,
            "_remaining_process_groups",
            side_effect=lambda groups: groups,
        ), mock.patch.object(
            NavigationRuntime,
            "_wait_for_process_groups",
            side_effect=[(nested_group,), (nested_group,), ()],
        ), mock.patch("plugins.navigation.runtime.os.killpg") as killpg:
            stopped = runtime.stop()

        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(
            [(call.args[0], call.args[1]) for call in killpg.call_args_list],
            [
                (nested_group.group_id, signal.SIGINT),
                (root_pid, signal.SIGINT),
                (nested_group.group_id, signal.SIGTERM),
                (nested_group.group_id, signal.SIGKILL),
            ],
        )

    def test_runtime_drops_reused_descendant_group_identity(self):
        owned_group = _OwnedProcessGroup(1234, "original-start")
        with mock.patch(
            "plugins.navigation.runtime.os.getpgid", return_value=1234
        ), mock.patch.object(
            NavigationRuntime,
            "_process_start_time",
            return_value="replacement-start",
        ):
            remaining = NavigationRuntime._remaining_process_groups(
                (owned_group,)
            )

        self.assertEqual(remaining, ())

    def test_runtime_uses_cached_group_after_root_exited_before_stop(self):
        def factory(command, **kwargs):
            del kwargs
            return self.Process(command)

        runtime = NavigationRuntime(popen_factory=factory, startup_grace_sec=0)
        runtime._children = runtime._children[:1]
        runtime.start()
        child = runtime._children[0]
        nested_group = _OwnedProcessGroup(child.process.pid + 1000, "cached")
        with child.independent_groups_lock:
            child.independent_groups[nested_group.group_id] = nested_group
        child.process.return_code = 0

        with mock.patch.object(
            NavigationRuntime,
            "_independent_descendant_process_groups",
            return_value=(),
        ), mock.patch.object(
            NavigationRuntime,
            "_remaining_process_groups",
            side_effect=lambda groups: groups,
        ), mock.patch.object(
            NavigationRuntime,
            "_wait_for_process_groups",
            side_effect=[(nested_group,), ()],
        ), mock.patch("plugins.navigation.runtime.os.killpg") as killpg:
            stopped = runtime.stop()

        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(
            [(call.args[0], call.args[1]) for call in killpg.call_args_list],
            [
                (nested_group.group_id, signal.SIGINT),
                (nested_group.group_id, signal.SIGTERM),
            ],
        )

    def test_runtime_never_signals_reused_cached_group_identity(self):
        def factory(command, **kwargs):
            del kwargs
            return self.Process(command)

        runtime = NavigationRuntime(popen_factory=factory, startup_grace_sec=0)
        runtime._children = runtime._children[:1]
        runtime.start()
        child = runtime._children[0]
        reused_group = _OwnedProcessGroup(child.process.pid + 1000, "old-start")
        with child.independent_groups_lock:
            child.independent_groups[reused_group.group_id] = reused_group
        child.process.return_code = 0

        with mock.patch.object(
            NavigationRuntime,
            "_independent_descendant_process_groups",
            return_value=(),
        ), mock.patch.object(
            NavigationRuntime,
            "_remaining_process_groups",
            return_value=(),
        ), mock.patch("plugins.navigation.runtime.os.killpg") as killpg:
            stopped = runtime.stop()

        self.assertEqual(stopped["state"], "idle")
        killpg.assert_not_called()

    @unittest.skipUnless(
        Path("/proc/self/stat").exists(), "requires Linux procfs"
    )
    def test_runtime_reaps_cached_nested_group_after_root_already_exited(self):
        nested_code = (
            "import signal,time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "time.sleep(60)\n"
        )
        launcher_code = (
            "import subprocess,sys,time\n"
            "child=subprocess.Popen(\n"
            f"    [sys.executable, '-c', {nested_code!r}],\n"
            "    stdin=subprocess.DEVNULL,\n"
            "    stdout=subprocess.DEVNULL,\n"
            "    stderr=subprocess.DEVNULL,\n"
            "    start_new_session=True,\n"
            ")\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(0.4)\n"
        )
        created = []
        nested_pid = None

        def factory(command, **kwargs):
            del command
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.PIPE
            kwargs["text"] = True
            process = subprocess.Popen(
                [sys.executable, "-c", launcher_code], **kwargs
            )
            created.append(process)
            return process

        def pid_is_running(pid):
            try:
                raw_stat = Path(f"/proc/{pid}/stat").read_text(
                    encoding="ascii"
                )
            except OSError:
                return False
            command_end = raw_stat.rfind(")")
            fields = raw_stat[command_end + 1 :].split()
            return bool(fields) and fields[0] != "Z"

        runtime = NavigationRuntime(
            popen_factory=factory,
            startup_grace_sec=0,
            stop_timeout_sec=0.5,
        )
        runtime._children = runtime._children[:1]
        try:
            runtime.start()
            root_process = created[0]
            nested_pid = int(root_process.stdout.readline().strip())
            root_process.wait(timeout=2.0)
            self.assertTrue(pid_is_running(nested_pid))

            stopped = runtime.stop()

            self.assertEqual(stopped["state"], "idle")
            deadline = time.monotonic() + 2.0
            while pid_is_running(nested_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(pid_is_running(nested_pid))
        finally:
            if nested_pid is not None and pid_is_running(nested_pid):
                try:
                    os.killpg(nested_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class FastLivo2BackendTest(unittest.TestCase):
    def test_stop_mapping_timeout_covers_supervisor_shutdown_and_map_save(self):
        backend = object.__new__(RosTopicFastLivo2Backend)
        backend._request_timeout = 5.0

        self.assertEqual(backend._response_timeout("start_mapping"), 5.0)
        self.assertEqual(backend._response_timeout("stop_mapping"), 360.0)
        self.assertEqual(backend._response_timeout("unload_map"), 360.0)
        self.assertEqual(backend._response_timeout("load_map"), 900.0)
        self.assertEqual(backend._response_timeout("relocalize"), 180.0)
        backend._request_timeout = 400.0
        self.assertEqual(backend._response_timeout("stop_mapping"), 400.0)
        self.assertEqual(backend._response_timeout("unload_map"), 400.0)
        self.assertEqual(backend._response_timeout("load_map"), 900.0)
        self.assertEqual(backend._response_timeout("relocalize"), 400.0)

    def test_late_backend_response_is_not_retained(self):
        backend = object.__new__(RosTopicFastLivo2Backend)
        backend._condition = threading.Condition()
        backend._last_status = {}
        backend._responses = {}
        backend._pending_requests = {"active"}

        backend._on_status(
            type(
                "Message",
                (),
                {
                    "data": json.dumps(
                        {
                            "event": "response",
                            "request_id": "late",
                            "status": "saved",
                        }
                    )
                },
            )()
        )
        self.assertEqual(backend._responses, {})

        backend._on_status(
            type(
                "Message",
                (),
                {
                    "data": json.dumps(
                        {
                            "event": "response",
                            "request_id": "active",
                            "status": "saved",
                        }
                    )
                },
            )()
        )
        self.assertIn("active", backend._responses)


if __name__ == "__main__":
    unittest.main()
