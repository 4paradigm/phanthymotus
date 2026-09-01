from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))

from plugins.navigation.contract import navigation_tool_definition  # noqa: E402
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


def _external_bindings():
    return [
        {"port": item["port"], "topic": item["topic"]}
        for item in navigation_tool_definition("ubuntu")["topic_in"]
        if item["required"]
    ]


class NavigationContractTest(unittest.TestCase):
    def test_one_public_card_hides_internal_ros_edges(self):
        tool = navigation_tool_definition("ubuntu")
        self.assertEqual(tool["name"], "ControlledSemanticSpatial")
        self.assertEqual(tool["displayName"], "ControlledSemanticSpatial")
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
        self.assertNotIn("livo_odom", {item["port"] for item in tool["topic_in"]})
        self.assertEqual(
            [item["port"] for item in tool["topic_out"]],
            [
                "map_view",
                "status",
                "collection_status",
                "velocity_proposal",
                "costmap",
            ],
        )
        topic_action = tool["inputSchema"]["x-topic-actions"][0]
        self.assertNotIn("x-topic-actions", tool)
        self.assertEqual(topic_action["action"], "navigate_to_pose")
        self.assertEqual(topic_action["port"], "goal_pose")
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

    def test_long_running_navigation_does_not_declare_acp_completion(self):
        """agent-core 的 await_pending() 是全局 barrier，导航不能挂上去。

        一次导航可能几分钟，声明 x-completion 会让这段时间里所有
        actuator/processor 调用（含 TTS 与本卡片自己的 stop_nav）都被挡住，
        等于移动中失去叫停通路。阻塞语义走显式 wait_navigation_done。
        """
        tool = navigation_tool_definition("ubuntu")
        self.assertNotIn("x-completion", tool["inputSchema"])
        self.assertNotIn("x-completion", tool)
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
        self.assertIn('plugins_cfg.get("navigation"', main)
        self.assertNotIn('plugins_cfg.get("fast_livo2"', main)
        self.assertNotIn('plugins_cfg.get("nav2"', main)
        self.assertNotIn('plugins_cfg.get("vln"', main)
        self.assertIn("plugins.navigation.namespace must be a string", main)
        self.assertIn("requires namespace=ubuntu", main)
        self.assertNotIn("socket.gethostname()", main)

    def test_navigation_runtime_uses_default_fastdds_transport(self):
        manifests = (
            ACTUCORE_ROOT / "Dockerfile.jetson",
            ACTUCORE_ROOT / "deploy" / "service.yml",
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "deploy"
            / "scripts"
            / "owner-start-g1-test-containers.sh",
        )
        for manifest in manifests:
            content = manifest.read_text(encoding="utf-8")
            self.assertIn("FASTDDS_BUILTIN_TRANSPORTS=DEFAULT", content)
            self.assertNotIn("FASTDDS_BUILTIN_TRANSPORTS=UDPv4", content)

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
        self.assertNotIn("pip3 install", dockerfile)
        self.assertNotIn("python3-pip", dockerfile)
        self.assertIn("import numpy, rosbag2_py, yaml, em", dockerfile)
        self.assertNotIn("import requests", dockerfile)
        self.assertNotIn("deploy/ros-base/audio_msgs", dockerfile)
        self.assertIn("COPY actucore/main.py", dockerfile)
        self.assertIn("COPY actucore/utils/", dockerfile)
        self.assertIn("COPY actucore/deploy/     /deploy/", dockerfile)
        self.assertIn("echo /work", dockerfile)
        self.assertIn('python3 -c "from utils import logsafe"', dockerfile)

    def test_every_python_process_installs_the_atomic_log_writer(self):
        sources = [
            ACTUCORE_ROOT / "main.py",
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "runtime"
            / "g1_fast_livo2"
            / "g1_fast_livo2"
            / "adapter_node.py",
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "runtime"
            / "g1_fast_livo2"
            / "g1_fast_livo2"
            / "runtime_supervisor.py",
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "runtime"
            / "g1_nav2"
            / "g1_nav2"
            / "planner_command_node.py",
        ]
        for source in sources:
            content = source.read_text(encoding="utf-8")
            self.assertIn("from utils import logsafe", content)
            self.assertIn("logsafe.install()", content)
            self.assertLess(
                content.index("logsafe.install()"), content.index("import rclpy")
            )
        self.assertEqual(
            (ACTUCORE_ROOT / "utils" / "logsafe.py").read_bytes(),
            (ACTUCORE_ROOT.parent / "perception" / "utils" / "logsafe.py").read_bytes(),
        )

    def test_navigation_base_owns_locked_third_party_builds(self):
        base_dockerfile = (
            ACTUCORE_ROOT / "Dockerfile.navigation-base"
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
        self.assertIn("FROM ${ACTUCORE_NAVIGATION_BASE_IMAGE}", dockerfile)
        self.assertIn("ARG ACTUCORE_NAVIGATION_BASE_IMAGE\n", dockerfile)
        self.assertNotIn("actucore-navigation-base@sha256:", dockerfile)
        self.assertNotIn("missing-digest", dockerfile)
        self.assertIn("--base", build_script)
        self.assertIn("ACTUCORE_NAVIGATION_BASE_IMAGE override", build_script)
        self.assertIn("JP${JP_VERSION} navigation base is not published", build_script)
        self.assertIn(
            "actucore-navigation-base@sha256:ce5c52b9fe7451a8c202632267b270460f20483fe94f35e9a130e580aaecddc9",
            build_script,
        )
        self.assertIn(
            "build the navigation base on native ARM64, not through QEMU",
            build_script,
        )
        self.assertNotIn("git fetch", dockerfile)
        self.assertNotIn("FAST_LIVO2_REPO", dockerfile)
        self.assertNotIn("NAVIGATION2_REPO", dockerfile)
        self.assertIn("g1_fast_livo2 g1_segmented_controller g1_nav2", dockerfile)
        self.assertIn("ros2 pkg prefix g1_nav2", dockerfile)
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

    def test_g1_deploy_uses_default_actucore_builder(self):
        deploy_script = (
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "deploy"
            / "scripts"
            / "deploy-g1.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "./deploy/build_actucore.sh --mirror tuna", deploy_script
        )
        self.assertNotIn("--variant navigation", deploy_script)
        self.assertIn("STAGE=stop", deploy_script)
        self.assertIn("STAGE=preflight", deploy_script)
        self.assertIn("STAGE=start", deploy_script)

        runtime_script = (
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "deploy"
            / "scripts"
            / "owner-start-g1-test-containers.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('OWNER_VALUE="navigation-card"', runtime_script)
        self.assertIn('LEGACY_OWNER_VALUE="nav2-card"', runtime_script)
        self.assertIn(
            "refusing to remove container owned by", runtime_script
        )
        self.assertIn('"ControlledSemanticSpatial" in tools', runtime_script)
        self.assertNotIn('"navigation" in tools', runtime_script)

    def test_g1_status_probe_is_quiet_while_mcp_is_starting(self):
        runtime_script = (
            ACTUCORE_ROOT
            / "plugins"
            / "navigation"
            / "deploy"
            / "scripts"
            / "owner-start-g1-test-containers.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            fake_docker = Path(directory) / "docker"
            fake_docker.write_text(
                """#!/usr/bin/env bash
if [[ "$*" == *"--format {{.State.Running}}"* ]]; then
  echo true
elif [[ "$*" == *"--format"* ]]; then
  echo '/fake|image=fake|status=running|running=true|owner=navigation-card'
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            result = subprocess.run(
                ["bash", str(runtime_script)],
                env={
                    **os.environ,
                    "PATH": f"{directory}:{os.environ['PATH']}",
                    "STAGE": "status",
                    "ACTUCORE_CONTAINER": "fake",
                    "MCP_URL": "http://127.0.0.1:1/mcp",
                },
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fake|image=fake|status=running", result.stdout)
        self.assertNotIn("curl:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("JSONDecodeError", result.stderr)


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

    def test_only_ControlledSemanticSpatial_is_a_public_dispatch_name(self):
        plugin = self.make_plugin()

        self.assertEqual(plugin.PREFIX, "ControlledSemanticSpatial")
        self.assertIsNone(plugin.dispatch("navigation", {"action": "info"}))
        self.assertEqual(
            plugin.dispatch("ControlledSemanticSpatial", {"action": "info"})[
                "name"
            ],
            "ControlledSemanticSpatial",
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
            "ControlledSemanticSpatial",
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
        stopped = plugin.dispatch("ControlledSemanticSpatial", {"action": "stop"})
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
            "ControlledSemanticSpatial",
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
                    "ControlledSemanticSpatial",
                    {"action": "start", "input_bindings": bindings},
                )

                self.assertEqual(result["error_code"], "invalid_canvas_wiring")
                self.assertFalse(runtime.started)

    def test_collection_reuses_rgb_and_requires_depth_binding(self):
        mapping = CollectionMappingComponent("mapping")
        plugin = self.make_plugin(mapping=mapping)

        missing = plugin.dispatch(
            "ControlledSemanticSpatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )
        self.assertEqual(missing["error_code"], "invalid_canvas_wiring")
        self.assertIn("rgb", missing["error"])
        self.assertIn("depth_frame", missing["error"])

        all_bindings = [
            {"port": item["port"], "topic": item["topic"]}
            for item in navigation_tool_definition("ubuntu")["topic_in"]
        ]
        started = plugin.dispatch(
            "ControlledSemanticSpatial",
            {"action": "start", "input_bindings": all_bindings},
        )
        self.assertEqual(started["state"], "ready")

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
            "ControlledSemanticSpatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        self.assertEqual(result["error_code"], "navigation_start_failed")
        self.assertFalse(runtime.started)
        self.assertEqual(runtime.stop_calls, 1)
        self.assertTrue(any(args["action"] == "stop" for _, args in mapping.calls))

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
            "ControlledSemanticSpatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        self.assertEqual(result["state"], "ready")
        self.assertEqual(planning.start_attempts, 2)
        self.assertEqual(runtime.stop_calls, 0)
        self.assertEqual(
            sum(args["action"] == "start" for _, args in mapping.calls),
            1,
        )

    def test_stop_waits_for_inflight_start_instead_of_closing_its_backend(self):
        mapping = BlockingStartComponent()
        plugin = self.make_plugin(mapping=mapping)
        start_result = {}
        stop_result = {}

        start_thread = threading.Thread(
            target=lambda: start_result.update(
                plugin.dispatch(
                    "ControlledSemanticSpatial",
                    {"action": "start", "input_bindings": _external_bindings()},
                )
            )
        )
        start_thread.start()
        self.assertTrue(mapping.start_entered.wait(timeout=1.0))

        stop_thread = threading.Thread(
            target=lambda: stop_result.update(
                plugin.dispatch("ControlledSemanticSpatial", {"action": "stop"})
            )
        )
        stop_thread.start()
        self.assertFalse(mapping.stop_entered.wait(timeout=0.05))

        mapping.release_start.set()
        start_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)

        self.assertFalse(start_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(start_result["state"], "ready")
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
            "ControlledSemanticSpatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )
        self.assertEqual(
            plugin.dispatch(
                "ControlledSemanticSpatial",
                {"action": "start_mapping", "map_name": "room"},
            )["component"],
            "mapping",
        )
        self.assertEqual(
            plugin.dispatch(
                "ControlledSemanticSpatial",
                {"action": "navigate_to_pose", "x": 1, "y": 2, "yaw": 0},
            )["component"],
            "planning",
        )
        self.assertEqual(
            plugin.dispatch("ControlledSemanticSpatial", {"action": "capture"})[
                "component"
            ],
            "semantic",
        )

    def test_semantic_goal_without_control_id_lets_planner_create_task_id(self):
        planning = FakeComponent("planning")
        plugin = self.make_plugin(planning=planning)

        plugin._handle_semantic_goal(
            {"x": 1.0, "y": -0.5, "yaw": 0.25, "speed": 0.5}
        )

        request = planning.calls[-1][1]
        self.assertEqual(request["action"], "navigate_to_pose")
        self.assertNotIn("_control_nav_id", request)

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
            "ControlledSemanticSpatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        result = plugin.dispatch("ControlledSemanticSpatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_pending")
        self.assertTrue(result["retryable"])
        self.assertTrue(runtime.started)
        self.assertEqual(runtime.stop_calls, 0)
        self.assertTrue(plugin._started)
        self.assertFalse(any(args["action"] == "stop" for _, args in planning.calls))
        self.assertFalse(any(args["action"] == "stop" for _, args in semantic.calls))

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
            "ControlledSemanticSpatial",
            {"action": "start", "input_bindings": _external_bindings()},
        )
        self.assertEqual(started["state"], "ready")
        mapping_started = plugin.dispatch(
            "ControlledSemanticSpatial",
            {"action": "start_mapping", "map_name": "office"},
        )
        self.assertEqual(mapping_started["status"], "mapping")

        result = plugin.dispatch("ControlledSemanticSpatial", {"action": "stop"})

        self.assertEqual(result["error_code"], "navigation_stop_pending")
        self.assertTrue(result["retryable"])
        self.assertTrue(runtime.started)
        self.assertEqual(runtime.stop_calls, 0)
        self.assertTrue(plugin._started)
        self.assertEqual(backend.stop_calls, 0)
        self.assertTrue(
            mapping.dispatch("fast_livo2", {"action": "info"})["canvas_wired"]
        )
        self.assertFalse(any(args["action"] == "stop" for _, args in planning.calls))
        self.assertFalse(any(args["action"] == "stop" for _, args in semantic.calls))

    def test_unified_config_rejects_unknown_and_forwards_partial_vlm_fields(self):
        plugin = self.make_plugin()

        unknown = plugin.dispatch(
            "ControlledSemanticSpatial",
            {"action": "config", "legacy_companion": True},
        )
        self.assertEqual(unknown["error_code"], "invalid_config")
        self.assertIn("legacy_companion", unknown["error"])

        partial = plugin.dispatch(
            "ControlledSemanticSpatial",
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
            "ControlledSemanticSpatial",
            {
                "action": "config",
                "obstacle_min_height_m": -0.8,
                "obstacle_max_height_m": 0.4,
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
            "ControlledSemanticSpatial",
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
            "ControlledSemanticSpatial",
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
        )
        self.assertEqual(started["state"], "running")
        self.assertEqual([item[0][2] for item in commands], ["g1_fast_livo2", "g1_nav2"])
        self.assertTrue(all(item[1]["start_new_session"] for item in commands))
        self.assertFalse(started["docker_runtime_dependency"])
        self.assertNotIn("docker", str(commands))
        fast_command = commands[0][0]
        nav2_command = commands[1][0]
        self.assertIn("lidar_topic:=/sensors/lidar", fast_command)
        self.assertIn("imu_topic:=/sensors/imu", fast_command)
        self.assertIn("rgb_topic:=/camera/rgb", fast_command)
        self.assertIn("odom_topic:=/robot/navigation/odom", fast_command)
        self.assertIn("odom_topic:=/robot/navigation/odom", nav2_command)
        self.assertIn(
            "velocity_proposal_topic:=/robot/navigation/nav2/velocity_proposal",
            nav2_command,
        )
        with mock.patch("plugins.navigation.runtime.os.killpg") as killpg:
            stopped = runtime.stop()
        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(killpg.call_count, 2)

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
