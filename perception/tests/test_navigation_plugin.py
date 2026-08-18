from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.navigation.contract import navigation_tool_definition  # noqa: E402
from plugins.navigation.plugin import NavigationPlugin  # noqa: E402
from plugins.navigation.runtime import NavigationRuntime  # noqa: E402


class FakeRuntime:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.started = False
        self.stop_calls = 0

    def start(self):
        if self.fail:
            raise RuntimeError("runtime boom")
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
            "container_model": "single_perception_container",
            "docker_runtime_dependency": False,
        }


class FakeComponent:
    def __init__(self, name, *, fail_start=False):
        self.name = name
        self.fail_start = fail_start
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
            return {"state": "idle", "status": "idle"}
        if action == "info":
            return {"state": "ready" if self.started else "idle"}
        return {"status": action, "component": self.name}


def _external_bindings():
    return [
        {"port": item["port"], "topic": item["topic"]}
        for item in navigation_tool_definition("ubuntu")["topic_in"]
        if item.get("required", True)
    ]


class NavigationContractTest(unittest.TestCase):
    def test_one_public_card_hides_internal_ros_edges(self):
        tool = navigation_tool_definition("ubuntu")
        self.assertEqual(tool["name"], "navigation")
        self.assertEqual(tool["displayName"], "Navigation")
        self.assertEqual(
            {item["port"] for item in tool["topic_in"]},
            {"lidar", "imu", "rgb", "goal_pose"},
        )
        self.assertFalse(
            next(item for item in tool["topic_in"] if item["port"] == "goal_pose")[
                "required"
            ]
        )
        self.assertNotIn("livo_odom", {item["port"] for item in tool["topic_in"]})
        actions = tool["inputSchema"]["properties"]["action"]["enum"]
        for action in (
            "start_mapping",
            "load_map",
            "navigate_to_pose",
            "capture",
            "navigate",
        ):
            self.assertIn(action, actions)
        self.assertEqual(
            tool["x-execution-control"]["start_actions"],
            ["navigate_to_pose", "navigate"],
        )

    def test_unified_action_schema_only_exposes_supported_fields(self):
        tool = navigation_tool_definition("ubuntu")
        properties = tool["inputSchema"]["properties"]
        action_params = tool["inputSchema"]["x-action-params"]
        config_fields = list(tool["configSchema"]["properties"])

        self.assertEqual(action_params["config"]["params"], config_fields)
        self.assertEqual(
            tool["configSchema"]["required"],
            ["vlm_base_url", "vlm_api_key", "vlm_model"],
        )
        self.assertIn("vlm_api_key", properties)
        self.assertIn("planning_request_timeout_sec", properties)
        self.assertNotIn("backend", properties)
        self.assertNotIn("request_timeout_sec", properties)
        self.assertNotIn("base_url", properties)
        for action in action_params.values():
            self.assertTrue(set(action["params"]) <= set(properties))

    def test_formal_service_has_exactly_one_container(self):
        service = (PERCEPTION_ROOT / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        main = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertEqual(service.count("container_name:"), 1)
        self.assertNotIn("depends_on:", service)
        self.assertNotIn("embodied-perception-fast-livo2", service)
        self.assertNotIn("embodied-perception-nav2", service)
        self.assertIn('plugins_cfg.get("navigation"', main)
        self.assertNotIn('plugins_cfg.get("fast_livo2"', main)
        self.assertNotIn('plugins_cfg.get("nav2"', main)
        self.assertNotIn('plugins_cfg.get("vln"', main)

    def test_unified_image_retains_full_perception_vop_dependencies(self):
        dockerfile = (PERCEPTION_ROOT / "Dockerfile.navigation").read_text(
            encoding="utf-8"
        )
        self.assertIn("ultralytics/CLIP.git@488e81a", dockerfile)
        self.assertIn("/work/weights/clip/ViT-B-32.pt", dockerfile)
        self.assertIn("ARG PYTORCH_VERSION=2.2.2", dockerfile)
        self.assertIn('"torch==${PYTORCH_VERSION}"', dockerfile)
        self.assertIn("ARG TORCHAUDIO_VERSION=2.2.2", dockerfile)
        self.assertIn('"torchaudio==${TORCHAUDIO_VERSION}"', dockerfile)
        self.assertIn("ARG TORCHVISION_VERSION=0.17.2", dockerfile)
        self.assertIn('"torchvision==${TORCHVISION_VERSION}"', dockerfile)
        self.assertIn("ARG NUMPY_VERSION=1.26.4", dockerfile)
        self.assertIn('"numpy==${NUMPY_VERSION}"', dockerfile)
        self.assertIn("ARG OPENCV_PYTHON_VERSION=4.11.0.86", dockerfile)
        self.assertIn('"opencv-python==${OPENCV_PYTHON_VERSION}"', dockerfile)
        self.assertIn("ARG SETUPTOOLS_VERSION=75.8.2", dockerfile)
        self.assertIn("ARG WHEEL_VERSION=0.45.1", dockerfile)
        self.assertIn("ARG PACKAGING_VERSION=24.2", dockerfile)
        self.assertIn("--no-build-isolation", dockerfile)
        self.assertNotIn("download.pytorch.org", dockerfile)
        self.assertIn("YOLO_CONFIG_DIR=/work", dockerfile)
        self.assertIn("COPY perception/deploy/ /deploy/", dockerfile)

    def test_unified_image_builds_fast_livo2_without_external_base(self):
        dockerfile = (PERCEPTION_ROOT / "Dockerfile.navigation").read_text(
            encoding="utf-8"
        )
        build_script = (
            PERCEPTION_ROOT.parent / "deploy" / "build_perception.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("FROM ${ROS_BASE_IMAGE}", dockerfile)
        self.assertNotIn("FROM ${FAST_LIVO2_BASE_IMAGE}", dockerfile)
        self.assertIn(
            "/usr/share/ros-apt-source/ros2.sources", dockerfile
        )
        self.assertIn("/etc/apt/sources.list.d/ros2.sources", dockerfile)
        self.assertNotIn("FAST_LIVO2_BASE_IMAGE=", build_script)
        self.assertIn('VARIANT="navigation"', build_script)
        for variable in (
            "ROS_BASE_IMAGE",
            "GIT_MIRROR_PREFIX",
            "FAST_LIVO2_REPO",
            "FAST_LIVO2_RUNTIME_PATCH_SHA256",
            "FAST_LIVO2_PCD_SAVE_PATCH_SHA256",
            "RPG_VIKIT_REPO",
            "LIVOX_ROS_DRIVER2_REPO",
            "LIVOX_SDK2_REPO",
            "SOPHUS_REPO",
        ):
            self.assertIn(f'"{variable}=${{{variable}}}"', build_script)

    def test_g1_deploy_uses_default_perception_builder(self):
        deploy_script = (
            PERCEPTION_ROOT
            / "plugins"
            / "navigation"
            / "deploy"
            / "scripts"
            / "deploy-g1.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "./deploy/build_perception.sh --mirror tuna", deploy_script
        )
        self.assertNotIn("--variant navigation", deploy_script)
        self.assertIn("STAGE=stop", deploy_script)
        self.assertIn("STAGE=preflight", deploy_script)
        self.assertIn("STAGE=start", deploy_script)

        runtime_script = (
            PERCEPTION_ROOT
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

    def test_g1_status_probe_is_quiet_while_mcp_is_starting(self):
        runtime_script = (
            PERCEPTION_ROOT
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
                    "PERCEPTION_CONTAINER": "fake",
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
            "navigation",
            {
                "action": "start",
                "instance_id": "canvas-navigation",
                "input_bindings": _external_bindings(),
            },
        )

        self.assertEqual(result["state"], "ready")
        self.assertTrue(runtime.started)
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
        semantic_start = next(args for _, args in semantic.calls if args["action"] == "start")
        self.assertEqual(
            {item["port"] for item in semantic_start["input_bindings"]},
            {"rgb", "livo_odom", "livo_status"},
        )
        stopped = plugin.dispatch("navigation", {"action": "stop"})
        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(runtime.stop_calls, 1)

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
            "navigation",
            {"action": "start", "input_bindings": _external_bindings()},
        )

        self.assertEqual(result["error_code"], "navigation_start_failed")
        self.assertFalse(runtime.started)
        self.assertEqual(runtime.stop_calls, 1)
        self.assertTrue(any(args["action"] == "stop" for _, args in mapping.calls))

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
            "navigation",
            {"action": "start", "input_bindings": _external_bindings()},
        )
        self.assertEqual(
            plugin.dispatch(
                "navigation", {"action": "start_mapping", "map_name": "room"}
            )["component"],
            "mapping",
        )
        self.assertEqual(
            plugin.dispatch(
                "navigation",
                {"action": "navigate_to_pose", "x": 1, "y": 2, "yaw": 0},
            )["component"],
            "planning",
        )
        self.assertEqual(
            plugin.dispatch("navigation", {"action": "capture"})["component"],
            "semantic",
        )

    def test_unified_config_rejects_unknown_and_partial_vlm_fields(self):
        plugin = self.make_plugin()

        unknown = plugin.dispatch(
            "navigation", {"action": "config", "legacy_companion": True}
        )
        self.assertEqual(unknown["error_code"], "invalid_config")
        self.assertIn("legacy_companion", unknown["error"])

        partial = plugin.dispatch(
            "navigation",
            {"action": "config", "vlm_base_url": "https://vlm.example.test"},
        )
        self.assertEqual(partial["error_code"], "invalid_config")
        self.assertIn("vlm_api_key", partial["error"])
        self.assertFalse(
            any(args["action"] == "config" for _, args in plugin._semantic.calls)
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
        started = runtime.start()
        self.assertEqual(started["state"], "running")
        self.assertEqual([item[0][2] for item in commands], ["g1_fast_livo2", "g1_nav2"])
        self.assertTrue(all(item[1]["start_new_session"] for item in commands))
        self.assertFalse(started["docker_runtime_dependency"])
        self.assertNotIn("docker", str(commands))
        with mock.patch("plugins.navigation.runtime.os.killpg") as killpg:
            stopped = runtime.stop()
        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(killpg.call_count, 2)


if __name__ == "__main__":
    unittest.main()
