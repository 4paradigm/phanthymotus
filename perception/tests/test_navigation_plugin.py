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


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

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
        if item.get("required", True)
    ]


class NavigationContractTest(unittest.TestCase):
    def test_one_public_card_hides_internal_ros_edges(self):
        tool = navigation_tool_definition("ubuntu")
        self.assertEqual(tool["name"], "controlled_semantic_spatial")
        self.assertEqual(tool["displayName"], "controlled_semantic_spatial")
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
        self.assertEqual(
            [item["port"] for item in tool["topic_out"]],
            ["map_view", "status", "velocity_proposal", "plan", "costmap"],
        )
        self.assertTrue(
            {
                "livo_odom",
                "registered_cloud",
                "obstacle_map",
                "static_map",
                "collection_status",
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
        self.assertEqual(
            tool["x-execution-control"]["start_actions"],
            ["navigate_to_pose", "navigate"],
        )
        self.assertEqual(tool["x-execution-control"]["version"], 2)
        self.assertEqual(
            tool["x-execution-control"]["authorize_action"],
            "authorize_navigation",
        )
        self.assertEqual(
            tool["x-execution-control"]["revoke_action"],
            "revoke_navigation",
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
        self.assertIn("obstacle_min_height_m", properties)
        self.assertIn("obstacle_max_height_m", properties)
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
        self.assertIn('"controlled_semantic_spatial" in tools', runtime_script)
        self.assertNotIn('"navigation" in tools', runtime_script)

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
        semantic_start = next(args for _, args in semantic.calls if args["action"] == "start")
        self.assertEqual(
            {item["port"] for item in semantic_start["input_bindings"]},
            {"rgb", "livo_odom", "livo_status"},
        )
        stopped = plugin.dispatch("controlled_semantic_spatial", {"action": "stop"})
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
            "controlled_semantic_spatial",
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
            "controlled_semantic_spatial",
            {"action": "start", "input_bindings": _external_bindings()},
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
        self.assertFalse(any(args["action"] == "stop" for _, args in planning.calls))
        self.assertFalse(any(args["action"] == "stop" for _, args in semantic.calls))

    def test_unified_config_rejects_unknown_and_partial_vlm_fields(self):
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
        self.assertEqual(partial["error_code"], "invalid_config")
        self.assertIn("vlm_api_key", partial["error"])
        self.assertFalse(
            any(args["action"] == "config" for _, args in plugin._semantic.calls)
        )

        configured = plugin.dispatch(
            "controlled_semantic_spatial",
            {
                "action": "config",
                "obstacle_min_height_m": -0.8,
                "obstacle_max_height_m": 0.4,
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
