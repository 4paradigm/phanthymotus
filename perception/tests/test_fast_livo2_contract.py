from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.navigation.mapping.contract import (  # noqa: E402
    FAST_LIVO2_ACTIONS,
    FAST_LIVO2_LIFECYCLE_ACTIONS,
    fast_livo2_tool_definition,
)


class FastLivo2ContractTest(unittest.TestCase):
    def test_identity_actions_and_frozen_sensor_contract(self) -> None:
        tool = fast_livo2_tool_definition("ubuntu")

        self.assertEqual(tool["name"], "fast_livo2")
        self.assertEqual(tool["displayName"], "FAST-LIVO2")
        self.assertEqual(tool["type"], "processor")
        actions = tool["inputSchema"]["properties"]["action"]["enum"]
        self.assertEqual(actions[:4], list(FAST_LIVO2_LIFECYCLE_ACTIONS))
        self.assertEqual(actions[4:], list(FAST_LIVO2_ACTIONS))
        self.assertEqual(
            actions[4:],
            ["start_mapping", "stop_mapping", "load_map", "relocalize"],
        )

        inputs = {item["port"]: item for item in tool["topic_in"]}
        self.assertEqual(set(inputs), {"lidar", "imu"})
        self.assertEqual(inputs["lidar"]["topic"], "/ubuntu/navigation/lidar")
        self.assertEqual(inputs["lidar"]["ros_type"], "sensor_msgs/msg/PointCloud2")
        self.assertEqual(
            inputs["lidar"]["qos"], "RELIABLE + KEEP_LAST(depth=2) + VOLATILE"
        )
        self.assertEqual(inputs["imu"]["topic"], "/ubuntu/navigation/imu")
        self.assertEqual(
            inputs["imu"]["qos"], "RELIABLE + KEEP_LAST(depth=200) + VOLATILE"
        )

        outputs = {item["port"]: item for item in tool["topic_out"]}
        self.assertEqual(
            set(outputs),
            {
                "livo_odom",
                "registered_cloud",
                "obstacle_map",
                "map_view",
                "status",
                "collection_status",
            },
        )
        self.assertEqual(outputs["livo_odom"]["frame_id"], "map -> base_link")
        self.assertEqual(outputs["registered_cloud"]["frame_id"], "map")
        self.assertEqual(
            outputs["obstacle_map"]["topic"], "/ubuntu/navigation/obstacle_map"
        )
        self.assertEqual(outputs["obstacle_map"]["frame_id"], "map")
        self.assertEqual(outputs["map_view"]["schema"], "phanthy.navigation.map_view.v1")
        self.assertEqual(
            outputs["collection_status"]["schema"],
            "phanthy.navigation.fast_livo2_collection_status.v1",
        )
        config = tool["configSchema"]["properties"]
        self.assertFalse(config["collection_enabled"]["default"])
        self.assertEqual(
            config["collection_directory"]["default"],
            "/opt/phanthy-motus/data/fast_livo2/recordings",
        )
        self.assertNotIn("start_recording", actions)
        self.assertNotIn("stop_recording", actions)

    def test_runtime_is_locked_inside_the_single_perception_image(self) -> None:
        runtime = PERCEPTION_ROOT / "plugins" / "navigation" / "runtime"
        source_lock = (runtime / "fast_livo2-source-lock.env").read_text(
            encoding="utf-8"
        )
        dockerfile = (PERCEPTION_ROOT / "Dockerfile.navigation").read_text(
            encoding="utf-8"
        )
        service = (PERCEPTION_ROOT / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        main = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")
        runtime_patch = runtime / "patches" / "fast-livo2-runtime.patch"
        pcd_patch = runtime / "patches" / "fast-livo2-pcd-save.patch"

        self.assertIn(
            "FAST_LIVO2_COMMIT=1fcd0d05cadaeb25ca59fd87cda95aaaee41e3ea",
            source_lock,
        )
        self.assertIn(
            "FAST_LIVO2_RUNTIME_PATCH_SHA256=534b15ab7559d572b1be56611ab1b5f5d73809f91727de5e853cd04612f4fc3b",
            source_lock,
        )
        self.assertIn(
            "FAST_LIVO2_PCD_SAVE_PATCH_SHA256=b3afa3e64b5743898c829fe34891f828027eb372324d05a8c94357f9cacd6ec4",
            source_lock,
        )
        self.assertIn(
            "APT_UBUNTU_MIRROR=mirrors.tuna.tsinghua.edu.cn", source_lock
        )
        self.assertIn(
            "APT_ROS_MIRROR=mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu",
            source_lock,
        )
        self.assertIn("ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/", source_lock)
        for variable in (
            "FAST_LIVO2_REPO",
            "FAST_LIVO2_COMMIT",
            "RPG_VIKIT_REPO",
            "RPG_VIKIT_COMMIT",
            "LIVOX_ROS_DRIVER2_REPO",
            "LIVOX_ROS_DRIVER2_COMMIT",
            "LIVOX_SDK2_REPO",
            "LIVOX_SDK2_COMMIT",
            "SOPHUS_REPO",
            "SOPHUS_COMMIT",
        ):
            self.assertIn(f"{variable}=", source_lock)
            self.assertIn(f"${{{variable}}}", dockerfile)
        self.assertEqual(
            hashlib.sha256(runtime_patch.read_bytes()).hexdigest(),
            "534b15ab7559d572b1be56611ab1b5f5d73809f91727de5e853cd04612f4fc3b",
        )
        self.assertEqual(
            hashlib.sha256(pcd_patch.read_bytes()).hexdigest(),
            "b3afa3e64b5743898c829fe34891f828027eb372324d05a8c94357f9cacd6ec4",
        )
        self.assertIn("/etc/apt/sources.list.d/*.list", dockerfile)
        self.assertIn("/etc/apt/sources.list.d/*.sources", dockerfile)
        self.assertIn("ports\\.ubuntu\\.com/ubuntu-ports", dockerfile)
        self.assertIn("packages\\.ros\\.org/ros2/ubuntu", dockerfile)
        self.assertIn("/^[[:space:]]*deb-src[[:space:]]/d", dockerfile)
        self.assertIn("^Types:[[:space:]]*deb", dockerfile)
        self.assertIn(
            "APT source-package entries remain after binary-only rewrite",
            dockerfile,
        )
        self.assertIn("GPL-2.0-only AND GPL-3.0-only", dockerfile)
        self.assertIn("FROM ${ROS_BASE_IMAGE}", dockerfile)
        self.assertNotIn("FAST_LIVO2_BASE_IMAGE", dockerfile)
        self.assertIn("git fetch --depth 1 origin", dockerfile)
        self.assertIn("git apply --check /tmp/fast-livo2-runtime.patch", dockerfile)
        self.assertIn("git apply --check /tmp/fast-livo2-pcd-save.patch", dockerfile)
        self.assertIn("--packages-select livox_ros_driver2 vikit_common vikit_ros", dockerfile)
        self.assertIn("--packages-select fast_livo", dockerfile)
        self.assertIn("PCD finalization completed", dockerfile)
        self.assertIn("g1_fast_livo2", dockerfile)
        self.assertIn("g1_nav2", dockerfile)
        self.assertNotIn("container_name: embodied-perception-fast-livo2", service)
        self.assertNotIn("container_name: embodied-perception-nav2", service)
        self.assertEqual(service.count("container_name:"), 1)
        self.assertIn("/opt/phanthy-motus/data/fast_livo2/maps", service)
        self.assertIn(
            "/opt/phanthy-motus/data/fast_livo2/recordings", service
        )
        self.assertIn("ros-humble-rosbag2-storage-mcap", dockerfile)
        self.assertIn("> /opt/g1-nav2-package-lock.txt", dockerfile)
        self.assertIn(
            'test "$(wc -l < /opt/g1-nav2-package-lock.txt)" -eq 5',
            dockerfile,
        )
        self.assertIn('full_name == p.PREFIX', main)
        self.assertIn('qualified_prefix = f"{p.PREFIX}_"', main)

    def test_contract_exposes_bounded_relocalization_not_navigation(self) -> None:
        tool = fast_livo2_tool_definition("ubuntu")
        actions = tool["inputSchema"]["properties"]["action"]["enum"]
        self.assertIn("load_map", actions)
        self.assertIn("relocalize", actions)
        self.assertNotIn("unload_map", actions)
        self.assertNotIn("global_localization", actions)
        self.assertNotIn("navigate_to_tag", actions)
        self.assertEqual(
            tool["inputSchema"]["x-action-params"]["relocalize"]["params"][:4],
            ["initial_x", "initial_y", "initial_z", "initial_yaw"],
        )

        runtime_package = (
            PERCEPTION_ROOT
            / "plugins"
            / "navigation"
            / "runtime"
            / "g1_fast_livo2"
            / "g1_fast_livo2"
        )
        supervisor = (runtime_package / "runtime_supervisor.py").read_text(
            encoding="utf-8"
        )
        adapter = (runtime_package / "adapter_node.py").read_text(encoding="utf-8")
        self.assertIn("_MAP_NAME_RE.fullmatch(map_name)", supervisor)
        self.assertIn("self._algorithm_command(save_pcd=False)", supervisor)
        self.assertIn('"/livox/lidar:=/ubuntu/navigation/lidar"', supervisor)
        self.assertNotIn("lidar_fast_livo", supervisor)
        self.assertIn('self._adapter_execute("unload_map", {})', supervisor)
        self.assertIn("self._runtime_lifecycle_lock = threading.Lock()", supervisor)
        self.assertIn("self._collection_lifecycle_lock = threading.Lock()", supervisor)
        self.assertIn('"rollback_status"', supervisor)
        self.assertIn("finalize_collection_session(", supervisor)
        self.assertIn("self._reference_points = loaded.points", adapter)
        self.assertIn("reference = self._reference_points", adapter)
        self.assertIn('"obstacle_min_height_m", -1.25', adapter)
        self.assertIn('"obstacle_max_height_m", 0.30', adapter)

        frame_adapter = (runtime_package / "frame_adapter_core.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("payload = stream.read()", frame_adapter)
        self.assertIn("stream.seek(payload_offset + point_index * byte_offset)", frame_adapter)

    def test_g1_deploy_entrypoint_stays_narrow(self) -> None:
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
        self.assertNotIn("build-companion.sh", deploy_script)
        self.assertNotIn("docker compose", deploy_script)
        self.assertIn("STAGE=stop", deploy_script)
        self.assertIn("STAGE=preflight", deploy_script)
        self.assertIn("STAGE=start", deploy_script)
        self.assertNotIn('${PERCEPTION_IMAGE:-', deploy_script)
        self.assertNotIn("git pull", deploy_script)
        self.assertNotIn("git reset", deploy_script)


if __name__ == "__main__":
    unittest.main()
