from __future__ import annotations

import sys
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.fast_livo2.contract import (  # noqa: E402
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
        self.assertEqual(
            inputs["lidar"]["topic"], "/ubuntu/navigation/lidar_fast_livo"
        )
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

    def test_companion_is_locked_and_navigation_compose_owns_the_service(self) -> None:
        companion = PERCEPTION_ROOT / "plugins" / "fast_livo2" / "companion"
        source_lock = (companion / "source-lock.env").read_text(encoding="utf-8")
        dockerfile = (companion / "Dockerfile").read_text(encoding="utf-8")
        build_script = (companion / "build-companion.sh").read_text(
            encoding="utf-8"
        )
        service = (PERCEPTION_ROOT / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        main = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")

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
        self.assertIn("GPL-2.0-only AND GPL-3.0-only", dockerfile)
        self.assertIn("org.opencontainers.image.revision", build_script)
        self.assertIn("org.opencontainers.image.fast-livo2-runtime-patch", build_script)
        self.assertIn("org.opencontainers.image.fast-livo2-pcd-save-patch", build_script)
        self.assertIn('"${actual_arch}" == "arm64"', build_script)
        self.assertNotIn("FAST_LIVO2_BASE_IMAGE_ID", build_script)
        self.assertIn("fast_livo2:", service)
        self.assertIn("/opt/phanthy-motus/data/fast_livo2/maps", service)
        self.assertIn(
            "/opt/phanthy-motus/data/fast_livo2/recordings", service
        )
        self.assertIn("ros-humble-rosbag2-storage-mcap", dockerfile)
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

        companion_package = (
            PERCEPTION_ROOT
            / "plugins"
            / "fast_livo2"
            / "companion"
            / "g1_fast_livo2"
            / "g1_fast_livo2"
        )
        supervisor = (companion_package / "runtime_supervisor.py").read_text(
            encoding="utf-8"
        )
        adapter = (companion_package / "adapter_node.py").read_text(encoding="utf-8")
        self.assertIn("_MAP_NAME_RE.fullmatch(map_name)", supervisor)
        self.assertIn("self._algorithm_command(save_pcd=False)", supervisor)
        self.assertIn('self._adapter_execute("unload_map", {})', supervisor)
        self.assertIn("self._reference_points = loaded.points", adapter)
        self.assertIn("reference = self._reference_points", adapter)

    def test_g1_build_and_start_entrypoint_stays_narrow(self) -> None:
        deploy_script = (
            PERCEPTION_ROOT
            / "plugins"
            / "nav2"
            / "deploy"
            / "scripts"
            / "build-and-start-g1.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("./deploy/build_perception.sh", deploy_script)
        self.assertIn("build-companion.sh", deploy_script)
        self.assertIn(
            "docker compose --env-file source-lock.env build nav2",
            deploy_script,
        )
        self.assertIn("STAGE=preflight", deploy_script)
        self.assertIn("STAGE=start", deploy_script)
        self.assertNotIn('${PERCEPTION_IMAGE:-', deploy_script)
        self.assertNotIn('${FAST_LIVO2_IMAGE:-', deploy_script)
        self.assertNotIn('${NAV2_IMAGE:-', deploy_script)
        self.assertNotIn("git pull", deploy_script)
        self.assertNotIn("git reset", deploy_script)
        self.assertNotIn("STAGE=stop", deploy_script)


if __name__ == "__main__":
    unittest.main()
