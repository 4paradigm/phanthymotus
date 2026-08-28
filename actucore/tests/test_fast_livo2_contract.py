from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))

from plugins.navigation.contract import (  # noqa: E402
    CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME,
)
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
                "static_map",
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
        self.assertEqual(outputs["static_map"]["topic"], "/ubuntu/navigation/static_map")
        self.assertEqual(outputs["static_map"]["ros_type"], "nav_msgs/msg/OccupancyGrid")
        self.assertEqual(
            outputs["static_map"]["qos"],
            "RELIABLE + KEEP_LAST(depth=1) + TRANSIENT_LOCAL",
        )
        self.assertEqual(outputs["static_map"]["frame_id"], "map")
        self.assertEqual(outputs["map_view"]["schema"], "phanthy.navigation.map_view.v1")
        self.assertEqual(
            outputs["collection_status"]["topic"],
            "/ubuntu/navigation/fast_livo2/collection_preview",
        )
        self.assertEqual(outputs["collection_status"]["format"], "image/jpeg")
        self.assertEqual(
            outputs["collection_status"]["ros_type"],
            "sensor_msgs/msg/CompressedImage",
        )
        self.assertEqual(
            outputs["collection_status"]["schema"],
            "phanthy.navigation.collection_preview.v1",
        )
        config = tool["configSchema"]["properties"]
        self.assertEqual(config["obstacle_min_height_m"]["default"], -0.30)
        self.assertEqual(config["obstacle_max_height_m"]["default"], 0.30)
        self.assertFalse(config["collection_enabled"]["default"])
        self.assertEqual(
            config["collection_directory"]["default"],
            "/opt/phanthy-motus/data/fast_livo2/recordings",
        )
        self.assertNotIn("start_recording", actions)
        self.assertNotIn("stop_recording", actions)

    def test_runtime_is_locked_in_navigation_base_and_actucore_image(self) -> None:
        runtime = ACTUCORE_ROOT / "plugins" / "navigation" / "runtime"
        source_lock = (runtime / "fast_livo2-source.lock").read_text(
            encoding="utf-8"
        )
        base_dockerfile = (ACTUCORE_ROOT / "Dockerfile.navigation-base").read_text(
            encoding="utf-8"
        )
        dockerfile = (ACTUCORE_ROOT / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )
        service = (ACTUCORE_ROOT / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        main = (ACTUCORE_ROOT / "main.py").read_text(encoding="utf-8")
        runtime_patch = runtime / "patches" / "fast-livo2-runtime.patch"
        pcd_patch = runtime / "patches" / "fast-livo2-pcd-save.patch"
        flush_patch = runtime / "patches" / "fast-livo2-pcd-flush.patch"
        livox_messages = runtime / "livox_ros_driver2_msgs"

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
            "FAST_LIVO2_PCD_FLUSH_PATCH_SHA256=1484bfba11408e3efd87360a63fef1787f2b2ceaf75e8d8abdd5a17e3474beeb",
            source_lock,
        )
        self.assertIn("pcd_save.flush_sequence", flush_patch.read_text(encoding="utf-8"))
        self.assertIn("raw PCD flush failed", flush_patch.read_text(encoding="utf-8"))
        self.assertIn("import numpy, rosbag2_py", dockerfile)
        self.assertIn("ros2 bag record --help", dockerfile)
        self.assertIn(
            "from g1_fast_livo2.camera_rgb_frame import decode as decode_rgb",
            dockerfile,
        )
        self.assertIn(
            "from g1_fast_livo2.camera_depth_frame import decode as decode_depth",
            dockerfile,
        )
        self.assertIn("assert RGB_MAGIC == DEPTH_MAGIC == b'PSE1'", dockerfile)
        # 基础镜像 digest 由构建脚本按 --jp-version 选择，不放在源码锁中。
        self.assertNotIn("ROS_BASE_IMAGE=", source_lock)
        self.assertNotIn("APT_ROS_MIRROR=", source_lock)
        for variable in (
            "FAST_LIVO2_REPO",
            "FAST_LIVO2_COMMIT",
            "RPG_VIKIT_REPO",
            "RPG_VIKIT_COMMIT",
            "SOPHUS_REPO",
            "SOPHUS_COMMIT",
        ):
            self.assertIn(f"{variable}=", source_lock)
            self.assertIn(f"${{{variable}}}", base_dockerfile)
        for removed in (
            "LIVOX_ROS_DRIVER2_REPO",
            "LIVOX_ROS_DRIVER2_COMMIT",
            "LIVOX_SDK2_REPO",
            "LIVOX_SDK2_COMMIT",
        ):
            self.assertNotIn(removed, source_lock)
            self.assertNotIn(removed, base_dockerfile)
        self.assertTrue((livox_messages / "msg" / "CustomMsg.msg").is_file())
        self.assertTrue((livox_messages / "msg" / "CustomPoint.msg").is_file())
        self.assertIn(
            "COPY actucore/plugins/navigation/runtime/livox_ros_driver2_msgs/",
            base_dockerfile,
        )
        self.assertIn(
            'test -z "$(ros2 pkg executables livox_ros_driver2)"',
            base_dockerfile,
        )
        self.assertEqual(
            hashlib.sha256(runtime_patch.read_bytes()).hexdigest(),
            "534b15ab7559d572b1be56611ab1b5f5d73809f91727de5e853cd04612f4fc3b",
        )
        self.assertEqual(
            hashlib.sha256(pcd_patch.read_bytes()).hexdigest(),
            "b3afa3e64b5743898c829fe34891f828027eb372324d05a8c94357f9cacd6ec4",
        )
        self.assertEqual(
            hashlib.sha256(flush_patch.read_bytes()).hexdigest(),
            "1484bfba11408e3efd87360a63fef1787f2b2ceaf75e8d8abdd5a17e3474beeb",
        )
        # base 是 Focal，packages.ros.org 上没有 humble 二进制，所以镜像必须
        # 把 ROS 的 apt 源整体删掉，ROS 侧全部来自源码编译。
        self.assertIn("rm -f /etc/apt/sources.list.d/*.list", base_dockerfile)
        self.assertIn(
            "ROS APT source remains; humble has no Focal binaries",
            base_dockerfile,
        )
        self.assertNotIn("ros-humble-navigation2", base_dockerfile)
        self.assertNotIn("ros-humble-nav2-bringup", base_dockerfile)
        self.assertIn("GPL-2.0-only AND GPL-3.0-only", base_dockerfile)
        self.assertIn("FROM ${ACTUCORE_NAVIGATION_PARENT_IMAGE}", base_dockerfile)
        self.assertNotIn("FAST_LIVO2_BASE_IMAGE", base_dockerfile)
        self.assertIn("git fetch --depth 1 origin", base_dockerfile)
        self.assertIn(
            "git apply --check /tmp/fast-livo2-runtime.patch", base_dockerfile
        )
        self.assertIn(
            "patch --dry-run --batch --fuzz=0 --ignore-whitespace -p1",
            base_dockerfile,
        )
        self.assertIn(
            "git apply --check /tmp/fast-livo2-pcd-save.patch", base_dockerfile
        )
        self.assertIn(
            "--packages-select livox_ros_driver2 vikit_common vikit_ros",
            base_dockerfile,
        )
        self.assertNotIn("/tmp/livox-sdk2", base_dockerfile)
        self.assertIn("--packages-select fast_livo", base_dockerfile)
        self.assertIn("PCD finalization completed", base_dockerfile)
        self.assertIn("g1_fast_livo2", dockerfile)
        self.assertIn("g1_nav2", dockerfile)
        self.assertNotIn("FAST_LIVO2_REPO", dockerfile)
        self.assertNotIn("NAVIGATION2_REPO", dockerfile)
        self.assertNotIn("container_name: embodied-perception-fast-livo2", service)
        self.assertNotIn("container_name: embodied-perception-nav2", service)
        self.assertEqual(service.count("container_name:"), 1)
        self.assertIn("/opt/phanthy-motus/data/fast_livo2/maps", service)
        self.assertIn(
            "/opt/phanthy-motus/data/fast_livo2/recordings", service
        )
        # 录包用 mcap，它由 base（dustynv 的 rosinstall 列表里带）提供，
        # 卡片侧只声明 exec_depend；镜像必须留下可核对的 ROS 包清单。
        self.assertIn(
            "> /opt/actucore-ros-package-lock.txt", base_dockerfile
        )
        self.assertIn(
            "grep -Fqx nav2_bringup /opt/actucore-ros-package-lock.txt",
            base_dockerfile,
        )
        # 工具名不含下划线，所以 ActuCoreBundle 上游那套朴素的
        # partition("_") 路由就够用了 —— 不需要给宿主打 exact-match 补丁。
        self.assertNotIn("_", CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME)
        self.assertIn('prefix, sep, tool_name = full_name.partition("_")', main)
        self.assertIn("name = tool_name if sep else prefix", main)
        self.assertIn('plugins_cfg.get("navigation"', main)

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
            ACTUCORE_ROOT
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
        self.assertIn(
            'f"/livox/lidar:={self.get_parameter(\'lidar_topic\').value}"',
            supervisor,
        )
        self.assertNotIn("lidar_fast_livo", supervisor)
        self.assertIn('self._adapter_execute("unload_map", {})', supervisor)
        self.assertIn("self._runtime_lifecycle_lock = threading.Lock()", supervisor)
        self.assertIn("self._collection_lifecycle_lock = threading.Lock()", supervisor)
        self.assertIn('"rollback_status"', supervisor)
        self.assertIn("finalize_collection_session(", supervisor)
        self.assertIn("self._reference_points = static_loaded.points", adapter)
        self.assertIn("reference = self._reference_points", adapter)
        self.assertIn('"obstacle_min_height_m", -0.30', adapter)
        self.assertIn('"obstacle_max_height_m", 0.30', adapter)
        self.assertIn('action == "configure_obstacle_filter"', adapter)
        self.assertIn('action == "save_static_map"', adapter)
        self.assertIn('action == "validate_map"', adapter)
        self.assertIn("self._static_save_result", adapter)
        self.assertNotIn("and self._static_save_result is not None", adapter)
        self.assertIn('"map_load_max_points", 200000', adapter)
        self.assertIn('"static_map_load_max_points", 200000', adapter)
        self.assertNotIn("static_confirmation_frames", adapter)
        self.assertNotIn("static_dynamic_filter_enabled", adapter)
        self.assertIn("bracketed_stamped_pose(", adapter)
        self.assertIn("latest_sensor_qos = QoSProfile(", adapter)
        self.assertIn("self._pending_cloud", adapter)
        self.assertIn('self._mode = "finalizing"', adapter)
        self.assertIn("self._static_map.observe_scan(", adapter)
        self.assertIn("self._mapping_worker_main", adapter)
        self.assertIn("self._queue_mapping_scan(", adapter)
        self.assertIn("self._mapping_work_dropped", adapter)
        self.assertIn("MultiThreadedExecutor(num_threads=4)", adapter)
        publish_fast_path = adapter.split("    def _drain_pending_cloud", 1)[1].split(
            "    def _on_reset", 1
        )[0]
        self.assertLess(
            publish_fast_path.index("self._cloud_pub.publish(navigation_cloud)"),
            publish_fast_path.index("self._queue_mapping_scan("),
        )
        self.assertNotIn("self._static_map.observe_scan(", publish_fast_path)
        self.assertNotIn("candidate_points", adapter)
        self.assertIn('map_view_context.add(work["out_of_band_points"])', adapter)
        self.assertIn("encode_map_view_points(", adapter)
        self.assertIn("self._static_map.map_view_points", adapter)
        self.assertNotIn("self._static_map.as_voxel_map()", adapter)
        self.assertIn("max_points=_MAP_VIEW_MAX_POINTS", adapter)
        self.assertIn("_MAP_VIEW_MAX_POINTS = 40_000", adapter)
        self.assertIn("_MAP_VIEW_POSE_REFRESH_HZ = 1.0", adapter)
        self.assertIn("MutuallyExclusiveCallbackGroup", adapter)
        self.assertIn("self._display_callbacks = MutuallyExclusiveCallbackGroup()", adapter)
        self.assertIn("map_view_qos = QoSProfile(", adapter)
        self.assertIn("callback_group=self._display_callbacks", adapter)
        self.assertIn("_RELOCALIZATION_MIN_MATCH_RATIO = 0.35", adapter)
        self.assertIn("self._invalidate_map_view_cache_locked()", adapter)
        self.assertIn("self._static_map.prepare_confirmed(static_loaded.points)", adapter)
        self.assertIn("self._static_map.apply_prepared_confirmed(prepared)", adapter)
        self.assertIn('result.pop("_post_response", None)', adapter)
        self.assertIn('action == "configure_obstacle_filter"', supervisor)
        self.assertIn("self._map_artifacts_from_manifest(map_name)", supervisor)
        self.assertIn('self._adapter_execute("validate_map", validation_args)', supervisor)
        self.assertIn("self._snapshot_session_pcd_files(", supervisor)
        self.assertIn("_MAX_MAP_ARTIFACT_FILES = 64", supervisor)
        self.assertIn("_MAX_MAP_ARTIFACT_BYTES = 1_073_741_824", supervisor)
        self.assertIn("_MAX_MAP_ARTIFACT_TOTAL_BYTES = 536_870_912", supervisor)
        self.assertIn("_MAX_MAP_MANIFEST_BYTES = 65_536", supervisor)
        self.assertIn('"static_map_save_failed"', supervisor)
        self.assertIn('"manifest_write_failed"', supervisor)
        self.assertIn("self._pending_mapping_finalize = pending", supervisor)
        self.assertIn('"mapping_finalize_pending"', supervisor)
        self.assertIn('"static_map_format_version": 2', supervisor)
        self.assertIn('"retryable": True', supervisor)
        self.assertIn("acquire(blocking=False)", supervisor)
        self.assertIn('"static_map_status_unavailable"', supervisor)
        self.assertNotIn("time.sleep(0.10)", supervisor)
        self.assertIn('"static_map_pcd"', supervisor)
        self.assertIn('args["obstacle_height_range_m"]', supervisor)
        self.assertIn("normalize_obstacle_height_range(", supervisor)
        self.assertIn("obstacle_height_ranges_match(", adapter)
        self.assertIn(
            "saved static map obstacle height range does not match",
            adapter,
        )
        frame_adapter = (runtime_package / "frame_adapter_core.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MVFILT2", frame_adapter)
        self.assertIn("class TemporalOccupancyMap", frame_adapter)
        self.assertIn("def write_pcd_xyz_atomic(", frame_adapter)
        self.assertIn("max_declared_points", frame_adapter)
        self.assertIn("_MAX_PCD_HEADER_BYTES = 65_536", frame_adapter)
        self.assertIn("_MAX_PCD_ASCII_RECORD_BYTES = 65_536", frame_adapter)
        self.assertIn("deadline_monotonic", frame_adapter)
        self.assertNotIn("payload = stream.read()", frame_adapter)
        self.assertIn("stream.seek(payload_offset + point_index * byte_offset)", frame_adapter)
        mapping_backend = (
            ACTUCORE_ROOT / "plugins" / "navigation" / "mapping" / "backend.py"
        ).read_text(encoding="utf-8")
        self.assertIn('if action == "stop_mapping"', mapping_backend)

    def test_g1_deploy_entrypoint_stays_narrow(self) -> None:
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
        self.assertNotIn("build-companion.sh", deploy_script)
        self.assertNotIn("docker compose", deploy_script)
        self.assertIn("STAGE=stop", deploy_script)
        self.assertIn("STAGE=preflight", deploy_script)
        self.assertIn("STAGE=start", deploy_script)
        self.assertNotIn('${ACTUCORE_IMAGE:-', deploy_script)
        self.assertNotIn("git pull", deploy_script)
        self.assertNotIn("git reset", deploy_script)


if __name__ == "__main__":
    unittest.main()
