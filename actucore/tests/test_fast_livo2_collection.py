from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


RUNTIME_PACKAGE = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "navigation"
    / "runtime"
    / "g1_fast_livo2"
)
sys.path.insert(0, str(RUNTIME_PACKAGE))

from g1_fast_livo2.collection_core import (  # noqa: E402
    COLLECTION_EVENT_TOPICS,
    COLLECTION_SOURCES,
    CollectionHealth,
    CollectionSampler,
    finalize_collection_session,
    normalize_collection_directory,
    read_rosbag_recording_summary,
    rosbag_record_command,
)


class FastLivo2CollectionTest(unittest.TestCase):
    def test_directory_and_rosbag_command_are_bounded(self) -> None:
        directory = normalize_collection_directory(
            "/opt/phanthy-motus/data/fast_livo2/recordings/room-a"
        )
        self.assertEqual(
            directory,
            "/opt/phanthy-motus/data/fast_livo2/recordings/room-a",
        )
        for invalid in ("relative", "/tmp/recordings", "/opt/../tmp"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    normalize_collection_directory(invalid)

        command = rosbag_record_command("/safe/session.partial")
        self.assertEqual(command[:7], [
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",
            "--output",
            "/safe/session.partial",
        ])
        self.assertEqual(
            command[7:],
            [item["record_topic"] for item in COLLECTION_SOURCES]
            + [item["topic"] for item in COLLECTION_EVENT_TOPICS],
        )
        self.assertEqual(COLLECTION_SOURCES[0]["topic"], "/ubuntu/navigation/lidar")
        rgb_frame = next(
            item for item in COLLECTION_SOURCES if item["port"] == "rgb_frame"
        )
        self.assertEqual(rgb_frame["topic"], "/ubuntu/camera/rgb_frame")
        depth_frame = next(
            item for item in COLLECTION_SOURCES if item["port"] == "depth_frame"
        )
        self.assertEqual(
            depth_frame["topic"], "/ubuntu/camera/depth_frame"
        )

    def test_health_exposes_normal_and_missing_source_failure(self) -> None:
        health = CollectionHealth(grace_sec=5.0, stale_sec=2.0)
        health.start("session-a", "/recordings/session-a", now_monotonic=10.0)

        starting = health.snapshot(process_running=True, now_monotonic=11.0)
        self.assertEqual(starting["state"], "starting")
        self.assertIsNone(starting["failure_reason"])

        for index, item in enumerate(COLLECTION_SOURCES):
            if item["port"] == "rgb_frame":
                continue
            health.observe(
                item["port"],
                source_stamp_ns=1_000 + index,
                now_monotonic=12.0,
            )
        degraded = health.snapshot(process_running=True, now_monotonic=16.0)
        self.assertEqual(degraded["state"], "degraded")
        self.assertEqual(degraded["missing_sources"], ["rgb_frame"])
        self.assertEqual(
            degraded["failure_reason"], "missing_sources:rgb_frame"
        )

        for index, item in enumerate(COLLECTION_SOURCES):
            health.observe(
                item["port"],
                source_stamp_ns=2_000 + index,
                now_monotonic=16.5,
            )
        recording = health.snapshot(process_running=True, now_monotonic=17.0)
        self.assertEqual(recording["state"], "recording")
        self.assertTrue(recording["healthy"])
        self.assertEqual(recording["missing_sources"], [])

        stale = health.snapshot(process_running=True, now_monotonic=20.0)
        self.assertEqual(stale["state"], "degraded")
        self.assertEqual(
            stale["stale_sources"],
            [item["port"] for item in COLLECTION_SOURCES],
        )
        self.assertEqual(
            stale["failure_reason"],
            "stale_sources:" + ",".join(item["port"] for item in COLLECTION_SOURCES),
        )

    def test_rosbag_exit_is_an_explicit_error(self) -> None:
        health = CollectionHealth()
        health.start("session-b", "/recordings/session-b", now_monotonic=1.0)
        status = health.snapshot(
            process_running=False,
            process_return_code=7,
            now_monotonic=2.0,
        )
        self.assertEqual(status["state"], "error")
        self.assertEqual(status["failure_reason"], "rosbag_exited:7")

    def test_decode_error_is_visible_in_collection_status(self) -> None:
        health = CollectionHealth(grace_sec=0.0)
        health.start("session-decode", "/recordings/session-decode")
        health.observe_error("rgb_frame", "unsupported distortion model")
        status = health.snapshot(process_running=True)
        self.assertEqual(status["state"], "degraded")
        self.assertEqual(status["source_errors"], ["rgb_frame"])
        self.assertEqual(
            status["failure_reason"], "source_decode_errors:rgb_frame"
        )
        self.assertIn(
            "unsupported distortion",
            status["sources"]["rgb_frame"]["error"],
        )

    def test_time_alignment_reports_software_sync_and_pair_skew(self) -> None:
        health = CollectionHealth(grace_sec=5.0, stale_sec=2.0)
        health.start("session-sync", "/recordings/session-sync", now_monotonic=1.0)
        base = 1_700_000_000_000_000_000
        stamps = {
            "lidar": base,
            "imu": base + 1_000_000,
            "rgb_frame": base + 2_000_000,
            "depth_frame": base + 3_000_000,
            "odom": base + 1_000_000,
        }
        for port, stamp in stamps.items():
            health.observe(
                port,
                source_stamp_ns=stamp,
                now_monotonic=6.0,
                receive_epoch_ns=base + 10_000_000,
                metadata=(
                    {
                        "frame_id": "camera_color_optical_frame",
                        "width": 640,
                        "height": 480,
                        "calibration_id": "g1-camera-a",
                    }
                    if port == "rgb_frame"
                    else {"frame_id": port}
                ),
            )

        status = health.snapshot(process_running=True, now_monotonic=6.5)
        alignment = status["time_alignment"]
        self.assertEqual(status["state"], "recording")
        self.assertTrue(alignment["alignment_ready"])
        self.assertEqual(alignment["clock_domain"], "ros_system_time")
        self.assertFalse(alignment["hardware_synchronized"])
        self.assertEqual(
            alignment["pairs"]["rgb_frame_lidar"]["nearest_skew_ms"]["p95"],
            2.0,
        )
        self.assertEqual(
            status["sources"]["rgb_frame"]["source_timestamp_coverage"], 1.0
        )
        self.assertEqual(
            status["sources"]["rgb_frame"]["metadata"]["calibration_id"],
            "g1-camera-a",
        )

    def test_time_alignment_degrades_on_skew_or_non_monotonic_stamp(self) -> None:
        health = CollectionHealth(grace_sec=1.0, stale_sec=2.0)
        health.start("session-skew", "/recordings/session-skew", now_monotonic=1.0)
        base = 1_700_000_000_000_000_000
        stamps = {
            "lidar": base,
            "imu": base + 1_000_000,
            "rgb_frame": base + 200_000_000,
            "depth_frame": base + 210_000_000,
            "odom": base + 1_000_000,
        }
        for port, stamp in stamps.items():
            health.observe(port, source_stamp_ns=stamp, now_monotonic=2.0)
        health.observe(
            "imu",
            source_stamp_ns=base - 1_000_000,
            now_monotonic=2.1,
        )

        status = health.snapshot(process_running=True, now_monotonic=2.5)
        self.assertEqual(status["state"], "degraded")
        self.assertFalse(status["time_alignment"]["alignment_ready"])
        self.assertIn(
            "imu:source_timestamp_out_of_order",
            status["time_alignment"]["reasons"],
        )
        self.assertIn(
            "rgb_frame_lidar:nearest_skew",
            status["time_alignment"]["reasons"],
        )
        self.assertTrue(status["failure_reason"].startswith("timestamp_alignment:"))

    def test_alignment_only_compares_overlapping_long_session_history(self) -> None:
        health = CollectionHealth(grace_sec=1.0, stale_sec=2.0)
        health.start("session-long", "/recordings/session-long", now_monotonic=1.0)
        base = 1_700_000_000_000_000_000
        for offset_ms in range(0, 60_000, 100):
            health.observe(
                "lidar",
                source_stamp_ns=base + offset_ms * 1_000_000,
                now_monotonic=2.0,
            )
        for offset_ms in range(0, 60_000, 5):
            health.observe(
                "imu",
                source_stamp_ns=base + offset_ms * 1_000_000,
                now_monotonic=2.0,
            )
        pair = health.snapshot(
            process_running=True, now_monotonic=2.5
        )["time_alignment"]["pairs"]["lidar_imu"]
        self.assertTrue(pair["ready"], pair)
        self.assertLessEqual(pair["nearest_skew_ms"]["p95"], 5.0)

    def test_sampler_emits_one_aligned_bundle_per_second(self) -> None:
        sampler = CollectionSampler(interval_sec=1.0)
        sampler.start()
        base = 1_700_000_000_000_000_000

        for port, offset_ms in (
            ("depth_frame", -60),
            ("lidar", -30),
            ("imu", -15),
            ("odom", -80),
        ):
            self.assertIsNone(
                sampler.observe(
                    port,
                    source_stamp_ns=base + offset_ms * 1_000_000,
                    message=port,
                    metadata={"port": port},
                    now_monotonic=10.0,
                )
            )
        bundle = sampler.observe(
            "rgb_frame",
            source_stamp_ns=base,
            message="rgb",
            metadata={"port": "rgb_frame"},
            now_monotonic=10.0,
        )
        self.assertEqual(set(bundle), {item["port"] for item in COLLECTION_SOURCES})
        self.assertIsNone(
            sampler.observe(
                "rgb_frame",
                source_stamp_ns=base + 100_000_000,
                message="rgb-too-soon",
                metadata=None,
                now_monotonic=10.1,
            )
        )

    def test_sampler_matches_imu_to_selected_lidar_not_rgb(self) -> None:
        sampler = CollectionSampler(interval_sec=1.0)
        sampler.start()
        base = 1_700_000_000_000_000_000
        for port, offset_ms in (
            ("depth_frame", 100),
            ("lidar", 50),
            ("imu", 51),
            ("odom", 0),
        ):
            sampler.observe(
                port,
                source_stamp_ns=base + offset_ms * 1_000_000,
                message=port,
                metadata={"port": port},
                now_monotonic=10.0,
            )

        bundle = sampler.observe(
            "rgb_frame",
            source_stamp_ns=base,
            message="rgb",
            metadata={"port": "rgb_frame"},
            now_monotonic=10.0,
        )

        self.assertIsNotNone(bundle)
        self.assertEqual(bundle["imu"]["source_stamp_ns"], base + 51_000_000)
        self.assertEqual(sampler.snapshot()["emitted_count"], 1)

    def test_sampler_exposes_rejection_reason(self) -> None:
        sampler = CollectionSampler(interval_sec=1.0)
        sampler.start()

        self.assertIsNone(
            sampler.observe(
                "rgb_frame",
                source_stamp_ns=1_700_000_000_000_000_000,
                message="rgb",
                metadata=None,
                now_monotonic=10.0,
            )
        )

        status = sampler.snapshot()
        self.assertEqual(status["emitted_count"], 0)
        self.assertEqual(status["last_rejection_reason"], "missing_depth_frame")
        self.assertEqual(status["rejections"], {"missing_depth_frame": 1})

    def test_health_separates_source_arrival_from_sampled_count(self) -> None:
        health = CollectionHealth()
        health.start("session-counts", "/recordings/session-counts")
        health.observe("lidar", source_stamp_ns=1_700_000_000_000_000_000)
        health.observe_sampled("lidar")

        source = health.snapshot(process_running=True)["sources"]["lidar"]
        self.assertEqual(source["count"], 1)
        self.assertEqual(source["sampled_count"], 1)

    def test_recording_summary_exposes_rosbag_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observed = {
                item["port"]: {"count": 100, "sampled_count": 10}
                for item in COLLECTION_SOURCES
            }
            entries = []
            for item in COLLECTION_SOURCES:
                entries.append(
                    {
                        "topic_metadata": {"name": item["record_topic"]},
                        "message_count": 2 if item["port"] == "imu" else 10,
                    }
                )
            entries.append(
                {
                    "topic_metadata": {
                        "name": COLLECTION_EVENT_TOPICS[0]["topic"]
                    },
                    "message_count": 3,
                }
            )
            Path(temporary, "metadata.yaml").write_text(
                "rosbag2_bagfile_information:\n"
                "  message_count: 42\n"
                "  duration:\n"
                "    nanoseconds: 10000000000\n"
                "  topics_with_message_count:\n"
                + "".join(
                    "  - topic_metadata:\n"
                    f"      name: {entry['topic_metadata']['name']}\n"
                    f"    message_count: {entry['message_count']}\n"
                    for entry in entries
                ),
                encoding="utf-8",
            )
            summary = read_rosbag_recording_summary(temporary, observed)
            self.assertFalse(summary["healthy"])
            self.assertEqual(
                summary["failure_reasons"], ["imu:recording_coverage"]
            )
            self.assertEqual(
                summary["topics"]["imu"]["recording_coverage"], 0.2
            )
            self.assertEqual(
                summary["topics"]["imu"]["source_observed_count"], 100
            )
            self.assertEqual(
                summary["events"]["sensor_rejection"]["recorded_count"], 3
            )

    def test_empty_rosbag_is_never_reported_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "metadata.yaml").write_text(
                "rosbag2_bagfile_information:\n"
                "  message_count: 0\n"
                "  duration:\n"
                "    nanoseconds: 0\n"
                "  topics_with_message_count: []\n",
                encoding="utf-8",
            )
            observed = {
                item["port"]: {"count": 12, "sampled_count": 0}
                for item in COLLECTION_SOURCES
            }

            summary = read_rosbag_recording_summary(temporary, observed)

            self.assertFalse(summary["healthy"])
            self.assertEqual(summary["failure_reasons"], ["recording_empty"])
            self.assertEqual(
                summary["topics"]["lidar"]["source_observed_count"], 12
            )
            self.assertEqual(summary["topics"]["lidar"]["sampled_count"], 0)

    def test_finalize_writes_receipt_and_only_renames_complete_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            partial = root / "session.partial"
            final = root / "session"
            partial.mkdir()
            (partial / "session_0.mcap").write_bytes(b"mcap-fixture")

            result = finalize_collection_session(
                str(partial),
                str(final),
                {"schema": "receipt.v1", "state": "complete"},
                storage_complete=True,
            )

            self.assertFalse(partial.exists())
            self.assertTrue((final / "session_0.mcap").is_file())
            self.assertTrue((final / "collection.json").is_file())
            self.assertEqual(result["directory"], str(final))

            failed_partial = root / "failed.partial"
            failed_final = root / "failed"
            failed_partial.mkdir()
            failed = finalize_collection_session(
                str(failed_partial),
                str(failed_final),
                {"schema": "receipt.v1", "state": "failed"},
                storage_complete=False,
            )
            self.assertTrue(failed_partial.is_dir())
            self.assertFalse(failed_final.exists())
            self.assertEqual(failed["directory"], str(failed_partial))


if __name__ == "__main__":
    unittest.main()
