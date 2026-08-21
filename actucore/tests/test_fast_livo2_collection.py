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
    COLLECTION_SOURCES,
    CollectionHealth,
    finalize_collection_session,
    normalize_collection_directory,
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
            command[7:], [item["topic"] for item in COLLECTION_SOURCES]
        )
        self.assertEqual(COLLECTION_SOURCES[0]["topic"], "/ubuntu/navigation/lidar")
        rgb_v2 = next(
            item for item in COLLECTION_SOURCES if item["port"] == "rgb_v2"
        )
        self.assertEqual(rgb_v2["topic"], "/ubuntu/navigation/camera/rgb")

    def test_health_exposes_normal_and_missing_source_failure(self) -> None:
        health = CollectionHealth(grace_sec=5.0, stale_sec=2.0)
        health.start("session-a", "/recordings/session-a", now_monotonic=10.0)

        starting = health.snapshot(process_running=True, now_monotonic=11.0)
        self.assertEqual(starting["state"], "starting")
        self.assertIsNone(starting["failure_reason"])

        for index, item in enumerate(COLLECTION_SOURCES):
            if item["port"] == "rgb_v2":
                continue
            health.observe(
                item["port"],
                source_stamp_ns=1_000 + index,
                now_monotonic=12.0,
            )
        degraded = health.snapshot(process_running=True, now_monotonic=16.0)
        self.assertEqual(degraded["state"], "degraded")
        self.assertEqual(degraded["missing_sources"], ["rgb_v2"])
        self.assertEqual(
            degraded["failure_reason"], "missing_sources:rgb_v2"
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

    def test_time_alignment_reports_software_sync_and_pair_skew(self) -> None:
        health = CollectionHealth(grace_sec=5.0, stale_sec=2.0)
        health.start("session-sync", "/recordings/session-sync", now_monotonic=1.0)
        base = 1_700_000_000_000_000_000
        stamps = {
            "lidar": base,
            "imu": base + 1_000_000,
            "rgb_v2": base + 2_000_000,
            "depth": base + 3_000_000,
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
                    if port == "rgb_v2"
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
            alignment["pairs"]["rgb_v2_lidar"]["nearest_skew_ms"]["p95"],
            2.0,
        )
        self.assertEqual(
            status["sources"]["rgb_v2"]["source_timestamp_coverage"], 1.0
        )
        self.assertEqual(
            status["sources"]["rgb_v2"]["metadata"]["calibration_id"],
            "g1-camera-a",
        )

    def test_time_alignment_degrades_on_skew_or_non_monotonic_stamp(self) -> None:
        health = CollectionHealth(grace_sec=1.0, stale_sec=2.0)
        health.start("session-skew", "/recordings/session-skew", now_monotonic=1.0)
        base = 1_700_000_000_000_000_000
        stamps = {
            "lidar": base,
            "imu": base + 1_000_000,
            "rgb_v2": base + 50_000_000,
            "depth": base + 52_000_000,
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
            "rgb_v2_lidar:nearest_skew",
            status["time_alignment"]["reasons"],
        )
        self.assertTrue(status["failure_reason"].startswith("timestamp_alignment:"))

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
