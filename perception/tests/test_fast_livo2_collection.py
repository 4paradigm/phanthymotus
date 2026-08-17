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

    def test_health_exposes_normal_and_missing_source_failure(self) -> None:
        health = CollectionHealth(grace_sec=5.0, stale_sec=2.0)
        health.start("session-a", "/recordings/session-a", now_monotonic=10.0)

        starting = health.snapshot(process_running=True, now_monotonic=11.0)
        self.assertEqual(starting["state"], "starting")
        self.assertIsNone(starting["failure_reason"])

        for index, item in enumerate(COLLECTION_SOURCES):
            if item["port"] == "camera_info":
                continue
            health.observe(
                item["port"],
                source_stamp_ns=1_000 + index,
                now_monotonic=12.0,
            )
        degraded = health.snapshot(process_running=True, now_monotonic=16.0)
        self.assertEqual(degraded["state"], "degraded")
        self.assertEqual(degraded["missing_sources"], ["camera_info"])
        self.assertEqual(
            degraded["failure_reason"], "missing_sources:camera_info"
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
