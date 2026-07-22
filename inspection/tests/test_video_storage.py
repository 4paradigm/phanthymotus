from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


INSPECTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSPECTION_ROOT))

from plugins.videoinspector import VideoInspectorPlugin  # noqa: E402
from plugins.videoinspector.runtime import VideoFramePump, VideoRecorderRuntime  # noqa: E402
from storage.ledger import SegmentLedger  # noqa: E402
from storage.models import SegmentState  # noqa: E402
from storage.video_writer import VideoFragmentStore, reconcile_video_store  # noqa: E402


class VideoStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.ledger = SegmentLedger(self.root / "state" / "ledger.sqlite3")

    def tearDown(self) -> None:
        self.ledger.close()
        self.tempdir.cleanup()

    def make_store(self) -> VideoFragmentStore:
        return VideoFragmentStore(
            data_root=self.root / "data",
            ledger=self.ledger,
            card_id="videoinspector",
            instance_id="camera-1",
            input_topic="/robot/camera/rgb",
            session_id="session-1",
            device_id="g1-sh",
            encoder="nvv4l2h264enc",
            target_bitrate_kbps=4000,
        )

    def test_closed_fragment_is_atomically_finalized_and_registered(self) -> None:
        store = self.make_store()
        location = store.create_location(0, first_source_stamp_ns=2_000_000_000)
        Path(location).write_bytes(b"fake-closed-mp4")
        store.note_frame(source_stamp_ns=2_100_000_000, receive_monotonic_ns=9_000_000_000, dropped=2)

        metadata = store.finalize_location(location)

        final_path = Path(location[:-5])
        metadata_path = final_path.with_suffix(".json")
        self.assertTrue(final_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertFalse(Path(location).exists())
        self.assertEqual(1, metadata["samples_or_frames"])
        self.assertEqual(2, metadata["dropped_before_writer"])
        record = self.ledger.get(metadata["segment_id"])
        assert record is not None
        self.assertEqual(SegmentState.FINALIZED.value, record["state"])
        self.assertEqual(str(final_path), record["local_path"])
        relative = final_path.relative_to(self.root / "data")
        self.assertEqual("video-inspector", relative.parts[0])
        self.assertTrue(relative.parts[1].startswith("camera-rgb--"))
        self.assertRegex(relative.parts[2], r"^utc-hour=\d{4}-\d{2}-\d{2}T\d{2}Z$")
        self.assertRegex(relative.name, r"^\d{8}T\d{6}\.\d{9}Z--\d{6}\.mp4$")
        self.assertEqual("videoinspector", metadata["card_id"])
        self.assertEqual("camera-1", metadata["instance_id"])

    def test_valid_interrupted_fragment_can_be_recovered(self) -> None:
        store = self.make_store()
        location = store.create_location(4)
        Path(location).write_bytes(b"validated-mp4")

        stats = reconcile_video_store(self.root / "data", self.ledger, validator=lambda _path: True)

        self.assertEqual(1, stats["parts_recovered"])
        metadata_path = next((self.root / "data").rglob("*.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(metadata["recovered_after_unclean_shutdown"])
        self.assertFalse(metadata["source_stamp_valid"])

    def test_invalid_interrupted_fragment_is_preserved_as_corrupt(self) -> None:
        store = self.make_store()
        location = store.create_location(5)
        Path(location).write_bytes(b"broken")

        stats = reconcile_video_store(self.root / "data", self.ledger, validator=lambda _path: False)

        self.assertEqual(1, stats["corrupt_parts"])
        self.assertTrue(Path(location + ".corrupt").exists())
        self.assertTrue(Path(location + ".corrupt.json").exists())

    def test_terminal_pipeline_error_aborts_before_drain_and_preserves_part(self) -> None:
        store = self.make_store()
        location = store.create_location(6)
        Path(location).write_bytes(b"incomplete-mp4")
        runtime = VideoRecorderRuntime(
            executor=object(),
            store=store,
            instance_id="camera-1",
            input_topic="/robot/camera/rgb",
            encoder="nvv4l2h264enc",
            target_bitrate_kbps=4000,
            segment_seconds=60,
            max_segment_bytes=64 * 1024 * 1024,
            max_fps=15,
            queue_frames=8,
            shutdown_timeout_seconds=1,
        )
        runtime._pipeline = object()
        runtime._appsrc = object()
        runtime.last_error = "native decoder failed"

        with self.assertRaisesRegex(RuntimeError, "native decoder failed"):
            runtime.stop()

        self.assertIsNone(runtime._pipeline)
        self.assertTrue(Path(location + ".corrupt").exists())
        reason = json.loads(Path(location + ".corrupt.json").read_text(encoding="utf-8"))
        self.assertEqual("CORRUPT", reason["state"])
        self.assertIn("native decoder failed", reason["reason"])
        self.assertFalse(Path(location[:-5] + ".open.json").exists())

    def test_video_frame_pump_enforces_format_fps_and_drains(self) -> None:
        consumed = []
        ticks = iter((1_000_000_000, 1_050_000_000, 1_200_000_000))
        pump = VideoFramePump(
            lambda jpeg, stamp, received, dropped: consumed.append((jpeg, stamp, received, dropped)),
            queue_frames=4,
            max_fps=10,
            monotonic_ns=lambda: next(ticks),
        )
        pump.start()
        message = SimpleNamespace(
            format="jpeg",
            data=[255, 216, 255, 217],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=3, nanosec=4)),
        )

        self.assertTrue(pump.submit_message(message))
        self.assertFalse(pump.submit_message(message))
        self.assertTrue(pump.submit_message(message))
        pump.stop(timeout=2)

        self.assertEqual(2, len(consumed))
        self.assertEqual(3_000_000_004, consumed[0][1])
        self.assertEqual(1, pump.stats()["rate_limited"])
        self.assertEqual(1, pump.stats()["dropped"])

    def test_video_frame_pump_rejects_repeated_invalid_jpeg_payloads(self) -> None:
        pump = VideoFramePump(
            lambda *_args: None,
            queue_frames=2,
            max_fps=15,
        )
        pump.start()
        broken = SimpleNamespace(format="jpeg", data=b"not-a-jpeg")

        self.assertFalse(pump.submit_message(broken))
        self.assertFalse(pump.submit_message(broken))
        self.assertFalse(pump.submit_message(broken))
        pump.stop(timeout=2)

        stats = pump.stats()
        self.assertEqual(3, stats["invalid_payload"])
        self.assertEqual(3, stats["consecutive_invalid"])
        self.assertIn("invalid JPEG payload", stats["last_error"])

    def test_video_runtime_reports_first_frame_timeout_and_stream_stall(self) -> None:
        now_ns = [12_000_000_000]
        runtime = VideoRecorderRuntime(
            executor=object(),
            store=self.make_store(),
            instance_id="camera-health",
            input_topic="/robot/camera/rgb",
            encoder="nvv4l2h264enc",
            target_bitrate_kbps=4000,
            segment_seconds=60,
            max_segment_bytes=64 * 1024 * 1024,
            max_fps=15,
            queue_frames=8,
            shutdown_timeout_seconds=15,
            input_start_timeout_seconds=10,
            input_stall_timeout_seconds=5,
            monotonic_ns=lambda: now_ns[0],
        )
        runtime._started_ns = 1_000_000_000

        missing = runtime.stats()
        self.assertTrue(missing["input_failed"])
        self.assertEqual("input_start_timeout", missing["error_kind"])
        self.assertEqual("stalled", missing["input_state"])

        runtime.pump.last_received_ns = 11_500_000_000
        healthy = runtime.stats()
        self.assertFalse(healthy.get("input_failed", False))
        self.assertEqual("healthy", healthy["input_state"])

        now_ns[0] = 17_000_000_000
        stalled = runtime.stats()
        self.assertTrue(stalled["input_failed"])
        self.assertEqual("input_stalled", stalled["error_kind"])
        self.assertGreaterEqual(stalled["last_frame_age_seconds"], 5)

    def test_instance_runtime_error_survives_ledger_reopen(self) -> None:
        self.ledger.set_instance_state(
            card_id="videoinspector",
            instance_id="camera-error",
            input_topic="/robot/camera/rgb",
            desired_state="idle",
            auto_resume=False,
            session_id="session-error",
            config={},
        )
        self.ledger.set_instance_error(
            card_id="videoinspector",
            instance_id="camera-error",
            runtime_state="degraded",
            last_error="JPEG input stalled",
            error_kind="input_stalled",
        )
        ledger_path = self.root / "state" / "ledger.sqlite3"
        self.ledger.close()
        self.ledger = SegmentLedger(ledger_path)

        saved = self.ledger.list_instance_states(card_id="videoinspector")[0]
        self.assertEqual("degraded", saved["runtime_state"])
        self.assertEqual("input_stalled", saved["error_kind"])
        self.assertEqual("JPEG input stalled", saved["last_error"])
        self.assertGreater(saved["error_at_ns"], 0)

    def test_jetson_pipeline_uses_only_explicit_hardware_codec(self) -> None:
        runtime = VideoRecorderRuntime(
            executor=object(),
            store=self.make_store(),
            instance_id="camera-1",
            input_topic="/robot/camera/rgb",
            encoder="nvv4l2h264enc",
            target_bitrate_kbps=4000,
            segment_seconds=60,
            max_segment_bytes=64 * 1024 * 1024,
            max_fps=15,
            queue_frames=8,
            shutdown_timeout_seconds=15,
        )

        pipeline = runtime._pipeline_description()

        self.assertIn("nvjpegdec", pipeline)
        self.assertIn("nvvidconv", pipeline)
        self.assertIn("nvv4l2h264enc bitrate=4000000", pipeline)
        self.assertNotIn("x264enc", pipeline)
        self.assertIn("splitmuxsink", pipeline)
        self.assertIn("max-size-bytes=67108864", pipeline)

    def test_reboot_defaults_video_to_explicit_resume(self) -> None:
        config = {
            "runtime_mode": "ros2-gstreamer",
            "data_root": str(self.root / "plugin-data"),
            "state_root": str(self.root / "plugin-state"),
        }
        plugin = VideoInspectorPlugin(config, executor=object())
        assert plugin._ledger is not None
        plugin._ledger.set_instance_state(
            card_id="videoinspector",
            instance_id="canvas-camera",
            input_topic="/robot/camera/rgb",
            desired_state="recording",
            auto_resume=False,
            session_id="session-before-reboot",
            config={"auto_resume_after_reboot": False},
        )
        plugin.shutdown()

        restarted = VideoInspectorPlugin(config, executor=object())
        info = restarted.dispatch("videoinspector", {"action": "info", "instance_id": "canvas-camera"})

        assert info is not None
        self.assertEqual("degraded", info["state"])
        self.assertTrue(info["resume_required"])
        self.assertEqual("unclean_shutdown", info["error_kind"])
        self.assertIn("interrupted", info["last_error"])
        restarted.shutdown()


if __name__ == "__main__":
    unittest.main()
