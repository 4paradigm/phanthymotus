from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGE = (
    ACTUCORE_ROOT
    / "plugins"
    / "navigation"
    / "runtime"
    / "g1_fast_livo2"
)
sys.path.insert(0, str(ACTUCORE_ROOT))
sys.path.insert(0, str(RUNTIME_PACKAGE))

from g1_fast_livo2.camera_rgb_v2 import (  # noqa: E402
    InvalidCameraRgbV2,
    decode,
    encode,
)
from plugins.navigation.mapping.collection_postprocess import (  # noqa: E402
    CollectionPostprocessManager,
    OfflineAnnotationProcessor,
    SessionTracker,
    annotate_frame,
)


def metadata(stamp: int = 1_700_000_000_000_000_000) -> dict:
    return {
        "schema": "phanthy.sensor.camera_rgb.v2",
        "source_stamp_ns": stamp,
        "receive_stamp_ns": stamp + 1_000_000,
        "frame_id": "camera_color_optical_frame",
        "calibration_id": "sha256:test",
        "width": 100,
        "height": 100,
        "encoding": "jpeg",
        "intrinsics": {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 50.0,
            "cy": 50.0,
            "distortion_model": "none",
            "coefficients": [],
        },
        "t_camera_lidar": np.eye(4).reshape(-1).tolist(),
        "t_base_camera": np.eye(4).reshape(-1).tolist(),
    }


def driver_metadata(stamp: int = 1_700_000_000_000_000_000) -> dict:
    identity = np.eye(4).reshape(-1).tolist()
    return {
        "schema": "phanthy.sensor.camera_rgb.v2",
        "header": {
            "stamp_ns": stamp,
            "frame_id": "camera_color_optical_frame",
        },
        "timing": {
            "source_stamp_ns": stamp,
            "driver_receive_stamp_ns": stamp + 1_000_000,
            "clock_domain": "ros_system_time",
        },
        "sequence": 7,
        "image": {
            "encoding": "jpeg",
            "width": 100,
            "height": 100,
            "payload_size": 4,
        },
        "calibration": {
            "calibration_id": "sha256:test",
            "width": 100,
            "height": 100,
            "distortion_model": "none",
            "k": [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0],
            "d": [],
            "lidar_to_camera": {
                "status": "factory_nominal",
                "transform": {
                    "source_frame": "livox_frame",
                    "target_frame": "camera_color_optical_frame",
                    "convention": "target_from_source",
                    "matrix_row_major": identity,
                },
            },
        },
    }


def synthetic_points() -> np.ndarray:
    ground = np.asarray(
        [(x, y, 0.0) for x in np.linspace(-1.0, 1.0, 8) for y in np.linspace(-1.0, 1.0, 8)],
        dtype=np.float64,
    )
    obstacle = np.asarray(
        [
            (x, y, z)
            for x in (-0.10, 0.0, 0.10)
            for y in (-0.10, 0.0, 0.10)
            for z in (1.90, 2.00, 2.10)
        ],
        dtype=np.float64,
    )
    return np.vstack((ground, obstacle))


class FakeReader:
    def __init__(self, records: list[dict]):
        self.records = records

    def count_images(self) -> int:
        return sum(record["kind"] == "rgb_v2" for record in self.records)

    def iter_records(self):
        return iter(self.records)


class NavigationCollectionPostprocessTest(unittest.TestCase):
    def test_camera_rgb_v2_round_trip_and_rejects_missing_calibration(self) -> None:
        jpeg = b"\xff\xd8\xff\xd9"
        payload = encode(driver_metadata(), jpeg)
        self.assertEqual(payload[:4], b"PSE2")
        decoded_metadata, decoded_jpeg = decode(payload)
        self.assertEqual(decoded_metadata["calibration_id"], "sha256:test")
        self.assertEqual(
            decoded_metadata["base_transform_source"],
            "actucore_g1_base_to_lidar+driver_lidar_to_camera",
        )
        self.assertEqual(len(decoded_metadata["t_base_camera"]), 16)
        self.assertEqual(decoded_jpeg, jpeg)

        invalid = driver_metadata()
        invalid["calibration"].pop("lidar_to_camera")
        with self.assertRaisesRegex(InvalidCameraRgbV2, "lidar_to_camera"):
            encode(invalid, jpeg)

        unsupported = driver_metadata()
        unsupported["calibration"]["distortion_model"] = (
            "inverse_brown_conrady"
        )
        with self.assertRaisesRegex(InvalidCameraRgbV2, "distortion_model"):
            encode(unsupported, jpeg)

        with self.assertRaisesRegex(InvalidCameraRgbV2, "JPEG"):
            encode(driver_metadata(), b"not-a-jpeg")

        with self.assertRaisesRegex(InvalidCameraRgbV2, "magic"):
            decode(b"CRGB" + payload[4:])

    def test_annotation_outputs_session_id_nearest_point_and_distance(self) -> None:
        stamp = metadata()["source_stamp_ns"]
        result = annotate_frame(
            {
                "image_id": "frame-00000001",
                "image_path": "rgb/frame-00000001.jpg",
                "stamp_ns": stamp,
                "metadata": metadata(stamp),
            },
            {"stamp_ns": stamp, "points": synthetic_points()},
            {"stamp_ns": stamp, "gravity": np.asarray((0.0, 0.0, -9.81))},
            {"stamp_ns": stamp, "t_map_base": np.eye(4)},
            SessionTracker(),
            0,
        )
        self.assertEqual(result["status"], "valid", result)
        self.assertTrue(result["obstacles"])
        obstacle = result["obstacles"][0]
        self.assertEqual(obstacle["obstacle_id"], "obs-000001")
        point = obstacle["nearest_point_camera_m"]
        expected = (point["x"] ** 2 + point["y"] ** 2 + point["z"] ** 2) ** 0.5
        self.assertAlmostEqual(obstacle["distance_m"], expected, places=5)

    def test_processor_writes_one_json_per_image_and_atomic_manifest(self) -> None:
        stamp = metadata()["source_stamp_ns"]
        records = [
            {"kind": "lidar", "stamp_ns": stamp, "points": synthetic_points()},
            {"kind": "imu", "stamp_ns": stamp, "gravity": np.asarray((0.0, 0.0, -9.81))},
            {"kind": "odom", "stamp_ns": stamp, "t_map_base": np.eye(4)},
            {
                "kind": "rgb_v2",
                "stamp_ns": stamp,
                "metadata": metadata(stamp),
                "jpeg": b"\xff\xd8\xff\xd9",
            },
        ]
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            session.mkdir()
            processor = OfflineAnnotationProcessor(lambda _: FakeReader(records))
            manifest = processor.process_session(
                session,
                lambda *args: events.append(args),
                lambda: None,
            )
            self.assertEqual(manifest["processed_images"], 1)
            self.assertTrue((session / "derived" / "rgb" / "frame-00000001.jpg").is_file())
            frame_path = session / "derived" / "frames" / "frame-00000001.json"
            self.assertTrue(frame_path.is_file())
            frame = json.loads(frame_path.read_text(encoding="utf-8"))
            self.assertEqual(frame["schema"], "phanthy.navigation.obstacle_frame.v1")
            self.assertFalse((session / "derived.partial").exists())
            self.assertEqual(events[-1][0], "finalizing")

    def test_processor_rejects_a_session_without_versioned_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            session.mkdir()
            processor = OfflineAnnotationProcessor(lambda _: FakeReader([]))
            with self.assertRaisesRegex(RuntimeError, "no_rgb_v2_images"):
                processor.process_session(session, lambda *args: None, lambda: None)

    def test_manager_pauses_for_runtime_and_resumes_after_stop(self) -> None:
        class FakeProcessor:
            def process_session(self, session, progress, wait_if_paused):
                progress("scanning", 0, 1, None)
                wait_if_paused()
                progress("processing", 1, 1, None)
                derived = session / "derived"
                derived.mkdir()
                manifest = {
                    "state": "complete",
                    "processed_images": 1,
                    "total_images": 1,
                }
                (derived / "manifest.json").write_text(json.dumps(manifest))
                return manifest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "ubuntu" / "2026-08-21" / "session-a"
            session.mkdir(parents=True)
            manager = CollectionPostprocessManager(
                str(root), processor_factory=FakeProcessor
            )
            self.assertFalse(
                manager.enqueue_receipt(
                    {
                        "storage_complete": True,
                        "directory": str(root.parent / "outside-session"),
                    }
                )
            )
            manager.set_runtime_active(True)
            self.assertTrue(
                manager.enqueue_receipt(
                    {
                        "state": "complete",
                        "storage_complete": True,
                        "directory": str(session),
                    }
                )
            )
            raw_status = manager.snapshot()
            self.assertEqual(raw_status["state"], "disabled")
            self.assertTrue(raw_status["healthy"])
            self.assertEqual(raw_status["last_receipt"]["state"], "complete")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if manager.snapshot()["postprocess"]["state"] == "paused":
                    break
                time.sleep(0.01)
            self.assertEqual(manager.snapshot()["postprocess"]["state"], "paused")
            manager.set_runtime_active(False)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if manager.snapshot()["postprocess"]["state"] == "complete":
                    break
                time.sleep(0.01)
            status = manager.snapshot()["postprocess"]
            self.assertEqual(status["state"], "complete", status)
            self.assertEqual(status["percent"], 100.0)
            journal = json.loads(
                (session / "postprocess.json").read_text(encoding="utf-8")
            )
            self.assertEqual(journal["state"], "complete")
            self.assertEqual(journal["processed_images"], 1)


if __name__ == "__main__":
    unittest.main()
