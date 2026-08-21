from __future__ import annotations

import json
import struct
import sys
import tempfile
import time
import unittest
import zlib
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

from g1_fast_livo2.camera_depth_v2 import (  # noqa: E402
    InvalidCameraDepthV2,
    decode as decode_depth,
    encode as encode_depth,
)
from g1_fast_livo2.camera_rgb_v2 import (  # noqa: E402
    InvalidCameraRgbV2,
    decode,
    encode,
)
from plugins.navigation.mapping.collection_postprocess import (  # noqa: E402
    CollectionPreviewWorker,
    CollectionPostprocessManager,
    LiveCollectionSynchronizer,
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


def depth_driver_metadata(stamp: int = 1_700_000_000_000_000_000) -> dict:
    identity = np.eye(4).reshape(-1).tolist()
    intrinsics = {
        "width": 2,
        "height": 2,
        "distortion_model": "plumb_bob",
        "k": [100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0],
        "d": [0.0] * 5,
    }
    transform = {
        "source_frame": "camera_depth_optical_frame",
        "target_frame": "camera_color_optical_frame",
        "convention": "target_from_source",
        "matrix_row_major": identity,
    }
    return {
        "schema": "phanthy.sensor.camera_depth.v2",
        "header": {
            "stamp_ns": stamp,
            "frame_id": "camera_depth_optical_frame",
        },
        "timing": {
            "source_stamp_ns": stamp,
            "driver_receive_stamp_ns": stamp + 1_000_000,
            "clock_domain": "ros_system_time",
        },
        "sequence": 8,
        "image": {
            "encoding": "z16_le",
            "width": 2,
            "height": 2,
            "step_bytes": 4,
            "payload_size": 8,
            "unit": "meter",
            "depth_scale_m": 0.001,
            "aligned_to_rgb": False,
        },
        "calibration": {
            "calibration_id": "sha256:test",
            **intrinsics,
            "depth_scale_m": 0.001,
            "aligned_to_rgb": False,
            "depth_to_rgb": transform,
            "rgb_intrinsics": intrinsics,
            "lidar_to_camera": {
                "status": "factory_nominal",
                "transform": {
                    **transform,
                    "source_frame": "livox_frame",
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


def decode_grayscale16_png(payload: bytes) -> tuple[int, int, np.ndarray]:
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(payload):
        size = struct.unpack_from(">I", payload, offset)[0]
        name = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + size]
        offset += 12 + size
        if name == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(
                ">IIBBBBB", data
            )
            assert (bit_depth, color_type) == (16, 0)
        elif name == b"IDAT":
            compressed.extend(data)
        elif name == b"IEND":
            break
    rows = zlib.decompress(bytes(compressed))
    stride = width * 2 + 1
    assert len(rows) == stride * height
    image = np.vstack(
        [
            np.frombuffer(rows[row * stride + 1 : (row + 1) * stride], dtype=">u2")
            for row in range(height)
        ]
    )
    return width, height, image


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

        inverse = driver_metadata()
        inverse["calibration"]["distortion_model"] = (
            "realsense_inverse_brown_conrady"
        )
        inverse["calibration"]["d"] = [0.01, -0.01, 0.0, 0.0, 0.0]
        decoded_inverse, _ = decode(encode(inverse, jpeg))
        self.assertEqual(
            decoded_inverse["intrinsics"]["distortion_model"],
            "realsense_inverse_brown_conrady",
        )

        with self.assertRaisesRegex(InvalidCameraRgbV2, "JPEG"):
            encode(driver_metadata(), b"not-a-jpeg")

        with self.assertRaisesRegex(InvalidCameraRgbV2, "magic"):
            decode(b"CRGB" + payload[4:])

    def test_camera_depth_v2_round_trip_and_rejects_scale_mismatch(self) -> None:
        raw_depth = np.asarray((100, 200, 300, 400), dtype="<u2").tobytes()
        payload = encode_depth(depth_driver_metadata(), raw_depth)
        decoded, depth = decode_depth(payload)
        self.assertEqual(decoded["calibration_id"], "sha256:test")
        self.assertEqual(decoded["depth_scale_m"], 0.001)
        self.assertEqual(depth, raw_depth)

        invalid = depth_driver_metadata()
        invalid["calibration"]["depth_scale_m"] = 0.002
        with self.assertRaisesRegex(InvalidCameraDepthV2, "scales disagree"):
            encode_depth(invalid, raw_depth)

    def test_annotation_outputs_session_id_nearest_point_and_distance(self) -> None:
        stamp = metadata()["source_stamp_ns"]
        result = annotate_frame(
            {
                "image_id": "frame-00000001",
                "image_path": "rgb/frame-00000001.jpg",
                "stamp_ns": stamp,
                "metadata": metadata(stamp),
            },
            {
                "stamp_ns": stamp,
                "frame_id": "livox_frame",
                "lidar_id": f"lidar-{stamp:019d}",
                "lidar_path": f"lidar/lidar-{stamp:019d}.pcd",
                "points": synthetic_points(),
            },
            {"stamp_ns": stamp, "gravity": np.asarray((0.0, 0.0, -9.81))},
            {"stamp_ns": stamp, "t_map_base": np.eye(4)},
            SessionTracker(),
            0,
            {
                "stamp_ns": stamp,
                "frame_id": "camera_depth_optical_frame",
                "depth_id": f"depth-{stamp:019d}",
                "depth_path": f"depth/depth-{stamp:019d}.png",
                "metadata": {
                    "width": 2,
                    "height": 2,
                    "encoding": "z16_le",
                    "depth_scale_m": 0.001,
                    "aligned_to_rgb": False,
                    "receive_stamp_ns": stamp + 1_000_000,
                },
            },
        )
        self.assertEqual(result["status"], "valid", result)
        self.assertTrue(result["obstacles"])
        obstacle = result["obstacles"][0]
        self.assertEqual(obstacle["obstacle_id"], "obs-000001")
        point = obstacle["nearest_point_camera_m"]
        expected = (point["x"] ** 2 + point["y"] ** 2 + point["z"] ** 2) ** 0.5
        self.assertAlmostEqual(obstacle["distance_m"], expected, places=5)
        self.assertAlmostEqual(
            obstacle["distance_ground_truth_m"], expected, places=5
        )
        self.assertEqual(result["lidar_frame_id"], "livox_frame")
        self.assertEqual(result["camera_parameters"]["equivalent_focal_length_px"], 100.0)
        self.assertAlmostEqual(
            result["distance_ground_truth"]["nearest_obstacle_distance_m"],
            expected,
            places=5,
        )
        self.assertGreaterEqual(obstacle["image_pixel"]["x"], 0)
        self.assertLess(obstacle["image_pixel"]["x"], metadata()["width"])
        self.assertGreaterEqual(obstacle["image_pixel"]["y"], 0)
        self.assertLess(obstacle["image_pixel"]["y"], metadata()["height"])

    def test_live_preview_reassembles_sampled_topics_and_renders_in_background(self) -> None:
        stamp = metadata()["source_stamp_ns"]
        synchronizer = LiveCollectionSynchronizer()
        synchronizer.update_session("session-a")
        records = [
            {
                "kind": "lidar",
                "stamp_ns": stamp + 10_000_000,
                "points": synthetic_points(),
            },
            {
                "kind": "imu",
                "stamp_ns": stamp + 11_000_000,
                "gravity": np.asarray((0.0, 0.0, -9.81)),
            },
            {
                "kind": "depth_v2",
                "stamp_ns": stamp + 20_000_000,
                "metadata": {"source_stamp_ns": stamp + 20_000_000},
                "depth": b"depth",
            },
            {"kind": "odom", "stamp_ns": stamp, "t_map_base": np.eye(4)},
            {
                "kind": "rgb_v2",
                "stamp_ns": stamp,
                "metadata": metadata(stamp),
                "jpeg": b"jpeg",
            },
        ]
        ready = None
        for record in records:
            ready = synchronizer.observe(record) or ready
        self.assertIsNotNone(ready)
        frame_number, bundle = ready
        self.assertEqual(frame_number, 1)
        self.assertEqual(set(bundle), {"rgb_v2", "depth_v2", "lidar", "imu", "odom"})

        def fake_renderer(value, number, tracker):
            self.assertIs(value, bundle)
            self.assertEqual(number, 1)
            self.assertIsInstance(tracker, SessionTracker)
            return b"\xff\xd8preview\xff\xd9", {
                "status": "valid",
                "obstacles": [{"distance_m": 1.25}],
            }

        worker = CollectionPreviewWorker(fake_renderer)
        worker.submit(frame_number, bundle)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if worker.snapshot()["frame_number"] == 1:
                break
            time.sleep(0.01)
        preview = worker.snapshot()
        self.assertEqual(preview["jpeg"], b"\xff\xd8preview\xff\xd9")
        self.assertEqual(preview["annotation_status"], "valid")
        self.assertEqual(preview["obstacle_count"], 1)
        self.assertIsNone(preview["failure_reason"])

        synchronizer.update_session("session-b")
        self.assertEqual(synchronizer.session_id, "session-b")

    def test_processor_writes_one_json_per_image_and_atomic_manifest(self) -> None:
        stamp = metadata()["source_stamp_ns"]
        lidar_stamp = stamp + 50_000_000
        imu_stamp = lidar_stamp + 1_000_000
        depth_stamp = stamp + 40_000_000
        raw_depth = np.asarray((100, 200, 300, 400), dtype="<u2").tobytes()
        records = [
            {
                "kind": "lidar",
                "stamp_ns": lidar_stamp,
                "frame_id": "livox_frame",
                "points": synthetic_points(),
            },
            {
                "kind": "imu",
                "stamp_ns": imu_stamp,
                "gravity": np.asarray((0.0, 0.0, -9.81)),
            },
            {"kind": "odom", "stamp_ns": stamp, "t_map_base": np.eye(4)},
            {
                "kind": "depth_v2",
                "stamp_ns": depth_stamp,
                "frame_id": "camera_depth_optical_frame",
                "metadata": {
                    "source_stamp_ns": depth_stamp,
                    "receive_stamp_ns": depth_stamp + 1_000_000,
                    "frame_id": "camera_depth_optical_frame",
                    "calibration_id": "sha256:test",
                    "width": 2,
                    "height": 2,
                    "encoding": "z16_le",
                    "step_bytes": 4,
                    "depth_scale_m": 0.001,
                    "aligned_to_rgb": False,
                },
                "depth": raw_depth,
            },
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
            self.assertEqual(manifest["lidar_frames"], 1)
            self.assertEqual(manifest["depth_frames"], 1)
            self.assertTrue((session / "derived" / "rgb" / "frame-00000001.jpg").is_file())
            lidar_id = f"lidar-{lidar_stamp:019d}"
            lidar_path = session / "derived" / "lidar" / f"{lidar_id}.pcd"
            self.assertTrue(lidar_path.is_file())
            lidar_payload = lidar_path.read_bytes()
            self.assertIn(b"FIELDS x y z\n", lidar_payload[:256])
            lidar_header, lidar_binary = lidar_payload.split(b"DATA binary\n", 1)
            self.assertIn(
                f"POINTS {len(synthetic_points())}\n".encode("ascii"), lidar_header
            )
            self.assertEqual(len(lidar_binary), len(synthetic_points()) * 3 * 4)
            depth_id = f"depth-{depth_stamp:019d}"
            depth_path = session / "derived" / "depth" / f"{depth_id}.png"
            width, height, decoded_depth = decode_grayscale16_png(
                depth_path.read_bytes()
            )
            self.assertEqual((width, height), (2, 2))
            np.testing.assert_array_equal(
                decoded_depth,
                np.asarray(((100, 200), (300, 400)), dtype=np.uint16),
            )
            frame_path = session / "derived" / "frames" / "frame-00000001.json"
            self.assertTrue(frame_path.is_file())
            frame = json.loads(frame_path.read_text(encoding="utf-8"))
            self.assertEqual(frame["schema"], "phanthy.navigation.obstacle_frame.v1")
            self.assertEqual(frame["lidar_id"], lidar_id)
            self.assertEqual(frame["lidar_path"], f"lidar/{lidar_id}.pcd")
            self.assertEqual(frame["depth_id"], depth_id)
            self.assertEqual(frame["depth_path"], f"depth/{depth_id}.png")
            self.assertEqual(frame["timestamps_ns"]["image_source"], stamp)
            self.assertEqual(frame["timestamps_ns"]["lidar_source"], lidar_stamp)
            self.assertEqual(frame["timestamps_ns"]["imu_source"], imu_stamp)
            self.assertEqual(frame["timestamps_ns"]["depth_source"], depth_stamp)
            self.assertEqual(frame["time_skew_ms"]["image_lidar"], 50.0)
            self.assertEqual(frame["time_skew_ms"]["lidar_imu"], 1.0)
            self.assertEqual(frame["time_skew_ms"]["image_depth"], 40.0)
            self.assertEqual(frame["depth_parameters"]["depth_scale_m"], 0.001)
            self.assertFalse(frame["depth_parameters"]["aligned_to_rgb"])
            self.assertEqual(
                frame["camera_parameters"]["equivalent_focal_length_px"], 100.0
            )
            self.assertEqual(
                manifest["artifacts"]["lidar"]["format"],
                "pcd_binary_xyz_float32_m",
            )
            self.assertEqual(
                manifest["artifacts"]["depth"]["format"],
                "png_grayscale_16bit_z16",
            )
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
