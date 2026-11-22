import json
import time
import tracemalloc
import unittest
import uuid
import zlib

import numpy as np

from plugins.cross_view_object_consistency import (
    CrossViewObjectConsistencyPlugin, Synchronizer, decode_depth,
)
from plugins.cross_view_geometry import calibration, compare
from utils.frame_source import capture_identity


def record(camera_id, x, focal):
    return dict(camera_id=camera_id, world_frame="bench", direction="camera_to_world",
                rectified=True, depth_registered=True, size=[640, 480], error_m=.003,
                K=[focal, 0, 320, 0, focal, 240, 0, 0, 1],
                T_world_camera=[1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])


def scene(points, camera, occluder=None):
    k, pose, _, _, _ = camera
    depth = np.full((480, 640), 2.0, np.float32)
    objects = []
    for point in points:
        xyz = pose[:3, :3].T @ (np.asarray(point) - pose[:3, 3])
        u, v = (k @ xyz)[:2] / xyz[2]
        if 8 <= u < 632 and 8 <= v < 472:
            objects.append(dict(name="block", confidence=.95, bbox=[u-5, v-5, u+5, v+5]))
            depth[int(v)-5:int(v)+6, int(u)-5:int(u)+6] = xyz[2]
    if occluder is not None:
        u, v, extent = occluder
        depth[v-extent:v+extent, u-extent:u+extent] = .7
    return objects, depth


def rig():
    return dict(camera_ids=["a", "b"], session_id="test",
                calibration={"a": record("a", 0, 420), "b": record("b", .2, 510)})


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.records = rig()["calibration"]
        self.cameras = [calibration(self.records[i], i) for i in "ab"]

    def run_scene(self, a, b, occluder=None):
        oa, da = scene(a, self.cameras[0])
        ob, db = scene(b, self.cameras[1], occluder)
        return compare([oa, ob], [da, db], self.cameras, .06, .25, .03, .35)

    def test_different_intrinsics_and_extrinsics(self):
        result = self.run_scene([(0, 0, 1), (.32, 0, 1)], [(.32, 0, 1), (0, 0, 1)])
        self.assertEqual(result["state"], "consistent")
        self.assertEqual([(m["a_id"], m["b_id"]) for m in result["matched_targets"]], [(0, 1), (1, 0)])
        self.assertLess(result["position_residual_m"], 1e-6)

    def test_known_offset(self):
        result = self.run_scene([(0, 0, 1)], [(.12, 0, 1)])
        self.assertEqual(result["state"], "inconsistent")
        self.assertAlmostEqual(result["position_residual_m"], .12)

    def test_close_same_class(self):
        self.assertEqual(self.run_scene([(0, 0, 1), (.025, 0, 1)],
                                       [(0, 0, 1), (.025, 0, 1)])["state"], "ambiguous")

    def test_full_occlusion_and_missed_detection(self):
        for occluder, reason in [(None, "not_detected_in_other_view"),
                                 ((218, 240, 4), "possible_occlusion")]:
            result = self.run_scene([(0, 0, 1)], [], occluder)
            self.assertEqual(result["state"], "insufficient_evidence")
            self.assertEqual(result["unmatched_targets"][0]["reason"], reason)
        result = self.run_scene([(-.57, 0, 1)], [])
        self.assertEqual(result["unmatched_targets"][0]["reason"], "outside_other_view")

    def test_partial_occlusion(self):
        result = self.run_scene([(0, 0, 1)], [(0, 0, 1)], (218, 240, 1))
        self.assertIn("mixed_target_depth", result["unmatched_targets"][-1]["reason"])

    def test_wrong_direction_and_invalid_rotation(self):
        bad = self.records["b"]
        bad["direction"] = "world_to_camera"
        with self.assertRaises(ValueError):
            calibration(bad, "b")
        bad["direction"] = "camera_to_world"
        bad["T_world_camera"][0] = -1
        with self.assertRaises(ValueError):
            calibration(bad, "b")

    def test_wrong_but_rigid_translation_is_detectable_in_fixture(self):
        oa, da = scene([(0, 0, 1)], self.cameras[0])
        ob, db = scene([(0, 0, 1)], self.cameras[1])
        self.records["b"]["T_world_camera"][3] = .33
        bad = calibration(self.records["b"], "b")
        outcome = compare([oa, ob], [da, db], [self.cameras[0], bad], .06, .25, .03, .35)
        self.assertEqual(outcome["state"], "inconsistent")

    def test_rotated_camera(self):
        pose = np.eye(4)
        pose[:3, :3] = [[.8, 0, .6], [0, 1, 0], [-.6, 0, .8]]
        pose[0, 3] = -.4
        self.records["b"]["T_world_camera"] = pose.flatten().tolist()
        self.cameras[1] = calibration(self.records["b"], "b")
        self.assertEqual(self.run_scene([(0, 0, 1)], [(0, 0, 1)])["state"], "consistent")


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.sync = Synchronizer({**rig(), "buffer_size": 2})
        self.sync.start_ns = 1_000_000_000

    def send(self, side, ns, session="test", depth=True):
        source = dict(frame_id=f"{'ab'[side]}@{session}", stamp_ns=ns)
        obj, grid = scene([(0, 0, 1)], self.sync.cameras[side])
        self.sync.ingest(side, "rgb", source, None, ns / 1e9)
        self.sync.ingest(side, "objects", source, dict(objects=obj, source_size=[640, 480]), ns / 1e9)
        if depth:
            self.sync.ingest(side, "depth", source, grid, ns / 1e9)

    def test_join_and_stale(self):
        self.send(0, 1_010_000_000)
        self.send(1, 1_020_000_000)
        result = self.sync.result(1.03)
        self.assertEqual(result["state"], "consistent")
        self.assertEqual(result["time_offset_ms"], 10)
        self.assertEqual(self.sync.result(3)["state"], "stale")

    def test_skew_session_and_missing_depth(self):
        self.send(0, 1_010_000_000)
        self.send(1, 1_020_000_000, session="old")
        self.assertEqual(self.sync.result(1.03)["reason"], "unverified_camera_or_session")
        self.send(1, 1_120_000_000, depth=False)
        self.assertEqual(self.sync.result(1.13)["reason"], "cross_view_time_offset_exceeded")
        self.send(0, 1_120_000_000)
        self.assertEqual(self.sync.result(1.13)["reason"], "camera_b_aligned_depth_unavailable")
        self.sync.cameras = None
        self.assertEqual(self.sync.result(1.13)["reason"], "missing_calibration")

    def test_capacity_duplicates_and_reordering(self):
        for ns in (1_010_000_000, 1_020_000_000, 1_030_000_000, 1_030_000_000, 1_025_000_000):
            self.send(0, ns)
        self.assertEqual(list(self.sync.buffers[0]["rgb"]), [1_020_000_000, 1_030_000_000])

    def test_bounded_decode(self):
        raw = np.full((480, 640), 1234, dtype="<u2").tobytes()
        self.assertAlmostEqual(float(decode_depth(zlib.compress(raw))[0, 0]), 1.234, places=6)
        for invalid in (raw[:-1], raw + b"00", raw * 20):
            with self.assertRaises(ValueError):
                decode_depth(zlib.compress(invalid))

    def test_mcp_config_and_ports(self):
        plugin = CrossViewObjectConsistencyPlugin({}, None)
        tool = plugin.get_tools()[0]
        self.assertEqual(len(tool["topic_in"]), 6)
        self.assertFalse(plugin.dispatch(tool["name"], {"action": "config", "rig_json": "{}"})["adapter_ok"])
        self.assertTrue(plugin.dispatch(tool["name"], {
            "action": "config", "rig_json": json.dumps(rig()),
        })["calibrated"])
        for _ in range(5):
            self.assertEqual(plugin.dispatch(tool["name"], {"action": "stop"})["state"], "idle")


def capture_rig(*, calibrated=True, clock_bound=True):
    config = rig()
    config.pop("session_id")
    config["device_ids"] = ["a" * 32, "b" * 32]
    for side, camera in enumerate("ab"):
        config["calibration"][camera].update(
            device_id=config["device_ids"][side], calibration_version="survey-v1")
    if not calibrated:
        config.pop("calibration")
    if clock_bound:
        config["time_error_ms"] = 5
    return config


def capture_source(side, ns, sequence, generation=1):
    camera = "ab"[side]
    session = f"{side * 100 + generation:032x}"
    tag = f"motus.capture/1|{camera}|{camera}_color_optical_frame|{session}|{sequence}|host_read|{camera * 32}"
    return {"stamp_ns": ns, "frame_id": tag}


class CaptureSourceTest(unittest.TestCase):
    def configure(self, **options):
        self.sync = Synchronizer(capture_rig(**options))
        self.sync.start_ns = self.sync.cutover_ns = 1_000_000_000

    def feed(self, side, ns, seq, generation=1, *, depth=True, source_side=None):
        source = capture_source(side if source_side is None else source_side, ns, seq, generation)
        camera = calibration(rig()["calibration"]["ab"[side]], "ab"[side])
        objects, grid = scene([(0, 0, 1)], camera)
        self.sync.ingest(side, "rgb", source, None, ns / 1e9)
        self.sync.ingest(side, "objects", source, dict(objects=objects, source_size=[640, 480]), ns / 1e9)
        if depth:
            self.sync.ingest(side, "depth", source, grid, ns / 1e9)

    def test_independent_sessions_and_qualified_depth(self):
        self.configure()
        for side in range(2):
            self.feed(side, 1_010_000_000, 1)
        result = self.sync.result(1.02)
        self.assertEqual(result["state"], "consistent")
        self.assertEqual(result["capability_level"], 2)
        self.assertNotEqual(*result["capture_sessions"])
        self.assertEqual([frame["camera_id"] for frame in result["capture_frames"]], ["a", "b"])

    def test_restart_flushes_pair_and_rejects_retired_session(self):
        self.configure()
        for side in range(2):
            self.feed(side, 1_010_000_000, 1)
        self.feed(0, 1_030_000_000, 1, generation=2)
        self.assertNotEqual(self.sync.result(1.03)["state"], "consistent")
        self.feed(0, 1_040_000_000, 2)  # Delayed retired session even with a higher stamp.
        self.feed(1, 1_045_000_000, 2)
        result = self.sync.result(1.05)
        self.assertEqual(result["state"], "consistent")
        self.assertEqual(result["capture_frames"][0]["capture_timestamp_ns"], 1_030_000_000)

    def test_offline_and_wrong_camera_reject_cached_pair_then_recover(self):
        self.configure()
        for side in range(2):
            self.feed(side, 1_010_000_000, 1)
        self.sync.ingest(0, "rgb", {"frame_id": "motus.capture/1|offline",
                                   "frame_valid": False, "stamp_ns": 0}, None, 1.02)
        self.assertEqual(self.sync.result(1.02)["reason"], "invalid_or_offline_capture_frame")
        self.feed(0, 1_025_000_000, 2, source_side=1)
        self.assertEqual(self.sync.result(1.03)["reason"], "capture_camera_or_device_mismatch")
        for side in range(2):
            self.feed(side, 1_035_000_000, 3)
        self.assertEqual(self.sync.result(1.04)["state"], "consistent")

    def test_levels_and_unverified_clock(self):
        self.configure(calibrated=False)
        for side in range(2):
            self.feed(side, 1_010_000_000, 1, depth=False)
        result = self.sync.result(1.02)
        self.assertEqual((result["capability_level"], result["reason"]), (0, "missing_calibration"))
        self.configure()
        for side in range(2):
            self.feed(side, 1_010_000_000, 1, depth=False)
        result = self.sync.result(1.02)
        self.assertEqual((result["capability_level"], result["epipolar_candidate_count"]), (1, 1))
        self.assertIsNone(result["position_residual_m"])
        self.configure(clock_bound=False)
        for side in range(2):
            self.feed(side, 1_010_000_000, 1)
        self.assertEqual(self.sync.result(1.02)["reason"], "capture_clock_error_unverified_or_exceeds_limit")

    def test_stale_b_then_fresh_frames_only(self):
        self.configure()
        for side in range(2):
            self.feed(side, 1_010_000_000, 1)
        self.feed(0, 2_020_000_000, 2)
        self.assertEqual(self.sync.result(2.03)["reason"], "camera_b_rgb_stream_stale")
        self.feed(1, 2_040_000_000, 1, generation=2)
        self.assertNotEqual(self.sync.result(2.04)["state"], "consistent")
        self.feed(0, 2_045_000_000, 3)
        self.assertEqual(self.sync.result(2.05)["state"], "consistent")

    def test_unmatched_calibration_versions_and_invalid_depth(self):
        config = capture_rig()
        config["calibration"]["b"]["calibration_version"] = "stale-survey"
        self.sync = Synchronizer(config)
        self.sync.start_ns = 1_000_000_000
        for side in range(2):
            self.feed(side, 1_010_000_000, 1)
        self.assertEqual(self.sync.result(1.02)["reason"], "invalid_calibration")
        self.configure()
        self.feed(0, 1_010_000_000, 1)
        self.feed(1, 1_010_000_000, 1, depth=False)
        source = capture_source(1, 1_010_000_000, 1)
        self.sync.ingest(1, "depth", source, np.zeros((480, 640), np.float32), 1.01)
        result = self.sync.result(1.02)
        self.assertEqual((result["capability_level"], result["reason"]), (1, "camera_b_depth_invalid"))

    def test_rgb_only_without_detectors_reports_observed_timing(self):
        config = capture_rig(calibrated=False, clock_bound=False)
        config["topics"] = {"a": {"rgb": "/a"}, "b": {"rgb": "/b"}}
        sync = Synchronizer(config)
        sync.start_ns = 1_000_000_000
        for side in range(2):
            stamp = 1_010_000_000 + 5_000_000 * side
            sync.ingest(side, "rgb", capture_source(side, stamp, 1), None, stamp / 1e9)
        result = sync.result(1.02)
        self.assertEqual(result["time_offset_ms"], 5)
        self.assertEqual(result["reason"], "camera_a_objects_not_connected")
        self.assertEqual(result["sync_status"], "unqualified")

    def test_replay_preserves_timing_and_independent_sessions(self):
        from tools.replay_cross_view import rebase_source

        tags = [capture_source(side, 100 + side, 7) for side in range(2)]
        run = uuid.uuid4()
        rebased = [rebase_source(row["frame_id"], row["stamp_ns"], 1000, run) for row in tags]
        self.assertEqual([stamp for _, stamp in rebased], [1100, 1101])
        parsed = [capture_identity(tag) for tag, _ in rebased]
        self.assertNotEqual(parsed[0]["capture_session_id"], parsed[1]["capture_session_id"])
        self.assertEqual(rebased[0], rebase_source(tags[0]["frame_id"], 100, 1000, run))
        self.assertEqual(rebase_source("motus.capture/1|offline", 0, 1000, run),
                         ("motus.capture/1|offline", 0))

    def test_equal_stamp_with_different_frame_sequence_does_not_join(self):
        self.configure()
        self.feed(1, 1_010_000_000, 1)
        self.sync.ingest(0, "rgb", capture_source(0, 1_010_000_000, 1), None, 1.01)
        self.sync.ingest(0, "objects", capture_source(0, 1_010_000_000, 2), {}, 1.01)
        self.assertEqual(self.sync.result(1.02)["reason"], "awaiting_synchronized_frames")


def benchmark(count=500):
    cameras = [calibration(rig()["calibration"][i], i) for i in "ab"]
    rng = np.random.default_rng(713)
    timings, residuals = [], []
    correct = incorrect = 0
    tracemalloc.start()
    for _ in range(count):
        a = [(x, float(rng.uniform(-.1, .1)), 1.) for x in (-.12, .15, .42)]
        b = [a[i] for i in (2, 0, 1)]
        oa, da = scene(a, cameras[0])
        ob, db = scene(b, cameras[1])
        before = time.perf_counter_ns()
        result = compare([oa, ob], [da, db], cameras, .06, .25, .03, .35)
        timings.append((time.perf_counter_ns() - before) / 1e6)
        for item in result["matched_targets"]:
            correct += (2, 0, 1)[item["b_id"]] == item["a_id"]
            incorrect += (2, 0, 1)[item["b_id"]] != item["a_id"]
            residuals.append(item["position_residual_m"])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(json.dumps(dict(expected=count * 3, correct=correct, incorrect=incorrect,
                          max_residual_m=max(residuals), median_ms=float(np.median(timings)),
                          p95_ms=float(np.percentile(timings, 95)), traced_peak_bytes=peak)))


if __name__ == "__main__":
    import sys

    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
