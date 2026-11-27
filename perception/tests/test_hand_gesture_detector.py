#!/usr/bin/env python3
"""Product-neutral unit tests for the opt-in hand gesture detector backend
(no GPU, TensorRT, network, or real model weights required)."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.hand_gesture_detector import (  # noqa: E402
    ALL_DETECTOR_GESTURES,
    HAGRID_CLASS_NAMES,
    HAGRID_TO_CANONICAL,
    TWO_HAND_GESTURES,
    HandGestureDetectorChannel,
    assign_hands_lr,
    pick_primary_person_index,
)
from plugins.vision_runtime import COCO_INDEX, LetterboxMeta  # noqa: E402


class _FakeVisionSession:
    """Stub VisionEngineSession emitting deterministic (1, N, 6) rows in frame coords."""

    def __init__(self, rows_by_call):
        self._rows_by_call = list(rows_by_call)
        self.calls = 0
        self.input_size = (640, 640)

    def infer(self, frame):
        h, w = frame.shape[:2]
        meta = LetterboxMeta(scale=1.0, pad_x=0.0, pad_y=0.0, orig_w=w, orig_h=h)
        rows = self._rows_by_call[min(self.calls, len(self._rows_by_call) - 1)]
        self.calls += 1
        arr = (
            np.asarray(rows, dtype=np.float32).reshape(1, len(rows), 6)
            if rows
            else np.zeros((1, 0, 6), dtype=np.float32)
        )
        return [arr], meta


class _FakePoseSession:
    """Stub pose engine returning one person with configurable wrists."""

    input_size = (640, 640)

    def infer(self, frame):
        h, w = frame.shape[:2]
        meta = LetterboxMeta(scale=1.0, pad_x=0.0, pad_y=0.0, orig_w=w, orig_h=h)
        row = np.zeros(57, dtype=np.float32)
        row[:6] = [150, 80, 490, 470, 0.95, 0]
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[COCO_INDEX["left_shoulder"]] = [390, 160, 0.9]
        kpts[COCO_INDEX["right_shoulder"]] = [250, 160, 0.9]
        kpts[COCO_INDEX["left_wrist"]] = [360, 240, 0.9]
        kpts[COCO_INDEX["right_wrist"]] = [280, 240, 0.9]
        row[6:] = kpts.reshape(-1)
        return [row.reshape(1, 1, 57)], meta


def _make_person(
    l_wrist=None,
    r_wrist=None,
    l_elbow=None,
    r_elbow=None,
    box=(100, 100, 500, 900),
    score=0.95,
):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[COCO_INDEX["left_shoulder"]] = [380, 260, 0.9]
    kpts[COCO_INDEX["right_shoulder"]] = [220, 260, 0.9]
    if l_elbow is not None:
        kpts[COCO_INDEX["left_elbow"]] = [*l_elbow, 0.9]
    if r_elbow is not None:
        kpts[COCO_INDEX["right_elbow"]] = [*r_elbow, 0.9]
    if l_wrist is not None:
        kpts[COCO_INDEX["left_wrist"]] = [*l_wrist, 0.9]
    if r_wrist is not None:
        kpts[COCO_INDEX["right_wrist"]] = [*r_wrist, 0.9]
    return {"id": 1, "score": score, "box": list(box), "keypoints": kpts, "verdict": {}}


class TestHandGestureDetector(unittest.TestCase):
    def test_vocabulary_completeness(self):
        self.assertEqual(len(HAGRID_CLASS_NAMES), 34)
        self.assertEqual(len(ALL_DETECTOR_GESTURES), 30)
        self.assertEqual(len(TWO_HAND_GESTURES), 7)
        mapped = {v for v in HAGRID_TO_CANONICAL.values() if v is not None}
        self.assertEqual(mapped, set(ALL_DETECTOR_GESTURES))
        self.assertEqual(HAGRID_TO_CANONICAL["two_up_inverted"], "two_up")
        self.assertEqual(HAGRID_TO_CANONICAL["peace_inverted"], "victory")
        self.assertEqual(HAGRID_TO_CANONICAL["stop_inverted"], "stop")
        self.assertIsNone(HAGRID_TO_CANONICAL["no_gesture"])

    def test_praying_hands_holy_do_not_swap(self):
        person = _make_person(
            l_elbow=(420, 500),
            l_wrist=(325, 420),
            r_elbow=(180, 500),
            r_wrist=(275, 420),
        )
        dets = [
            {"box": [305, 340, 355, 450], "conf": 0.93},
            {"box": [245, 340, 295, 450], "conf": 0.92},
        ]
        assigned = assign_hands_lr(dets, person, img_w=640)
        self.assertEqual(assigned["left"]["box"], [305, 340, 355, 450])
        self.assertEqual(assigned["right"]["box"], [245, 340, 295, 450])

    def test_crossed_arms_xsign_follows_wrists(self):
        person = _make_person(
            l_elbow=(410, 520),
            l_wrist=(220, 360),
            r_elbow=(190, 520),
            r_wrist=(380, 360),
        )
        dets = [
            {"box": [190, 310, 250, 390], "conf": 0.91},
            {"box": [350, 310, 410, 390], "conf": 0.90},
        ]
        assigned = assign_hands_lr(dets, person, img_w=640)
        self.assertEqual(assigned["left"]["box"], [190, 310, 250, 390])
        self.assertEqual(assigned["right"]["box"], [350, 310, 410, 390])

    def test_two_hand_gesture_keeps_single_hand_bboxes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = _make_person(l_wrist=(360, 240), r_wrist=(280, 240))
        hand_sess = _FakeVisionSession([[
            [330, 180, 400, 260, 0.94, 0],
            [240, 180, 310, 260, 0.93, 0],
        ]])
        gest_sess = _FakeVisionSession([[
            [240, 180, 400, 260, 0.96, 8],
        ]])
        channel = HandGestureDetectorChannel(gest_sess, hand_sess, adaptive_zoom=False)
        persons = [person]
        note = channel.update(persons, frame)
        self.assertTrue(note["ok"])
        self.assertEqual(note["found"], 2)
        self.assertEqual(person["hand_bboxes"]["left"], [330.0, 180.0, 400.0, 260.0])
        self.assertEqual(person["hand_bboxes"]["right"], [240.0, 180.0, 310.0, 260.0])
        self.assertEqual(person["hand_gestures"], {"left": "hand_heart", "right": "hand_heart"})
        self.assertAlmostEqual(person["hand_gesture_scores"]["left"], 0.96, places=3)
        self.assertAlmostEqual(person["hand_gesture_scores"]["right"], 0.96, places=3)

    def test_single_hand_plus_resting_no_gesture_hand(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = _make_person(l_wrist=(400, 380), r_wrist=(200, 200))
        hand_sess = _FakeVisionSession([[
            [170, 150, 230, 240, 0.95, 0],
            [370, 340, 430, 420, 0.88, 0],
        ]])
        gest_sess = _FakeVisionSession([[
            [170, 150, 230, 240, 0.94, 24],  # stop
            [370, 340, 430, 420, 0.85, 33],  # no_gesture
        ]])
        channel = HandGestureDetectorChannel(gest_sess, hand_sess, adaptive_zoom=False)
        channel.update([person], frame)
        self.assertEqual(person["hand_bboxes"]["right"], [170.0, 150.0, 230.0, 240.0])
        self.assertEqual(person["hand_bboxes"]["left"], [370.0, 340.0, 430.0, 420.0])
        self.assertEqual(person["hand_gestures"], {"left": None, "right": "stop"})
        self.assertIsNone(person["hand_gesture_scores"]["left"])
        self.assertAlmostEqual(person["hand_gesture_scores"]["right"], 0.94, places=3)

    def test_zero_person_synthesizes_person_on_photo(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand_sess = _FakeVisionSession([[[420, 180, 500, 290, 0.92, 0]]])
        gest_sess = _FakeVisionSession([[[420, 180, 500, 290, 0.91, 18]]])
        channel = HandGestureDetectorChannel(gest_sess, hand_sess, adaptive_zoom=False)
        persons = []
        channel.update(persons, frame, synthesize_person=True)
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0]["hand_gestures"], {"left": "ok", "right": None})
        self.assertEqual(persons[0]["hand_bboxes"]["right"], [])

    def test_multi_person_promotes_gesturing_person_first(self):
        bg = _make_person(l_wrist=(60, 60), r_wrist=(40, 60), box=(20, 20, 90, 180), score=0.98)
        fg = _make_person(l_wrist=(400, 240), r_wrist=(260, 240), box=(220, 80, 460, 470), score=0.91)
        idx = pick_primary_person_index([bg, fg], [{"box": [380, 200, 430, 270], "conf": 0.95}])
        self.assertEqual(idx, 1)

    def test_adaptive_zoom_rescues_distant_missed_hands(self):
        frame = np.zeros((800, 1400, 3), dtype=np.uint8)
        person = _make_person(
            l_wrist=(740, 260),
            r_wrist=(660, 420),
            box=(600, 140, 780, 620),
            score=0.95,
        )
        hand_sess = _FakeVisionSession([[], [[200, 180, 250, 240, 0.88, 0]]])
        gest_sess = _FakeVisionSession([[], [[200, 180, 250, 240, 0.86, 18]]])
        channel = HandGestureDetectorChannel(gest_sess, hand_sess, adaptive_zoom=True)
        note = channel.update([person], frame)
        self.assertTrue(note["ok"])
        self.assertTrue(note["zoomed"])
        self.assertEqual(note["found"], 1)
        self.assertEqual(person["hand_gestures"]["left"], "ok")
        self.assertAlmostEqual(person["hand_gesture_scores"]["left"], 0.86, places=3)

    def test_channel_reset_clears_throttle(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = _make_person(l_wrist=(360, 240), r_wrist=(280, 240))
        hand_sess = _FakeVisionSession([[[330, 180, 400, 260, 0.94, 0]]])
        gest_sess = _FakeVisionSession([[[330, 180, 400, 260, 0.95, 18]]])
        channel = HandGestureDetectorChannel(
            gest_sess, hand_sess, adaptive_zoom=False, interval_s=1.0
        )
        n1 = channel.update([person], frame, now=10.0)
        self.assertFalse(n1.get("throttled", False))
        n2 = channel.update([person], frame, now=10.1)
        self.assertTrue(n2.get("throttled", False))
        channel.reset()
        n3 = channel.update([person], frame, now=10.1)
        self.assertFalse(n3.get("throttled", False))
        self.assertEqual(channel.ran, 2)

    def test_pose_plugin_recognize_by_photo_detector_contract(self):
        from plugins.pose import PosePerceptionPlugin

        hand_sess = _FakeVisionSession([[
            [330, 180, 400, 260, 0.94, 0],
            [240, 180, 310, 260, 0.93, 0],
        ]])
        gest_sess = _FakeVisionSession([[
            [240, 180, 400, 260, 0.96, 8],
        ]])

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "sample.jpg")
            cv2.imwrite(img_path, np.zeros((480, 640, 3), dtype=np.uint8))

            plugin = PosePerceptionPlugin(
                {
                    "hands": True,
                    "hand_backend": "detector",
                    "action_backend": "rules",
                    "image_roots": [tmpdir],
                },
                namespace="",
                executor=None,
            )
            plugin._model = _FakePoseSession()
            plugin._gesture_det_model = gest_sess
            plugin._hand_det_model = hand_sess

            res = plugin.dispatch(
                "pose",
                {"action": "recognize_by_photo", "image_path": img_path},
            )
            self.assertTrue(res["ok"])
            self.assertEqual(res["count"], 1)
            self.assertEqual(res["image_size"], [640, 480])
            self.assertEqual(len(res["keypoint_names"]), 17)
            p0 = res["persons"][0]
            self.assertEqual(p0["hand_bboxes"]["left"], [330.0, 180.0, 400.0, 260.0])
            self.assertEqual(p0["hand_bboxes"]["right"], [240.0, 180.0, 310.0, 260.0])
            self.assertEqual(p0["hand_gestures"], {"left": "hand_heart", "right": "hand_heart"})
            self.assertAlmostEqual(p0["hand_gesture_scores"]["left"], 0.96, places=3)
            self.assertAlmostEqual(p0["hand_gesture_scores"]["right"], 0.96, places=3)
            self.assertIn("left", p0["hands"])
            self.assertIn("right", p0["hands"])

    def test_stream_describe_synthesizes_person_and_preserves_two_hand_hands_seen(self):
        from plugins.pose import PoseTracker, _PoseNode, build_backend

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand_sess = _FakeVisionSession([[[340, 180, 410, 260, 0.92, 0]]])
        gest_sess = _FakeVisionSession([[[260, 180, 410, 260, 0.94, 2]]])  # holy
        channel = HandGestureDetectorChannel(gest_sess, hand_sess, adaptive_zoom=False)

        node = object.__new__(_PoseNode)
        node._max_persons = 5
        node._tracker = PoseTracker()
        node._activity_interval_s = 0.0
        node._classifier = build_backend("rules")
        node._last_prediction = {}
        node._hand_channel = channel
        node._hand_gestures = None
        node._last_actions = {}
        node._publish_bbox = True
        node._publish_keypoints = "off"
        node._publish_hand_keypoints = "off"

        persons = node._describe([], [], [], frame, now_wall=1.0)
        self.assertEqual(len(persons), 1)
        lean = node._lean_record(persons[0])
        self.assertEqual(lean["hand_gestures"], {"left": "holy", "right": "holy"})
        self.assertEqual(lean["hands_seen"], ["left", "right"])
        self.assertEqual(lean["hand_bboxes"]["left"], [340.0, 180.0, 410.0, 260.0])
        self.assertEqual(lean["hand_bboxes"]["right"], [])
        self.assertFalse(node._has_hand_keypoints())

    def test_lazy_init_config_info_and_backward_compatible_backend_fallback(self):
        from plugins.pose import (
            DEFAULT_HAND_BACKEND,
            HAND_DETECTOR_MODEL,
            HAND_MODEL,
            TOOLS,
            PosePerceptionPlugin,
        )

        # Default config preserves upstream rtmpose backend and does not load models.
        props = TOOLS[0]["configSchema"]["properties"]
        self.assertEqual(props["hand_backend"]["default"], DEFAULT_HAND_BACKEND)
        self.assertIn("detector", props["hand_backend"]["enum"])

        plugin = PosePerceptionPlugin(
            {"hands": True, "hand_backend": "detector", "action_backend": "rules"},
            namespace="",
            executor=None,
        )
        self.assertIsNone(plugin._model)
        self.assertIsNone(plugin._gesture_det_model)
        self.assertIsNone(plugin._hand_det_model)

        info = plugin.dispatch("pose", {"action": "info"})
        self.assertEqual(info["state"], "idle")
        self.assertTrue(info["hands"])
        self.assertEqual(info["hand_backend"], "detector")
        self.assertEqual(info["hand_model"], HAND_DETECTOR_MODEL)
        self.assertNotIn("hand_keypoint_names", info)

        actions = plugin.dispatch("pose", {"action": "list_actions"})
        self.assertEqual(len(actions["gestures"]["hand_shapes"]), 30)

        # Unknown / legacy hand_backend falls back safely without raising.
        plugin.dispatch("pose", {"action": "config", "hand_backend": "legacy_unknown"})
        info_fb = plugin.dispatch("pose", {"action": "info"})
        self.assertEqual(info_fb["hand_backend"], DEFAULT_HAND_BACKEND)
        self.assertEqual(info_fb["hand_model"], HAND_MODEL)
        self.assertEqual(len(info_fb["hand_keypoint_names"]), 21)

    def test_start_node_does_not_hold_nodes_lock_and_surfaces_hand_download_progress(self):
        from plugins.pose import PosePerceptionPlugin

        plugin = PosePerceptionPlugin(
            {"hands": True, "hand_backend": "detector", "action_backend": "rules"},
            namespace="",
            executor=None,
        )
        plugin._model = _FakePoseSession()

        entered = threading.Event()
        release = threading.Event()
        info_during_load = {}
        errors = []

        def fake_ensure_models(model_dir, family=None, progress_cb=None):
            if progress_cb is not None:
                progress_cb("yolov10n_hands_640.engine", 500, 1000)
            entered.set()
            release.wait(timeout=5.0)
            raise RuntimeError("stop after progress check")

        with mock.patch(
            "utils.model_downloader.ensure_hand_detector_models",
            side_effect=fake_ensure_models,
        ):
            t = threading.Thread(
                target=lambda: plugin._hand_channel_for(
                    plugin._merged_config("/cam/rgb")
                ),
                daemon=True,
            )
            t.start()
            self.assertTrue(entered.wait(timeout=5.0))
            try:
                # info() must remain non-blocking and visibly report loading progress.
                info_during_load.update(plugin.dispatch("pose", {"action": "info"}))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                release.set()
                t.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual(info_during_load["state"], "loading")
        self.assertTrue(info_during_load["hand_loading"])
        self.assertIn("hand_load_status", info_during_load)
        self.assertFalse(plugin._hand_loading)
        self.assertIsNotNone(plugin._hand_load_error)

    def test_concurrent_stop_during_hand_download_prevents_node_revival(self):
        from plugins.pose import PosePerceptionPlugin

        class _FakeExecutor:
            def __init__(self):
                self.added = []
                self.removed = []

            def add_node(self, node):
                self.added.append(node)

            def remove_node(self, node):
                self.removed.append(node)

        for cancel_args, expected_key, expected_val in (
            ({"action": "stop", "instance_id": "/cam/rgb"}, "state", "idle"),
            ({"action": "stop"}, "state", "idle"),
            (
                {
                    "action": "config",
                    "instance_id": "/cam/rgb",
                    "hands": False,
                },
                "status",
                "configured",
            ),
        ):
            with self.subTest(cancel_args=cancel_args):
                executor = _FakeExecutor()
                plugin = PosePerceptionPlugin(
                    {
                        "hands": True,
                        "hand_backend": "detector",
                        "action_backend": "rules",
                    },
                    namespace="",
                    executor=executor,
                )
                plugin._model = _FakePoseSession()

                entered = threading.Event()
                release = threading.Event()
                start_reply = {}
                errors = []

                def fake_ensure_detector():
                    entered.set()
                    release.wait(timeout=5.0)
                    return _FakeVisionSession([[]]), _FakeVisionSession([[]])

                with mock.patch.object(
                    plugin,
                    "_ensure_hand_detector_models",
                    side_effect=fake_ensure_detector,
                ):
                    def run_start():
                        try:
                            start_reply["value"] = plugin.dispatch(
                                "pose",
                                {
                                    "action": "start",
                                    "input_topic": "/cam/rgb",
                                    "instance_id": "/cam/rgb",
                                },
                            )
                        except Exception as exc:  # noqa: BLE001
                            errors.append(exc)

                    t = threading.Thread(target=run_start, daemon=True)
                    t.start()
                    self.assertTrue(entered.wait(timeout=5.0))
                    try:
                        cancel_reply = plugin.dispatch("pose", cancel_args)
                    finally:
                        release.set()
                        t.join(timeout=5.0)

                self.assertEqual(errors, [])
                self.assertEqual(cancel_reply[expected_key], expected_val)
                self.assertEqual(start_reply["value"]["state"], "idle")
                self.assertEqual(plugin._nodes, {})
                self.assertEqual(executor.added, [])

                # A fresh start after cancellation must succeed cleanly.
                with mock.patch.object(
                    plugin,
                    "_ensure_hand_detector_models",
                    return_value=(_FakeVisionSession([[]]), _FakeVisionSession([[]])),
                ):
                    restart_reply = plugin.dispatch(
                        "pose",
                        {
                            "action": "start",
                            "input_topic": "/cam/rgb",
                            "instance_id": "/cam/rgb",
                        },
                    )
                    try:
                        self.assertEqual(restart_reply["state"], "running")
                        self.assertIn("/cam/rgb", plugin._nodes)
                    finally:
                        plugin.dispatch("pose", {"action": "stop", "instance_id": "/cam/rgb"})

    def test_model_downloader_hand_detector_bundles_pinned_and_public(self):
        import ipaddress
        from urllib.parse import urlparse
        from utils import model_downloader as md

        self.assertRegex(md.HAND_DETECTOR_MODEL_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(set(md.HAND_DETECTOR_MODEL_BUNDLES), {"jp511", "jp61"})
        for family, bundle in md.HAND_DETECTOR_MODEL_BUNDLES.items():
            parsed = urlparse(bundle["base_url"])
            self.assertEqual(parsed.scheme, "https")
            self.assertTrue(parsed.hostname)
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                self.assertFalse(ip.is_private or ip.is_loopback)
            except ValueError:
                pass
            self.assertEqual(
                set(bundle["files"]),
                {"yolov10n_hands_640.engine", "yolov10n_gestures_640.engine"},
            )
            for fname, meta in bundle["files"].items():
                self.assertGreater(meta["size"], 0, f"{family}/{fname}")
                self.assertTrue(
                    bool(re.fullmatch(r"[0-9a-f]{64}", meta["sha256"])),
                    f"{family}/{fname}",
                )

        captured = {}

        def fake_ensure_vision(name, bundles, target_dir, family, progress_cb=None):
            captured["name"] = name
            captured["target_dir"] = target_dir
            captured["family"] = family
            return {
                "yolov10n_hands_640.engine": os.path.join(target_dir, "yolov10n_hands_640.engine"),
                "yolov10n_gestures_640.engine": os.path.join(target_dir, "yolov10n_gestures_640.engine"),
            }

        with mock.patch.object(md, "_ensure_vision_bundle", side_effect=fake_ensure_vision):
            out = md.ensure_hand_detector_models("/models/hand_gesture", family="jp61")
        self.assertEqual(captured["name"], "hand_detector")
        self.assertEqual(captured["family"], "jp61")
        self.assertEqual(captured["target_dir"], "/models/hand_gesture/jp61")
        self.assertIn("yolov10n_gestures_640.engine", out)


if __name__ == "__main__":
    unittest.main()
