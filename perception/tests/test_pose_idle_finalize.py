"""Actualnode worker pause/resume and backend confidence; no GPU/ROS proof."""
import json
import time

import numpy as np
import pytest

from test_pose_plugin import _plugin, _feed, _publisher
from test_pose_plugin import _FakeModel, _standing_row
from test_pose_stgcn import _FakeEngine, _frames, _standing
from plugins.pose_stgcn import SkeletonActionBackend, HybridActionBackend


class _FinalClassifier:
    history_s = 5.
    last_error = None

    def __init__(self):
        self.calls = 0

    def classify(self, frames, want_activity=True):
        return {"posture": "standing", "posture_confidence": .9,
                "activity": None, "evidence": {}}

    def finalize(self, frames):
        self.calls += 1
        return {"activity": {"name": "brush teeth", "name_zh": "刷牙",
                             "action_id": 3, "score": .81, "source": "stgcn"},
                "evidence": {"frames_used": len(frames)}}


def _running(cfg):
    plugin, _ = _plugin({"label_hold": 1, **cfg})
    classifier = _FinalClassifier()
    plugin._classifier_for = lambda merged: classifier
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb", frames=4)
    return node, classifier


def test_worker_finishes_once_and_does_not_invent_a_skeleton_frame():
    node, classifier = _running({"activity_idle_flush_s": .15})
    try:
        lean = _publisher(node, node._output_topic)
        skeleton = _publisher(node, node._skeleton_topic)
        from vision_stubs import _wait_until
        _wait_until(lambda: classifier.calls == 1, timeout=2.)
        _wait_until(lambda: len(lean.messages) == 5, timeout=2.)
        final = json.loads(lean.messages[-1])
        assert final["persons"][0]["activity"] == "brush teeth"
        assert final["persons"][0]["activity_score"] == .81
        assert "latency_ms" not in final
        assert node._detect_count == 4
        assert len(skeleton.messages) == 4
        time.sleep(.25)
        assert classifier.calls == 1 and len(lean.messages) == 5
        _feed_before = len(lean.messages)
        from vision_stubs import _FakeCompressedImage, frame_bytes
        node._last_inference_time -= 1.
        node.subscriptions[0].callback(_FakeCompressedImage(frame_bytes(640, 480)))
        _wait_until(lambda: len(lean.messages) > _feed_before, timeout=2.)
        _wait_until(lambda: classifier.calls == 2, timeout=2.)
        assert len(skeleton.messages) == 5
        assert node._detect_count == 5
    finally:
        node.stop()


def test_legacy_default_does_not_finish_on_a_pause():
    node, classifier = _running({})
    try:
        node._last_image_received_at -= 2.
        node._flush_idle_activity()
        assert classifier.calls == 0
        assert len(_publisher(node, node._output_topic).messages) == 4
    finally:
        node.stop()


def test_one_action_update_preserves_the_other_person_in_the_snapshot():
    class Partial(_FinalClassifier):
        def finalize(self, frames):
            self.calls += 1
            if frames[-1].box[0] > 320:
                return {"activity": None, "evidence": {}}
            return {"activity": {"name": "brush teeth", "name_zh": "刷牙",
                                 "action_id": 3, "score": .81,
                                 "source": "stgcn"}, "evidence": {}}

    plugin, _ = _plugin({"label_hold": 1, "activity_idle_flush_s": .15},
                        model=_FakeModel([_standing_row(cx=200),
                                          _standing_row(cx=440)]))
    classifier = Partial()
    plugin._classifier_for = lambda merged: classifier
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb", frames=4)
    try:
        from vision_stubs import _wait_until
        lean = _publisher(node, node._output_topic)
        _wait_until(lambda: len(lean.messages) == 5, timeout=2.)
        final = json.loads(lean.messages[-1])
        assert final["count"] == 2
        assert {p["id"] for p in final["persons"]} == {1, 2}
        assert sum(p.get("activity") == "brush teeth"
                   for p in final["persons"]) == 1
        assert node._detect_count == 4
    finally:
        node.stop()


def test_expired_track_and_stop_do_not_republish_people():
    node, classifier = _running({"activity_idle_flush_s": 2., "tracker_timeout_s": .6})
    try:
        node._last_image_received_at -= 3.
        for person in node._idle_persons:person["track"].last_seen -= 3.
        node._flush_idle_activity()
        assert classifier.calls == 0
        node.stop()
        assert node._idle_persons == []
        node._flush_idle_activity()
        assert classifier.calls == 0
    finally:
        node.stop()


def test_finalization_preserves_minimum_frames_and_uses_true_model_probability():
    values = np.zeros(126, np.float32);values[2] = .81;values[37] = .19
    engine = _FakeEngine(outputs=[values[None]])
    backend = SkeletonActionBackend(engine=engine, window_maturity_s=2.8,
                                    min_window_frames=4, min_motion=0., min_score=.04)
    frames = _frames(_standing, n=5, fps=10.)
    normal = backend.classify(frames)["activity"]
    final = backend.finalize(frames)["activity"]
    assert final["name"] == "brush teeth" and final["score"] == .81
    assert normal["score"] < final["score"]
    assert backend.finalize(frames[:3])["activity"] is None
    hybrid = HybridActionBackend(learned=backend, strict_vocabulary=True)
    assert hybrid.finalize(frames)["activity"] == final


def test_persisted_instance_config_retains_idle_setting():
    plugin, _ = _plugin({"activity_idle_flush_s": .4})
    assert plugin._merged_config("fresh")["activity_idle_flush_s"] == .4
    plugin._instance_configs["custom"] = {"activity_idle_flush_s": .7}
    assert plugin._merged_config("custom")["activity_idle_flush_s"] == .7
    assert _plugin()[0]._merged_config("fresh")["activity_idle_flush_s"] == 0.
