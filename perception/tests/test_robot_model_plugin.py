"""One-shot robot model card contract."""

import numpy as np

from vision_stubs import _FakeExecutor, frame_bytes

from plugins.robot_model_catalog import CANDIDATES
from plugins.robot_model import RobotModelPlugin


class Detector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect_frame(self, frame, confidence):
        self.calls += 1
        return self.boxes


class Proxy:
    def __init__(self):
        self.vectors = np.zeros((len(CANDIDATES), 512), dtype=np.float32)
        self.vectors[0, 0] = 1
        self.vectors[1, 1] = 1
        self.calls = []

    def embed(self, image):
        self.calls.append(image)
        vector = np.zeros(512, dtype=np.float32)
        vector[(len(self.calls) - 1) % 2] = 1
        return vector.tolist()


def box(name, coords, confidence=0.9):
    return {"name": name, "bbox": coords, "confidence": confidence}


def plugin(boxes, cfg=None):
    card = RobotModelPlugin(cfg or {"min_score": 0.5, "min_margin": 0.2},
                            Detector(boxes), _FakeExecutor())
    card._proxy = Proxy()
    return card


def test_two_robots_get_separate_predictions_and_crops():
    card = plugin([box("quadruped robot", [0, 0, 50, 40]),
                   box("person", [30, 0, 50, 20]),
                   box("humanoid robot", [50, 0, 100, 40])])
    result = card.recognize_frame(np.zeros((40, 100, 3), dtype=np.uint8))
    assert result["ok"] is True
    assert [item["model_id"] for item in result["objects"]] == [
        CANDIDATES[0][0], CANDIDATES[1][0]]
    assert all(set(("bbox", "detector_class", "detector_score", "model_id",
                    "score", "margin", "reason")) <= item.keys()
               for item in result["objects"])
    assert len(card._proxy.calls) == 2


def test_no_robot_does_not_load_model():
    card = plugin([box("person", [0, 0, 20, 20])])
    card._proxy = None
    assert card.recognize_frame(np.zeros((40, 40, 3), dtype=np.uint8)) == {
        "ok": True, "objects": []}


def test_invalid_and_tiny_boxes_are_skipped():
    card = plugin([box("robot", [-5, -5, 1, 1]),
                   box("drone", [20, 20, 10, 10]),
                   box("robot", [2, 2, 2, 2])])
    result = card.recognize_frame(np.zeros((30, 30, 3), dtype=np.uint8))
    assert result == {"ok": True, "objects": []}


def test_uncalibrated_and_ambiguous_results_abstain():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    card = plugin([box("robot", [0, 0, 40, 40])],
                  {"min_score": None, "min_margin": None})
    assert card.recognize_frame(frame)["objects"][0]["reason"] == "uncalibrated"
    card._cfg = {"min_score": 0.5, "min_margin": 1.1}
    item = card.recognize_frame(frame)["objects"][0]
    assert item["model_id"] == "unknown"
    assert item["reason"] == "ambiguous"


def test_missing_detector_or_bundle_is_engine_unavailable(monkeypatch):
    card = RobotModelPlugin({}, None, _FakeExecutor())
    assert "VOP" in card.dispatch("robot_model", {"action": "info"})["detail"]
    assert card.recognize_frame(np.zeros((10, 10, 3), dtype=np.uint8))["reason"] == "engine_unavailable"
    card = RobotModelPlugin({}, Detector([box("robot", [0, 0, 20, 20])]), _FakeExecutor())
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing bundle")
    monkeypatch.setattr("plugins.robot_model.RobotModelProxy", missing)
    assert card.recognize_frame(np.zeros((30, 30, 3), dtype=np.uint8))["reason"] == "engine_unavailable"


def test_photo_and_url_use_shared_input_guard_and_recognition(monkeypatch, tmp_path):
    card = plugin([box("robot", [0, 0, 20, 20])],
                  {"min_score": 0.5, "min_margin": 0.2,
                   "image_roots": [str(tmp_path)]})
    image = tmp_path / "robot.jpg"
    image.write_bytes(frame_bytes(30, 30))
    local = card.dispatch("robot_model", {"action": "recognize_by_photo", "image_path": str(image)})
    card._proxy.calls.clear()
    monkeypatch.setattr("plugins.image_input.fetch_url", lambda url, cap: frame_bytes(30, 30))
    remote = card.dispatch("robot_model", {"action": "recognize_by_url", "url": "https://example.test/a.jpg"})
    assert local["objects"] == remote["objects"]
    assert card.dispatch("robot_model", {"action": "recognize_by_photo", "image_path": "/etc/passwd"})["reason"] == "bad_input"
    image.write_bytes(b"broken")
    assert card.dispatch("robot_model", {"action": "recognize_by_photo", "image_path": str(image)})["reason"] == "bad_input"
