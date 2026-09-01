"""Host-side contract checks for person_area; no model or ROS hardware needed."""

from __future__ import annotations

import json

from vision_stubs import _FakeCompressedImage, _FakeExecutor, _wait_until  # noqa: F401
import plugins.person_area as person_area  # noqa: E402


def test_person_area_publishes_only_the_largest_person(monkeypatch):
    class Detector:
        def detect(self, jpeg, confidence):
            assert jpeg == b"frame" and confidence == 0.6
            return 100, 50, [
                {"bbox_xyxy": [1, 2, 11, 12], "area_px": 100, "area_ratio": 0.02, "confidence": 0.95},
                {"bbox_xyxy": [20, 5, 60, 45], "area_px": 1600, "area_ratio": 0.32, "confidence": 0.83},
            ]

    monkeypatch.setattr(person_area, "_build_detector", lambda cfg: Detector())
    executor = _FakeExecutor()
    plugin = person_area.PersonAreaPlugin({"confidence": 0.6, "fps": 15}, "q5", executor)
    assert plugin.dispatch("person_area", {"action": "start", "input_topic": "/q5/camera/rgb"})["state"] == "loading"
    assert _wait_until(lambda: len(executor.nodes) == 1)
    node = executor.nodes[0]
    node.subscriptions[0].callback(_FakeCompressedImage(b"frame"))
    assert _wait_until(lambda: len(node.publishers[0].messages) == 1)
    payload = json.loads(node.publishers[0].messages[0])
    assert payload["person_count"] == 2
    assert payload["largest_person"] == {"bbox_xyxy": [20, 5, 60, 45], "area_px": 1600,
                                         "area_ratio": 0.32, "confidence": 0.83}
    plugin.dispatch("person_area", {"action": "stop"})


def test_person_area_publishes_null_when_no_person_is_detected(monkeypatch):
    class Detector:
        def detect(self, jpeg, confidence):
            return 848, 480, []

    monkeypatch.setattr(person_area, "_build_detector", lambda cfg: Detector())
    executor = _FakeExecutor()
    plugin = person_area.PersonAreaPlugin({}, "q5", executor)
    plugin.dispatch("person_area", {"action": "start", "input_topic": "/q5/camera/rgb"})
    assert _wait_until(lambda: len(executor.nodes) == 1)
    node = executor.nodes[0]
    node.subscriptions[0].callback(_FakeCompressedImage(b"frame"))
    assert _wait_until(lambda: len(node.publishers[0].messages) == 1)
    payload = json.loads(node.publishers[0].messages[0])
    assert payload["person_count"] == 0 and payload["largest_person"] is None
    plugin.dispatch("person_area", {"action": "stop"})
