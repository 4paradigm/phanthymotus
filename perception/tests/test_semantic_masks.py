import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.semantic_masks import decode_camera_jpeg, overlay, semantic_maps
from plugins.vision_runtime import LetterboxMeta, VisionDecodeError

META = LetterboxMeta(1, 0, 2, 8, 4)
NAMES = ["person", "car", "chair"]
PRESETS = {"person": ["person"], "car": ["car"], "room": ["person","chair"]}


def heads():
    # The car is on the right. The person and the lower-confidence chair overlap.
    proto = np.ones((1,2,4,4),np.float32)
    proto[:,1,:,:2] = -1
    rows = np.array([[0,2,8,6,.99,0,1,0], [0,2,8,6,.85,1,0,1], [2,2,8,6,.5,2,1,0],
                     [0,2,8,6,.05,1,1,0]],np.float32)[None]
    return rows, proto


@pytest.mark.parametrize("transpose", [True,False])
def test_class_selection_letterbox_background_and_overlap(transpose):
    rows, proto = heads()
    outs = [proto, rows.transpose(0,2,1)] if transpose else [rows, proto]
    masks = semantic_maps(outs, META, (8,8), NAMES, PRESETS, .3)
    assert all(m.shape == (4,8) and m.dtype == np.uint8 for m in masks.values())
    assert np.all(masks["person"] == 1) and np.all(masks["room"] == 1)
    assert np.all(masks["car"][:,:4] == 0) and np.all(masks["car"][:,4:] == 1)


@pytest.mark.skipif(getattr(cv2, "__motus_fake__", False), reason="overlay needs OpenCV blending")
def test_overlay_preserves_background_pixels():
    source = np.zeros((4, 8, 3), np.uint8)
    mask = np.zeros((4, 8), np.uint8)
    mask[:, 4:] = 1
    preview = overlay(source, mask)
    np.testing.assert_array_equal(preview[:, :4], source[:, :4])
    assert np.any(preview[:, 4:])


def test_empty_input_no_claimed_pixels_and_crop():
    rows, proto = heads()
    masks = semantic_maps([rows[:,:0],proto], META,(8,8),NAMES,PRESETS,.3)
    assert not any(m.any() for m in masks.values())
    rows = rows[:,:1].copy()
    rows[0,0,:4] = [1.2, 3, 4, 5]
    m = semantic_maps([rows,proto],META,(8,8),NAMES,PRESETS,.3)["person"]
    expected = np.zeros_like(m)
    expected[1:3,2:4] = 1
    np.testing.assert_array_equal(m,expected)


def test_detection_only_head_and_bad_image():
    with pytest.raises(VisionDecodeError):
        semantic_maps([np.zeros((1,300,6))],META,(8,8),NAMES,PRESETS,.3)
    from io import BytesIO
    from PIL import Image
    png = BytesIO()
    Image.new("RGB", (8, 4)).save(png, format="PNG")
    with pytest.raises(ValueError):
        decode_camera_jpeg(png.getvalue())


def test_card_registration_precedes_jetson_opencv_initialization(monkeypatch):
    import runpy
    monkeypatch.setitem(sys.modules, "cv2", None)
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "plugins/semantic_masks.py"))
    assert callable(module["decode_camera_jpeg"])


@pytest.mark.skipif(getattr(cv2, "__motus_fake__", False), reason="verify real JPEG and PNG codecs")
def test_camera_publishes_stamped_mask_report_and_optional_overlay_from_one_frame():
    import json
    from types import SimpleNamespace
    from sensor_msgs.msg import CompressedImage
    from plugins.segment import _Camera, _ports
    from vision_stubs import _FakeExecutor, _wait_until, frame_bytes

    class Engine:
        input_size = (8, 8)
        calls = 0
        def infer(self, image):
            assert image.shape[:2] == (4, 8)
            self.calls += 1
            return heads(), META

    engine = Engine()
    cam = _Camera("/test/rgb", _FakeExecutor(), lambda: (engine, NAMES), .3, 10)
    try:
        person = _ports("/test/rgb", "person", "a", False)
        car = _ports("/test/rgb", "car", "b", True)
        cam.add(("person", "a"), "person", ["person"], person)
        cam.add(("car", "b"), "car", ["car"], car)
        frame = CompressedImage()
        frame.data = frame_bytes(8, 4)
        frame.header.frame_id = "cam_rgb_optical"
        frame.header.stamp = SimpleNamespace(sec=123, nanosec=456)
        cam.subscriptions[0].callback(frame)
        pubs = {p.topic: p for p in cam.publishers}
        assert len(cam.subscriptions) == 1 and len(pubs) == 5
        assert _wait_until(lambda: all(p.raw_messages for p in pubs.values())), cam.error
        assert engine.calls == 1
        mask, report, preview = [pubs[port["topic"]].raw_messages[0] for port in car]
        assert mask.format == "png" and preview.format == "jpeg"
        assert mask.header == preview.header == frame.header
        decoded = cv2.imdecode(np.frombuffer(mask.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert decoded.shape == (4, 8) and np.count_nonzero(decoded) == 16
        info = json.loads(report.data)
        assert info["stamp"] == {"sec": 123, "nanosec": 456}
        assert info["frame_id"] == "cam_rgb_optical" and (info["width"], info["height"]) == (8, 4)
        assert info["classes"] == [{"id": 0, "name": "background", "pixels": 16},
                                   {"id": 1, "name": "car", "pixels": 16}]
    finally:
        cam.close()


def test_card_registration_and_independent_instances(monkeypatch):
    import types
    import plugins.segment as cards
    from plugins.segment import SegmentationPlugin

    topics = {}
    class Stream:
        def __init__(self, topic, *args):
            import threading
            self.cv = threading.Condition()
            self.ready = True
            self.load_failed = False
            self.vocabulary = set(NAMES) | {"bicycle", "motorcycle", "bus", "truck"}
            self.error = ""
            self.targets = {}
            topics[topic] = self
        def add(self,key,name,labels,ports):
            if "/unavailable/" in ports[0]["topic"]:
                raise RuntimeError("publisher unavailable")
            self.targets[key] = (name, labels, ports)
        def remove(self,key):
            self.targets.pop(key)
            return not self.targets
        def close(self):
            self.closed = True

    validator = types.ModuleType("rclpy.validate_full_topic_name")
    validator.validate_full_topic_name = lambda topic: None
    monkeypatch.setitem(sys.modules,"rclpy.validate_full_topic_name",validator)
    monkeypatch.setattr(cards,"_Camera",Stream)
    plugin = SegmentationPlugin({},None,lambda:None)
    assert {t["name"] for t in plugin.get_tools()} == {"car","person","chair","traffic"}
    a = {"instance_id":"left","input_topic":"/cam/jpeg"}
    b = {"instance_id":"right","input_topic":"/cam/jpeg"}
    pre = plugin.dispatch("person",dict(a,action="info"))
    left = plugin.dispatch("person",dict(a,action="start"))
    right = plugin.dispatch("car",dict(b,action="start"))
    assert pre["topic_out"] == left["topic_out"] and left["topic_out"] != right["topic_out"]
    assert len(plugin.nodes) == 1 and len(topics["/cam/jpeg"].targets) == 2
    plugin.dispatch("car",dict(b,action="stop"))
    assert plugin.dispatch("person",dict(a,action="info"))["state"] == "running"
    with pytest.raises(RuntimeError, match="publisher unavailable"):
        plugin.dispatch("person",dict(a,action="start",input_topic="/unavailable/jpeg"))
    assert plugin.dispatch("person",dict(a,action="info"))["topic_out"] == left["topic_out"]
    assert topics["/unavailable/jpeg"].closed
    plugin.shutdown()
    assert topics["/cam/jpeg"].closed
