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
