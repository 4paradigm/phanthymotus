"""Scope-specific tests for the opt-in, weak-privacy mosaic demo Card.

No robot, motion, network, camera, private video, model download or PR needed.
Run in the existing Perception image:
  PYTHONPATH=perception python3 -m unittest \
    discover -s perception/tests -p test_mosaic_smooth_demo.py -v
"""
from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from plugins.mosaic_smooth import (
    MosaicSmoothDemoPlugin, output_topics, validate_topic,
)
from plugins.mosaic_smooth_runtime import (
    DemoConfig, MosaicDemoError, SmoothMosaic, validate_model,
)
from utils.model_downloader import MOSAIC_SMOOTH_MODEL_FILENAME, MOSAIC_SMOOTH_MODEL_FILES


class NoFaceDetector:
    def __init__(self):
        self.calls = 0
        self.shape = None

    def setInputSize(self, size):
        self.shape = size

    def detect(self, image):
        self.calls += 1
        return 1, None


class OneFaceDetector(NoFaceDetector):
    def detect(self, image):
        self.calls += 1
        # 80x80 textured central face fixture (we do not claim it is a face).
        row = np.array([[230, 90, 90, 80] + [0]*10 + [0.93]], np.float32)
        return 1, row


def fixture_jpeg():
    rng = np.random.default_rng(20261121)
    image = rng.integers(0,255,(360,640,3),dtype=np.uint8)
    ok,jpg = cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,90])
    assert ok
    return jpg.tobytes()


class TestConfig(unittest.TestCase):
    def test_requires_explicit_consent(self):
        c = DemoConfig()
        self.assertFalse(c.consent_acknowledged)
        self.assertTrue(c.updated({"consent_acknowledged":True}).consent_acknowledged)
        for bad in (
            {"confidence_threshold":float("nan")},
            {"output_width":True},
            {"output_width":0},
            {"detect_every_frames":13},
            {"bad_field":1},
            {"consent_acknowledged":1},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(MosaicDemoError):
                    c.updated(bad)

    def test_model_wrong_digest_is_rejected(self):
        with tempfile.NamedTemporaryFile() as f:
            size = MOSAIC_SMOOTH_MODEL_FILES[MOSAIC_SMOOTH_MODEL_FILENAME]["size"]
            f.write(b"x" * size)
            f.flush()
            with self.assertRaisesRegex(MosaicDemoError,"model_sha256_mismatch"):
                validate_model(f.name)

    def test_model_wrong_size_is_rejected(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"invalid-model" * 2000)
            f.flush()
            with self.assertRaisesRegex(MosaicDemoError, "model_size_mismatch"):
                validate_model(f.name)

    def test_topic_and_instance_isolation(self):
        self.assertNotEqual(output_topics("/robot/camera","a"),
                            output_topics("/robot/camera","b"))
        for value in ("camera","/bad-topic","/","",None):
            with self.subTest(value=value):
                with self.assertRaises(MosaicDemoError):
                    validate_topic(value)
        self.assertEqual(validate_topic("/ubuntu/camera/color"),
                         "/ubuntu/camera/color")

    def test_mcp_consent_guard_and_no_side_effects(self):
        plugin = MosaicSmoothDemoPlugin({"model_path":"/nonexistent/mosaic_yunet_for_unit_test.onnx"},executor=None)
        self.assertEqual(plugin.PREFIX,"mosaic_smooth_demo")
        self.assertEqual(plugin.get_tools()[0]["type"],"processor")
        self.assertEqual(plugin.get_tools()[0]["multiInstance"],True)
        start = plugin.dispatch("mosaic_smooth_demo",{
            "action":"start","instance_id":"a",
            "input_topic":"/test/camera",
        })
        self.assertEqual(start["state"],"error")
        self.assertIn("consent_acknowledgment_required",start["error"])
        config = plugin.dispatch("mosaic_smooth_demo",{
            "action":"config","instance_id":"a",
            "config":{"consent_acknowledged":True,"detect_every_frames":4},
        })
        self.assertEqual(config["state"],"configured")
        info = plugin.dispatch("mosaic_smooth_demo",{
            "action":"info","instance_id":"a","input_topic":"/test/camera",
        })
        self.assertFalse(info["privacy_guaranteed"])
        self.assertEqual(len(info["topic_out"]),2)
        # Model is absent: no ROS node is constructed and no model is downloaded.
        start = plugin.dispatch("mosaic_smooth_demo",{
            "action":"start","instance_id":"a",
            "input_topic":"/test/camera",
        })
        self.assertEqual(start["state"],"error")
        self.assertIn("model_missing",start["error"])
        self.assertEqual(plugin.dispatch("mosaic_smooth_demo",{
            "action":"stop","instance_id":"a",
        })["state"],"idle")
        plugin.shutdown()


class TestPixels(unittest.TestCase):
    def test_no_face_is_coarse_mosaic_never_source_bytes(self):
        source=fixture_jpeg()
        detector=NoFaceDetector()
        cfg=DemoConfig(consent_acknowledged=True)
        current=cv2.getNumThreads()
        pipeline=SmoothMosaic("",cfg,detector=detector)
        self.assertEqual(cv2.getNumThreads(),current)
        results=[]
        for _ in range(8):
            output,faces,coarse=pipeline.process(source)
            self.assertNotEqual(output,source)
            self.assertTrue(output.startswith(b"\xff\xd8"))
            self.assertEqual(faces,0)
            self.assertTrue(coarse)
            results.append(output)
        self.assertEqual(detector.calls,2)
        self.assertEqual(pipeline.detect_calls,2)
        image=cv2.imdecode(np.frombuffer(results[0],np.uint8),cv2.IMREAD_COLOR)
        original=cv2.imdecode(np.frombuffer(source,np.uint8),cv2.IMREAD_COLOR)
        self.assertGreater(np.abs(image.astype(np.int16)
                        - original.astype(np.int16)).mean(),25)

    def test_every_frame_detection_option(self):
        detector=NoFaceDetector()
        pipeline=SmoothMosaic(
            "",DemoConfig(consent_acknowledged=True,detect_every_frames=1),
            detector=detector)
        frame=fixture_jpeg()
        for _ in range(4):
            pipeline.process(frame)
        self.assertEqual(detector.calls,4)

    def test_roi_and_intermediate_frames_are_reencoded(self):
        source=fixture_jpeg()
        detector=OneFaceDetector()
        pipeline=SmoothMosaic("",DemoConfig(consent_acknowledged=True),
                              detector=detector)
        outputs=[]
        for _ in range(8):
            data,faces,fallback=pipeline.process(source)
            self.assertNotEqual(data,source)
            outputs.append((faces,fallback))
        self.assertEqual(detector.calls,2)
        self.assertTrue(any(n>0 and not coarse for n,coarse in outputs))
        self.assertFalse(all(coarse for _,coarse in outputs))

    def test_invalid_frame_is_not_published(self):
        pipeline=SmoothMosaic(
            "",DemoConfig(consent_acknowledged=True),detector=NoFaceDetector())
        for jpeg in (b"",b"raw image",b"\xff\xd8broken\xff\xd9"):
            with self.subTest(jpeg=jpeg):
                with self.assertRaises(MosaicDemoError):
                    pipeline.process(jpeg)


if __name__=="__main__":
    unittest.main()
