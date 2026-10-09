"""ROS-free contracts for the merged SCRFD-2.5G detector work.

The w600k_mbf/sface recognizer candidates are still in stash@{0}; tests for
_ONNX_ARCFACE_REF/_similarity_transform live with that work.
"""
import ast
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from typing import Optional
from unittest.mock import patch
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins import ort_worker  # noqa: E402
from plugins.face_corpus import MAX_IMAGE_BYTES, corpus_entries, load_image  # noqa: E402
from plugins.image_input import (  # noqa: E402
    DEFAULT_IMAGE_ROOTS, BadInput, check_under_roots,
)
from utils.log_sampling import escape_log_text  # noqa: E402
from utils.model_downloader import (  # noqa: E402
    FACE_SCRFD_DETECTOR_BUNDLES, FACE_SCRFD_EDGEFACE_BASE,
    FACE_SCRFD_RECOGNIZER_BUNDLES, ensure_verified_bundle,
)
from utils.model_progress import fetch_status  # noqa: E402
from utils.ros_lifecycle import dispose_node  # noqa: E402
from urllib.parse import urlsplit  # noqa: E402

import cv2

# These contracts feed real pixel content through cv2 encode/decode and, in the
# detector case, cv2.dnn — the "WxH" marker fake cv2 (vision_stubs.__motus_fake__)
# cannot stand in for that. They run where cv2 is real: the perception image,
# whose container-side suite run is the documented acceptance pass (see
# perception/README.md § Running the tests inside a perception image). Elsewhere
# they skip rather than assert against stub pixels.
_NEEDS_REAL_CV2 = not getattr(cv2, "__motus_fake__", False)


def load_face():
    path = Path(__file__).resolve().parents[1] / "plugins" / "face_scrfd.py"
    tree = ast.parse(path.read_text())
    # Extract production definitions without importing ROS or EdgeFace dependencies.
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and getattr(node, "name", "") != "_FaceNode"]
    nodes += [node for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in {
                  "_ARCFACE_REF", "_SCRFD_STRIDES", "_SCRFD_NUM_ANCHORS",
                  "_DETECT_TARGET_W", "_CONTAINER_SWEEP_THRESHOLDS",
                  "_CONTAINER_PORT_BASE", "_CONTAINER_PORT_STRIDE",
                  "_DEVICE_PROVIDERS",
                  # Plain-data module constants the extracted methods read.
                  "DEFAULT_SIMILARITY_THRESHOLD", "ENROLL_WINDOW_S",
                  "ENROLL_WINDOW_MAX_FRAMES", "ENROLL_MAX_ANALYZED",
                  "VISIT_GAP_S", "TOOLS",
              } for t in node.targets)]
    # The namespace mirrors the module's globals for everything the extracted
    # code reads. test_face_scrfd_plugin asserts statically that it stays
    # complete — adding a module-level dependency here is what twice shipped a
    # NameError that only the image could see.
    ns = dict(np=np, os=os, log=logging.getLogger(__name__), urllib=urllib,
              threading=threading, time=time, Path=Path, Optional=Optional,
              DEFAULT_MODEL_NAME="edgeface_s_gamma_05", _MODEL_BASE_URL="https://example.invalid",
              check_under_roots=check_under_roots, BadInput=BadInput,
              escape_log_text=escape_log_text, urlsplit=urlsplit,
              dispose_node=dispose_node, corpus_entries=corpus_entries,
              load_image=load_image, ort_worker=ort_worker,
              ensure_verified_bundle=ensure_verified_bundle, fetch_status=fetch_status,
              FACE_SCRFD_RECOGNIZER_BUNDLES=FACE_SCRFD_RECOGNIZER_BUNDLES,
              FACE_SCRFD_DETECTOR_BUNDLES=FACE_SCRFD_DETECTOR_BUNDLES,
              FACE_SCRFD_EDGEFACE_BASE=FACE_SCRFD_EDGEFACE_BASE,
              DEFAULT_IMAGE_ROOTS=DEFAULT_IMAGE_ROOTS, MAX_IMAGE_BYTES=MAX_IMAGE_BYTES)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + sorted(nodes, key=lambda n: n.lineno), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


@unittest.skipUnless(_NEEDS_REAL_CV2, "needs real cv2")
class FaceContracts(unittest.TestCase):
    def setUp(self):
        self.ns = load_face()

    def test_detector_files(self):
        self.assertEqual(
            self.ns["_ensure_weights"] and True, True)  # module loads
        # Every selectable artifact now lives pinned in the downloader; the
        # plugin resolves detector names through that registry, not a local
        # filename dict. Assert both halves: pins where they live, wiring
        # where it lives.
        md = (Path(__file__).resolve().parents[1] / "utils" / "model_downloader.py").read_text()
        for needle in ("scrfd_2.5g_bnkps_hsuyabc.onnx", "scrfd_500m_kps.onnx",
                       "face_detection_yunet_2023mar.onnx", "edgeface_base.int8.onnx",
                       "edgeface_s_gamma_05.onnx", "edgeface_s_gamma_05.onnx.data"):
            self.assertIn(needle, md)
        src = (Path(__file__).resolve().parents[1] / "plugins" / "face_scrfd.py").read_text()
        for needle in ('"scrfd_2.5g"', "FACE_SCRFD_DETECTOR_BUNDLES", "_detector_key"):
            self.assertIn(needle, src)

    def test_bbox_json(self):
        detector = self.ns["SCRFDDetector"].__new__(self.ns["SCRFDDetector"])
        detector._confidence = .5
        detector._forward = lambda *a: (
            [np.array([[.9]], np.float32)],
            [np.array([[1, 2, 30, 40]], np.float32)],
            [np.zeros((1, 5, 2), np.float32)],
        )
        result = detector.detect(np.zeros((640, 640, 3), np.uint8))[0]
        # bbox must be plain floats — np.float32 is not JSON serializable
        self.assertTrue(all(type(v) is float for v in result["bbox"]))
        json.dumps({"bbox_relative": [round(v / 640, 6) for v in result["bbox"]]})

    def test_bbox_relative_contract(self):
        class Database:
            _lock = __import__("threading").Lock()

            def match(self, embedding, threshold):
                return "unknown", 0.0

            def get_person(self, person_id):
                return None

        result = self.ns["_face_result"](
            Database(),
            {"bbox": [82.0, 87.0, 230.0, 270.0],
             "embedding": np.array([1.0, 0.0]), "confidence": 0.9},
            (363, 304, 3),
            0.4,
        )
        self.assertEqual(result["bbox_relative"], [82.0 / 304, 87.0 / 363,
                                                     148.0 / 304, 183.0 / 363])
        self.assertTrue(all(0.0 <= value <= 1.0
                            for value in result["bbox_relative"]))

    def test_container_sweep_disabled_by_default(self):
        for port in ('15720', '15820', '15920'):
            with patch.dict(os.environ, {'MCP_PORT': port}, clear=True):
                self.assertEqual(self.ns['_container_sweep_overrides'](), {})

    def test_container_threshold_sweep(self):
        with patch.dict(os.environ, {"FACE_CONTAINER_SWEEP": "1",
                                     "MCP_PORT": "15720"}):
            self.assertEqual(self.ns["_container_sweep_overrides"](),
                             {"similarity_threshold": 0.40})
        with patch.dict(os.environ, {"FACE_CONTAINER_SWEEP": "1",
                                     "MCP_PORT": "15820"}):
            self.assertEqual(self.ns["_container_sweep_overrides"](),
                             {"similarity_threshold": 0.39})
        with patch.dict(os.environ, {"FACE_CONTAINER_SWEEP": "1",
                                     "MCP_PORT": "15920"}):
            self.assertEqual(self.ns["_container_sweep_overrides"](),
                             {"similarity_threshold": 0.38})

    def test_device_provider_selection(self):
        """`device` selects the ORT provider list; CPU stays last so an image or
        host without CUDA still starts instead of raising at session creation.
        Unknown values are rejected rather than defaulted to CPU — that default
        silently hid every typo (see _providers_for_device)."""
        providers = self.ns["_providers_for_device"]
        self.assertEqual(providers("cpu"), ["CPUExecutionProvider"])
        self.assertEqual(providers("gpu"),
                         ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(providers("CUDA"), providers("gpu"))
        self.assertEqual(providers(" GPU "), providers("gpu"))
        for value in ("", "npu", "tensorrt", None):
            with self.assertRaises(ValueError):
                providers(value)

    def test_sessions_take_device(self):
        """Both sessions must accept a device kwarg — the hardcoded CPU provider
        list was the reason `device: gpu` was a no-op before."""
        import inspect
        for cls in ("SCRFDDetector", "EdgeFaceAdapter"):
            params = inspect.signature(self.ns[cls].__init__).parameters
            self.assertIn("device", params, cls)
            self.assertEqual(params["device"].default, "cpu", cls)
        src = (Path(__file__).resolve().parents[1] / "plugins" / "face_scrfd.py").read_text()
        self.assertNotIn('providers=["CPUExecutionProvider"]', src)
        # One provider list per session: the SCRFD detector and the recognizer.
        self.assertIn("_providers_for_device(device)", src)
        self.assertIn("_providers_for_device(recognizer_device or device)", src)

    def test_split_device_overrides(self):
        """The detector and recognizer can be pinned to different providers:
        SCRFD-2.5G (all FP32) on CUDA, EdgeFace INT8 on CPU — the INT8 graph
        partitions across EPs because the CUDA EP has no DynamicQuantizeLinear."""
        import inspect
        params = inspect.signature(self.ns["EdgeFaceAdapter"].__init__).parameters
        for name in ("detector_device", "recognizer_device"):
            self.assertIn(name, params, name)
            self.assertIsNone(params[name].default, name)
        src = (Path(__file__).resolve().parents[1] / "plugins" / "face_scrfd.py").read_text()
        # Both overrides must fall back to `device` when unset.
        self.assertIn("_providers_for_device(recognizer_device or device)", src)
        self.assertIn("device=detector_device or device", src)
        # And the plugin must forward them from config into the adapter.
        self.assertIn("detector_device=self._detector_device", src)
        self.assertIn("recognizer_device=self._recognizer_device", src)

    def test_forward_reshape_contract(self):
        """2.5G outputs carry a leading batch dim / named outputs; _forward must
        normalize to 2D and index scores[:, 0]."""
        import types

        detector = self.ns["SCRFDDetector"].__new__(self.ns["SCRFDDetector"])
        detector._confidence = .5
        detector._cache = {}
        detector._input_name = "input"
        # 640x640 input: strides 8/16/32 → 80x80/40x40/20x20 grids, 2 anchors.
        def mk(h, w):
            return np.zeros((1, h * w * 2, 1), np.float32)
        det_out = np.zeros((1, 80 * 80 * 2, 4), np.float32)
        det_out_mid = np.zeros((1, 40 * 40 * 2, 4), np.float32)
        det_out_lo = np.zeros((1, 20 * 20 * 2, 4), np.float32)
        kps_out = np.zeros((1, 80 * 80 * 2, 10), np.float32)
        kps_out_mid = np.zeros((1, 40 * 40 * 2, 10), np.float32)
        kps_out_lo = np.zeros((1, 20 * 20 * 2, 10), np.float32)
        # 9 outputs: scores(3) + bboxes(3) + kps(3); all scores 1.0 on first cell
        outs = [mk(80, 80), mk(40, 40), mk(20, 20),
                det_out, det_out_mid, det_out_lo,
                kps_out, kps_out_mid, kps_out_lo]
        outs[0][0, 0, 0] = 0.9
        outs[3][0, 0] = [1, 2, 30, 40]
        detector._sess = types.SimpleNamespace(run=lambda _, __: outs)
        s, b, k = detector._forward(np.zeros((640, 640, 3), np.uint8), 0.5)
        self.assertEqual(len(s), 3)
        self.assertEqual(s[0].shape, (1,))
        self.assertEqual(b[0].shape, (1, 4))
        self.assertEqual(k[0].shape, (1, 5, 2))


if __name__ == "__main__":
    unittest.main()
