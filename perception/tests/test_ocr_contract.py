import importlib
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))


def _install_ros_stubs():
    class FakeNode:
        def __init__(self, name):
            self.name = name

        def create_publisher(self, *args, **kwargs):
            return mock.Mock()

        def create_subscription(self, *args, **kwargs):
            return mock.Mock()

        def destroy_subscription(self, *args, **kwargs):
            return True

        def destroy_node(self):
            return True

    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = FakeNode
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.QoSProfile = lambda **kwargs: kwargs
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(
        RELIABLE="RELIABLE", BEST_EFFORT="BEST_EFFORT"
    )
    rclpy.qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="KEEP_LAST")
    rclpy.qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="VOLATILE")

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs.msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs.msg.CompressedImage = type("CompressedImage", (), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg.String = type("String", (), {})

    sys.modules.update(
        {
            "rclpy": rclpy,
            "rclpy.node": rclpy.node,
            "rclpy.qos": rclpy.qos,
            "sensor_msgs": sensor_msgs,
            "sensor_msgs.msg": sensor_msgs.msg,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs.msg,
        }
    )


class OCRContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ros_stubs()
        cls.ocr = importlib.import_module("plugins.ocr")
        cls.ocr_runtime = importlib.import_module("plugins.ocr_runtime")

    @staticmethod
    def _jpeg(width, height):
        return (
            b"\xff\xd8\xff\xc0\x00\x11\x08"
            + height.to_bytes(2, "big")
            + width.to_bytes(2, "big")
            + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"
        )

    def test_tool_contract_uses_compressed_images_and_json_results(self):
        tool = self.ocr.TOOLS[0]

        self.assertEqual(tool["name"], "ocr")
        self.assertEqual(
            tool["inputSchema"]["properties"]["action"]["enum"],
            ["start", "stop", "info", "config"],
        )
        self.assertEqual(
            tool["topic_in"],
            [{"format": "image/jpeg", "desc": "camera image input"}],
        )
        self.assertEqual(
            tool["topic_out"],
            [{"format": "data/json", "desc": "OCR result with text boxes"}],
        )
        self.assertEqual(
            tool["configSchema"]["properties"]["provider"]["enum"],
            ["rapidocr", "openai", "qwen", "tesseract"],
        )
        properties = tool["configSchema"]["properties"]
        self.assertEqual(
            properties["backend"]["enum"],
            ["mnn", "onnxruntime", "tensorrt"],
        )
        self.assertEqual(
            properties["fallback_backend"]["enum"],
            ["", "mnn", "onnxruntime"],
        )
        self.assertEqual(properties["max_side_len"]["default"], 1600)
        self.assertEqual(properties["det_unclip_ratio"]["default"], 0.7)
        self.assertEqual(properties["rec_min_score"]["default"], 0.9)

    def test_camera_and_result_topics_use_appropriate_reliability(self):
        self.assertEqual(self.ocr._CAMERA_QOS["reliability"], "RELIABLE")
        self.assertEqual(self.ocr._RESULT_QOS["reliability"], "RELIABLE")
        self.assertEqual(self.ocr._CAMERA_QOS["depth"], 1)
        self.assertEqual(self.ocr._RESULT_QOS["depth"], 1)

    def test_output_topic_is_derived_from_input_topic(self):
        self.assertEqual(
            self.ocr._ocr_output_topic("/robot/camera/image"),
            "/robot/camera/image/ocr",
        )

    def test_default_provider_builds_local_adapter(self):
        expected = object()
        with mock.patch(
            "plugins.ocr.RapidOCRAdapter", return_value=expected
        ) as adapter:
            result = self.ocr._build_ocr_adapter(
                {
                    "provider": "rapidocr",
                    "model_dir": "/models/ocr/ppocrv6-tiny",
                    "device": "cuda",
                    "device_id": 0,
                    "gpu_mem_mb": 512,
                    "use_angle_cls": True,
                    "num_threads": 2,
                    "max_side_len": 1600,
                    "large_image_strategy": {
                        "enabled": True,
                        "trigger_side": 2400,
                    },
                }
            )

        self.assertIs(result, expected)
        adapter.assert_called_once_with(
            "/models/ocr/ppocrv6-tiny",
            backend="onnxruntime",
            fallback_backend="",
            fallback_model_dir="",
            device="cuda",
            device_id=0,
            gpu_mem_mb=512,
            use_angle_cls=True,
            num_threads=2,
            max_side_len=1600,
            rec_min_score=0.9,
            enable_preprocess=True,
            det_thresh=0.3,
            det_box_thresh=0.5,
            det_unclip_ratio=0.7,
            large_image_strategy={
                "enabled": True,
                "trigger_side": 2400,
            },
        )

    def test_adapter_signature_changes_with_large_image_strategy(self):
        first = self.ocr._adapter_signature(
            {
                "provider": "rapidocr",
                "large_image_strategy": {"enabled": True, "max_tiles": 6},
            }
        )
        second = self.ocr._adapter_signature(
            {
                "provider": "rapidocr",
                "large_image_strategy": {"enabled": True, "max_tiles": 4},
            }
        )

        self.assertNotEqual(first, second)

    def test_adapter_signature_changes_with_inference_tuning(self):
        baseline = {
            "provider": "rapidocr",
            "rec_min_score": 0.9,
            "det_unclip_ratio": 0.7,
        }

        for key, value in (
            ("rec_min_score", 0.8),
            ("enable_preprocess", False),
            ("det_thresh", 0.2),
            ("det_box_thresh", 0.4),
            ("det_unclip_ratio", 1.0),
        ):
            changed = {**baseline, key: value}
            with self.subTest(key=key):
                self.assertNotEqual(
                    self.ocr._adapter_signature(baseline),
                    self.ocr._adapter_signature(changed),
                )

    def test_adapter_signature_changes_with_device(self):
        cpu = self.ocr._adapter_signature(
            {"provider": "rapidocr", "device": "cpu"}
        )
        cuda = self.ocr._adapter_signature(
            {"provider": "rapidocr", "device": "cuda", "device_id": 0}
        )

        self.assertNotEqual(cpu, cuda)

    def test_local_adapter_initialization_failure_is_fatal(self):
        with mock.patch(
            "plugins.ocr._build_ocr_adapter", side_effect=RuntimeError("load failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                self.ocr.OCRPlugin(
                    {"provider": "rapidocr"}, object()
                )

    def test_mnn_config_is_forwarded_to_local_adapter(self):
        expected = object()
        with mock.patch(
            "plugins.ocr.RapidOCRAdapter", return_value=expected
        ) as adapter:
            result = self.ocr._build_ocr_adapter(
                {
                    "provider": "rapidocr",
                    "backend": "mnn",
                    "fallback_backend": "",
                    "model_dir": "/models/ocr/ppocrv6-tiny-mnn",
                    "fallback_model_dir": "/models/ocr/ppocrv6-tiny-ort",
                    "use_angle_cls": False,
                    "num_threads": 1,
                    "max_side_len": 960,
                    "large_image_strategy": {"enabled": True},
                }
            )

        self.assertIs(result, expected)
        adapter.assert_called_once_with(
            "/models/ocr/ppocrv6-tiny-mnn",
            backend="mnn",
            fallback_backend="",
            fallback_model_dir="/models/ocr/ppocrv6-tiny-ort",
            device="cpu",
            device_id=0,
            gpu_mem_mb=512,
            use_angle_cls=False,
            num_threads=1,
            max_side_len=960,
            rec_min_score=0.9,
            enable_preprocess=True,
            det_thresh=0.3,
            det_box_thresh=0.5,
            det_unclip_ratio=0.7,
            large_image_strategy={"enabled": True},
        )

    def test_mnn_session_uses_low_memory_config_and_uint8_pointer(self):
        import numpy as np

        state = types.SimpleNamespace(config=None, conversion=None)

        class FakeTensor:
            def getShape(self):
                return [1, 1, 1, 1]

            def copyToHostTensor(self, _host):
                state.copied_to_host = True

            def getNumpyData(self):
                return np.array([[[[0.75]]]], dtype=np.float32)

        class FakeHostTensor:
            def __init__(self, *_args):
                pass

            def getData(self):
                return [0.75]

        class FakeInterpreter:
            def __init__(self, model_path):
                state.model_path = model_path

            def createSession(self, config):
                state.config = config
                return object()

            def getSessionInput(self, _session):
                return FakeTensor()

            def resizeTensor(self, _tensor, shape):
                state.shape = shape

            def resizeSession(self, _session):
                state.resized = True

            def runSession(self, _session):
                state.ran = True

            def getSessionOutput(self, _session):
                return FakeTensor()

        class FakeImageProcess:
            def __init__(self, config):
                state.image_config = config

            def convert(self, ptr, width, height, stride, destination):
                state.conversion = (ptr, width, height, stride, destination)

        mnn = types.ModuleType("MNN")
        mnn.Interpreter = FakeInterpreter
        mnn.CVImageProcess = FakeImageProcess
        mnn.CV_ImageFormat_RGB = "RGB"
        mnn.CV_ImageFormat_BGR = "BGR"
        mnn.CV_Filter_BILINEAL = "BILINEAR"
        mnn.Tensor = FakeHostTensor
        mnn.Halide_Type_Float = "float"
        mnn.Tensor_DimensionType_Caffe = "caffe"

        with tempfile.TemporaryDirectory() as model_tmp:
            model_path = Path(model_tmp) / "det.mnn"
            model_path.write_bytes(b"model")
            with mock.patch.dict(sys.modules, {"MNN": mnn}):
                session = self.ocr_runtime._MNNModelSession(
                    model_path,
                    num_threads=1,
                    mean=(127.5, 127.5, 127.5),
                    normal=(1 / 127.5, 1 / 127.5, 1 / 127.5),
                )
                image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
                output = session.run_uint8(image, (1, 3, 2, 3))

        self.assertEqual(
            state.config,
            {
                "backend": "CPU",
                "numThread": 1,
                "precision": "low",
                "memory": 2,
                "power": 0,
            },
        )
        self.assertEqual(state.shape, (1, 3, 2, 3))
        self.assertEqual(state.conversion[:4], (image.ctypes.data, 3, 2, 9))
        self.assertEqual(output.shape, (1, 1, 1, 1))

    def test_mnn_adapter_requires_no_classifier_model(self):
        with tempfile.TemporaryDirectory() as model_tmp:
            root = Path(model_tmp)
            for filename in ("det.mnn", "rec.mnn", "keys.txt"):
                (root / filename).write_bytes(b"model")

            with mock.patch("plugins.ocr_runtime._MNNPipeline") as pipeline:
                adapter = self.ocr_runtime.RapidOCRAdapter(
                    str(root),
                    backend="mnn",
                    use_angle_cls=False,
                    num_threads=1,
                )

        pipeline.assert_called_once_with(
            root,
            num_threads=1,
            max_side_len=1600,
            rec_min_score=0.9,
            enable_preprocess=True,
            det_thresh=0.3,
            det_box_thresh=0.5,
            det_unclip_ratio=0.7,
        )
        pipeline.return_value.warm_up.assert_called_once_with()
        self.assertEqual(adapter._backend_name, "mnn")

    def test_tensorrt_adapter_loads_external_engines(self):
        with tempfile.TemporaryDirectory() as model_tmp:
            root = Path(model_tmp)
            for filename in ("det.engine", "rec.engine", "keys.txt"):
                (root / filename).write_bytes(b"model")

            with mock.patch(
                "plugins.ocr_runtime._TensorRTPipeline"
            ) as pipeline:
                adapter = self.ocr_runtime.RapidOCRAdapter(
                    str(root),
                    backend="tensorrt",
                    device="cuda",
                    device_id=0,
                    use_angle_cls=False,
                )

        pipeline.assert_called_once_with(
            root,
            device_id=0,
            max_side_len=1600,
            rec_min_score=0.9,
            enable_preprocess=True,
            det_thresh=0.3,
            det_box_thresh=0.5,
            det_unclip_ratio=0.7,
        )
        pipeline.return_value.warm_up.assert_called_once_with()
        self.assertEqual(adapter._backend_name, "tensorrt")

    def test_tensorrt_adapter_falls_back_to_mnn(self):
        with tempfile.TemporaryDirectory() as primary_tmp, \
                tempfile.TemporaryDirectory() as fallback_tmp:
            primary = Path(primary_tmp)
            fallback = Path(fallback_tmp)
            for filename in ("det.engine", "rec.engine", "keys.txt"):
                (primary / filename).write_bytes(b"model")
            for filename in ("det.mnn", "rec.mnn", "keys.txt"):
                (fallback / filename).write_bytes(b"model")

            with mock.patch(
                "plugins.ocr_runtime._TensorRTPipeline",
                side_effect=RuntimeError("engine rejected"),
            ), mock.patch("plugins.ocr_runtime._MNNPipeline") as mnn:
                adapter = self.ocr_runtime.RapidOCRAdapter(
                    str(primary),
                    backend="tensorrt",
                    fallback_backend="mnn",
                    fallback_model_dir=str(fallback),
                    device="cuda",
                    use_angle_cls=False,
                    num_threads=1,
                )

        mnn.assert_called_once_with(
            fallback,
            num_threads=1,
            max_side_len=1600,
            rec_min_score=0.9,
            enable_preprocess=True,
            det_thresh=0.3,
            det_box_thresh=0.5,
            det_unclip_ratio=0.7,
        )
        mnn.return_value.warm_up.assert_called_once_with()
        self.assertEqual(adapter._backend_name, "mnn")

    def test_tensorrt_adapter_requires_cuda_and_disables_classifier(self):
        with self.assertRaisesRegex(ValueError, "requires device='cuda'"):
            self.ocr_runtime.RapidOCRAdapter(
                "/models/ocr/trt", backend="tensorrt", device="cpu"
            )

        with tempfile.TemporaryDirectory() as model_tmp:
            root = Path(model_tmp)
            for filename in ("det.engine", "rec.engine", "keys.txt"):
                (root / filename).write_bytes(b"model")
            with self.assertRaisesRegex(ValueError, "angle classifier"):
                self.ocr_runtime.RapidOCRAdapter(
                    str(root), backend="tensorrt", device="cuda"
                )

    def test_tensorrt_session_executes_supported_dynamic_shape(self):
        import contextlib
        import numpy as np

        state = types.SimpleNamespace(shape=None, addresses={}, executed=False)

        class FakeTensor:
            def __init__(self, value):
                self.value = np.asarray(value)
                self.dtype = "float32"

            def to(self, **_kwargs):
                return self

            def data_ptr(self):
                return id(self.value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class FakeStream:
            cuda_stream = 123

            @staticmethod
            def synchronize():
                state.synchronized = True

        class FakeContext:
            @staticmethod
            def set_input_shape(_name, shape):
                state.shape = shape
                return True

            @staticmethod
            def get_tensor_shape(_name):
                return (1, 1, 32, 32)

            @staticmethod
            def set_tensor_address(name, address):
                state.addresses[name] = address

            @staticmethod
            def execute_async_v3(stream):
                state.executed = stream == 123
                return True

        class FakeEngine:
            num_io_tensors = 2
            num_optimization_profiles = 1

            @staticmethod
            def create_execution_context():
                return FakeContext()

            @staticmethod
            def get_tensor_name(index):
                return ("x", "y")[index]

            @staticmethod
            def get_tensor_mode(name):
                return "input" if name == "x" else "output"

            @staticmethod
            def get_tensor_shape(_name):
                return (-1, 3, -1, -1)

            @staticmethod
            def get_tensor_profile_shape(_name, _index):
                return (
                    (1, 3, 32, 32),
                    (1, 3, 64, 64),
                    (1, 3, 128, 128),
                )

            @staticmethod
            def get_tensor_dtype(_name):
                return "float32"

        class FakeRuntime:
            def __init__(self, _logger):
                pass

            @staticmethod
            def deserialize_cuda_engine(_data):
                return FakeEngine()

        trt = types.ModuleType("tensorrt")
        trt.Logger = type("Logger", (), {
            "WARNING": 1,
            "__init__": lambda self, _level: None,
        })
        trt.Runtime = FakeRuntime
        trt.TensorIOMode = types.SimpleNamespace(
            INPUT="input", OUTPUT="output"
        )
        trt.nptype = lambda _dtype: np.float32

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device=lambda _device: contextlib.nullcontext(),
            Stream=lambda **_kwargs: FakeStream(),
            stream=lambda _stream: contextlib.nullcontext(),
        )
        torch.from_numpy = lambda value: FakeTensor(value)
        torch.empty = lambda shape, **_kwargs: FakeTensor(
            np.zeros(shape, dtype=np.float32)
        )

        with tempfile.TemporaryDirectory() as model_tmp:
            engine_path = Path(model_tmp) / "det.engine"
            engine_path.write_bytes(b"engine")
            with mock.patch.dict(
                sys.modules, {"tensorrt": trt, "torch": torch}
            ):
                session = self.ocr_runtime._TensorRTModelSession(
                    engine_path,
                    device_id=0,
                    mean=(127.5, 127.5, 127.5),
                    normal=(1 / 127.5, 1 / 127.5, 1 / 127.5),
                )
                output = session.run_uint8(
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    (1, 3, 32, 32),
                )
                with self.assertRaisesRegex(ValueError, "outside profiles"):
                    session.run_uint8(
                        np.zeros((160, 160, 3), dtype=np.uint8),
                        (1, 3, 160, 160),
                    )

        self.assertEqual(state.shape, (1, 3, 32, 32))
        self.assertEqual(set(state.addresses), {"x", "y"})
        self.assertTrue(state.executed)
        self.assertTrue(state.synchronized)
        self.assertEqual(output.shape, (1, 1, 32, 32))

    def test_tensorrt_session_selects_nearest_profile_and_batch_limit(self):
        session = object.__new__(self.ocr_runtime._TensorRTModelSession)
        session._profiles = [
            (
                (1, 3, 48, 320),
                (8, 3, 48, 320),
                (16, 3, 48, 320),
            ),
            (
                (1, 3, 48, 320),
                (2, 3, 48, 640),
                (4, 3, 48, 1024),
            ),
        ]

        self.assertEqual(session._select_profile((8, 3, 48, 320)), 0)
        self.assertEqual(session._select_profile((2, 3, 48, 640)), 1)
        self.assertEqual(session.max_batch_size(48, 320), 16)
        self.assertEqual(session.max_batch_size(48, 640), 4)
        with self.assertRaisesRegex(
            self.ocr_runtime.TensorRTShapeError, "outside profiles"
        ):
            session.max_batch_size(48, 2048)

    def test_tensorrt_pipeline_batches_equal_width_crops_in_box_order(self):
        import numpy as np

        pipeline = object.__new__(self.ocr_runtime._TensorRTPipeline)
        pipeline._enable_preprocess = False
        pipeline._rec_min_score = 0.9
        boxes = np.asarray(
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[20, 0], [30, 0], [30, 10], [20, 10]],
                [[40, 0], [50, 0], [50, 10], [40, 10]],
            ],
            dtype=np.float32,
        )
        pipeline._detect = mock.Mock(return_value=(boxes, [1.0] * 3))
        pipeline._crop = mock.Mock(side_effect=[object(), object(), object()])
        pipeline._prepare_recognition_crop = mock.Mock(side_effect=[
            (np.zeros((48, 320, 3), dtype=np.uint8), 1.0),
            (np.zeros((48, 328, 3), dtype=np.uint8), 2.0),
            (np.zeros((48, 320, 3), dtype=np.uint8), 3.0),
        ])
        pipeline._rec = mock.Mock()
        pipeline._rec.max_batch_size.return_value = 8
        pipeline._rec.run_uint8_batch.side_effect = lambda images, _shape: (
            np.zeros((len(images), 2, 2), dtype=np.float32)
        )

        def decode(_prediction, _use_space_char, *, wh_ratio_list, **_kwargs):
            return (
                [(f"text-{int(ratio)}", 0.95) for ratio in wh_ratio_list],
                [],
            )

        pipeline._rec_decode = mock.Mock(side_effect=decode)

        result = pipeline.infer(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(
            [item["text"] for item in result],
            ["text-1", "text-2", "text-3"],
        )
        calls = pipeline._rec.run_uint8_batch.call_args_list
        self.assertEqual(calls[0].args[0].shape, (2, 48, 320, 3))
        self.assertEqual(calls[0].args[1], (2, 3, 48, 320))
        self.assertEqual(calls[1].args[0].shape, (1, 48, 328, 3))

    def test_tensorrt_shape_error_lazily_loads_mnn_fallback_once(self):
        adapter = object.__new__(self.ocr_runtime.RapidOCRAdapter)
        adapter._inference_lock = threading.Lock()
        adapter._pipeline = mock.Mock()
        adapter._pipeline.infer.side_effect = (
            self.ocr_runtime.TensorRTShapeError("outside profiles")
        )
        fallback = mock.Mock()
        fallback.infer.return_value = [{"text": "fallback"}]
        adapter._runtime_fallback_pipeline = None
        adapter._runtime_fallback_loader = mock.Mock(return_value=fallback)

        first = adapter._infer_image(object())
        second = adapter._infer_image(object())

        self.assertEqual(first, [{"text": "fallback"}])
        self.assertEqual(second, first)
        adapter._runtime_fallback_loader.assert_called_once_with()
        self.assertEqual(fallback.infer.call_count, 2)

    def test_mnn_pipeline_closes_detector_when_recognizer_load_fails(self):
        det_session = mock.Mock()
        det_utils = types.ModuleType("rapidocr.ch_ppocr_det.utils")
        det_utils.DBPostProcess = mock.Mock()
        rec_utils = types.ModuleType("rapidocr.ch_ppocr_rec.utils")
        rec_utils.CTCLabelDecode = mock.Mock()
        image_utils = types.ModuleType("rapidocr.utils.process_img")
        image_utils.get_rotate_crop_image = mock.Mock()
        modules = {
            "rapidocr": types.ModuleType("rapidocr"),
            "rapidocr.ch_ppocr_det": types.ModuleType("rapidocr.ch_ppocr_det"),
            "rapidocr.ch_ppocr_det.utils": det_utils,
            "rapidocr.ch_ppocr_rec": types.ModuleType("rapidocr.ch_ppocr_rec"),
            "rapidocr.ch_ppocr_rec.utils": rec_utils,
            "rapidocr.utils": types.ModuleType("rapidocr.utils"),
            "rapidocr.utils.process_img": image_utils,
        }

        with mock.patch.dict(sys.modules, modules), mock.patch(
            "plugins.ocr_runtime._MNNModelSession",
            side_effect=[det_session, RuntimeError("rec load failed")],
        ):
            with self.assertRaisesRegex(RuntimeError, "rec load failed"):
                self.ocr_runtime._MNNPipeline(
                    Path("/models/ocr/mnn"),
                    num_threads=1,
                    max_side_len=960,
                )

        det_session.close.assert_called_once_with()

    def test_mnn_recognition_padding_is_neutral_before_normalization(self):
        import numpy as np

        pipeline = object.__new__(self.ocr_runtime._MNNPipeline)
        captured = {}
        pipeline._rec = mock.Mock()

        def run_uint8(image, shape):
            captured["image"] = image.copy()
            captured["shape"] = shape
            return np.zeros((1, 2, 2), dtype=np.float32)

        pipeline._rec.run_uint8.side_effect = run_uint8
        pipeline._rec_decode = mock.Mock(
            return_value=([("text", 0.9)], [])
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.INTER_LINEAR = 1
        cv2_module.resize = mock.Mock(
            return_value=np.full((48, 100, 3), 200, dtype=np.uint8)
        )
        crop = np.zeros((48, 100, 3), dtype=np.uint8)

        with mock.patch.dict(sys.modules, {"cv2": cv2_module}):
            result = pipeline._recognize_crop(crop)

        self.assertEqual(result, ("text", 0.9))
        self.assertEqual(captured["shape"], (1, 3, 48, 320))
        self.assertTrue(np.all(captured["image"][:, :100] == 200))
        self.assertTrue(np.all(captured["image"][:, 100:] == 128))

    def test_single_pass_uses_bounded_vips_overview_for_any_source_size(self):
        adapter = object.__new__(self.ocr_runtime.RapidOCRAdapter)
        adapter._max_side_len = 960
        adapter._probe_image_header = mock.Mock(
            return_value=self.ocr_runtime.ImageHeader("PNG", 12000, 9000)
        )
        adapter._infer_image = mock.Mock(
            return_value=[
                {"text": "bounded", "bbox": [10, 20, 110, 60], "score": 0.9}
            ]
        )
        overview = types.SimpleNamespace(shape=(720, 960, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview",
            return_value=overview,
        ) as decode:
            result = adapter._recognize_single_pass(b"compressed-image")

        decode.assert_called_once_with(b"compressed-image", 960)
        self.assertEqual(result[0]["bbox"], [125, 250, 1375, 750])

    def test_rapidocr_output_normalizes_polygon_to_pixel_bbox(self):
        output = types.SimpleNamespace(
            boxes=[
                [[10.2, 20.8], [110.4, 19.9], [111.0, 50.1], [9.7, 51.2]]
            ],
            txts=("你好 123",),
            scores=(0.9876,),
        )

        items = self.ocr.normalize_rapidocr_output(output)

        self.assertEqual(
            items,
            [
                {
                    "text": "你好 123",
                    "bbox": [9, 19, 111, 52],
                    "score": 0.9876,
                }
            ],
        )

    def test_rapidocr_output_scales_bbox_to_source_image(self):
        output = types.SimpleNamespace(
            boxes=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
            txts=("scaled",),
            scores=(0.9,),
        )

        items = self.ocr.normalize_rapidocr_output(
            output, scale_x=4.0, scale_y=3.0
        )

        self.assertEqual(items[0]["bbox"], [40, 60, 400, 150])

    def test_inference_error_becomes_publishable_empty_payload(self):
        adapter = mock.Mock()
        adapter.recognize.side_effect = ValueError("invalid image")

        payload = self.ocr.recognize_to_payload(
            adapter, b"not-an-image", "zh", 123.0
        )

        self.assertEqual(
            payload,
            {
                "text": "",
                "items": [],
                "error": "invalid image",
                "timestamp": 123.0,
                "language": "zh",
            },
        )

    def test_rapidocr_adapter_uses_only_external_cpu_models(self):
        fake_engine = mock.Mock()
        captured_config = {}

        def create_engine(*, config_path):
            captured_config.update(
                json.loads(Path(config_path).read_text(encoding="utf-8"))
            )
            return fake_engine

        rapidocr_module = types.ModuleType("rapidocr")
        rapidocr_module.RapidOCR = mock.Mock(side_effect=create_engine)
        rapidocr_main_module = types.ModuleType("rapidocr.main")

        with tempfile.TemporaryDirectory() as model_dir:
            root = Path(model_dir)
            for name in ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt"):
                (root / name).write_bytes(b"model")
            default_config = root / "default.yaml"
            default_config.write_text(
                json.dumps(
                    {
                        "Det": {},
                        "Cls": {},
                        "Rec": {},
                        "Global": {},
                        "EngineConfig": {},
                    }
                ),
                encoding="utf-8",
            )
            rapidocr_main_module.DEFAULT_CFG_PATH = str(default_config)
            yaml_module = types.ModuleType("yaml")
            yaml_module.safe_load = json.load
            yaml_module.safe_dump = lambda data, stream, **_kwargs: json.dump(
                data, stream
            )
            with mock.patch.dict(
                sys.modules,
                {
                    "rapidocr": rapidocr_module,
                    "rapidocr.main": rapidocr_main_module,
                    "yaml": yaml_module,
                },
            ):
                self.ocr.RapidOCRAdapter(
                    model_dir,
                    device="cpu",
                    use_angle_cls=True,
                    num_threads=2,
                    max_side_len=1600,
                )

        self.assertEqual(Path(captured_config["Det"]["model_path"]).name, "det.onnx")
        self.assertEqual(Path(captured_config["Cls"]["model_path"]).name, "cls.onnx")
        self.assertEqual(Path(captured_config["Rec"]["model_path"]).name, "rec.onnx")
        self.assertEqual(Path(captured_config["Rec"]["rec_keys_path"]).name, "keys.txt")
        self.assertEqual(captured_config["Det"]["model_type"], "tiny")
        self.assertEqual(captured_config["Det"]["ocr_version"], "PP-OCRv6")
        self.assertEqual(captured_config["Det"]["thresh"], 0.3)
        self.assertEqual(captured_config["Det"]["box_thresh"], 0.5)
        self.assertEqual(captured_config["Det"]["unclip_ratio"], 0.7)
        self.assertEqual(captured_config["Cls"]["model_type"], "mobile")
        self.assertEqual(captured_config["Cls"]["ocr_version"], "PP-OCRv4")
        self.assertEqual(captured_config["Rec"]["model_type"], "tiny")
        self.assertEqual(captured_config["Rec"]["ocr_version"], "PP-OCRv6")
        self.assertTrue(captured_config["Global"]["use_cls"])
        engine_config = captured_config["EngineConfig"]["onnxruntime"]
        self.assertEqual(engine_config["intra_op_num_threads"], 2)
        self.assertEqual(engine_config["inter_op_num_threads"], 1)
        self.assertFalse(engine_config["use_cuda"])
        self.assertEqual(captured_config["Global"]["max_side_len"], 1600)

    def test_rapidocr_adapter_configures_cuda_execution_provider(self):
        captured_config = {}

        def create_engine(*, config_path):
            captured_config.update(
                json.loads(Path(config_path).read_text(encoding="utf-8"))
            )
            return mock.Mock()

        rapidocr_module = types.ModuleType("rapidocr")
        rapidocr_module.RapidOCR = mock.Mock(side_effect=create_engine)
        rapidocr_main_module = types.ModuleType("rapidocr.main")
        onnxruntime_module = types.ModuleType("onnxruntime")
        onnxruntime_module.get_available_providers = mock.Mock(
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        with tempfile.TemporaryDirectory() as model_dir:
            root = Path(model_dir)
            for name in ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt"):
                (root / name).write_bytes(b"model")
            default_config = root / "default.yaml"
            default_config.write_text(
                json.dumps(
                    {
                        "Det": {}, "Cls": {}, "Rec": {}, "Global": {},
                        "EngineConfig": {},
                    }
                ),
                encoding="utf-8",
            )
            rapidocr_main_module.DEFAULT_CFG_PATH = str(default_config)
            yaml_module = types.ModuleType("yaml")
            yaml_module.safe_load = json.load
            yaml_module.safe_dump = lambda data, stream, **_kwargs: json.dump(
                data, stream
            )
            with mock.patch.dict(
                sys.modules,
                {
                    "rapidocr": rapidocr_module,
                    "rapidocr.main": rapidocr_main_module,
                    "onnxruntime": onnxruntime_module,
                    "yaml": yaml_module,
                },
            ):
                self.ocr.RapidOCRAdapter(
                    model_dir,
                    device="cuda",
                    device_id=0,
                    gpu_mem_mb=512,
                    num_threads=1,
                )

        engine_config = captured_config["EngineConfig"]["onnxruntime"]
        self.assertTrue(engine_config["use_cuda"])
        self.assertEqual(engine_config["intra_op_num_threads"], 1)
        self.assertEqual(
            engine_config["cuda_ep_cfg"],
            {"device_id": 0, "gpu_mem_limit": 512 * 1024 * 1024},
        )

    def test_cuda_adapter_rejects_missing_cuda_execution_provider(self):
        onnxruntime_module = types.ModuleType("onnxruntime")
        onnxruntime_module.get_available_providers = mock.Mock(
            return_value=["CPUExecutionProvider"]
        )

        with tempfile.TemporaryDirectory() as model_dir:
            root = Path(model_dir)
            for name in ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt"):
                (root / name).write_bytes(b"model")
            with mock.patch.dict(
                sys.modules, {"onnxruntime": onnxruntime_module}
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "CUDAExecutionProvider"
                ):
                    self.ocr.RapidOCRAdapter(model_dir, device="cuda")

    def test_large_jpeg_delegates_to_strategy(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._large_image_strategy = mock.Mock()
        adapter._large_image_strategy.should_handle.return_value = True
        adapter._large_image_strategy.recognize.return_value = [
            {"text": "tile"}
        ]
        adapter._infer_image = mock.Mock()
        image_bytes = self._jpeg(4000, 3000)

        result = adapter.recognize(image_bytes)

        self.assertEqual(result, [{"text": "tile"}])
        adapter._large_image_strategy.should_handle.assert_called_once_with(
            (4000, 3000)
        )
        adapter._large_image_strategy.recognize.assert_called_once_with(
            image_bytes, adapter._infer_image
        )

    def test_small_image_keeps_existing_single_pass_path(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._large_image_strategy = mock.Mock()
        adapter._large_image_strategy.should_handle.return_value = False
        adapter._recognize_single_pass = mock.Mock(
            return_value=[
                {"text": "small", "bbox": [10, 20, 100, 50]}
            ]
        )
        image_bytes = self._jpeg(800, 600)

        result = adapter.recognize(image_bytes)

        self.assertEqual(result[0]["bbox"], [10, 20, 100, 50])
        adapter._recognize_single_pass.assert_called_once_with(image_bytes)
        adapter._large_image_strategy.recognize.assert_not_called()

    def test_strategy_uses_same_locked_engine_callback(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._use_angle_cls = True
        adapter._inference_lock = mock.MagicMock()
        adapter._engine = mock.Mock(
            return_value=types.SimpleNamespace(boxes=[], txts=(), scores=())
        )
        image = object()

        result = adapter._infer_image(image)

        self.assertEqual(result, [])
        adapter._inference_lock.__enter__.assert_called_once_with()
        adapter._inference_lock.__exit__.assert_called_once()
        adapter._engine.assert_called_once_with(
            image, use_det=True, use_cls=True, use_rec=True
        )

    def test_shared_adapter_serializes_complete_large_image_requests(self):
        first_entered = threading.Event()
        both_entered = threading.Event()
        release = threading.Event()
        state_lock = threading.Lock()
        state = {"active": 0, "max_active": 0}

        class TrackingStrategy:
            @staticmethod
            def should_handle(_source_size):
                return True

            @staticmethod
            def recognize(_image_bytes, _infer_image):
                with state_lock:
                    state["active"] += 1
                    state["max_active"] = max(
                        state["max_active"], state["active"]
                    )
                    if state["active"] == 1:
                        first_entered.set()
                    if state["active"] == 2:
                        both_entered.set()
                release.wait(timeout=2)
                with state_lock:
                    state["active"] -= 1
                return []

        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._request_lock = threading.Lock()
        adapter._large_image_strategy = TrackingStrategy()
        adapter._infer_image = mock.Mock()
        image_bytes = self._jpeg(4000, 3000)
        first = threading.Thread(target=adapter.recognize, args=(image_bytes,))
        second = threading.Thread(target=adapter.recognize, args=(image_bytes,))

        first.start()
        self.assertTrue(first_entered.wait(timeout=1))
        second.start()
        both_entered.wait(timeout=0.1)
        release.set()
        first.join(timeout=1)
        second.join(timeout=1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(state["max_active"], 1)

    def _single_pass_adapter(self, source_size, output, *, max_side=1600):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._large_image_strategy = None
        adapter._use_angle_cls = False
        adapter._max_side_len = max_side
        adapter._probe_image_header = mock.Mock(
            return_value=self.ocr_runtime.ImageHeader(
                "JPEG", source_size[0], source_size[1]
            )
        )
        adapter._inference_lock = threading.Lock()
        adapter._rec_min_score = 0.3
        adapter._engine = mock.Mock(return_value=output)
        return adapter

    def test_rapidocr_adapter_uses_bounded_overview_before_inference(self):
        decoded_image = types.SimpleNamespace(shape=(100, 200, 3))
        adapter = self._single_pass_adapter(
            (200, 100),
            types.SimpleNamespace(boxes=[], txts=(), scores=()),
        )

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview",
            return_value=decoded_image,
        ) as decode:
            result = adapter.recognize(b"jpeg-bytes")

        decode.assert_called_once_with(b"jpeg-bytes", 1600)
        adapter._engine.assert_called_once_with(
            decoded_image, use_det=True, use_cls=False, use_rec=True
        )
        self.assertEqual(result, [])

    def test_large_image_overview_restores_source_bbox(self):
        adapter = self._single_pass_adapter(
            (4000, 3000),
            types.SimpleNamespace(
                boxes=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
                txts=("large",),
                scores=(0.8,),
            ),
        )
        overview = types.SimpleNamespace(shape=(750, 1000, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview", return_value=overview
        ):
            result = adapter.recognize(b"image-bytes")

        self.assertEqual(result[0]["bbox"], [40, 80, 400, 200])

    def test_single_pass_scales_float_polygon_before_rounding(self):
        adapter = self._single_pass_adapter(
            (4000, 3000),
            types.SimpleNamespace(
                boxes=[
                    [
                        [10.8, 20.8],
                        [100.2, 20.8],
                        [100.2, 50.2],
                        [10.8, 50.2],
                    ]
                ],
                txts=("precise",),
                scores=(0.8,),
            ),
        )
        overview = types.SimpleNamespace(shape=(750, 1000, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview", return_value=overview
        ):
            result = adapter.recognize(b"image-bytes")

        self.assertEqual(result[0]["bbox"], [43, 83, 401, 201])

    def test_small_image_keeps_source_coordinates(self):
        adapter = self._single_pass_adapter(
            (800, 600),
            types.SimpleNamespace(
                boxes=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
                txts=("small",),
                scores=(0.8,),
            ),
        )
        overview = types.SimpleNamespace(shape=(600, 800, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview", return_value=overview
        ):
            result = adapter.recognize(b"image-bytes")

        self.assertEqual(result[0]["bbox"], [10, 20, 100, 50])

    def test_non_jpeg_overview_restores_source_bbox(self):
        adapter = self._single_pass_adapter(
            (4000, 3000),
            types.SimpleNamespace(
                boxes=[[[10, 10], [100, 10], [100, 40], [10, 40]]],
                txts=("png",),
                scores=(0.7,),
            ),
        )
        adapter._probe_image_header.return_value = self.ocr_runtime.ImageHeader(
            "PNG", 4000, 3000
        )
        overview = types.SimpleNamespace(shape=(1200, 1600, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview", return_value=overview
        ):
            result = adapter.recognize(b"png-bytes")

        self.assertEqual(result[0]["bbox"], [25, 25, 250, 100])

    def test_frame_queue_keeps_only_latest_image(self):
        node = self.ocr._OCRNode("/camera", object())

        self.assertEqual(node._frame_queue.maxsize, 1)

    def test_language_only_instance_config_reuses_shared_adapter(self):
        shared_adapter = object()
        executor = mock.Mock()
        with mock.patch(
            "plugins.ocr._build_ocr_adapter", return_value=shared_adapter
        ) as build:
            plugin = self.ocr.OCRPlugin(
                {
                    "provider": "rapidocr",
                    "model_dir": "/models/ocr/ppocrv6-tiny",
                    "language": "zh",
                },
                executor,
            )
            plugin.dispatch(
                "ocr",
                {"action": "config", "instance_id": "case-1", "language": "en"},
            )
            with mock.patch.object(self.ocr, "_OCRNode") as node_type:
                node_type.return_value.start.return_value = {"state": "running"}
                plugin.dispatch(
                    "ocr",
                    {
                        "action": "start",
                        "instance_id": "case-1",
                        "input_topic": "/camera",
                    },
                )

        self.assertEqual(build.call_count, 1)
        node_type.assert_called_once_with(
            "/camera",
            shared_adapter,
            "en",
            node_suffix="case_1",
            min_interval=0.0,
        )

    def test_empty_shared_config_reuses_adapter(self):
        shared_adapter = object()
        with mock.patch(
            "plugins.ocr._build_ocr_adapter", return_value=shared_adapter
        ) as build:
            plugin = self.ocr.OCRPlugin(
                {"provider": "rapidocr", "language": "zh"}, mock.Mock()
            )
            result = plugin.dispatch("ocr", {"action": "config"})

        self.assertEqual(build.call_count, 1)
        self.assertIs(plugin._adapter, shared_adapter)
        self.assertTrue(result["reused"])

    def test_shared_tuning_config_rebuilds_adapter(self):
        shared_adapter = object()
        tuned_adapter = object()
        with mock.patch(
            "plugins.ocr._build_ocr_adapter",
            side_effect=[shared_adapter, tuned_adapter],
        ) as build:
            plugin = self.ocr.OCRPlugin(
                {"provider": "rapidocr", "rec_min_score": 0.9}, mock.Mock()
            )
            result = plugin.dispatch(
                "ocr", {"action": "config", "rec_min_score": 0.8}
            )

        self.assertEqual(build.call_count, 2)
        self.assertIs(plugin._adapter, tuned_adapter)
        self.assertFalse(result["reused"])

    def test_repeated_start_stop_reuses_one_ros_node(self):
        executor = mock.Mock()
        node = mock.Mock(_input_topic="/camera", state="idle")
        node.start.return_value = {"state": "running"}
        node.stop.return_value = {"state": "idle"}

        with mock.patch("plugins.ocr._build_ocr_adapter", return_value=object()):
            plugin = self.ocr.OCRPlugin({"provider": "rapidocr"}, executor)
        with mock.patch.object(self.ocr, "_OCRNode", return_value=node) as node_type:
            for _ in range(250):
                plugin.dispatch(
                    "ocr",
                    {
                        "action": "start",
                        "input_topic": "/camera",
                    },
                )
                plugin.dispatch("ocr", {"action": "stop"})

        node_type.assert_called_once()
        executor.add_node.assert_called_once_with(node)
        executor.remove_node.assert_not_called()
        node.destroy_node.assert_not_called()
        self.assertIs(plugin._nodes["/camera"], node)
        self.assertEqual(node.start.call_count, 250)
        self.assertEqual(node.stop.call_count, 250)

    def test_concurrent_starts_create_one_ocr_node(self):
        executor = mock.Mock()
        first_constructor_entered = threading.Event()
        release_first_constructor = threading.Event()

        def build_node(*_args, **_kwargs):
            first_constructor_entered.set()
            release_first_constructor.wait(timeout=1)
            node = mock.Mock(_input_topic="/camera", state="idle")
            node.start.return_value = {"state": "running"}
            return node

        with mock.patch("plugins.ocr._build_ocr_adapter", return_value=object()):
            plugin = self.ocr.OCRPlugin({"provider": "rapidocr"}, executor)

        errors = []

        def start():
            try:
                plugin.dispatch(
                    "ocr",
                    {
                        "action": "start",
                        "input_topic": "/camera",
                    },
                )
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(self.ocr, "_OCRNode", side_effect=build_node) as node_type:
            first = threading.Thread(target=start)
            second = threading.Thread(target=start)
            first.start()
            self.assertTrue(first_constructor_entered.wait(timeout=1))
            second.start()
            time.sleep(0.05)
            release_first_constructor.set()
            first.join(timeout=1)
            second.join(timeout=1)

        self.assertEqual(errors, [])
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        node_type.assert_called_once()
        executor.add_node.assert_called_once()

    def test_stale_worker_cannot_publish_after_restart(self):
        old_inference_started = threading.Event()
        release_old_inference = threading.Event()

        def recognize(_adapter, image_bytes, _language, _timestamp):
            if image_bytes == b"old":
                old_inference_started.set()
                release_old_inference.wait(timeout=1)
            return {"text": image_bytes.decode(), "items": []}

        node = self.ocr._OCRNode("/camera", object())
        with mock.patch("plugins.ocr.recognize_to_payload", side_effect=recognize):
            node.start()
            first_stop_event = node._stop_event
            node._image_cb(types.SimpleNamespace(data=b"old", format="jpeg"))
            self.assertTrue(old_inference_started.wait(timeout=1))

            old_worker = node._worker_thread
            with mock.patch.object(old_worker, "join", return_value=None):
                node.stop()
            node.start()
            second_stop_event = node._stop_event
            release_old_inference.set()
            time.sleep(0.05)

            node._image_cb(types.SimpleNamespace(data=b"new", format="jpeg"))
            deadline = time.time() + 1
            while node._pub.publish.call_count < 1 and time.time() < deadline:
                time.sleep(0.01)
            node.stop()

        self.assertIsNot(first_stop_event, second_stop_event)
        published = [json.loads(call.args[0].data)["text"] for call in node._pub.publish.call_args_list]
        self.assertEqual(published, ["new"])

    def test_topic_change_retires_node_without_destroying_it(self):
        executor = mock.Mock()
        old_node = mock.Mock(_input_topic="/camera/old")
        new_node = mock.Mock(_input_topic="/camera/new")
        old_node.start.return_value = {"state": "running"}
        old_node.stop.return_value = {"state": "idle"}
        new_node.start.return_value = {"state": "running"}

        with mock.patch("plugins.ocr._build_ocr_adapter", return_value=object()):
            plugin = self.ocr.OCRPlugin({"provider": "rapidocr"}, executor)
        with mock.patch.object(
            self.ocr, "_OCRNode", side_effect=[old_node, new_node]
        ):
            plugin.dispatch(
                "ocr",
                {
                    "action": "start",
                    "instance_id": "case-1",
                    "input_topic": "/camera/old",
                },
            )
            plugin.dispatch(
                "ocr",
                {
                    "action": "start",
                    "instance_id": "case-1",
                    "input_topic": "/camera/new",
                },
            )

        old_node.stop.assert_called_once_with()
        executor.remove_node.assert_called_once_with(old_node)
        old_node.destroy_node.assert_not_called()
        self.assertEqual(plugin._retired_nodes, [old_node])
        self.assertIs(plugin._nodes["case-1"], new_node)

    def test_instance_config_updates_existing_node_without_removing_it(self):
        executor = mock.Mock()
        shared_adapter = object()
        node = mock.Mock(_input_topic="/camera")
        node.start.return_value = {"state": "running"}
        node.stop.return_value = {"state": "idle"}

        with mock.patch(
            "plugins.ocr._build_ocr_adapter", return_value=shared_adapter
        ):
            plugin = self.ocr.OCRPlugin(
                {"provider": "rapidocr", "language": "zh"}, executor
            )
        with mock.patch.object(self.ocr, "_OCRNode", return_value=node):
            plugin.dispatch(
                "ocr",
                {
                    "action": "start",
                    "instance_id": "case-1",
                    "input_topic": "/camera",
                },
            )
            plugin.dispatch(
                "ocr",
                {"action": "config", "instance_id": "case-1", "language": "en"},
            )

        node.stop.assert_called_once_with()
        executor.remove_node.assert_not_called()
        node.destroy_node.assert_not_called()
        self.assertIs(node._adapter, shared_adapter)
        self.assertEqual(node._language, "en")

    def test_nodes_are_destroyed_only_during_final_shutdown(self):
        plugin = object.__new__(self.ocr.OCRPlugin)
        plugin._lifecycle_lock = threading.RLock()
        active = mock.Mock()
        retired = mock.Mock()
        plugin._nodes = {"case-1": active}
        plugin._retired_nodes = [retired, active]

        plugin.prepare_shutdown()

        active.stop.assert_called_once_with()
        retired.stop.assert_called_once_with()
        active.destroy_node.assert_not_called()
        retired.destroy_node.assert_not_called()

        plugin.destroy_nodes()

        active.destroy_node.assert_called_once_with()
        retired.destroy_node.assert_called_once_with()
        self.assertEqual(plugin._nodes, {})
        self.assertEqual(plugin._retired_nodes, [])


if __name__ == "__main__":
    unittest.main()
