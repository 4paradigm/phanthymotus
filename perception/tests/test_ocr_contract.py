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
                    "max_input_mb": 16,
                    "max_decode_mb": 64,
                    "memory_guard": {
                        "enabled": True,
                        "expected_workers": 10,
                    },
                    "large_image_strategy": {
                        "enabled": True,
                        "trigger_side": 2400,
                    },
                }
            )

        self.assertIs(result, expected)
        adapter.assert_called_once_with(
            "/models/ocr/ppocrv6-tiny",
            device="cuda",
            device_id=0,
            gpu_mem_mb=512,
            use_angle_cls=True,
            num_threads=2,
            max_side_len=1600,
            max_input_mb=16,
            max_decode_mb=64,
            memory_guard={
                "enabled": True,
                "expected_workers": 10,
            },
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

    def test_adapter_signature_changes_with_device(self):
        cpu = self.ocr._adapter_signature(
            {"provider": "rapidocr", "device": "cpu"}
        )
        cuda = self.ocr._adapter_signature(
            {"provider": "rapidocr", "device": "cuda", "device_id": 0}
        )

        self.assertNotEqual(cpu, cuda)

    def test_adapter_initialization_failure_does_not_stop_bundle(self):
        with mock.patch(
            "plugins.ocr._build_ocr_adapter", side_effect=RuntimeError("load failed")
        ):
            with self.assertLogs("plugins.ocr", level="ERROR"):
                plugin = self.ocr.OCRPlugin(
                    {"provider": "rapidocr"}, object()
                )

        self.assertIsNone(plugin._adapter)

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

    def test_image_limit_error_only_fails_current_payload(self):
        adapter = mock.Mock()
        adapter.recognize.side_effect = [
            self.ocr_runtime.ImageTooLargeError("decode limit exceeded"),
            [{"text": "next", "bbox": [1, 2, 3, 4], "score": 0.9}],
        ]

        rejected = self.ocr.recognize_to_payload(
            adapter, b"large", "zh", 123.0
        )
        following = self.ocr.recognize_to_payload(
            adapter, b"normal", "zh", 124.0
        )

        self.assertEqual(rejected["items"], [])
        self.assertEqual(rejected["error"], "decode limit exceeded")
        self.assertEqual(following["text"], "next")
        self.assertNotIn("error", following)

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

    def test_rapidocr_adapter_decodes_compressed_image_before_inference(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._use_angle_cls = True
        adapter._max_side_len = 1600
        adapter._probe_image_header = mock.Mock(
            return_value=self.ocr_runtime.ImageHeader("JPEG", 200, 100)
        )
        adapter._inference_lock = threading.Lock()
        adapter._engine = mock.Mock(
            return_value=types.SimpleNamespace(boxes=[], txts=(), scores=())
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.INTER_AREA = 3
        decoded_image = types.SimpleNamespace(shape=(100, 200, 3))
        cv2_module.imdecode = mock.Mock(return_value=decoded_image)
        cv2_module.resize = mock.Mock()
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded-buffer")

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            result = adapter.recognize(b"jpeg-bytes")

        numpy_module.frombuffer.assert_called_once_with(b"jpeg-bytes", dtype="uint8")
        cv2_module.imdecode.assert_called_once_with("encoded-buffer", 1)
        adapter._engine.assert_called_once_with(
            decoded_image, use_det=True, use_cls=True, use_rec=True
        )
        self.assertEqual(result, [])

    def test_large_jpeg_uses_reduced_decode_and_restores_source_bbox(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._use_angle_cls = False
        adapter._max_side_len = 1600
        adapter._inference_lock = threading.Lock()
        adapter._engine = mock.Mock(
            return_value=types.SimpleNamespace(
                boxes=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
                txts=("large",),
                scores=(0.8,),
            )
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.INTER_AREA = 3
        reduced_image = types.SimpleNamespace(shape=(750, 1000, 3))
        cv2_module.imdecode = mock.Mock(return_value=reduced_image)
        cv2_module.resize = mock.Mock()
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded-buffer")
        jpeg = (
            b"\xff\xd8\xff\xc0\x00\x11\x08"
            + (3000).to_bytes(2, "big")
            + (4000).to_bytes(2, "big")
            + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"
        )

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            result = adapter.recognize(jpeg)

        cv2_module.imdecode.assert_called_once_with("encoded-buffer", 4)
        cv2_module.resize.assert_not_called()
        self.assertEqual(result[0]["bbox"], [40, 80, 400, 200])

    def test_oversized_non_jpeg_is_rejected_before_decode(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._large_image_strategy = None
        adapter._max_side_len = 960
        adapter._max_input_bytes = 16 * 1024 * 1024
        adapter._max_decode_bytes = 64 * 1024 * 1024
        adapter._probe_image_header = mock.Mock(
            return_value=self.ocr_runtime.ImageHeader("PNG", 10000, 10000)
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.imdecode = mock.Mock()
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded-buffer")

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            with self.assertRaisesRegex(
                self.ocr_runtime.ImageTooLargeError, "10000x10000"
            ):
                adapter.recognize(b"small-compressed-png")

        cv2_module.imdecode.assert_not_called()
        numpy_module.frombuffer.assert_not_called()

    def test_dynamic_memory_pressure_rejects_before_decode(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._large_image_strategy = None
        adapter._max_side_len = 960
        adapter._max_input_bytes = 16 * 1024 * 1024
        adapter._max_decode_bytes = 64 * 1024 * 1024
        adapter._memory_guard = mock.Mock()
        adapter._memory_guard.decode_limit_bytes.return_value = 8 * 1024 * 1024
        adapter._probe_image_header = mock.Mock(
            return_value=self.ocr_runtime.ImageHeader("PNG", 2000, 2000)
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.imdecode = mock.Mock()
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded-buffer")

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            with self.assertRaisesRegex(
                self.ocr_runtime.ImageTooLargeError, "dynamic limit"
            ):
                adapter.recognize(b"compressed-png")

        cv2_module.imdecode.assert_not_called()
        numpy_module.frombuffer.assert_not_called()

    def test_jpeg_too_large_after_reduced_decode_is_rejected_before_decode(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._large_image_strategy = None
        adapter._max_side_len = 960
        adapter._max_input_bytes = 16 * 1024 * 1024
        adapter._max_decode_bytes = 64 * 1024 * 1024
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.imdecode = mock.Mock()
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded-buffer")

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            with self.assertRaisesRegex(
                self.ocr_runtime.ImageTooLargeError, "65000x65000"
            ):
                adapter.recognize(self._jpeg(65000, 65000))

        cv2_module.imdecode.assert_not_called()
        numpy_module.frombuffer.assert_not_called()

    def test_single_pass_scales_float_polygon_before_rounding(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._large_image_strategy = None
        adapter._use_angle_cls = False
        adapter._max_side_len = 1600
        adapter._inference_lock = threading.Lock()
        adapter._engine = mock.Mock(
            return_value=types.SimpleNamespace(
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
            )
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.INTER_AREA = 3
        cv2_module.imdecode = mock.Mock(
            return_value=types.SimpleNamespace(shape=(750, 1000, 3))
        )
        cv2_module.resize = mock.Mock()
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded-buffer")

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            result = adapter.recognize(self._jpeg(4000, 3000))

        self.assertEqual(result[0]["bbox"], [43, 83, 401, 201])

    def test_small_jpeg_keeps_full_decode_resolution(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._use_angle_cls = False
        adapter._max_side_len = 1600
        adapter._inference_lock = threading.Lock()
        adapter._engine = mock.Mock(
            return_value=types.SimpleNamespace(
                boxes=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
                txts=("small",),
                scores=(0.8,),
            )
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.INTER_AREA = 3
        full_image = types.SimpleNamespace(shape=(600, 800, 3))
        cv2_module.imdecode = mock.Mock(return_value=full_image)
        cv2_module.resize = mock.Mock()
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded-buffer")
        jpeg = (
            b"\xff\xd8\xff\xc0\x00\x11\x08"
            + (600).to_bytes(2, "big")
            + (800).to_bytes(2, "big")
            + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"
        )

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            result = adapter.recognize(jpeg)

        cv2_module.imdecode.assert_called_once_with("encoded-buffer", 1)
        cv2_module.resize.assert_not_called()
        self.assertEqual(result[0]["bbox"], [10, 20, 100, 50])

    def test_non_jpeg_large_image_is_resized_and_bbox_is_restored(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._use_angle_cls = True
        adapter._max_side_len = 1600
        adapter._probe_image_header = mock.Mock(
            return_value=self.ocr_runtime.ImageHeader("PNG", 4000, 3000)
        )
        adapter._inference_lock = threading.Lock()
        adapter._engine = mock.Mock(
            return_value=types.SimpleNamespace(
                boxes=[[[10, 10], [100, 10], [100, 40], [10, 40]]],
                txts=("png",),
                scores=(0.7,),
            )
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.IMREAD_REDUCED_COLOR_2 = 2
        cv2_module.IMREAD_REDUCED_COLOR_4 = 4
        cv2_module.IMREAD_REDUCED_COLOR_8 = 8
        cv2_module.INTER_AREA = 3
        full_image = types.SimpleNamespace(shape=(3000, 4000, 3))
        resized_image = types.SimpleNamespace(shape=(1200, 1600, 3))
        cv2_module.imdecode = mock.Mock(return_value=full_image)
        cv2_module.resize = mock.Mock(return_value=resized_image)
        numpy_module = types.ModuleType("numpy")
        numpy_module.uint8 = "uint8"
        numpy_module.frombuffer = mock.Mock(return_value="encoded-buffer")

        with mock.patch.dict(
            sys.modules, {"cv2": cv2_module, "numpy": numpy_module}
        ):
            result = adapter.recognize(b"png-bytes")

        cv2_module.resize.assert_called_once_with(
            full_image, (1600, 1200), interpolation=3
        )
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
