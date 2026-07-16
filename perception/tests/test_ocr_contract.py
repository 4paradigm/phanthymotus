import importlib
import json
import sys
import tempfile
import threading
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
            use_angle_cls=True,
            num_threads=2,
            max_side_len=1600,
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

    def test_rapidocr_adapter_decodes_compressed_image_before_inference(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._use_angle_cls = True
        adapter._max_side_len = 1600
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
            "/camera", shared_adapter, "en", node_suffix="case_1"
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

    def test_removing_node_stops_removes_and_destroys_it(self):
        plugin = object.__new__(self.ocr.OCRPlugin)
        node = mock.Mock()
        node.worker_alive = False
        plugin._nodes = {"case-1": node}
        plugin._executor = mock.Mock()
        plugin._retired_nodes = []

        plugin._remove_node("case-1")

        node.stop.assert_called_once_with()
        plugin._executor.remove_node.assert_called_once_with(node)
        node.destroy_node.assert_called_once_with()
        self.assertNotIn("case-1", plugin._nodes)


if __name__ == "__main__":
    unittest.main()
