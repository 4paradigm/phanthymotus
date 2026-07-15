import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))


def _install_ros_stubs():
    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = object
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
        self.assertEqual(self.ocr._CAMERA_QOS["reliability"], "BEST_EFFORT")
        self.assertEqual(self.ocr._RESULT_QOS["reliability"], "RELIABLE")

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
                }
            )

        self.assertIs(result, expected)
        adapter.assert_called_once_with(
            "/models/ocr/ppocrv6-tiny", use_angle_cls=True, num_threads=2
        )

    def test_adapter_initialization_failure_does_not_stop_bundle(self):
        with mock.patch(
            "plugins.ocr._build_ocr_adapter", side_effect=RuntimeError("load failed")
        ):
            plugin = self.ocr.OCRPlugin({"provider": "rapidocr"}, object())

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
            import yaml

            captured_config.update(
                yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
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
                "Det: {}\nCls: {}\nRec: {}\nGlobal: {}\nEngineConfig: {}\n",
                encoding="utf-8",
            )
            rapidocr_main_module.DEFAULT_CFG_PATH = str(default_config)
            with mock.patch.dict(
                sys.modules,
                {
                    "rapidocr": rapidocr_module,
                    "rapidocr.main": rapidocr_main_module,
                },
            ):
                self.ocr.RapidOCRAdapter(
                    model_dir, use_angle_cls=True, num_threads=2
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

    def test_rapidocr_adapter_decodes_compressed_image_before_inference(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._use_angle_cls = True
        adapter._engine = mock.Mock(
            return_value=types.SimpleNamespace(boxes=[], txts=(), scores=())
        )
        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.imdecode = mock.Mock(return_value="decoded-image")
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
            "decoded-image", use_det=True, use_cls=True, use_rec=True
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
