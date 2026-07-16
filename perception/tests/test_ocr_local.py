"""离线 OCR 纯逻辑单元测试。"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).parents[1] / "plugins" / "ocr_local.py"
SPEC = importlib.util.spec_from_file_location("ocr_local", MODULE_PATH)
ocr_local = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ocr_local)


class OCRLocalHelpersTest(unittest.TestCase):
    def test_sanitize_bbox_orders_and_clips_coordinates(self):
        self.assertEqual(
            ocr_local.sanitize_bbox([120.4, -2, 10.2, 88.8], 100, 80),
            [10, 0, 99, 79],
        )

    def test_sanitize_bbox_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            ocr_local.sanitize_bbox([1, 2, 3], 100, 100)

    def test_sort_ocr_results_uses_reading_order(self):
        results = [
            {"text": "第二行", "bbox": [5, 30, 30, 40]},
            {"text": "world", "bbox": [50, 5, 90, 15]},
            {"text": "hello", "bbox": [5, 7, 40, 17]},
        ]
        sorted_results = ocr_local.sort_ocr_results(results)
        self.assertEqual(
            [item["text"] for item in sorted_results],
            ["hello", "world", "第二行"],
        )

    def test_load_characters_reserves_ctc_blank(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "inference.yml"
            path.write_text(
                "PostProcess:\n  character_dict:\n    - A\n    - 中\n",
                encoding="utf-8",
            )
            self.assertEqual(
                ocr_local.PPOCRv6ONNXAdapter._load_characters(str(path)),
                ["blank", "A", "中"],
            )

    def test_environment_model_url_overrides_yaml(self):
        with mock.patch.dict(
            "os.environ",
            {"OCR_MODEL_BASE_URL": "http://runtime/ocr/"},
        ):
            self.assertEqual(
                ocr_local.resolve_model_base_url(
                    {"model_base_url": "http://yaml/ocr/"}
                ),
                "http://runtime/ocr/",
            )

    def test_yaml_model_url_is_fallback(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                ocr_local.resolve_model_base_url(
                    {"model_base_url": "http://yaml/ocr/"}
                ),
                "http://yaml/ocr/",
            )


if __name__ == "__main__":
    unittest.main()
