"""OCR 模型包下载脚本测试。"""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (
    pathlib.Path(__file__).parents[1] / "scripts" / "download_ocr_model.py"
)
SPEC = importlib.util.spec_from_file_location("download_ocr_model", MODULE_PATH)
download_ocr_model = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(download_ocr_model)


class DownloadOCRModelTest(unittest.TestCase):
    def test_download_model_installs_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "models"

            def fake_download(url, destination):
                destination.write_bytes(url.rsplit("/", 1)[-1].encode("utf-8"))

            with mock.patch.object(
                download_ocr_model,
                "download_file",
                side_effect=fake_download,
            ):
                download_ocr_model.download_model(
                    "http://models.internal/ocr",
                    str(output),
                )

            self.assertEqual((output / "det.onnx").read_bytes(), b"det.onnx")
            self.assertEqual((output / "rec.onnx").read_bytes(), b"rec.onnx")
            self.assertEqual(
                (output / "inference.yml").read_bytes(),
                b"inference.yml",
            )

    def test_download_model_rejects_oversized_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "models"

            def fake_download(_url, destination):
                destination.write_bytes(b"1234")

            with mock.patch.object(
                download_ocr_model,
                "download_file",
                side_effect=fake_download,
            ):
                with self.assertRaises(ValueError):
                    download_ocr_model.download_model(
                        "http://models.internal/ocr",
                        str(output),
                        max_bundle_bytes=3,
                    )


if __name__ == "__main__":
    unittest.main()
