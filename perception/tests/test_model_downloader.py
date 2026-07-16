"""模型下载器测试。"""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).parents[1] / "utils" / "model_downloader.py"
SPEC = importlib.util.spec_from_file_location("model_downloader", MODULE_PATH)
model_downloader = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(model_downloader)


class ModelDownloaderTest(unittest.TestCase):
    def test_download_files_uses_base_url_and_atomic_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_download(url, destination):
                calls.append(url)
                pathlib.Path(destination).write_bytes(b"model")

            with mock.patch.object(
                model_downloader,
                "urlretrieve",
                side_effect=fake_download,
            ):
                model_downloader._download_files(
                    ("det/inference.onnx",),
                    directory,
                    "http://models.internal/ppocr/",
                )

            self.assertEqual(
                calls,
                ["http://models.internal/ppocr/det/inference.onnx"],
            )
            self.assertEqual(
                (pathlib.Path(directory) / "det" / "inference.onnx").read_bytes(),
                b"model",
            )

    def test_ocr_download_requires_private_base_url(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "base_url is required"):
                model_downloader.ensure_model(
                    "ocr_ppocrv6_tiny",
                    directory,
                )

    def test_ocr_download_checks_all_required_files(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = pathlib.Path(directory)
            (model_dir / "inference.yml").write_text("config", encoding="utf-8")
            downloaded = []

            def fake_download(files, target_dir, base_url):
                downloaded.extend(files)
                for filename in files:
                    (pathlib.Path(target_dir) / filename).write_bytes(b"model")

            with mock.patch.object(
                model_downloader,
                "_download_files",
                side_effect=fake_download,
            ):
                model_downloader.ensure_model(
                    "ocr_ppocrv6_tiny",
                    directory,
                    base_url="http://models.internal/ocr/",
                )

            self.assertEqual(
                downloaded,
                ["det.onnx", "rec.onnx", "inference.yml"],
            )


if __name__ == "__main__":
    unittest.main()
