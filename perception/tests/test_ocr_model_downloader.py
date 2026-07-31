import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OCRModelDownloaderTest(unittest.TestCase):
    def test_default_bundle_contains_only_mnn_detector_and_recognizer(self):
        from utils.ocr_model_downloader import MODEL_FILES

        self.assertEqual(MODEL_FILES, ("det.mnn", "rec.mnn", "keys.txt"))

    def test_cli_accepts_explicit_model_filenames(self):
        from utils import ocr_model_downloader

        argv = [
            "ocr_model_downloader.py",
            "--base-url",
            "https://models.example.test/mnn",
            "--output-dir",
            "/models/ocr/mnn",
            "--filenames",
            "det.mnn",
            "rec.mnn",
            "keys.txt",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            ocr_model_downloader, "download_model"
        ) as download:
            ocr_model_downloader.main()

        download.assert_called_once_with(
            "https://models.example.test/mnn",
            "/models/ocr/mnn",
            filenames=("det.mnn", "rec.mnn", "keys.txt"),
        )

    def test_downloads_complete_bundle_without_checksum_pins(self):
        from utils.ocr_model_downloader import MODEL_FILES, download_model

        with tempfile.TemporaryDirectory() as source_tmp:
            with tempfile.TemporaryDirectory() as output_tmp:
                source = Path(source_tmp)
                for index, filename in enumerate(MODEL_FILES, start=1):
                    (source / filename).write_bytes(bytes([index]) * index)

                download_model(source.as_uri(), output_tmp)

                self.assertEqual(
                    {path.name for path in Path(output_tmp).iterdir()},
                    set(MODEL_FILES),
                )

    def test_rejects_empty_file_and_leaves_no_partial_bundle(self):
        from utils.ocr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as output_tmp:
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = b""

            with mock.patch(
                "utils.ocr_model_downloader.urlopen", return_value=response
            ), mock.patch("utils.ocr_model_downloader.time.sleep"):
                with self.assertRaisesRegex(ValueError, "empty"):
                    download_model("https://models.example.test", output_tmp)

            self.assertEqual(list(Path(output_tmp).iterdir()), [])

    def test_rejects_bundle_over_configured_limit(self):
        from utils.ocr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as output_tmp:
            def oversized_download(_url, destination):
                destination.write_bytes(b"x" * 6_000_000)

            with mock.patch(
                "utils.ocr_model_downloader.download_file",
                side_effect=oversized_download,
            ):
                with self.assertRaisesRegex(ValueError, "configured 15 byte"):
                    download_model(
                        "https://models.example.test",
                        output_tmp,
                        max_bundle_bytes=15,
                    )

            self.assertEqual(list(Path(output_tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
