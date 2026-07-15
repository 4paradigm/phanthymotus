import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OCRModelDownloaderTest(unittest.TestCase):
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
            def empty_download(_url, destination):
                Path(destination).write_bytes(b"")

            with mock.patch(
                "utils.ocr_model_downloader.urlretrieve",
                side_effect=empty_download,
            ):
                with self.assertRaisesRegex(ValueError, "empty"):
                    download_model("https://models.example.test", output_tmp)

            self.assertEqual(list(Path(output_tmp).iterdir()), [])

    def test_rejects_bundle_over_fifteen_mebibytes(self):
        from utils.ocr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as output_tmp:
            def oversized_download(_url, destination):
                Path(destination).write_bytes(b"x" * 4_000_000)

            with mock.patch(
                "utils.ocr_model_downloader.urlretrieve",
                side_effect=oversized_download,
            ):
                with self.assertRaisesRegex(ValueError, "15 MiB"):
                    download_model("https://models.example.test", output_tmp)

            self.assertEqual(list(Path(output_tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
