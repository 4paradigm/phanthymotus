import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from tools import build_ocr_tensorrt_engines as builder  # noqa: E402


class OCRTensorRTBuilderTest(unittest.TestCase):
    def test_detector_profiles_cover_all_resized_dimensions(self):
        for height in range(32, 1601, 32):
            for width in range(32, 1601, 32):
                shape = (1, 3, height, width)
                self.assertTrue(
                    any(
                        profile.contains(shape)
                        for profile in builder.DETECTOR_PROFILES
                    ),
                    shape,
                )

    def test_recognizer_profiles_cover_supported_widths(self):
        for width in range(320, 2049, 8):
            shape = (1, 3, 48, width)
            self.assertTrue(
                any(
                    profile.contains(shape)
                    for profile in builder.RECOGNIZER_PROFILES
                ),
                shape,
            )

    def test_classifier_profile_supports_batch_eight(self):
        for batch in range(1, 9):
            shape = (batch, 3, 48, 192)
            self.assertTrue(
                any(
                    profile.contains(shape)
                    for profile in builder.CLASSIFIER_PROFILES
                ),
                shape,
            )

    def test_classifier_only_build_uses_classifier_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classifier = root / "cls.onnx"
            keys = root / "keys.txt"
            output = root / "output"
            classifier.write_bytes(b"cls")
            keys.write_text("key", encoding="utf-8")
            argv = [
                "build_ocr_tensorrt_engines.py",
                "--component", "cls",
                "--cls-onnx", str(classifier),
                "--keys", str(keys),
                "--output-dir", str(output),
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                builder, "build_engine"
            ) as build_engine:
                builder.main()

            build_engine.assert_called_once_with(
                classifier,
                output / "cls.engine",
                builder.CLASSIFIER_PROFILES,
                workspace_mb=512,
                optimization_level=3,
            )

    def test_recognizer_only_build_does_not_require_detector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rec = root / "rec.onnx"
            keys = root / "keys.txt"
            output = root / "output"
            rec.write_bytes(b"rec")
            keys.write_text("key", encoding="utf-8")
            argv = [
                "build_ocr_tensorrt_engines.py",
                "--component", "rec",
                "--rec-onnx", str(rec),
                "--keys", str(keys),
                "--output-dir", str(output),
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                builder, "build_engine"
            ) as build_engine:
                builder.main()

            build_engine.assert_called_once_with(
                rec,
                output / "rec.engine",
                builder.RECOGNIZER_PROFILES,
                workspace_mb=512,
                optimization_level=3,
            )
            self.assertEqual(
                (output / "keys.txt").read_text(encoding="utf-8"), "key"
            )


if __name__ == "__main__":
    unittest.main()
