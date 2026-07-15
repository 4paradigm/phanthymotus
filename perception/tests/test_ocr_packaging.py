import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class OCRPackagingTest(unittest.TestCase):
    def test_bundle_registers_ocr_plugin(self):
        source = (REPO_ROOT / "perception" / "main.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("from plugins.ocr import OCRPlugin", source)
        self.assertIn(
            'plugins_cfg.get("ocr", {}).get("enabled", False)', source
        )

    def test_default_config_enables_local_ocr_without_changing_asr(self):
        config = (REPO_ROOT / "perception" / "config.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("  asr:\n    enabled: true\n    mode: offline", config)
        self.assertIn("  ocr:\n    enabled: true\n    provider: rapidocr", config)
        self.assertIn("model_dir: /models/ocr/ppocrv6-tiny", config)

    def test_jetson_image_uses_external_ocr_models(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )

        self.assertIn("rapidocr==3.9.1", dockerfile)
        self.assertIn("--no-deps", dockerfile)
        self.assertIn("onnxruntime", dockerfile)
        self.assertIn("rapidocr.__file__", dockerfile)
        self.assertIn("-name '*.onnx' -delete", dockerfile)
        self.assertIn("ocr_model_downloader.py", dockerfile)
        self.assertIn(
            "http://172.28.4.81:34567/zengzhitao/embodied-ai/ppocrv6-tiny",
            dockerfile,
        )
        self.assertIn(
            "http://172.28.4.81:34567/zengzhitao/embodied-ai/ocr/ppocrv6-tiny",
            dockerfile,
        )
        self.assertIn("/models/ocr/ppocrv6-tiny", dockerfile)
        self.assertNotIn("COPY perception/models", dockerfile)

    def test_ocr_does_not_override_process_wide_fastdds_transport(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("FASTRTPS_DEFAULT_PROFILES_FILE", dockerfile)
        self.assertFalse(
            (REPO_ROOT / "perception" / "config" / "fastdds_large_message.xml").exists()
        )


if __name__ == "__main__":
    unittest.main()
