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

    def test_default_config_is_bounded_for_ocr_leaderboard(self):
        config = (REPO_ROOT / "perception" / "config.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("  asr:\n    enabled: false\n    mode: offline", config)
        self.assertIn("  ocr:\n    enabled: true\n    provider: rapidocr", config)
        self.assertIn("model_dir: /models/ocr/ppocrv6-tiny", config)
        self.assertIn("    max_side_len: 1600", config)
        self.assertIn("    num_threads: 1", config)
        self.assertIn('    url: ""', config)
        self.assertIn('    key: ""', config)
        self.assertIn('    model: ""', config)

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

    def test_jetson_image_loads_large_message_fastdds_profile(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )
        service = (REPO_ROOT / "perception" / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        profile_path = (
            REPO_ROOT
            / "perception"
            / "config"
            / "fastdds_large_message.xml"
        )

        self.assertTrue(profile_path.is_file())
        profile = profile_path.read_text(encoding="utf-8")
        self.assertIn("<maxMessageSize>65000</maxMessageSize>", profile)
        self.assertIn("<sendBufferSize>8388608</sendBufferSize>", profile)
        self.assertIn("<receiveBufferSize>8388608</receiveBufferSize>", profile)
        self.assertIn("<useBuiltinTransports>true</useBuiltinTransports>", profile)
        self.assertIn(
            "COPY perception/config/fastdds_large_message.xml "
            "/opt/phanthy-motus/config/fastdds_large_message.xml",
            dockerfile,
        )
        self.assertIn(
            "FASTRTPS_DEFAULT_PROFILES_FILE="
            "/opt/phanthy-motus/config/fastdds_large_message.xml",
            service,
        )
        self.assertNotIn("FASTDDS_BUILTIN_TRANSPORTS", service)


if __name__ == "__main__":
    unittest.main()
