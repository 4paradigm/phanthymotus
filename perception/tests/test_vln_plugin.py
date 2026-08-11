from __future__ import annotations

import sys
import unittest
from pathlib import Path

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.vln import VisionLanguageNavigationPlugin  # noqa: E402


class VisionLanguageNavigationPluginTest(unittest.TestCase):
    def test_minimal_plugin_contract(self) -> None:
        plugin = VisionLanguageNavigationPlugin(
            {"enabled": True, "namespace": ""}, "test_host", None
        )

        self.assertEqual(plugin.PREFIX, "vln")
        self.assertEqual(plugin.get_tools(), [])
        self.assertIsNone(plugin.dispatch("vln", {}))

    def test_bundle_loader_and_default_config_are_persistent(self) -> None:
        main = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")
        config = (PERCEPTION_ROOT / "config.yaml").read_text(encoding="utf-8")

        self.assertIn("from plugins.vln import VisionLanguageNavigationPlugin", main)
        self.assertIn("VisionLanguageNavigationPlugin(", main)
        self.assertIn(
            '  vln:\n'
            '    enabled: true\n'
            '    namespace: ""\n',
            config,
        )


if __name__ == "__main__":
    unittest.main()
