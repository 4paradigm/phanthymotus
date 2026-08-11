from __future__ import annotations

import sys
import unittest
from pathlib import Path

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.semantic_navigation import SemanticNavigationPlugin  # noqa: E402


class SemanticNavigationPluginTest(unittest.TestCase):
    def test_minimal_plugin_contract(self) -> None:
        plugin = SemanticNavigationPlugin(
            {"enabled": True, "namespace": "ubuntu"}, "ubuntu", None
        )

        self.assertEqual(plugin.PREFIX, "semantic")
        self.assertEqual(plugin.get_tools(), [])
        self.assertIsNone(plugin.dispatch("semantic", {}))

    def test_bundle_loader_and_default_config_are_persistent(self) -> None:
        main = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")
        config = (PERCEPTION_ROOT / "config.yaml").read_text(encoding="utf-8")

        self.assertIn("from plugins.semantic_navigation import", main)
        self.assertIn("SemanticNavigationPlugin(", main)
        self.assertIn(
            '  semantic_navigation:\n'
            '    enabled: true\n'
            '    namespace: "ubuntu"\n',
            config,
        )


if __name__ == "__main__":
    unittest.main()
