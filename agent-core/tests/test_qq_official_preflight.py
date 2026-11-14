"""Preflight report must identify disabled capabilities without making live claims."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qq_official_preflight import inspect_database


class QQPreflightTests(unittest.TestCase):
    def _db(self, folder):
        path = Path(folder) / "test.db"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
            channels = [{
                "id": "qq-test", "platform": "qq", "enabled": True,
                "config": {
                    "protocol": "official", "app_id": "1900000000",
                    "enabled_scopes": ["c2c", "group"],
                    "enable_inbound_media": True,
                    "enable_outbound_media": False,
                    "enable_typing": True,
                },
            }]
            db.execute("INSERT INTO config VALUES (?, ?)", (
                "channel_configs", json.dumps(channels)))
            db.execute("INSERT INTO config VALUES (?, ?)", (
                "event", json.dumps({"llm": {"vision_input": False}})))
        return path

    def test_report_lists_unverified_capabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            report = inspect_database(self._db(folder), "qq-test")
        self.assertTrue(report["channel_enabled"])
        self.assertTrue(report["features_enabled"]["enable_inbound_media"])
        self.assertFalse(report["features_enabled"]["enable_outbound_media"])
        self.assertIn("image understanding requires VLM",
                      report["not_ready_or_unverified"])
        self.assertNotIn("channel_configs", report)
        self.assertNotIn("config", report)

    def test_missing_channel_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "not found"):
                inspect_database(self._db(folder), "different-channel")


if __name__ == "__main__":
    unittest.main()
