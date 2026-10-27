"""Reviewed supplements must not turn ordinary words into finance tickers."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


FRONTEND = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/frontend"


def load_module():
    package = types.ModuleType("_hotword_test_frontend")
    package.__path__ = [str(FRONTEND)]
    with patch.dict(sys.modules, {package.__name__: package}):
        spec = importlib.util.spec_from_file_location(package.__name__ + ".hotwords", FRONTEND / "hotwords.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def entry(text="WEN", **overrides):
    return dict({"text": text, "arpa": ["W", "EH1", "N", "D", "IY0", "Z"],
                 "case": "exact", "languages": ["en"], "domain": "finance", "priority": 10,
                 "override_cmu": True, "enabled": True, "reviewed": True,
                 "source": "reviewed-test-fixture", "source_sha256": "a" * 64}, **overrides)


class HotwordTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.module = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "reviewed.json"
        bundled = Path(self.tmp.name) / "bundled.json"
        bundled.write_text(json.dumps({"version": 1, "entries": []}))
        self.module.bundled_hotwords_source = lambda: (
            str(bundled), hashlib.sha256(bundled.read_bytes()).hexdigest())

    def manifest(self, entries):
        self.path.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
        os.environ["VITS2_HOTWORDS_PATH"] = str(self.path)
        self.module.reload_hotwords()

    def test_default_has_no_word_specific_inline_overrides(self):
        for token in ("Airbnb", "airbnb", "AIRBNB", "ASML", "asml", "WEN", "Wen",
                      "car", "CAR", "apps", "APPS", "NVIDIA", "CUDA"):
            with self.subTest(token=token):
                self.assertIsNone(self.module.lookup_hotword(token))

    def test_lookup_is_whole_token_not_substring(self):
        for token in ("myAirbnb", "Airbnbish", "foo_Airbnb_bar", "Airbnb2", "Airbnb ASML", "Airbnb."):
            with self.subTest(token=token):
                self.assertIsNone(self.module.lookup_hotword(token))

    def test_language_is_token_language_and_cmu_guard(self):
        self.manifest([entry(text="AIRBNB", case="fold", domain="*", override_cmu=False)])
        self.assertIsNone(self.module.lookup_hotword("Airbnb", language="zh"))
        self.assertIsNotNone(self.module.lookup_hotword("Airbnb", language="en"))
        self.assertIsNone(self.module.lookup_hotword("Airbnb", has_cmu=True))

    def test_domain_and_exact_case_gate_finance_override(self):
        self.manifest([entry()])
        self.assertIsNone(self.module.lookup_hotword("WEN", has_cmu=True))
        os.environ["VITS2_HOTWORD_DOMAIN"] = "finance"
        self.assertEqual(self.module.lookup_hotword("WEN", has_cmu=True), [entry()["arpa"]])
        for token in ("Wen", "wen", "WENDY", "newWEN", "foo_WEN"):
            self.assertIsNone(self.module.lookup_hotword(token, has_cmu=True))
        self.assertIsNone(self.module.lookup_hotword("WEN", domain="general", has_cmu=True))

    def test_explicit_priorities_resolve_overlapping_entries(self):
        self.manifest([entry(priority=1), entry(priority=2, arpa=["W", "EH1", "N"])])
        self.assertEqual(self.module.lookup_hotword("WEN", domain="finance", has_cmu=True), [["W", "EH1", "N"]])
        self.manifest([entry(priority=1), entry(priority=1, arpa=["W", "EH1", "N"])])
        with self.assertRaisesRegex(ValueError, "priorities"):
            self.module.lookup_hotword("WEN", domain="finance")

    def test_disabled_candidates_and_unreviewed_enabled(self):
        self.manifest([entry(enabled=False, reviewed=False)])
        self.assertIsNone(self.module.lookup_hotword("WEN", domain="finance"))
        self.manifest([entry(reviewed=False)])
        with self.assertRaisesRegex(ValueError, "reviewed"):
            self.module.lookup_hotword("WEN", domain="finance")

    def test_same_path_changes_require_explicit_reload(self):
        self.manifest([entry()])
        self.assertEqual(self.module.lookup_hotword("WEN", domain="finance"), [entry()["arpa"]])
        self.path.write_text(json.dumps({"version": 1, "entries": [entry(arpa=["W", "EH1", "N"])]}))
        # Atomic configuration: a disk edit alone cannot change queued requests.
        self.assertEqual(self.module.lookup_hotword("WEN", domain="finance"), [entry()["arpa"]])
        self.module.reload_hotwords()
        self.assertEqual(self.module.lookup_hotword("WEN", domain="finance"), [["W", "EH1", "N"]])

    def test_rejects_unsafe_cmu_override_scopes(self):
        for change in ({"case": "fold"}, {"domain": "general"}, {"domain": "*"}):
            with self.subTest(change=change):
                self.manifest([entry(**change)])
                with self.assertRaisesRegex(ValueError, "CMU overrides"):
                    self.module.lookup_hotword("WEN", domain="finance")

    def test_arpa_and_model_symbol_validation(self):
        for arpa in ([], ["AH"], ["AH3"], ["T1"], ["AX1"], ["BAD"], [1], [["W"]]):
            with self.subTest(arpa=arpa):
                self.manifest([entry(arpa=arpa)])
                with self.assertRaises(ValueError):
                    self.module.lookup_hotword("WEN", domain="finance")

    def test_raw_gold_and_oversized_files_are_not_autoactivated(self):
        os.environ["VITS2_HOTWORDS_PATH"] = str(self.path)
        self.path.write_text(json.dumps({"WEN": ["W", "EH1", "N"]}))
        with self.assertRaisesRegex(ValueError, "raw Gold"):
            self.module.lookup_hotword("WEN")
        self.path.write_bytes(b" " * (64 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            self.module.lookup_hotword("WEN")

    def test_optional_sha_pin_and_provenance_are_checked(self):
        self.manifest([entry()])
        os.environ["VITS2_HOTWORDS_SHA256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertIsNotNone(self.module.lookup_hotword("WEN", domain="finance"))
        os.environ["VITS2_HOTWORDS_SHA256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.module.lookup_hotword("WEN", domain="finance")
        del os.environ["VITS2_HOTWORDS_SHA256"]
        self.manifest([entry(source_sha256="unverified")])
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.module.lookup_hotword("WEN", domain="finance")

    def test_cache_does_not_leak_across_files_env_or_caller_mutation(self):
        self.manifest([entry()])
        selected = self.module.lookup_hotword("WEN", domain="finance")
        selected[0][0] = "BAD"
        self.assertEqual(self.module.lookup_hotword("WEN", domain="finance"), [entry()["arpa"]])
        self.path = Path(self.tmp.name) / "other.json"
        self.manifest([entry(arpa=["W", "EH1", "N"])])
        self.assertEqual(self.module.lookup_hotword("WEN", domain="finance"), [["W", "EH1", "N"]])
        del os.environ["VITS2_HOTWORDS_PATH"]
        self.assertIsNone(self.module.lookup_hotword("WEN", domain="finance"))


if __name__ == "__main__":
    unittest.main()
