"""External lexicon compilation, lookup and CMU contracts, without inference.

Download, process-lock and resource-cache behavior stays in
``test_vits2_lexicon_resources.py``. The published CMU check at the end is
optional unless VITS2_MODEL_DIR points at the verified release.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pytest

PERCEPTION = Path(__file__).resolve().parents[1]
PLUGIN = PERCEPTION / "plugins/vits2_tts_trt"
FRONTEND = PLUGIN / "frontend"


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module("_external_lexicon_builder_test", PERCEPTION / "tools/vits2_prepare_external_lexicon.py")


class _TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)


class _RuntimeCase(_TemporaryCase):
    """One isolated environment and frontend package per lookup/binding case."""
    def setUp(self):
        super().setUp()
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.package_name = "_vits2_external_lexicon_tests"
        package = types.ModuleType(self.package_name)
        package.__path__ = [str(PLUGIN)]
        frontend = types.ModuleType(self.package_name + ".frontend")
        frontend.__path__ = [str(FRONTEND)]
        modules = patch.dict(sys.modules, {self.package_name: package, frontend.__name__: frontend})
        modules.start()
        self.addCleanup(modules.stop)
        self.module = _load_module(frontend.__name__ + ".external_lexicon", FRONTEND / "external_lexicon.py")
        self.paths = sys.modules[frontend.__name__ + ".release_paths"]


class CompilationTests(_TemporaryCase):
    """Pinned source, lossless compilation and default CMU conflict repair."""
    def setUp(self):
        super().setUp()
        self.source = self.directory / "source.json"
        self.gold = {"ABC": ["EY1", "B", "IY1", "S", "IY1"],
                     "ACME": ["AE1", "K", "M", "IY0"],
                     "CAR": ["S", "IY1", "EY1", "AA1", "R"]}
        self.cmu = {"CAR": [["K", "AA1", "R"]], "OTHER": [["AH1"], ["DH", "ER0"]]}
        self.source.write_text(json.dumps(self.gold))

    def compile(self, **kwargs):
        kwargs.setdefault("expected_source_sha256", hashlib.sha256(self.source.read_bytes()).hexdigest())
        return builder.compile_lexicon(self.source, self.cmu, **kwargs)

    def test_every_source_entry_is_preserved_and_conflicts_keep_cmu(self):
        payload, audit = self.compile()
        self.assertEqual(set(payload["entries"]), set(self.gold))
        self.assertEqual(payload["entries"]["CAR"], {
            "arpa": ["K", "AA1", "R"], "mode": "cmu", "original_arpa": self.gold["CAR"]})
        self.assertEqual(audit["pronunciation_conflicts_defaulted_to_cmu"], 1)
        self.assertFalse(audit["acoustic_acceptance"])
        self.assertEqual(payload["cmu"]["canonical_layout"], "nested_syllables")

    def test_spelling_is_exact_and_oov_lexeme_is_fold(self):
        payload, _ = self.compile()
        self.assertEqual(payload["entries"]["ABC"]["mode"], "exact")
        self.assertEqual(payload["entries"]["ACME"]["mode"], "fold")

    def test_hyphenated_letter_spelling_is_exact_without_rewriting_phones(self):
        phones = ["D", "IY1", "AY1", "D", "IY1"]
        self.source.write_text(json.dumps({"D-ID": phones, "NEW-WORD": ["N", "UW1", "W", "ER1", "D"]}))
        payload, audit = self.compile()
        self.assertEqual(payload["entries"]["D-ID"], {"arpa": phones, "mode": "exact"})
        self.assertEqual(payload["entries"]["NEW-WORD"]["mode"], "fold")
        self.assertEqual(audit["counts"]["source_separated_letter_spellings"], 1)

    def test_external_case_policy_can_select_mode_but_not_cmu_override(self):
        payload, _ = self.compile(case_policies={"ABC": "fold", "ACME": "exact", "CAR": "fold"})
        self.assertEqual(payload["entries"]["ABC"]["mode"], "fold")
        self.assertEqual(payload["entries"]["ACME"]["mode"], "exact")
        self.assertEqual(payload["entries"]["CAR"]["mode"], "cmu")
        for policies in ({"UNKNOWN": "fold"}, {"ABC": "all"}):
            with self.subTest(policies=policies), self.assertRaises(ValueError):
                self.compile(case_policies=policies)
        with self.assertRaisesRegex(ValueError, "object"):
            self.compile(case_policies=[("ABC", "fold")])

    def test_invalid_cmu_byte_pin_rejected(self):
        with self.assertRaisesRegex(ValueError, "CMU source"):
            self.compile(cmu_source_sha256="unknown")

    def test_semantic_pin_allows_whitespace_only_repackaging(self):
        canonical = builder.canonical_sha256(self.gold)
        before, _ = builder.compile_lexicon(self.source, self.cmu, expected_source_canonical_sha256=canonical)
        self.source.write_text(json.dumps(self.gold, indent=4))
        after, _ = builder.compile_lexicon(self.source, self.cmu, expected_source_canonical_sha256=canonical)
        self.assertEqual(before["entries"], after["entries"])
        self.assertNotEqual(before["source"]["sha256"], after["source"]["sha256"])
        self.assertEqual(before["source"]["canonical_sha256"], after["source"]["canonical_sha256"])

    def test_source_pin_required_and_mismatches_fail(self):
        with self.assertRaises(ValueError):
            builder.compile_lexicon(self.source, self.cmu)
        for changes in ({"expected_source_sha256": "0" * 64},
                        {"expected_source_canonical_sha256": "0" * 64},
                        {"expected_source_sha256": "invalid"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.compile(**changes)

    def test_cmu_hash_retains_syllable_structure_and_ignores_key_order(self):
        reverse = dict(reversed(list(self.cmu.items())))
        self.assertEqual(builder.canonical_sha256(self.cmu), builder.canonical_sha256(reverse))
        modified = {**self.cmu, "OTHER": [["AH1", "DH", "ER0"]]}
        self.assertNotEqual(builder.canonical_sha256(self.cmu), builder.canonical_sha256(modified))

    def test_invalid_phone_stress_and_model_inventory_rejected(self):
        for phones in (["AH"], ["AH3"], ["T1"], ["AX1"], ["BAD"], [["K"]], [], [1]):
            with self.subTest(phones=phones):
                self.source.write_text(json.dumps({"ABC": phones}))
                with self.assertRaises(ValueError):
                    self.compile()

    def test_duplicate_source_keys_and_non_token_keys_rejected(self):
        self.source.write_text('{"ABC":["EY1"],"ABC":["B"]}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.compile()
        for key in ("abc", "ABC DEF", "ABC123", "ABC.", "ÉTÉ"):
            self.source.write_text(json.dumps({key: ["EY1"]}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compile()

    def test_parser_ignores_headers_and_keeps_release_syllables(self):
        path = Path(self.tmp.name) / "cmudict.rep"
        path.write_text('## Date:  2026\nCAR  K AA1 R\nOTHER  AH1 - DH ER0\n')
        self.assertEqual(builder.parse_cmu_rep(path), self.cmu)

    def test_parser_reproduces_release_reader_for_duplicate_words(self):
        path = Path(self.tmp.name) / "cmudict.rep"
        path.write_text('BING  B IH0 NG\nBING  B IH1 NG\n')
        self.assertEqual(builder.parse_cmu_rep(path), {"BING": [["B", "IH1", "NG"]]})

    def test_output_deterministic_and_cli_refuses_overwrite(self):
        cmu = Path(self.tmp.name) / "cmudict.rep"
        cmu.write_text('CAR  K AA1 R\nOTHER  AH1 - DH ER0\n')
        output = Path(self.tmp.name) / "output"
        policy = Path(self.tmp.name) / "policies.json"
        policy.write_text("{}")
        arguments = ["--source", str(self.source), "--source-sha256",
                     hashlib.sha256(self.source.read_bytes()).hexdigest(), "--cmu", str(cmu),
                     "--output-dir", str(output), "--case-policies", str(policy)]
        builder.main(arguments)
        data = (output / "external_lexicon.json").read_bytes()
        self.assertEqual(data, builder.encode_payload(json.loads(data)))
        with self.assertRaises(FileExistsError):
            builder.main(arguments)
        self.assertEqual(data, (output / "external_lexicon.json").read_bytes())

class LookupTests(_RuntimeCase):
    """Whole-token/case rules, validated file shapes and cached lookups."""
    def setUp(self):
        super().setUp()
        self.path = self.directory / "external_lexicon.json"
        self.cmu = {"WEN": [["W", "EH1", "N"]]}
        self.data = {"format": "vits2-external-pronunciations", "version": 1,
            "source": {"sha256": "a" * 64, "entries": 4},
            "cmu": {"sha256": _canonical_sha(self.cmu), "entries": 1,
                    "canonical_sha256": _canonical_sha(self.cmu), "canonical_layout": "nested_syllables"},
            "entries": {
                "AIRBNB": {"arpa": ["EH1", "R", "B", "IY1", "EH1", "N", "B", "IY1"], "mode": "fold"},
                "ASML": {"arpa": ["EY1", "EH1", "S", "EH1", "M", "EH1", "L"], "mode": "exact"},
                "WEN": {"arpa": ["W", "EH1", "N"], "mode": "cmu",
                        "original_arpa": ["W", "EH1", "N", "D", "IY0", "Z"]},
                "OOV-BRAND": {"arpa": ["B", "R", "AE1", "N", "D"], "mode": "fold"},
            }}
        os.environ["VITS2_LEXICON_PATH"] = str(self.path)
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        self.module.reload_external_lexicon()

    def test_shared_lookup_case_policy_and_cmu_guard(self):
        self.assertEqual(self.module.lookup_external_lexicon("Airbnb"),
                         self.module.lookup_external_lexicon("airbnb"))
        self.assertIsNotNone(self.module.lookup_external_lexicon("ASML"))
        self.assertIsNone(self.module.lookup_external_lexicon("asml"))
        self.assertIsNone(self.module.lookup_external_lexicon("WEN", has_cmu=True))
        self.assertIsNone(self.module.lookup_external_lexicon("Wen"))
        self.assertIsNotNone(self.module.lookup_external_lexicon("oov-brand"))

    def test_full_word_boundaries_and_unknown_fallback(self):
        for word in ("Airbnb2", "myAirbnb", "Airbnb.", "Airbnb ASML", "foo_Airbnb",
                     "éAirbnb", "你好Airbnb", "UNKNOWN", "", None):
            with self.subTest(word=word):
                self.assertIsNone(self.module.lookup_external_lexicon(word))

    def test_status_and_results_cannot_mutate_cached_dictionary(self):
        phones = self.module.lookup_external_lexicon("ASML")
        phones[0][0] = "BAD"
        status = self.module.external_lexicon_status()
        status["mode_counts"]["exact"] = 99
        self.assertEqual(self.module.lookup_external_lexicon("ASML")[0][0], "EY1")
        self.assertEqual(self.module.external_lexicon_status()["mode_counts"]["exact"], 1)
        self.assertEqual(status["entry_count"], 4)

    def test_cmu_conflicts_are_bound_to_actual_nested_dictionary(self):
        self.assertTrue(self.module.validate_external_lexicon_cmu(self.cmu)["cmu_verified"])
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.module.validate_external_lexicon_cmu({"WEN": [["B", "AE1", "D"]]})
        self.data["entries"]["WEN"]["arpa"] = ["B", "AE1", "D"]
        self.save()
        with self.assertRaisesRegex(ValueError, "conflict repair"):
            self.module.validate_external_lexicon_cmu(self.cmu)

    def test_missing_or_unexpected_cmu_modes_rejected(self):
        for key, mode in (("WEN", "fold"), ("AIRBNB", "cmu")):
            with self.subTest(key=key):
                self.data["entries"][key]["mode"] = mode
                self.save()
                with self.assertRaisesRegex(ValueError, "CMU"):
                    self.module.validate_external_lexicon_cmu(self.cmu)
                self.data["entries"][key]["mode"] = "cmu" if key == "WEN" else "fold"

    def test_explicit_missing_resource_never_appears_loaded(self):
        os.environ["VITS2_LEXICON_PATH"] = str(self.path.with_name("missing.json"))
        with self.assertRaises(FileNotFoundError):
            self.module.external_lexicon_status()

    def test_optional_file_pin_and_invalid_sha(self):
        os.environ["VITS2_LEXICON_SHA256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertIsNotNone(self.module.lookup_external_lexicon("ASML"))
        os.environ["VITS2_LEXICON_SHA256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.module.external_lexicon_status()
        os.environ["VITS2_LEXICON_SHA256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            self.module.external_lexicon_status()

    def test_same_path_is_stable_until_explicit_reload(self):
        first = self.module.lookup_external_lexicon("ASML")
        self.data["entries"]["ASML"]["arpa"] = ["B", "AE1", "D"]
        self.path.write_text(json.dumps(self.data))
        self.assertEqual(self.module.lookup_external_lexicon("ASML"), first)
        self.module.reload_external_lexicon()
        self.assertEqual(self.module.lookup_external_lexicon("ASML"), [["B", "AE1", "D"]])

    def test_resource_loaded_once_for_many_token_queries(self):
        with patch("builtins.open", wraps=open) as opening:
            for _ in range(25):
                self.module.lookup_external_lexicon("ASML")
                self.module.lookup_external_lexicon("Airbnb")
                self.module.external_lexicon_status()
            self.assertEqual(opening.call_count, 1)

    def test_resource_is_not_limited_to_small_128_entry_manifest(self):
        def letters(number):
            return chr(65 + number // 26) + chr(65 + number % 26)
        self.data["entries"] = {"BRAND" + letters(i): {
            "arpa": ["B", "R", "AE1", "N", "D"], "mode": "fold"} for i in range(512)}
        self.data["source"]["entries"] = 512
        self.save()
        self.assertEqual(self.module.external_lexicon_status()["entry_count"], 512)
        self.assertIsNotNone(self.module.lookup_external_lexicon("brandta"))

    def test_rejects_invalid_shape_modes_and_phones(self):
        original = copy.deepcopy(self.data)
        changes = [lambda d: d.update(version=2),
                   lambda d: d.update(entries=[]),
                   lambda d: d["source"].update(entries=3),
                   lambda d: d["source"].update(sha256="bad"),
                   lambda d: d["entries"]["ASML"].update(mode="guess"),
                   lambda d: d["entries"]["ASML"].update(arpa=["AH"]),
                   lambda d: d["entries"]["ASML"].update(arpa=["BAD"]),
                   lambda d: d["entries"].update({"bad": {"mode": "fold", "arpa": ["B"]}})]
        for change in changes:
            with self.subTest(change=change):
                self.data = copy.deepcopy(original)
                change(self.data)
                self.save()
                with self.assertRaises(ValueError):
                    self.module.external_lexicon_status()

    def test_duplicate_keys_and_bounds_rejected(self):
        self.path.write_text('{"version":1,"version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.module.external_lexicon_status()
        self.path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
        self.module.reload_external_lexicon()
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            self.module.external_lexicon_status()

class CmuBindingTests(_RuntimeCase):
    """Bind the active CMU, reject incompatible resources and recover safely."""
    def setUp(self):
        super().setUp()
        self.resources = _load_module(self.package_name + ".lexicon_resources", PLUGIN / "lexicon_resources.py")
        self.cmu = {"CAR": [["K", "AA1", "R"]]}
        self.payload = self.make_payload(self.cmu)
        self.path = self.directory / "external.json"
        self.write(self.path, self.payload)
        self.select(self.path)

    def make_payload(self, cmu):
        return {"format": "vits2-external-pronunciations", "version": 1,
                "source": {"entries": 2, "sha256": "a" * 64},
                "cmu": {"entries": len(cmu), "sha256": "b" * 64,
                        "canonical_sha256": _canonical_sha(cmu), "canonical_layout": "nested_syllables"},
                "entries": {
                    "CAR": {"mode": "cmu", "arpa": [p for syllable in cmu["CAR"] for p in syllable]},
                    "ACME": {"mode": "cmu" if "ACME" in cmu else "fold",
                             "arpa": ([p for syllable in cmu["ACME"] for p in syllable]
                                      if "ACME" in cmu else ["AE1", "K", "M", "IY0"])}}}

    def write(self, path, payload):
        path.write_text(json.dumps(payload))

    def select(self, path):
        os.environ["VITS2_LEXICON_PATH"] = str(path)
        os.environ["VITS2_LEXICON_SHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def bind(self):
        return self.module.bind_external_lexicon_cmu(self.cmu)

    def incompatible(self):
        other = {**self.cmu, "ACME": [["B", "AE1", "D"]]}
        return self.make_payload(other)

    def test_schema_preparation_is_not_a_cmu_binding(self):
        self.assertFalse(self.module.validate_external_lexicon_file(self.path)["cmu_verified"])
        self.assertTrue(self.module.validate_external_lexicon_cmu(self.cmu)["cmu_verified"])
        # Pure validators remain suitable for audits and fixture cache resets.
        self.assertFalse(self.module.external_lexicon_status()["cmu_verified"])
        self.assertTrue(self.bind()["cmu_verified"])
        self.assertTrue(self.module.external_lexicon_status()["cmu_verified"])

    def test_documented_reload_rejects_other_cmu_and_following_lookup_raises(self):
        self.bind()
        self.assertEqual(self.module.lookup_external_lexicon("Acme"), [["AE1", "K", "M", "IY0"]])
        self.write(self.path, self.incompatible())
        self.select(self.path)
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.module.reload_external_lexicon()
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.module.lookup_external_lexicon("Acme")
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.module.external_lexicon_status()

    def test_explicit_new_path_must_validate_bound_cmu_without_manual_reload(self):
        self.bind()
        new_path = self.directory / "different-cmu.json"
        self.write(new_path, self.incompatible())
        self.select(new_path)
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.resources.ensure_external_lexicon(self.directory)
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.module.lookup_external_lexicon("Acme")

    def test_restore_compatible_file_and_reload_recovers(self):
        self.bind()
        self.write(self.path, self.incompatible())
        self.select(self.path)
        with self.assertRaises(ValueError):
            self.module.reload_external_lexicon()
        self.write(self.path, self.payload)
        self.select(self.path)
        self.module.reload_external_lexicon()
        self.assertEqual(self.module.lookup_external_lexicon("Acme"), [["AE1", "K", "M", "IY0"]])
        self.assertTrue(self.module.external_lexicon_status()["cmu_verified"])

    def test_failed_new_binding_keeps_previous_valid_identity(self):
        self.bind()
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.module.bind_external_lexicon_cmu({**self.cmu, "OTHER": [["AH1", "DH", "ER0"]]})
        self.assertEqual(self.module.lookup_external_lexicon("Acme"), [["AE1", "K", "M", "IY0"]])
        self.assertTrue(self.module.external_lexicon_status()["cmu_verified"])

    def test_same_cmu_does_not_permit_incorrect_conflict_phones(self):
        self.bind()
        changed = copy.deepcopy(self.payload)
        changed["entries"]["CAR"]["arpa"] = ["B", "AE1", "D"]
        self.write(self.path, changed)
        self.select(self.path)
        with self.assertRaisesRegex(ValueError, "conflict repair"):
            self.module.reload_external_lexicon()
        with self.assertRaisesRegex(ValueError, "conflict repair"):
            self.module.validate_external_lexicon_file(self.path)

    def test_valid_reload_keeps_matching_policy_and_can_update_oov_phones(self):
        self.bind()
        changed = copy.deepcopy(self.payload)
        changed["entries"]["ACME"]["arpa"] = ["AE1", "K", "M", "EY1"]
        self.write(self.path, changed)
        self.select(self.path)
        self.module.reload_external_lexicon()
        self.assertEqual(self.module.lookup_external_lexicon("Acme"), [["AE1", "K", "M", "EY1"]])
        self.assertIsNone(self.module.lookup_external_lexicon("CAR", has_cmu=True))

    def test_bound_lookups_do_not_hash_or_scan_cmu_for_each_word(self):
        self.bind()
        with patch.object(self.module.hashlib, "sha256", side_effect=AssertionError("per-token hash")), \
             patch.object(self.module, "_check_cmu", side_effect=AssertionError("per-token CMU scan")):
            for _ in range(100):
                self.assertIsNotNone(self.module.lookup_external_lexicon("Acme"))
                self.assertTrue(self.module.external_lexicon_status()["cmu_verified"])

    def test_missing_or_checksum_failure_after_reload_is_explicit(self):
        self.bind()
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.module.reload_external_lexicon()
        self.write(self.path, self.payload)
        os.environ["VITS2_LEXICON_SHA256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.module.reload_external_lexicon()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.module.lookup_external_lexicon("Acme")

    def test_resource_preparation_validates_output_before_release_path_change(self):
        self.bind()
        self.paths.configure_release_paths(self.directory / "not-installed-old-release")
        os.environ.pop("VITS2_LEXICON_PATH")
        os.environ.pop("VITS2_LEXICON_SHA256")
        root = self.directory / "new-release"
        (root / "frontend_data").mkdir(parents=True)
        (root / "frontend_data/cmudict.rep").write_text("CAR  K AA1 R\n")
        source = self.directory / "source.json"
        source_data = {"CAR": ["S", "IY1", "EY1", "AA1", "R"], "ACME": ["AE1", "K", "M", "IY0"]}
        source.write_text(json.dumps(source_data))
        policy = self.directory / "policy.json"
        policy.write_text("{}")
        with patch.object(self.resources, "_SOURCE_CANONICAL_SHA", _canonical_sha(source_data)), \
             patch.object(self.resources, "_POLICY_PATH", policy), \
             patch.object(self.resources, "_download", return_value=source):
            output = self.resources.ensure_external_lexicon(root)
        self.assertTrue(self.module.validate_external_lexicon_file(output)["cmu_verified"])
        self.paths.configure_release_paths(root)
        self.assertIsNotNone(self.module.lookup_external_lexicon("Acme"))

    def test_actual_english_startup_binds_before_pronunciation_lookup(self):
        tree = ast.parse((PLUGIN / "frontend/english.py").read_text())
        body = [node for node in tree.body if
                (isinstance(node, ast.ImportFrom) and node.module == "external_lexicon")
                or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == "_external_lexicon_status" for target in node.targets))
                or (isinstance(node, ast.FunctionDef) and node.name == "english_pronunciation")]
        namespace = {"__package__": self.package_name + ".frontend", "eng_dict": self.cmu,
                     "CUSTOM_EN_PRONUNCIATIONS": {}, "VERIFIED_EN_PRONUNCIATIONS": {},
                     "lookup_hotword": lambda *args, **kwargs: None}
        exec(compile(ast.Module(body=body, type_ignores=[]), "english-startup-probe", "exec"), namespace)
        self.assertTrue(namespace["_external_lexicon_status"]["cmu_verified"])
        self.write(self.path, self.incompatible())
        self.select(self.path)
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.module.reload_external_lexicon()
        # None here would silently invoke G2P in the caller: it must raise.
        with self.assertRaisesRegex(ValueError, "another CMU"):
            namespace["english_pronunciation"]("Acme")


def test_uncached_release_rep_rebuild_matches_published_dictionary(tmp_path):
    configured = os.environ.get("VITS2_MODEL_DIR", "")
    data = Path(configured) / "frontend_data"
    rep, cache = data / "cmudict.rep", data / "cmudict_cache.pickle"
    if not configured or not rep.is_file() or not cache.is_file():
        pytest.skip("Requires the verified VITS2 release via VITS2_MODEL_DIR")
    source = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/frontend/english.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = ast.Module(body=[node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"read_dict", "cache_dict", "get_dict"}],
        type_ignores=[])
    rebuilt_cache = tmp_path / "rebuilt-cmu.pickle"
    namespace = {"CMU_DICT_PATH": str(rep), "CACHE_PATH": str(rebuilt_cache),
                 "os": os, "pickle": pickle}
    exec(compile(selected, str(source), "exec"), namespace)
    expected = pickle.loads(cache.read_bytes())
    assert not rebuilt_cache.exists()
    rebuilt = namespace["get_dict"]()
    assert len(expected) == len(rebuilt) == 129464
    assert rebuilt == expected
    assert pickle.loads(rebuilt_cache.read_bytes()) == expected
    assert not any(key.startswith("##") for key in rebuilt)
    # The published representation has duplicate BING entries. Retain the
    # original dictionary's last-value semantics instead of rejecting a word.
    definitions = [line.strip().split("  ")[1]
                   for line in rep.read_text(encoding="utf-8").splitlines()
                   if line.startswith("BING  ")]
    assert len(definitions) > 1
    assert rebuilt["BING"] == [syllable.split() for syllable in definitions[-1].split(" - ")]
