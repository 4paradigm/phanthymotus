"""Gold lexicon integration, without acoustic inference or resource downloads.

The full shipped overlay, production lookup/conversion functions, both G2P
routes, inflect, and structured TN rules are real. Small CMU and OOV fixtures
make lookup precedence observable without importing NLTK models. A deliberately
destructive normalizer tests the external FST boundary; it is not an acoustic
or real-FST quality test.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy  # Load the extension before temporary sys.modules isolation.


FRONTEND = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/frontend"
GOLD_SHA256 = "0c6fd2adb51aeae0fd83968f1902dedb16c73db695a3e3ad088bfc23cc05d57a"


def load_frontends(cache):
    # The frontend now receives resource locations from the verified release
    # root. Keep test resources behind that same contract, not environment
    # overrides that production deliberately no longer accepts.
    release_root = cache.parent
    data_dir = release_root / "frontend_data"
    data_dir.mkdir()
    cache = data_dir / "cmudict_cache.pickle"
    name = "_gold_lexicon_test_frontend"
    package = types.ModuleType(name)
    package.__path__ = [str(FRONTEND)]
    nltk = types.ModuleType("nltk")
    nltk.data = types.SimpleNamespace(path=[])
    g2p = types.ModuleType("g2p_en")
    # Distinct, valid phones expose an unwanted fallback rather than echoing
    # the input or the expected Gold pronunciation.
    g2p.G2p = lambda: Mock(return_value=["Z", "AY1", "K", "S"])
    cache.write_bytes(pickle.dumps({
        "KNOWN": [["N", "OW1", "N"]],
        "SQL": [["B", "AE1", "D"]],
        "NVIDIA": [["B", "AE1", "D"]],
        "G": [["JH", "IY1"]],
        "B": [["B", "IY1"]],
    }))

    def load(module_name):
        spec = importlib.util.spec_from_file_location(
            f"{name}.{module_name}", FRONTEND / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    with patch.dict(sys.modules, {name: package, "nltk": nltk, "g2p_en": g2p}):
        load("release_paths").configure_release_paths(release_root)
        english = load("english")
        mixed = load("unified_g2p")
        special = load("special_text")
        math_units = load("math_units")
        lexicon = load("lexicon_rules")
    return english, mixed, special, math_units, lexicon


class GoldLexiconTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        (self.english, self.mixed, self.special,
         self.math_units, self.lexicon) = load_frontends(self.directory / "cmu.pickle")

    def assert_both_routes(self, token, phones, tones):
        expected = tuple(phones), tuple(tones)
        self.assertEqual(self.english.g2p_token(token), expected)
        actual = self.mixed._en_token_to_phones(token)
        self.assertEqual((tuple(actual[0]), tuple(actual[1])), expected)

    def normalized(self, text, preserve=True):
        english = self.english

        def plain(fragment):
            return (english.normalize_preserving_english_pronunciations(
                fragment, english.text_normalize_without_numbers,
            ) if preserve else english.text_normalize_without_numbers(fragment))

        return self.special.normalize_special_text(
            text, plain, lambda value, language: self.math_units._english_quantity(value),
            language="EN",
        )

    def test_full_source_overlay_is_exact_valid_and_below_one_megabyte(self):
        source = FRONTEND / "en_gold_pronunciations.json"
        self.assertLess(source.stat().st_size, 1_000_000)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), GOLD_SHA256)
        entries = self.english.gold_en_pronunciations()
        self.assertEqual(len(entries), 13083)
        self.assertIs(self.english.gold_en_pronunciations(), entries)
        self.assertFalse(any(word.isdecimal() or len(word) == 1 for word in entries))
        for word, pronunciation in entries.items():
            phones, tones = self.english.refine_syllables((pronunciation,))
            converted = [self.english.post_replace_ph(phone) for phone in phones]
            self.assertNotIn("UNK", converted, word)
            self.assertEqual(len(phones), len(tones), word)
            self.assertTrue(set(tones) <= {1, 2, 3}, word)

    def test_known_gold_phones_and_stress_are_identical_in_both_routes(self):
        examples = (
            ("SQL", ("eh", "s", "k", "y", "uw", "eh", "l"), (3, 3, 3, 3, 3, 2, 3)),
            ("NVIDIA", ("eh", "n", "V", "ih", "d", "iy", "ah"), (2, 3, 3, 2, 3, 1, 1)),
            ("QWEN", ("k", "w", "iy", "n"), (3, 3, 2, 3)),
            ("AIRBNB", ("eh", "r", "b", "iy", "eh", "n", "b", "iy"),
             (2, 3, 3, 2, 2, 3, 3, 2)),
            ("TAKE-TWO", ("t", "ey", "k", "t", "uw"), (3, 2, 3, 3, 2)),
            ("RISC-V", ("r", "ih", "s", "k", "f", "ay", "V"),
             (3, 2, 3, 3, 3, 2, 3)),
        )
        for token, phones, tones in examples:
            for spelling in (token, token.lower(), token.title()):
                with self.subTest(spelling=spelling):
                    self.assert_both_routes(spelling, phones, tones)
        self.english._g2p.assert_not_called()
        self.assertIsNone(self.mixed._g2p_en)

    def test_multiword_sentence_g2p_preserves_gold_phones_stress_and_alignment(self):
        expected_phones = (
            "eh", "n", "V", "ih", "d", "iy", "ah",
            "eh", "s", "k", "y", "uw", "eh", "l",
        )
        expected_tones = (2, 3, 3, 2, 3, 1, 1, 3, 3, 3, 3, 3, 2, 3)
        phones, tones, word2ph = self.english.g2p("Nvidia sql")
        self.assertEqual(tuple(phones), ("_",) + expected_phones + ("_",))
        self.assertEqual(tuple(tones), (0,) + expected_tones + (0,))
        self.assertEqual(word2ph, [1, 7, 7, 1])
        mixed_phones, mixed_tones, languages = self.mixed._non_zh_token_to_phones("Nvidia sql")
        self.assertEqual(tuple(mixed_phones), expected_phones)
        self.assertEqual(tuple(mixed_tones), expected_tones)
        self.assertEqual(languages, ["EN"] * len(expected_phones))

    def test_explicit_custom_reading_wins_over_gold_and_conflicting_cmu(self):
        self.english.CUSTOM_EN_PRONUNCIATIONS["NVIDIA"] = [["K", "UW1", "D", "AH0"]]
        self.assert_both_routes("Nvidia", ("k", "uw", "d", "ah"), (3, 2, 3, 1))
        self.english._g2p.assert_not_called()
        self.assertIsNone(self.mixed._g2p_en)

    def test_gold_wins_over_conflicting_cmu_pronunciation(self):
        self.assertEqual(self.english.eng_dict["SQL"], [["B", "AE1", "D"]])
        self.assert_both_routes("sql", ("eh", "s", "k", "y", "uw", "eh", "l"),
                               (3, 3, 3, 3, 3, 2, 3))
        self.english._g2p.assert_not_called()

    def test_unlisted_words_keep_cmu_then_oov_fallback(self):
        self.assertNotIn("KNOWN", self.english.gold_en_pronunciations())
        self.assert_both_routes("known", ("n", "ow", "n"), (3, 2, 3))
        self.english._g2p.assert_not_called()
        self.assertIsNone(self.mixed._g2p_en)
        self.assert_both_routes("zyxqnoncetest", ("z", "ay", "k", "s"), (3, 2, 3, 3))
        self.english._g2p.assert_called_once_with("zyxqnoncetest")
        self.mixed._g2p_en.assert_called_once_with("zyxqnoncetest")

    def test_existing_custom_names_aliases_units_and_letter_a_keep_semantics(self):
        self.assert_both_routes("CUDA", ("k", "uw", "d", "ah"), (3, 2, 3, 1))
        self.assert_both_routes("HAMi", ("hh", "aa", "m", "iy"), (3, 2, 3, 1))
        self.assert_both_routes("A", ("ey",), (2,))
        self.assert_both_routes("GB", ("jh", "iy", "b", "iy"), (3, 2, 3, 2))
        self.assertEqual(self.lexicon.apply_lexeme_rules("Qwen3", language="ZH"), "千问三")
        self.assertEqual(self.lexicon.apply_lexeme_rules("Qwen", language="EN"), "Qwen")
        self.assertEqual(self.english.g2p_token("."), ((".",), (0,)))

    def test_english_text_normalization_keeps_gold_acronyms_intact(self):
        for token in ("NVIDIA", "nvidia", "Nvidia", "SQL", "sql", "airbnb",
                      "TAKE-TWO", "risc-v"):
            with self.subTest(token=token):
                normalized = self.english.text_normalize("Use " + token + ".")
                self.assertIn(token, normalized)
                self.assertNotIn("N V I D I A", normalized)
                self.assertNotIn("S Q L", normalized)
        self.assertNotIn("ZXQZX", self.english.gold_en_pronunciations())
        self.assertEqual(self.english.text_normalize("ZXQZX"), "Z X Q Z X")

    def test_brand_case_survives_destructive_fst_boundary_before_gold_g2p(self):
        for token in ("NVIDIA", "nvidia", "Nvidia", "SQL", "sql", "airbnb"):
            fragments = []

            def destructive_fst(fragment):
                fragments.append(fragment)
                return re.sub(re.escape(token), "错误中文", fragment, flags=re.I).strip()

            with self.subTest(token=token):
                original = "Use " + token + " today."
                result = self.english.normalize_preserving_english_pronunciations(
                    original, destructive_fst,
                )
                self.assertEqual(result, original)
                self.assertEqual(fragments, ["Use ", " today."])
                self.assertNotIn("错误中文", result)
                self.assertNotIn("UNK", self.english.g2p_token(token)[0])

    def test_known_word_protection_does_not_claim_unknown_models_or_substrings(self):
        text = "NVIDIAish myNVIDIA my_NVIDIA Qwen3foo Qwen2.5 Qwen-3-extra Nvidia_X"
        fragments = []

        def normalizer(fragment):
            fragments.append(fragment)
            return fragment.replace("Qwen2.5", "existing version handling")

        result = self.english.normalize_preserving_english_pronunciations(text, normalizer)
        self.assertEqual(fragments, [text])
        self.assertEqual(result, text.replace("Qwen2.5", "existing version handling"))

    def test_quantities_decimals_and_calendar_context_keep_existing_readings(self):
        examples = (
            ("There are 5690 samples and 2026 robots.",
             ("five thousand six hundred ninety", "two thousand twenty-six robots")),
            ("The values are 1,024.05 and -0.005.",
             ("one thousand twenty-four point zero five", "minus zero point zero zero five")),
            ("The date is 2026-08-07 at 14:25.",
             ("August seventh", "two thousand twenty-six", "fourteen hours twenty-five minutes")),
        )
        for raw, fragments in examples:
            with self.subTest(raw=raw):
                normalized = self.normalized(raw)
                self.assertEqual(normalized, self.normalized(raw, preserve=False))
                for fragment in fragments:
                    self.assertIn(fragment, normalized)
                self.assertFalse(any(char.isdecimal() for char in normalized))

    def test_identifier_phone_and_unit_context_keep_existing_readings(self):
        examples = (
            ("The order ID is 001234.", ("zero zero one two three four",)),
            ("The last four digits of your phone number are 5690.", ("five six nine zero",)),
            ("The phone is 020-87654321.", ("zero two zero", "eight seven six five four three two one")),
            ("The threshold is 8.5GB and the temperature is -12°C.",
             ("eight point five gigabytes", "minus twelve degrees Celsius")),
        )
        for raw, fragments in examples:
            with self.subTest(raw=raw):
                normalized = self.normalized(raw)
                self.assertEqual(normalized, self.normalized(raw, preserve=False))
                for fragment in fragments:
                    self.assertIn(fragment, normalized)
                self.assertFalse(any(char.isdecimal() for char in normalized))

    def test_addresses_are_literal_before_brand_lookup(self):
        examples = (
            ("Visit https://nvidia.com/sql?q=2.",
             ("N V I D I A dot C O M slash S Q L", "question mark Q equals two")),
            ("Email nvidia+sql_2@example.com.",
             ("N V I D I A plus S Q L underscore two at E X A M P L E dot C O M",)),
        )
        for raw, fragments in examples:
            with self.subTest(raw=raw):
                normalized = self.normalized(raw)
                self.assertEqual(normalized, self.normalized(raw, preserve=False))
                for fragment in fragments:
                    self.assertIn(fragment, normalized)
                self.assertNotIn("NVIDIA", normalized)
                self.assertNotIn("SQL", normalized)

    def test_numeric_and_unit_overlay_entries_cannot_preempt_structured_tn(self):
        extra = {**self.english.gold_en_pronunciations(),
                 **{word: ("B", "AE1", "D") for word in ("12", "001234", "G", "B", "USD")}}
        raw = "Order ID: 001234. We need 12 GB for 12 USD on 2026-08-07."
        baseline = self.normalized(raw)
        with patch.object(self.english, "gold_en_pronunciations", return_value=extra):
            normalized = self.normalized(raw)
        self.assertEqual(normalized, baseline)
        self.assertIn("zero zero one two three four", normalized)
        self.assertIn("twelve gigabytes", normalized)
        self.assertIn("August seventh", normalized)

    def test_restored_curated_names_precede_gold_in_english_and_mixed_context(self):
        for token, spoken in (("OpenMLDB", "Open M L D B"),
                              ("FeatInsight", "Feat Insight")):
            # These two restored curated corrections are not Gold entries.
            self.assertNotIn(token.upper(), self.english.gold_en_pronunciations())
            for spelling in (token, token.lower(), token.upper()):
                for language, prefix, suffix in (("EN", "Use ", "."),
                                                  ("MIX", "请使用", "。")):
                    with self.subTest(token=spelling, language=language):
                        raw = prefix + spelling + suffix
                        curated = self.lexicon.apply_lexeme_rules(raw, language=language)
                        self.assertEqual(curated, prefix + spoken + suffix)
                        protected = self.english.normalize_preserving_english_pronunciations(
                            curated, lambda ordinary: ordinary,
                        )
                        self.assertIn(spoken, protected)

    def test_curated_product_names_do_not_rewrite_identifiers_or_addresses(self):
        identifier = "foo_OpenMLDB_v2 featinsight_api"
        prior_rules = [
            rule for rule in self.lexicon.RULES
            if rule[1] not in ("Open M L D B", "Feat Insight")
        ]
        for language in ("EN", "MIX"):
            actual = self.lexicon.apply_lexeme_rules(identifier, language=language)
            with patch.object(self.lexicon, "RULES", prior_rules):
                prior = self.lexicon.apply_lexeme_rules(identifier, language=language)
            self.assertEqual(actual, prior)
            # The pre-existing API rule spells the suffix after an underscore.
            # Restoring these two product names must not alter that baseline.
            self.assertEqual(actual, "foo_OpenMLDB_v2 featinsight_A P I")
            self.assertNotIn("Open M L D B", actual)
            self.assertNotIn("Feat Insight", actual)

        def normalize(raw, curated):
            def plain(ordinary):
                if curated:
                    ordinary = self.lexicon.apply_lexeme_rules(ordinary, language="EN")
                return self.english.normalize_preserving_english_pronunciations(
                    ordinary, self.english.text_normalize_without_numbers,
                )
            return self.special.normalize_special_text(
                raw, plain, lambda value, language: self.math_units._english_quantity(value),
                language="EN",
            )

        for raw, spoken in (
                ("Visit https://openmldb.com/FeatInsight?x=2.",
                 "O P E N M L D B dot C O M slash F E A T I N S I G H T"),
                ("Email openmldb+FeatInsight@example.com.",
                 "O P E N M L D B plus F E A T I N S I G H T at")):
            with self.subTest(raw=raw):
                actual = normalize(raw, curated=True)
                self.assertEqual(actual, normalize(raw, curated=False))
                self.assertIn(spoken, actual)
                self.assertNotIn("Open M L D B", actual)
                self.assertNotIn("Feat Insight", actual)

    def test_missing_lexicon_raises_instead_of_silently_using_cmu(self):
        self.english.current_file_path = str(self.directory)
        self.english.gold_en_pronunciations.cache_clear()
        with self.assertRaises(FileNotFoundError):
            self.english.g2p_token("known")
        self.english._g2p.assert_not_called()

    def test_malformed_json_and_schema_fail_explicitly(self):
        path = self.directory / "en_gold_pronunciations.json"
        self.english.current_file_path = str(self.directory)
        path.write_text("{malformed", encoding="utf-8")
        self.english.gold_en_pronunciations.cache_clear()
        with self.assertRaises(json.JSONDecodeError):
            self.english.gold_en_pronunciations()
        for invalid in ([], {"word": ["W", "ER1", "D"]}, {"": ["B"]},
                        {"WORD": []}, {"WORD": "W ER1 D"}, {"WORD": ["NOTAPHONE"]},
                        {"WORD": [None]}):
            with self.subTest(invalid=invalid):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                self.english.gold_en_pronunciations.cache_clear()
                with self.assertRaisesRegex(ValueError, "[Ee]nglish gold"):
                    self.english.gold_en_pronunciations()


if __name__ == "__main__":
    unittest.main()
