"""Liaoqianqian CUSTOM → verified terms → CMU → G2P contracts.

Production lookup, phone conversion, both G2P routes and structured text
normalization run against small explicit CMU/OOV fixtures. Conflicting Gold
files are deliberately present in some cases to prove they are never consulted.
No acoustic inference, network access or full dictionary copy is needed.
"""

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
# All Gold entries conflict with the intended lookup, including a Gold-only OOV.
GOLD_FIXTURE = {word: ["B", "AE1", "D"] for word in (
    "CUDA", "HAMI", "IELTS", "APPS", "WEN", "GILD", "QWEN", "AGENTIC", "NVIDIA",
)}
CUSTOM_EXPECTED = {
    "CUDA": [["K", "UW1", "D", "AH0"]],
    "HAMI": [["HH", "AE1", "M", "IY0"]],
    "IELTS": [["AY1"], ["EH0", "L", "T", "S"]],
}
VERIFIED_EXPECTED = {
    "AGENTIC": [["AH0", "JH", "EH1", "N", "T", "IH0", "K"]],
    "NVIDIA": [["EH0", "N", "V", "IH1", "D", "IY0", "AH0"]],
}


def load_frontends(cache, gold_state="valid"):
    # The frontend now receives resource locations from the verified release
    # root. Keep test resources behind that same contract, not environment
    # overrides that production deliberately no longer accepts.
    release_root = cache.parent
    gold_path = release_root / "frontend" / "en_gold_pronunciations.json"
    gold_path.parent.mkdir()
    if gold_state == "valid":
        gold_path.write_text(json.dumps(GOLD_FIXTURE), encoding="utf-8")
    elif gold_state == "malformed":
        gold_path.write_text("{malformed", encoding="utf-8")
    elif gold_state != "missing":
        raise ValueError(gold_state)
    data_dir = release_root / "frontend_data"
    data_dir.mkdir()
    cache = data_dir / "cmudict_cache.pickle"
    name = "_liaoqianqian_lexicon_test_frontend"
    package = types.ModuleType(name)
    package.__path__ = [str(FRONTEND)]
    nltk = types.ModuleType("nltk")
    nltk.data = types.SimpleNamespace(path=[])
    g2p = types.ModuleType("g2p_en")
    # Distinct, valid phones expose an unwanted fallback rather than echoing
    # the input or the expected dictionary pronunciation.
    g2p.G2p = lambda: Mock(return_value=["Z", "AY1", "K", "S"])
    cache.write_bytes(pickle.dumps({
        "KNOWN": [["N", "OW1", "N"]],
        "CUDA": [["B", "AE1", "D"]],
        "HAMI": [["B", "AE1", "D"]],
        "IELTS": [["B", "AE1", "D"]],
        "AGENTIC": [["B", "AE1", "D"]],
        "NVIDIA": [["B", "AE1", "D"]],
        "WEN": [["W", "EH1", "N"]],
        "GILD": [["G", "IH1", "L", "D"]],
        "G": [["JH", "IY1"]],
        "B": [["B", "IY1"]],
        "APPS": [["AE1", "P", "S"]],
        "ATE": [["EY1", "T"]],
        "ATOM": [["AE1"], ["T", "AH0", "M"]],
        "ANNA": [["AE1"], ["N", "AH0"]],
        "CAR": [["K", "AA1", "R"]],
        "CASH": [["K", "AE1", "SH"]],
        "US": [["AH1", "S"]],
        "IT": [["IH1", "T"]],
        "ID": [["IH1", "D"]],
        "IP": [["IH1", "P"]],
        "U": [["Y", "UW1"]],
        "S": [["EH1", "S"]],
        "I": [["AY1"]],
        "T": [["T", "IY1"]],
        "D": [["D", "IY1"]],
        "P": [["P", "IY1"]],
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


class LiaoqianqianLexiconTest(unittest.TestCase):
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

    def test_custom_dictionary_matches_exactly_the_three_liaoqianqian_entries(self):
        self.assertEqual(self.english.CUSTOM_EN_PRONUNCIATIONS, CUSTOM_EXPECTED)
        self.assertNotIn("AGENTIC", self.english.CUSTOM_EN_PRONUNCIATIONS)
        self.assertEqual(self.english.VERIFIED_EN_PRONUNCIATIONS, VERIFIED_EXPECTED)
        self.assertTrue(self.english.CUSTOM_EN_PRONUNCIATIONS.keys().isdisjoint(
            self.english.VERIFIED_EN_PRONUNCIATIONS))

    def test_verified_terms_keep_whole_word_phones_and_stress_in_both_routes(self):
        for word, phones, tones in (
                ("AGENTIC", ("ah", "jh", "eh", "n", "t", "ih", "k"), (1, 3, 2, 3, 3, 1, 3)),
                ("NVIDIA", ("eh", "n", "V", "ih", "d", "iy", "ah"), (1, 3, 3, 2, 3, 1, 1))):
            self.assertEqual(self.english.eng_dict[word], [["B", "AE1", "D"]])
            for token in (word, word.lower(), word.title()):
                with self.subTest(token=token):
                    self.assertEqual(self.english.english_pronunciation(token), VERIFIED_EXPECTED[word])
                    self.assert_both_routes(token, phones, tones)
        self.english._g2p.assert_not_called()
        self.assertIs(self.mixed._g2p_en, self.english._g2p)

    def test_custom_wins_over_conflicting_cmu_and_gold_in_both_routes(self):
        examples = (
            ("CUDA", ("k", "uw", "d", "ah"), (3, 2, 3, 1)),
            ("HAMI", ("hh", "ae", "m", "iy"), (3, 2, 3, 1)),
            ("IELTS", ("ay", "eh", "l", "t", "s"), (2, 1, 3, 3, 3)),
        )
        for word, phones, tones in examples:
            self.assertEqual(self.english.eng_dict[word], [["B", "AE1", "D"]])
            for token in (word, word.lower(), word.title()):
                with self.subTest(token=token):
                    self.assertEqual(self.english.english_pronunciation(token), CUSTOM_EXPECTED[word])
                    self.assert_both_routes(token, phones, tones)
        self.english._g2p.assert_not_called()
        self.assertIs(self.mixed._g2p_en, self.english._g2p)

    def test_cmu_words_use_whole_word_readings_in_every_case(self):
        examples = (
            ("apps", ("ae", "p", "s"), (2, 3, 3)),
            ("ate", ("ey", "t"), (2, 3)),
            ("atom", ("ae", "t", "ah", "m"), (2, 3, 1, 3)),
            ("anna", ("ae", "n", "ah"), (2, 3, 1)),
            ("car", ("k", "aa", "r"), (3, 2, 3)),
            ("cash", ("k", "ae", "sh"), (3, 2, 3)),
            ("wen", ("w", "eh", "n"), (3, 2, 3)),
            ("gild", ("g", "ih", "l", "d"), (3, 2, 3, 3)),
        )
        for word, phones, tones in examples:
            for token in (word, word.title(), word.upper()):
                with self.subTest(token=token):
                    self.assertEqual(self.english.english_pronunciation(token),
                                     self.english.eng_dict[word.upper()])
                    self.assert_both_routes(token, phones, tones)
                    self.assertEqual(self.english.text_normalize(token), token)
        self.english._g2p.assert_not_called()
        self.assertIs(self.mixed._g2p_en, self.english._g2p)

    def test_first_mixed_unknown_word_reuses_loaded_english_predictor(self):
        expected = self.english.g2p_token("Qwen")
        calls_before = self.english._g2p.call_count
        phones, tones = self.mixed._en_token_to_phones("Qwen")
        self.assertEqual((tuple(phones), tuple(tones)), expected)
        self.assertEqual(self.english._g2p.call_count, calls_before + 1)
        self.assertIs(self.mixed._g2p_en, self.english._g2p)

    def test_other_oov_words_use_g2p_in_both_routes(self):
        for token in ("zyxqnoncetest", "Qwen"):
            with self.subTest(token=token):
                self.assertIsNone(self.english.english_pronunciation(token))
                self.assert_both_routes(token, ("z", "ay", "k", "s"), (3, 2, 3, 3))
                self.english._g2p.assert_called_with(token)
                self.mixed._g2p_en.assert_called_with(token)

    def test_gold_present_malformed_or_missing_cannot_change_lookup(self):
        for state in ("valid", "malformed", "missing"):
            with self.subTest(state=state):
                directory = self.directory / state
                directory.mkdir()
                english, mixed, *_ = load_frontends(directory / "cmu.pickle", state)
                for word, expected in (("CUDA", CUSTOM_EXPECTED["CUDA"]),
                                       ("HAMI", CUSTOM_EXPECTED["HAMI"]),
                                       ("Agentic", VERIFIED_EXPECTED["AGENTIC"]),
                                       ("NVIDIA", VERIFIED_EXPECTED["NVIDIA"]),
                                       ("APPS", [["AE1", "P", "S"]]),
                                       ("Wen", [["W", "EH1", "N"]]),
                                       ("GILD", [["G", "IH1", "L", "D"]])):
                    self.assertEqual(english.english_pronunciation(word), expected)
                    actual = mixed._en_token_to_phones(word)
                    self.assertEqual(english.g2p_token(word),
                                     (tuple(actual[0]), tuple(actual[1])))
                self.assertIsNone(english.english_pronunciation("Qwen"))
                self.assertEqual(english.g2p_token("Qwen"), (("z", "ay", "k", "s"), (3, 2, 3, 3)))
                self.assertEqual(english.text_normalize("APPS CUDA ZXQZX"), "APPS CUDA Z X Q Z X")

    def test_sentence_g2p_keeps_phone_stress_and_alignment_in_both_routes(self):
        phones = ("ae", "p", "s", "w", "eh", "n", "g", "ih", "l", "d", "hh", "ae", "m", "iy")
        tones = (2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 1)
        actual_phones, actual_tones, word2ph = self.english.g2p("Apps Wen GILD HAMi")
        self.assertEqual(tuple(actual_phones), ("_",) + phones + ("_",))
        self.assertEqual(tuple(actual_tones), (0,) + tones + (0,))
        self.assertEqual(word2ph, [1, 3, 3, 4, 4, 1])
        mixed_phones, mixed_tones, languages = self.mixed._non_zh_token_to_phones("Apps Wen GILD HAMi")
        self.assertEqual(tuple(mixed_phones), phones)
        self.assertEqual(tuple(mixed_tones), tones)
        self.assertEqual(languages, ["EN"] * len(phones))

    def test_acronyms_preserve_known_words_but_spell_unknown_uppercase(self):
        for token in ("CUDA", "HAMI", "IELTS", "APPS", "WEN", "GILD", "KNOWN", "AGENTIC", "NVIDIA"):
            with self.subTest(token=token):
                self.assertEqual(self.english.text_normalize(token), token)
        for token in ("ZXQZX", "QWEN"):
            with self.subTest(token=token):
                self.assertEqual(self.english.text_normalize(token), " ".join(token))

    def test_explicit_uppercase_acronyms_expand_before_cmu_and_fst(self):
        for token, phones in (
                ("US", ("y", "uw", "eh", "s")),
                ("IT", ("ay", "t", "iy")),
                ("ID", ("ay", "d", "iy")),
                ("IP", ("ay", "p", "iy"))):
            with self.subTest(token=token):
                normalized = self.english.text_normalize(token)
                self.assertEqual(normalized, " ".join(token))
                self.assertEqual(tuple(self.english.g2p(normalized)[0][1:-1]), phones)
                plain = Mock(side_effect=lambda value: value)
                self.assertEqual(self.english.normalize_preserving_english_pronunciations(token, plain),
                                 " ".join(token))
                plain.assert_not_called()
                for ordinary in (token.lower(), token.title()):
                    self.assertEqual(self.english.text_normalize(ordinary), ordinary)
                    self.assertEqual(self.english.english_pronunciation(ordinary),
                                     self.english.eng_dict[token])
                    self.assertEqual(self.english.normalize_preserving_english_pronunciations(
                        ordinary, lambda value: value), ordinary)

        identifiers = "my_US US2 IT_API ID-12 IP.address"
        plain = Mock(side_effect=lambda value: value)
        self.assertEqual(self.english.normalize_preserving_english_pronunciations(identifiers, plain), identifiers)
        plain.assert_called_once_with(identifiers)

    def test_only_custom_and_verified_whole_words_are_protected_from_plain_normalization(self):
        text = "CUDA apps HAMi Wen IELTS Qwen Agentic NVIDIA"
        fragments = []

        def destructive_fst(fragment):
            fragments.append(fragment)
            return re.sub(r"[A-Za-z]+", "BROKEN", fragment)

        result = self.english.normalize_preserving_english_pronunciations(text, destructive_fst)
        self.assertEqual(result, "CUDA BROKEN HAMi BROKEN IELTS BROKEN Agentic NVIDIA")
        self.assertEqual("".join(fragments), " apps  Wen  Qwen  ")

    def test_custom_word_protection_does_not_claim_identifiers_or_substrings(self):
        text = ("CUDAish myCUDA my_CUDA HAMI3foo HAMI2.5 HAMI-3-extra IELTS_X "
                "agentically Agentic3 my_NVIDIA Nvidia2.5 Nvidia-X")
        normalizer = Mock(side_effect=lambda fragment: fragment.replace("HAMI2.5", "version handled"))
        result = self.english.normalize_preserving_english_pronunciations(text, normalizer)
        normalizer.assert_called_once_with(text)
        self.assertEqual(result, text.replace("HAMI2.5", "version handled"))

    def test_explicit_letter_sequences_do_not_become_roman_numbers(self):
        for text in ("A I.", "A P I.", "今天A I.", "Read 'A I'.", "A\tI."):
            with self.subTest(text=text):
                normalizer = Mock(side_effect=lambda part: part.replace("I.", "一."))
                result = self.english.normalize_preserving_english_pronunciations(text, normalizer)
                self.assertNotIn("一", result)
                self.assertTrue(all("I." not in call.args[0] for call in normalizer.call_args_list))

        for text in ("I.", "II.", "IV.", "A I2", "my_A I", "A I.foo", "A I-X"):
            with self.subTest(text=text):
                normalizer = Mock(return_value="ordinary normalization")
                self.assertEqual(self.english.normalize_preserving_english_pronunciations(
                    text, normalizer), "ordinary normalization")
                normalizer.assert_called_once_with(text)

    def test_car_keeps_cmu_word_while_other_tokens_keep_their_normalizer(self):
        for text in ("CAR", "Car", "car"):
            with self.subTest(text=text):
                normalizer = Mock(return_value="CA R")
                self.assertEqual(self.english.normalize_preserving_english_pronunciations(
                    text, normalizer), text)
                normalizer.assert_not_called()
                self.assert_both_routes(text, ("k", "aa", "r"), (3, 2, 3))
        for text in ("SCAR", "CARGO", "CART", "CAR2", "CAR_model", "CAR-T", "my_CAR"):
            with self.subTest(text=text):
                normalizer = Mock(return_value="ordinary normalization")
                self.assertEqual(self.english.normalize_preserving_english_pronunciations(
                    text, normalizer), "ordinary normalization")
                normalizer.assert_called_once_with(text)

    def test_expanded_kilogram_words_are_not_partially_read_as_logarithms(self):
        for text in ("kilogram", "kilograms", "Kilograms", "KILOGRAMS"):
            with self.subTest(text=text):
                normalizer = Mock(return_value="ki对数rams")
                self.assertEqual(self.english.normalize_preserving_english_pronunciations(
                    text, normalizer), text)
                normalizer.assert_not_called()
        for text in ("kilogram_id", "kilograms2", "mykilogram", "log"):
            with self.subTest(text=text):
                normalizer = Mock(return_value="ordinary normalization")
                self.assertEqual(self.english.normalize_preserving_english_pronunciations(
                    text, normalizer), "ordinary normalization")
                normalizer.assert_called_once_with(text)

    def test_existing_names_units_punctuation_and_letter_a_keep_semantics(self):
        self.assert_both_routes("A", ("ey",), (2,))
        self.assertEqual(self.mixed._en_token_to_phones("GB"),
                         (["jh", "iy", "b", "iy"], [3, 2, 3, 2]))
        self.assertEqual(self.lexicon.apply_lexeme_rules("Qwen3", language="ZH"), "千问三")
        self.assertEqual(self.lexicon.apply_lexeme_rules("Qwen", language="EN"), "Qwen")
        self.assertEqual(self.english.g2p_token("."), ((".",), (0,)))

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

    def test_legacy_gold_numeric_entries_cannot_preempt_structured_tn(self):
        raw = "Order ID: 001234. We need 12 GB for 12 USD on 2026-08-07."
        baseline = self.normalized(raw)
        path = self.directory / "frontend" / "en_gold_pronunciations.json"
        path.write_text(json.dumps({word: ["B", "AE1", "D"]
                                   for word in ("12", "001234", "G", "B", "USD")}), encoding="utf-8")
        self.assertEqual(self.normalized(raw), baseline)
        self.assertIn("zero zero one two three four", baseline)
        self.assertIn("twelve gigabytes", baseline)
        self.assertIn("August seventh", baseline)

    def test_curated_names_keep_explicit_rewrites_in_english_and_mixed_context(self):
        for token, spoken in (("OpenMLDB", "Open M L D B"),
                              ("FeatInsight", "Feat Insight")):
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



if __name__ == "__main__":
    unittest.main()
