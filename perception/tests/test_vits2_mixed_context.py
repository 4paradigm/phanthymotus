"""Local language/entity integrity without FST, neural G2P, ROS or GPU.

Number markers deliberately test routing, not number pronunciation. The
optional real-frontend runner and real-inflect tests validate those separately.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.frontend import math_units, special_text
from plugins.vits2_tts_trt.frontend.mixed_context import context_diagnostics, normalize_mixed_context
from plugins.vits2_tts_trt.frontend.number_context import find_spans as number_spans
from plugins.vits2_tts_trt.frontend.special_text import normalize_special_text


def number(value, language):
    return f"N[{language}:{value}]"


def normalize_for_routing(text):
    identity = lambda value: value
    original = lambda value: normalize_special_text(value, identity, number)
    with patch.object(math_units, "_english_quantity", side_effect=lambda value: number(value, "EN")) as cardinal, \
            patch.object(special_text, "_english_quantity", cardinal):
        return normalize_mixed_context(text, original, identity, number)


class MixedContextTest(unittest.TestCase):
    def test_pure_languages_keep_the_original_entry(self):
        for text in ("请检查网络和电量。", "The memory limit is 8.5GB."):
            original = Mock(return_value="original normalized result")
            self.assertEqual(normalize_mixed_context(text, original, Mock(), number), "original normalized result")
            original.assert_called_once_with(text)

    def test_english_insertion_keeps_chinese_grammar(self):
        text = "请说 hello world，然后检查网络。"
        normalized = normalize_for_routing(text)
        for phrase in ("请说", "hello world", "然后检查网络"):
            self.assertIn(phrase, normalized)

    def test_chinese_name_keeps_english_numbers(self):
        for text in ("NVIDIA provides 8.5GB for 北京 and 3 cameras.",
                     "Yesterday 北京 received 12 samples and 3 cameras.",
                     "北京 has 8.5GB of memory and 3 cameras."):
            with self.subTest(text=text):
                normalized = normalize_for_routing(text)
                self.assertIn("北京", normalized)
                self.assertIn("N[EN:", normalized)
                self.assertNotIn("N[ZH:", normalized)

    def test_brand_names_do_not_force_english_on_chinese_grammar(self):
        normalized = normalize_for_routing("GPU 内存阈值设为8.5GB，请确认。")
        self.assertIn("N[ZH:8.5]", normalized)
        self.assertNotIn("N[EN:8.5]", normalized)

    def test_curated_name_in_chinese_list_keeps_chinese_reading_context(self):
        for name in ("qwen", "Qwen", "QWEN", "Qwen3", "Qwen2.5"):
            text = f"我们在waic会议上看到许多产品和模型，{name}、豆包等都有出现"
            with self.subTest(name=name):
                diagnostic = context_diagnostics(text)
                piece = next(p for p in diagnostic["segments"] if p["text"] == name + "、")
                self.assertEqual(piece["language"], "ZH")
                self.assertEqual(piece["reason"], "chinese-name-in-list")
                self.assertEqual("".join(p["text"] for p in diagnostic["segments"]), text)

    def test_english_clauses_and_similar_names_are_not_chinese_list_aliases(self):
        for text, phrase in (
            ("请读下面这句英文，Qwen, please answer.", "Qwen,"),
            ("我们已经准备好，Qwen, please respond in English.", "Qwen,"),
            ("我们看到许多产品，Qwen is ready，然后继续。", "Qwen is ready"),
            ("At the meeting in 北京, Qwen is ready.", "Qwen is ready"),
            ("我们看到许多产品，myqwen、豆包都有出现。", "myqwen"),
            ("我们看到许多产品，qwenish、豆包都有出现。", "qwenish"),
        ):
            with self.subTest(text=text):
                piece = next(p for p in context_diagnostics(text)["segments"] if phrase in p["text"])
                self.assertEqual(piece["language"], "EN")
                self.assertNotEqual(piece["reason"], "chinese-name-in-list")

    def test_english_possessive_does_not_hide_later_chinese_list(self):
        for possessive in ("users'", "users’", "user's"):
            text = f"我们看了 {possessive} dashboards，Qwen、豆包等都有出现。"
            with self.subTest(possessive=possessive):
                piece = next(p for p in context_diagnostics(text)["segments"]
                             if p["text"] == "Qwen、")
                self.assertEqual(piece["language"], "ZH")
                self.assertEqual(piece["reason"], "chinese-name-in-list")

    def test_quoted_names_and_addresses_are_not_reclassified_as_list_aliases(self):
        for text in (
            '请念英文“qwen”，然后继续。',
            '请念英文（“qwen”），然后继续。',
            "请念英文'qwen'，然后继续。",
            "请念英文'qwen、doubao'，然后继续。",
            '请查看https://qwen.ai/qwen，然后继续。',
            '请联系qwen@example.com，然后继续。',
        ):
            with self.subTest(text=text):
                diagnostic = context_diagnostics(text)
                self.assertFalse(any(p["reason"] == "chinese-name-in-list"
                                     for p in diagnostic["segments"]))
                self.assertEqual("".join(p["text"] for p in diagnostic["segments"]), text)

    def test_comma_separated_clauses_keep_each_number_language(self):
        normalized = normalize_for_routing("The memory limit is 8.5GB, 请确认电流为1.2A。")
        self.assertIn("N[EN:8.5]", normalized)
        self.assertIn("N[ZH:1.2]", normalized)
        normalized = normalize_for_routing("请确认，the memory limit is 8.5GB.")
        self.assertIn("N[EN:8.5]", normalized)

    def test_quoted_language_does_not_change_surrounding_quantities(self):
        normalized = normalize_for_routing("Please say “电流1.2A” and check the 8.5GB memory limit.")
        self.assertIn("N[ZH:1.2]", normalized)
        self.assertIn("N[EN:8.5]", normalized)
        normalized = normalize_for_routing("他说“memory is ready”，电流为1.2A。")
        self.assertIn("memory is ready", normalized)
        self.assertIn("N[ZH:1.2]", normalized)

    def test_protected_entities_never_split_at_internal_punctuation(self):
        for entity in ("1,024.5GB", "2026-08-07", "14:25", "020-87654321",
                       "NO.0917", "https://example.com/v1.2?q=8.5", "test0917@example.com"):
            text = "Please check 北京: " + entity + ", then continue."
            with self.subTest(entity=entity):
                diagnostic = context_diagnostics(text)
                self.assertEqual("".join(piece["text"] for piece in diagnostic["segments"]), text)
                start = text.index(entity)
                end = start + len(entity)
                self.assertTrue(any(item["start"] <= start and item["end"] >= end
                                    for item in diagnostic["protected"]))
                self.assertFalse(any(start < piece["end"] < end for piece in diagnostic["segments"]))

    def test_apostrophes_are_not_quote_boundaries(self):
        text = "Alice's 北京 server isn't ready and doesn't have 8.5GB of memory."
        segments = context_diagnostics(text)["segments"]
        for word in ("Alice's", "isn't", "doesn't"):
            self.assertTrue(any(word in piece["text"] for piece in segments))
        self.assertEqual("".join(piece["text"] for piece in segments), text)

    def test_ambiguous_labels_remain_explicit_review_cases(self):
        diagnostic = context_diagnostics("北京 server 8.5GB")
        self.assertTrue(any(piece["reason"].startswith("ambiguous-") for piece in diagnostic["segments"]))
        # No assertion that either numeral language is universally correct.


class IdentifierContextTest(unittest.TestCase):
    def test_named_owner_keeps_id_digits_and_leading_zero(self):
        text = "The order ID for 北京 is 001234."
        spans = number_spans(text, "EN", number)
        self.assertTrue(any(spoken == "zero zero one two three four" for _, _, spoken in spans))
        self.assertTrue(any(text[start:end] == "001234" for start, end, _ in spans))

    def test_id_mention_does_not_capture_later_counts_or_versions(self):
        for text in ("The order ID for 北京 is based on 12 samples.",
                     "The order ID for 北京 has 12 characters.",
                     "Please validate the order ID for 北京 after 12 minutes.",
                     "The order ID for 北京 uses version 12 of the protocol."):
            with self.subTest(text=text):
                self.assertFalse(any("one two" in spoken for _, _, spoken in number_spans(text, "EN", number)))
                self.assertIn("N[EN:12]", normalize_for_routing(text))

    def test_phone_and_equation_are_independent_of_id_rules(self):
        phone = normalize_for_routing("The phone number for 北京 is 020-87654321.")
        self.assertIn("zero two zero", phone)
        self.assertIn("eight seven six five four three two one", phone)
        self.assertNotIn("minus", phone)
        formula = normalize_for_routing("For 北京, y=2x-3, and 12 samples are ready.")
        self.assertIn("y equals N[EN:2] times x minus N[EN:3]", formula)
        self.assertIn("N[EN:12]", formula)


if __name__ == "__main__":
    unittest.main(verbosity=2)
