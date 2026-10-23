"""Address integrity and compatibility with the author's normal lexicon."""
from pathlib import Path
import importlib.util
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.frontend.special_text import normalize_special_text, verbalize_chinese_number
from plugins.vits2_tts_trt.frontend.lexicon_rules import apply_lexeme_rules


def number(value, language):
    return {"ZH": {"1": "一", "2": "二", "3": "三", "7": "七", "2.5": "二点五"},
            "EN": {"1": "one", "2": "two", "3": "three", "7": "seven", "2.5": "two point five"}}[language][value]


class SpecialTextTests(unittest.TestCase):
    def normalize(self, text):
        return normalize_special_text(text, apply_lexeme_rules, number)

    def test_waic_is_spelled_as_letters_without_changing_longer_names(self):
        for spelling in ("WAIC", "waic", "Waic"):
            self.assertEqual(apply_lexeme_rules(spelling), "W A I C")
            self.assertEqual(apply_lexeme_rules(spelling, language="EN"), "W A I C")
        for word in ("waicx", "mywaic", "WAIC2026", "WAIC_2026", "my_WAIC"):
            self.assertEqual(apply_lexeme_rules(word), word)

    def test_waic_inside_address_is_not_a_named_acronym(self):
        normalized = self.normalize("请访问https://waic.ai，或联系waic@example.com。")
        self.assertIn("W A I C", normalized)
        self.assertNotIn("世界人工智能大会", normalized)
        self.assertIn("艾特", normalized)

    def test_ordinary_text_uses_original_lexicon(self):
        for text in ("使用Qwen3、A100、H800和CNCF。", "Tier 1 IDC AutoML PhanRouter",
                     "在OpenMLDB里打开FeatInsight的SQL Playground。"):
            self.assertEqual(self.normalize(text), apply_lexeme_rules(text))

    def test_mail_local_part_is_not_a_model_name(self):
        text = self.normalize("请联系qwen3+tts_2@4paradigm.com。")
        # Chinese adjacency spaces are removed to avoid spurious G2P pauses;
        # the literal address content and English spelling must remain intact.
        self.assertIn("QWEN三加号TTS下划线二艾特四PARADIGM点COM", "".join(text.split()))
        self.assertIn("Q W E N", text)
        self.assertIn("P A R A D I G M", text)
        self.assertNotIn("千问", text)
        self.assertTrue(text.endswith("。"))

    def test_english_address_uses_english_symbols(self):
        text = self.normalize("Please email api_2+qa@example.com.")
        self.assertIn("A P I underscore two plus Q A at E X A M P L E dot C O M", text)
        self.assertFalse(any("\u4e00" <= c <= "\u9fff" for c in text))
        self.assertTrue(text.endswith("."))

    def test_url_path_query_and_fragment_are_preserved(self):
        text = self.normalize("请访问https://www.4paradigm.com/v2?a=3&b=2#top。")
        for phrase in ("H T T P S 冒号 斜杠 斜杠 W W W 点 四", "斜杠 V 二",
                       "问号 A 等号 三 与号 B 等号 二 井号 T O P"):
            self.assertIn("".join(phrase.split()), "".join(text.split()))
        self.assertIn("H T T P S", text)
        self.assertIn("W W W", text)

    def test_url_period_is_outside_address(self):
        text = self.normalize("Visit https://example.com/docs.")
        self.assertTrue(text.endswith("D O C S ."))

    def test_preserve_balanced_url_parentheses(self):
        text = self.normalize("Visit https://example.com/a(b).")
        self.assertIn("A left parenthesis B right parenthesis", text)

    def test_quantity_is_cardinal_and_decimal_digits_are_preserved(self):
        for raw, spoken in (("512", "五百一十二"), ("3.05", "三点零五"),
                            ("0.005", "零点零零五"), ("-2.5", "负二点五"),
                            ("10", "十"), ("101", "一百零一"),
                            ("10001", "一万零一"), ("10012", "一万零一十二"),
                            ("100000010", "一亿零一十"), ("0", "零")):
            with self.subTest(raw=raw):
                self.assertEqual(verbalize_chinese_number(raw), spoken)


@unittest.skipUnless(importlib.util.find_spec("inflect"), "Actual quantities require the released inflect dependency")
class EnglishQuantityTests(unittest.TestCase):
    def normalize(self, text, **kwargs):
        def year_style_reader(value, language):
            return {"2026": "twenty twenty-six", "1": "one", "7": "seven"}.get(value, value)
        return normalize_special_text(text, lambda value: value, year_style_reader, **kwargs)

    def test_counts_are_quantities_not_years(self):
        result = self.normalize("There are 5690 samples and 2026 robots.")
        self.assertIn("five thousand six hundred ninety", result)
        self.assertIn("two thousand twenty-six robots", result)
        self.assertNotIn("twenty twenty-six", result)

    def test_decimal_digits_and_grouping_are_preserved(self):
        result = self.normalize("The values are 1,024.05 and -0.005.")
        self.assertIn("one thousand twenty-four point zero five", result)
        self.assertIn("minus zero point zero zero five", result)

    def test_explicit_dates_and_identifiers_keep_priority(self):
        result = self.normalize("Order ID: 2026. Date: 2026-08-07.")
        self.assertIn("two zero two six", result)
        self.assertIn("August seventh, twenty twenty-six", result)

    def test_fraction_sentence_and_explicit_division_keep_distinct_readings(self):
        self.assertEqual(self.normalize("2/3 of the samples passed."),
                         "two thirds of the samples passed.")
        self.assertEqual(self.normalize("Calculate 2/3."),
                         "Calculate two divided by three .")
        self.assertEqual(self.normalize("y = 2/3."),
                         "y equals two divided by three .")
        self.assertEqual(
            normalize_special_text("合格样本占2/3。", lambda text: text,
                                   lambda value, language: verbalize_chinese_number(value),
                                   language="ZH"),
            "合格样本占三分之二。",
        )

    def test_date_and_address_slashes_are_not_fraction_nouns(self):
        self.assertEqual(self.normalize("Date: 2026/10/20."),
                         "Date: October twentieth, twenty twenty-six .")
        self.assertEqual(self.normalize("https://example.com/2/3"),
                         "H T T P S colon slash slash E X A M P L E dot C O M slash two slash three")

    def test_local_chinese_quantity_resolver_is_respected(self):
        result = normalize_special_text(
            "There are 7 samples.", lambda value: value,
            lambda value, language: "七" if language == "ZH" else "seven",
            language="EN", language_resolver=lambda start, end, default: "ZH",
        )
        self.assertIn("七 samples", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
