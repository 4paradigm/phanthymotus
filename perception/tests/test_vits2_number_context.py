"""Contextual numeric-field tests, without ROS, FST or acoustic inference."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.frontend.number_context import find_spans


def spans(text, language="EN"):
    return find_spans(text, language, lambda number, lang: f"CARDINAL[{lang}:{number}]")


def replacements(text, language="EN"):
    return [(text[start:end], spoken) for start, end, spoken in spans(text, language)]


class TrailingDigitsContextTest(unittest.TestCase):
    def test_original_listening_case_uses_individual_digits(self):
        text = "Please confirm that the last four digits of your phone number are 5690."
        self.assertEqual(replacements(text), [("5690", "five six nine zero")])

    def test_numeric_four_case_and_leading_zeros(self):
        examples = (
            "The last 4 digits of your phone number are 0056.",
            "THE LAST FOUR DIGITS OF YOUR PHONE NUMBER ARE 0056.",
            "Last 4 digits of your phone number: 0056.",
            "Please check the final four digits of your telephone number: 0056.",
            "Your last 4 digits are 0056.",
            "Last four digits: 0056.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(replacements(text), [("0056", "zero zero five six")])

    def test_bounded_owner_and_explicit_identifier_fields(self):
        examples = (
            "The last four digits of Alice's phone number are 5690.",
            "The last four digits of Alice’s phone number are 5690.",
            "The last four digits of the customer's mobile number are 5690.",
            "The last four digits of the business bank account number are 5690.",
            "The last four digits on your credit card are 5690.",
            "The last four digits of your order ID are 5690.",
            "The last four digits of the serial number for Alice are 5690.",
            "The last four digits of the phone number for 北京 are 5690.",
            "The last four digits of 张三的 phone number are 5690.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(replacements(text), [("5690", "five six nine zero")])

    def test_multiple_fields_keep_original_offsets_and_other_numbers(self):
        text = "Last 4 digits of your phone number: 0056; 12 samples are ready. Last 4 digits of the account: 0917."
        found = spans(text)
        self.assertEqual(replacements(text), [
            ("0056", "zero zero five six"), ("0917", "zero nine one seven"),
        ])
        self.assertTrue(all(text[start:end] != "12" for start, end, _ in found))
        self.assertTrue(all(first[1] <= second[0] for first, second in zip(found, found[1:])))

    def test_ordinary_quantities_without_identifier_cue_are_untouched(self):
        for text in (
            "There are 5690 samples.",
            "The order contains 5690 samples.",
            "We measured 0056 grams.",
            "Your phone number has 12 digits.",
            "The package weighs 5690 grams.",
            "We have 4 digits and 5690 samples.",
        ):
            with self.subTest(text=text):
                self.assertEqual(replacements(text), [])

    def test_tail_mention_cannot_capture_later_quantity(self):
        for text in (
            "The last four digits of your phone number are hidden among 5690 samples.",
            "The last four digits of your phone number are based on 5690 samples.",
            "The last four digits of your phone number are not 5690.",
            "The last four digits of your phone number change after 5690 calls.",
            "The last four digits of your phone number are shown here. There are 5690 samples.",
            "The last four digits of your phone number are shown here, with 5690 samples.",
            "The last four digits of your phone number are shown here; there are 5690 samples.",
        ):
            with self.subTest(text=text):
                self.assertEqual(replacements(text), [])

    def test_unrecognized_ownership_field_does_not_enable_digit_reading(self):
        for text in (
            "The last four digits of the measured quantity are 5690.",
            "The last four digits of the sample count are 5690.",
            "The last four digits of these experiments are 5690.",
        ):
            with self.subTest(text=text):
                self.assertEqual(replacements(text), [])

    def test_field_value_must_be_exactly_four_digits_without_unit_or_decimal(self):
        for value in ("569", "56900", "5690.5", "5690,5", "5690A", "5690GB", "A5690"):
            text = "The last four digits of your phone number are " + value + "."
            with self.subTest(value=value):
                self.assertEqual(replacements(text), [])

    def test_chinese_last_digits_and_identifiers_are_preserved(self):
        for text in (
            "请确认手机号码的最后四位是5690。",
            "请确认最后4位为0056。",
            "我的电话号码是13800138000，末四位是5690。",
            "请确认末4位为0056。",
            "后 四 位：0056。",
            "订单编号为8492610。",
        ):
            with self.subTest(text=text):
                expected = ("0056", "零零五六") if "0056" in text else (
                    ("8492610", "八四九二六一零") if "8492610" in text else ("5690", "五六九零")
                )
                self.assertEqual(replacements(text, "ZH"), [expected])

    def test_chinese_trailing_digits_require_a_complete_four_digit_value(self):
        for label in ("末四位", "最后四位", "后4位"):
            for value in ("569", "56900", "5690.5", "5690,5", "5690A", "5690GB", "A5690"):
                with self.subTest(label=label, value=value):
                    self.assertEqual(replacements(label + "是" + value + "。", "ZH"), [])

    def test_last_digit_labels_do_not_capture_later_quantities(self):
        for text in (
            "末四位需要核对。共有5690个样本。",
            "末四位尚不明确，累计5690次调用。",
            "最后四位不是5690。",
            "最后四位是根据5690条数据计算的。",
        ):
            with self.subTest(text=text):
                self.assertEqual(replacements(text, "ZH"), [])

    def test_short_english_trailing_label_cannot_bypass_four_digit_boundary(self):
        for value in ("569", "56900", "5690.5", "5690GB"):
            with self.subTest(value=value):
                self.assertEqual(replacements("Last four digits are " + value + "."), [])

    def test_last_digits_context_uses_explicit_requested_number_language(self):
        text = "The last four digits of your phone number are 0056."
        self.assertEqual(replacements(text, "ZH"), [("0056", "零零五六")])


class EmbeddedDateTest(unittest.TestCase):
    def test_spaced_date_is_one_field_inside_either_language(self):
        for text in ("2026 10/23 message", "2026 10/ 23 星期五", "Date: 2026\t10/23."):
            with self.subTest(text=text):
                result = replacements(text)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0][1], "October twenty third, CARDINAL[EN:2026]")

    def test_date_match_does_not_cross_lines_or_truncate_identifiers(self):
        for text in ("2026\n10/23", "v2026 10/23", "2026 10/23.5", "2026 10/23_suffix"):
            with self.subTest(text=text):
                self.assertEqual(replacements(text), [])

    def test_identifier_cue_keeps_priority_over_date_shape(self):
        self.assertEqual(replacements("Order ID: 2026-10-23"), [
            ("2026-10-23", "two zero two six hyphen one zero hyphen two three")
        ])

    def test_invalid_calendar_date_is_not_claimed_as_valid(self):
        for text in ("message 2026 02/30", "message 2026 13/23"):
            with self.subTest(text=text):
                self.assertEqual(replacements(text), [])

    def test_numeric_only_date_with_decimal_suffix_still_rejects(self):
        from plugins.vits2_tts_language.neutral_text import InvalidDateError, normalize_neutral_text
        for text in ("2026 10/23.5", "2026-10-23.5"):
            with self.subTest(text=text), self.assertRaises(InvalidDateError):
                normalize_neutral_text(text, "zh")


class FixedMixedReadingTest(unittest.TestCase):
    def test_preprocessing_preserves_explicit_line_breaks(self):
        from plugins.vits2_tts_language.neutral_text import normalize_mixed_text
        self.assertEqual(normalize_mixed_text("AI\n2/3\r\nCAR", "zh"), "AI\n三分之二\r\nCAR")
        self.assertEqual(normalize_mixed_text("AI\n2/3\r\nCAR", "en"), "AI\ntwo thirds\r\nCAR")

    def test_ordinary_words_versions_and_unknown_identifiers_are_not_split(self):
        from plugins.vits2_tts_language.neutral_text import normalize_mixed_text
        text = "Qwen3 abc123 foo_123_bar 123-ABC 1,234XYZ"
        for language in ("zh", "en"):
            with self.subTest(language=language):
                self.assertEqual(normalize_mixed_text(text, language), text)

    def test_addresses_keep_literal_digits_and_separators(self):
        from plugins.vits2_tts_language.neutral_text import normalize_mixed_text
        text = "Visit https://example.com/2026/10/23?q=2/3 now"
        zh = normalize_mixed_text(text, "zh")
        en = normalize_mixed_text(text, "en")
        self.assertIn("斜杠二零二六斜杠一零斜杠二三", zh)
        self.assertIn("slash two zero two six slash one zero slash two three", en)
        self.assertNotIn("October", en)
        self.assertNotIn("thirds", en)
        self.assertTrue(zh.startswith("Visit ") and zh.endswith(" now"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
