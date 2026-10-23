"""Targeted normalization spans without ROS, GPU, or frontend dependencies."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/frontend/math_units.py"
SPEC = importlib.util.spec_from_file_location("vits2_math_units_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MathUnitsTest(unittest.TestCase):
    def setUp(self):
        self.numbers = Mock(side_effect=lambda value, language: f"NUM({value})")
        # These tests isolate span boundaries/operator composition. Actual
        # English cardinal rendering is checked separately below, without FST.
        reader = patch.object(MODULE, "_english_quantity",
                              side_effect=lambda value: self.numbers(value, "EN"))
        reader.start()
        self.addCleanup(reader.stop)

    def spans(self, text, language="ZH"):
        return MODULE.find_spans(text, language, self.numbers)

    def spoken(self, text, language="ZH"):
        return self.spans(text, language)[0][2]

    def test_explicit_math_and_equation(self):
        self.assertEqual(self.spoken("x^2+3=7"), "x 的平方 加 NUM(3) 等于 NUM(7)")
        self.assertEqual(self.spoken("x=2"), "x 等于 NUM(2)")
        self.assertEqual(self.spoken("πr²"), "派 乘以 r 的平方")
        self.assertEqual(self.spoken("x^2+3=7", "EN"), "x squared plus NUM(3) equals NUM(7)")

    def test_decimal_spelling_reaches_number_callback_unchanged(self):
        self.assertEqual(self.spoken("3.05kg"), "NUM(3.05) 千克")
        self.numbers.assert_called_once_with("3.05", "ZH")
        self.numbers.reset_mock()
        self.assertEqual(self.spoken("3.00 kg", "EN"), "NUM(3.00) kilograms")
        self.numbers.assert_called_once_with("3.00", "EN")

    def test_common_units_in_both_languages(self):
        units = {
            "℃": ("摄氏度", "degrees Celsius"),
            "°C": ("摄氏度", "degrees Celsius"),
            "°F": ("华氏度", "degrees Fahrenheit"),
            "kg": ("千克", "kilograms"), "g": ("克", "grams"),
            "mg": ("毫克", "milligrams"), "km/h": ("千米每小时", "kilometers per hour"),
            "m/s": ("米每秒", "meters per second"), "m²": ("平方米", "square meters"),
            "m/s²": ("米每二次方秒", "meters per second squared"),
            "cm²": ("平方厘米", "square centimeters"), "m³": ("立方米", "cubic meters"),
            "kg/m³": ("千克每立方米", "kilograms per cubic meter"),
            "GB": ("G B", "gigabytes"), "MB": ("M B", "megabytes"),
            "KB": ("K B", "kilobytes"), "degrees Celsius": ("摄氏度", "degrees Celsius"),
            "V": ("伏特", "volts"), "mV": ("毫伏", "millivolts"),
            "kV": ("千伏", "kilovolts"), "W": ("瓦特", "watts"),
            "mW": ("毫瓦", "milliwatts"), "kW": ("千瓦", "kilowatts"),
        }
        for unit, (zh, en) in units.items():
            with self.subTest(unit=unit):
                self.assertEqual(self.spoken(f"2.5 {unit}"), f"NUM(2.5) {zh}")
                self.assertEqual(self.spoken(f"2.5 {unit}", "EN"), f"NUM(2.5) {en}")
        self.assertEqual(self.spoken("2.5%"), "百分之 NUM(2.5)")
        self.assertEqual(self.spoken("2.5%", "EN"), "NUM(2.5) percent")
        self.assertEqual(self.spoken("1kg", "EN"), "NUM(1) kilogram")

    def test_omitted_zero_decimals_in_math_measurements_and_uncertainty(self):
        for text, expected in (
            ("x=.5", "x 等于 NUM(.5)"),
            ("x=-.5", "x 等于 负 NUM(.5)"),
            ("y=.5x-.25", "y 等于 NUM(.5) 乘以 x 减 NUM(.25)"),
            ("-.5V", "NUM(-.5) 伏特"),
            ("±.5°C", "正负 NUM(.5) 摄氏度"),
            ("-.5°C～.5°C", "NUM(-.5) 摄氏度 到 NUM(.5) 摄氏度"),
            (".50mW", "NUM(.50) 毫瓦"),
            (".5e-3V", "NUM(.5) 乘以 十的 NUM(-3) 次方 伏特"),
        ):
            with self.subTest(text=text):
                spans = self.spans(text)
                self.assertEqual(spans, [(0, len(text), expected)])

    def test_decimal_and_new_unit_boundaries_preserve_versions_and_ids(self):
        for text in (
            "v1.2.3", "version 1.2.3", "x=1.2.3", "NO.0917", "ID_0.5V",
            "v1.2.3V", "model1.5W", "3.05VModel", "2.5mWh", "1kVA",
            "2V/day", "2kW/h", "1V.2", "1W.2", "file.5V",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.spans(text), [])

    def test_parentheses_comparators_negative_values_and_powers(self):
        self.assertEqual(self.spoken("(x+2)^3=27"),
                         "左括号 x 加 NUM(2) 右括号 的立方 等于 NUM(27)")
        self.assertEqual(self.spoken("x>=-2"), "x 大于等于 负 NUM(2)")
        self.assertEqual(self.spoken("x^-2=0.25"), "x 的 负 NUM(2) 次方 等于 NUM(0.25)")
        self.assertEqual(self.spoken("x!=2", "EN"), "x not equal to NUM(2)")
        self.assertEqual(self.spoken("3-x=2"), "NUM(3) 减 x 等于 NUM(2)")

    def test_model_names_dates_phone_brand_and_ordinary_text_are_untouched(self):
        for text in ["A100", "H800", "Qwen3", "PhanRouter", "2026-10-13", "2026/10/13",
                     "1-800-FLOWERS.COM", "I/O", "The price is 123.",
                     "The API is ready.", "3.05kgModel", "100MBps"]:
            with self.subTest(text=text):
                self.assertEqual(self.spans(text), [])

    def test_unsupported_unit_suffix_is_not_partially_consumed(self):
        self.assertEqual(self.spans("2kg/day"), [])

    def test_spans_preserve_original_offsets_and_do_not_overlap(self):
        text = "公式x=2，重3.05kg，面积πr²。"
        spans = self.spans(text)
        self.assertEqual([text[start:end] for start, end, _ in spans], ["x=2", "3.05kg", "πr²"])
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))

    def test_excessively_long_expression_stays_with_existing_frontend(self):
        self.assertEqual(self.spans("+".join(["2"] * 1000)), [])


@unittest.skipUnless(importlib.util.find_spec("inflect"), "Real cardinal checks require the released inflect dependency")
class EnglishCardinalTest(unittest.TestCase):
    def spoken(self, text):
        def no_year_reader(value, language):
            self.fail("Explicit English quantity reached the general/year-like reader")
        return MODULE.find_spans(text, "EN", no_year_reader)[0][2]

    def test_grouped_measurements_use_cardinals_not_years(self):
        for text, expected in (
            ("1,024.5GB", "one thousand twenty-four point five gigabytes"),
            ("1,200.5GB", "one thousand two hundred point five gigabytes"),
            ("2026GB", "two thousand twenty-six gigabytes"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.spoken(text), expected)

    def test_signs_fractional_zeros_and_unit_plurality(self):
        for text, expected in (
            ("0.05GB", "zero point zero five gigabytes"),
            ("3.00kg", "three point zero zero kilograms"),
            ("-0.05℃", "minus zero point zero five degrees Celsius"),
            ("1kg", "one kilogram"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.spoken(text), expected)

    def test_math_uses_actual_english_number_readings(self):
        self.assertEqual(self.spoken("y=2x-3"), "y equals two times x minus three")

    def test_omitted_zero_and_electrical_units_use_real_english_quantities(self):
        for text, expected in (
            ("x=.5", "x equals zero point five"),
            ("x=-.5", "x equals negative zero point five"),
            ("-.5V", "minus zero point five volts"),
            ("±.5°C", "plus or minus zero point five degrees Celsius"),
            (".5e-3V", "zero point five times ten to the power of minus three volts"),
            ("1V", "one volt"), ("2.5mV", "two point five millivolts"),
            ("1kV", "one kilovolt"), ("2W", "two watts"),
            (".50mW", "zero point five zero milliwatts"),
            ("1kW", "one kilowatt"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.spoken(text), expected)

    def test_standalone_fractions_use_natural_singular_and_plural_nouns(self):
        for text, expected in (
            ("1/2", "one half"),
            ("2/2", "two halves"),
            ("1/3", "one third"),
            ("2/3", "two thirds"),
            ("1/4", "one quarter"),
            ("3/4", "three quarters"),
            ("5/6", "five sixths"),
            ("0/8", "zero eighths"),
            ("2/21", "two twenty-firsts"),
            ("1/100", "one hundredth"),
            ("2/100", "two hundredths"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.spoken(text), expected)

    def test_degenerate_fraction_denominators_keep_literal_over_reading(self):
        self.assertEqual(self.spoken("3/0"), "three over zero")
        self.assertEqual(self.spoken("7/1"), "seven over one")

    def test_explicit_arithmetic_slash_still_means_division(self):
        for text, expected in (
            ("Calculate 2/3.", "two divided by three"),
            ("x=2/3", "x equals two divided by three"),
            ("2/3+1", "two divided by three plus one"),
            ("2/3-1", "two divided by three minus one"),
            ("(2/3)", "open parenthesis two divided by three close parenthesis"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.spoken(text), expected)


if __name__ == "__main__":
    unittest.main()
