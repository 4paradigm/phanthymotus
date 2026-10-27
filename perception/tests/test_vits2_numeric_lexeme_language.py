"""Numeric names and unknown identifiers retain their explicit reading language."""
from test_vits2_liaoqianqian_lexicon import lexical_resource

import os
from pathlib import Path
import subprocess
import sys
import unittest

import pytest


PERCEPTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION))


class NumericLexemeLanguageTests(unittest.TestCase):
    def test_fixed_language_resolves_known_names_in_both_prose_languages(self):
        from plugins.vits2_tts_language.neutral_text import normalize_mixed_text

        for original, zh, en in (
            ("StyleTTS2", "style T T S 二", "style T T S two"),
            ("Velo3D", "Velo 三 D", "Velo three D"),
            ("X3 Acquisition", "X 三 Acquisition", "X three Acquisition"),
            ("X4 Pharmaceuticals", "X 四 Pharmaceuticals", "X four Pharmaceuticals"),
            ("FACT II Acquisition", "Fact 二 Acquisition", "Fact two Acquisition"),
        ):
            for mode, expected in (("zh", zh), ("en", en)):
                for template in ("{}", "我们使用{}合成语音。", "We use {} today."):
                    with self.subTest(original=original, mode=mode, template=template):
                        result = normalize_mixed_text(template.format(original), mode)
                        self.assertIn(expected, result)
                        self.assertNotIn("3D", result)

    def test_names_in_structured_addresses_keep_address_spelling(self):
        from plugins.vits2_tts_language.neutral_text import normalize_mixed_text

        for mode in ("zh", "en"):
            for text in ("https://example.com/StyleTTS2", "Velo3D@example.com"):
                with self.subTest(mode=mode, text=text):
                    result = normalize_mixed_text(text, mode)
                    # Address parsers own these spans. They spell letters and
                    # separators; product aliases must not rewrite the name.
                    self.assertNotIn("style T T S", result)
                    self.assertNotIn("Velo three D", result)
                    self.assertNotIn("Velo 三 D", result)

    def test_word_and_line_boundaries_are_preserved(self):
        from plugins.vits2_tts_language.neutral_text import normalize_mixed_text

        for mode in ("zh", "en"):
            for text in ("éVelo3Dé", "FACT II\nAcquisition"):
                with self.subTest(mode=mode, text=text):
                    self.assertEqual(normalize_mixed_text(text, mode), text)
            # An unknown whole identifier is spelled literally. The embedded
            # StyleTTS2 alias must not expand into "style T T S" inside it.
            expected = ("foo 下划线 StyleTTS 二 下划线 bar" if mode == "zh"
                        else "foo underscore StyleTTS two underscore bar")
            actual = normalize_mixed_text("foo_StyleTTS2_bar", mode)
            self.assertEqual(actual, expected)
            self.assertNotIn("style T T S", actual)

    def test_named_versions_do_not_rewrite_inside_unicode_latin_words(self):
        from plugins.vits2_tts_language.name_text import normalize_standalone_name

        for mode in ("zh", "en"):
            for text in ("Qwen3é", "éQwen3", "Qwen3\u0301", "请读Qwen3é。"):
                with self.subTest(mode=mode, text=text):
                    self.assertEqual(normalize_standalone_name(text, mode), text)


class AlphanumericTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plugins.vits2_tts_trt.frontend.alphanumeric_text import normalize_unknown_alphanumeric_tokens
        cls.read = staticmethod(normalize_unknown_alphanumeric_tokens)

    def test_unknown_ascii_tokens_keep_all_digits_in_english(self):
        for text, expected in [
            ("rtx4090a", "rtx four zero nine zero A"),
            ("RTX4090A", "RTX four zero nine zero A"),
            ("B2B", "B two B"),
            ("abc123", "abc one two three"),
            ("AX-4090Z", "AX hyphen four zero nine zero Z"),
            ("sku_001b", "sku underscore zero zero one B"),
            ("rev2.part7", "rev two dot part seven"),
            ("123abc", "one two three abc"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.read(text, "en"), expected)

    def test_unknown_tokens_use_chinese_digits_when_explicitly_selected(self):
        self.assertEqual(self.read("rtx4090a", "ZH"), "rtx 四零九零 A")
        self.assertEqual(self.read("型号abc123", "zh"), "型号abc 一二三")
        self.assertEqual(self.read("B2B", "zh"), "B 二 B")

    def test_ordinary_prose_and_quantities_are_not_reclassified(self):
        for text in ["4090", "-.25", "我们需要123个盒子。", "We need 1200 units.",
                     "The old car is parked outside.", "Hello, I'm Xiao Fang."]:
            for language in ["zh", "en"]:
                with self.subTest(text=text, language=language):
                    self.assertEqual(self.read(text, language), text)

    def test_existing_structured_readers_keep_ownership(self):
        for text in ["RTX4090", "rtx-4090", "ID: abc123", "serial number: AB-0042",
                     "2029-03-04", "12.5mA", "1e-4", "1.25GB/s", "y=2x-3",
                     "v2.03.004", "https://example.com/a1b?x=2",
                     "abc123+demo@example.com"]:
            for language in ["zh", "en"]:
                with self.subTest(text=text, language=language):
                    self.assertEqual(self.read(text, language), text)

    def test_ascii_pattern_cannot_split_accented_latin_or_combining_words(self):
        for text in ["préabc123", "abc123é", "abc123\u0301", "é12x"]:
            with self.subTest(text=text):
                self.assertEqual(self.read(text, "en"), text)

    def test_grouped_unknown_quantity_cannot_create_partial_identifier(self):
        for text in ["1,234XYZ", "12,345abc", "ABC1,234"]:
            for language in ("zh", "en"):
                with self.subTest(text=text, language=language):
                    self.assertEqual(self.read(text, language), text)

    def test_complete_curated_name_stays_with_existing_name_rule(self):
        for text in ("Qwen3", "qwen3", "Qwen2.5", "QWEN2.5"):
            for language in ("zh", "en"):
                with self.subTest(text=text, language=language):
                    self.assertEqual(self.read(text, language), text)

    def test_unknown_model_never_uses_the_curated_known_model_number(self):
        for text in ["rtx4090a", "RTX4090-A", "RTX4090.5"]:
            with self.subTest(text=text):
                result = self.read(text, "en")
                self.assertIn("four zero nine zero", result)
                self.assertNotIn("forty ninety", result)

    def test_explicit_language_is_required(self):
        for language in ["auto", "", None]:
            with self.subTest(language=language):
                with self.assertRaises(ValueError):
                    self.read("abc123", language)
        with self.assertRaises(TypeError):
            self.read(None, "en")

    def test_import_does_not_load_model_or_pronunciation_packages(self):
        env = dict(os.environ, PYTHONPATH=str(PERCEPTION), PYTHONDONTWRITEBYTECODE="1")
        code = """
import sys
from plugins.vits2_tts_trt.frontend.alphanumeric_text import normalize_unknown_alphanumeric_tokens
assert normalize_unknown_alphanumeric_tokens('abc123','en') == 'abc one two three'
assert 'plugins.vits2_tts_trt.frontend.english' not in sys.modules
assert 'g2p_en' not in sys.modules and 'torch' not in sys.modules and 'tensorrt' not in sys.modules
"""
        done = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


@pytest.fixture(scope="module")
def real_frontend():
    """Load release-dependent checks lazily; pure text cases stay host-runnable."""
    from test_vits2_frontend import production_text_adapter
    from test_vits2_curated_pronunciations import prepared
    return production_text_adapter.__wrapped__(), prepared


@pytest.mark.parametrize('text', [
    '我们使用 Qwen3 检查机器人。',
    'We use Qwen3 to check the robot.',
    '请读出 "Qwen3 is ready"，然后继续。',
    'Please read "你好，Qwen3" and continue.',
])
@pytest.mark.parametrize('mode,expected', [('zh', '千问三'), ('en', 'Qwen three')])
def test_fixed_name_preference_survives_chinese_english_prose_and_quotes(real_frontend, text, mode, expected):
    normalized, _, _, _ = real_frontend[1](real_frontend[0], text, mode)
    assert expected.lower() in normalized.lower()
    assert '千问 three' not in normalized
    assert 'Qwen 三' not in normalized
    if mode == 'en':
        assert '千问' not in normalized


@pytest.mark.parametrize('text', [
    '你好，Qwen3已经准备好。', 'Hello, Qwen3 is ready.',
    '2026 10/20', '你好，机器人。',
])
def test_none_preference_retains_existing_auto_frontend_entry(text, real_frontend):
    from plugins.vits2_tts_trt.frontend import chinese
    assert chinese.mix_normalize(text) == chinese.mix_normalize(text, reading_language=None)


@pytest.mark.parametrize('text', [
    'https://example.com/Qwen3', 'user+Qwen3@example.com', '订单 ID: Qwen3',
])
@pytest.mark.parametrize('mode', ['zh', 'en'])
def test_name_preference_does_not_override_structured_address_or_identifier(real_frontend, text, mode):
    normalized, _, _, _ = real_frontend[1](real_frontend[0], text, mode)
    assert 'Qwen three' not in normalized
    assert '千问' not in normalized


def test_invalid_reading_preference_is_rejected(real_frontend):
    from plugins.vits2_tts_trt.frontend import chinese
    with pytest.raises(ValueError, match='reading_language'):
        chinese.mix_normalize('Hello', reading_language='fr')
