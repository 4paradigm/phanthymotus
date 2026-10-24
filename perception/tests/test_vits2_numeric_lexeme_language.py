"""Keep fixed reading preferences through late product-name expansion."""
import unittest


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
            for text in ("foo_StyleTTS2_bar", "éVelo3Dé", "FACT II\nAcquisition"):
                with self.subTest(mode=mode, text=text):
                    self.assertEqual(normalize_mixed_text(text, mode), text)

    def test_named_versions_do_not_rewrite_inside_unicode_latin_words(self):
        from plugins.vits2_tts_language.name_text import normalize_standalone_name

        for mode in ("zh", "en"):
            for text in ("Qwen3é", "éQwen3", "Qwen3\u0301", "请读Qwen3é。"):
                with self.subTest(mode=mode, text=text):
                    self.assertEqual(normalize_standalone_name(text, mode), text)
