"""Entry-point lexeme routing without loading the acoustic model or FST assets.

Execute the production entry-point functions with an identity FST and the real
lexeme/context rules. The heavy G2P imports are irrelevant to this regression.
"""

import ast
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.frontend.lexicon_rules import apply_lexeme_rules
from plugins.vits2_tts_trt.frontend.special_text import normalize_special_text


class EnglishLexemeContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/frontend/chinese.py"
        selected = {"_remove_redundant_newlines_after_punctuation",
                    "_normalize_ordinary_text", "_normalize_raw_text", "mix_normalize"}
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = ast.Module(body=[node for node in tree.body
                                     if isinstance(node, ast.FunctionDef) and node.name in selected],
                               type_ignores=[])
        cls.entry = {
            "__package__": "plugins.vits2_tts_trt.frontend", "re": re,
            "_normalizer": types.SimpleNamespace(normalize=lambda text: text),
            "apply_lexeme_rules": apply_lexeme_rules,
            "normalize_special_text": normalize_special_text,
            "_special_number": lambda value, language: f"N[{language}:{value}]",
            "normalize_technical_lexemes": lambda text: text,
            "_post_replace": lambda text: text,
        }
        exec(compile(functions, str(source), "exec"), cls.entry)

    def setUp(self):
        english = types.ModuleType("plugins.vits2_tts_trt.frontend.english")
        english.replace_punctuation = lambda value: value
        # These tests isolate context routing with an identity FST. The real
        # Gold protection/lookup is exercised in test_vits2_gold_lexicon.py.
        english.normalize_preserving_english_pronunciations = (
            lambda value, normalize_plain: normalize_plain(value)
        )
        boundary = patch.dict(sys.modules, {english.__name__: english})
        boundary.start()
        self.addCleanup(boundary.stop)

    def normalize(self, text):
        return self.entry["mix_normalize"](text)

    def test_pure_english_names_are_not_translated_to_chinese(self):
        for text in ("We saw Qwen at WAIC.", "Qwen is ready.", "Qwen", "qwen"):
            with self.subTest(text=text):
                normalized = self.normalize(text)
                self.assertNotIn("千问", normalized)
                self.assertIn("qwen", normalized.lower())
        self.assertEqual(self.normalize("We saw Qwen at WAIC."), "We saw Qwen at W A I C.")

    def test_chinese_product_list_retains_chinese_alias(self):
        text = "我们在waic会议上看到许多产品和模型，qwen、豆包等都有出现"
        normalized = self.normalize(text)
        self.assertIn("W A I C", normalized)
        self.assertIn("千问", normalized)
        self.assertNotIn("qwen", normalized.lower())

    def test_default_callback_keeps_context_for_isolated_chinese_list_names(self):
        self.assertEqual(self.entry["_normalize_raw_text"]("Qwen"), "千问")
        self.assertEqual(self.entry["_normalize_raw_text"]("Qwen", lexeme_language="EN"), "Qwen")

    def test_english_clause_inside_mixed_input_still_keeps_english_name(self):
        normalized = self.normalize("我们看到许多产品，Qwen is ready，然后继续。")
        self.assertIn("Qwen is ready", normalized)
        self.assertNotIn("千问", normalized)

    def test_existing_english_numeric_routing_is_unchanged(self):
        for text in ("The order ID is 001234.", "The date is 2026-08-07 at 14:25.",
                     "The threshold is 8.5GB.", "Please check 3 cameras."):
            with self.subTest(text=text):
                self.assertEqual(self.normalize(text), self.entry["_normalize_raw_text"](text))


if __name__ == "__main__":
    unittest.main()
