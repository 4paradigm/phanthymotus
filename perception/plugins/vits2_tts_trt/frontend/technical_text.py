"""Stable lexical handling for technology names after Chinese TN."""

import re

from .lexicon_rules import _sub_lexeme


def _edge(pattern: str) -> re.Pattern:
    """CJK-safe ASCII word boundary (``\\b`` fails next to 中文)."""
    return re.compile(rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])",
                      re.IGNORECASE)


_PRODUCT_FORMS = (
    (re.compile(r"\bchatgpt\b", re.IGNORECASE), "chat GPT"),
    (re.compile(r"\bopenai\b", re.IGNORECASE), "open A I"),
    (re.compile(r"\bchattts\b", re.IGNORECASE), "chat T T S"),
    (_edge(r"rise[^\S\r\n\v\f\x85\u2028\u2029]+vgpu"), "rise V G P U"),
    (_edge(r"phanthymotus"), "Phanthy Motus"),
    (_edge(r"https"), "H T T P S"),
    (_edge(r"http"), "H T T P"),
)


def normalize_technical_lexemes(text: str) -> str:
    """Make known product names deterministic without touching TN semantics."""
    for pattern, replacement in _PRODUCT_FORMS:
        text = _sub_lexeme(pattern, replacement, text)
    return text
