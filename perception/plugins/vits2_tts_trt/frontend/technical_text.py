"""Stable lexical handling for technology names after Chinese TN."""

import re


def _edge(pattern: str) -> re.Pattern:
    """CJK-safe ASCII word boundary (``\\b`` fails next to 中文)."""
    return re.compile(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])",
                      re.IGNORECASE)


_PRODUCT_FORMS = (
    (re.compile(r"\bchatgpt\b", re.IGNORECASE), "chat GPT"),
    (re.compile(r"\bopenai\b", re.IGNORECASE), "open AI"),
    (re.compile(r"\bchattts\b", re.IGNORECASE), "chat T T S"),
    (re.compile(r"\bstyletts2\b", re.IGNORECASE), "style T T S二"),
    (_edge(r"rise\s+vgpu"), "rise V G P U"),
    (_edge(r"phanthymotus"), "Phanthy Motus"),
    (_edge(r"https"), "H T T P S"),
    (_edge(r"http"), "H T T P"),
)


def normalize_technical_lexemes(text: str) -> str:
    """Make known product names deterministic without touching TN semantics."""
    for pattern, replacement in _PRODUCT_FORMS:
        text = pattern.sub(replacement, text)
    return text
