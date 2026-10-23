"""Read known names in the selected language or their local bilingual clause."""

import re

from ..vits2_tts_trt.frontend.lexicon_rules import (
    _QWEN_VERSION_RE, apply_lexeme_rules, has_chinese_name_reading,
)
from ..vits2_tts_trt.frontend.math_units import _english_quantity
from ..vits2_tts_trt.frontend.mixed_context import context_diagnostics
from ..vits2_tts_trt.frontend.number_context import _ID_CUED, _ID_OWNER_CUED, _NUMBERED_ID
from ..vits2_tts_trt.frontend.special_text import _ADDRESSES


_TERMINAL_PUNCTUATION = re.compile(r"[.。!?！？]+$")
_ID_PREFIX = re.compile(
    r"\b(?:order\s+(?:id|number)|serial\s+number|verification\s+code|model_id|identifier|id)[\"']?"
    r"(?:\s+(?:for|of)\s+[A-Za-z '\-]{1,50})?\s*(?:is|are)?\s*[:=]?\s*[\"']?\s*$",
    re.IGNORECASE,
)
_NAME_JOINERS = "_/@\\+‐‑‒–—−-"


def _english_version_names(text, allowed_ranges=None):
    protected = [
        (match.start(), match.end())
        for pattern in (_ADDRESSES, _ID_CUED, _ID_OWNER_CUED, _NUMBERED_ID)
        for match in pattern.finditer(text)
    ]

    def replace(match):
        start, end = match.span()
        if allowed_ranges is not None and not any(
            a <= start and end <= b for a, b in allowed_ranges
        ):
            return match.group()
        if any(start < b and end > a for a, b in protected):
            return match.group()
        # A version inside a longer model, path or identifier is not this name.
        if start and text[start - 1] in _NAME_JOINERS + ".":
            return match.group()
        if end < len(text) and text[end] in _NAME_JOINERS:
            return match.group()
        if end + 1 < len(text) and text[end] in ".:" and text[end + 1].isalnum():
            return match.group()
        if _ID_PREFIX.search(text[:start]):
            return match.group()
        brand = text[start:match.start(1)].rstrip()
        return brand + " " + _english_quantity(match.group(1))

    return _QWEN_VERSION_RE.sub(replace, text)


def normalize_standalone_name(text, language):
    """Keep Chinese name readings while preserving English clauses in mixed text."""
    token = text.strip()
    terminal = _TERMINAL_PUNCTUATION.search(token)
    core = token[:terminal.start()].rstrip() if terminal else token
    if language == "zh" and has_chinese_name_reading(core):
        return apply_lexeme_rules(core, language="ZH") + (terminal.group() if terminal else "")
    if language == "en" and not re.search(r"[\u4e00-\u9fff]", text):
        normalized = _english_version_names(text)
        return normalized.strip() if _QWEN_VERSION_RE.fullmatch(core) else normalized
    if language in ("zh", "en") and re.search(r"[\u4e00-\u9fff]", text) and _QWEN_VERSION_RE.search(text):
        english_ranges = [
            (piece["start"], piece["end"])
            for piece in context_diagnostics(text)["segments"]
            if piece["language"] == "EN"
        ]
        # Match against the original full text: splitting first would lose an
        # ID prefix outside quotes and could rewrite its value as a model name.
        # Only the frontend copy changes; punctuation and synthesis stay intact.
        return _english_version_names(text, english_ranges)
    return text
