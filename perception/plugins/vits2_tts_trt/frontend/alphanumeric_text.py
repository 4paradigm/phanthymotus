"""Read unclaimed ASCII letter/digit tokens in an explicit language.

This is a text-only fallback, after known product aliases and structured-field
parsers. It neither loads a pronunciation dictionary nor invents product names.
"""
import re
import unicodedata

from .math_units import find_spans as math_spans
from .number_context import find_spans as field_spans
from .special_text import _ADDRESSES
from .lexicon_rules import has_chinese_name_reading


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
_PART = re.compile(r"[A-Za-z]+|[0-9]+|[._-]")
_DIGITS = {
    "zh": tuple("零一二三四五六七八九"),
    "en": tuple("zero one two three four five six seven eight nine".split()),
}
_SEPARATORS = {
    "zh": {".": "点", "_": "下划线", "-": "连字符"},
    "en": {".": "dot", "_": "underscore", "-": "hyphen"},
}


def _latin_neighbor(char):
    return (char.isalpha() and "LATIN" in unicodedata.name(char, "")
            or unicodedata.category(char).startswith("M")
            or char.isascii() and (char.isalnum() or char == "_"))


def normalize_unknown_alphanumeric_tokens(text, language):
    """Expand identifier digits after existing aliases and fields have run.

    Pure quantities and complete known fields, formulas, units, addresses and
    versions remain owned by their established parsers. Remaining ASCII tokens
    containing both letters and digits are read as identifiers, digit by digit.
    Letter runs keep their spelling; a one-letter run denotes a spelled letter.
    """
    if not isinstance(text, str):
        raise TypeError("Alphanumeric text must be a string")
    if not isinstance(language, str) or language.lower() not in _DIGITS:
        raise ValueError("Alphanumeric reading language must be zh or en")
    language = language.lower()
    if not re.search(r"[A-Za-z]", text) or not re.search(r"[0-9]", text):
        return text
    # These parsers are text-only. ZH plus an identity reader obtains protected
    # offsets without importing CMU/G2P or the English cardinal-number engine.
    unchanged = lambda value, chosen: value
    protected = [(match.start(), match.end()) for match in _ADDRESSES.finditer(text)]
    protected += [(a, b) for a, b, _ in field_spans(text, "ZH", unchanged)]
    protected += [(a, b) for a, b, _ in math_spans(text, "ZH", unchanged)]

    def replace(match):
        token, start, end = match.group(), match.start(), match.end()
        if not (re.search(r"[A-Za-z]", token) and re.search(r"[0-9]", token)):
            return token
        if ((start and _latin_neighbor(text[start - 1]))
                or (end < len(text) and _latin_neighbor(text[end]))):
            return token
        # An unsupported grouped quantity such as 1,234XYZ must not be split
        # into a quantity prefix plus an unrelated "234XYZ" identifier.
        if ((start > 1 and text[start - 1] == "," and text[start - 2].isdigit())
                or (end + 1 < len(text) and text[end] == "," and text[end + 1].isdigit())):
            return token
        if any(a <= start and end <= b for a, b in protected):
            return token
        # Leave complete curated names to their existing whole-name rule.
        # Splitting the version from such a name would lose the boundary that
        # its later EN/MIX alias pass uses (and could produce a hybrid reading).
        if has_chinese_name_reading(token):
            return token
        parts = []
        for part in _PART.findall(token):
            if part.isascii() and part.isdigit():
                separator = " " if language == "en" else ""
                parts.append(separator.join(_DIGITS[language][int(char)] for char in part))
            elif part in _SEPARATORS[language]:
                parts.append(_SEPARATORS[language][part])
            else:
                parts.append(part.upper() if len(part) == 1 else part)
        return " ".join(parts)

    return _TOKEN.sub(replace, text)
