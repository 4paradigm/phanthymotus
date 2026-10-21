"""Read numeric-only speech in the resolved conversation language.

The original request remains intact. This produces a frontend input copy using
the released number/date readers, without loading a model or changing MIX rules.
"""

import datetime
import re

from ..vits2_tts_trt.frontend.math_units import _english_quantity
from ..vits2_tts_trt.frontend.number_context import _DATE
from ..vits2_tts_trt.frontend.special_text import (
    normalize_special_text, verbalize_chinese_number,
)


_SPACED_DATE = re.compile(
    r"(?P<year>\d{4})[ \t]+(?P<month>\d{1,2})[ \t]*/[ \t]*(?P<day>\d{1,2})"
)
_NUMBER = re.compile(
    r"[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
)
_FULLWIDTH_ASCII = {code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)}
_FULLWIDTH_ASCII[0x3000] = ord(" ")
_TERMINAL_PUNCTUATION = re.compile(r"[.。!?！？]+$")


class InvalidDateError(ValueError):
    """A complete numeric date shape contains an invalid calendar date."""


def is_neutral_numeric(text):
    return any(char.isdecimal() for char in text) and not any(char.isalpha() for char in text)


def normalize_neutral_text(text, language):
    """Normalize numeric speech only, with canonical zh/en chosen by policy."""
    if language not in ("zh", "en") or not is_neutral_numeric(text):
        raise ValueError("Numeric speech requires a resolved Chinese or English language")
    # Fold width only: NFKC would turn mathematical 2² into the number 22.
    original = text.strip()
    text = original.translate(_FULLWIDTH_ASCII)
    # The supported spaced variant is deliberately a complete date shape.
    # Three whitespace-separated numbers and cross-line fragments are not dates.
    # Width folding is one-to-one, so a folded fullwidth stop can be recognized
    # while the exact original punctuation is retained for the frontend copy.
    terminal = _TERMINAL_PUNCTUATION.search(text)
    date_text = text[:terminal.start()].rstrip() if terminal else text
    suffix = original[terminal.start():] if terminal else ""
    match = _SPACED_DATE.fullmatch(date_text) or _DATE.fullmatch(date_text)
    if match is None:
        prefix = _SPACED_DATE.match(date_text) or _DATE.match(date_text)
        if prefix and re.match(r"\.\d", date_text[prefix.end():]):
            raise InvalidDateError("A complete date cannot have a numeric suffix")
    if match:
        year, month, day = (int(match[key]) for key in ("year", "month", "day"))
        try:
            datetime.date(year, month, day)
        except ValueError:
            raise InvalidDateError("The numeric date is not a valid calendar date") from None
        text = f"{year:04d}/{month:02d}/{day:02d}" + suffix

    selected = language.upper()

    def verbalize(value, chosen):
        value = value.replace(",", "").replace("−", "-")
        return verbalize_chinese_number(value) if chosen == "ZH" else _english_quantity(value)

    def normalize_plain(fragment):
        # Special spans have already consumed dates, fractions, math and units.
        # Explicitly read remaining numbers in the selected language. Keep
        # whitespace-separated numbers distinct when Chinese TN removes spaces.
        output, cursor, previous_number = [], 0, False
        for number in _NUMBER.finditer(fragment):
            separator = fragment[cursor:number.start()]
            output.append(", " if previous_number and separator.isspace() else separator)
            output.append(verbalize(number.group(), selected))
            cursor, previous_number = number.end(), True
        output.append(fragment[cursor:])
        return "".join(output)

    normalized = normalize_special_text(text, normalize_plain, verbalize, language=selected)
    if any(char.isnumeric() and not char.isalpha() for char in normalized):
        raise ValueError("Numeric speech contains an unsupported numeric form")
    return normalized
