"""Read numeric/symbolic speech in the resolved conversation language.

The original request remains intact. This produces a frontend input copy using
the released number/date readers, without loading a model or changing MIX rules.
"""

import datetime
import re

from ..vits2_tts_trt.frontend.math_units import _english_quantity, find_spans as math_spans
from ..vits2_tts_trt.frontend.lexicon_rules import normalize_numeric_lexemes
from ..vits2_tts_trt.frontend.number_context import _DATE, _ID_CUED, find_spans as field_spans
from ..vits2_tts_trt.frontend.special_text import (
    _ADDRESSES, normalize_special_text, verbalize_chinese_number,
)


# Complete numeric-only dates also need prefix validation: the bounded field
# parser intentionally does not match an invalid suffix such as 10/23.5.
_SPACED_DATE_SHAPE = re.compile(
    r"(?P<year>\d{4})[ \t]+(?P<month>\d{1,2})[ \t]*/[ \t]*(?P<day>\d{1,2})"
)
_NUMBER = re.compile(
    r"[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
)
_PROSE_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.,$£€¥￥/\-‐‑‒–−])[+\-−]?"
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?![A-Za-z0-9_]|[.,/]\d|[\-‐‑‒–−][A-Za-z0-9_])"
)
_FULLWIDTH_ASCII = {code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)}
_FULLWIDTH_ASCII[0x3000] = ord(" ")
_TERMINAL_PUNCTUATION = re.compile(r"[.。!?！？]+$")
_NEUTRAL_ID_LABEL = re.compile(
    r"[ \t]*(?:ID|order[ \t]+(?:ID|number)|serial[ \t]+number|verification[ \t]+code)"
    r"[ \t]*[:：]?[ \t]*", re.IGNORECASE,
)


class InvalidDateError(ValueError):
    """A complete numeric date shape contains an invalid calendar date."""


def is_neutral_numeric(text):
    return any(char.isdecimal() for char in text) and not any(char.isalpha() for char in text)


def is_neutral_symbolic(text):
    """Recognize complete supported fields/formulas, not English prose.

    Variable letters, unit symbols and model prefixes do not establish a
    spoken language. Reuse the existing parsers and require complete coverage;
    arbitrary words and unsupported suffixes keep their old route. Complete
    addresses carry no prose-language evidence either. Protect them before
    structured fields and math, as in special_text, so URL path/query numbers
    cannot affect the coverage check or be reinterpreted as dates/formulas.
    """
    text = text.translate(_FULLWIDTH_ASCII)
    spans = [(match.start(), match.end(), "") for match in _ADDRESSES.finditer(text)]
    for start, end, spoken in field_spans(text, "ZH", lambda value, language: value):
        if not any(start < b and end > a for a, b, _ in spans):
            spans.append((start, end, spoken))
    fields = sorted(spans)
    cursor = 0
    for start, end, _ in fields + [(len(text), len(text), "")]:
        spans.extend((cursor + a, cursor + b, spoken) for a, b, spoken in
                     math_spans(text[cursor:start], "ZH", lambda value, language: value))
        cursor = end
    remaining = list(text)
    for start, end, _ in spans:
        remaining[start:end] = " " * (end - start)
        # A measured/scientific RHS consumes its comparison but deliberately
        # leaves the variable label outside the span (x=2kg, x=1e-4). Accept
        # the same single-letter variables as the existing math grammar.
        if re.match(r"(?:<=|>=|!=|[=<>≤≥≠])", text[start:end]):
            label = re.search(r"(?<![A-Za-z0-9_])([A-Za-z])\s*$", text[:start])
            if label:
                remaining[label.start():start] = " " * (start - label.start())
    # A complete metadata field such as "ID: abc123" is not English prose.
    # Only discard this exact ASCII label for detection, never a surrounding
    # sentence ("The order ID is ...") or a Chinese language-bearing label.
    for match in _ID_CUED.finditer(text):
        if (_NEUTRAL_ID_LABEL.fullmatch(text[:match.start("value")])
                and not text[match.end("value"):].strip(" \t\r\n.,，。;；!?！？")
                and any(start == match.start("value") and end == match.end("value")
                        for start, end, _ in spans)):
            remaining[:match.start("value")] = " " * match.start("value")
    return bool(spans) and not "".join(remaining).strip(" \t\r\n.,，。;；:：!?！？")


def is_neutral_text(text):
    return is_neutral_numeric(text) or is_neutral_symbolic(text)


def has_reading_fields(text):
    """Cheap eligibility for deferring a primary-script check until TN.

    This does not accept the final reply: the prepared frontend output must
    still contain the selected primary language before any audio is queued.
    """
    text = text.translate(_FULLWIDTH_ASCII)
    unchanged_number = lambda value, language: value
    # An arbitrary number in foreign prose (e.g. "Qwen 3 is ready") is not a
    # reason to weaken the explicit dialogue contract. Only recognized fields
    # defer the check; neutral numeric-only speech already has its own path.
    return bool(_ADDRESSES.search(text)
                or field_spans(text, "ZH", unchanged_number)
                or math_spans(text, "ZH", unchanged_number))


def normalize_mixed_text(text, language):
    """Apply an explicit reading preference to fields, preserving prose.

    Fixed panel modes and explicit dialogue turns may contain words from both
    languages. Those words are not translated. Only supported structured fields
    and standalone quantities are verbalized here, before MIX chooses its G2P.
    Auto-mode prose keeps the original clause-local routing instead.
    """
    if language not in ("zh", "en"):
        raise ValueError("A mixed reading preference must be zh or en")
    selected = language.upper()

    def verbalize(value, chosen):
        value = value.replace(",", "").replace("−", "-")
        return verbalize_chinese_number(value) if chosen == "ZH" else _english_quantity(value)

    def normalize_plain(fragment):
        # Known names can introduce digits only after ordinary field parsing
        # (e.g. a generated Roman-numeral pronunciation). Resolve those narrow
        # name rules here so the fixed request language survives later MIX TN.
        fragment = normalize_numeric_lexemes(fragment, language=selected)
        from ..vits2_tts_trt.frontend.alphanumeric_text import normalize_unknown_alphanumeric_tokens
        fragment = normalize_unknown_alphanumeric_tokens(fragment, selected)
        # Do not split numbers inside names/IDs, versions or currency notation.
        # Their existing frontend rules still run after this preprocessing.
        return _PROSE_NUMBER.sub(lambda match: verbalize(match.group(), selected), fragment)

    # normalize_special_text joins whitespace inside a line. Preserve explicit
    # paragraph boundaries rather than silently changing the speaking rhythm.
    pieces = re.split(r"(\r\n|\r|\n)", text)
    for index in range(0, len(pieces), 2):
        line = pieces[index].translate(_FULLWIDTH_ASCII)
        pieces[index] = normalize_special_text(
            line, normalize_plain, verbalize, language=selected, normalize_cardinals=False,
        )
    return "".join(pieces)


def normalize_neutral_text(text, language):
    """Read numeric/symbolic speech with canonical zh/en chosen by policy."""
    if language not in ("zh", "en") or not is_neutral_text(text):
        raise ValueError("Neutral speech requires a resolved Chinese or English language")
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
    match = _SPACED_DATE_SHAPE.fullmatch(date_text) or _DATE.fullmatch(date_text)
    if match:
        year, month, day = (int(match[key]) for key in ("year", "month", "day"))
        try:
            datetime.date(year, month, day)
        except ValueError:
            # Keep invalid calendar input intact for the literal numeric span;
            # correcting it to another date would change the supplied content.
            pass
        else:
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
