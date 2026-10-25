"""Explicit monetary amounts, ordinals and labelled version numbers.

These are written formats, not pronunciation dictionary entries. Callers must
protect addresses and explicitly labelled IDs before accepting these spans.
Unknown alphanumeric names are deliberately not guessed to be identifiers.
"""

import re

from .math_units import _english_cardinals, _english_quantity


_AMOUNT = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
_MONEY = re.compile(
    rf"(?<![A-Za-z0-9_.$£€¥￥,+\-−])(?P<sign>[+\-−]?)(?P<symbol>[$£€¥￥])"
    rf"[ \t]*(?P<inner_sign>[+\-−]?)(?P<amount>{_AMOUNT})"
    r"(?![A-Za-z0-9_]|[.,]\d)"
)
_MONEY_SHAPE = re.compile(
    r"(?<![A-Za-z0-9_.$£€¥￥,+\-−])(?P<sign>[+\-−]?)(?P<symbol>[$£€¥￥])"
    r"[ \t]*(?P<body>[+\-−\d.,]+)"
)
_CURRENCIES = {
    "$": ("美元", "dollar", "dollars", "cent", "cents"),
    "£": ("英镑", "pound", "pounds", "penny", "pence"),
    "€": ("欧元", "euro", "euros", "cent", "cents"),
    # Keep this frontend's existing yuan interpretation of these symbols.
    # Read the written decimal in full instead of guessing minor-unit wording.
    "¥": ("元", "yuan", "yuan", None, None),
    "￥": ("元", "yuan", "yuan", None, None),
}
_ORDINAL = re.compile(
    r"(?<![A-Za-z0-9_./+\-])(?P<number>\d+)(?P<suffix>st|nd|rd|th)"
    r"(?![A-Za-z0-9_])", re.IGNORECASE,
)
_VERSION_VALUE = r"\d+(?:\.\d+){1,3}"
_VERSION_END = r"(?![A-Za-z0-9_]|\.\d|[+\-‐‑‒–−﹣－][A-Za-z0-9])"
# A bare dotted number could be an address, date or decimal. Require v/V or
# an explicit version label; avoid claiming a prefix of a prerelease/build ID.
_VERSION = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<prefix>[vV])(?P<value>{_VERSION_VALUE})"
    + _VERSION_END
)
_VERSION_CUED = re.compile(
    rf"(?:\bversion(?=[ \t:：\d])|版本(?:号)?)[ \t]*[:：]?[ \t]*(?P<value>{_VERSION_VALUE})"
    + _VERSION_END, re.IGNORECASE,
)
_DIGITS_EN = "zero one two three four five six seven eight nine".split()


def _english_number(value):
    # Hyphens in generated English words are not mathematical minus signs;
    # the legacy Chinese postprocessor would otherwise turn them into pauses.
    return _english_quantity(value).replace("-", " ")


def _money(match, language, verbalize_number):
    sign = match["sign"] or match["inner_sign"]
    amount = match["amount"].replace(",", "")
    zh_label, singular, plural, minor_one, minor_many = _CURRENCIES[match["symbol"]]
    if language == "ZH":
        canonical_sign = sign.replace("−", "-")
        return verbalize_number(canonical_sign + amount, language) + zh_label
    prefix = "minus " if sign in ("-", "−") else "plus " if sign == "+" else ""
    if minor_one is None:
        return prefix + _english_number(amount) + " " + plural
    whole, dot, fraction = amount.partition(".")
    units = int(whole or "0")
    if dot and len(fraction) > 2:
        return prefix + _english_number(amount) + " " + plural
    minor = int(fraction.ljust(2, "0")) if dot else 0
    pieces = []
    if units or not minor:
        pieces.append(_english_number(str(units)) + " " + (singular if units == 1 else plural))
    if minor:
        pieces.append(_english_number(str(minor)) + " " + (minor_one if minor == 1 else minor_many))
    return prefix + " and ".join(pieces)


def _valid_ordinal(match):
    n = int(match["number"])
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return match["suffix"].lower() == suffix


def _version(value, language, verbalize_number):
    parts = []
    for part in value.split("."):
        if len(part) > 1 and part.startswith("0"):
            spoken = ("".join("零一二三四五六七八九"[int(c)] for c in part)
                      if language == "ZH" else " ".join(_DIGITS_EN[int(c)] for c in part))
        else:
            spoken = (verbalize_number(part, language) if language == "ZH"
                      else _english_number(part))
        parts.append(spoken)
    return ("点" if language == "ZH" else " point ").join(parts)


def find_spans(text, language, verbalize_number):
    """Return complete formats only; never consume partial malformed fields."""
    language = "ZH" if language.lower().startswith("zh") else "EN"
    spans = []
    for match in _MONEY.finditer(text):
        if not (match["sign"] and match["inner_sign"]):
            spans.append((match.start(), match.end(), _money(match, language, verbalize_number)))
    for match in _ORDINAL.finditer(text):
        if _valid_ordinal(match):
            number = int(match["number"])
            spoken = ("第" + verbalize_number(str(number), language) if language == "ZH"
                      else _english_cardinals().number_to_words(
                          _english_cardinals().ordinal(number), andword=""
                      ).replace(",", "").replace("-", " "))
            spans.append((match.start(), match.end(), spoken))
    for match in _VERSION.finditer(text):
        spans.append((match.start(), match.end(), "V " + _version(match["value"], language, verbalize_number)))
    for match in _VERSION_CUED.finditer(text):
        spans.append((match.start("value"), match.end("value"), _version(match["value"], language, verbalize_number)))
    return sorted(spans)


def malformed_money_ranges(text):
    """Find numeric currency shapes that cannot be read as one amount.

    This is a diagnostic query; ordinary field finders keep returning only
    valid spans. Callers exclude addresses and identifiers before literal fallback.
    Trailing sentence punctuation is not part of the monetary amount.
    """
    for match in _MONEY_SHAPE.finditer(text):
        body = match["body"].rstrip(".,")
        if not any(char.isdecimal() for char in body):
            continue
        end = match.start("body") + len(body)
        complete = _MONEY.fullmatch(text[match.start():end])
        if complete is None or complete["sign"] and complete["inner_sign"]:
            yield match.start(), end


def protected_version_ranges(text):
    """Version separators are data, even when their prefix looks like a date."""
    for pattern in (_VERSION, _VERSION_CUED):
        for match in pattern.finditer(text):
            yield match.start(), match.end()


def protected_ranges(text):
    """Protect separators from MIX clause splitting without reading numbers."""
    for pattern in (_MONEY, _ORDINAL, _VERSION, _VERSION_CUED):
        for match in pattern.finditer(text):
            yield match.start(), match.end()
