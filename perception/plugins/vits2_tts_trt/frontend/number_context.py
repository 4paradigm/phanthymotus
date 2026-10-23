"""Read structured numeric fields before general lexical/math normalization.

Only explicit fields and recognizable formats are handled here. An unlabelled
2026 or 5690 remains a normal number; its value alone cannot determine intent.
Offsets always refer to the original text, including Unicode hyphens.
"""

import datetime
import re

from .numeric_formats import find_spans as find_numeric_format_spans


_DASH = r"\-‐‑‒–−﹣－"
_DIGITS = {"ZH": tuple("零一二三四五六七八九"),
           "EN": "zero one two three four five six seven eight nine".split()}
_DATE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<year>\d{{4}})(?P<sep>[{_DASH}/.])"
    rf"(?P<month>\d{{1,2}})(?P=sep)(?P<day>\d{{1,2}})(?![A-Za-z0-9_])"
)
_SPACED_DATE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<year>\d{4})[ \t]+"
    r"(?P<month>\d{1,2})[ \t]*/[ \t]*(?P<day>\d{1,2})"
    r"(?![A-Za-z0-9_]|[./]\d)"
)
_TIME = re.compile(
    r"(?<![A-Za-z0-9_:])(?P<hour>\d{1,2})[:：](?P<minute>\d{2})"
    r"(?:[:：](?P<second>\d{2}))?(?![A-Za-z0-9_:：])"
)
_PHONE = re.compile(
    rf"(?<![A-Za-z0-9_+{_DASH}])(?P<area>0\d{{2,3}})\s*[{_DASH}]\s*"
    r"(?P<local>\d{7,8})(?![A-Za-z0-9_]|\.\d)"
)
_PHONE_CUED = re.compile(
    rf"(?:客服电话|联系电话|电话号码|电话|热线|(?i:\b(?:phone|tel|telephone)))"
    rf"\s*(?:号码)?\s*(?:是|为|is)?\s*[:：]?\s*"
    rf"(?P<number>\d{{2,4}}\s*[{_DASH}]\s*\d{{7,8}})(?![A-Za-z0-9_]|\.\d)"
)
_ID_VALUE = rf"[A-Za-z0-9]+(?:[{_DASH}_.][A-Za-z0-9]+)*"
_ID_CUED = re.compile(
    r"(?:订单\s*(?i:ID)|订单号|项目编号|订单编号|编号|序列号|验证码|尾号|"
    r"(?i:\b(?:order\s+(?:id|number)|serial\s+number|verification\s+code|ID\b)))"
    rf"\s*(?:是|为|(?i:is|are))?\s*[:：]?\s*(?P<value>{_ID_VALUE})(?![A-Za-z0-9_]|\.\d)"
)
# An explicitly named identifier may have an owner between its field name and
# value: "order ID for 北京 is 001234". Require both the ownership preposition
# and the copula; bound the owner to one short nonnumeric phrase so this cannot
# turn a later arbitrary quantity into an identifier.
_ID_OWNER_CUED = re.compile(
    r"(?i:\b(?:order\s+(?:id|number)|serial\s+number|verification\s+code))"
    r"\s+(?i:for|of)\s+[A-Za-z\u4e00-\u9fff]"
    r"[A-Za-z\u4e00-\u9fff '\-]{0,48}?\s+(?i:is|are)\s*[:：]?\s*"
    rf"(?P<value>{_ID_VALUE})(?![A-Za-z0-9_]|\.\d)"
)
# A trailing-digit field may name its owner before the number. Keep the scope
# narrow: an explicit four-digit label, a recognized identifier field, short
# alphabetic owners, and exactly four digits. Do not scan across arbitrary
# prose, punctuation, quantities, decimal values or another sentence.
_TAIL_OWNER_WORD = r"[A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff'’\-]{0,23}"
_TAIL_FIELD = (
    r"(?:(?:phone|telephone|mobile)\s+number|"
    r"(?:bank\s+)?account(?:\s+number)?|"
    r"(?:credit\s+|debit\s+)?card(?:\s+number)?|"
    r"order\s+(?:id|number)|serial\s+number|"
    r"(?:tracking|reference)\s+(?:id|number)|verification\s+code)"
)
_TRAILING_FOUR_DIGITS = re.compile(
    r"\b(?:last|final)\s+(?:four|4)\s+digits"
    rf"(?:\s+(?:of|on|from)\s+(?:{_TAIL_OWNER_WORD}\s+){{0,3}}{_TAIL_FIELD}"
    rf"(?:\s+(?:for|of)\s+{_TAIL_OWNER_WORD}(?:\s+{_TAIL_OWNER_WORD}){{0,2}})?)?"
    r"(?:\s+(?:is|are)\s*[:：]?\s*|\s*[:：]\s*|\s+)"
    r"(?P<value>\d{4})(?![A-Za-z0-9_]|[.,]\d)",
    re.IGNORECASE,
)
_ZH_TRAILING_FOUR_DIGITS = re.compile(
    r"(?:最后|末|后)\s*(?:四|4)\s*位\s*(?:是|为)?\s*[:：]?\s*"
    r"(?P<value>\d{4})(?![A-Za-z0-9_]|[.,]\d)"
)
_NUMBERED_ID = re.compile(
    r"(?<![A-Za-z0-9_])(?i:NO)\.(?P<value>\d+)(?![A-Za-z0-9_]|\.\d)"
)
_GPU_MODEL = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<series>(?i:RTX|GTX))\s*[{_DASH}]?\s*"
    r"(?P<number>\d{3,4})(?![A-Za-z0-9_]|\.\d)"
)
_SYMBOL_LABEL = re.compile(
    r"(?<![A-Za-z0-9_])(?P<label>(?i:lr|snr))(?=\s*(?:=|>=|<=|≥|≤|>|<))"
)
_MONTHS = ("", "January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")
_DAYS = ("", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
         "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
         "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
         "nineteenth", "twentieth", "twenty first", "twenty second",
         "twenty third", "twenty fourth", "twenty fifth", "twenty sixth",
         "twenty seventh", "twenty eighth", "twenty ninth", "thirtieth", "thirty first")


def _digits(value, language):
    return ("" if language == "ZH" else " ").join(_DIGITS[language][int(c)] for c in value)


def _identifier(value, language):
    if value.isdecimal():
        return _digits(value, language)
    return " ".join(
        _DIGITS[language][int(c)] if c.isdecimal() else c.upper() if c.isalpha()
        else ("下划线" if language == "ZH" else "underscore") if c == "_"
        else ("点" if language == "ZH" else "dot") if c == "."
        else ("连字符" if language == "ZH" else "hyphen")
        for c in value
    )


def find_spans(text, language, verbalize_number):
    """Return nonoverlapping (start, end, spoken) spans with contextual readings."""
    language = "ZH" if language.lower().startswith("zh") else "EN"
    zh = language == "ZH"
    spans = []

    def add(start, end, spoken):
        if not any(start < b and end > a for a, b, _ in spans):
            spans.append((start, end, spoken))

    def cardinal(value):
        return verbalize_number(str(int(value)), language)

    # An explicit field label takes priority over date/phone/unit shapes:
    # order ID 12A is an identifier, not twelve amperes.
    for pattern in (_TRAILING_FOUR_DIGITS, _ZH_TRAILING_FOUR_DIGITS):
        for match in pattern.finditer(text):
            add(match.start("value"), match.end("value"), _digits(match["value"], language))
    for match in _ID_CUED.finditer(text):
        if any(c.isdigit() for c in match["value"]):
            add(match.start("value"), match.end("value"), _identifier(match["value"], language))
    for match in _ID_OWNER_CUED.finditer(text):
        if any(c.isdigit() for c in match["value"]):
            add(match.start("value"), match.end("value"), _identifier(match["value"], language))
    for match in _NUMBERED_ID.finditer(text):
        add(match.start(), match.end(), "N O " + _digits(match["value"], language))

    for start, end, spoken in find_numeric_format_spans(text, language, verbalize_number):
        add(start, end, spoken)

    # Recognize a date inside prose before math can consume its month/day as a
    # fraction. Horizontal whitespace is intentional: never join separate lines.
    for match in sorted(
        (match for pattern in (_DATE, _SPACED_DATE) for match in pattern.finditer(text)),
        key=lambda match: match.start(),
    ):
        year, month, day = (int(match[key]) for key in ("year", "month", "day"))
        try:
            datetime.date(year, month, day)
        except ValueError:
            continue
        if zh:
            spoken = _digits(match["year"], language) + "年" + cardinal(month) + "月" + cardinal(day) + "日"
        else:
            spoken = f"{_MONTHS[month]} {_DAYS[day]}, {cardinal(year)}"
        add(match.start(), match.end(), spoken)

    for match in _TIME.finditer(text):
        hour, minute = int(match["hour"]), int(match["minute"])
        second = int(match["second"]) if match["second"] is not None else None
        if hour > 23 or minute > 59 or (second is not None and second > 59):
            continue
        left = re.split(r"[。！？!?，,；;\n]", text[:match.start()])[-1][-24:]
        if re.search(r"比例|比值|(?i:\bratio\b)", left):
            continue
        if zh:
            spoken = cardinal(hour) + "点"
            if minute:
                spoken += ("零" if minute < 10 else "") + cardinal(minute) + "分"
            elif second is None:
                spoken += "整"
            else:
                spoken += "零分"
            if second is not None:
                spoken += cardinal(second) + "秒"
        else:
            spoken = f"{cardinal(hour)} {'hour' if hour == 1 else 'hours'}"
            spoken += f" {cardinal(minute)} {'minute' if minute == 1 else 'minutes'}"
            if second is not None:
                spoken += f" {cardinal(second)} {'second' if second == 1 else 'seconds'}"
        add(match.start(), match.end(), spoken)

    for match in _PHONE_CUED.finditer(text):
        groups = re.split(rf"\s*[{_DASH}]\s*", match["number"])
        add(match.start("number"), match.end("number"), ", ".join(_digits(x, language) for x in groups))
    for match in _PHONE.finditer(text):
        left = re.split(r"[。！？!?，,；;\n]", text[:match.start()])[-1][-24:]
        if re.search(r"计算|算式|减去|(?i:\b(?:calculate|compute|subtract|equation)\b)", left):
            continue
        if re.search(r"[=+*×÷^<>≤≥≠]\s*$", left) or re.match(r"\s*[=+*×÷^<>≤≥≠]", text[match.end():]):
            continue
        add(match.start(), match.end(), _digits(match["area"], language) + ", " + _digits(match["local"], language))

    for match in _GPU_MODEL.finditer(text):
        add(match.start(), match.end(), " ".join(match["series"].upper()) + " " + _digits(match["number"], language))
    for match in _SYMBOL_LABEL.finditer(text):
        add(match.start(), match.end(), " ".join(match["label"].upper()))
    return sorted(spans)
