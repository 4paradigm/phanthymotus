"""Read explicit addresses, formulas and measurements before lexical TN.

The delivered lexicon is still used for every ordinary text span. Special spans
are spoken separately so a product-name rule cannot change an address, decimal
or exponent. This is deterministic text processing, not model inference.
"""
import re

from .math_units import find_spans as find_math_unit_spans, _english_quantity
from .number_context import (
    find_spans as find_number_context_spans, literal_date_ranges, protected_identifier_ranges,
)
from .numeric_formats import malformed_money_ranges, protected_version_ranges

_EMAIL = r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,63}"
_URL = r"(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+"
_ADDRESSES = re.compile(r"(?<![A-Za-z0-9_])(?:" + _URL + "|" + _EMAIL + ")", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]")
_EN_CARDINAL = re.compile(
    r"(?<![A-Za-z0-9_.,$£€¥￥/\-])[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?![A-Za-z0-9_]|\.\d)"
)
_DIGITS = {"ZH": tuple("零一二三四五六七八九"),
           "EN": "zero one two three four five six seven eight nine".split()}
_SYMBOLS = {
    "ZH": {".": "点", "@": "艾特", "/": "斜杠", ":": "冒号", "_": "下划线",
           "-": "连字符", "+": "加号", "?": "问号", "&": "与号", "=": "等号",
           "#": "井号", "%": "百分号", "~": "波浪号", "!": "感叹号",
           "$": "美元符号", "'": "单引号", "(": "左括号", ")": "右括号",
           "[": "左方括号", "]": "右方括号", ",": "逗号", ";": "分号",
           "*": "星号", "^": "脱字符", "`": "反引号", "{": "左花括号",
           "}": "右花括号", "|": "竖线"},
    "EN": {".": "dot", "@": "at", "/": "slash", ":": "colon", "_": "underscore",
           "-": "hyphen", "+": "plus", "?": "question mark", "&": "ampersand",
           "=": "equals", "#": "hash", "%": "percent", "~": "tilde", "!": "exclamation mark",
           "$": "dollar sign", "'": "apostrophe", "(": "left parenthesis", ")": "right parenthesis",
           "[": "left bracket", "]": "right bracket", ",": "comma", ";": "semicolon",
           "*": "asterisk", "^": "caret", "`": "backtick", "{": "left brace",
           "}": "right brace", "|": "vertical bar"},
}


_LITERAL_SYMBOLS = {
    "ZH": {"/": "斜杠", ".": "点", ",": "逗号", "-": "连字符", "+": "加号",
           "−": "减号", "$": "美元符号", "£": "英镑符号", "€": "欧元符号",
           "¥": "元符号", "￥": "元符号"},
    "EN": {"/": "slash", ".": "dot", ",": "comma", "-": "hyphen", "+": "plus sign",
           "−": "minus sign", "$": "dollar sign", "£": "pound sign", "€": "euro sign",
           "¥": "yuan sign", "￥": "yuan sign"},
}
_LITERAL_WIDTH = {code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)}
_LITERAL_WIDTH[0x3000] = ord(" ")
_LITERAL_DASHES = dict.fromkeys(map(ord, "‐‑‒–﹣－"), "-")


def fold_numeric_width(text):
    """Fold ASCII width one-to-one so diagnostic offsets stay unchanged."""
    return text.translate(_LITERAL_WIDTH)


def read_numeric_literal(text, language):
    """Preserve every written digit and internal separator in the chosen language."""
    language = "ZH" if language.lower().startswith("zh") else "EN"
    words = []
    for char in fold_numeric_width(text).translate(_LITERAL_DASHES):
        if char.isspace():
            continue
        words.append(_DIGITS[language][int(char)] if char.isdecimal()
                     else _LITERAL_SYMBOLS[language].get(char, char))
    return ("" if language == "ZH" else " ").join(words)


def verbalize_chinese_number(value):
    """Cardinal reading for a formula/measurement, with exact decimal digits.

    The released FST can classify an isolated 512 as a telephone fragment
    (五幺二). Inside a quantity its reading must be 五百一十二.
    """
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = ("正" if value[0] == "+" else "负"), value[1:]
    whole, dot, fraction = value.partition(".")
    digits = _DIGITS["ZH"]
    n = int(whole or "0")
    groups = []
    while n:
        n, part = divmod(n, 10000)
        groups.append(part)
    scales = ("", "万", "亿", "万亿", "亿亿")
    if len(groups) > len(scales):
        # Spell exceptionally large values, never truncate them.
        spoken = "".join(digits[int(c)] for c in whole)
    else:
        spoken, previous = "", None
        for index in range(len(groups) - 1, -1, -1):
            part = groups[index]
            if not part:
                continue
            if spoken and (previous - index > 1 or part < 1000):
                spoken += "零"
            small, pending_zero = "", False
            for char, unit in zip(str(part).zfill(4), ("千", "百", "十", "")):
                if char == "0":
                    pending_zero = bool(small)
                else:
                    if pending_zero:
                        small += "零"
                    small += digits[int(char)] + unit
                    pending_zero = False
            if not spoken and 10 <= part < 20:
                small = small[1:]
            spoken += small + scales[index]
            previous = index
        spoken = spoken or "零"
    if dot:
        spoken += "点" + "".join(digits[int(c)] for c in fraction)
    return sign + spoken


def _read_address(address, language):
    # Spell identifiers literally. This avoids turning digits into quantities
    # or reinterpreting a local part such as qwen3/API as a brand/acronym rule.
    return " ".join(
        char.upper() if char.isascii() and char.isalpha()
        else _DIGITS[language][int(char)] if char.isascii() and char.isdigit()
        else _SYMBOLS[language][char]
        for char in address
    )


def normalize_special_text(text, normalize_plain, verbalize_number, language=None,
                           language_resolver=None, normalize_cardinals=True):
    """Normalize only recognized special spans; delegate all other text.

    Without an override, language follows the original legacy behavior.
    Mixed routing can supply a clause language and a field-local resolver;
    the resolver receives raw-text start/end offsets. Address punctuation is
    pronounced literally and is never sent through lexical rewriting.
    `verbalize_number(value, language)` must preserve decimal digits.
    A caller providing its own quantity reader can disable the English
    cardinal pass and handle unclaimed numbers in normalize_plain instead.
    """
    if language is None:
        language = "ZH" if _CJK.search(text) else "EN"
    if language not in {"ZH", "EN"}:
        raise ValueError("Special-text language must be ZH or EN")

    def local_language(start, end):
        selected = language if language_resolver is None else language_resolver(start, end, language)
        if selected not in {"ZH", "EN"}:
            raise ValueError("Resolved special-text language must be ZH or EN")
        return selected

    def local_spans(finder, fragment, offset=0):
        primary = finder(fragment, language, verbalize_number)
        if language_resolver is None:
            return primary
        other = "ZH" if language == "EN" else "EN"
        alternative = {(a, b): spoken for a, b, spoken in finder(fragment, other, verbalize_number)}
        return [(a, b, alternative.get((a, b), spoken)
                 if local_language(offset + a, offset + b) == other else spoken)
                for a, b, spoken in primary]
    spans = []
    for match in _ADDRESSES.finditer(text):
        address = match.group().rstrip(".,;")
        while address.endswith(")") and address.count(")") > address.count("("):
            address = address[:-1]
        while address.endswith("]") and address.count("]") > address.count("["):
            address = address[:-1]
        if not address:
            continue
        end = match.start() + len(address)
        spans.append((match.start(), end, _read_address(address, local_language(match.start(), end))))
    # Unsupported numeric syntax is still speakable. Preserve its exact digits
    # and separators before ordinary fields/FST can claim a prefix or drop a
    # currency symbol. Explicit addresses, IDs and versions keep precedence.
    protected_literals = [(a, b) for a, b, _ in spans]
    shape_text = fold_numeric_width(text)
    protected_literals.extend(protected_identifier_ranges(shape_text))
    protected_literals.extend(protected_version_ranges(shape_text))
    literal_candidates = list(malformed_money_ranges(shape_text)) + list(literal_date_ranges(shape_text))
    for start, end in sorted(literal_candidates, key=lambda item: (item[0], -item[1])):
        if (not any(start < b and end > a for a, b in protected_literals)
                and not any(start < b and end > a for a, b, _ in spans)):
            spoken = read_numeric_literal(text[start:end], local_language(start, end))
            spans.append((start, end, spoken))
    # Context is read from the entire original sentence. Addresses take priority,
    # so an identifier-looking fragment inside a URL/email is always read literally.
    for start, end, spoken in local_spans(find_number_context_spans, text):
        if not any(start < b and end > a for a, b, _ in spans):
            spans.append((start, end, spoken))
    spans.sort()
    protected_spans = list(spans)
    cursor = 0
    for start, end, _ in protected_spans + [(len(text), len(text), "")]:
        for a, b, spoken in local_spans(find_math_unit_spans, text[cursor:start], cursor):
            spans.append((cursor + a, cursor + b, spoken))
        cursor = end
    # MIX also sends English-only requests through the Chinese FST. Protect
    # remaining numeric quantities using the existing English number reader;
    # otherwise even "111 samples" can become a Chinese quantity plus "秒amples".
    # Structured fields and formulas have priority over cardinal readings.
    if language == "EN" and normalize_cardinals:
        for match in _EN_CARDINAL.finditer(text):
            if not any(match.start() < b and match.end() > a for a, b, _ in spans):
                number_language = local_language(match.start(), match.end())
                value = match.group().replace(",", "")
                # The general English TN reader treats 2026 as a year even in
                # "2026 robots". Remaining numeric values are quantities;
                # recognized dates/identifiers already have protected spans.
                spoken = (_english_quantity(value) if number_language == "EN"
                          else verbalize_number(value, number_language))
                spans.append((match.start(), match.end(), spoken))
    spans.sort()
    if not spans:
        return normalize_plain(text)
    output = []
    cursor = 0
    for start, end, spoken in spans:
        if cursor < start:
            output.append(normalize_plain(text[cursor:start]))
        output.append(" " + spoken + " ")
        cursor = end
    if cursor < len(text):
        output.append(normalize_plain(text[cursor:]))
    normalized = re.sub(r"[^\S\r\n\v\f\x85\u2028\u2029]+", " ", "".join(output)).strip(" \t")
    # A MIX request can later produce a Chinese-only chunk. Chinese G2P
    # removes whitespace after punctuation while splitting sentences, so
    # retaining it here breaks word2ph alignment in those merged chunks.
    # Preserve English word/letter spacing and ZH/EN boundary spacing.
    normalized = re.sub(
        r"(?<=[\u4e00-\u9fff，。！？；：、,.!?;:]) +"
        r"(?=[\u4e00-\u9fff，。！？；：、,.!?;:])", "", normalized
    )
    if _CJK.search(normalized) and not re.search(r"[A-Za-z]", normalized):
        # e.g. "电流1.2A" starts on MIX's Latin-containing route, but becomes
        # entirely Chinese. Its G2P maps tokenization spaces to pause phones.
        # Join Chinese words without adding those artificial pauses; real
        # commas/stops (including the phone area-code separator) remain intact.
        normalized = normalized.replace(" ", "")
    return normalized
