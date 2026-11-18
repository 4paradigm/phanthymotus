"""Resident Kokoro Multi text normalization, pronunciation dictionaries and G2P.

All language resources remain in the verified external release. Thai tones and
Russian/German substitutions retain the legacy114-symbol audition baseline.
The trained119-symbol release preserves native phonemes; review is still required. No TTSV4 source or acoustic weights are imported here.
"""
from __future__ import annotations

from datetime import date
from dataclasses import replace
from decimal import Decimal
from functools import lru_cache
from itertools import accumulate, groupby
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from unicodedata import category


_LANGUAGES = {"zh", "en", "ja", "ru", "de", "fr", "es", "th"}
_PHONE_HINT = re.compile(r"\[[^\]\n]+\]\(/[^/\n]+/\)")
_ADDRESS = re.compile(
    r"`[^`\n]+`|(?<!\w)--[A-Za-z][\w-]*(?:=[^\s;,]+)?|"
    r"(?<!\w)-[A-Za-z]\s+[+\-]?\d+(?:\.\d+)?|https?://[^\s<>`]+|www\.[^\s<>`]+|"
    r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+|"
    r"(?<!\w)(?:[A-Za-z]:[\\/]|\.?\.?/)[\w./\\+@~%-]+|"
    r"(?<![\w/])[\w.-]+(?:[/\\][\w.+@~-]+)*[/\\][\w.-]+\.[A-Za-z]{1,8}(?!\w)|"
    r"(?<![\w.])[\w+-]+(?:\.[\w+-]+)*\.(?:txt|csv|jsonl?|onnx|pt|pth|wav|mp3|py|sh|zip|pdf|md|yaml|yml|log)(?!\w)",
    re.IGNORECASE,
)
_VERSION = re.compile(
    r"(?:(?P<label>\b(?:version|ver\.?|versión|верси(?:я|и|ю|ей|й))\s*|版本\s*|バージョン\s*|เวอร์ชัน\s*|รุ่น\s*)(?P<v1>[vV]?\d+(?:\.\d+)+)|"
    r"(?<![A-Za-z0-9_.])(?P<v2>[vV]\d+(?:\.\d+)+))(?![A-Za-z0-9_]|\.[A-Za-z0-9_])", re.IGNORECASE,
)
_DATE = re.compile(r"(?<![A-Za-z0-9_.\-])\d{4}-\d{2}-\d{2}(?![A-Za-z0-9_\-]|\.[A-Za-z0-9_])")
_DATE_LIKE = re.compile(r"(?<![A-Za-z0-9_])\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[-/.]\d+)*(?![A-Za-z0-9_])")
_AMBIGUOUS_PAIR = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:-\d+|\s*/\s*(?:\d+|[A-Za-z]{1,3}))(?![A-Za-z0-9_.])")
_DOTTED = re.compile(r"(?<![A-Za-z0-9_.])\d{1,4}(?:\.\d{1,4}){2,}(?![A-Za-z0-9_.])")
_CLOCK = re.compile(r"(?<![A-Za-z0-9_:])\d{1,3}:\d{2}(?::\d{2})?(?![A-Za-z0-9_:])")
_FR_CLOCK = re.compile(r"(?<![\w:.,])(?P<h>\d{1,2})[ \t]*h[ \t]*(?P<m>\d{2})(?![\w:]|[.,]\d)")
_FR_METRE = re.compile(r"(?<![\w.,+\-−])(?P<n>[+\-−]?\d+(?:[.,]\d+)?)[ \t]+m(?![\w/^²³])")
_NUM = r"(?:\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
_EN_GROUPED_NUM = r"[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?"
_EU_GROUPED_DECIMAL = r"[1-9]\d{0,2}(?:\.\d{3})+,\d+"
_EN_GROUPED = re.compile(r"(?<![A-Za-z0-9_.,])(?P<n>[+\-−]?" +
                         _EN_GROUPED_NUM + r")(?![A-Za-z0-9_]|[.,]\d)")
_INTEGER_COUNT = re.compile(r"(?<![\w.,])(?P<n>[+\-−]?(?:[1-9]\d{0,2}(?P<sep>[.,])\d{3}"
                            r"(?:(?P=sep)\d{3})*|[1-9]\d{3,}))(?!\w|[.,]\d)")
_SPACE_GROUPED_NUM = r"[1-9]\d{0,2}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?"
_SPACE_GROUPED = re.compile(r"(?<![A-Za-z0-9_.,])(?<!\d )(?<!\d\u00a0)(?<!\d\u202f)"
                            r"(?P<n>[+\-−]?" + _SPACE_GROUPED_NUM +
                            r")(?![A-Za-z0-9_]|[.,]\d|[ \u00a0\u202f]\d)")
_JA_MINUTES = re.compile(r"(?<![A-Za-z0-9_.])(?P<n>\d{1,2})[ \t]*分"
                         r"(?!の[\d〇零一二三四五六七八九十百千万億兆])"
                         r"(?=間|前|後|以上|以下|以内|未満|程度|[^\u4e00-\u9fff]|$)")
_IPV4 = re.compile(r"(?<![A-Za-z0-9_.])(?:\d{1,3}\.){3}\d{1,3}(?![A-Za-z0-9_]|\.\d)")
_IPV4_CUES = {
    "zh": r"(?:IPv4|IP(?:地址)?|地址)(?:是|为|為)?",
    "en": r"\b(?:ipv4|ip|address)(?:[ \t]+(?:address|is|at))?",
    "ja": r"(?:IPv4|IP(?:アドレス)?|アドレス)(?:は|が)?",
    "de": r"\b(?:ipv4|ip(?:[- ]adresse)?|adresse)(?:[ \t]+(?:ist|lautet))?",
    "fr": r"\b(?:ipv4|ip|adresse(?:[ \t]+ip)?)(?:[ \t]+(?:d['’]exemple|de[ \t]+(?:test|documentation))){0,2}(?:[ \t]+(?:est|vaut))?",
    "es": r"\b(?:ipv4|ip|dirección(?:[ \t]+ip)?)(?:[ \t]+de[ \t]+(?:prueba|ejemplo|documentación)){0,2}(?:[ \t]+es)?",
    "ru": r"\b(?:ipv4|ip(?:[- ]адрес)?|адрес)(?:[ \t]+(?:сервера|хоста|равен))?",
    "th": r"(?:IPv4|IP|ไอพี|ที่อยู่(?:ไอพี)?)(?:คือ|เป็น)?",
}
_EN_MONEY = re.compile(r"(?<![A-Za-z0-9_])(?P<left>[+\-−]?)(?P<c>[$£€])"
                       r"(?P<right>[+\-−]?)(?P<n>" + _EN_GROUPED_NUM + "|" +
                       _NUM + r")(?![A-Za-z0-9_]|[.,]\d)")
_EN_HARDWARE = re.compile(r"(?<![A-Za-z0-9_])(?P<family>RTX|GTX)[ \t]*"
                          r"(?P<n>[1-9]\d{3})(?:[ \t]*(?P<suffix>Ti|SUPER))?"
                          r"(?![A-Za-z0-9_])", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])" + _NUM + r"(?![A-Za-z0-9_]|\.[A-Za-z0-9_])")
_CODE = re.compile(r"(?<![A-Za-z0-9_.])(?:[A-Z]+\d[A-Z0-9]*|\d+[A-Z][A-Z0-9]*)"
                   r"(?:[-_][A-Z0-9]+)*(?![A-Za-z0-9_-]|\.[A-Za-z0-9_])")
_FORMULA_LIKE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[+\-−][ \t]*)?[A-Za-z\u0370-\u03ff0-9_().]+"
    r"(?:[²³][A-Za-z\u0370-\u03ff0-9_().]*|"
    r"[ \t]*[+\-−‑×·*/÷^=<>≤≥≠≈][ \t]*[+\-−]?[ \t]*[A-Za-z\u0370-\u03ff0-9_().]*)+"
    r"|(?<![A-Za-z0-9_])[A-Za-z]+[ \t]*\([A-Za-z\u0370-\u03ff0-9_().+\-−‑×·*/÷^²³ \t]+\)"
)
_TOKEN = re.compile(_NUM + r"|[A-Za-z]+|π|[()+\-−×·*÷/^²³=<>≤≥≠≈]")
_COMMA_LANGUAGES = {"de", "fr", "es", "ru"}
_RU_UNIT_ALIASES = {
    "ГБ": "GB", "мс": "ms", "ms": "ms", "кВт": "kW", "Вт": "W",
    "мВт": "mW", "МВт": "MW", "В": "V", "мВ": "mV", "МВ": "MV",
    "А": "A", "мА": "mA", "Гц": "Hz", "кГц": "kHz", "МГц": "MHz",
    "ГГц": "GHz", "кг": "kg", "мг": "mg", "см": "cm",
    "мм": "mm", "км": "km", "мкм": "μm", "кДж": "kJ", "°С": "°C",
}
_COMMA_NUM = _NUM.replace(r"\.", "[.,]")
_COMMA_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(?<!\d,)" + _COMMA_NUM + r"(?![A-Za-z0-9_]|\.[A-Za-z0-9_])(?!,\d)")
_COMMA_TOKEN = re.compile(_TOKEN.pattern.replace(_NUM, _COMMA_NUM))
_COMMA_FORMULA_LIKE = re.compile(_FORMULA_LIKE.pattern.replace("_().", "_().,"))
_RELATIONS = {"=": "equals", "<": "less", ">": "greater", "≤": "le", "≥": "ge", "≠": "ne", "≈": "approx"}
_EN_MATH_COPULAS = {"is", "are", "was", "were"}
_IDENTIFIER_CUES = {
    "zh": r"验证码", "ja": r"認証コード|確認コード", "en": r"verification code",
    "ru": r"проверочный код|код подтверждения|номер заказа", "de": r"Bestätigungscode",
    "fr": r"code de vérification(?: fictif)?|numéro de commande",
    "es": r"código de verificación(?: ficticio)?|número de pedido",
    "th": r"รหัสยืนยัน(?:สมมติ)?(?:ของคุณ)?|หมายเลขคำสั่งซื้อ",
}

_lexicon_path = None


def configure_lexicon(path):
    global _lexicon_path
    _lexicon_path = Path(path)
    _config.cache_clear()
    _unit_pattern.cache_clear()


@lru_cache(maxsize=1)
def _config():
    if _lexicon_path is None:
        raise RuntimeError("Kokoro Multi lexicon is not configured")
    cfg = json.loads(_lexicon_path.read_text(encoding="utf-8"))["normalization"]
    # Mathematical variable names must stay in the sentence's language.
    # Unlisted letters otherwise take the English branch in these three G2Ps.
    for language, names in {
        "ja": "エー ビー シー ディー イー エフ ジー エイチ アイ ジェー ケー エル エム エヌ オー ピー キュー アール エス ティー ユー ブイ ダブリュー エックス ワイ ゼット",
        "ru": "а бэ цэ дэ е эф гэ аш и йот ка эль эм эн о пэ ку эр эс тэ у вэ дубльвэ икс игрек зет",
        "th": "เอ บี ซี ดี อี เอฟ จี เอช ไอ เจ เค แอล เอ็ม เอ็น โอ พี คิว อาร์ เอส ที ยู วี ดับเบิลยู เอ็กซ์ วาย แซด",
    }.items():
        for letter, name in zip("abcdefghijklmnopqrstuvwxyz", names.split()):
            cfg["letters"][language].setdefault(letter, name)
    cfg["units"].setdefault("%", {
        "zh": ["百分之"], "en": ["percent"], "ja": ["パーセント"],
        "ru": ["процент", "процента", "процентов"], "de": ["Prozent"],
        "fr": ["pour cent"], "es": ["por ciento"], "th": ["ร้อยละ"]})
    cfg["units"].setdefault("GHz", {
        "zh": ["吉赫兹"], "en": ["gigahertz"], "ja": ["ギガヘルツ"],
        "ru": ["гигагерц", "гигагерца", "гигагерц"], "de": ["Gigahertz"],
        "fr": ["gigahertz"], "es": ["gigahercios"], "th": ["กิกะเฮิรตซ์"]})
    cfg["unit_symbol_aliases"].setdefault("％", "%")
    templates = cfg.setdefault("quantity_templates", {}).setdefault("%", {})
    for lang, template in (("zh", "百分之{number}"), ("th", "ร้อยละ{number}")):
        templates.setdefault(lang, template)
    return cfg


@lru_cache(maxsize=8)
def _unit_pattern(language=""):
    cfg = _config()
    symbols = (set(cfg["units"]) - {"hour", "minute", "second"}) | set(cfg["unit_symbol_aliases"])
    if language == "ru":
        symbols |= _RU_UNIT_ALIASES.keys()
    names = "|".join(re.escape(symbol) for symbol in sorted(symbols, key=len, reverse=True))
    number = r"(?:" + _EN_GROUPED_NUM + "|" + _NUM + ")" if language == "en" else _NUM
    if language in _COMMA_LANGUAGES:
        # Preserve the existing separate-number reading of comma lists.
        number = r"(?:(?<!\d,)" + _COMMA_NUM + r"|(?<=\d,)" + _NUM + r")"
    if language in {"de", "es"}:
        number = r"(?:" + _EU_GROUPED_DECIMAL + "|" + number + ")"
    if language in {"fr", "es"}:
        number = r"(?:" + _SPACE_GROUPED_NUM + "|" + number + ")"
    boundary = r"(?<!\d,)" if language in {"de", "es"} else ""
    left = r"(?<![\w.\-−])" if language == "ru" else r"(?<![A-Za-z0-9_.\-−])"
    right = r"(?!\w|[.\-]\w)" if language == "ru" else r"(?![A-Za-z0-9_]|[.\-][A-Za-z0-9_])"
    return re.compile(left + boundary + r"(?P<n>[+\-−]?" + number +
        r")\s*(?P<u>" + names + r")" + right)


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Original-text offsets for addresses, paths, filenames, code and versions.

    Protect these when matching lexical aliases; nothing is removed or replaced.
    The normalizer separately handles explicit versions as typed numeric spans.
    """
    spans = [(m.start(), m.end()) for m in _address_matches(text)]
    spans += [(m.start(), m.end()) for m in _VERSION.finditer(text)]
    return sorted(spans)


def _address_matches(text):
    hints = [m.span() for m in _PHONE_HINT.finditer(text)]
    for match in _ADDRESS.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in hints):
            continue
        # The sentence following a URL/path is not part of that literal.
        end = match.start() + len(match.group().rstrip(".,!?;，。！？；"))
        if end < match.end():
            match = _ADDRESS.match(text, match.start(), end)
            if match is None:
                continue
        # A single /x after a numeric operand may be division, not a root path.
        if re.fullmatch(r"/[A-Za-z]{1,3}", match.group()) and re.search(r"\d\s*$", text[:match.start()]):
            continue
        yield match


def _join(parts, language):
    return ("" if language in {"zh", "ja", "th"} else " ").join(parts)


def _context(text, start, end):
    """Do not borrow a semantic cue from a neighbouring sentence/clause."""
    left = re.split(r"[;；。!?？！\n]", text[max(0, start-64):start])[-1]
    right = re.split(r"[;；。!?？！\n]", text[end:end+40])[0]
    return (left + text[start:end] + right).lower()


def _ja_integer(value):
    if not 0 <= value < 10**16:
        raise ValueError("Japanese cardinal outside supported range")
    if value == 0:
        return "零"
    digits = "零一二三四五六七八九"
    groups = []
    for group_unit in ("", "万", "億", "兆"):
        group, value = value % 10000, value // 10000
        if group:
            words = ""
            for power, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
                digit, group = divmod(group, power)
                if digit:
                    words += ("" if digit == 1 and power > 1 else digits[digit]) + unit
            groups.append(words + group_unit)
    return "".join(reversed(groups))


def _ja_minutes(value):
    """Read ordinary 0–99 minute counters; spaces keep kana suffixes separate."""
    if value >= 100:
        return _ja_integer(value) + "分"
    tens, ones = divmod(value, 10)
    if value and not ones:
        reading = (_ja_integer(tens) if tens > 1 else "") + "じゅっぷん"
    else:
        endings = ("れいふん", "いっぷん", "にふん", "さんぷん", "よんぷん",
                   "ごふん", "ろっぷん", "ななふん", "はっぷん", "きゅうふん")
        reading = (_ja_integer(tens * 10) if tens else "") + endings[ones]
    return " " + reading + " "


def _integer(value, language, **kwargs):
    if language == "zh":
        from cn2an import an2cn
        return an2cn(str(value))
    if language == "ja":
        return _ja_integer(value)
    from num2words import num2words
    return num2words(value, lang=language, **kwargs)


def _digits(value, language, cfg):
    return _join([cfg["digits"][language][int(c)] for c in value], language)


def _number(value, language, cfg):
    value = str(value).replace("−", "-")
    if language in {"fr", "es"}:
        value = re.sub(r"[ \u00a0\u202f]", "", value)
    if language == "en" and re.fullmatch(r"[+\-]?" + _EN_GROUPED_NUM, value):
        value = value.replace(",", "")
    if language in {"de", "es"} and re.fullmatch(r"[+\-]?" + _EU_GROUPED_DECIMAL, value):
        value = value.replace(".", "")
    if language in _COMMA_LANGUAGES:
        value = value.replace(",", ".")
    if value[:1] in {"-", "+"}:
        key = "negative" if value[0] == "-" else "positive"
        return _join([cfg["operator_words"][language][key], _number(value[1:], language, cfg)], language)
    if "e" in value.lower():
        mantissa, exponent = re.split("[eE]", value)
        return _join([_number(mantissa, language, cfg), cfg["operator_words"][language]["times"],
                      _power(_integer(10, language), _number(str(int(exponent)), language, cfg), language, cfg)], language)
    if "." in value:
        if language == "ru":
            return _integer(value, language)
        whole, fraction = value.split(".")
        return _join([_integer(int(whole), language), cfg["operator_words"][language]["point"],
                      _digits(fraction, language, cfg)], language)
    return _integer(int(value), language)


def _power(base, exponent, language, cfg):
    pattern = cfg["operator_words"][language]["power"]
    return _join([base, pattern.replace("{n}", exponent)], language)


def _unit_form(forms, value, language):
    raw = value.replace("−", "-")
    if language == "en" and re.fullmatch(r"[+\-]?" + _EN_GROUPED_NUM, raw):
        raw = raw.replace(",", "")
    if language in {"de", "es"} and re.fullmatch(r"[+\-]?" + _EU_GROUPED_DECIMAL, raw):
        raw = raw.replace(".", "")
    number = Decimal(raw.replace(",", ".") if language in _COMMA_LANGUAGES else raw)
    if language == "ru":
        if number != int(number) or "." in value or "," in value:
            return forms[1]
        n = abs(int(number))
        return forms[0] if n % 10 == 1 and n % 100 != 11 else forms[1] if n % 10 in {2, 3, 4} and n % 100 not in {12, 13, 14} else forms[2]
    return forms[0] if abs(number) == 1 else forms[-1]


def _quantity(value, unit, language, cfg, genitive_decimal=False):
    if language == "ja" and unit == "minute" and value.isdigit():
        return _ja_minutes(int(value))
    if language in {"fr", "es"}:
        value = re.sub(r"[ \u00a0\u202f]", "", value)
    words = _number(value, language, cfg)
    if genitive_decimal:
        whole, fraction = re.split(r"[.,]", value)
        singular = lambda n: n % 10 == 1 and n % 100 != 11
        words = " ".join([
            _integer(int(whole), "ru", gender="f", case="g"),
            "целой" if singular(int(whole)) else "целых",
            _integer(int(fraction), "ru", gender="f", case="g"),
            _integer(10 ** len(fraction), "ru", to="ordinal", gender="f", case="g",
                     plural=not singular(int(fraction)))])
    if language == "de" and re.fullmatch(r"[+\-−]?1", value):
        words = re.sub(r"eins$", "eine" if unit in {"hour", "minute", "second"} else "ein", words)
    if language == "ru" and unit in {"minute", "second"} and re.fullmatch(r"\d+", value):
        words = _integer(int(value), language, gender="feminine")
    if language == "ru" and unit == "ms" and re.fullmatch(r"[+\-−]?\d+", value):
        words = _integer(int(value.replace("−", "-")), language, gender="feminine")
        if value.startswith("+"):
            words = _join([cfg["operator_words"][language]["positive"], words], language)
    if language == "fr" and unit in {"hour", "minute", "second"} and re.fullmatch(r"[+\-−]?\d+", value):
        words = re.sub(r"\bun$", "une", words)
    if language == "es" and re.fullmatch(r"[+\-−]?\d+", value):
        if unit == "hour":
            words = re.sub(r"uno$", "una", words)
        else:
            words = re.sub(r"veintiuno$", "veintiún", words)
            words = re.sub(r"uno$", "un", words)
    forms = (["милли" + form for form in cfg["units"]["second"]["ru"]]
             if language == "ru" and unit == "ms" else cfg["units"][unit][language])
    unit_words = _unit_form(forms, value, language)
    template = cfg.get("quantity_templates", {}).get(unit, {}).get(language)
    return template.format(number=words, unit=unit_words) if template else _join([words, unit_words], language)


def _date(raw, language, cfg, *, german_dative=False, russian_accusative=False):
    year, month, day = map(int, raw.split("-"))
    date(year, month, day)  # Gregorian validation, including leap years.
    if language == "zh":
        return f"{_digits(str(year), language, cfg)}年{_integer(month, language)}月{_integer(day, language)}日"
    if language == "ja":
        return f"{_integer(year, language)}年{_integer(month, language)}月{_integer(day, language)}日"
    if language == "ru":
        d = _integer(day, language, to="ordinal", gender="neuter",
                     case="accusative" if russian_accusative else "genitive")
        y = _integer(year, language, to="ordinal", case="genitive")
        return f"{d} {cfg['months'][language][month-1]} {y} года"
    d = _integer(day, language, to="ordinal") if language in {"en", "de"} else _integer(day, language)
    if language == "de":
        d += "n" if german_dative else "r"
    if language == "fr" and day == 1:
        d = "premier"
    return cfg["date_templates"][language].format(day=d, month=cfg["months"][language][month-1], year=_integer(year, language))


def _clock(raw, duration, language, cfg):
    values = list(map(int, raw.split(":")))
    if any(v > 59 for v in values[1:]) or (not duration and values[0] > 23):
        raise ValueError("invalid clock/duration fields")
    if duration:
        units = ["minute", "second"] if len(values) == 2 else ["hour", "minute", "second"]
        return _join([_quantity(str(v), u, language, cfg) for v, u in zip(values, units) if v or not any(values)], language)
    hour, minute = values[:2]
    if language == "en":
        result = _integer(hour, language)
        result += " " + (("oh " if minute < 10 else "") + _integer(minute, language) if minute else "o'clock")
    elif language in {"zh", "ja"}:
        result = _integer(hour, language) + "点" if language == "zh" else cfg["clock_hour_readings"]["ja"][hour]
        if language == "ja":
            result += _ja_minutes(minute) if minute or len(values) == 3 else "ちょうど"
        else:
            result += ("零" if 0 < minute < 10 else "") + _integer(minute, language) + "分" if minute or len(values) == 3 else "整"
    elif language == "de":
        result = ("ein" if hour == 1 else _integer(hour, language)) + " Uhr" + (" " + _integer(minute, language) if minute else "")
    elif language == "th":
        result = _integer(hour, language) + "นาฬิกา" + (_integer(minute, language) + "นาที" if minute else "")
    else:
        result = _join([_quantity(str(hour), "hour", language, cfg), _quantity(str(minute), "minute", language, cfg)], language)
    if len(values) == 3:
        result = _join([result, _quantity(str(values[2]), "second", language, cfg)], language)
    return result


class _MathParser:
    """Small expression AST with retained parentheses; no eval or Python code."""
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0

    def expression(self, minimum=0):
        if self.pos >= len(self.tokens):
            raise ValueError("missing operand")
        token = self.tokens[self.pos]
        self.pos += 1
        if token in {"+", "-", "−"}:
            node = ("unary", token, self.expression(30))
        elif token == "(":
            node = ("group", self.expression())
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ")":
                raise ValueError("unbalanced parentheses")
            self.pos += 1
        elif re.fullmatch(_NUM, token) or re.fullmatch("[A-Za-z]{1,3}|π", token):
            node = ("atom", token)
        else:
            raise ValueError("unsupported operand")
        while self.pos < len(self.tokens):
            op = self.tokens[self.pos]
            # Only an explicit numeric coefficient or pi may imply a product
            # with one variable.  Parsing the right operand above multiply
            # precedence keeps 2x² as 2 * (x²), not (2x)².
            implicit = node[0] == "atom" and (re.fullmatch(_NUM, node[1]) or node[1] in {"π", "pi"}) and re.fullmatch("[A-Za-z]|π", op)
            precedence = 20 if implicit else {"+": 10, "-": 10, "−": 10, "*": 20, "×": 20, "·": 20, "/": 20, "÷": 20, "^": 40, "²": 40, "³": 40}.get(op, -1)
            if precedence < minimum:
                break
            if implicit:
                node = ("binary", "*", node, self.expression(precedence + 1))
                continue
            self.pos += 1
            right = ("atom", "2" if op == "²" else "3") if op in {"²", "³"} else self.expression(precedence if op == "^" else precedence + 1)
            node = ("binary", "^" if op in {"²", "³"} else op, node, right)
        return node

    def parse(self):
        left = self.expression()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in _RELATIONS:
            relation = self.tokens[self.pos]
            self.pos += 1
            left = ("relation", relation, left, self.expression())
        if self.pos != len(self.tokens):
            raise ValueError("unsupported expression")
        return left


def _speak_math(node, language, cfg):
    words = cfg["operator_words"][language]
    kind = node[0]
    if kind == "atom":
        atom = node[1]
        if atom in {"π", "pi"}:
            return cfg["constant_words"]["π"][language]
        letters = cfg["letters"][language]
        return _number(atom, language, cfg) if atom[0].isdigit() else _join([letters.get(c.lower(), c) for c in atom], language)
    if kind == "group":
        return _join([words["left"], _speak_math(node[1], language, cfg), words["right"]], language)
    if kind == "unary":
        operand = _speak_math(node[2], language, cfg)
        if node[2][0] in {"binary", "relation"}:
            operand = _join([words["left"], operand, words["right"]], language)
        return _join([words["positive" if node[1] == "+" else "negative"], operand], language)
    op, left, right = node[1], _speak_math(node[2], language, cfg), _speak_math(node[3], language, cfg)
    if op == "^":
        if node[3] in {("atom", "2"), ("atom", "3")}:
            return _join([left, words["square" if node[3][1] == "2" else "cube"]], language)
        if node[3][0] == "binary":
            right = _join([words["left"], right, words["right"]], language)
        return _power(left, right, language, cfg)
    key = _RELATIONS[op] if kind == "relation" else {"+": "plus", "-": "minus", "−": "minus", "*": "times", "×": "times", "·": "times", "/": "divide", "÷": "divide"}[op]
    return _join([left, words[key], right], language)


def normalize_notation(text: str, language: str, *, lexical_spans=()) -> tuple[str, list[str]]:
    """Return spoken input plus decision notes; preserve uncertain source spans."""
    language = language.lower().split("-")[0].split("_")[0]
    notes, spans = [], []
    if language not in _LANGUAGES:
        return text, [f"notation:unsupported_language:{language}"]
    try:
        cfg = _config()
    except (OSError, KeyError, ValueError) as exc:
        return text, [f"notation:configuration_unavailable:{type(exc).__name__}"]
    number_pattern, token_pattern, formula_pattern = ((_COMMA_NUMBER, _COMMA_TOKEN, _COMMA_FORMULA_LIKE)
        if language in _COMMA_LANGUAGES else (_NUMBER, _TOKEN, _FORMULA_LIKE))
    if language == "ru":
        number_pattern = re.compile(r"(?<![\w.,])" + _COMMA_NUM + r"(?!\w|[.,]\d)")
    cues = [re.escape(cue) for cue in cfg.get("unit_identifier_cues", {}).get(language, [])]
    cues.extend(re.escape(cue) for cue in cfg.get("numeric_identifier_prefixes", {}).get(language, []))
    cues.append(_IDENTIFIER_CUES[language])
    bridge = {"zh": "是|为|為", "en": "is|are", "ja": "は|が", "de": "lautet|ist",
              "fr": "est", "es": "es", "th": "คือ|เป็น"}.get(language)
    gap = r"[ \t]*" if language in {"zh", "ja", "th"} else r"[ \t]+"
    bridge = r"(?:" + gap + r"(?:" + bridge + r"))?" if bridge else ""
    identifier_prefix = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(cues) +
                                   r")" + bridge + r"[ \t:：#=—–\"'「]*$", re.IGNORECASE)

    def overlaps(start, end):
        return any(start < b and end > a for a, b, _ in spans)

    def add(start, end, kind, render):
        if overlaps(start, end):
            return
        try:
            spoken = render()
        except (ImportError, ValueError, TypeError, NotImplementedError, OverflowError, KeyError) as exc:
            spoken = text[start:end]
            notes.append(f"notation:preserved:{kind}:{type(exc).__name__}:{text[start:end]}")
        else:
            notes.append(f"notation:{kind}:{text[start:end]}")
        spans.append((start, end, spoken))

    spans.extend((m.start(), m.end(), m.group()) for m in _PHONE_HINT.finditer(text))
    for match in _address_matches(text):
        spans.append((match.start(), match.end(), match.group()))
        notes.append(f"notation:protected_literal:{match.group()}")
    for start, end in lexical_spans:
        add(start, end, "dictionary_reading", lambda start=start, end=end: text[start:end])
    for abbreviation, reading in cfg.get("native_abbreviations", {}).get(language, {}).items():
        for match in re.finditer(r"(?<!\w)" + re.escape(abbreviation) + r"(?!\w)", text):
            add(match.start(), match.end(), "native_abbreviation", lambda value=reading: value)
    code_prefixes = cfg.get("technical_code_prefixes", {}).get(language, [])
    if code_prefixes:
        prefix = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, code_prefixes)) +
                            r")" + bridge + r"[ \t:：=]*$", re.I)
        for match in _NUMBER.finditer(text):
            if re.fullmatch(r"\d{3}", match.group()) and prefix.search(text[max(0, match.start()-64):match.start()]):
                add(match.start(), match.end(), "technical_code",
                    lambda value=match.group(): _digits(value, language, cfg))
    count_nouns = cfg.get("integer_count_nouns", {}).get(language, [])
    if count_nouns:
        following = re.compile(r"[ \t]+(?:" + "|".join(map(re.escape, count_nouns)) + r")(?!\w)", re.I)
        for match in _INTEGER_COUNT.finditer(text):
            if (following.match(text[match.end():]) and
                    not identifier_prefix.search(text[max(0, match.start()-64):match.start()])):
                def count(m=match):
                    value = re.sub(r"[.,]", "", m["n"])
                    spoken = _number(value, language, cfg)
                    if language == "es" and abs(int(value)) >= 10**6 and int(value) % 10**6 == 0:
                        spoken += " de"
                    return spoken
                add(match.start(), match.end(), "integer_count", count)
    # Keep the programming-language name available for the pronunciation
    # dictionary; its plus signs are not mathematical operators.
    for match in re.finditer(r"(?<![A-Za-z0-9_])C\+\+(?![A-Za-z0-9_+])", text):
        add(match.start(), match.end(), "programming_language", lambda m=match: m.group())
    if language == "en":
        for match in _EN_MONEY.finditer(text):
            def money(m=match):
                if m["left"] and m["right"]:
                    raise ValueError("ambiguous currency sign")
                sign = m["left"] or m["right"]
                word = "negative" if sign in {"-", "−"} else "positive"
                prefix = cfg["operator_words"]["en"][word] + " " if sign else ""
                if m["c"] == "£":
                    amount = Decimal(m["n"].replace(",", ""))
                    if amount * 100 != (amount * 100).to_integral_value():
                        return prefix + _number(m["n"], "en", cfg) + " pounds sterling"
                    words = _integer(amount, "en", to="currency", currency="GBP")
                    if amount == int(amount):
                        words = words.removesuffix(", zero pence")
                    elif amount < 1:
                        words = words.removeprefix("zero pounds sterling, ")
                    return prefix + words
                return prefix + m["c"] + m["n"]
            add(match.start(), match.end(), "english_currency", money)
        for match in _EN_HARDWARE.finditer(text):
            def hardware(m=match):
                suffix = "T I" if (m["suffix"] or "").lower() == "ti" else (m["suffix"] or "")
                return (m["family"].upper() + m["n"] + " " + suffix).strip()
            add(match.start(), match.end(), "english_hardware_model", hardware)
    elif language == "zh":
        for match in _EN_HARDWARE.finditer(text):
            def hardware(m=match):
                suffix = "T I" if (m["suffix"] or "").lower() == "ti" else (m["suffix"] or "")
                return (m["family"].upper() + " " + _digits(m["n"], "zh", cfg) + " " + suffix).strip()
            add(match.start(), match.end(), "chinese_hardware_model", hardware)
    code_cue = _IDENTIFIER_CUES[language] + (r"|code" if language == "en" else "")
    code_prefix = re.compile(r"(?<![A-Za-z0-9_])(?:" + code_cue + r")" + bridge +
                             r"[ \t:：#=—–\"'「]*$", re.I)
    for match in _CODE.finditer(text):
        if not code_prefix.search(text[max(0, match.start()-64):match.start()]):
            continue
        def code(m=match):
            pieces = []
            for char in m.group():
                if char.isdigit():
                    pieces.append(cfg["digits"][language][int(char)])
                elif char.isalpha():
                    pieces.append(cfg["letters"][language].get(char.lower(), char))
                else:
                    pieces.append(cfg["literal_symbol_words"][char][language])
            return _join(pieces, language)
        add(match.start(), match.end(), "verification_code", code)
    for match in _VERSION.finditer(text):
        value = match.group("v1") or match.group("v2")
        start = match.start("v1") if match.group("v1") else match.start("v2")
        def version(value=value):
            parts = value.lstrip("vV").split(".")
            def component(part):
                return _digits(part, language, cfg) if len(part) >= 4 or (len(part) > 1 and part.startswith("0")) else _number(part, language, cfg)
            pieces = [component(parts[0])]
            for part in parts[1:]:
                pieces.extend([cfg["operator_words"][language]["dot"], component(part)])
            spoken = _join(pieces, language)
            if value[:1].lower() == "v":
                prefix = cfg["version_prefix_words"][language]["lower" if value[0] == "v" else "upper"]
                return _join([prefix, spoken], language)
            return spoken
        add(start, start + len(value), "version", version)
    for match in _DATE.finditer(text):
        local = _context(text, match.start(), match.end())
        if any(cue in local for cue in ("พ.ศ.", "佛历", "佛曆", "buddhist")):
            add(match.start(), match.end(), "calendar_requires_context", lambda m=match: m.group())
        else:
            dative = language == "de" and bool(re.search(r"\bam\s*$", text[:match.start()], re.I))
            accusative = language == "ru" and bool(re.search(r"\bна[ \t]+$", text[:match.start()], re.I))
            label = re.search(r"วันที่[ \t]*$", text[:match.start()]) if language == "th" else None
            add(label.start() if label else match.start(), match.end(), "iso_date",
                lambda m=match, d=dative, a=accusative: _date(
                    m.group(), language, cfg, german_dative=d, russian_accusative=a))
    if language == "de":
        for match in re.finditer(r"\bam\s+(?P<day>\d{1,2})\.(?P<month>\d{1,2})\."
                                 r"(?P<year>\d{4})(?![A-Za-z0-9_]|\.\d)", text, re.I):
            raw = f"{match['year']}-{match['month']}-{match['day']}"
            add(match.start("day"), match.end("year"), "german_date",
                lambda value=raw: _date(value, language, cfg, german_dative=True))
    if language == "en":
        months = "|".join(re.escape(month) for month in cfg["months"]["en"])
        for match in re.finditer(r"\b(?P<month>" + months + r")\s+(?P<day>\d{1,2})"
                                 r"(?:st|nd|rd|th)?(?![A-Za-z0-9_.])", text, re.I):
            year = re.match(r"(?:[ \t]*,[ \t]*|[ \t]+)(\d{4})(?!\d)", text[match.end():])
            date_cue = re.search(r"\b(?:on|dated|date(?: is)?)\s*$", text[:match.start()], re.I)
            if not year and not date_cue:
                continue
            def named_date(m=match, y=int(year[1]) if year else 2000):
                month = [name.lower() for name in cfg["months"]["en"]].index(m["month"].lower()) + 1
                date(y, month, int(m["day"]))
                return m["month"] + " " + _integer(int(m["day"]), "en", to="ordinal")
            add(match.start(), match.end(), "english_named_date", named_date)
    for match in _DATE_LIKE.finditer(text):
        add(match.start(), match.end(), "ambiguous_date_format", lambda m=match: m.group())
    for match in _IPV4.finditer(text):
        prefix = text[max(0, match.start()-48):match.start()]
        valid = all(int(part) <= 255 for part in match.group().split("."))
        cue = re.search(_IPV4_CUES[language] + r"[ \t:=：]*$", prefix, re.I)
        if valid and cue:
            add(match.start(), match.end(), "ipv4",
                lambda m=match: (" " + cfg["operator_words"][language]["dot"] + " ").join(
                    _number(part, language, cfg)
                    if cfg.get("ipv4_octet_reading", {}).get(language) == "cardinal"
                    and not (len(part) > 1 and part.startswith("0"))
                    else _digits(part, language, cfg) for part in m.group().split(".")))
    if language in {"de", "es"}:
        for match in _unit_pattern(language).finditer(text):
            prefix = text[max(0, match.start()-64):match.start()]
            if (not re.fullmatch(r"[+\-−]?" + _EU_GROUPED_DECIMAL, match["n"])
                    or match["u"] in {"KJ", "KM"} or identifier_prefix.search(prefix)
                    or re.match(r"[/^²³]", text[match.end():])):
                continue
            unit = cfg["unit_symbol_aliases"].get(match["u"], match["u"])
            add(match.start(), match.end(), "quantity",
                lambda m=match, u=unit: _quantity(m["n"], u, language, cfg))
    for match in _DOTTED.finditer(text):
        if not overlaps(match.start(), match.end()):
            add(match.start(), match.end(), "ambiguous_dotted_identifier", lambda m=match: m.group())
    for match in _CLOCK.finditer(text):
        context = _context(text, match.start(), match.end())
        duration = any(re.search(r"(?<!\w)"+re.escape(cue)+r"(?!\w)", context) if cue.isascii() else cue in context for cue in cfg["duration_cues"][language])
        if not duration and any(re.search(r"(?<!\w)" + re.escape(cue) +
                                         (r"s?" if language == "en" else "") + r"(?!\w)", context)
                                if cue.isascii() else cue in context
                                for cue in cfg["ambiguous_clock_cues"][language]):
            add(match.start(), match.end(), "ambiguous_clock_or_offset", lambda m=match: m.group())
            continue
        end = match.end()
        if language == "th" and not duration:
            suffix = re.match(r"[ \t]*นาฬิกา", text[end:])
            if suffix:
                end += suffix.end()
        elif language == "de" and not duration:
            suffix = re.match(r"[ \t]*Uhr\b", text[end:], re.I)
            if suffix:
                end += suffix.end()
        add(match.start(), end, "duration" if duration else "clock", lambda m=match, d=duration: _clock(m.group(), d, language, cfg))
    if language == "fr":
        for match in _FR_CLOCK.finditer(text):
            prefix = re.split(r"[.!?;\n]", text[max(0, match.start()-64):match.start()])[-1]
            if any(re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", prefix, re.I)
                   for cue in cfg["duration_cues"]["fr"]):
                continue
            if re.search(r"(?<!\w)(?:à|de|entre|vers|dès|jusqu['’]à|horaire(?: est)?|heure(?: est)?)[ \t:]*$", prefix, re.I):
                add(match.start(), match.end(), "french_clock",
                    lambda m=match: _clock(m["h"] + ":" + m["m"], False, "fr", cfg))
    if language == "ja":
        for match in _JA_MINUTES.finditer(text):
            if text[:match.start()].rstrip(" \t").endswith(("+", "-", "−", "＋", "－")):
                continue
            add(match.start(), match.end(), "japanese_minutes",
                lambda m=match: _ja_minutes(int(m["n"])))
    formula_ranges = [(match.start(), match.end()) for match in formula_pattern.finditer(text)]
    if language == "fr":
        for match in _FR_METRE.finditer(text):
            prefix = re.split(r"[;!?\n]|(?<!\d)\.|\.(?!\d)", text[max(0, match.start()-64):match.start()])[-1]
            if (identifier_prefix.search(prefix) or
                    any(start < match.end() and match.start() < end for start, end in formula_ranges) or
                    not re.search(r"(?<!\w)(?:mesure(?:nt)?|longueur|largeur|hauteur|distance|passage|câble)(?!\w)", prefix, re.I)):
                continue
            add(match.start(), match.end(), "french_metres",
                lambda m=match: _join([_number(m["n"], "fr", cfg),
                    "mètre" if abs(Decimal(m["n"].replace("−", "-").replace(",", "."))) < 2
                    else "mètres"], "fr"))
    for match in _unit_pattern(language).finditer(text):
        unit = cfg["unit_symbol_aliases"].get(match["u"], match["u"])
        if language == "ru":
            unit = _RU_UNIT_ALIASES.get(unit, unit)
        # Keep single-letter operands in algebra on the existing math path.
        if len(unit) == 1 and any(start <= match.start() and match.end() <= end for start, end in formula_ranges):
            continue
        compound = re.match(r"[/^²³][A-Za-z0-9µμ°℃²³^/().+\-−]*", text[match.end():])
        if compound:
            rate = re.match(r"/s(?![A-Za-z0-9_/^²³()+\-−×*÷=<>≤≥≠≈]|\.[A-Za-z0-9_])", text[match.end():])
            if (language == "en" and unit in {"Gb", "GB"} and rate and
                    not identifier_prefix.search(text[max(0, match.start()-64):match.start()]) and
                    not text[:match.start()].rstrip(" \t").endswith(tuple("=<>≤≥≠≈+-−×*÷^²³/")) and
                    not text[match.end()+rate.end():].lstrip(" \t").startswith(tuple("=<>≤≥≠≈+-−×*÷^²³/"))):
                end = match.end() + rate.end()
                spans[:] = [s for s in spans if not (match.start() <= s[0] and s[1] <= end and s[2] == text[s[0]:s[1]])]
                add(match.start(), end, "data_rate", lambda m=match, u=unit:
                    _quantity(m["n"], u, "en", cfg) + " per " + cfg["units"]["second"]["en"][0])
                continue
            end = match.end() + compound.end()
            # A suffix such as /s may already be protected as a path. Merge
            # only unchanged literals into this larger literal span.
            spans[:] = [s for s in spans if not (match.start() <= s[0] and s[1] <= end and s[2] == text[s[0]:s[1]])]
            add(match.start(), end, "unsupported_compound_unit", lambda m=match, e=end: text[m.start():e])
            continue
        prefix = text[max(0, match.start()-48):match.start()]
        if ((language in {"de", "es"} and re.fullmatch(r"[+\-−]?" + _EU_GROUPED_DECIMAL, match["n"])
             and identifier_prefix.search(prefix)) or
                any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(cue) + r"[ \t:：#=\"'「]*$", prefix, re.I)
                    for cue in cfg.get("unit_identifier_cues", {}).get(language, []))):
            add(match.start(), match.end(), "quantity_or_identifier_requires_context", lambda m=match: m.group())
            continue
        context = _context(text, match.start(), match.end())
        if match["u"] in {"KJ", "KM"}:
            cues = cfg["unit_alias_context"][unit][language]
            if not any(cue in context for cue in cues):
                add(match.start(), match.end(), "ambiguous_nonstandard_unit", lambda m=match: m.group())
                continue
        genitive_decimal = (language == "ru" and unit == "%"
                            and re.fullmatch(r"\d+[.,]\d+", match["n"])
                            and (re.search(r"(?<!\w)до[ \t]+$", prefix, re.I)
                                 or (re.search(r"(?<!\w)с[ \t]+$", prefix, re.I)
                                     and re.search(r"(?<!\w)до[ \t]+[+\-−]?" +
                                                   _COMMA_NUM + r"[ \t]*[%％]",
                                                   re.split(r"[;；。!?？！\n]|(?<!\d)\.|\.(?!\d)",
                                                            text[match.end():], 1)[0], re.I))))
        add(match.start(), match.end(), "quantity",
            lambda m=match, u=unit, g=bool(genitive_decimal): _quantity(m["n"], u, language, cfg, g))

    if language == "en":
        for match in _EN_GROUPED.finditer(text):
            prefix = text[max(0, match.start()-64):match.start()]
            add(match.start(), match.end(), "english_grouped_number",
                lambda m=match, p=prefix: _digits(m["n"].replace(",", ""), "en", cfg) if m["n"][0].isdigit() and identifier_prefix.search(p) else _number(m["n"], "en", cfg))

    if language in {"fr", "es"}:
        for match in _SPACE_GROUPED.finditer(text):
            if any(start < match.end() and match.start() < end for start, end in formula_ranges):
                continue
            value = re.sub(r"[ \u00a0\u202f]", "", match["n"])
            prefix = text[max(0, match.start()-64):match.start()]
            add(match.start(), match.end(), "space_grouped_number",
                lambda v=value, p=prefix: _digits(v, language, cfg)
                if v.isdigit() and identifier_prefix.search(p) else _number(v, language, cfg))

    for match in _AMBIGUOUS_PAIR.finditer(text):
        context = _context(text, match.start(), match.end())
        if not any(cue in context for cue in cfg["math_cues"][language]) and not any(op in context for op in "=<>≤≥≠≈"):
            add(match.start(), match.end(), "ambiguous_fraction_or_identifier", lambda m=match: m.group())

    # Claim a complete equation/power before looking for smaller expressions.
    # Otherwise an unsupported lhs/body could leave a perfectly parseable tail
    # (e.g. -3 or r²) that would silently change only part of the source formula.
    for match in formula_pattern.finditer(text):
        start, end = match.start(), match.end()
        while end > start and text[end-1] in ("., \t" if language in _COMMA_LANGUAGES else ". \t"):
            end -= 1
        if language == "en" and text[:start].rstrip(" \t").endswith(("^", "²", "³", "(")):
            continue
        if overlaps(start, end):
            continue
        raw = text[start:end]
        leadin = re.match(r"([A-Za-z]+)[ \t]+(?=[+\-−(0-9])", raw) if language == "en" else None
        if leadin and (len(leadin[1]) > 3 or leadin[1].lower() in _EN_MATH_COPULAS):
            # Parse the following expression, not a verb such as "solve" or "is".
            start += leadin.end()
            expression = formula_pattern.match(text, start)
            if expression is None:
                continue
            end = expression.end()
            while end > start and text[end-1] in ". \t":
                end -= 1
            if overlaps(start, end):
                continue
            raw = text[start:end]
        mathematical = any(op in raw for op in "=<>≤≥≠≈^²³×·*/÷") or any(
            cue in _context(text, start, end) for cue in cfg["math_cues"][language])
        parsed = raw.replace("‑", "-") if mathematical else raw
        matches = list(token_pattern.finditer(parsed))
        values = [m.group().replace(",", ".") if language in _COMMA_LANGUAGES else m.group() for m in matches]
        covered = token_pattern.sub("", parsed).strip()
        strong = any(v in _RELATIONS or v in {"^", "²", "³", "×", "·", "*", "÷"} for v in values)
        strong |= bool(re.fullmatch(r"[A-Za-z][ \t]*\+[ \t]*[A-Za-z]", parsed))
        strong |= any(v.isdigit() or (language in _COMMA_LANGUAGES and re.fullmatch(_NUM, v)) for v in values) and any(v in {"+", "-", "−"} for v in values)
        strong |= any(cue in _context(text, start, end) for cue in cfg["math_cues"][language])
        try:
            if covered or not strong:
                raise ValueError("unclassified expression")
            node = _MathParser(values).parse()
        except ValueError:
            # Read an explicit plus without claiming the unknown expression parsed.
            add(start, end, "unsupported_formula", lambda raw=raw: raw.replace(
                "+", " " + cfg["operator_words"][language]["plus"] + " "))
        else:
            add(start, end, "math", lambda node=node: _speak_math(node, language, cfg))

    # Search only bounded token runs.  Only explicit numeric/pi coefficients
    # may imply multiplication; group nodes retain the source formula scope.
    tokens = list(token_pattern.finditer(text))
    for first, token in enumerate(tokens):
        if overlaps(token.start(), token.end()):
            continue
        if (language == "en" and token.group().lower() in _EN_MATH_COPULAS and
                re.match(r"[ \t]+[+\-−(0-9]", text[token.end():])):
            continue
        best = None
        for last in range(first + 1, min(first + 40, len(tokens))):
            end_token = tokens[last]
            if text[tokens[last-1].end():end_token.start()].strip() or overlaps(end_token.start(), end_token.end()):
                break
            values = [t.group().replace(",", ".") if language in _COMMA_LANGUAGES else t.group() for t in tokens[first:last+1]]
            if not any(v in _RELATIONS or v in {"^", "²", "³", "+", "-", "−", "×", "·", "*", "÷", "/"} for v in values):
                continue
            # A bare a-b or 3/4 is also a label/date/fraction: require context.
            strong = any(v in _RELATIONS or v in {"^", "²", "³", "×", "·", "*", "÷"} for v in values)
            strong |= any(v.isdigit() or (language in _COMMA_LANGUAGES and re.fullmatch(_NUM, v)) for v in values) and any(v in {"+", "-", "−"} for v in values)
            strong |= any(cue in _context(text, token.start(), end_token.end()) for cue in cfg["math_cues"][language])
            if not strong:
                continue
            try:
                node = _MathParser(values).parse()
            except ValueError:
                continue
            # A final period is punctuation; .digits/.name still continues an
            # identifier.  Never accept a prefix before a remaining exponent,
            # bracket or operator (in particular the c before c² at sentence end).
            start, end = token.start(), end_token.end()
            if (start and re.match(r"[A-Za-z0-9_.(]", text[start-1])) or re.match(r"[A-Za-z0-9_)]|\.[A-Za-z0-9_]|[ \t]*[+\-−‑×·*/÷^²³=<>≤≥≠≈]", text[end:]):
                continue
            best = (end, node)
        if best:
            end, node = best
            add(token.start(), end, "math", lambda node=node: _speak_math(node, language, cfg))
    if language == "en":
        for match in re.finditer(r"(?<![A-Za-z0-9_./+\-−])(?P<sign>[+\-−])[ \t]*(?P<n>" +
                                 _NUM + r")(?![A-Za-z0-9_]|\.[A-Za-z0-9_])", text):
            if identifier_prefix.search(text[max(0, match.start()-64):match.start()]):
                continue
            add(match.start(), match.end(), "signed_number",
                lambda m=match: _number(m["sign"] + m["n"], "en", cfg))
    numbers = list(number_pattern.finditer(text))
    if language in _COMMA_LANGUAGES:
        # Two commas identify an ambiguous list, not a single decimal. Retain
        # the existing independent-number readings, including leading-zero rules.
        lists = [match.span() for match in re.finditer(r"\d+(?:,\d+){2,}", text)]
        numbers.extend(match for match in _NUMBER.finditer(text)
                       if any(start <= match.start() and match.end() <= end for start, end in lists))
        numbers.sort(key=lambda match: match.start())
    for match in numbers:
        prefix = text[max(0, match.start()-64):match.start()]
        phone = language == "zh" and re.fullmatch(r"\d{7,15}", match.group()) and re.search(
            r"(?:联系电话|电话号码|手机号码|手机号|电话)(?:为|是)?[ \t:：]*$", prefix)
        if language not in _COMMA_LANGUAGES and re.search(r"\d,$", prefix) and text[match.end():].startswith("."):
            # Preserve an unsupported comma-number fragment rather than just its tail.
            add(match.start(), match.end(), "ambiguous_comma_fragment", lambda m=match: m.group())
        elif match.group().isdigit() and (phone or identifier_prefix.search(prefix)):
            add(match.start(), match.end(), "identifier_digits", lambda m=match: _digits(m.group(), language, cfg))
        elif (language == "de" and re.fullmatch(r"\d+\.\d+", match.group())
              and re.search(r"\b(?:Raum|Zimmer)[ \t]*$", prefix, re.I)):
            def room_number(m=match):
                whole, fraction = m.group().split(".")
                floor = _digits(whole, language, cfg) if whole.startswith("0") else _number(whole, language, cfg)
                return " ".join([floor, cfg["literal_symbol_words"]["."][language], _digits(fraction, language, cfg)])
            add(match.start(), match.end(), "room_number", room_number)
        elif (language == "zh" and re.fullmatch(r"[1-9]\d{3}", match.group())
              and re.match(r"[ \t]*年", text[match.end():])
              and not re.search(r"(?:[+\-−=<>≤≥≠≈×*/^]|负|正)[ \t]*$", prefix)
              and (re.match(r"[ \t]*年[ \t]*\d{1,2}月", text[match.end():])
                   or (not re.search(r"(?:每|隔|持续|运行|历时|历经|经过|累计|长达|为期|"
                                     r"周期|寿命|保修|服役|使用|已有)(?:了|为)?[ \t]*(?:约|近)?[ \t]*$", prefix)
                       and not re.match(r"[ \t]*年(?:期|率|整|历史|之久|左右|以上|以内|一遇|[/^²³])",
                                        text[match.end():])))):
            # Calendar years use digit readings; elapsed years remain quantities.
            add(match.start(), match.end(), "calendar_year",
                lambda m=match: _digits(m.group(), language, cfg))
        elif len(match.group()) > 1 and match.group().startswith("0") and match.group()[1].isdigit():
            if language == "en" and re.search(r"\bcode[ \t:#=]*$", prefix, re.I):
                add(match.start(), match.end(), "identifier_digits",
                    lambda m=match: _digits(m.group(), language, cfg))
            else:
                add(match.start(), match.end(), "ambiguous_leading_zero_number", lambda m=match: m.group())
        else:
            def cardinal(m=match):
                if language == "ru" and m.group().isdigit() and re.search(r"%[ \t]+от[ \t]+$", text[:m.start()]):
                    return _integer(int(m.group()), language, case="g")
                return _number(m.group(), language, cfg)
            add(match.start(), match.end(), "number", cardinal)
    # These operators can occur outside an equation, e.g. AI Agent＋平台 or
    # ±0.5℃. Protected addresses and already-rendered expressions stay intact.
    for match in re.finditer(r"[＋±×÷−]|(?<![A-Za-z0-9_])\+(?![A-Za-z0-9_])", text):
        key = {"±": "pm", "×": "times", "÷": "divide", "−": "minus"}.get(match.group(), "plus")
        add(match.start(), match.end(), "operator",
            lambda key=key: " " + cfg["operator_words"][language][key] + " ")
    result, cursor = [], 0
    for start, end, spoken in sorted(spans):
        result.extend((text[cursor:start], spoken))
        cursor = end
    result.append(text[cursor:])
    return "".join(result), notes


LANGUAGES = ("zh", "en", "ja", "ru", "de", "fr", "es", "th")
_VOWEL_ONSETS = frozenset("AIOQWYaeiouyæɑɒɔəɛɜɪʊʌɐøœ")
_LATIN_LETTERS = r"A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff"
_LATIN_WORD = (rf"[{_LATIN_LETTERS}][{_LATIN_LETTERS}0-9]*"
               rf"(?:[-'&][{_LATIN_LETTERS}0-9]+)*")
_LATIN = re.compile(r"(" + _LATIN_WORD + r"(?:[ \t]+(?:&[ \t]+)?" + _LATIN_WORD + r")*)")
_SYMBOL_MAP = {"ru": {"ɭ": "l", "ɵ": "o", "ʑ": "zʲ"}, "de": {"ʏ": "y"}}
_CLOSERS = r'''[)\]}'"’”」』】》]*'''
_STOPS = re.compile(r'(?:[!?…。！？;；:：,，、—]+|\.(?![\w.]))' + _CLOSERS + r'\s*')
_SENTENCE_STOPS = re.compile(r'(?:[!?…。！？]+|\.(?![\w.]))' + _CLOSERS + r'\s*')
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "no.", "fig.", "inc.", "ltd.", "dept.", "a.m.", "p.m.",
    "z.b.", "d.h.", "u.a.", "bzw.", "ca.", "nr.", "ggf.", "vgl.", "usw.",
    "m.", "mme.", "mlle.", "pr.", "p.ex.", "env.",
    "sra.", "srta.", "dra.", "ud.", "uds.", "p.ej.", "núm.", "aprox.",
    "г.", "гг.", "ул.", "д.", "кв.", "стр.", "рис.", "см.", "т.", "т.д.",
    "т.п.", "т.е.", "им.", "руб.",
}


def first_sentence_parts(text, protected=(), *, clause=False):
    """Return a natural opening boundary and its intact tail, without repacking."""
    opens = {"(": ")", "[": "]", "{": "}", "（": "）", "【": "】",
             "《": "》", "「": "」", "『": "』", "“": "”"}
    stack = []
    for index, char in enumerate(text):
        if stack and char == stack[-1]:
            stack.pop()
            continue
        if char in opens:
            stack.append(opens[char])
            continue
        clause_stop = clause and char in ",，、;；:："
        if stack or (char not in "。！？!?." and not clause_stop) or any(a <= index < b for a, b in protected):
            continue
        if clause_stop:
            head = text[:index]
            native_letters = re.findall(r"[\u3040-\u30ff\u4e00-\u9fff\u0e00-\u0e7f]", head)
            words = re.findall(r"[^\W\d_][\w'’'-]*", head)
            # Keep a greeting such as "Hello," attached to its following clause.
            if len(native_letters) < 4 and len(words) < 4:
                continue
            # The comma before an English calendar year belongs to the date.
            if char == "," and re.match(r"\s*(?:\d{4}\b|two thousand\b|nineteen\b|twenty\b)", text[index+1:]):
                continue
        left, right = text[index - 1:index], text[index + 1:index + 2]
        if char == ".":
            if left == "." or right == "." or (left.isalnum() and right.isalnum()):
                continue
            match = re.search(r"[^\W\d_][\w.]*\.$", text[:index + 1])
            word = match[0] if match else ""
            if (word.lower() in _ABBREVIATIONS
                    or re.fullmatch(r"(?:[^\W\d_]\.){2,}", word)
                    or len(word) == 2 and word[0].isupper()):
                continue
        elif char == "!" and right == "=":
            continue
        end = index + 1
        while end < len(text) and text[end] in "。！？!?\"'”’》」』)]}）】":
            end += 1
        if end > index + 1 and end < len(text) and text[end].isalnum():
            continue
        if char == "." and end < len(text) and not text[end].isspace():
            continue
        if text[:end].strip() and text[end:].strip():
            return [text[:end], text[end:]]
        break
    return [text]


def phoneme_chunks(phones, limit=220, preferred=180, first_limit=0):
    """Split only at whole-word/punctuation boundaries, retaining every phone.

    ``first_limit`` is a soft latency target: a nearby whole word takes
    precedence over emitting a tiny opening fragment. No phone is truncated
    to make a TensorRT profile fit.
    """
    if not 1 <= preferred <= limit <= 510 or not 0 <= first_limit <= limit:
        raise ValueError("Invalid Kokoro Multi phoneme chunk limits")
    remaining, chunks = phones.strip(), []
    if not remaining:
        return chunks
    while remaining:
        first = not chunks and first_limit > 0
        if len(remaining) <= (first_limit if first else limit):
            chunks.append(remaining)
            break
        stops = [m.end() for m in _STOPS.finditer(remaining) if m.end() <= limit]
        words = [m.end() for m in re.finditer(r"\s+", remaining[:limit])]
        # A leading quote or parenthesis is not a speakable first fragment.
        boundaries = sorted({end for end in stops + words
                             if any(c.isalpha() for c in remaining[:end])})
        if first:
            early = [end for end in boundaries if end <= first_limit]
            phrase = [end for end in stops if first_limit // 2 <= end <= first_limit]
            substantial = [end for end in boundaries if end >= first_limit // 2]
            choices = phrase or [end for end in early if end >= first_limit // 2] or substantial or boundaries
            end = min(choices, key=lambda value: abs(value - first_limit)) if choices else None
        else:
            natural = [end for end in stops if preferred // 2 <= end <= limit]
            choices = natural or boundaries
            end = min(choices, key=lambda value: abs(value - preferred)) if choices else None
        if end is None:
            raise ValueError(f"No whole-word Kokoro Multi boundary within {limit} phonemes")
        chunks.append(remaining[:end].strip())
        remaining = remaining[end:].lstrip()
    if "".join("".join(chunk.split()) for chunk in chunks) != "".join(phones.split()):
        raise RuntimeError("Kokoro Multi chunking changed phoneme content")
    return chunks


def sentence_parts(phones):
    ends = [0, *[m.end() for m in _SENTENCE_STOPS.finditer(phones)], len(phones)]
    return [phones[start:end].strip() for start, end in zip(ends, ends[1:])
            if phones[start:end].strip()]


def sentence_phoneme_chunks(phones, *, pack_tail=False):
    """Pack sentences within the limit, optionally carrying a long sentence's tail."""
    chunks, pending = [], ""
    for sentence in sentence_parts(phones):
        if len(sentence) > 220:
            if pending:
                chunks.append(pending)
                pending = ""
            parts = phoneme_chunks(sentence)
            chunks.extend(parts[:-1] if pack_tail else parts)
            if pack_tail:
                pending = parts[-1]
        elif pending and len(pending) + len(sentence) + 1 <= 220:
            pending += " " + sentence
        else:
            if pending:
                chunks.append(pending)
            pending = sentence
    if pending:
        chunks.append(pending)
    return chunks


class ReadingSession:
    """One node's language state; update only after complete PCM delivery."""

    def __init__(self):
        self.previous_language = None
        self.pending_language = None

    def resolve(self, frontend, text, preference):
        preference = str(preference or "auto").lower().replace("_", "-").split("-")[0]
        if preference not in ("auto", *LANGUAGES):
            raise ValueError(f"Unsupported reading language: {preference}")
        evidence = (frontend.language_evidence(text) if preference == "auto"
                    else preference)
        self.pending_language = evidence
        return evidence or self.previous_language or "zh"

    def commit(self):
        if self.pending_language is not None:
            self.previous_language = self.pending_language
        self.pending_language = None


def _hint_tokens(tokenize, subtokenize, text, words, features):
    """Bind English hints by source characters, preserving unhinted suffixes.

    Misaki's flattened token alignment can shift later hints when, for example,
    the hinted `Kokoro` is part of the single spaCy token `Kokoro-82M`.
    """
    tokens = tokenize(text, words, {})
    if not features:
        return tokens
    spans, cursor = [], 0
    for index, word in enumerate(words):
        start = text.index(word, cursor)
        cursor = start + len(word)
        if index in features:
            spans.append((start, cursor, features[index]))
    boundaries = {edge for start, end, _ in spans for edge in (start, end)}
    result, offsets, cursor = [], [], 0
    for token in tokens:
        start = text.index(token.text, cursor)
        cursor = start + len(token.text)
        cuts = [start, *sorted(edge for edge in boundaries if start < edge < cursor), cursor]
        if len(cuts) > 2:
            position = start
            for piece in subtokenize(token.text):
                position += len(piece)
                cuts.append(position)
            cuts = sorted(set(cuts))
        for left, right in zip(cuts, cuts[1:]):
            result.append(replace(token, text=text[left:right],
                whitespace=token.whitespace if right == cursor else "",
                _=type(token._)(token._)))
            offsets.append((left, right))
    for start, end, value in spans:
        head = True
        for token, (left, right) in zip(result, offsets):
            if start <= left and right <= end:
                if isinstance(value, str) and value.startswith("/"):
                    token._.is_head = head
                    token.phonemes = value.lstrip("/") if head else ""
                    token._.rating = 5
                    head = False
                elif isinstance(value, str) and value.startswith("#"):
                    token._.num_flags = value.lstrip("#")
                else:
                    token._.stress = value
    return result


class Frontend:
    """Load dictionaries once; expose stateless per-utterance language evidence."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        vocab_path = model_dir / "vocab.json"
        vocab = json.loads(vocab_path.read_text()) if vocab_path.is_file() else {}
        self._native_symbols = all(vocab.get(symbol) == token for symbol, token in
                                   {"ʏ": 7, "ɭ": 8, "ɵ": 26, "ʑ": 27, "↑": 170}.items())
        # The release vendors only FastThai's pure Python code and offline data.
        # Compiled ARM dependencies come from the image's pinned requirements.
        extra = model_dir / "frontend_data"
        if extra.is_dir():
            sys.path.insert(0, str(extra))
        cache = Path(tempfile.gettempdir()) / "kokoro-multi-frontend"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PYTHAINLP_DATA", str(cache / "pythainlp"))
        os.environ.setdefault("PYTHAINLP_OFFLINE", "1")
        import jieba
        jieba.dt.cache_file = str(cache / "jieba.cache")
        jieba.initialize()
        from langid.langid import LanguageIdentifier, model
        from misaki.cutlet import Cutlet
        from misaki.en import G2P as EnglishG2P, subtokenize
        from misaki import espeak
        from misaki.tone_sandhi import ToneSandhi
        from fastthaig2p import G2P as ThaiG2P
        from pythainlp.tokenize import Tokenizer
        from phonemizer.backend import EspeakBackend

        class EnglishWithHints(EnglishG2P):
            def tokenize(self, text, tokens, features):
                return _hint_tokens(super().tokenize, subtokenize, text, tokens, features)

        if importlib.util.find_spec("en_core_web_sm") is None:
            raise RuntimeError("Kokoro Multi requires the pinned offline en_core_web_sm model")
        self._lock = threading.RLock()
        self._identifier = LanguageIdentifier.from_modelstring(model, norm_probs=False)
        self._identifier.set_languages(LANGUAGES)
        self._lexicon_path = model_dir / "frontend_lexicon.json"
        lexicon = json.loads(self._lexicon_path.read_text(encoding="utf-8"))
        cfg = lexicon["normalization"]
        self._chinese_readings = lexicon.get("chinese_readings", {})
        self._unstressed_linkers = {language: frozenset(words)
                                   for language, words in lexicon.get("unstressed_linkers", {}).items()}
        self._mandatory_liaisons = lexicon.get("mandatory_liaisons", {})
        self._literal_digits = cfg["digits"]["en"]
        symbols = {"_": "underscore", ".": "dot", "-": "dash", "/": "slash",
                   "\\": "backslash", ":": "colon", "+": "plus"}
        self._literal_symbol_words = {
            symbol: {lang: cfg["operator_words"][lang][key] for lang in LANGUAGES}
            for symbol, key in symbols.items()}
        self._literal_symbol_words["@"] = {
            "zh": "at", "en": "at", "ja": "アットマーク", "ru": "собака",
            "de": "ät", "fr": "arobase", "es": "arroba", "th": "แอท"}
        for symbol, words in cfg.get("literal_symbol_words", {}).items():
            self._literal_symbol_words.setdefault(symbol, {}).update(words)
        configure_lexicon(self._lexicon_path)
        self._english = EnglishWithHints(british=False, fallback=espeak.EspeakFallback(british=False))
        self._english.lexicon.golds.update(lexicon.get("english_overrides", {}))
        for word, overrides in lexicon.get("pos_overrides", {}).items():
            self._english.lexicon.golds[word].update(overrides)
        self._read_present = self._english.lexicon.golds["read"]["DEFAULT"]
        self._lead_metal = self._english.lexicon.golds["led"]
        self._japanese, self._sandhi = Cutlet(), ToneSandhi()
        available = EspeakBackend.supported_languages()
        french = next(code for code in ("fr-fr", "fr") if code in available)
        self._espeak = {lang: espeak.EspeakG2P(language=french if lang == "fr" else lang)
                        for lang in ("ru", "de", "fr", "es")}
        self._thai = ThaiG2P()
        overrides = lexicon.get("thai_ipa_overrides", {})
        self._thai._ipa.update(overrides)
        self._thai._tokenizer._dict = self._thai._tokenizer.vocab | set(overrides)
        self._thai._tokenizer._engine = Tokenizer(
            custom_dict=self._thai._tokenizer.vocab, engine="newmm")
        entries = [entry for entry in lexicon["entries"] if entry["enabled"]]
        alternatives, self._entries, self._native_entries = [], {}, {}
        self._entry_languages = {}
        self._filename_languages = {}
        for index, entry in enumerate(entries):
            default = self._dictionary_phonemes(entry)
            native = {language: self._native(reading, language)
                      for language, reading in entry.get("native_readings", {}).items()}
            if languages := lexicon.get("initialism_group_stress", ()):
                grouped = self._dictionary_phonemes(entry, group_stress=True)
                native = {**dict.fromkeys(languages, grouped), **native}
            for form in [entry["term"], *entry["aliases"]]:
                name = f"w{index}_{len(alternatives)}"
                body = re.escape(form).replace(r"\ ", r"[ \t]+")
                body = body if entry["case_sensitive"] else "(?i:" + body + ")"
                alternatives.append((len(form), f"(?P<{name}>{body})"))
                self._entries[name] = default
                self._native_entries[name] = native
                self._entry_languages[name] = tuple(entry.get("languages", ()))
                self._filename_languages[name] = tuple(entry.get("filename_languages", ()))
        alternatives.sort(reverse=True)
        self._terms = re.compile(r"(?<![A-Za-z0-9_])(?:" +
                                 "|".join(body for _, body in alternatives) +
                                 r")(?![A-Za-z0-9_]|\.[A-Za-z0-9_])")
        # Load lazy corpora before the first queued utterance, without running a
        # neural model or downloading resources during requests.
        for language, text in zip(LANGUAGES, (
                "你好", "Hello", "こんにちは", "Здравствуйте", "Guten Tag",
                "Bonjour", "Buenos días", "สวัสดี")):
            self.convert(text, language)

    def detect_language(self, text):
        return self.language_evidence(text) or "en"

    def language_evidence(self, text):
        """Return this request's language, or None for script-neutral input."""
        # Literal contents do not establish the surrounding spoken language.
        # Mask them before script checks or lexical aliases can break a filename.
        ordinary = list(text)
        for start, end in self._literal_spans(text) + [m.span() for m in _CODE.finditer(text)]:
            ordinary[start:end] = " " * (end - start)
        ordinary = "".join(ordinary)
        if re.search(r"[\u0e00-\u0e7f]", ordinary):
            return "th"
        if re.search(r"[\u3040-\u30ff]", ordinary):
            return "ja"
        if re.search(r"[\u0400-\u04ff]", ordinary):
            return "ru"
        if re.search(r"[\u4e00-\u9fff]", ordinary):
            # Pure Han text without kana is normally Chinese in this deployment;
            # callers can explicitly select Japanese for ambiguous short names.
            return "zh"
        if re.fullmatch(r"(?:Hallo|Danke)[.!?…]*", ordinary.strip(), re.I):
            return "de"
        greeting = ordinary.strip(" \t\r\n.!?…¡¿").lower()
        if greeting in {"bonjour", "bonsoir", "merci", "salut"}:
            return "fr"
        if greeting in {"hola", "gracias", "buenas tardes", "buenas noches"}:
            return "es"
        ordinary = self._terms.sub(lambda m: m[0] if self._entry_languages[m.lastgroup] else " ", ordinary)
        ordinary = re.sub(r"(?<!\w)IP(?:v[46])?(?!\w)", " ", ordinary, flags=re.I)
        spans = protected_spans(ordinary)
        spans += [m.span() for m in _unit_pattern().finditer(ordinary)]
        spans += [m.span() for m in _EN_HARDWARE.finditer(ordinary)]
        for match in _FORMULA_LIKE.finditer(ordinary):
            expression = match[0].rstrip(". \t")
            try:
                _MathParser(_TOKEN.findall(expression)).parse()
            except ValueError:
                continue
            spans.append(match.span())
        masked = list(ordinary)
        for start, end in spans:
            masked[start:end] = " " * (end - start)
        ordinary = "".join(masked)
        words = re.findall(r"[^\W\d_]+", ordinary)
        if not any(len(word) > 1 for word in words):
            return None
        if len(words) <= 3 and words[0].lower() in {
                "compruebe", "comprueba", "verifique", "verifica", "resuelva", "resuelve"}:
            return "es"
        with self._lock:
            return self._identifier.classify(ordinary)[0]

    def _dictionary_matches(self, text, language):
        return (match for match in self._terms.finditer(text)
                if not self._entry_languages[match.lastgroup]
                or language in self._entry_languages[match.lastgroup])

    def _dictionary_phonemes(self, entry, *, group_stress=False):
        if entry.get("phonemes"):
            return entry["phonemes"]
        parts = []
        for letters, words in groupby(entry["spoken_english"].split(),
                key=lambda word: bool(re.fullmatch(r"[A-Z]", word))):
            if letters:
                # Explicitly spaced letters remain separate spoken words.
                values = [self._english.lexicon.get_NNP(word)[0] for word in words]
                if group_stress:
                    # Match Misaki's initialism stress without joining its words.
                    values[:-1] = [value.replace("ˈ", "ˌ") for value in values[:-1]]
                parts.extend(values)
            else:
                parts.append(self._english(" ".join(words).lower())[0])
        return " ".join(parts)

    def _english_context_readings(self, text, protected, source_text):
        protected = [*protected, *(m.span() for m in _PHONE_HINT.finditer(text))]
        calendar = re.search(r"\b(?:in|during)\s+(?:the year\s+)?\d{4}\b", source_text, re.I)
        readings = []
        for match in re.finditer(r"\b(read|lead)\b", text, re.I):
            if any(match.start() < end and match.end() > start for start, end in protected):
                continue
            before, after = text[:match.start()], text[match.end():]
            left = re.split(r"[.!?;\n]", before)[-1]
            right = re.split(r"[.!?;\n]", after, 1)[0]
            clause = left + match[0] + right
            if match[0].lower() == "read":
                habitual = re.search(r"\b(?:every|each) (?:day|morning|week)\b|\b(?:daily|usually|regularly)\b", clause, re.I)
                subject = re.fullmatch(r"\s*(?:(?:every|each) (?:day|morning|week)[,\s]+)?"
                                       r"(?:I|we|you|they)\s+(?:(?:usually|often|regularly)\s+)?", left, re.I)
                past = re.search(r"\b(?:yesterday|ago|last|had|did|was|were)\b|\bused\s+to\b|"
                                 r"\b(?:in|during)\s+(?:the year\s+)?\d{4}\b|"
                                 r"\b(?:before|after|when|until|while)\s+(?:I|we|you|they|he|she|it)\b", clause, re.I)
                if habitual and subject and not past and not calendar:
                    readings.append((*match.span(), self._read_present))
            elif (re.match(r"\s+(?:pipes?|paint|poisoning|metal)\b", right, re.I) or
                  re.search(r"\bpipes?\s+(?:contain|contains|contained)\s+$", left, re.I) or
                  (re.search(r"\b(?:metal|element)\s+$", left, re.I) and
                   re.match(r"\s+(?:is|has|was)\b|\s*$", right, re.I))):
                # Metal lead and the native dictionary's "led" have the same reading.
                readings.append((*match.span(), self._lead_metal))
        for match in re.finditer(r"\b([aApP])\.[mM]\.(?=\s*$|\s+(?:Which|Please|Then|We|It|This|That|I)\b)", text):
            if not any(match.start() < end and match.end() > start for start, end in protected):
                # Preserve an explicit sentence stop in the native phone hint;
                # spaCy otherwise folds the final dot into the abbreviation.
                readings.append((match.start(), match.end(),
                                 self._english.lexicon.get_NNP(match[1].upper() + "M")[0] + "."))
        return readings

    def _chinese(self, text):
        import cn2an
        import jieba.posseg as psg
        from misaki.zh import ZHG2P
        from pypinyin import lazy_pinyin, Style
        readings = self._chinese_readings
        # A transliterated name's middle dot is a word boundary, not a phone.
        text = re.sub(r"(?<=[\u4e00-\u9fff])·(?=[\u4e00-\u9fff])", " ", text)
        text = ZHG2P.map_punctuation(cn2an.transform(text, "an2cn"))
        parts = []
        for span in re.findall(r"[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+", text):
            if not re.fullmatch(r"[\u4e00-\u9fff]+", span):
                parts.append(span)
                continue
            words = [(word, pos) for word, pos in psg.lcut(span)]
            modal_de = {
                index for index, (word, pos) in enumerate(words)
                if (not index or words[index - 1][0] in {
                    "还", "必须", "我", "你", "他", "她", "我们", "你们", "他们"})
                and ((word == "得" and pos == "ud" and index + 1 < len(words)
                      and words[index + 1][0] in {"先", "尽快", "重新", "重新启动", "重试"})
                     or (word == "得去" and pos == "v"))
            }
            # Retain pypinyin's phrase context, including phrases Jieba splits differently.
            contextual = lazy_pinyin(span, style=Style.TONE3, neutral_tone_with_five=True,
                                     errors=lambda unknown: list(unknown))
            # A title and a surname may be separate Jieba tokens. Match full
            # phrases across token boundaries without changing larger words.
            boundaries = set(accumulate((len(word) for word, _ in words), initial=0))
            for phrase, reading in readings.items():
                for match in re.finditer(re.escape(phrase), span):
                    start, end = match.span()
                    if start in boundaries and end in boundaries:
                        contextual[start:end] = reading
            syllables = []
            offset = 0
            for index, (word, pos) in enumerate(words):
                phones = list(readings[word]) if word in readings else contextual[offset:offset + len(word)]
                offset += len(word)
                previous, previous_pos = words[index - 1] if index else ("", "")
                following, following_pos = words[index + 1] if index + 1 < len(words) else ("", "")
                if word == "地" and pos == "uv":
                    phones[0] = "de5"
                elif index in modal_de:
                    phones[0] = "dei3"
                elif word == "得" and pos == "ud":
                    complement = following_pos in {"a", "ad"}
                    if following in {"很", "太", "再", "非常", "更", "不", "比较"} and index + 2 < len(words):
                        complement = words[index + 2][1] in {"a", "ad"}
                    if previous_pos.startswith("v") and complement:
                        phones[0] = "de5"
                elif word == "长" and previous in {"耗时", "更", "较", "比较"}:
                    phones[0] = "chang2"
                if index + 1 in modal_de and phones[-1].endswith("3"):
                    phones[-1] = phones[-1][:-1] + "2"
                syllables.extend(phones)
            # Choose contextual readings before native merging of 不/一/three-tone groups.
            rendered = []
            offset = 0
            for word, pos in self._sandhi.pre_merge_for_modify(words):
                phones = list(readings[word]) if word in readings else syllables[offset:offset + len(word)]
                offset += len(word)
                phones = self._sandhi._bu_sandhi(word, phones)
                phones = self._sandhi._yi_sandhi(word, phones)
                phones = self._sandhi._three_sandhi(word, phones)
                rendered.append("".join(ZHG2P.py2ipa(phone) for phone in phones))
            parts.append(" ".join(rendered))
        return "".join(parts).replace(chr(815), "")

    def _thai_phones(self, text):
        from fastthaig2p import normalize, ipa_to_kokoro
        from fastthaig2p.fallback import fallback_g2p
        parts = []
        for token in self._thai._tokenizer.tokenize(normalize(text)):
            if not token.strip():
                continue
            parts.append(token if all(category(c).startswith("P") for c in token)
                         else self._thai._ipa.get(token) or fallback_g2p(token))
        ipa = " ".join(parts)
        phones = re.sub(r"/[^/]+/", lambda match: ipa_to_kokoro(match[0]).replace(".", ""), ipa)
        return phones if self._native_symbols else phones.translate(str.maketrans({"→": "", "↑": "→"}))

    def _native(self, text, language):
        if language == "en":
            return self._english(text)[0]
        if language == "zh":
            return self._chinese(text)
        if language == "ja":
            return self._japanese(text.replace("/", " スラッシュ "))[0].translate(
                str.maketrans({"«": "(", "»": ")"}))
        if language == "es":
            text = text.replace("¿", "").replace("¡", "")
        if language == "th":
            return self._thai_phones(text)
        phones = self._espeak[language](text)[0]
        return phones if self._native_symbols else phones.translate(str.maketrans(_SYMBOL_MAP.get(language, {})))

    def _literal_spans(self, text):
        spans = protected_spans(text)
        # The notation parser's Unicode word boundary misses "report-v2.txtを".
        # Reuse that parser on ASCII runs, preserving offsets and still requiring
        # a recognized literal; a bare a_b remains ordinary frontend input.
        for match in re.finditer(r"[!-~]*[_.][!-~]*", text):
            spans.extend((match.start() + start, match.start() + end)
                         for start, end in protected_spans(match[0]))
        return spans

    def _expand_literal_symbols(self, text, language, spans=None):
        words = self._literal_symbol_words
        if not any(symbol in text for symbol in words):
            return text
        spans = self._literal_spans(text) if spans is None else spans
        replacements = {}
        if language == "en":
            for match in re.finditer(r"(?<!\d)0\d+(?!\d)", text):
                if any(start <= match.start() and match.end() <= end for start, end in spans):
                    replacements.update({i: " " + self._literal_digits[int(text[i])] + " "
                                         for i in range(match.start(), match.end())})
        elif language == "ru":
            cfg = _config()
            for start, end in spans:
                for match in re.finditer(r"(?<![A-Za-z0-9])[vV](\d+(?:\.\d+)*)(?![A-Za-z0-9]|\.\d)", text[start:end]):
                    # In a filename, v2 must not become English "vee two"
                    # between native-language underscore and dot readings.
                    pieces = [cfg["letters"][language]["v"]]
                    for index, component in enumerate(match[1].split(".")):
                        if index:
                            pieces.append(words["."][language])
                        pieces.append(_digits(component, language, cfg)
                                      if len(component) >= 4 or len(component) > 1 and component.startswith("0")
                                      else _number(component, language, cfg))
                    replacements.update({i: "" for i in range(start + match.start(), start + match.end())})
                    replacements[start + match.start()] = " " + _join(pieces, language) + " "
        return "".join(
            replacements[index] if index in replacements else
            " " + words[char][language] + " "
            if char in words and any(start <= index < end for start, end in spans)
            else char for index, char in enumerate(text))

    def _prepare_text(self, text, language):
        if language == "auto":
            language = self.detect_language(text)
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported Kokoro Multi language: {language}")
        flattened = re.sub(r"[\r\n]", " ", text)
        protected_lines = (protected_spans(flattened)
                           + [m.span() for m in self._dictionary_matches(flattened, language)]
                           + [m.span() for m in _unit_pattern(language).finditer(flattened)]
                           + [m.span() for m in _FORMULA_LIKE.finditer(flattened)])
        text = re.sub(r"[\r\n]", lambda m: " " if any(a <= m.start() < b for a, b in protected_lines)
                      else m[0], text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            parts = [lines[0]]
            for left, right in zip(lines, lines[1:]):
                stopped = any(m.end() == len(left) for m in _STOPS.finditer(left))
                continuation = re.search(r"[=+−*/^×÷-]$", left) or re.match(r"[=+−*/^×÷-]", right)
                parts.extend((" " if stopped or continuation else ", ", right))
            text = "".join(parts)
        with self._lock:
            # Keep numbered names intact until their pronunciation entry is read.
            names = [m.span() for m in self._dictionary_matches(text, language)
                     if any(char.isdigit() for char in m[0])]
            spoken, _ = normalize_notation(text, language, lexical_spans=names)
        return text, spoken, language

    def convert(self, text, language="auto"):
        source, spoken, language = self._prepare_text(text, language)
        return self._phonemize(spoken, language, source)

    def iter_phonemes(self, text, language="auto"):
        """Normalize the full request once; defer tail G2P until its turn."""
        source, spoken, language = self._prepare_text(text, language)
        protected = self._literal_spans(spoken) + [m.span() for m in self._dictionary_matches(spoken, language)]
        if language == "en":
            # Keep first-sentence context; prepare the remainder after first PCM.
            parts = first_sentence_parts(spoken, protected)
            first = self._phonemize(parts[0], language, source)
            chunks = sentence_phoneme_chunks(first, pack_tail=True)
            if len(chunks) > 1:
                yield chunks[0]
                tail = " ".join(chunks[1:])
            else:
                yield first
                tail = ""
            if len(parts) > 1:
                tail += " " + self._phonemize(parts[1], language, source)
            if tail.strip():
                yield tail
            return
        # Native Latin prose keeps sentence context; mixed-name passages retain
        # their smaller opening clause for first-audio latency.
        clause = language not in ("en", "fr", "es") or any(self._dictionary_matches(spoken, language))
        parts = first_sentence_parts(spoken, protected, clause=clause)
        if (language == "zh" and len(spoken) <= 32 and not _LATIN.search(spoken)
                and parts[0].rstrip().endswith((",", "，", "、"))):
            # Keep the opening comma or list separator inside a short phrase.
            yield self._phonemize(spoken, language, source)
            return
        if language in ("fr", "es"):
            phones = [self._phonemize(part, language, source) for part in parts]
            joined = re.sub(r"\s+", " ", " ".join(phones)).strip()
            # A dictionary hint must not split an otherwise single short request.
            yield from [joined] if len(joined) <= 220 else phones
            return
        for index, part in enumerate(parts):
            if language != "th" or index:
                yield self._phonemize(part, language, source)
                continue
            # Prepare Thai phrases lazily, retaining the existing acoustic chunks.
            # More than one chunk of lookahead fixes its original first boundary.
            pending, cursor = [], 0
            for match in re.finditer(r"\s+", part):
                end = match.end()
                if part[end:].startswith("ๆ") or any(start < end < stop for start, stop in protected):
                    continue
                pending.append(self._phonemize(part[cursor:end], language, source))
                cursor = end
                prepared = re.sub(r"\s+", " ", " ".join(pending)).strip()
                if len(prepared) > 220:
                    head, *tail = sentence_phoneme_chunks(prepared)
                    yield head
                    rest = self._phonemize(part[cursor:], language, source)
                    yield " ".join([*tail, rest])
                    break
            else:
                pending.append(self._phonemize(part[cursor:], language, source))
                yield " ".join(pending)

    def _phonemize(self, spoken, language, source_text):
        with self._lock:
            protected = self._literal_spans(spoken)
            matches = [match for match in self._dictionary_matches(spoken, language)
                       if not any(match.start() < end and match.end() > start
                                  for start, end in protected)
                       or (language in self._filename_languages[match.lastgroup]
                           and spoken[match.start()-1:match.start()] == "."
                           and any(match.end() == end for _, end in protected))]
            cursor, parts = 0, []

            def expand_literals(value, offset):
                # A dictionary suffix must not discard the enclosing filename.
                spans = [(max(0, start-offset), min(len(value), end-offset))
                         for start, end in protected
                         if start < offset+len(value) and end > offset]
                return self._expand_literal_symbols(value, language, spans)

            if language == "en":
                readings = []
                for match in matches:
                    end = match.end()
                    phones = self._native_entries[match.lastgroup].get(language, self._entries[match.lastgroup])
                    suffix = re.match(r"-\d+[A-Za-z]?(?![A-Za-z0-9_.-])", spoken[end:])
                    if suffix:
                        # A model suffix belongs to the hinted name: -82M in
                        # Kokoro-82M is a size label, not a negative quantity.
                        end += suffix.end()
                        phones += " " + self._english(suffix[0][1:])[0]
                    readings.append((match.start(), end, phones))
                readings += self._english_context_readings(spoken, protected + [match.span() for match in matches], source_text)
                for start, end, phones in sorted(readings):
                    parts.extend((expand_literals(spoken[cursor:start], cursor),
                                  "[" + spoken[start:end] + "](/" + phones + "/)"))
                    cursor = end
                parts.append(expand_literals(spoken[cursor:], cursor))
                return self._english("".join(parts))[0]

            compounds = []
            for word in re.finditer(_LATIN_WORD, spoken):
                if "-" not in word[0] or any(word.start() < end and word.end() > start for start, end in protected):
                    continue
                hits = [match for match in matches if word.start() <= match.start() and match.end() <= word.end()]
                if not hits or any(match.span() == word.span() for match in hits):
                    continue
                if (language in ("zh", "ja", "ru", "th")
                        or re.fullmatch(r"[A-Za-z]+\d+-TTS", word[0])
                        or re.search(r"-\d+[A-Za-z]?$", word[0])):
                    # Keep the whole English compound in one G2P call.  Splitting
                    # F5-TTS at TTS leaves a dangling hyphen in the native branch.
                    compounds.append((word.start(), word.end(), self.convert(word[0], "en")))
                    matches = [match for match in matches if match not in hits]
            readings = sorted([(match.start(), match.end(),
                                self._native_entries[match.lastgroup].get(language, self._entries[match.lastgroup]))
                               for match in matches] + compounds)

            def plain(value, offset, following=""):
                value = expand_literals(value, offset)
                pieces = _LATIN.split(value) if language in ("zh", "ja", "ru", "th") else [value]
                for piece in pieces:
                    if piece.strip():
                        lang = "en" if language in ("zh", "ja", "ru", "th") and _LATIN.fullmatch(piece) else language
                        phones = self._native(piece, lang)
                        if following and (lang in self._unstressed_linkers or lang in self._mandatory_liaisons):
                            linker = re.search(r'''([^\W\d_]+)[\s"“«(]*$''', piece)
                            if linker:
                                word = linker[1].lower()
                                unstressed = word in self._unstressed_linkers.get(lang, ())
                                liaison = self._mandatory_liaisons.get(lang, {}).get(word, "")
                                if (following.lstrip(" ˈˌ")[:1] not in _VOWEL_ONSETS
                                        or piece[linker.start(1)+len(linker[1]):].strip()):
                                    liaison = ""
                                def connect(match):
                                    token = match[1].replace("ˈ", "").replace("ˌ", "") if unstressed else match[1]
                                    if liaison and not token.endswith(liaison):
                                        token += liaison
                                    return token + match[2]
                                # Keep weak words and mandatory links across dictionary insertions.
                                phones = re.sub(r'''([^\s"“”«»()]+)([\s"“”«»()]*)$''', connect, phones)
                        parts.append(phones)

            for start, end, phones in readings:
                prefix = spoken[cursor:start]
                clitic = re.search(r"(?<!\w)([lLdD])['’]$", prefix) if language == "fr" else None
                # Elided l'/d' attach to the name; do not spell them as letters.
                plain(prefix[:clitic.start()] if clitic else prefix, cursor, following=phones)
                parts.append((clitic[1].lower() if clitic else "") + phones)
                cursor = end
            plain(spoken[cursor:], cursor)
            return " ".join(parts)
