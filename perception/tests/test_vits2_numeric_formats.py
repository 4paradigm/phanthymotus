"""Format reading and boundaries, using real number readers without a model."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_language.neutral_text import normalize_mixed_text, normalize_neutral_text
from plugins.vits2_tts_trt.frontend.numeric_formats import find_spans
from plugins.vits2_tts_trt.frontend.special_text import verbalize_chinese_number
from plugins.vits2_tts_language.neutral_text import is_neutral_text
from plugins.vits2_tts_language.policy import LanguagePolicy


def formats(text, language):
    return [(text[a:b], spoken) for a, b, spoken in find_spans(
        text, language, lambda value, lang: verbalize_chinese_number(value))]


@pytest.mark.parametrize("raw,zh,en", [
    ("$1,234.50", "一千二百三十四点五零美元", "one thousand two hundred thirty four dollars and fifty cents"),
    ("$1.01", "一点零一美元", "one dollar and one cent"),
    ("$0.50", "零点五零美元", "fifty cents"),
    ("$.50", "零点五零美元", "fifty cents"),
    ("$1.00", "一点零零美元", "one dollar"),
    ("$0.00", "零点零零美元", "zero dollars"),
    ("-$12.05", "负十二点零五美元", "minus twelve dollars and five cents"),
    ("$-12.05", "负十二点零五美元", "minus twelve dollars and five cents"),
    ("€2.50", "二点五零欧元", "two euros and fifty cents"),
    ("£0.01", "零点零一英镑", "one penny"),
    ("£0.02", "零点零二英镑", "two pence"),
    ("$0.001", "零点零零一美元", "zero point zero zero one dollars"),
])
def test_complete_currency_values_preserve_sign_precision_and_minor_units(raw, zh, en):
    assert formats(raw, "ZH") == [(raw, zh)]
    assert formats(raw, "EN") == [(raw, en)]
    for mode, expected in (("zh", zh), ("en", en)):
        assert normalize_mixed_text("Price " + raw + " today", mode) == "Price " + expected + " today"
        assert normalize_neutral_text(raw, mode) == expected


@pytest.mark.parametrize("raw,zh,en", [
    ("1st", "第一", "first"), ("2nd", "第二", "second"),
    ("3rd", "第三", "third"), ("11th", "第十一", "eleventh"),
    ("12th", "第十二", "twelfth"), ("13th", "第十三", "thirteenth"),
    ("21st", "第二十一", "twenty first"), ("22ND", "第二十二", "twenty second"),
    ("101st", "第一百零一", "one hundred first"),
    ("111th", "第一百一十一", "one hundred eleventh"),
    ("1000th", "第一千", "one thousandth"),
])
def test_ordinals_follow_selected_language_without_hyphen_pause(raw, zh, en):
    for mode, expected in (("zh", zh), ("en", en)):
        assert normalize_mixed_text("The " + raw + " sample", mode) == "The " + expected + " sample"


@pytest.mark.parametrize("raw,expected", [
    ("v1.2.3", "V one point two point three"),
    ("version 1.12.003", "version one point twelve point zero zero three"),
    ("version1.2", "version one point two"),
    ("版本号：1.12.003", "版本号: one point twelve point zero zero three"),
    ("version 2026.10.23", "version two thousand twenty six point ten point twenty three"),
])
def test_only_explicit_version_formats_are_read_as_versions(raw, expected):
    assert normalize_mixed_text(raw, "en") == expected
    assert "一点二点三" in normalize_mixed_text("v1.2.3", "zh")


@pytest.mark.parametrize("raw", [
    "$12,34.50", "$1,2345", "-$-1", "$1.2.3", "$1_2", "$12USD",
    "11st", "12nd", "13rd", "21th", "A21st", "21stModel",
    "v1.2.3.4.5", "v1.2.3-rc1", "v1.2+build5", "v1.2.3‑beta", "version1.2–rc1", "file1.2.3",
    "conversion1.2", "versioned1.2", "version\n1.2.3", "版本\n1.2.3",
    "abc123", "1.2.3", "192.168.1.1", "foo_v1.2.3", "v1.2suffix",
])
def test_no_partial_claims_or_guessing_unknown_ids(raw):
    assert formats(raw, "EN") == []


def test_explicit_id_takes_priority_over_versions_and_math():
    en = normalize_mixed_text("ID: abc123; Order ID: v1.02.3", "en")
    assert "A B C one two three" in en
    assert "V one dot zero two dot three" in en
    assert "point" not in en
    zh = normalize_mixed_text("编号: abc123; ID: v1.02.3", "zh")
    assert "ABC一二三" in zh.replace(" ", "")
    assert "V一点零二点三" in zh.replace(" ", "")
    for raw in ("identity123", "identifier123", "IDLE123", "abc123"):
        assert normalize_mixed_text(raw, "en") == raw


def test_addresses_override_currency_versions_and_ordinals():
    text = "https://example.com/v1.2.3/$1,234.50/21st abc123@example.com"
    en = normalize_mixed_text(text, "en")
    assert "dollars" not in en and "twenty first" not in en and "point" not in en
    assert "dollar sign one comma two three four dot five zero" in en
    assert "A B C one two three at" in en


def test_decimal_zero_omission_and_lines_keep_numeric_preference():
    assert normalize_mixed_text("Value .5 and -.25", "en") == "Value zero point five and minus zero point two five"
    assert normalize_mixed_text("Value .5 and -.25", "zh") == "Value 零点五 and 负零点二五"
    text = "$1.50\r\n21st\nv1.2.3"
    result = normalize_mixed_text(text, "en")
    assert result == "one dollar and fifty cents\r\ntwenty first\nV one point two point three"


@pytest.mark.parametrize("text", ["ID abc123", "ID:abc123", "Order ID:abc123", "Serial number: A001", "ID: v1.02.3", "  ID：abc123。\n"])
def test_complete_identifier_labels_inherit_auto_language_but_not_between_instances(text):
    assert is_neutral_text(text)
    policy = LanguagePolicy("auto")
    assert policy.begin_turn(text)["reply_language"] == "zh"
    policy.begin_turn("Hello")
    turn = policy.begin_turn(text)
    assert (turn["reply_language"], turn["language_source"]) == ("en", "history")
    assert LanguagePolicy("auto").begin_turn(text)["reply_language"] == "zh"
    policy.begin_turn("你好")
    turn = policy.begin_turn(text)
    assert (turn["reply_language"], turn["language_source"]) == ("zh", "history")


@pytest.mark.parametrize("text", [
    "The order ID is abc123", "Please read ID: abc123", "ID abc123 is ready",
    "ID\nabc123", "订单号abc123", "identity123", "IDLE123", "IDabc123",
])
def test_identifier_label_detection_does_not_remove_language_bearing_prose(text):
    assert not is_neutral_text(text)
