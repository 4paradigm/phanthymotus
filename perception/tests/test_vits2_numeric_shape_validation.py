"""Unsupported numeric syntax is spoken literally; supported fields keep priority."""
from test_vits2_liaoqianqian_lexicon import lexical_resource

import pytest

from plugins.vits2_tts_language.neutral_text import (
    is_neutral_text, normalize_mixed_text, normalize_neutral_text,
)
from plugins.vits2_tts_trt.frontend.number_context import find_spans
from plugins.vits2_tts_trt.frontend.numeric_formats import find_spans as numeric_spans
from test_vits2_language_integration import make_plugin, start, call, finished


@pytest.mark.parametrize("text", [
    "2029/03/04/5", "2029-03-04-5", "2029.03.04.5", "2029 03/04/5",
    "2029/03/04.5", "2029 03/ 04.5", "2029/03/04 / 5",
    "Read 2029/03/04/5 now.", "预约日期是2029/03/04/5。",
])
def test_malformed_date_shapes_are_read_literal(text):
    for language in ("zh", "en"):
        spoken = normalize_mixed_text(text, language)
        assert spoken and not any(char.isdecimal() for char in spoken)
        assert "March" not in spoken and "年" not in spoken


@pytest.mark.parametrize("text", [
    "$1,23.45", "¥1,299.5.0", "£--2.00", "+$-1.00", "$12,3456",
    "$1.2.3", "$1,,234", "Price $1,23.45 today.", "金额是¥1,299.5.0。",
])
def test_malformed_currency_shapes_are_read_literal(text):
    for language in ("zh", "en"):
        spoken = normalize_mixed_text(text, language)
        assert spoken and not any(char.isdecimal() for char in spoken)
        assert "符号" in spoken if language == "zh" else "sign" in spoken


def test_diagnostic_queries_remain_pure_and_do_not_claim_date_prefix():
    for text in ("2029/03/04/5", "$1,23.45", "¥1,299.5.0"):
        assert isinstance(is_neutral_text(text), bool)
    assert find_spans("2029/03/04/5", "EN", lambda value, language: value) == []
    assert numeric_spans("$1,23.45", "EN", lambda value, language: value) == []


@pytest.mark.parametrize("text", [
    "https://example.com/2029/03/04/5?q=$1,23.45",
    "$1.2.3@example.com", "2029.03.04.5@example.com",
    "ID:2029.03.04.5", "订单ID:2029-03-04-5", "Order ID: v2029.03.04.5",
    "v2029.03.04.5", "version 2029.03.04.5", "version 2029.03.04.5.6",
    "version 2026.10.23", "v2026.123.7",
])
def test_addresses_explicit_ids_and_versions_take_priority(text):
    for language in ("zh", "en"):
        normalize_mixed_text(text, language)


@pytest.mark.parametrize("text", [
    "$1,234.50", "¥1,299.50", "￥+0.005", "−¥.75", "$1.00.", "$1.00, next",
    "2029/03/04", "2029 03/ 04", "日期=2029/03/04", "date=2029/03/04",
    "2029/03/04. 5 samples are ready.", "2029/03/04。5个样本已准备好。",
])
def test_supported_fields_and_sentence_punctuation_remain_valid(text):
    for language in ("zh", "en"):
        normalize_mixed_text(text, language)


@pytest.mark.parametrize("text", [
    "计算2029/03/04/5", "x=2029/03/04/5", "Calculate 2029/03/04/5.",
    "x=(2029/03/04)/5", "y=8/2", "2029/3", "ratio 3/4",
])
def test_unambiguous_arithmetic_is_not_rejected_or_read_as_a_calendar(text):
    zh, en = normalize_mixed_text(text, "zh"), normalize_mixed_text(text, "en")
    assert "March" not in en
    assert "年" not in zh
    if text.count("/") >= 3:
        assert "divided by" in en
        assert "除以" in zh


@pytest.mark.parametrize("mode", ["zh", "en", "auto"])
@pytest.mark.parametrize("text,zh_marker,en_marker", [
    ("2029/03/04/5", "斜杠零三", "slash zero three"),
    ("预约日期2029/03/04/5", "斜杠零三", "slash zero three"),
    ("$1,23.45", "美元符号", "dollar sign"),
    ("Price ¥1,299.5.0 today.", "元符号", "yuan sign"),
])
def test_unified_entry_prepares_readable_malformed_text(make_plugin, mode, text, zh_marker, en_marker):
    state = make_plugin(reply_language=mode)
    start(state)
    before = len(state.adapter.prepared)
    result = call(state, "speak", instance_id="a", text=text)
    finished(state, result)
    assert len(state.adapter.prepared) == before + 1
    assert state.adapter.spoken
    # This lifecycle fixture leaves AUTO prose's final FST to the adapter;
    # the actual-FST probe separately checks those complete model inputs.
    if mode != "auto" or is_neutral_text(text):
        spoken = " ".join(state.adapter.spoken)
        assert (zh_marker if result["reply_language"] == "zh" else en_marker) in spoken


@pytest.mark.parametrize("seed,bad,language", [
    ("Hello there.", "2029/03/04/5", "en"),
    ("你好。", "$1,23.45", "zh"),
])
def test_numeric_literal_uses_auto_history_without_inventing_a_new_language(make_plugin, seed, bad, language):
    state = make_plugin(reply_language="auto")
    start(state)
    finished(state, call(state, "speak", instance_id="a", text=seed))
    before = call(state, "info", instance_id="a")
    result = call(state, "speak", instance_id="a", text=bad)
    assert result["reply_language"] == language
    finished(state, result)
    after = call(state, "info", instance_id="a")
    for field in ("remembered_language", "resolved_language"):
        assert before[field] == after[field]
    assert after["remembered_language"] == language
    follow = call(state, "speak", instance_id="a", text="2029/03/04")
    assert follow["reply_language"] == language
    finished(state, follow)


@pytest.mark.parametrize("mode", ["zh", "en", "auto"])
def test_neutral_normalizer_spells_complete_bad_currency_or_date(mode):
    selected = "zh" if mode == "auto" else mode
    assert ("斜杠" if selected == "zh" else "slash") in normalize_neutral_text("2029/03/04/5", selected)
    assert ("元符号" if selected == "zh" else "yuan sign") in normalize_neutral_text("¥1,299.5.0", selected)


@pytest.mark.parametrize("text,zh,en", [
    ("2029/03/04/5", "二零二九斜杠零三斜杠零四斜杠五",
     "two zero two nine slash zero three slash zero four slash five"),
    ("$1,23.45", "美元符号一逗号二三点四五", "dollar sign one comma two three dot four five"),
    ("¥1,299.5.0", "元符号一逗号二九九点五点零", "yuan sign one comma two nine nine dot five dot zero"),
    ("$001,23.4500", "美元符号零零一逗号二三点四五零零",
     "dollar sign zero zero one comma two three dot four five zero zero"),
    ("2027/02/29", "二零二七斜杠零二斜杠二九", "two zero two seven slash zero two slash two nine"),
    ("2029-13-04", "二零二九连字符一三连字符零四", "two zero two nine hyphen one three hyphen zero four"),
    ("２０２９／０３／０４／５", "二零二九斜杠零三斜杠零四斜杠五",
     "two zero two nine slash zero three slash zero four slash five"),
    ("＄００１，２３．４５００", "美元符号零零一逗号二三点四五零零",
     "dollar sign zero zero one comma two three dot four five zero zero"),
    ("2030--02--28", "二零三零连字符连字符零二连字符连字符二八",
     "two zero three zero hyphen hyphen zero two hyphen hyphen two eight"),
    ("2026/10-23", "二零二六斜杠一零连字符二三",
     "two zero two six slash one zero hyphen two three"),
    ("2026-123-7", "二零二六连字符一二三连字符七",
     "two zero two six hyphen one two three hyphen seven"),
    ("2026-10-", "二零二六连字符一零连字符", "two zero two six hyphen one zero hyphen"),
    ("2026/10/", "二零二六斜杠一零斜杠", "two zero two six slash one zero slash"),
    ("2026//10/23", "二零二六斜杠斜杠一零斜杠二三",
     "two zero two six slash slash one zero slash two three"),
])
def test_literal_output_retains_every_digit_symbol_and_leading_zero(text, zh, en):
    assert normalize_neutral_text(text, "zh") == zh
    assert normalize_neutral_text(text, "en") == en


def test_incomplete_date_in_prose_and_explicit_math_are_distinguished():
    assert normalize_mixed_text("日期2026-10-", "zh") == "日期二零二六连字符一零连字符"
    assert "two zero two six hyphen one zero hyphen" in normalize_mixed_text("日期2026-10-", "en")
    assert "divided by" in normalize_mixed_text("y=2026/10-23", "en")
    assert "minus" in normalize_mixed_text("计算2026-123-7", "en")
    assert "slash" not in normalize_mixed_text("y=2026/10-23", "en")
    assert "hyphen" not in normalize_mixed_text("计算2026-123-7", "en")


@pytest.mark.parametrize("text", ["2026.10.", "2029/03/04.", "2029/03/04. 5 samples."])
def test_terminal_period_does_not_turn_a_valid_date_or_decimal_into_literal(text):
    en = normalize_mixed_text(text, "en")
    assert "hyphen" not in en and "slash" not in en and "dot" not in en


@pytest.mark.parametrize("text", ["RTX-4090A", "RTX4090-A", "rtx-4090a", "rtx4090-a",
                                  "RTX‑4090A", "RTX4090‑A"])
def test_model_suffix_is_an_unknown_identifier_instead_of_a_partial_model_or_unit(text):
    en = normalize_mixed_text(text, "en")
    zh = normalize_mixed_text(text, "zh")
    assert "four zero nine zero" in en
    assert "forty ninety" not in en and "amperes" not in en
    assert "四零九零" in zh and "安培" not in zh


@pytest.mark.parametrize("text", ["4090A", "x-5A", "x−5A"])
def test_true_current_and_single_variable_subtraction_keep_measurement_reading(text):
    en = normalize_mixed_text(text, "en")
    zh = normalize_mixed_text(text, "zh")
    assert "amperes" in en and "安培" in zh
