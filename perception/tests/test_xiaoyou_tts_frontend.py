"""CPU-only regressions for Xiaoyou text normalization and source boundaries."""

from types import SimpleNamespace

import pytest

from plugins.xiaoyou_tts.frontend import numbers
from plugins.xiaoyou_tts.frontend.phonemizer import Frontend


@pytest.mark.parametrize(
    "language,text,readings",
    [
        ("zh", "今天是2026.11.20，功耗为1.5kJ，结果 x = 3.5。",
         ("二零二六年十一月二十日", "一点五千焦", "x 等于 三 点 五")),
        ("en", "Today is 2026.11.20. Use 1.5 ms and 2 GB. Solve x = 3.5.",
         ("November twentieth two thousand twenty six", "one point five milliseconds",
          "two gigabytes", "x equals three point five")),
    ],
    ids=["chinese", "english"],
)
def test_dates_units_and_equations_use_the_selected_language(language, text, readings):
    normalized = Frontend.normalize_text(text, language)
    for reading in readings:
        assert reading in normalized


@pytest.mark.parametrize("separator", ["-", "–"])
def test_short_slash_range_follows_a_calendar_field(separator):
    text = f"日期：2027 年 1 月 1 日是周五，1/1{separator}1/3 可以预约讲解。"
    assert Frontend.normalize_text(text, "zh") == (
        "日期:二零二七年一月一日是周五,一月一日至一月三日 可以预约讲解。"
    )


def test_slash_range_still_means_division_in_an_equation():
    assert Frontend.normalize_text("计算 1/1-1/3。", "zh") == "计算 一 除以 一 减 一 除以 三。"


def test_invalid_calendar_date_is_rejected():
    with pytest.raises(numbers.NumericNormalizationError, match="Invalid calendar date"):
        Frontend.normalize_text("日期：2027 年 2 月 29 日。", "zh")


def test_stock_code_retains_digit_identity_instead_of_a_cardinal_reading():
    text = "股票代码 6682.HK，订单金额 6682 元。"
    assert Frontend.normalize_text(text, "zh") == (
        "股票代码 六六八二 HK,订单金额 六千六百八十二 元。"
    )
    assert "six six eight two HK" in Frontend.normalize_text("Stock code 6682.HK.", "en")


def test_normalization_preserves_source_words_for_the_pronunciation_stage():
    text = "华为发布 iPhoneXS 兼容性报告，孩子快乐地长大，一起参观藏文展品并储藏资料。"
    assert Frontend.normalize_text(text) == (
        "华为发布 iPhoneXS 兼容性报告,孩子快乐地长大,一起参观藏文展品并储藏资料。"
    )


@pytest.mark.parametrize(
    "carrier,language",
    [("请打开设置。", "zh"), ("Open the settings.", "en")],
)
def test_identifier_only_input_does_not_replace_the_auto_language(carrier, language):
    literals = "model_v2.4.json。https://example.com/v2.4?q=1/3。192.168.0.1。"
    previous, found = numbers.auto_context_language(carrier)
    assert (previous, found) == (language, True)
    assert numbers.auto_context_language(literals, previous) == (language, False)

    date = "二零二六年十一月二十日" if language == "zh" else "November twentieth two thousand twenty six"
    assert Frontend.normalize_text(literals + "2026.11.20。", auto_fallback=previous) == (
        literals + date + "。"
    )


def test_auto_follows_a_natural_language_switch_within_the_request():
    normalized = Frontend.normalize_text("今天是 2026.11.20。The date is 2026.11.20.")
    assert normalized == (
        "今天是 二零二六年十一月二十日。The date is November twentieth two thousand twenty six."
    )


@pytest.mark.parametrize(
    "text,literal",
    [
        ("Open model.config.json now.", "model.config.json"),
        ("Visit https://example.com/v2.4?q=1/3 next.", "https://example.com/v2.4?q=1/3"),
        ("Access 192.168.0.1 next.", "192.168.0.1"),
        ("Use TensorRT 2.11.0 here.", "TensorRT 2.11.0"),
        ("Set x = 3.5 now.", "x = 3.5"),
    ],
)
def test_safe_split_keeps_a_complete_literal(text, literal):
    left, right = numbers.safe_split(text)
    assert left and right
    assert left + right == text
    assert literal in left or literal in right


@pytest.mark.parametrize(
    "literal,marker,error",
    [
        ("model_v2.4.json", ".", "complete filename"),
        ("https://example.com/v2.4?q=1/3", "?", "complete address"),
        ("TensorRT 2.11.0", ".", "version component sequence"),
    ],
)
def test_semantic_span_cannot_end_inside_a_protected_literal(literal, marker, error):
    normalized = Frontend.normalize_text(f"Check {literal} then continue.", "en")
    end = normalized.index(literal) + literal.index(marker) + 1
    # The frontend must reject the offset before creating or invoking any G2P.
    frontend = Frontend.__new__(Frontend)
    with pytest.raises(ValueError, match=error):
        frontend.phonemize_normalized(normalized, "en", end=end)


def test_phonemizing_a_semantic_span_does_not_normalize_again(monkeypatch):
    frontend = Frontend.__new__(Frontend)
    normalized = Frontend.normalize_text("Check the meter. Use 1.5 ms.", "en")
    start = normalized.index("Use")
    calls = []

    def call_span(text, first, end):
        calls.append((text, first, end))
        return "a", []

    def unexpected_normalization(*args, **kwargs):
        pytest.fail("phonemize_normalized normalized the source a second time")

    frontend.en = SimpleNamespace(call_span=call_span)
    frontend.vocab = {"a": 7}
    monkeypatch.setattr(Frontend, "normalize_text", staticmethod(unexpected_normalization))
    monkeypatch.setattr(numbers, "normalize_context", unexpected_normalization)

    assert frontend.phonemize_normalized(normalized, "en", start=start) == ("a", [7])
    assert calls == [(normalized, start, len(normalized))]
    assert frontend.last_plan["text"] == "Use one point five milliseconds."
