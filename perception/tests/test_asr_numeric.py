"""Numeric-span failures must not roll back the rest of a transcript."""
import pytest

from plugins.asr_numeric import NumericOutputNormalizer


@pytest.fixture
def normalizer(tmp_path):
    renderer = NumericOutputNormalizer(tmp_path)
    english = "zero one two three four five six seven eight nine".split()
    renderer._lexicon = {
        "digits": {"en": english, "th": "ศูนย์ หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า".split(),
                   "zh": ["零 〇", "一 幺", "二 两", "三", "四", "五", "六", "七", "八", "九"],
                   "ja": ["零 〇", "一", "二", "三", "四", "五", "六", "七", "八", "九"]},
        "words": {"en": english}, "extra": {"en": ""},
        "connectors": {"en": []},
        "percent": {"th": "เปอร์เซ็นต์", "en": "percent", "zh": "%", "ja": "パーセント"},
    }
    return renderer


@pytest.mark.parametrize("source,expected", [
    ("ร้อยละสิบ", "10%"),
    ("ร้อยละสามจุดห้า", "3.5%"),
    ("ร้อยละศูนย์จุดห้า", "0.5%"),
    ("ร้อยละศูนย์สิบ", "ร้อยละศูนย์สิบ"),
    ("ร้อยละสามจุดสิบ", "ร้อยละสามจุดสิบ"),
    ("ร้อยละจุดห้า", "ร้อยละจุดห้า"),
])
def test_thai_percent_preserves_invalid_spans(normalizer, source, expected):
    assert normalizer.normalize(source, "th") == expected


def test_invalid_percent_keeps_other_number_conversions(normalizer, caplog):
    source = "จำนวน ๑๒ คน ร้อยละศูนย์สิบ และ ร้อยละสิบ"
    assert normalizer.normalize(source, "th") == "จำนวน 12 คน ร้อยละศูนย์สิบ และ 10%"
    assert "numeric normalization unavailable" not in caplog.text


def test_ordinary_thai_is_preserved(normalizer):
    text = "วันนี้อากาศดี"
    assert normalizer.normalize(text, "th") == text


@pytest.mark.parametrize("language,source,expected", [
    ("en", "zero zero one two three four five six seven", "001234567"),
    ("zh", "零零一二三四五六七", "001234567"),
    ("ja", "二〇二五年、93パーセント", "2025年、93%"),
    ("zh", "二〇二五年", "2025年"),
])
def test_digit_dictation_and_cjk_rendering(normalizer, language, source, expected):
    assert normalizer.normalize(source, language) == expected


def test_protected_names_versions_and_urls_do_not_enter_grammar(normalizer, monkeypatch):
    text = "Qwen3 uses GPT-4.1, model-v2 and version 3.10. https://example.com/v3 user02@example.com"
    monkeypatch.setattr(normalizer, "_rewrite", lambda *a: pytest.fail("protected text entered grammar"))
    assert normalizer.normalize(text, "en") == text


def test_rejected_grammar_preserves_recognized_transcript(normalizer, monkeypatch, caplog):
    def reject(*args):
        raise ValueError("rejected span")
    monkeypatch.setattr(normalizer, "_rewrite", reject)
    text = "keep one item"
    assert normalizer.normalize(text, "en") == text
    assert "preserving text" in caplog.text
