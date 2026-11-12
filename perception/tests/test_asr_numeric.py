"""Numeric-span failures must not roll back the rest of a transcript."""
import pytest

from plugins.asr_numeric import NumericOutputNormalizer


@pytest.fixture
def normalizer(tmp_path):
    renderer = NumericOutputNormalizer(tmp_path)
    english = "zero one two three four five six seven eight nine".split()
    renderer._lexicon = {
        "digits": {"en": english, "th": "ศูนย์ หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า".split()},
        "words": {"en": english}, "extra": {"en": ""},
        "connectors": {"en": []}, "percent": {"th": "เปอร์เซ็นต์"},
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
