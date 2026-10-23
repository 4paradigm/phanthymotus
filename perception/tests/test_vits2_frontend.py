"""
tests/test_vits2_frontend.py — VITS2 text-frontend regressions.

Skipped unless the VITS2 frontend can actually be imported: it needs
jieba/pypinyin/wetext/g2p_en plus the installed model release (frontend_data,
tn_cache, nltk_data). That means these run on a Jetson with the release in
place and skip on a dev host.

On device:
    docker exec -w /work embodied-perception python3 -m pytest \
        /work/tests/test_vits2_frontend.py -q
or from the repo, with a release under /models/vits2:
    python -m pytest perception/tests/test_vits2_frontend.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

import vision_stubs  # noqa: F401  (installs ROS stubs, puts perception on sys.path)

MODEL_DIR = Path(os.getenv("VITS2_MODEL_DIR", "/models/vits2"))

# Presence is checked with find_spec, never by importing: on Python 3.8 `import
# wetext` fails until wetext_compat has patched importlib.resources.files (which
# frontend/fst_tn.py does before it touches wetext), and importing g2p_en calls
# nltk.download() unless nltk.data.path already points at the release. Both
# orderings belong to the frontend package — a test must not front-run them.
_MISSING = [
    name for name in
    ("jieba", "pypinyin", "wetext", "kaldifst", "g2p_en", "inflect", "nltk")
    if importlib.util.find_spec(name) is None
]
if _MISSING:
    pytest.skip(f"VITS2 frontend dependencies missing: {', '.join(_MISSING)}",
                allow_module_level=True)
if not (MODEL_DIR / "tn_cache").is_dir():
    pytest.skip(f"no VITS2 release at {MODEL_DIR}", allow_module_level=True)

from plugins.vits2_tts_trt.frontend.release_paths import (  # noqa: E402
    configure_release_paths,
)

configure_release_paths(MODEL_DIR)

from plugins.vits2_tts_trt.frontend import chinese  # noqa: E402
from plugins.vits2_tts_trt.frontend.chunking import (  # noqa: E402
    split_punctuation_units,
)
from plugins.vits2_tts_trt.frontend.cleaner import (  # noqa: E402
    g2p_normalized_text_mix,
    normalize_text_mix,
)


# The paragraph that failed on device: pasted prose with two ASCII spaces plus
# two ideographic spaces at each paragraph start, mixed with latin (BOT, G1,
# T800) so the whole input takes the mix_normalize() path.
DEVICE_TEXT = (
    "你好一、从“翻车”名场面到硬核技术展示  　　事件出圈：2026年1月，宁波一商场"
    "机器人表演武术时，一台机器人回旋踢失误将同伴踹倒，视频获近200万点赞，网友"
    "戏称“把自己干趴下”。  　　反差认知：这类“翻车”视频反而成为技术窗口——被踹"
    "机器人小碎步后退、尝试平衡、最终倒地，恰恰展现了动态平衡与跌倒保护算法的"
    "真实水平。  　　全球刷屏：2026年春晚《武BOT》节目中，宇树G1等机器人完成醉拳、"
    "双截棍、连续单腿后空翻等高难动作。"
)


def _units(text: str) -> list[str]:
    return [u for u in split_punctuation_units(normalize_text_mix(text)) if u.strip()]


@pytest.mark.parametrize("text,expected", [
    ("1kg", "one kilogram"),
    ("3.05kg", "three point zero five kilograms"),
    ("x=2kg", "x equals two kilograms"),
    ("2kg/m³", "two kilograms per cubic meter"),
])
def test_selected_english_unit_reading_survives_real_fst_and_g2p(text, expected):
    from plugins.vits2_tts_language.neutral_text import normalize_neutral_text

    normalized = normalize_text_mix(normalize_neutral_text(text, "en"))
    assert normalized == expected
    phones, tones, languages, _ = g2p_normalized_text_mix(normalized)
    assert phones and len(phones) == len(tones) == len(languages)
    assert set(languages) == {"EN"}


def test_every_unit_of_a_pasted_paragraph_converts():
    """A whitespace run used to abort the whole utterance.

    pypinyin collapses a run of non-Chinese characters into one element, so
    "  　　事件出圈," produced fewer word2ph entries than characters and
    chinese.g2p asserted — raised from token counting, so nothing was spoken at
    all. Every unit must convert.
    """
    units = _units(DEVICE_TEXT)
    assert len(units) > 10
    for unit in units:
        g2p_normalized_text_mix(unit)   # must not raise


@pytest.mark.parametrize("text", [
    "  　　事件出圈，",          # leading ASCII + ideographic spaces
    "全球 刷屏，",               # space between Chinese words
    "第一句。\n\n第二句。",      # newlines
    "　",                    # nothing but an ideographic space
])
def test_whitespace_shapes_convert(text):
    g2p_normalized_text_mix(normalize_text_mix(text))


def test_unvoiceable_characters_do_not_silence_the_rest(monkeypatch):
    """OOV must cost the character, never the utterance."""
    # The warning is asserted through the module logger rather than caplog: the
    # perception image configures logging handlers, and what matters here is
    # that the code reports the drop at all.
    warnings = []
    monkeypatch.setattr(chinese.log, "warning",
                        lambda msg, *args: warnings.append(msg % args if args else msg))

    phones, _, word2ph = chinese.g2p("扭矩四百五十牛✅米")

    assert len(phones) > 10, "the Chinese around the unknown character was dropped"
    assert any("cannot voice" in message for message in warnings), \
        "an unvoiceable character disappeared without a warning"
    assert any("✅" in message for message in warnings)


def test_digits_are_not_stripped_by_the_chinese_path():
    """A digit that survived TN must still be read, not silently deleted.

    Dispatching on "has no latin letters" sent such a chunk to the Chinese
    frontend, whose cleaner drops digits — the listener simply never heard them.
    """
    assert chinese.zh_speakable("第3名") is False
    phones, _, langs, _ = g2p_normalized_text_mix("第3名")
    assert "EN" in langs, "the digit was dropped instead of pronounced"
    assert len(phones) > 4


def test_pure_chinese_still_takes_the_chinese_path():
    assert chinese.zh_speakable("今天天气不错，") is True
    _, _, langs, _ = g2p_normalized_text_mix(normalize_text_mix("今天天气不错。"))
    assert set(langs) == {"ZH"}


@pytest.fixture(scope="module")
def production_text_adapter():
    """Use the real chunking/ID methods without constructing a CUDA engine."""
    from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter
    from plugins.vits2_tts_trt.runtime.backends.trt_numpy_tts_engine import TensorRTNumpyTTSEngine

    engine = TensorRTNumpyTTSEngine.__new__(TensorRTNumpyTTSEngine)
    engine.add_blank = bool(json.loads((MODEL_DIR / "config.json").read_text())["data"].get("add_blank", True))
    adapter = Vits2TensorRTAdapter.__new__(Vits2TensorRTAdapter)
    adapter._engine = engine
    adapter.max_chunk_tokens = 256
    return adapter


def _production_chunks(adapter, text):
    from plugins.vits2_tts_language.plugin import _direct_request, _prepare_chunks
    from plugins.vits2_tts_language.policy import LanguagePolicy

    request = _direct_request(LanguagePolicy("auto"), text)
    return _prepare_chunks(adapter, request).chunks


@pytest.mark.parametrize("text", [
    "NVIDIA uses Agentic.",
    "今天 NVIDIA 使用 Agentic。",
    "请读这句英文，Nvidia uses AGENTIC，然后继续。",
])
def test_verified_terms_survive_real_fst_in_production_language_routes(production_text_adapter, text):
    from plugins.vits2_tts_trt.frontend.symbols import symbols

    chunks = _production_chunks(production_text_adapter, text)
    normalized = " ".join(value for value, _ in chunks)
    assert "nvidia" in normalized.lower()
    assert "agentic" in normalized.lower()
    assert "英伟达" not in normalized
    all_phones, all_tones, all_languages = [], [], []
    for value, ids in chunks:
        phones, tones, languages, word2ph = g2p_normalized_text_mix(value)
        assert len(phones) == len(tones) == len(languages) == sum(word2ph)
        expected_ids = 2 * len(phones) + 1 if production_text_adapter._engine.add_blank else len(phones)
        assert len(ids[0]) == len(ids[1]) == len(ids[2]) == expected_ids
        assert all(phone in symbols and phone != "UNK" for phone in phones)
        all_phones.extend(phones)
        all_tones.extend(tones)
        all_languages.extend(languages)
    for expected_phones, expected_tones in (
            (["eh", "n", "V", "ih", "d", "iy", "ah"], [1, 3, 3, 2, 3, 1, 1]),
            (["ah", "jh", "eh", "n", "t", "ih", "k"], [1, 3, 2, 3, 3, 1, 3])):
        matches = [index for index in range(len(all_phones) - len(expected_phones) + 1)
                   if all_phones[index:index + len(expected_phones)] == expected_phones]
        assert len(matches) == 1
        start, end = matches[0], matches[0] + len(expected_phones)
        assert all_tones[start:end] == expected_tones
        assert all_languages[start:end] == ["EN"] * len(expected_phones)
        if expected_phones[-1] == "k":
            assert all_phones[end] in {".", ","}, "Agentic acquired an extra final phone"


def test_verified_terms_leave_real_numeric_and_address_spans_intact(production_text_adapter):
    text = ("NVIDIA uses 8.5GB. Agentic reads https://nvidia.com/Agentic?x=2 "
            "and agentic+nvidia_2@example.com.")
    normalized = " ".join(" ".join(
        value for value, _ in _production_chunks(production_text_adapter, text)
    ).split())
    assert "NVIDIA" in normalized
    assert "Agentic reads" in normalized
    assert "eight point five gigabytes" in normalized
    assert "N V I D I A dot C O M slash A G E N T I C question mark X equals two" in normalized
    assert "A G E N T I C plus N V I D I A underscore two at E X A M P L E dot C O M" in normalized
    assert not any(char.isdecimal() for char in normalized)


def test_uppercase_abbreviations_keep_letter_readings_through_real_fst(production_text_adapter):
    text = "The US team uses this IP address and order ID. IT support is available."
    chunks = _production_chunks(production_text_adapter, text)
    normalized = " ".join(value for value, _ in chunks)
    for phrase in ("U S team", "I P address", "order I D", "I T support"):
        assert phrase in normalized
    for value, _ in chunks:
        phones, tones, languages, word2ph = g2p_normalized_text_mix(value)
        assert len(phones) == len(tones) == len(languages) == sum(word2ph)
        assert "UNK" not in phones


@pytest.mark.parametrize("text,spelling,expected_phones", [
    ("NVIDIA supports Agentic AI.", "A I", ["ey", "ay"]),
    ("Please check the API.", "A P I", ["ey", "p", "iy", "ay"]),
    ("今天测试AI.", "A I", ["ey", "ay"]),
    ("今天测试AI。", "A I", ["ey", "ay"]),
    ("Please read A I.", "A I", ["ey", "ay"]),
    ("The CAR is ready.", "CAR", ["k", "aa", "r"]),
    ("Car is ready.", "Car", ["k", "aa", "r"]),
    ("今天 CAR 可以用 cash 支付。", "CAR", ["k", "aa", "r"]),
])
def test_letter_spellings_and_cmu_car_survive_real_fst(
        production_text_adapter, text, spelling, expected_phones):
    chunks = _production_chunks(production_text_adapter, text)
    normalized = " ".join(value for value, _ in chunks)
    assert spelling in normalized
    assert "CA R" not in normalized
    assert "一" not in normalized
    all_phones, all_languages = [], []
    for value, _ in chunks:
        phones, tones, languages, word2ph = g2p_normalized_text_mix(value)
        assert len(phones) == len(tones) == len(languages) == sum(word2ph)
        assert "UNK" not in phones
        all_phones.extend(phones)
        all_languages.extend(languages)
    matches = [i for i in range(len(all_phones) - len(expected_phones) + 1)
               if all_phones[i:i + len(expected_phones)] == expected_phones]
    assert len(matches) == 1
    start = matches[0]
    assert all_languages[start:start + len(expected_phones)] == ["EN"] * len(expected_phones)


def test_letter_spelling_protection_preserves_roman_number_and_unit_context():
    for roman, spoken in (("I.", "一."), ("II.", "二."), ("IV.", "四.")):
        assert normalize_text_mix(roman) == spoken
    normalized = normalize_text_mix("电流1.2A，日期2026-08-07，订单ID是0056。")
    assert "一点二安培" in normalized
    assert "二零二六年八月七日" in normalized
    assert "零零五六" in normalized


@pytest.mark.parametrize("text,expected", [
    ("AI\nNext step.", "AI.Nextstep."),
    ("今天学习AI\n明天测试API", "今天学习AI.明天测试API"),
    ("AI.\nII. Methods", "AI.二.Methods"),
    ("API.\r\nIV. Methods", "API.四.Methods"),
])
def test_protected_letters_keep_paragraph_boundaries(production_text_adapter, text, expected):
    chunks = _production_chunks(production_text_adapter, text)
    assert "".join("".join(value.split()) for value, _ in chunks) == expected
