"""
tests/test_vits2_reading_language.py — what the VITS2 frontend actually says.

Like test_vits2_frontend.py these need jieba/pypinyin/wetext/g2p_en plus the
installed model release (frontend_data, tn_cache, nltk_data, and the compiled
pronunciation dictionary), so they run on a Jetson with the release in place
and skip on a dev host:

    docker exec -w /work embodied-perception python3 -m pytest \\
        /work/tests/test_vits2_reading_language.py -q

or anywhere with the frontend deps and an unpacked release:

    VITS2_MODEL_DIR=/path/to/release python -m pytest \\
        perception/tests/test_vits2_reading_language.py -q

Every expected string here was read and judged by a person, not snapshotted.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

import vision_stubs  # noqa: F401  (installs ROS stubs, puts perception on sys.path)

MODEL_DIR = Path(os.getenv("VITS2_MODEL_DIR", "/models/vits2"))

# Presence via find_spec, never import — see test_vits2_frontend.py for why.
_MISSING = [
    name for name in
    ("jieba", "pypinyin", "wetext", "kaldifst", "g2p_en", "inflect", "nltk")
    if importlib.util.find_spec(name) is None
]
if _MISSING:
    pytest.skip(f"VITS2 frontend dependencies missing: {', '.join(_MISSING)}",
                allow_module_level=True)
if not (MODEL_DIR / "frontend_data" / "external_lexicon.json").is_file():
    pytest.skip(f"no VITS2 release with a pronunciation dictionary at {MODEL_DIR}",
                allow_module_level=True)

from plugins.vits2_tts_trt.frontend.release_paths import (  # noqa: E402
    configure_release_paths,
)

configure_release_paths(MODEL_DIR)

from plugins.vits2_tts_trt.frontend.cleaner import normalize_text_mix  # noqa: E402
from plugins.vits2_tts_trt.frontend.language import prepare, resolve  # noqa: E402
from plugins.vits2_tts_trt.lexicon_resources import load_external_lexicon  # noqa: E402


def _say(text, preference="auto", previous=None):
    """The exact text the G2P sees, through the same two steps as the plugin."""
    frontend_text, reading, _ = prepare(text, preference, previous)
    if reading:
        return normalize_text_mix(frontend_text, reading_language=reading)
    return normalize_text_mix(frontend_text)


@pytest.mark.parametrize("text,preference,previous,expected", [
    # Notation with no script of its own follows the setting.
    ("2026 10/20", "auto", None, "二零二六年十月二十日"),
    ("2026 10/20", "en", None, "October twentieth, two thousand twenty six"),
    ("2/3", "zh", None, "三分之二"),
    ("2/3", "en", None, "two thirds"),
    ("y = 2x-3", "zh", None, "y 等于二乘以 x 减三"),
    ("y = 2x-3", "en", None, "y equals two times x minus three"),
    ("RTX 4090", "zh", None, "R T X 四零九零"),
    ("RTX 4090", "en", None, "R T X forty ninety"),
    ("Qwen3", "zh", None, "千问三"),
    ("Qwen3", "en", None, "Qwen three"),
    ("SNR≥28dB", "zh", None, "S N R 大于等于二十八分贝"),
    # Auto reads by the sentence around the notation.
    ("Tell me about Qwen3", "auto", None, "Tell me about Qwen three"),
    ("最后四位是 5690", "auto", None, "最后四位是五六九零"),
    # ...and falls back on the previous utterance when there is no sentence.
    ("2026 10/20", "auto", "en", "October twentieth, two thousand twenty six"),
    # Micro units: the PR's English-word splitter used to cut "m" off "5μm",
    # leaving the TN graph a bare "5μ" it read as 五十缪.
    ("精度是5μm", "auto", None, "精度是五微米"),
    ("精度是5μm", "en", None, "精度是 five micrometers"),
    # A clock time, not a duration ("three hours thirty minutes").
    ("The meeting is at 3:30 pm.", "auto", None, "The meeting is at three thirty pm."),
    ("Wake me at 7:05.", "auto", None, "Wake me at seven oh five ."),
    # The released TN graph deletes 啊; the frontend keeps it.
    ("你好啊", "auto", None, "你好啊"),
])
def test_reading(text, preference, previous, expected):
    assert _say(text, preference, previous) == expected


@pytest.mark.parametrize("text,previous,expected", [
    ("Hello, please check the battery.", None, "en"),
    ("电池电量还有百分之八十", None, "zh"),
    ("你好，Qwen3 is ready，然后继续。", None, "zh"),
    ("2026 10/20", None, "zh"),
    ("2026 10/20", "en", "en"),
    # A sentence's own evidence beats the previous utterance.
    ("电池电量还有百分之八十", "en", "zh"),
])
def test_auto_resolution(text, previous, expected):
    assert resolve(text, "auto", previous) == expected


def test_a_fixed_preference_always_wins():
    assert resolve("Hello there", "zh") == "zh"
    assert resolve("你好", "en") == "en"


@pytest.mark.parametrize("text", [
    "こんにちは，你好", "Привет 你好", "😊好的", "μ是希腊字母", "2026-02-30",
])
@pytest.mark.parametrize("preference", ["auto", "zh", "en"])
def test_unsupported_input_is_read_not_refused(text, preference):
    """Characters the model cannot say are dropped by the cleaner as before;
    the rest of the sentence is still spoken. An earlier version of this change
    refused the whole utterance over one kana — the robot went silent."""
    assert _say(text, preference).strip()


def test_unknown_preference_is_an_error():
    with pytest.raises(ValueError):
        prepare("你好", "ja")


def test_the_release_dictionary_loads_and_matches_its_cmu_data():
    assert load_external_lexicon(MODEL_DIR).name == "external_lexicon.json"


def test_a_dictionary_built_for_other_cmu_data_is_refused(tmp_path):
    """A mismatched dictionary would silently mispronounce; fail the load."""
    root = tmp_path / "release"
    (root / "frontend_data").mkdir(parents=True)
    shutil.copy(MODEL_DIR / "frontend_data" / "external_lexicon.json",
                root / "frontend_data" / "external_lexicon.json")
    (root / "frontend_data" / "cmudict.rep").write_text("HELLO  HH AH0 L OW1\n")
    configure_release_paths(root)
    try:
        with pytest.raises(ValueError, match="CMU"):
            load_external_lexicon(root)
    finally:
        configure_release_paths(MODEL_DIR)
        load_external_lexicon(MODEL_DIR)


def test_a_release_without_its_dictionary_fails_loudly(tmp_path):
    root = tmp_path / "release"
    (root / "frontend_data").mkdir(parents=True)
    shutil.copy(MODEL_DIR / "frontend_data" / "cmudict.rep",
                root / "frontend_data" / "cmudict.rep")
    configure_release_paths(root)
    try:
        with pytest.raises(FileNotFoundError, match="dictionary"):
            load_external_lexicon(root)
    finally:
        configure_release_paths(MODEL_DIR)
