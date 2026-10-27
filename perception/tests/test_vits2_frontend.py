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
