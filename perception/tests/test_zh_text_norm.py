"""
tests/test_zh_text_norm.py — which numbers count as Chinese, and which do not.

The bug this guards: sherpa-onnx applies `rule_fsts` to the whole text *before* its
frontend decides what is Chinese, so handing it the ZH number FSTs rewrote every
digit into Chinese characters and the frontend then read them in Chinese — in
English, Spanish, Japanese, everything. `2026` came out as `二千零二十六` in an
English sentence.

The fix decides per number from its neighbours. That decision is pure text, so it is
testable here with no kaldifst, no model and no device — which matters, because the
alternative way to find a mistake in it is to listen to nine languages.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import zh_text_norm  # noqa: E402


def _chinese_parts(text: str):
    return [chunk for is_zh, chunk in zh_text_norm.segment(text) if is_zh]


def _has_chinese_number(text: str) -> bool:
    """True when at least one digit ends up in a Chinese segment."""
    return any(any(c.isdigit() for c in chunk) for chunk in _chinese_parts(text))


# ── the segmentation is lossless ──────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "", "hello", "你好", "2026", "   ",
    "Please follow me to the next exhibit, which opened in 2026.",
    "今天是2026年1月15日，有25个展品。",
    "agent core 延迟 200 毫秒。",
    "Hola, tenemos 25 exposiciones hoy.",
    "こんにちは、25の展示があります。",
])
def test_segments_reassemble_to_the_input(text):
    """Every byte must survive; a normaliser that drops text is worse than the bug."""
    assert "".join(chunk for _, chunk in zh_text_norm.segment(text)) == text


def test_empty_input_produces_no_segments():
    assert zh_text_norm.segment("") == []


# ── numbers that must NOT become Chinese ──────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Please follow me to the next exhibit, which opened in 2026.",
    "We have 25 exhibits today.",
    "Hola, tenemos 25 exposiciones hoy.",         # the Spanish case
    "Bonjour, nous avons 25 expositions.",
    "The meeting is on 2026-01-15.",
    "delay 200 ms",
    "Room 101, floor 3.",
    "2026",                                       # bare, no context at all
    "25 exhibits",                                # number leads the sentence
    "exhibits 25",                                # number ends the sentence
])
def test_numbers_in_non_chinese_text_stay_untouched(text):
    """This is the regression. Each of these read the number in Chinese before."""
    assert not _has_chinese_number(text), (
        f"{text!r} put a digit in a Chinese segment: {_chinese_parts(text)}")


def test_a_purely_english_sentence_is_one_non_chinese_segment():
    assert zh_text_norm.segment("We have 25 exhibits today.") == [
        (False, "We have 25 exhibits today.")]


# ── numbers that MUST become Chinese ──────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "今天是2026年1月15日",
    "第25个展品",
    "延迟200毫秒",
    "我在2026年出生",
    "有 25 个",              # spaces must not break the adjacency
    "延迟 200 毫秒",
])
def test_numbers_in_chinese_text_are_marked_chinese(text):
    assert _has_chinese_number(text), (
        f"{text!r} left its digits outside the Chinese segments: "
        f"{zh_text_norm.segment(text)}")


@pytest.mark.parametrize("text,expect_chinese", [
    # Chinese on both sides.
    ("agent core 延迟 200 毫秒", True),
    # Chinese only before the number.
    ("共25 items", True),
    # Chinese only after the number.
    ("In 2026 我们开业", True),
    # Latin on both sides, inside an otherwise Chinese paragraph.
    ("我们说 we have 25 exhibits 就这样", False),
])
def test_mixed_text_decides_per_number(text, expect_chinese):
    """Either-side-Chinese wins; that is what makes both directions work.

    The last case is the one a language-based gate could never get right: the
    sentence is Chinese, but that particular number belongs to the English clause.
    """
    assert _has_chinese_number(text) is expect_chinese, zh_text_norm.segment(text)


def test_full_width_punctuation_counts_as_chinese_context():
    """"有25个。" ends in a Chinese full stop, not a neutral one."""
    assert _has_chinese_number("2026年。")
    assert _has_chinese_number("有25个，很多。")


@pytest.mark.parametrize("text", [
    "a fundamental transition—from model-driven to a new engine",
    "Wait… what happened?",
    'He said “hello” to me.',
    "It's a well-known fact — really.",
])
def test_english_typography_does_not_make_a_sentence_chinese(text):
    """The em dash, ellipsis and curly quotes are English punctuation too.

    They were in _CJK_PUNCT at first, which split an English sentence in three and
    handed the bare dash to the ZH FSTs as "Chinese". It happened to be harmless —
    the FSTs leave punctuation alone — but it made English text look Chinese to
    anyone reading the segmentation, and an FST that did rewrite punctuation would
    have made it a real bug.
    """
    assert zh_text_norm.segment(text) == [(False, text)], zh_text_norm.segment(text)


def test_the_cjk_punctuation_set_holds_only_cjk_only_characters():
    """Guards the rule rather than the four cases above."""
    for ch in "—…“”‘’·":
        assert ch not in zh_text_norm._CJK_PUNCT, f"{ch!r} is used in English too"
    for ch in "。，、；：？！":
        assert ch in zh_text_norm._CJK_PUNCT, f"{ch!r} only appears in CJK text"


# ── Japanese must not be treated as Chinese ───────────────────────────────────

@pytest.mark.parametrize("text", [
    "今日は2026年10月1日です",
    "こんにちは！今日は2026年10月1日です。私はシャオ・ファンと申します。",
    "こんにちは、25の展示があります。",
    "第1話",                       # kanji + digit, but kana-free? no: no kana here
])
def test_japanese_numbers_are_left_for_the_japanese_frontend(text):
    """Kanji are inside the CJK range, so the adjacency rule would claim them.

    Left alone, "今日は2026年10月1日です" became "今日は二零二六年十月一日です" and
    was then read in Mandarin. Japanese dates want ジュウガツ / ツイタチ, which
    plugins/ja_text_norm.py produces; the ZH FSTs must not get there first.
    """
    if not zh_text_norm.has_kana(text):
        pytest.skip("no kana: this case is not distinguishable from Chinese")
    assert zh_text_norm.segment(text) == [(False, text)], zh_text_norm.segment(text)
    assert not _has_chinese_number(text)


def test_kana_detection():
    assert zh_text_norm.has_kana("こんにちは")          # hiragana
    assert zh_text_norm.has_kana("シャオ")              # katakana
    assert zh_text_norm.has_kana("今日は")              # mixed
    assert not zh_text_norm.has_kana("今天有25个展品")   # Chinese
    assert not zh_text_norm.has_kana("We have 25")
    assert not zh_text_norm.has_kana("")


def test_chinese_still_normalises_when_no_kana_present():
    """The kana guard must not disarm the Chinese path it sits next to."""
    assert _has_chinese_number("今天是2026年1月15日，有25个展品。")
    assert _has_chinese_number("延迟200毫秒")


# ── the FST is fed whole segments, not bare digits ────────────────────────────

class _RecordingNormalizer(zh_text_norm.ZhTextNormalizer):
    """Bypasses kaldifst so the wiring is testable without the dependency."""

    def __init__(self):
        self.seen = []
        self._normalizers = []
        self._available = True

    def _normalize_chinese(self, chunk):
        self.seen.append(chunk)
        return chunk.replace("2026", "二零二六").replace("25", "二十五")


def test_the_normaliser_only_sees_the_chinese_segments():
    n = _RecordingNormalizer()
    out = n.normalize("We have 25 exhibits. 今天有25个展品。")
    # The ". " between the two sentences stays with the English side: a neutral run
    # inherits the segment already open, so punctuation never opens a third piece.
    assert n.seen == ["今天有25个展品。"], n.seen
    assert out == "We have 25 exhibits. 今天有二十五个展品。"


def test_context_characters_reach_the_fst():
    """date-zh.fst needs the 年 to say 二零二六年 instead of 二千零二十六.

    Passing the bare digit run would silently downgrade every year to the quantity
    form, which is the kind of thing only a Chinese speaker would catch by ear.
    """
    n = _RecordingNormalizer()
    n.normalize("今天是2026年1月15日")
    assert n.seen == ["今天是2026年1月15日"]
    assert "年" in n.seen[0]


def test_a_normaliser_without_kaldifst_is_a_passthrough(tmp_path):
    """Degrades to a warning, not a refusal — see the class docstring."""
    n = zh_text_norm.ZhTextNormalizer(str(tmp_path))   # no FSTs on disk
    assert n.available is False
    assert n.normalize("今天有25个展品。") == "今天有25个展品。"
    assert n.normalize("") == ""


def test_normalize_is_a_noop_when_there_is_nothing_chinese():
    n = _RecordingNormalizer()
    assert n.normalize("We have 25 exhibits today.") == "We have 25 exhibits today."
    assert n.seen == []


def test_the_cjk_range_matches_what_sherpa_routes_on():
    """If these disagreed, text could be normalised here and routed to espeak.

    sherpa-onnx's KokoroMultiLangLexicon splits on "([\\u4e00-\\u9fff]+)".
    """
    assert zh_text_norm._CJK == "一-鿿"
    assert ord("一") == 0x4E00 and ord("鿿") == 0x9FFF


def test_no_adapter_hands_rule_fsts_to_sherpa_onnx():
    """The guard on the actual bug, checked in the source rather than by ear.

    Any non-empty `rule_fsts=` re-introduces it: sherpa applies those FSTs to the
    whole text before its frontend decides what is Chinese, so the digits are already
    Chinese characters by the time anything is routed. There is no way to scope them
    per language, which is why this module exists.

    Parsed as AST rather than grepped, so the comments explaining the rule — which
    necessarily contain the string `rule_fsts` — do not match.
    """
    import ast
    import inspect

    import plugins.tts as tts

    tree = ast.parse(inspect.getsource(tts))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "rule_fsts":
            continue
        value = node.value
        empty = isinstance(value, ast.Constant) and value.value == ""
        if not empty:
            offenders.append(ast.dump(value)[:120])
    assert not offenders, (
        "rule_fsts must be \"\" — the ZH FSTs are applied by plugins/zh_text_norm.py "
        f"so they only touch Chinese text. Found: {offenders}")


def test_both_sherpa_adapters_normalise_before_synthesising():
    """A rule_fsts="" with no replacement call would silently drop ZH numbers."""
    import inspect

    import plugins.tts as tts

    for adapter in (tts.MatchaTTSAdapter, tts.KokoroTTSAdapter):
        source = inspect.getsource(adapter.synthesize_stream)
        assert "_zh_norm.normalize" in source, (
            f"{adapter.__name__}.synthesize_stream does not normalise its text; "
            "Chinese numbers would reach the engine as bare digits")

