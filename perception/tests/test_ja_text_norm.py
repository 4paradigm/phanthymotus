"""
tests/test_ja_text_norm.py — Japanese numbers, dates and counters.

The rules here exist because a morphological analyser gets counters wrong: Janome
reads `10月` as `ツキ` and `1日` as `ニチ`, where a speaker says `ジュウガツ` and
`ツイタチ`. Everything in this file is pure text, so it runs with no janome, no
model and no device — which matters, because the alternative way to catch a wrong
reading is to find someone who reads Japanese.

The kanji->kana stage itself needs janome and is skipped without it.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import ja_text_norm as ja  # noqa: E402


# ── Sino-Japanese numerals ────────────────────────────────────────────────────

@pytest.mark.parametrize("n,expected", [
    (0, "ゼロ"),
    (1, "イチ"),
    (4, "ヨン"),
    (7, "ナナ"),
    (10, "ジュウ"),
    (11, "ジュウイチ"),
    (20, "ニジュウ"),
    (25, "ニジュウゴ"),
    (100, "ヒャク"),
    (200, "ニヒャク"),
    (1000, "セン"),
    (2026, "ニセンニジュウロク"),
])
def test_numbers_read_as_sino_japanese(n, expected):
    assert ja.read_number(n) == expected


@pytest.mark.parametrize("n,expected", [
    (300, "サンビャク"),   # not サンヒャク
    (600, "ロッピャク"),   # not ロクヒャク
    (800, "ハッピャク"),
    (3000, "サンゼン"),    # not サンセン
    (8000, "ハッセン"),
])
def test_the_euphonic_irregulars_are_not_derived_naively(n, expected):
    """These sound changes are why the hundreds and thousands need a table."""
    assert ja.read_number(n) == expected


def test_numbers_group_by_four_digits_not_three():
    """Japanese groups in 万/億, not thousands.

    Porting a Western number reader and grouping by three is the classic way to get
    this wrong; 20000 is ニマン, never "twenty thousand" spelled out in セン.
    """
    assert ja.read_number(10000) == "イチマン"
    assert ja.read_number(20000) == "ニマン"
    assert ja.read_number(100000000) == "イチオク"


# ── dates: the case that started this ─────────────────────────────────────────

def test_the_reported_date_reads_correctly():
    """2026年10月1日 was read in Mandarin, then as ツキ/ニチ. Both are wrong."""
    got = ja.normalize_dates("2026年10月1日")
    assert got == "ニセンニジュウロクネン ジュウガツ ツイタチ", got
    assert "ツキ" not in got, "月 must be ガツ in a date, not the isolated ツキ"
    assert "1ニチ" not in got


@pytest.mark.parametrize("n,expected", [
    (1, "イチガツ"),
    (4, "シガツ"),        # not ヨンガツ
    (7, "シチガツ"),      # not ナナガツ
    (9, "クガツ"),        # not キュウガツ
    (10, "ジュウガツ"),
    (12, "ジュウニガツ"),
])
def test_month_readings_including_the_three_irregulars(n, expected):
    assert ja.normalize_dates(f"{n}月") == expected


@pytest.mark.parametrize("n,expected", [
    (1, "ツイタチ"),
    (2, "フツカ"),
    (4, "ヨッカ"),
    (8, "ヨウカ"),
    (10, "トオカ"),
    (14, "ジュウヨッカ"),
    (20, "ハツカ"),
    (24, "ニジュウヨッカ"),
    (15, "ジュウゴニチ"),      # regular
    (28, "ニジュウハチニチ"),  # regular despite 8 being irregular alone
])
def test_day_readings_are_native_for_the_first_ten_and_14_20_24(n, expected):
    assert ja.normalize_dates(f"{n}日") == expected


def test_a_full_date_is_consumed_before_the_single_counters_match():
    """Ordering matters: 年月日 must not be eaten piecemeal by the 年 rule."""
    assert ja.normalize_dates("2026年1月15日") == "ニセンニジュウロクネン イチガツ ジュウゴニチ"
    assert ja.normalize_dates("10月1日") == "ジュウガツ ツイタチ"


def test_times_use_their_own_irregulars():
    assert ja.normalize_dates("4時") == "ヨジ"       # not ヨンジ
    assert ja.normalize_dates("9時") == "クジ"
    assert ja.normalize_dates("3分") == "サンプン"   # not サンフン
    assert ja.normalize_dates("10分") == "ジュップン"
    assert ja.normalize_dates("4時30分") == "ヨジ サンジュップン"


@pytest.mark.parametrize("n,expected", [
    (4, "ヨジ"), (7, "シチジ"), (9, "クジ"),
    (14, "ジュウヨジ"),      # not ジュウヨンジ — the irregular is on the ones digit
    (17, "ジュウシチジ"),
    (19, "ジュウクジ"),
    (12, "ジュウニジ"),      # regular
    (20, "ニジュウジ"),
])
def test_hour_irregulars_follow_the_trailing_digit(n, expected):
    assert ja.normalize_dates(f"{n}時") == expected


@pytest.mark.parametrize("n,expected", [
    (1, "イップン"), (3, "サンプン"), (6, "ロップン"), (8, "ハップン"),
    (2, "ニフン"), (5, "ゴフン"),
    (10, "ジュップン"), (20, "ニジュップン"), (30, "サンジュップン"),
    (21, "ニジュウイップン"),   # tens keep ジュウ, ones take the irregular
    (34, "サンジュウヨンプン"),
    (45, "ヨンジュウゴフン"),
])
def test_minute_irregulars_follow_the_trailing_digit(n, expected):
    """A flat table keyed on the whole value gave サンジュウフン for 30分."""
    assert ja.normalize_dates(f"{n}分") == expected


def test_spacing_between_number_and_counter_is_tolerated():
    assert ja.normalize_dates("2026 年 10 月 1 日") == "ニセンニジュウロクネン ジュウガツ ツイタチ"


# ── leftover numbers ──────────────────────────────────────────────────────────

def test_bare_numbers_are_read_only_after_the_counters():
    """Run in the other order and a date's digits get read twice."""
    text = ja.normalize_dates("今日は2026年10月1日で、25個あります")
    assert "2026" not in text, "the year should already be kana"
    assert ja.normalize_numbers(text).count("ニジュウゴ") == 1


def test_normalize_numbers_leaves_kana_alone():
    assert ja.normalize_numbers("ジュウガツ") == "ジュウガツ"
    assert ja.normalize_numbers("25") == "ニジュウゴ"


# ── the kanji stage (needs janome) ────────────────────────────────────────────

_HAS_JANOME = importlib.util.find_spec("janome") is not None
janome_only = pytest.mark.skipif(not _HAS_JANOME, reason="janome not installed")


def test_has_kanji_detects_what_would_reach_the_chinese_branch():
    """[一-鿿] is exactly the range sherpa routes to lexicon-zh.txt."""
    assert ja.has_kanji("今日")
    assert ja.has_kanji("キョウハ今日")
    assert not ja.has_kanji("キョウハニセンニジュウロクネン")
    assert not ja.has_kanji("こんにちは")
    assert not ja.has_kanji("")


# ── katakana -> romaji ────────────────────────────────────────────────────────
#
# Kana are never fed to the model: espeak-ja emits phonemes Kokoro's token table
# lacks (ʑ, U+0308, U+031E) and sherpa drops them, which is what made the first
# version unintelligible. These assertions are auditable by anyone who reads
# Japanese, which is the point — I cannot hear the result.

@pytest.mark.parametrize("kana,expected", [
    ("アイウエオ", "aiueo"),
    ("カキクケコ", "kakikukeko"),
    ("サシスセソ", "sashisuseso"),      # shi, not si
    ("タチツテト", "tachitsuteto"),     # chi and tsu, not ti/tu
    ("ハヒフヘホ", "hahifuheho"),       # fu, not hu
    ("ザジズゼゾ", "zajizuzezo"),       # ji, not zi
    ("ワヲン", "waon"),
])
def test_the_gojuon_uses_hepburn_spellings(kana, expected):
    """Hepburn, because that is what a Latin-script voice reads correctly."""
    assert ja.to_romaji(kana) == expected


@pytest.mark.parametrize("kana,expected", [
    ("キャキュキョ", "kyakyukyo"),
    ("シャシュショ", "shashusho"),
    ("チャチュチョ", "chachucho"),
    ("ジャジュジョ", "jajujo"),
    ("ファ", "fa"),          # loanword combinations, needed for names
    ("ティ", "ti"),
    ("シェ", "she"),
])
def test_digraphs_are_one_syllable(kana, expected):
    assert ja.to_romaji(kana) == expected


@pytest.mark.parametrize("kana,expected", [
    ("ニッキ", "nikki"),        # sokuon doubles the next consonant
    ("キッテ", "kitte"),
    ("イッショ", "issho"),
    ("ツイタチ", "tsuitachi"),  # no sokuon here; must not gain one
])
def test_sokuon_geminates_the_following_consonant(kana, expected):
    """Doubled consonants are why Italian is the natural target voice."""
    assert ja.to_romaji(kana) == expected


@pytest.mark.parametrize("kana,expected", [
    ("キョウ", "kyoo"),        # オウ is the ordinary long o — NOT kyou
    ("トウキョウ", "tookyoo"),
    ("ジュウ", "juu"),
    ("オオキイ", "ookii"),
    ("ラーメン", "raamen"),    # chounpu lengthens
    ("コーヒー", "koohii"),
])
def test_long_vowels_are_doubled_not_written_as_digraphs(kana, expected):
    """`kyou` reads as two syllables in Italian and Spanish; `kyoo` as one long one.

    This is the rule that decides whether the output sounds Japanese or like
    someone spelling out a transliteration.
    """
    assert ja.to_romaji(kana) == expected


def test_punctuation_and_digits_pass_through():
    assert ja.to_romaji("ABC123") == "ABC123"
    assert ja.to_romaji("") == ""
    # The interpunct between name parts becomes a space, or espeak runs them together.
    assert ja.to_romaji("シャオ・ファン") == "shao fan"


@pytest.mark.parametrize("kana,expected", [
    ("コンニチワ！", "konnichiwa!"),
    ("ソウデス。", "soodesu."),
    ("アレ、コレ", "are,kore"),
    ("ナニ？", "nani?"),
])
def test_full_width_punctuation_becomes_ascii(kana, expected):
    """espeak finds sentence boundaries in ASCII punctuation, not in 。！？.

    Left as full-width, the whole utterance is one breath group with no pauses.
    """
    assert ja.to_romaji(kana) == expected


def test_the_particle_ha_is_read_wa():
    """A dictionary gives the written kana; Japanese says something else.

    `今日は` is `kyoo wa`, never `kyoo ha`. The correction happens in to_kana where
    Janome's part-of-speech tags are available, so this test asserts the table those
    tags select from — the wiring itself needs janome and is covered below.
    """
    assert ja._PARTICLE_READINGS["ハ"] == "ワ"
    assert ja._PARTICLE_READINGS["ヘ"] == "エ"
    assert ja.to_romaji("キョウワ") == "kyoowa"
    assert ja._GREETINGS["コンニチハ"] == "コンニチワ"


def test_the_reported_sentence_romanises():
    """The whole pipeline's output, spelled out so a reader can check it."""
    kana = "コンニチハ！キョウハニセンニジュウロクネンジュウガツツイタチデス。"
    got = ja.to_romaji(kana)
    assert "kyoo" in got, got
    assert "nisen" in got and "nijuuroku" in got, got
    assert "juugatsu" in got, got
    assert "tsuitachi" in got, got
    assert not any(0x3040 <= ord(c) <= 0x30FF for c in got), f"kana survived: {got}"


def test_no_kana_survives_romanisation():
    """Anything left in kana would reach espeak-ja and lose phonemes again."""
    for kana in ["アイウエオ", "キャキュキョ", "ニッキ", "ラーメン",
                 "シャオ・ファン", "ヴィヴァ"]:
        got = ja.to_romaji(kana)
        assert not any(0x3040 <= ord(c) <= 0x30FF for c in got), (kana, got)



@janome_only
def test_the_reported_sentence_leaves_no_kanji():
    """The whole point: nothing may remain in [一-鿿], or it is read in Mandarin."""
    frontend = ja.JapaneseFrontend()
    out = frontend.normalize(
        "こんにちは！今日は2026年10月1日です。私はシャオ・ファンと申します。"
        "何かお手伝いしましょうか？")
    assert not ja.has_kanji(out), f"kanji survived: {out}"
    # Romaji, not katakana: `normalize` is the whole pipeline and its last stage
    # is `to_romaji` (see the note above it — feeding katakana to espeak-`ja`
    # spells the letters out instead of reading them). The katakana assertions
    # elsewhere in this file are on `normalize_dates`/`read_number`, which are
    # the intermediate stages and do return kana.
    assert "juugatsu" in out and "tsuitachi" in out, out


@janome_only
def test_kanji_become_kana():
    frontend = ja.JapaneseFrontend()
    assert not ja.has_kanji(frontend.normalize("私は人形ロボットです"))
    assert not ja.has_kanji(frontend.normalize("何かお手伝いしましょうか"))


@pytest.mark.skipif(_HAS_JANOME, reason="janome is installed here")
def test_without_janome_the_frontend_refuses_rather_than_degrading():
    """Unlike the Chinese normaliser, which warns and passes text through.

    Degrading here would not merely lose the numbers — every kanji would be spoken
    in Mandarin, so the card would come up `running` and speak the wrong language.
    Same call plugins/tts.py makes about pythainlp for Thai.
    """
    with pytest.raises(RuntimeError, match="janome"):
        ja.JapaneseFrontend()
