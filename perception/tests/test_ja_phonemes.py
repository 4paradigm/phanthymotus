"""
tests/test_ja_phonemes.py — the check whose absence caused the whole Japanese saga.

Kokoro was trained on misaki's phonemes; sherpa-onnx drives it with espeak's. The two
alphabets disagree, sherpa silently discards what it cannot look up, and Japanese came
out with 12 phonemes missing from a single sentence — audible as holes, and invisible
in any diff.

So the load-bearing test here is not "does it produce plausible output", it is **every
character this table can emit exists in the model's vocabulary**. That runs with no
model, no device and no janome.

The vocabulary is checked against a copy of Kokoro's own token list, so the assertion
does not depend on a downloaded release being present.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import ja_phonemes as jp  # noqa: E402

# Kokoro-82M's vocabulary, verified byte-identical to upstream's own config.json
# (114 tokens, zero difference) and to the tokens.txt inside the release we ship.
# Inlined rather than read from /models so this test needs no download.
KOKORO_VOCAB = set(
    ';:,.!?—…"()“” ̃ʣʥʦʨᵝꭧAIOQSTWYᵊabcdefhijk'
    'lmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳ'
    'ɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ↓→↗↘ᵻ'
)


def test_the_vocabulary_snapshot_is_the_right_size():
    """If this drifts, the snapshot above no longer describes the shipped model."""
    assert len(KOKORO_VOCAB) == 114, sorted(KOKORO_VOCAB)


# ── the assertion that matters ────────────────────────────────────────────────

def test_every_phoneme_the_table_can_emit_is_in_the_vocabulary():
    """The check that was missing. 12 phonemes were being dropped without it.

    `pip install misaki` today gives a table for a *newer* Kokoro whose vocabulary has
    G, K, g, ƫ and the palatalised ᶀᶁᶃᶄᶆᶈᶉ. Ours does not. If someone "updates" the
    vendored table to current misaki, this fails immediately instead of shipping
    silence.
    """
    missing = sorted(c for c in jp.all_table_phonemes() if c not in KOKORO_VOCAB)
    assert not missing, (
        f"these phonemes are not in Kokoro's vocabulary and would be silently "
        f"dropped: {missing}. The table is pinned to misaki fdc9c5e5e (2025-01-13) "
        f"for exactly this reason — see the comment in plugins/ja_phonemes.py")


@pytest.mark.parametrize("text", [
    "コンニチハ", "キョウハニセンニジュウロクネン", "ワタシハシャオ・ファントモウシマス",
    "ナニカオテツダイシマショウカ", "トウキョウ", "ラーメン", "ニッキ",
    "シャシュショ", "チャチュチョ", "ジャジュジョ", "ヴァヴィヴェヴォ",
])
def test_real_sentences_stay_in_vocabulary(text):
    out = jp.kana_to_phonemes(text)
    missing = sorted(c for c in out if c not in KOKORO_VOCAB)
    assert not missing, f"{text!r} -> {out!r} contains {missing}"


def test_no_kana_survives():
    """Anything left in kana would be looked up as a token and dropped."""
    for text in ["コンニチハ", "キョウ", "ラーメン", "シャオ・ファン", "ヴィヴァ"]:
        out = jp.kana_to_phonemes(text)
        assert not any(0x3040 <= ord(c) <= 0x30FF for c in out), (text, out)


# ── mora splitting ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kana,expected", [
    ("きゃ", ["きゃ"]),           # one mora, not two
    ("きや", ["き", "や"]),       # full-size や is its own mora
    ("しゃしゅしょ", ["しゃ", "しゅ", "しょ"]),
    ("にっき", ["に", "っ", "き"]),
    ("らーめん", ["ら", "ー", "め", "ん"]),
])
def test_digraphs_are_one_mora(kana, expected):
    """`きゃ` is one beat. Matching singles first would strand the small kana."""
    assert jp.split_moras(kana) == expected


def test_katakana_and_hiragana_give_the_same_phonemes():
    """The table is keyed on hiragana, as misaki's is; katakana converts down."""
    assert jp.kana_to_phonemes("コンニチハ") == jp.kana_to_phonemes("こんにちは")


# ── the three context-dependent moras ─────────────────────────────────────────

@pytest.mark.parametrize("kana,expected", [
    ("シンブン", "ɕimbɯɴ"),    # ん -> m before b
    ("サンポ", "sampo"),        # ん -> m before p
    ("ンゴ", "ŋɡo"),            # ん -> ŋ before ɡ
    ("サンニン", "saɲɲiɴ"),     # ん -> ɲ before ɲ
    ("ホンダ", "honda"),        # ん -> n before d
    ("ラーメン", "ɾaːmeɴ"),     # utterance-final -> uvular ɴ
])
def test_moraic_n_assimilates_to_what_follows(kana, expected):
    """ん takes the place of articulation of the next consonant.

    Five outcomes, and the fallback is the uvular ɴ rather than plain n — which is
    why the rule is spelled out rather than defaulted.
    """
    assert jp.kana_to_phonemes(kana) == expected


def test_sokuon_is_a_glottal_stop_not_a_doubled_consonant():
    """misaki writes っ as ʔ, and the model was trained that way.

    The romaji stage doubled the consonant instead, which was right for an Italian
    voice reading romaji and wrong here.
    """
    assert jp.kana_to_phonemes("ニッキ") == "ɲiʔkʲi"
    assert jp.kana_to_phonemes("キッテ") == "kʲiʔte"


def test_chounpu_lengthens_the_previous_vowel():
    assert jp.kana_to_phonemes("ラーメン") == "ɾaːmeɴ"
    assert jp.kana_to_phonemes("コーヒー") == "koːçiː"
    # Leading ー has nothing to lengthen and is dropped rather than emitted bare.
    assert not jp.kana_to_phonemes("ーア").startswith("ː")


def test_punctuation_becomes_tokens_the_model_actually_has():
    """Full-width 。、！？ are not in the vocabulary; their ASCII forms are.

    Left alone they would be looked up, missed and dropped — the same silent failure
    as the phonemes. `・` becomes a space because it separates name parts and is a
    boundary, not a sound.
    """
    out = jp.kana_to_phonemes("コンニチハ、ゲンキ？")
    assert "," in out and "?" in out, out
    assert "、" not in out and "？" not in out, out
    assert jp.kana_to_phonemes("シャオ・ファン") == "ɕao ɸaɴ"
    for text in ["コンニチハ、ゲンキ？", "ソウデス。", "エッ！", "「ハイ」"]:
        missing = sorted(c for c in jp.kana_to_phonemes(text) if c not in KOKORO_VOCAB)
        assert not missing, (text, missing)


def test_empty_input():
    assert jp.kana_to_phonemes("") == ""
    assert jp.split_moras("") == []
