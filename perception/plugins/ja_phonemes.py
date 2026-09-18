#!/usr/bin/env python3
"""
plugins/ja_phonemes.py — Japanese kana to Kokoro's own phoneme alphabet.

Why this exists, in one line: **Kokoro was trained on misaki's phonemes, and
sherpa-onnx drives it with espeak's.** The two alphabets disagree, sherpa silently
discards whatever it cannot look up, and Japanese came out full of holes — 12
phonemes dropped from a single sentence.

The earlier workaround romanised the text and let a Latin-script voice read it. That
dropped nothing and was intelligible, but it sounded like a non-native speaker,
because Italian phonemes are not Japanese ones. This module produces the *actual*
phonemes the model was trained on. Feeding them to it needs plugins/kokoro_direct,
since sherpa-onnx has no phoneme input path.

Everything here is pure text and is testable without a model, a device, or janome.
That matters: the failure mode is silence and gaps, which nobody notices in a diff.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


# ── misaki-compatible Japanese phonemes ──────────────────────────────────────
#
# Ported from misaki's `ja.py` at commit fdc9c5e5e (2025-01-13), the revision
# contemporary with Kokoro v1.0 — which is the release we ship. Apache-2.0, same as
# this repo's other vendored tables.
#
# The version matters more than anything else here. misaki's phoneme set has grown
# since, and `pip install misaki` today gives a table built for a **newer** Kokoro:
#
#   misaki 2025-01-13   31 distinct phonemes   0 missing from our vocab
#   misaki 2025-04-05   39 distinct phonemes  12 missing (G K g ƫ ᶀ ᶁ ᶃ ᶄ ᶆ ᶈ ᶉ)
#
# Twelve missing phonemes is exactly the failure this whole module exists to avoid,
# so the table is vendored and pinned rather than imported. Every character below is
# asserted against the model's own `tokens.txt` in tests/test_ja_phonemes.py.
#
# Keys are **hiragana** (misaki's own convention), so katakana readings from Janome
# are converted down first. Values are IPA in Kokoro's vocabulary.

_MORA_SINGLE = {
    "ぁ": "a", "あ": "a", "ぃ": "i", "い": "i", "ぅ": "ɯ", "う": "ɯ",
    "ぇ": "e", "え": "e", "ぉ": "o", "お": "o", "か": "ka", "が": "ɡa",
    "き": "kʲi", "ぎ": "ɡʲi", "く": "kɯ", "ぐ": "ɡɯ", "け": "ke", "げ": "ɡe",
    "こ": "ko", "ご": "ɡo", "さ": "sa", "ざ": "ʣa", "し": "ɕi", "じ": "ʥi",
    "す": "sɨ", "ず": "zɨ", "せ": "se", "ぜ": "ʣe", "そ": "so", "ぞ": "ʣo",
    "た": "ta", "だ": "da", "ち": "ʨi", "ぢ": "ʥi", "つ": "ʦɨ", "づ": "zɨ",
    "て": "te", "で": "de", "と": "to", "ど": "do", "な": "na", "に": "ɲi",
    "ぬ": "nɯ", "ね": "ne", "の": "no", "は": "ha", "ば": "ba", "ぱ": "pa",
    "ひ": "çi", "び": "bʲi", "ぴ": "pʲi", "ふ": "ɸɯ", "ぶ": "bɯ", "ぷ": "pɯ",
    "へ": "he", "べ": "be", "ぺ": "pe", "ほ": "ho", "ぼ": "bo", "ぽ": "po",
    "ま": "ma", "み": "mʲi", "む": "mɯ", "め": "me", "も": "mo", "ゃ": "ja",
    "や": "ja", "ゅ": "jɯ", "ゆ": "jɯ", "ょ": "jo", "よ": "jo", "ら": "ɾa",
    "り": "ɾʲi", "る": "ɾɯ", "れ": "ɾe", "ろ": "ɾo", "ゎ": "βa", "わ": "βa",
    "ゐ": "i", "ゑ": "e", "を": "o", "ゔ": "vɯ", "ゕ": "ka", "ゖ": "ke",
    "ヷ": "va", "ヸ": "vʲi", "ヹ": "ve", "ヺ": "vo",
}

# Digraphs: a base kana plus a small kana (sutegana), which form one mora together.
# Matched before singles, longest-first.
_MORA_DIGRAPH = {
    "いぇ": "je", "うぃ": "βi", "うぇ": "βe", "うぉ": "βo", "きぇ": "kʲe",
    "きゃ": "kʲa", "きゅ": "kʲɨ", "きょ": "kʲo", "ぎゃ": "ɡʲa", "ぎゅ": "ɡʲɨ",
    "ぎょ": "ɡʲo", "くぁ": "kᵝa", "くぃ": "kᵝi", "くぇ": "kᵝe", "くぉ": "kᵝo",
    "ぐぁ": "ɡᵝa", "ぐぃ": "ɡᵝi", "ぐぇ": "ɡᵝe", "ぐぉ": "ɡᵝo", "しぇ": "ɕe",
    "しゃ": "ɕa", "しゅ": "ɕɨ", "しょ": "ɕo", "じぇ": "ʥe", "じゃ": "ʥa",
    "じゅ": "ʥɨ", "じょ": "ʥo", "ちぇ": "ʨe", "ちゃ": "ʨa", "ちゅ": "ʨɨ",
    "ちょ": "ʨo", "ぢゃ": "ʥa", "ぢゅ": "ʥɨ", "ぢょ": "ʥo", "つぁ": "ʦa",
    "つぃ": "ʦʲi", "つぇ": "ʦe", "つぉ": "ʦo", "てぃ": "tʲi", "てゅ": "tʲɨ",
    "でぃ": "dʲi", "でゅ": "dʲɨ", "とぅ": "tɯ", "どぅ": "dɯ", "にぇ": "ɲe",
    "にゃ": "ɲa", "にゅ": "ɲɨ", "にょ": "ɲo", "ひぇ": "çe", "ひゃ": "ça",
    "ひゅ": "çɨ", "ひょ": "ço", "びゃ": "bʲa", "びゅ": "bʲɨ", "びょ": "bʲo",
    "ぴゃ": "pʲa", "ぴゅ": "pʲɨ", "ぴょ": "pʲo", "ふぁ": "ɸa", "ふぃ": "ɸʲi",
    "ふぇ": "ɸe", "ふぉ": "ɸo", "ふゅ": "ɸʲɨ", "ふょ": "ɸʲo", "みゃ": "mʲa",
    "みゅ": "mʲɨ", "みょ": "mʲo", "りゃ": "ɾʲa", "りゅ": "ɾʲɨ", "りょ": "ɾʲo",
    "ゔぁ": "va", "ゔぃ": "vʲi", "ゔぇ": "ve", "ゔぉ": "vo", "ゔゅ": "bʲɨ",
    "ゔょ": "bʲo",
}

# Not in the table, because each depends on what follows it.
_SOKUON = "っ"      # -> glottal stop
_CHOONPU = "ー"     # -> length mark on the previous vowel
_MORAIC_N = "ん"    # -> assimilates to the following consonant

# ん assimilates, and misaki spells the rule out; keeping it explicit rather than
# deriving it, because the categories are phonological, not orthographic.
#   m  before m, p, b        ŋ  before k, g
#   ɲ  before ɲ, ʨ, ʥ        n  before n, t, d, ɾ, z
#   ɴ  otherwise (utterance-final, before vowels and fricatives)
_N_BEFORE_LABIAL = "mpb"
_N_BEFORE_VELAR = "kɡ"
_N_BEFORE_PALATAL = ("ɲ", "ʨ", "ʥ")
_N_BEFORE_ALVEOLAR = "ntdɾz"

# A word or phrase boundary. ん before one of these is utterance-final for
# assimilation purposes and stays uvular.
_BOUNDARY = frozenset(' \t\n、。！？，．・「」『』（）,.!?;:()"“”')

# CJK punctuation mapped onto the tokens the model actually has. Kokoro's vocabulary
# contains `; : , . ! ? — … " ( ) “ ”` and a space, and nothing else — so anything
# left as its full-width form would be looked up, missed, and dropped, which is the
# same silent failure this module exists to prevent. `・` becomes a space because it
# separates the parts of a name (シャオ・ファン) and is a boundary, not a sound.
_PUNCT = {
    "、": ",", "，": ",", "。": ".", "．": ".", "！": "!", "？": "?",
    "：": ":", "；": ";", "（": "(", "）": ")",
    "「": "“", "」": "”", "『": "“", "』": "”", "〈": "“", "〉": "”",
    "《": "“", "》": "”", "【": "(", "】": ")",
    "・": " ", "〜": "—", "ー": "ː",
}


def _to_hiragana(text: str) -> str:
    """Katakana -> hiragana. The table is keyed on hiragana, as misaki's is."""
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
                   for c in text)


def split_moras(kana: str) -> list:
    """Split kana into moras, longest match first.

    Digraphs are two characters that form one mora (`きゃ` = kya, one beat, not two),
    so they must be matched before the singles or `き` would consume the `き` and
    leave `ゃ` stranded.
    """
    kana = _to_hiragana(kana)
    moras = []
    i = 0
    while i < len(kana):
        pair = kana[i:i + 2]
        if pair in _MORA_DIGRAPH:
            moras.append(pair)
            i += 2
            continue
        moras.append(kana[i])
        i += 1
    return moras


def _phoneme_for(mora: str) -> str:
    return _MORA_DIGRAPH.get(mora) or _MORA_SINGLE.get(mora, "")


def _moraic_n(next_phonemes: str) -> str:
    """ん takes the place of articulation of whatever follows it.

    Utterance-final and before a vowel it is the uvular ɴ, which is why the fallback
    is not simply `n`.
    """
    if next_phonemes:
        head = next_phonemes[0]
        if head in _N_BEFORE_LABIAL:
            return "m"
        if head in _N_BEFORE_VELAR:
            return "ŋ"
        if any(next_phonemes.startswith(p) for p in _N_BEFORE_PALATAL):
            return "ɲ"
        if head in _N_BEFORE_ALVEOLAR:
            return "n"
    return "ɴ"


def kana_to_phonemes(kana: str) -> str:
    """Kana -> a Kokoro phoneme string.

    Non-kana characters (punctuation, Latin, digits) pass through untouched; the
    caller has already normalised numbers and dates, and punctuation is in the
    model's vocabulary as its own tokens.
    """
    if not kana:
        return ""
    moras = split_moras(kana)
    out = []
    for i, mora in enumerate(moras):
        if mora == _SOKUON:
            # The geminate. misaki writes it as a glottal stop rather than doubling
            # the next consonant, and the model was trained that way.
            out.append("ʔ")
            continue
        if mora == _CHOONPU:
            # Length mark on whatever vowel preceded it. Dropped when it leads,
            # since there is nothing to lengthen.
            if out:
                out.append("ː")
            continue
        if mora == _MORAIC_N:
            # Look only at the *next* mora, and only within the word. misaki
            # assimilates per word (its `_romaji_word` never sees the next one), so
            # scanning past a space or punctuation would turn a word-final ん into
            # whatever the next word starts with — `ねん じゅう` came out `neɲ` where
            # it should be `neɴ`.
            nxt = ""
            if i + 1 < len(moras):
                candidate = moras[i + 1]
                if candidate not in _BOUNDARY:
                    nxt = _phoneme_for(candidate)
            out.append(_moraic_n(nxt))
            continue
        phonemes = _phoneme_for(mora)
        if phonemes:
            out.append(phonemes)
            continue
        mapped = _PUNCT.get(mora)
        if mapped is not None:
            out.append(mapped)
            continue
        # Latin, digits and anything already in the vocabulary passes through; the
        # caller has normalised numbers and dates before this point.
        out.append(mora)
    # The date rules and Janome's token joining both insert spaces; runs of them
    # would be separate tokens and read as separate pauses.
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def phoneme_chars(text: str) -> set:
    """Every distinct character in a phoneme string, for vocabulary checks."""
    return set(text)


def all_table_phonemes() -> set:
    """Every character the table can ever emit.

    Used by the test that asserts the whole table is representable in the model's
    vocabulary — the check whose absence caused this module to be needed.
    """
    chars = set()
    for value in list(_MORA_SINGLE.values()) + list(_MORA_DIGRAPH.values()):
        chars.update(value)
    chars.update("ʔːɴmŋɲn")     # the four context-dependent outputs above
    return chars
