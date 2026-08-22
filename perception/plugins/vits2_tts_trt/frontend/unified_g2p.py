"""
Unified ZH/EN mixed grapheme-to-phoneme converter.

Chinese runs use the native Chinese frontend (jieba, release dictionaries and
ToneSandhi); English runs use the release-controlled English frontend.

Design decisions:
  1. Contiguous CJK runs follow the same native path as pure Chinese text, so
     light tones and connected-speech tone sandhi remain consistent.
  2. English words and explicitly spaced letters remain separate tokens.
  3. Explicit whitespace remains a zero-phone boundary.
  4. English phonemes use refine_ph() + post_replace_ph() from english.py so
     the output symbols match en_symbols in symbols.py (lowercase ARPAbet).
  5. Punctuation is remapped by the caller via
     english.replace_punctuation *before* this function is called, so
     only ASCII punctuation survives here (in the `punctuation` set).

Returns:
  phones  : List[str]  — phoneme symbols (with leading/trailing '_')
  tones   : List[int]  — raw tone values (ZH 0-5, EN 0-3); no offset applied
  langs   : List[str]  — 'ZH' or 'EN' per phone (for tone_start offset lookup)
  word2ph : List[int]  — phoneme count per input position (char/word)
"""

import os
import re

from g2p_en import G2p
from .symbols import punctuation
from .english import eng_dict, refine_ph, post_replace_ph, arpa

current_file_path = os.path.dirname(__file__)
with open(os.path.join(current_file_path, "opencpop-strict.txt"), encoding="utf-8") as _f:
    pinyin_to_symbol_map = {
        line.split("\t")[0]: line.strip().split("\t")[1]
        for line in _f
    }

_g2p_en = None

# Chinese initials — longest-first so 'zh'/'ch'/'sh' match before 'z'/'c'/'s'
_INITIALS = [
    "zh", "ch", "sh",
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "r", "z", "c", "s",
    "y", "w",
]

# ASCII punctuation that survives mix_normalize (same set as symbols.punctuation)
_PUNCT_SET = set(punctuation)
_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9'-]+")
_NON_ZH_PIECE_RE = re.compile(r"[A-Za-z0-9'-]+|[^\w\s]", re.UNICODE)
_MIX_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff]+|[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|\s+|.",
    re.DOTALL,
)
_LETTER_SPELLED_UNITS = {"KB", "MB", "GB", "TB", "PB", "KBPS", "MBPS", "GBPS"}
# Deterministic pronunciations missing from the stock CMU dictionary. Keep this
# tiny and release-controlled: training/export and deployed runtime must match.
_CUSTOM_EN_PRONUNCIATIONS = {
    "IELTS": [["AY1"], ["EH0", "L", "T", "S"]],
}


def _split_pinyin(tone3: str):
    """Split a TONE3 pinyin string into (initial, final_no_tone, tone_int).

    Examples:
        'ba1'  → ('b',  'a',  1)
        'zhi4' → ('zh', 'i',  4)
        'an5'  → ('',   'an', 5)   ← neutral tone
    """
    if not tone3 or tone3[-1] not in "12345":
        return "", tone3, 0
    tone = int(tone3[-1])
    pny = tone3[:-1]
    for ini in _INITIALS:
        if pny.startswith(ini):
            return ini, pny[len(ini):], tone
    return "", pny, tone


def _zh_pinyin_to_phones(tone3: str):
    """Convert a TONE3 pinyin (e.g. 'ba1') to (phones_list, tone_int).

    Applies the same normalisations as chinese.py _g2p so that
    pinyin_to_symbol_map lookups are consistent.
    Returns ([], 0) when the pinyin is not in the map.
    """
    c, v, tone = _split_pinyin(tone3)

    # Handle punctuation represented as identical initial and final values.
    if c == v and c in _PUNCT_SET:
        return [c], 0

    pinyin = c + v
    if c:
        # Apply vowel contraction to syllables with an initial.
        v_rep = {"uei": "ui", "iou": "iu", "uen": "un"}
        if v in v_rep:
            pinyin = c + v_rep[v]
    else:
        # Apply leading-vowel normalization to syllables without an initial.
        whole_rep = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
        if pinyin in whole_rep:
            pinyin = whole_rep[pinyin]
        else:
            head_rep = {"v": "yu", "e": "e", "i": "y", "u": "w"}
            if pinyin and pinyin[0] in head_rep:
                pinyin = head_rep[pinyin[0]] + pinyin[1:]

    if pinyin not in pinyin_to_symbol_map:
        return [], tone

    phones = pinyin_to_symbol_map[pinyin].split(" ")
    return phones, tone


def _en_token_to_phones(word: str):
    """Convert a single English word/token to (phones, tones).

    Lookup order: CMU dict → g2p_en fallback.
    All outputs go through refine_ph() + post_replace_ph() so they match
    the lowercase ARPAbet symbols in en_symbols (e.g. 'IY1' → 'iy', tone=2).
    """
    phones: list = []
    tones: list = []
    if word == "A":
        return ["ey"], [2]
    if word.upper() in _LETTER_SPELLED_UNITS:
        # Byte units must use letter names. Passing ``GB`` to g2p_en as one
        # unknown word yields G EY B IY ("gay B"). Keep words such as IELTS
        # on their existing dictionary/G2P path.
        for letter in word:
            letter_phones, letter_tones = _en_token_to_phones(letter)
            phones.extend(letter_phones)
            tones.extend(letter_tones)
        return phones, tones
    syllables = _CUSTOM_EN_PRONUNCIATIONS.get(word.upper(), eng_dict.get(word.upper()))
    if syllables is not None:
        for syllable in syllables:
            for phn in syllable:
                phn, tone = refine_ph(phn)
                phones.append(post_replace_ph(phn))
                tones.append(tone)
    else:
        global _g2p_en
        if _g2p_en is None:
            _g2p_en = G2p()
        for ph in filter(lambda p: p != " ", _g2p_en(word)):
            if ph in arpa:
                ph, tone = refine_ph(ph)
                phones.append(post_replace_ph(ph))
                tones.append(tone)
            else:
                phones.append(post_replace_ph(ph))
                tones.append(0)
    return phones, tones


def _non_zh_token_to_phones(token: str):
    """Convert a non-CJK segment that may include spaces/punctuation.

    `simpleseg` may keep spans like `",z"`, `"RB,"` or `"face book"` as a
    single position. Split those spans into English/alphanumeric pieces and
    punctuation so they do not silently disappear from the phone stream.
    """
    phones: list = []
    tones: list = []
    langs: list = []

    for match in _NON_ZH_PIECE_RE.finditer(token):
        piece = match.group(0)
        if _EN_TOKEN_RE.fullmatch(piece):
            word = piece
            ph_list, tone_list = _en_token_to_phones(word)
            if ph_list:
                phones.extend(ph_list)
                tones.extend(tone_list)
                langs.extend(["EN"] * len(ph_list))
        elif piece in _PUNCT_SET:
            phones.append(piece)
            tones.append(0)
            langs.append("ZH")

    return phones, tones, langs


def _flatten_seg(segments):
    """Flatten pypinyin simpleseg output to a per-position list.

    CJK substrings are split char-by-char; non-CJK substrings are kept as units.
    This makes the result length-aligned with lazy_pinyin output.

    Example:
        seg("昨天CEO") → ["昨天", "CEO"]
        _flatten_seg(...)  → ["昨", "天", "CEO"]
        lazy_pinyin(...)   → ["zuo2", "tian1", "CEO"]  ← same length ✓
    """
    result = []
    for s in segments:
        if any("\u4e00" <= c <= "\u9fa5" for c in s):
            result.extend(list(s))
        else:
            result.append(s)
    return result


def unified_g2p(text: str):
    """Convert ZH/EN mixed text to a phoneme sequence.

    Args:
        text: Input text that has already been normalised via mix_normalize()
              (WeText + _post_replace + english.replace_punctuation).
              Chinese punctuation should already be ASCII at this point.

    Returns:
        phones  : List[str]  phoneme symbols, wrapped with '_' silence tokens
        tones   : List[int]  raw tone values (no language-start offset applied)
        langs   : List[str]  'ZH' or 'EN' per phone position
        word2ph : List[int]  phones-per-input-position (sums to len(phones))

    """
    phones: list = []
    tones: list = []
    langs: list = []
    word2ph: list = []

    # Import lazily to avoid the cleaner -> unified_g2p -> chinese cycle.
    from . import chinese

    for match in _MIX_TOKEN_RE.finditer(text):
        token = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            ph_list, tone_list, zh_word2ph = chinese.g2p(token)
            ph_list, tone_list = ph_list[1:-1], tone_list[1:-1]
            zh_word2ph = zh_word2ph[1:-1]
            if ph_list:
                phones.extend(ph_list)
                tones.extend(tone_list)
                langs.extend(["ZH"] * len(ph_list))
            word2ph.extend(zh_word2ph)
        elif _EN_TOKEN_RE.fullmatch(token):
            ph_list, tone_list = _en_token_to_phones(token)
            phones.extend(ph_list)
            tones.extend(tone_list)
            langs.extend(["EN"] * len(ph_list))
            word2ph.append(len(ph_list))
        elif token.isspace():
            # Preserve explicit token boundaries.
            word2ph.append(0)
        elif token in _PUNCT_SET:
            phones.append(token)
            tones.append(0)
            langs.append("ZH")
            word2ph.append(1)
        else:
            ph_list, tone_list, lang_list = _non_zh_token_to_phones(token)
            phones.extend(ph_list)
            tones.extend(tone_list)
            langs.extend(lang_list)
            word2ph.append(len(ph_list))

    # Match the silence-token convention used by the native Chinese frontend.
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    langs = ["ZH"] + langs + ["ZH"]
    word2ph = [1] + word2ph + [1]

    assert len(phones) == len(tones) == len(langs), (
        f"Internal length mismatch: phones={len(phones)} tones={len(tones)} langs={len(langs)}"
    )
    assert sum(word2ph) == len(phones), (
        f"word2ph sum {sum(word2ph)} != phones count {len(phones)}"
    )

    return phones, tones, langs, word2ph
