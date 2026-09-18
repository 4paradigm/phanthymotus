#!/usr/bin/env python3
"""
plugins/thai_frontend.py — text normalisation for the MMS Thai VITS model.

The model's tokenizer is character-level over exactly **71 characters** and it
drops everything outside that set. sherpa-onnx records the loss as a C++ stderr
line ("Skip unknown character. Unicode codepoint: \\U+0e33") that never reaches
the Python logger or the dashboard, and HF's tokenizer just maps the character to
`<unk>`; either way the *audio* simply comes back short, which is the same
failure the ZH/EN frontend documents at
``plugins/vits2_tts_trt/frontend/cleaner.py`` ("the digit was dropped instead of
pronounced"). That is why this module exists rather than passing text straight to
sherpa-onnx.

Three of the omissions are not obvious and each one silently mangles ordinary
Thai:

* ``ำ`` (U+0E33 SARA AM) **is not in the vocabulary**, but ``ํ`` (U+0E4D) and
  ``า`` (U+0E32) both are. MMS was trained on text that spelled the vowel as
  that pair, so every ``ำ`` has to be rewritten — otherwise น้ำ, ทำ, คำ, สำหรับ
  and a large fraction of the language lose their vowel. Measured: the sentence
  "น้ำ ทำ คำ สำหรับ น้ำหนัก" logs five skipped U+0E33 and synthesizes 1.34 s;
  rewritten it is 1.51 s, and those five vowels are audible.
* ``ๆ`` (maiyamok, the repeat marker) is absent, so ``ต่างๆ`` must be expanded to
  the repeated word before it reaches the model.
* Thai digits ``๐-๙`` are absent, and of the Arabic digits only ``0 1 2 4``
  survive — ``3 5 6 7 8 9`` are not in the set. So no numeral can be passed
  through; every number has to become words.

Everything the model cannot say is dropped **with a warning naming the
characters**, mirroring the "cannot voice" log the VITS2 frontend emits. Silence
about dropped text is the specific bug this module is written to prevent.

Latin and Chinese runs are transliterated into Thai script rather than routed to
another engine: the deployment speaks with one voice, and Thai-accented English
is what a Thai speaker actually produces for a foreign brand or model name.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Iterator, Optional

log = logging.getLogger(__name__)

# The MMS Thai tokenizer's alphabet. Verified byte-identical (same 71 entries,
# same ids) between facebook/mms-tts-tha and the VIZINTZOR fine-tunes, so it is
# safe as a default — but ThaiFrontend still prefers the tokens.txt shipped with
# the loaded model, so a future voice with a different table cannot drift.
MMS_THAI_VOCAB = frozenset(
    "านร่เ้องกวะัมทพยลจีคตดหขิแสบปไูใ็ื์ชุึํโผถญซธศณษฟภฉฝฐฤฏฮฆ๋ฎ'0๊ฑ142-ฬฒฌ "
)

# Obsolete or decorative characters with a spoken equivalent that *is* in the
# table. ๅ and ฯ have none worth guessing at, so they fall through to the final
# sweep and are reported as dropped.
_CHAR_FALLBACKS = {
    "ฃ": "ข",      # obsolete kho khuat
    "ฅ": "ค",      # obsolete kho khon
    "ฦ": "ล",      # obsolete lu
    "ฺ": "",  # phinthu — a silencing mark, nothing to say
    "๎": "",  # yamakkan
    "​": "",  # zero-width space, common in copy-pasted Thai
}

# Multi-character expansions applied before maiyamok, because ฯลฯ expands *into*
# a ๆ that then has to be expanded in turn.
_PHRASE_EXPANSIONS = {
    "ฯลฯ": "และอื่นๆ",
    "๏": "",
    "๚": "",
    "๛": "",
    "%": "เปอร์เซ็นต์",
    "°": "องศา",
    "+": "บวก",
    "=": "เท่ากับ",
    "&": "และ",
    "@": "แอท",
}

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Punctuation becomes a space rather than nothing. Space is in the vocabulary
# and is Thai's own phrase separator, so this buys the pause the punctuation
# implied; dropping it instead runs two clauses together.
_PUNCT_TO_SPACE = re.compile(r"[,.!?;:()\[\]{}<>\"“”„‘’«»/\\|~*#…、。，！？\n\r\t]+")

_THAI_RUN = re.compile(r"[฀-๿]+")
_CJK_RUN = re.compile(r"[一-鿿㐀-䶿]+")
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# Digit groups, optionally with thousands separators and a decimal part.
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
# Currency, both orders. Thai reads an amount of money with its own units
# (…บาท…สตางค์), so "฿120.50" is not "120.50" followed by the word for baht —
# replacing ฿ with บาท on its own produced "บาทหนึ่งร้อยยี่สิบ", the words in the
# wrong order.
_CURRENCY = re.compile(
    r"฿\s*(\d+(?:,\d{3})*(?:\.\d+)?)"
    r"|(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:฿|บาท)"
)

# Above this many digits a group is an identifier (phone, serial, order number)
# rather than a quantity, and is read digit by digit. Below it, cardinal reading
# is what a Thai speaker uses for counts, prices and years alike — 2026 reads as
# สองพันยี่สิบหก, which is correct for the year. A leading zero also forces
# digit-by-digit: 07 is never "seven".
_CARDINAL_MAX_DIGITS = 4

_THAI_DIGIT_WORDS = {
    "0": "ศูนย์", "1": "หนึ่ง", "2": "สอง", "3": "สาม", "4": "สี่",
    "5": "ห้า", "6": "หก", "7": "เจ็ด", "8": "แปด", "9": "เก้า",
}

# Thai names of the Latin letters, for acronyms. A Thai speaker says "เอ ไอ" for
# AI, never a transliterated syllable, so short all-caps tokens take this path.
_LATIN_LETTER_NAMES = {
    "a": "เอ", "b": "บี", "c": "ซี", "d": "ดี", "e": "อี", "f": "เอฟ",
    "g": "จี", "h": "เอช", "i": "ไอ", "j": "เจ", "k": "เค", "l": "แอล",
    "m": "เอ็ม", "n": "เอ็น", "o": "โอ", "p": "พี", "q": "คิว", "r": "อาร์",
    "s": "เอส", "t": "ที", "u": "ยู", "v": "วี", "w": "ดับเบิลยู",
    "x": "เอ็กซ์", "y": "วาย", "z": "แซด",
}

# Words the deployment actually says, where the rule-based fallback below is
# either wrong or merely worse than the form Thai speakers already use. This is
# the list to extend when a name comes out wrong — not the rule table.
_LATIN_LEXICON = {
    "robot": "โรบอต",
    "ok": "โอเค",
    "okay": "โอเค",
    "hello": "เฮลโล",
    "hi": "ไฮ",
    "wifi": "ไวไฟ",
    "internet": "อินเทอร์เน็ต",
    "computer": "คอมพิวเตอร์",
    "battery": "แบตเตอรี่",
    "start": "สตาร์ท",
    "stop": "สตอป",
    "menu": "เมนู",
    "email": "อีเมล",
    "online": "ออนไลน์",
    "offline": "ออฟไลน์",
    "server": "เซิร์ฟเวอร์",
    "sensor": "เซ็นเซอร์",
    "motor": "มอเตอร์",
    "camera": "กล้อง",
    "meter": "เมตร",
    "metre": "เมตร",
    "kilometer": "กิโลเมตร",
    "kilogram": "กิโลกรัม",
    "gram": "กรัม",
    "second": "วินาที",
    "minute": "นาที",
    "hour": "ชั่วโมง",
    "bangkok": "กรุงเทพ",
    "thailand": "ประเทศไทย",
    "china": "ประเทศจีน",
    "welcome": "เวลคัม",
    "thank": "แธงค์",
    "you": "ยู",
    "please": "พลีซ",
    "yes": "เยส",
    "no": "โน",
    "error": "เออเรอร์",
    "test": "เทสต์",
    "demo": "เดโม",
}

# ── Latin → Thai rule fallback ────────────────────────────────────────────────
#
# A reduced form of the Royal Institute's English transliteration guidance,
# enough for names and product codes. Digraphs are matched before single letters,
# so ordering inside each table matters and they are consulted longest-first.

_ONSET_DIGRAPHS = {
    "sch": "ช", "chr": "คร", "shr": "ชร", "thr": "ธร", "phr": "ฟร",
    "str": "สตร", "spr": "สปร", "scr": "สคร", "spl": "สปล",
    "ch": "ช", "sh": "ช", "th": "ธ", "ph": "ฟ", "wh": "ว", "kn": "น",
    "gh": "ก", "gn": "น", "ps": "ซ", "qu": "คว", "wr": "ร", "rh": "ร",
    "bl": "บล", "br": "บร", "cl": "คล", "cr": "คร", "dr": "ดร", "fl": "ฟล",
    "fr": "ฟร", "gl": "กล", "gr": "กร", "pl": "พล", "pr": "พร", "sc": "สค",
    "sk": "สค", "sl": "สล", "sm": "สม", "sn": "สน", "sp": "สป", "st": "สต",
    "sw": "สว", "tr": "ทร", "tw": "ทว",
}

_ONSET_SINGLE = {
    "b": "บ", "c": "ค", "d": "ด", "f": "ฟ", "g": "ก", "h": "ฮ", "j": "จ",
    "k": "ค", "l": "ล", "m": "ม", "n": "น", "p": "พ", "q": "ค", "r": "ร",
    "s": "ส", "t": "ท", "v": "ว", "w": "ว", "x": "ซ", "y": "ย", "z": "ซ",
}

# khanaa spells a vowel from its abstract form, with อ standing for the onset
# slot. Long/short choice follows the usual English reading of the grapheme.
_VOWELS = {
    "eau": "โอ", "eig": "เอ", "igh": "ไอ",
    "ai": "เอ", "ay": "เอ", "ea": "อี", "ee": "อี", "ei": "เอ", "eu": "อู",
    "ew": "อู", "ey": "อี", "ie": "อี", "oa": "โอ", "oe": "โอ", "oi": "ออย",
    "oo": "อู", "ou": "อาว", "ow": "โอ", "oy": "ออย", "ue": "อู", "ui": "อู",
    "au": "ออ", "aw": "ออ", "oa r": "ออ",
    "a": "อา", "e": "เอ", "i": "อิ", "o": "โอ", "u": "อู", "y": "อิ",
}

# Thai permits only these finals. Anything else is folded onto the nearest one,
# which is what the Royal Institute rules do too.
_CODAS = {
    "ng": "ง", "nk": "ง", "ck": "ก", "gh": "ก", "ph": "ฟ", "sh": "ช",
    "ch": "ช", "th": "ด", "tch": "ช",
    "b": "บ", "p": "บ", "f": "ฟ", "v": "ฟ", "d": "ด", "t": "ด", "s": "ส",
    "z": "ส", "c": "ก", "k": "ก", "g": "ก", "q": "ก", "x": "ก",
    "m": "ม", "n": "น", "l": "ล", "r": "ร", "y": "ย", "w": "ว", "j": "จ",
    "h": "",
}

_LATIN_VOWEL_CHARS = set("aeiouy")


def _longest_match(table: dict, text: str, start: int) -> tuple[Optional[str], int]:
    """Return (mapped, consumed) for the longest key of `table` at text[start:]."""
    for length in range(min(3, len(text) - start), 0, -1):
        key = text[start:start + length]
        if key in table:
            return table[key], length
    return None, 0


def _split_latin_syllables(word: str) -> list[tuple[str, str, str]]:
    """Split a lowercase Latin word into (onset, vowel, coda) grapheme groups.

    Deliberately naive — consonant runs, then vowel runs, then the consonants
    that a following vowel does not claim as its onset. Good enough for names
    and model codes, which is all the fallback is for; anything that matters is
    better added to _LATIN_LEXICON than modelled here.
    """
    groups: list[tuple[str, str, str]] = []
    index = 0
    length = len(word)
    while index < length:
        onset_start = index
        while index < length and word[index] not in _LATIN_VOWEL_CHARS:
            index += 1
        onset = word[onset_start:index]

        vowel_start = index
        while index < length and word[index] in _LATIN_VOWEL_CHARS:
            index += 1
        vowel = word[vowel_start:index]

        coda_start = index
        while index < length and word[index] not in _LATIN_VOWEL_CHARS:
            index += 1
        consonants = word[coda_start:index]
        # A single consonant between two vowels opens the next syllable rather
        # than closing this one: "moto" is โม-โท, not มด-โท.
        if index < length and len(consonants) == 1:
            coda, carry = "", consonants
        elif index < length and len(consonants) > 1:
            coda, carry = consonants[:-1], consonants[-1:]
        else:
            coda, carry = consonants, ""

        if not vowel and not coda and onset:
            # All-consonant tail, e.g. the "ng" of a word we already consumed.
            groups.append((onset, "", ""))
        elif vowel or onset or coda:
            groups.append((onset, vowel, coda))

        if carry:
            word = carry + word[index:]
            length = len(word)
            index = 0
    return groups


def _map_group(table: dict, graphemes: str) -> str:
    """Map a grapheme run through `table`, longest match first, dropping misses."""
    out: list[str] = []
    index = 0
    while index < len(graphemes):
        mapped, consumed = _longest_match(table, graphemes, index)
        if mapped is None:
            index += 1
            continue
        out.append(mapped)
        index += consumed
    return "".join(out)


def _transliterate_latin_word(word: str) -> str:
    """Latin word → Thai script, via the lexicon then the rule fallback."""
    lowered = word.lower().strip("'-")
    if not lowered:
        return ""
    if lowered in _LATIN_LEXICON:
        return _LATIN_LEXICON[lowered]
    # Acronyms and short codes: Thai speakers spell these out.
    if word.isupper() and len(lowered) <= 4:
        return " ".join(_LATIN_LETTER_NAMES.get(ch, "") for ch in lowered).strip()

    if _KHANAA_SPELL is None:
        # Already warned once at import; spelling the word out stays intelligible.
        return " ".join(_LATIN_LETTER_NAMES.get(ch, "") for ch in lowered).strip()

    syllables: list[str] = []
    for onset_gr, vowel_gr, coda_gr in _split_latin_syllables(lowered):
        onset = _map_group(_ONSET_DIGRAPHS, onset_gr) or _map_group(_ONSET_SINGLE, onset_gr)
        vowel, _ = _longest_match(_VOWELS, vowel_gr, 0)
        coda = _map_group(_CODAS, coda_gr)
        if not vowel:
            # Consonant-only group: give it the inherent short vowel so it is
            # pronounceable at all rather than dropped.
            vowel = "อะ" if onset else ""
        if not onset and not vowel:
            continue
        if not onset:
            onset = "อ"      # ตัวเต็ม placeholder onset for a vowel-initial syllable
        try:
            syllables.append(_KHANAA_SPELL(onset, vowel, coda))
        except Exception:
            # khanaa refuses impossible combinations (e.g. a coda the vowel
            # already carries). Keep the syllable audible rather than losing it.
            log.debug("[thai] khanaa refused onset=%r vowel=%r coda=%r from %r",
                      onset, vowel, coda, word)
            try:
                syllables.append(_KHANAA_SPELL(onset, vowel))
            except Exception:
                continue
    return "".join(syllables)


def _transliterate_cjk(run: str) -> str:
    """Chinese → Thai script, via pinyin and the Royal Institute's zh rules.

    pypinyin is already an image dependency (the VITS2 ZH frontend uses it) and
    wunsen transliterates *from* numeric-tone pinyin, so the two compose. Both
    are optional here: without them the run falls through to the final sweep and
    is reported as dropped rather than silently vanishing.
    """
    try:
        from pypinyin import Style, lazy_pinyin
        from wunsen import ThapSap
    except Exception as error:  # noqa: BLE001
        # Exception, not ImportError: wunsen imports khanaa, which can fail at
        # import time on Python 3.8 with a TypeError rather than an ImportError.
        # See _load_khanaa.
        log.warning("[thai] pypinyin/wunsen unusable (%s: %s); cannot say Chinese %r",
                    type(error).__name__, error, run)
        return ""
    try:
        pinyin = lazy_pinyin(run, style=Style.TONE3, neutral_tone_with_five=False)
        return ThapSap("zh", system="RI49").thap(" ".join(pinyin))
    except Exception as error:
        log.warning("[thai] Chinese transliteration failed for %r: %s", run, error)
        return ""


def _number_to_thai(literal: str) -> str:
    """Read a digit group as Thai words."""
    from pythainlp.util import num_to_thaiword

    cleaned = literal.replace(",", "")
    integer, _, fraction = cleaned.partition(".")

    digit_by_digit = (
        len(integer) > _CARDINAL_MAX_DIGITS
        or (len(integer) > 1 and integer.startswith("0"))
    )
    if digit_by_digit:
        head = "".join(_THAI_DIGIT_WORDS.get(ch, "") for ch in integer)
    else:
        try:
            head = num_to_thaiword(int(integer))
        except Exception:
            head = "".join(_THAI_DIGIT_WORDS.get(ch, "") for ch in integer)

    if not fraction:
        return head
    # num_to_thaiword_float only exists on pythainlp's dev branch, so the
    # fractional part is spelled out digit by digit after จุด — which is also
    # how Thai reads decimals aloud.
    tail = "".join(_THAI_DIGIT_WORDS.get(ch, "") for ch in fraction)
    return f"{head}จุด{tail}"


def _load_khanaa():
    """Return a `spell(onset, vowel, coda, tone) -> str`, or None if unavailable.

    Hides two independent problems behind one uniform callable.

    **The API differs by version, and the version differs by JetPack line.**
    khanaa 0.1.1 (jp6.1, cp310) exposes `Kham(...).form`; 0.0.6 — the newest that
    imports at all on cp38, so what jp5.11 gets — exposes
    `SpellWord().spell_out(...)`. Verified to produce identical output for the
    same inputs (สต+เอะ+ก+tone 3 -> เสต๊ก on both), so this is a rename, not a
    downgrade.

    **It catches Exception, not ImportError.** A pure-Python package can fail at
    *import* time without the module being missing: khanaa 0.1.1 annotates
    `def find_same_sound_consonant(...) -> list[str]` at module level, and PEP 585
    builtin generics in a function annotation are evaluated at def time, so
    importing it on Python 3.8 raises

        TypeError: 'type' object is not subscriptable

    An `except ImportError` did not catch that and the TypeError propagated out of
    normalize(), killing the utterance instead of degrading it. pythainlp has the
    same failure shape; see plugins/requirements.thai.txt.
    """
    try:
        from khanaa import Kham

        def spell(onset, vowel, coda="", tone=-1):
            return Kham(onset=onset, vowel=vowel, coda=coda, tone=tone).form

        return spell
    except Exception:  # noqa: BLE001 - see docstring; fall through to the old API
        pass
    try:
        from khanaa import SpellWord

        speller = SpellWord()

        def spell(onset, vowel, coda="", tone=-1):
            return speller.spell_out(onset=onset, vowel=vowel, coda=coda, tone=tone)

        return spell
    except Exception as error:  # noqa: BLE001 - see docstring
        log.warning("[thai] khanaa is unusable (%s: %s); Latin words will be "
                    "spelled out letter by letter instead of transliterated",
                    type(error).__name__, error)
        return None


# Resolved once: the failure is a property of the interpreter, not of the text, so
# re-importing per word would only repeat the same warning on every utterance.
_KHANAA_SPELL = _load_khanaa()


class ThaiFrontend:
    """Turn arbitrary text into something the MMS Thai tokenizer can say.

    `vocab` should be the character set of the loaded model's tokens.txt. It
    defaults to MMS_THAI_VOCAB so the class is unit-testable without a model,
    but the adapter passes the real set: the final sweep is only a guarantee if
    it is checked against the model actually running.
    """

    def __init__(self, vocab: Optional[Iterable[str]] = None,
                 phrase_spacing: bool = False):
        self._vocab = frozenset(vocab) if vocab else MMS_THAI_VOCAB
        # Inserting a space at every word boundary is *not* obviously right: MMS
        # was trained on Thai that only spaces phrase boundaries, so per-word
        # spacing may hurt prosody rather than help it. Off until measured on
        # device.
        self._phrase_spacing = phrase_spacing
        if "ำ" in self._vocab:
            # A future voice whose table does have sara am needs no rewrite, and
            # rewriting anyway would be wrong.
            log.info("[thai] model vocabulary includes ำ; skipping the sara-am rewrite")

    # ── normalisation ────────────────────────────────────────────────────

    def normalize(self, text: str) -> str:
        """Return text containing only characters the model can pronounce."""
        if not text:
            return ""

        stage = self._pre_clean(text)
        stage = self._expand_maiyamok(stage)
        # Numbers first: the punctuation sweep below turns "." and "," into
        # spaces, which cuts "1,250.50" into three unrelated numbers read as
        # "one / two hundred fifty / fifty".
        stage = _CURRENCY.sub(self._currency_to_thai, stage)
        stage = _NUMBER.sub(lambda m: _number_to_thai(m.group(0)), stage)
        stage = _PUNCT_TO_SPACE.sub(" ", stage)
        stage = self._transliterate_foreign(stage)
        stage = self._rewrite_sara_am(stage)
        if self._phrase_spacing:
            stage = self._space_words(stage)
        stage = self._drop_unspeakable(stage)
        return re.sub(r"\s{2,}", " ", stage).strip()

    @staticmethod
    def _currency_to_thai(match: "re.Match[str]") -> str:
        """Read a money amount with Thai's own units via bahttext."""
        literal = (match.group(1) or match.group(2) or "").replace(",", "")
        if not literal:
            return match.group(0)
        try:
            from pythainlp.util import bahttext

            return bahttext(float(literal))
        except Exception as error:
            log.warning("[thai] bahttext failed for %r: %s", literal, error)
            return f"{_number_to_thai(literal)}บาท"

    def _pre_clean(self, text: str) -> str:
        for source, target in _PHRASE_EXPANSIONS.items():
            if source in text:
                text = text.replace(source, target)
        text = text.translate(_THAI_DIGITS)
        for source, target in _CHAR_FALLBACKS.items():
            if source in text:
                text = text.replace(source, target)
        try:
            from pythainlp.util import normalize as thai_normalize
            from pythainlp.util import remove_zw
            # normalize() fixes reordered vowel/tone sequences and duplicated
            # marks — both produce a valid-looking string the model mispronounces.
            text = thai_normalize(remove_zw(text))
        except ImportError:
            log.warning("[thai] pythainlp unavailable; skipping Thai normalisation")
        return text

    @staticmethod
    def _expand_maiyamok(text: str) -> str:
        """Replace each ๆ with the word before it.

        Done here rather than through pythainlp's own helper because that helper's
        signature is not stable across the versions the two JetPack lines get.
        5.3.7 (jp6.1, cp310) exports `expand_maiyamok` and accepts a string;
        5.0.4 — the newest release that imports at all on cp38, so what jp5.11
        gets — exports only `maiyamok` and accepts a token *list*, raising
        IndexError on a string. Ten lines of our own beats branching on that, and
        it also handles a leading ๆ, which 5.0.4's helper crashes on.
        """
        if "ๆ" not in text:
            return text
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError:
            log.warning("[thai] pythainlp unavailable; ๆ will be dropped")
            return text
        try:
            tokens = word_tokenize(text, engine="newmm", keep_whitespace=True)
        except Exception as error:
            log.warning("[thai] ๆ expansion could not tokenise %r: %s", text, error)
            return text

        out: list[str] = []
        for token in tokens:
            if token.strip() == "ๆ":
                # The nearest preceding non-blank token is the one repeated.
                previous = next((t for t in reversed(out) if t.strip()), "")
                if previous:
                    out.append(previous)
                else:
                    # A sentence opening with ๆ has nothing to repeat; dropping it
                    # is the only option, but say so.
                    log.warning("[thai] leading ๆ has no preceding word to repeat")
            elif token.endswith("ๆ") and len(token) > 1:
                # newmm can return the marker glued to its word ("ต่างๆ").
                stem = token[:-1]
                out.append(stem)
                out.append(stem)
            else:
                out.append(token)
        return "".join(out)

    def _transliterate_foreign(self, text: str) -> str:
        text = _CJK_RUN.sub(lambda m: _transliterate_cjk(m.group(0)), text)
        return _LATIN_RUN.sub(lambda m: _transliterate_latin_word(m.group(0)), text)

    def _rewrite_sara_am(self, text: str) -> str:
        """ำ → ํ + า, the spelling MMS's character table actually has."""
        if "ำ" in self._vocab:
            return text
        return text.replace("ำ", "ํา")

    @staticmethod
    def _space_words(text: str) -> str:
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError:
            return text
        return " ".join(word_tokenize(text, engine="newmm", keep_whitespace=False))

    def _drop_unspeakable(self, text: str) -> str:
        kept: list[str] = []
        dropped: list[str] = []
        for char in text:
            if char in self._vocab:
                kept.append(char)
            elif char.isspace():
                kept.append(" ")
            else:
                dropped.append(char)
        if dropped:
            # Named, not counted: the ZH/EN frontend learned the hard way that a
            # silent drop reads downstream as a model bug. See
            # tests/test_vits2_frontend.py::test_unvoiceable_characters_do_not_
            # silence_the_rest.
            log.warning("[thai] no pronunciation for %s; not in the audio",
                        ", ".join(sorted(set(dropped))))
        return "".join(kept)

    # ── chunking ─────────────────────────────────────────────────────────

    def iter_chunks(self, text: str, max_chars: int = 180) -> Iterator[str]:
        """Split normalised text into synthesis chunks.

        Splits on Thai's own phrase separator — the space — rather than calling
        pythainlp's sent_tokenize, whose default `crfcut` engine needs
        python-crfsuite. That is a compiled wheel and one more thing to build
        for arm64, for a job a whitespace split already does: Thai marks clause
        boundaries with spaces and has no sentence-final period.
        """
        text = text.strip()
        if not text:
            return
        if len(text) <= max_chars:
            yield text
            return

        chunk: list[str] = []
        size = 0
        for piece in text.split(" "):
            if not piece:
                continue
            if size and size + 1 + len(piece) > max_chars:
                yield " ".join(chunk)
                chunk, size = [piece], len(piece)
            else:
                chunk.append(piece)
                size += (1 if size else 0) + len(piece)
        # A single phrase longer than the budget has no space to split on; hand
        # it over whole rather than cutting mid-syllable, which would change the
        # pronunciation.
        if chunk:
            yield " ".join(chunk)


# ── build-time self-check ────────────────────────────────────────────────────


def self_check() -> None:
    """Exercise the frontend and assert the one invariant the engine depends on.

    Lives here rather than as a `python3 -c` one-liner in the Dockerfile for two
    reasons. A Dockerfile `RUN` cannot contain a bare newline — every line needs a
    trailing backslash, and the parser reports the first unescaped line as an
    "unknown instruction", which is how the first attempt at this failed. And the
    assertions belong next to the code they guard, where a reader changing the
    frontend will see them.

    Run at image build time, after the source COPY, on the interpreter the image
    actually ships:

        python3 -m plugins.thai_frontend

    That is a stronger test than importing the dependencies, which is what the
    earlier build-time check did: it passed on jp5.11 while khanaa's and
    pythainlp's cp38 API differences were still waiting to crash on that one line.
    """
    import sys

    frontend = ThaiFrontend()
    cases = [
        "สวัสดีครับ ห้อง 305 พร้อมแล้ว",
        "น้ำ ทำ คำ สำหรับ น้ำหนัก",
        "ต่างๆ นานา และมากๆ",
        "สินค้าต่างๆ พร้อม",
        "ๆ นำหน้า",                     # 5.0.4's own maiyamok raises IndexError here
        "ราคา 1,250.50 บาท",
        "เบอร์ 0812345678",
        "ฯลฯ ๗๘๙ ฿100",
        "หุ่นยนต์ Bumi พร้อม",
        "เชื่อมต่อ WiFi แล้ว",
        "ระบบ AI และ USB",
        "ยินดีต้อนรับ 你好 ครับ",
        "ผลลัพธ์ ✅ ok",
    ]
    for text in cases:
        out = frontend.normalize(text)
        stray = sorted({ch for ch in out if ch not in MMS_THAI_VOCAB})
        assert not stray, f"{stray} cannot be pronounced but survived {text!r} -> {out!r}"

    # The rewrites, each of which silently mangles ordinary Thai if it regresses.
    assert "ำ" not in frontend.normalize("น้ำ"), "sara am was not rewritten"
    assert "ํา" in frontend.normalize("น้ำ"), "sara am was dropped, not respelled"
    assert "3" not in frontend.normalize("ห้อง 305"), "a digit reached the model"
    assert "สาม" in frontend.normalize("ห้อง 305"), "the digit was dropped, not spoken"
    assert frontend.normalize("ต่างๆ").count("ต่าง") == 2, "maiyamok was not expanded"
    assert "จุด" in frontend.normalize("ระยะ 1.5 เมตร"), "the decimal point was lost"

    print(f"[build] Thai frontend self-check ok on Python {sys.version.split()[0]}")


if __name__ == "__main__":
    self_check()
