"""Japanese text normalisation for the Kokoro TTS engine.

Why this exists at all, and why it cannot be skipped:

sherpa-onnx's Kokoro frontend splits text on `[一-鿿]` and sends everything in that
range to `ConvertChineseToTokenIDs`, which reads `lexicon-zh.txt`. That function
never receives the requested language — only the *non*-Chinese branch does. Japanese
kanji sit inside that range (`今` U+4ECA, `私` U+79C1, `何` U+4F55), so Japanese
arrives at the model as **Mandarin-pronounced kanji plus espeak-ja kana**. Measured
on Orin 6: `今日私何` produces 30682 samples under `ja`, `en-us`, `es` and `fr`
alike — the language setting has no effect on kanji whatsoever, while kana-only text
changes length 4x between `ja` and `en-us`.

No sherpa-onnx release ships a Japanese lexicon, so the fix has to happen before the
text reaches the engine: convert every kanji to kana here, leaving nothing in the CJK
range, so the whole utterance takes the espeak-ja path.

Same role as plugins/thai_frontend.py, which exists because sherpa-onnx could not be
handed raw Thai either.

Two stages, and the second is not optional:

1. **Dates and counters**, by rule. Japanese counter readings are irregular and
   context-dependent, and a morphological analyser gives the isolated-morpheme
   reading instead: Janome renders `10月` as `ツキ` and `1日` as `ニチ`, where a
   Japanese speaker says `ジュウガツ` and `ツイタチ`. Those have to be rewritten
   before tokenisation, from an explicit table.
2. **Everything else**, via Janome (Apache-2.0, pure Python, verified importable on
   cp38/jp5.11 and cp310/jp6.1). pykakasi does the same job but is GPL-3.0;
   fugashi/cutlet declare `requires_python >= 3.9` and so cannot run on jp5.11.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# ── numbers ───────────────────────────────────────────────────────────────────

_DIGITS = ("", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュウ")

# Sino-Japanese numerals, which is what counters take. Deliberately not the native
# ひとつ/ふたつ series — those go with different counters and are not what a date or
# a measurement uses.
_TENS = "ジュウ"
_HUNDREDS = "ヒャク"
_THOUSANDS = "セン"

# Irregular readings caused by sound changes; writing them out is shorter than
# encoding the euphonic rules, and these are the only cases below 10000.
_HUNDRED_IRREGULAR = {3: "サンビャク", 6: "ロッピャク", 8: "ハッピャク"}
_THOUSAND_IRREGULAR = {3: "サンゼン", 8: "ハッセン"}


def _read_under_10000(n: int) -> str:
    """0-9999 as Sino-Japanese numerals in katakana."""
    if n == 0:
        return "ゼロ"
    out = []
    thousands, n = divmod(n, 1000)
    hundreds, n = divmod(n, 100)
    tens, ones = divmod(n, 10)
    if thousands:
        out.append(_THOUSAND_IRREGULAR.get(thousands)
                   or ((_DIGITS[thousands] if thousands > 1 else "") + _THOUSANDS))
    if hundreds:
        out.append(_HUNDRED_IRREGULAR.get(hundreds)
                   or ((_DIGITS[hundreds] if hundreds > 1 else "") + _HUNDREDS))
    if tens:
        out.append((_DIGITS[tens] if tens > 1 else "") + _TENS)
    if ones:
        out.append(_DIGITS[ones])
    return "".join(out)


def read_number(n: int) -> str:
    """Any non-negative integer as katakana, grouped in Japanese 万/億 units.

    Japanese groups by four digits, not three, so 20260000 is ニセンニヒャクロクジュウ
    マン and not "twenty million". Getting the grouping wrong is the classic mistake
    when porting a Western number reader.
    """
    if n < 0:
        return "マイナス" + read_number(-n)
    if n < 10000:
        return _read_under_10000(n)
    units = ["", "マン", "オク", "チョウ"]
    parts = []
    idx = 0
    while n and idx < len(units):
        n, group = n // 10000, n % 10000
        if group:
            parts.append(_read_under_10000(group) + units[idx])
        idx += 1
    return "".join(reversed(parts))


# ── dates and counters ────────────────────────────────────────────────────────

# 月: 4, 7 and 9 are irregular (シ / シチ / ク, never ヨン / ナナ / キュウ).
_MONTHS = {
    1: "イチガツ", 2: "ニガツ", 3: "サンガツ", 4: "シガツ", 5: "ゴガツ", 6: "ロクガツ",
    7: "シチガツ", 8: "ハチガツ", 9: "クガツ", 10: "ジュウガツ", 11: "ジュウイチガツ",
    12: "ジュウニガツ",
}

# 日: the first ten days, plus 14/20/24, use native readings. Every other day is
# just the number + ニチ — 18 and 28 are regular, so the table below is complete.
_DAYS = {
    1: "ツイタチ", 2: "フツカ", 3: "ミッカ", 4: "ヨッカ", 5: "イツカ", 6: "ムイカ",
    7: "ナノカ", 8: "ヨウカ", 9: "ココノカ", 10: "トオカ", 14: "ジュウヨッカ",
    20: "ハツカ", 24: "ニジュウヨッカ",
}


def _read_month(n: int) -> str:
    return _MONTHS.get(n, read_number(n) + "ガツ")


def _read_day(n: int) -> str:
    return _DAYS.get(n, read_number(n) + "ニチ")


# Ordered longest-first so 年月日 is consumed as a date before 年 alone matches.
_DATE_FULL = re.compile(r"(\d{1,4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_DATE_MD = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_YEAR = re.compile(r"(\d{1,4})\s*年")
_MONTH = re.compile(r"(\d{1,2})\s*月")
_DAY = re.compile(r"(\d{1,2})\s*日")
_TIME = re.compile(r"(\d{1,2})\s*時\s*(\d{1,2})\s*分")
_HOUR = re.compile(r"(\d{1,2})\s*時")
_MINUTE = re.compile(r"(\d{1,2})\s*分")

# 時 and 分 irregulars are driven by the **trailing** digit, not by the whole value,
# so these tables cover the ones place and the readers below apply them positionally.
# A flat lookup keyed on the whole number gets 14時 (ジュウヨジ, not ジュウヨンジ) and
# 30分 (サンジュップン, not サンジュウフン) wrong.
_HOUR_ONES = {4: "ヨジ", 7: "シチジ", 9: "クジ"}
_MINUTE_ONES = {1: "イップン", 3: "サンプン", 4: "ヨンプン", 6: "ロップン",
                8: "ハップン"}


def _read_hour(n: int) -> str:
    tens, ones = divmod(n, 10)
    head = read_number(tens * 10) if tens else ""
    if ones in _HOUR_ONES:
        return head + _HOUR_ONES[ones]
    if ones == 0:
        return read_number(n) + "ジ"
    return head + read_number(ones) + "ジ"


def _read_minute(n: int) -> str:
    tens, ones = divmod(n, 10)
    if ones == 0 and tens:
        # 10分 ジュップン, 20分 ニジュップン, 30分 サンジュップン — the ジュウ
        # contracts to ジュッ before プン.
        base = read_number(n)
        assert base.endswith("ジュウ"), base
        return base[:-1] + "ップン"
    head = read_number(tens * 10) if tens else ""
    if ones in _MINUTE_ONES:
        return head + _MINUTE_ONES[ones]
    return head + read_number(ones) + "フン"


def normalize_dates(text: str) -> str:
    """Rewrite dates, times and their counters into katakana.

    Runs before Janome because Janome would give `月` the isolated reading `ツキ`
    and `日` the reading `ニチ`, so `10月1日` came out `10ツキ1ニチ` where a speaker
    says `ジュウガツツイタチ`.

    Components are joined with **spaces**. Running before Janome means the result is
    one unknown token to it, so nothing downstream will ever break the pieces apart —
    `ニセンニジュウロクネンジュウガツツイタチ` reached espeak as a single 33-character
    "word" and was read as one. The spaces are the only chance to say otherwise.
    """
    text = _DATE_FULL.sub(
        lambda m: (read_number(int(m.group(1))) + "ネン "
                   + _read_month(int(m.group(2))) + " "
                   + _read_day(int(m.group(3)))), text)
    text = _DATE_MD.sub(
        lambda m: _read_month(int(m.group(1))) + " " + _read_day(int(m.group(2))),
        text)
    text = _TIME.sub(
        lambda m: _read_hour(int(m.group(1))) + " " + _read_minute(int(m.group(2))),
        text)
    text = _YEAR.sub(lambda m: read_number(int(m.group(1))) + "ネン", text)
    text = _MONTH.sub(lambda m: _read_month(int(m.group(1))), text)
    text = _DAY.sub(lambda m: _read_day(int(m.group(1))), text)
    text = _HOUR.sub(lambda m: _read_hour(int(m.group(1))), text)
    text = _MINUTE.sub(lambda m: _read_minute(int(m.group(1))), text)
    return text


# Any remaining bare integer. Left until after the counters, so a date's digits are
# never read twice.
_BARE_NUMBER = re.compile(r"\d+")


def normalize_numbers(text: str) -> str:
    """Read any leftover digit run as a Japanese numeral."""
    return _BARE_NUMBER.sub(lambda m: read_number(int(m.group(0))), text)


# ── kanji ─────────────────────────────────────────────────────────────────────

_CJK = re.compile(r"[一-鿿]")


def has_kanji(text: str) -> bool:
    """True when anything would still reach sherpa-onnx's Chinese branch."""
    return bool(_CJK.search(text))


# ── katakana -> Hepburn romaji ────────────────────────────────────────────────
#
# The last stage, and the one that makes Japanese audible at all.
#
# Kana are *not* fed to the model. Kokoro's 114-token table is the misaki phoneme
# inventory — it has `ʣ ʥ ʦ ʨ ᵝ`, so the model was trained to speak Japanese — but it
# does **not** contain `ʑ`, which is exactly what espeak-ja emits for じ, along with
# combining diacritics U+0308 and U+031E. sherpa phonemises with espeak and looks the
# result up in that misaki-derived table, and silently discards whatever is missing.
# Measured on Orin 6: 12 phonemes dropped from one sentence, kana or hiragana alike.
# Those drops are audible holes, and they made the output unintelligible.
#
# Romanising instead and phonemising with a Latin-script voice drops **nothing** —
# measured 0 for `it`, `es` and `en-us`, at the same duration as the kana version
# (4.60-4.83 s vs 4.81 s), so it reads words rather than spelling them out. Feeding
# romaji to espeak-`ja` does *not* work: it spells the letters, 14.78 s for the same
# sentence.
#
# Hand-rolled rather than via `jaconv`: one fewer dependency, and the romanisation is
# the tuning surface for how the chosen voice reads it (`tsu` not `tu`, doubled
# vowels not digraphs). A library would hide exactly the decisions that matter here.
# Same call plugins/thai_frontend.py made.

_ROMAJI_DIGRAPHS = {
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo", "シャ": "sha", "シュ": "shu",
    "ショ": "sho", "チャ": "cha", "チュ": "chu", "チョ": "cho", "ニャ": "nya",
    "ニュ": "nyu", "ニョ": "nyo", "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo",
    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo", "リャ": "rya", "リュ": "ryu",
    "リョ": "ryo", "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo", "ジャ": "ja",
    "ジュ": "ju", "ジョ": "jo", "ヂャ": "ja", "ヂュ": "ju", "ヂョ": "jo",
    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo", "ピャ": "pya", "ピュ": "pyu",
    "ピョ": "pyo",
    # Katakana-only combinations, common in loanwords and names (シャオ・ファン).
    "ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo", "ヴァ": "va",
    "ヴィ": "vi", "ヴェ": "ve", "ヴォ": "vo", "ウィ": "wi", "ウェ": "we",
    "ウォ": "wo", "ティ": "ti", "ディ": "di", "トゥ": "tu", "ドゥ": "du",
    "チェ": "che", "ジェ": "je", "シェ": "she",
}

_ROMAJI = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "o", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ヴ": "vu",
    # Small kana left stranded by an unmatched digraph — better a vowel than a hole.
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
    "ャ": "ya", "ュ": "yu", "ョ": "yo",
    "・": " ",
}

_VOWELS = "aiueo"
_SOKUON = "ッ"
_CHOONPU = "ー"

# Readings a dictionary gives as written rather than as spoken. Applied only to
# tokens Janome tags 助詞 (particle), so the は inside a word like コンニチハ is not
# touched — that one is handled by _GREETINGS below.
_PARTICLE_READINGS = {"ハ": "ワ", "ヘ": "エ"}

# Lexicalised exceptions: single tokens whose dictionary reading keeps a は that is
# nevertheless pronounced わ. These are greetings, and they are frequent enough in a
# guide robot's script that getting them wrong is immediately noticeable.
_GREETINGS = {"コンニチハ": "コンニチワ", "コンバンハ": "コンバンワ"}

# Tokens that must not be pushed away from the word before them.
_LEADING_PUNCT = re.compile(r"^[、。！？，．!?,.\)）」』]")

# Tokens that are not words but the tail of the previous one. Janome splits the
# volitional `マショウ` into `マショ` + `ウ`, and `ー` can arrive alone; a space
# before either invents a boundary and blocks the long-vowel rule.
_VOWEL_CONTINUATIONS = {"ウ", "ー"}

# Full-width punctuation carries no meaning to a Latin-script espeak voice, which
# needs ASCII to find sentence boundaries — without this it runs the whole utterance
# together as one breath group.
_PUNCT_TO_ASCII = {"。": ".", "、": ",", "！": "!", "？": "?", "：": ":",
                   "；": ";", "（": "(", "）": ")", "「": '"', "」": '"',
                   "『": '"', "』": '"', "・": " "}


def _to_hiragana(text: str) -> str:
    """Katakana -> hiragana, so one table serves both."""
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
                   for c in text)


def _to_katakana(text: str) -> str:
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c
                   for c in text)


def to_romaji(text: str) -> str:
    """Kana -> Hepburn romaji, with gemination and long vowels spelled out.

    Three rules beyond the table, each chosen for how a Latin-script espeak voice
    will read the result rather than for orthographic tradition:

    - `ッ` doubles the next consonant (`ツイタチ` stays `tsuitachi`, but `ニッキ`
      becomes `nikki`). Italian reads doubled consonants as geminates natively,
      which is what Japanese actually does with them.
    - `ー` doubles the preceding vowel, and so does a vowel that simply repeats
      (`キョウ` -> `kyoo`, not `kyou`). `ou` would be read as two syllables.
    - `ン` is a bare `n`.
    """
    text = _to_katakana(text)
    out = []
    i = 0
    while i < len(text):
        pair = text[i:i + 2]
        if pair in _ROMAJI_DIGRAPHS:
            out.append(_ROMAJI_DIGRAPHS[pair])
            i += 2
            continue
        ch = text[i]
        if ch == _SOKUON:
            # Double whatever consonant comes next; a trailing ッ is dropped.
            nxt = text[i + 1:i + 3]
            syllable = (_ROMAJI_DIGRAPHS.get(nxt)
                        or _ROMAJI.get(text[i + 1:i + 2], ""))
            if syllable and syllable[0] not in _VOWELS:
                out.append(syllable[0])
            i += 1
            continue
        if ch == _CHOONPU:
            # Lengthen by repeating the vowel we just emitted.
            if out and out[-1] and out[-1][-1] in _VOWELS:
                out.append(out[-1][-1])
            i += 1
            continue
        mapped = _ROMAJI.get(ch)
        if mapped is None:
            # Punctuation, digits, Latin — pass through, translating full-width
            # punctuation to ASCII so espeak can find the sentence boundaries.
            out.append(_PUNCT_TO_ASCII.get(ch, ch))
            i += 1
            continue
        prev = out[-1][-1] if (out and out[-1]) else ""
        if prev in _VOWELS and (mapped == prev
                                or (ch == "ウ" and prev in "ou")):
            # A long vowel, written two ways in kana and needing one spelling here:
            #   オオ / ウウ  -> a repeated vowel
            #   オウ         -> the ordinary way to write long o (キョウ = kyoo)
            # Both become a doubled vowel, because `kyou` reads as two syllables in
            # Italian and Spanish while `kyoo` reads as one long one.
            out.append(prev)
            i += 1
            continue
        out.append(mapped)
        i += 1
    # Collapse the runs of spaces the interpunct and token joining can leave.
    return re.sub(r" {2,}", " ", "".join(out)).strip()



class JapaneseFrontend:
    """Turns Japanese text into kana so it takes the espeak-ja path.

    Holds the Janome tokenizer, which loads a ~180 MB dictionary on construction —
    so build once per adapter, never per utterance.
    """

    def __init__(self):
        self._tokenizer = None
        try:
            from janome.tokenizer import Tokenizer
        except ImportError as error:
            # Refuse rather than warn, unlike the Chinese normaliser. Without this
            # the kanji do not merely lose their numbers — they are pronounced in
            # Mandarin, so the card would come up `running` and speak the wrong
            # language. Same call the Thai adapter makes about pythainlp.
            raise RuntimeError(
                "the Japanese voice needs janome (installed when ENABLE_JA_TTS=1); "
                "without it, kanji reach sherpa-onnx's Chinese branch and are "
                f"pronounced in Mandarin: {error}"
            ) from error
        self._tokenizer = Tokenizer()
        log.info("[tts] Japanese frontend ready (janome)")

    def to_kana(self, text: str) -> str:
        """Kanji -> katakana, via Janome's per-token readings.

        Two readings are corrected here, because a dictionary gives the *written*
        kana and Japanese pronounces these two differently as particles:

            は as a particle -> ワ (wa), not ハ
            へ as a particle -> エ (e),  not ヘ

        `今日は` is `kyoo wa`, never `kyoo ha`. Janome tags the part of speech, so
        this is a lookup rather than a guess — and it has to happen here, where the
        tags exist, not in the romaji table, which sees only kana.

        Tokens are joined with **spaces**, which matters more than it looks. Japanese
        is written without them, but the romaji is read by a Latin-script espeak
        voice, and one unbroken `kyoowanisennijuurokunenjuugatsutsuitachidesu` is a
        single enormous "word" it has to guess the stress of. Measured on the
        reported sentence: 7.62 s unspaced against 4.60 s spaced, for the same text.
        Janome has already found the morpheme boundaries; this just keeps them.
        """
        parts = []
        for token in self._tokenizer.tokenize(text):
            reading = token.reading
            if not reading or reading == "*":
                # Punctuation, Latin and anything out-of-dictionary keep their
                # surface form; only kanji actually need converting.
                reading = token.surface
            if token.part_of_speech.split(",")[0] == "助詞":
                reading = _PARTICLE_READINGS.get(reading, reading)
            reading = _GREETINGS.get(reading, reading)
            # No space before punctuation, or espeak reads the gap as a pause.
            # Nor before a bare ウ / ー: Janome splits `マショウ` (mashoo) into
            # `マショ` + `ウ`, and a space there both invents a word boundary that
            # is not there and stops the long-vowel rule from firing, giving
            # `masho u` instead of `mashoo`.
            if (parts and not _LEADING_PUNCT.match(reading)
                    and reading not in _VOWEL_CONTINUATIONS):
                parts.append(" ")
            parts.append(reading)
        return "".join(parts)

    def to_kana_only(self, text: str) -> str:
        """Counters, then leftover numbers, then kanji->kana — stopping at kana.

        `normalize` continues on to romaji, which is what a Latin-script espeak voice
        needs. The direct runtime wants kana instead, because plugins/ja_phonemes maps
        kana to the phonemes the model was actually trained on. Same first three
        stages either way.
        """
        if not text:
            return text
        kana = self.to_kana(normalize_numbers(normalize_dates(text)))
        if has_kanji(kana):
            log.warning(
                "[tts] Japanese text still contains kanji after conversion (%s); "
                "those characters have no phoneme mapping and will be dropped",
                "".join(sorted(set(_CJK.findall(kana))))[:20])
        return kana

    def normalize(self, text: str) -> str:
        """Counters, then leftover numbers, then kanji->kana, then kana->romaji.

        The output is **romaji, not kana**, and that is deliberate: see the
        katakana->romaji section above. Kana handed to espeak-ja produce phonemes
        Kokoro's token table cannot represent, and sherpa drops them, which is what
        made the first version of this unintelligible.
        """
        if not text:
            return text
        kana = self.to_kana(normalize_numbers(normalize_dates(text)))
        if has_kanji(kana):
            # Not fatal — partial conversion still beats all-Mandarin — but those
            # characters stay in the CJK range and will be read in Chinese.
            log.warning(
                "[tts] Japanese text still contains kanji after conversion (%s); "
                "those characters will be pronounced in Mandarin",
                "".join(sorted(set(_CJK.findall(kana))))[:20])
        return to_romaji(kana)
