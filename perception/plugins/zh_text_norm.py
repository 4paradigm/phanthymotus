"""Chinese text normalisation for the sherpa-onnx TTS engines, applied selectively.

sherpa-onnx takes `rule_fsts` on `OfflineTtsConfig` and applies every FST to the
**whole text before its frontend routes anything** (offline-tts-kokoro-impl.h:224 —
`for (const auto &tn : tn_list_) text = tn->Normalize(text);`, then
`ConvertTextToTokenIds` at :243). The frontend then splits on `[一-鿿]` to decide
what is Chinese. So handing it the ZH number/date/phone FSTs rewrites the digits
into Chinese characters *first*, and the frontend duly reads them in Chinese —
in every language:

    "...opened in 2026."               -> "...opened in 二千零二十六."
    "We have 25 exhibits today."       -> "We have 二十五 exhibits today."
    "Hola, tenemos 25 exposiciones."   -> "Hola, tenemos 二十五 exposiciones."

Both `MatchaTTSAdapter` and `KokoroTTSAdapter` did exactly that. The fix is not to
drop the FSTs — for Chinese they are good and load-bearing:

    "2026年"          -> "二零二六年"
    "2026年1月15日"   -> "二零二六年一月十五日"
    "第25个展品"      -> "第二十五个展品"
    "延迟200毫秒"     -> "延迟二百毫秒"

— it is to run them here, before sherpa sees the text, and only on the parts that
are actually Chinese. Callers therefore pass `rule_fsts=""` and call `normalize()`.

Gating on the configured language would have been simpler and is worse in both
directions: `matcha-zh-en` has no language field at all, and even for Kokoro the
language is a property of the *speaker*, not reliably of the text — an operator on
`en-us` still reads Chinese sign text, and a `zh` tour still says "agent core".
Deciding per number, from its neighbours, gets both right.
"""

from __future__ import annotations

import logging
import os
import re

log = logging.getLogger(__name__)

# The three FSTs sherpa-onnx ships beside its Chinese-capable releases. Order
# matters: date before number, or "2026年1月15日" is consumed digit-group by
# digit-group as three plain quantities instead of a date.
FST_NAMES = ("date-zh.fst", "number-zh.fst", "phone-zh.fst")

# Same range sherpa-onnx's own frontend uses to decide what is Chinese
# (kokoro-multi-lang-lexicon.cc: "([\\u4e00-\\u9fff]+)"). Deliberately identical:
# if this disagreed with the router, text could be normalised here and then routed
# to espeak anyway, or vice versa.
_CJK = r"一-鿿"

_RUN = re.compile(
    rf"(?P<cjk>[{_CJK}]+)"
    r"|(?P<digit>\d+)"
    r"|(?P<latin>[A-Za-z]+)"
    r"|(?P<other>[^"
    rf"{_CJK}"
    r"\dA-Za-z]+)"
)

# Characters that carry no language of their own and so must not break the
# adjacency test: "延迟 200 毫秒" and "第25，26个" both have a Chinese neighbour.
_NEUTRAL = set(" \t\n\r\f\v.,:;!?()[]{}\"'`~@#$%^&*+=|\\/<>-_")

# Punctuation that only appears in CJK text, so it counts as Chinese context on its
# own: "有 25 个。" and "2026年。" read as Chinese even when the adjacent run is the
# sentence end.
#
# Deliberately excludes the characters an English typographer uses — the em dash
# `—` (U+2014), ellipsis `…` (U+2026), curly quotes `“ ” ‘ ’` and the middle dot
# `·`. Those were in here at first, which split
# "a fundamental transition—from model-driven" into three segments and handed the
# bare dash to the ZH FSTs as "Chinese". Harmless as it turned out (the FSTs leave
# punctuation alone, so the text came back byte-identical), but it made an English
# sentence look Chinese to every later reader of this code, and a future FST that
# *did* rewrite punctuation would have turned it into a real bug.
_CJK_PUNCT = set("。，、；：？！（）《》【】")

# Hiragana and katakana. Their presence means the text is Japanese, not Chinese —
# and Japanese kanji sit inside _CJK, so without this check the adjacency rule below
# marks a Japanese sentence's numbers as Chinese and the ZH FSTs rewrite them:
#
#   今日は2026年10月1日です  ->  今日は二零二六年十月一日です
#
# which is then read in Mandarin. Japanese dates want ジュウガツ / ツイタチ, which
# plugins/ja_text_norm.py handles. Chinese text does not contain kana, so treating
# any kana as "this is not Chinese" is safe in the direction that matters.
_KANA = re.compile(r"[぀-ヿ]")


def has_kana(text: str) -> bool:
    """True when the text contains hiragana or katakana, i.e. it is Japanese."""
    return bool(_KANA.search(text))


def _classify(text: str):
    """Split into (kind, start, end) runs, kind in cjk/digit/latin/other."""
    return [(m.lastgroup, m.start(), m.end()) for m in _RUN.finditer(text)]


def _other_is_chinese_context(chunk: str) -> bool:
    """True when an `other` run is Chinese punctuation rather than neutral filler."""
    return any(c in _CJK_PUNCT for c in chunk)


def _neighbour_is_cjk(runs, text: str, index: int, step: int) -> bool:
    """Look outward from runs[index] for the first run that carries a language.

    Neutral punctuation and whitespace are skipped; Chinese punctuation stops the
    walk and answers yes. A Latin run stops it and answers no — that is what keeps
    "we have 25 exhibits" out of the Chinese branch.
    """
    i = index + step
    while 0 <= i < len(runs):
        kind, start, end = runs[i]
        chunk = text[start:end]
        if kind == "cjk":
            return True
        if kind == "latin":
            return False
        if kind == "digit":
            # A second number tells us nothing on its own; keep looking past it.
            i += step
            continue
        if _other_is_chinese_context(chunk):
            return True
        if all(c in _NEUTRAL for c in chunk):
            i += step
            continue
        return False
    return False


def _is_chinese_number(runs, text: str, index: int) -> bool:
    """A digit run belongs to Chinese if *either* side is Chinese.

    Either-side rather than one-side-wins because both of these are Chinese:

        "延迟 200 毫秒"   (Chinese on both sides)
        "共25 items"      (Chinese only before)
        "In 2026 我们开业" (Chinese only after)

    while "we have 25 exhibits" and "delay 200 ms" have Chinese on neither and stay
    with espeak, which reads numbers correctly in each of its own languages.
    """
    return (_neighbour_is_cjk(runs, text, index, -1)
            or _neighbour_is_cjk(runs, text, index, +1))


def segment(text: str):
    """Split `text` into ``(is_chinese, substring)`` pieces covering it exactly.

    Pure text, no FST and no model — which is the point: this is the part of the
    fix that is testable on any host, and the part most likely to be wrong.

    Neutral runs join whichever segment is open, so punctuation never splits a
    sentence into extra pieces. Concatenating the substrings reproduces the input.

    Japanese is excluded wholesale: kanji are inside the CJK range, so a Japanese
    sentence would otherwise have its numbers rewritten into Chinese numerals and
    read in Mandarin. See has_kana.
    """
    if has_kana(text):
        return [(False, text)] if text else []

    runs = _classify(text)
    if not runs:
        return [(False, text)] if text else []

    flags = []
    for i, (kind, start, end) in enumerate(runs):
        if kind == "cjk":
            flags.append(True)
        elif kind == "digit":
            flags.append(_is_chinese_number(runs, text, i))
        elif kind == "latin":
            flags.append(False)
        else:
            chunk = text[start:end]
            # Chinese punctuation is Chinese; anything else inherits the run before
            # it so a trailing "." does not start a new segment.
            flags.append(_other_is_chinese_context(chunk)
                         or (flags[-1] if flags else False))

    pieces = []
    cur_flag, cur_start = flags[0], runs[0][1]
    for i in range(1, len(runs)):
        if flags[i] != cur_flag:
            pieces.append((cur_flag, text[cur_start:runs[i][1]]))
            cur_flag, cur_start = flags[i], runs[i][1]
    pieces.append((cur_flag, text[cur_start:]))
    return pieces


class ZhTextNormalizer:
    """Applies the ZH rule FSTs to the Chinese parts of mixed text.

    Holds the loaded FSTs, so construct once per adapter rather than per utterance
    (`kaldifst.TextNormalizer` compiles the FST on construction).
    """

    def __init__(self, model_dir: str, fst_names=FST_NAMES):
        self._normalizers = []
        self._available = False

        paths = [os.path.join(model_dir, name) for name in fst_names]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            log.info("[tts] no ZH rule FSTs in %s; numbers pass through to the "
                     "engine's own frontend", model_dir)
            return

        try:
            import kaldifst
        except ImportError as error:
            # A warning, not a refusal. Without this, Chinese digits are read by
            # espeak in the selected voice — wrong, but audible and confined to
            # numbers. (Contrast the Thai frontend, which refuses to load without
            # pythainlp because there the digits are *dropped* from the audio.)
            #
            # kaldifst reaches the image through
            # plugins/vits2_tts_trt/requirements.jetson.txt, installed when
            # ENABLE_VITS2_TRT=1 — the Dockerfile default. A build with it off
            # lands here.
            log.warning(
                "[tts] kaldifst is not installed (it ships with ENABLE_VITS2_TRT=1), "
                "so Chinese numbers will not be normalised and will be read by "
                "espeak instead: %s", error)
            return

        for path in paths:
            try:
                self._normalizers.append(kaldifst.TextNormalizer(path))
            except Exception as error:  # noqa: BLE001 - a bad FST must not be fatal
                log.warning("[tts] could not load %s: %s", os.path.basename(path), error)
        self._available = bool(self._normalizers)
        if self._available:
            log.info("[tts] ZH text normalisation active (%d FSTs from %s)",
                     len(self._normalizers), model_dir)

    @property
    def available(self) -> bool:
        return self._available

    def _normalize_chinese(self, chunk: str) -> str:
        for normalizer in self._normalizers:
            try:
                chunk = normalizer.normalize(chunk)
            except Exception as error:  # noqa: BLE001
                log.warning("[tts] ZH normalisation failed on %r: %s", chunk, error)
                return chunk
        return chunk

    def normalize(self, text: str) -> str:
        """Return `text` with only its Chinese parts run through the ZH FSTs.

        Whole segments are handed to the FSTs, never the bare digits: `date-zh.fst`
        has to see the `年` to produce `二零二六年` rather than the quantity form
        `二千零二十六`.
        """
        if not text or not self._available:
            return text
        out = []
        for is_chinese, chunk in segment(text):
            out.append(self._normalize_chinese(chunk) if is_chinese else chunk)
        return "".join(out)
