import logging
import os
import re
import runpy

import jieba
import numpy as np
from pypinyin import lazy_pinyin, Style, load_phrases_dict

from .symbols import punctuation
from .tone_sandhi import ToneSandhi
from .copula_context import wei2_positions as _copula_positions

from .fst_tn import FstTextNormalizer

from .heteronym import load_chinese_lexicon
from .release_paths import frontend_data_dir, tn_cache_dir
from .lexicon_rules import normalize_technical_lexemes
from .lexicon_rules import apply_lexeme_rules
from .special_text import normalize_special_text, verbalize_chinese_number

log = logging.getLogger(__name__)


def _load_phrase_pinyin_data():
    di_path = frontend_data_dir() / "phrase_pinyin_data" / "di.py"
    if di_path.is_file():
        runpy.run_path(str(di_path))["load"]()


custom_dict, jieba_phrases, robot_phrase_pinyin = load_chinese_lexicon()
_load_phrase_pinyin_data()
# External annotations retain the same order and styles as the inherited tables.
load_phrases_dict(custom_dict, style="tone2")
for phrase in jieba_phrases:
    jieba.add_word(phrase)
load_phrases_dict(robot_phrase_pinyin)
for phrase in robot_phrase_pinyin:
    # A default frequency near 1 can still split 请重发消息 into 重/发消息
    # or 请调高 into 请调/高. Keep existing higher frequencies intact.
    jieba.add_word(phrase, freq=max(100, jieba.dt.FREQ.get(phrase, 0)))

current_file_path = os.path.dirname(__file__)
_normalizer = FstTextNormalizer(tn_cache_dir())

with open(
    os.path.join(current_file_path, "opencpop-strict.txt"), encoding="utf-8"
) as symbol_file:
    pinyin_to_symbol_map = {
        line.split("\t")[0]: line.strip().split("\t")[1]
        for line in symbol_file
    }

import jieba.posseg as psg


rep_map = {
    "：": ",",
    ":": ",",
    "；": ",",
    ";": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

tone_modifier = ToneSandhi()


def _post_replace(text: str) -> str:
    # WeText owns semantic minus, range and identifier classification.  Any
    # remaining dash handled below is punctuation only.
    text = text.replace("/", "每")
    text = text.replace("①", "一，").replace("②", "二，").replace("③", "三，")
    text = text.replace("④", "四，").replace("⑤", "五，").replace("⑥", "六，")
    text = text.replace("⑦", "七，").replace("⑧", "八，").replace("⑨", "九，")
    text = text.replace("⑩", "十，")
    text = text.replace("α", "阿尔法").replace("β", "贝塔")
    text = text.replace("γ", "伽玛").replace("Γ", "伽玛")
    text = text.replace("δ", "德尔塔").replace("Δ", "德尔塔")
    text = text.replace("ε", "艾普西龙").replace("ζ", "捷塔").replace("η", "依塔")
    text = text.replace("θ", "西塔").replace("Θ", "西塔").replace("ι", "艾欧塔")
    text = text.replace("κ", "喀帕").replace("λ", "拉姆达").replace("Λ", "拉姆达")
    text = text.replace("μ", "缪").replace("ν", "拗")
    text = text.replace("ξ", "克西").replace("Ξ", "克西").replace("ο", "欧米克伦")
    text = text.replace("π", "派").replace("Π", "派").replace("ρ", "肉")
    text = text.replace("ς", "西格玛").replace("Σ", "西格玛").replace("σ", "西格玛")
    text = text.replace("τ", "套").replace("υ", "宇普西龙")
    text = text.replace("φ", "服艾").replace("Φ", "服艾").replace("χ", "器")
    text = text.replace("ψ", "普赛").replace("Ψ", "普赛")
    text = text.replace("ω", "欧米伽").replace("Ω", "欧米伽")
    text = text.replace("+", "加").replace("×", "乘").replace("÷", "除")
    text = text.replace("=", "等于").replace("＞", "大于").replace(">", "大于")
    text = re.sub(r"[-——《》【】<=>{}()（）#&@^_：、|\\]", "，", text)
    text = re.sub(r"，+", "，", text)
    return text


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


_WHITESPACE_RE = re.compile(r"\s+")


def zh_speakable(text: str) -> bool:
    """Return whether the Chinese path can voice every character of ``text``.

    Whitespace is ignored — it carries no sound. Everything else must survive
    replace_punctuation(), because whatever that drops would be silently
    missing from the audio.
    """
    stripped = _WHITESPACE_RE.sub("", text)
    return len(replace_punctuation(stripped)) == len(stripped)


def g2p(text, *, wei2_positions=None):
    # Whitespace is dropped up front, and only whitespace.
    #
    # This function is handed slices of text normalized either by
    # text_normalize() (which already removed everything it cannot voice) or by
    # mix_normalize(), which ends with the *English* cleaner and therefore keeps
    # spaces. pypinyin collapses a run of non-Chinese characters into a single
    # element, so "  　　事件出圈," produced fewer word2ph entries than
    # it has characters and tripped the assert below — an AssertionError while
    # merely counting a chunk's tokens, which is how a long paragraph failed to
    # synthesize at all rather than losing one space.
    #
    # Anything else unvoiceable is left in place on purpose: it must not vanish
    # without a trace, so it is reported and the caller (see
    # cleaner.g2p_normalized_text_mix) routes such chunks to the mixed frontend,
    # which can still pronounce digits, latin and symbols.
    original = text
    if wei2_positions is not None:
        # MIX supplies offsets within a contiguous Chinese run. Validate them
        # rather than silently applying a request's context to another string.
        wei2_positions = frozenset(wei2_positions)
        if any(not isinstance(i, int) or i < 0 or i >= len(text) or text[i] != "为"
               for i in wei2_positions):
            raise ValueError("Copula context does not match this Chinese run")
    text = _WHITESPACE_RE.sub("", text)
    cleaned = replace_punctuation(text)
    if len(cleaned) != len(text):
        dropped = "".join(sorted(set(text) - set(cleaned)))
        log.warning(
            "[vits2_tts_trt] chinese g2p cannot voice %r; those characters are "
            "not in the audio", dropped,
        )
        text = cleaned
    if wei2_positions is None:
        wei2_positions = _copula_positions(text)
    elif original != text and wei2_positions:
        raise ValueError("Copula offsets require an unchanged Chinese run")
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones, word2ph = _g2p(sentences, wei2_positions=wei2_positions)
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(text)
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _is_particle_de(segments, index):
    word, pos = segments[index]
    if len(word) == 3 and word[0] == word[1] and word.endswith("地"):
        # Jieba can keep AA + 地 together (轻轻地/z, 慢慢地/x).
        # Require a reduplicated modifier; a whole-token "z" tag alone also
        # occurs on place names such as 木樨地 and is not sufficient evidence.
        if pos == "z":
            return True
        modifier = psg.lcut(word[:-1])
        return (len(modifier) == 1 and modifier[0].word == word[:-1]
                and modifier[0].flag in ("a", "ad", "d", "z"))
    if word != "地" or pos != "uv" or index == 0:
        return False
    previous, previous_pos = segments[index - 1]
    # jieba also labels isolated 地 and 一块地's 地 as uv. A preceding
    # quantifier/demonstrative is not an adverbial modifier; retain dì there.
    return (previous not in punctuation
            and not previous_pos.startswith(("m", "q", "r")))


def _g2p(segments, *, wei2_positions=frozenset()):
    phones_list = []
    tones_list = []
    word2ph = []
    segment_offset = 0
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        word_offset = segment_offset
        for index, (word, pos) in enumerate(seg_cut):
            start = word_offset
            word_offset += len(word)
            if pos == "eng":
                continue
            sub_initials, sub_finals = _get_initials_finals(word)
            for char_index, char in enumerate(word):
                if start + char_index in wei2_positions:
                    if char != "为" or char_index >= len(sub_finals):
                        raise ValueError("Copula alignment changed during segmentation")
                    # Resolve before 不/一/third-tone sandhi. Changing final
                    # output tones would leave 不为零 incorrectly as bú wéi.
                    sub_finals[char_index] = sub_finals[char_index][:-1] + "2"
            particle_de = _is_particle_de(seg_cut, index)
            final_di = bool(word.endswith("地") and sub_finals and sub_finals[-1] == "i4")
            if particle_de:
                sub_finals[-1] = "e5"
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            if final_di and not particle_de:
                # The legacy suffix rule changes every 地 to tone 5, including
                # 土地/目的地. Keep lexical dì; only the particle above is de5.
                sub_finals[-1] = "i4"
            initials.append(sub_initials)
            finals.append(sub_finals)

        segment_offset += len(seg)
        initials = sum(initials, [])
        finals = sum(finals, [])
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # pypinyin can emit an empty initial/final for a stripped MIX
            # fragment (e.g. an English brand normalized inside Chinese).
            # It carries no acoustic token; ignore it instead of indexing an
            # empty tone string or failing the complete frontend conversion.
            if not c and not v:
                # Keep one alignment token for this source character.  A
                # silent punctuation token is safer than dropping a token
                # (which would invalidate word2ph and the training cache).
                c, v = ".", "."
            # Distinguish the three pypinyin representations of "i".
            if c == v:
                if c not in punctuation:
                    c = "."
                phone = [c]
                tone = "0"
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # Syllables with an initial.
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # Syllables without an initial.
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                phone = pinyin_to_symbol_map[pinyin].split(" ")
                word2ph.append(len(phone))

            phones_list += phone
            tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph


def _remove_redundant_newlines_after_punctuation(text):
    """Avoid creating duplicate stops from an existing stop plus a newline."""
    # Preserve a word boundary: AI.\nII. must not become the identifier AI.II.
    text = re.sub(r"\r\n|[\r\v\f\x85\u2028\u2029]", "\n", text)
    return re.sub(r"([。！？!?；;：:，,.])[\t ]*\n+[\t ]*", r"\1 ", text)


def _special_number(value, language):
    """Speak a number inside a protected formula/unit, bypassing lexeme rules."""
    if language == "ZH":
        return verbalize_chinese_number(value)
    from .english import normalize_numbers
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = ("plus " if value[0] == "+" else "minus "), value[1:]
    if "." in value:
        whole, fraction = value.split(".", 1)
        digits = "zero one two three four five six seven eight nine".split()
        return sign + normalize_numbers(whole or "0") + " point " + " ".join(digits[int(c)] for c in fraction)
    return sign + normalize_numbers(value)


def _normalize_ordinary_text(text):
    # Written entities are handled by normalize_special_text first. Preserve
    # explicit letter spellings and selected whole words so the FST cannot
    # change them before the shared English/MIX pronunciation lookup sees them.
    from .english import normalize_ordinary_latin_text
    return normalize_ordinary_latin_text(text, _normalizer.normalize)


def _normalize_raw_text(text, lexeme_language="MIX"):
    def normalize_plain(ordinary):
        ordinary = apply_lexeme_rules(ordinary, language=lexeme_language)
        if lexeme_language == "EN":
            from .alphanumeric_text import normalize_unknown_alphanumeric_tokens
            ordinary = normalize_unknown_alphanumeric_tokens(ordinary, "EN")
        return _normalize_ordinary_text(ordinary)

    selected = lexeme_language if lexeme_language in ("ZH", "EN") else None
    return normalize_special_text(text, normalize_plain, _special_number, language=selected)


def text_normalize(text):
    text = _remove_redundant_newlines_after_punctuation(text)
    text = _normalize_raw_text(text)
    text = _post_replace(normalize_technical_lexemes(text))
    text = replace_punctuation(text)
    return text


def mix_normalize(text: str, reading_language=None) -> str:
    """Normalise text for ZH/EN mixed input.

    Keep the complete mixed-language context so WeText can expand acronyms
    such as ``AI`` and ``CTO`` into letter-by-letter forms before G2P.
    """
    from .english import (replace_punctuation as en_replace_punct,
                          normalize_preserving_external_tokens)
    from .mixed_context import normalize_mixed_context
    text = _remove_redundant_newlines_after_punctuation(text)
    if reading_language not in (None, "zh", "en"):
        raise ValueError("reading_language must be zh, en or None")
    if reading_language is not None:
        # A fixed preference applies to supported names/fields throughout the
        # complete request. Ordinary Chinese/English prose is never translated.
        text = _normalize_raw_text(text, lexeme_language=reading_language.upper())
    else:
        # Auto keeps existing clause-local routing; do not infer a preference
        # from the Latin name within an otherwise Chinese sentence.
        normalize_original = _normalize_raw_text
        if re.search(r"[A-Za-z]", text) and not re.search(r"[\u4e00-\u9fff]", text):
            normalize_original = lambda value: _normalize_raw_text(value, lexeme_language="EN")
        text = normalize_mixed_context(
            text, normalize_original, _normalize_ordinary_text, _special_number
        )
    # English number readers emit e.g. twenty-six/twenty-third. Their
    # orthographic hyphen is not a pause or subtraction sign. Do not feed it
    # to the remaining-dash punctuation cleanup or rerun numeric TN.
    text = re.sub(
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-"
        r"(one|two|three|four|five|six|seven|eight|nine|first|second|third|"
        r"fourth|fifth|sixth|seventh|eighth|ninth)\b",
        r"\1 \2", text, flags=re.I,
    )
    text = normalize_preserving_external_tokens(
        text, lambda value: _post_replace(normalize_technical_lexemes(value)))
    text = en_replace_punct(text)
    return text


def get_bert_feature(text, word2ph):
    """Return the zero BERT feature tensor expected by this model."""
    return np.zeros((1024, sum(word2ph)), dtype=np.float32)
