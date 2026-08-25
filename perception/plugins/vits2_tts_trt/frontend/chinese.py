import logging
import os
import re
import runpy

import jieba
import numpy as np
from pypinyin import lazy_pinyin, Style, load_phrases_dict

from .symbols import punctuation
from .tone_sandhi import ToneSandhi

from .fst_tn import FstTextNormalizer

from .heteronym import custom_dict, jieba_phrases
from .release_paths import frontend_data_dir, tn_cache_dir

log = logging.getLogger(__name__)


def _load_phrase_pinyin_data():
    di_path = frontend_data_dir() / "phrase_pinyin_data" / "di.py"
    if di_path.is_file():
        runpy.run_path(str(di_path))["load"]()


_load_phrase_pinyin_data()
# Bundled pronunciation entries load last so they can override the broad dictionary.
load_phrases_dict(custom_dict, style="tone2")
for phrase in jieba_phrases:
    jieba.add_word(phrase)

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


def g2p(text):
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
    text = _WHITESPACE_RE.sub("", text)
    cleaned = replace_punctuation(text)
    if len(cleaned) != len(text):
        dropped = "".join(sorted(set(text) - set(cleaned)))
        log.warning(
            "[vits2_tts_trt] chinese g2p cannot voice %r; those characters are "
            "not in the audio", dropped,
        )
        text = cleaned
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones, word2ph = _g2p(sentences)
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


def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            initials.append(sub_initials)
            finals.append(sub_finals)

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
    return re.sub(r"([。！？!?；;：:，,.])[\t ]*(?:\r\n|\r|\n)+[\t ]*", r"\1", text)


def text_normalize(text):
    text = _remove_redundant_newlines_after_punctuation(text)
    text = _normalizer.normalize(text)
    text = _post_replace(text)
    text = replace_punctuation(text)
    return text


def mix_normalize(text: str) -> str:
    """Normalise text for ZH/EN mixed input.

    Keep the complete mixed-language context so WeText can expand acronyms
    such as ``AI`` and ``CTO`` into letter-by-letter forms before G2P.
    """
    from .english import replace_punctuation as en_replace_punct
    text = _remove_redundant_newlines_after_punctuation(text)
    text = _normalizer.normalize(text)
    text = _post_replace(text)
    text = en_replace_punct(text)
    return text


def get_bert_feature(text, word2ph):
    """Return the zero BERT feature tensor expected by this model."""
    return np.zeros((1024, sum(word2ph)), dtype=np.float32)
