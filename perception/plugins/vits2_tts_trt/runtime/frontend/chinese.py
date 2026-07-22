import os
import re
import runpy

import jieba
import numpy as np
from pypinyin import lazy_pinyin, Style, load_phrases_dict

from .symbols import punctuation
from .tone_sandhi import ToneSandhi

from tn.chinese.normalizer import Normalizer as ZhNormalizer

from .heteronym import custom_dict, jieba_phrases


def _load_phrase_pinyin_data():
    data_dir = os.getenv("VITS2_FRONTEND_DATA_DIR", os.path.dirname(__file__))
    di_path = os.path.join(data_dir, "phrase_pinyin_data", "di.py")
    if os.path.isfile(di_path):
        runpy.run_path(di_path)["load"]()


_load_phrase_pinyin_data()
# Project-specific entries load last so they can override the broad dictionary.
load_phrases_dict(custom_dict, style="tone2")
for phrase in jieba_phrases:
    jieba.add_word(phrase)

current_file_path = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_tn_cache_dir = os.path.join(project_root, "tn_cache")
_TN_CACHE_DIR = os.getenv("TN_CACHE_DIR", default_tn_cache_dir)
_normalizer = ZhNormalizer(cache_dir=_TN_CACHE_DIR)

pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg


rep_map = {
    "：": ",",
    "；": ",",
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


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones, word2ph = _g2p(sentences)
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
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

            # assert len(sub_initials) == len(sub_finals) == len(word)
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                tone = "0"
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
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


def text_normalize(text):
    text = _normalizer.normalize(text)
    text = _post_replace(text)
    text = replace_punctuation(text)
    return text


_MIX_LATIN_SPAN_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
_MIX_UNIT_RANGE_RE = re.compile(r"(?<=[A-Za-z])\s*[-~～—]\s*(?=\d)")
_MIX_NUMBER_UNIT_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)([A-Za-z]+(?:/[A-Za-z]+)?)"
)
_LETTER_SPELLED_UNITS = {"KB", "MB", "GB", "TB", "PB"}


def _normalize_unit_number(match: re.Match) -> str:
    # A temporary measure word forces WeText to treat values such as 900 as
    # cardinal numbers (九百), rather than digit sequences (九零零).
    normalized = _normalizer.normalize(f"{match.group(1)}个")
    number = normalized[:-1] if normalized.endswith("个") else normalized
    raw_unit = match.group(2).upper()
    numerator, separator, denominator = raw_unit.partition("/")
    unit = " ".join(numerator) if numerator in _LETTER_SPELLED_UNITS else numerator
    if separator:
        unit += separator + (
            " ".join(denominator)
            if denominator in _LETTER_SPELLED_UNITS
            else denominator
        )
    return number + unit


def _normalize_mixed_zh_numbers(text: str) -> str:
    """Run Chinese TN without letting it reinterpret Latin words or units."""
    # In Chinese technical text, ``900MB-1GB`` describes a range, not -1 GB.
    text = _MIX_UNIT_RANGE_RE.sub("到", text)
    # Keep number+unit together until the number has Chinese cardinal reading;
    # the Latin unit itself remains untouched for the English G2P stage.
    text = _MIX_NUMBER_UNIT_RE.sub(_normalize_unit_number, text)
    parts = _MIX_LATIN_SPAN_RE.split(text)
    latin_spans = _MIX_LATIN_SPAN_RE.findall(text)
    normalized = []
    for index, part in enumerate(parts):
        if part:
            normalized.append(_normalizer.normalize(part))
        if index < len(latin_spans):
            normalized.append(latin_spans[index])
    return "".join(normalized)




def mix_normalize(text: str) -> str:
    """Normalise text for ZH/EN mixed input.

    Latin spans are protected while Chinese WeText normalizes surrounding
    numbers. This keeps ``900MB`` as ``九百MB`` instead of either English
    ``nine hundred`` or WeText's erroneous ``九百米b`` interpretation.
    """
    from .english import replace_punctuation as en_replace_punct
    text = _normalize_mixed_zh_numbers(text)
    text = _post_replace(text)
    text = en_replace_punct(text)
    return text


def get_bert_feature(text, word2ph):
    # BERT feature disabled for inference-only deployment
    return np.zeros((1024, sum(word2ph)), dtype=np.float32)


if __name__ == "__main__":
    from .chinese_bert import get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
