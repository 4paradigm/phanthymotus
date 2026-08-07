import pickle
import os
import re
import inflect
import numpy as np

from .symbols import punctuation, symbols

current_file_path = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure the bundled NLTK data path before importing g2p_en.
NLTK_DATA_DIR = os.getenv("NLTK_DATA", os.path.join(project_root, "nltk_data"))
import nltk
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

from g2p_en import G2p

EN_TN_CACHE_DIR = os.getenv("EN_TN_CACHE_DIR", os.path.join(project_root, "tn_cache"))
EN_TN_MODE = os.getenv("EN_TN_MODE", "auto").lower()

_frontend_data_dir = os.getenv("VITS2_FRONTEND_DATA_DIR", "")
CMU_DICT_PATH = os.getenv(
    "VITS2_CMU_DICT_PATH",
    os.path.join(_frontend_data_dir, "cmudict.rep")
    if _frontend_data_dir
    else os.path.join(current_file_path, "cmudict.rep"),
)
CACHE_PATH = os.getenv(
    "VITS2_CMU_CACHE_PATH",
    os.path.join(_frontend_data_dir, "cmudict_cache.pickle")
    if _frontend_data_dir
    else os.path.join(current_file_path, "cmudict_cache.pickle"),
)
_g2p = G2p()
_normalizer = None

arpa = {
    "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B", "G",
    "AY0", "M", "AA0", "F", "AO0", "ER2", "UH1", "IY1", "AH2", "DH", "IY0",
    "EY1", "IH0", "K", "N", "W", "IY2", "T", "AA1", "ER1", "EH2", "OY0",
    "UH2", "UW1", "Z", "AW2", "AW1", "V", "UW2", "AA2", "ER", "AW0", "UW0",
    "R", "OW1", "EH1", "ZH", "AE0", "IH2", "IH", "Y", "JH", "P", "AY1",
    "EY0", "OY2", "TH", "HH", "D", "ER0", "CH", "AO1", "AE1", "AO2", "OY1",
    "AY2", "IH1", "OW0", "L", "SH",
}


def post_replace_ph(ph):
    rep_map = {
        "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
        "\n": ".", "·": ",", "、": ",", "…": "...", "···": "...",
        "・・・": "...", "v": "V",
    }
    if ph in rep_map:
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    return "UNK"


rep_map = {
    ":": ",", ";": ",",
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
    "\n": ".", "．": ".", "…": "...", "···": "...", "・・・": "...",
    "·": ",", "・": ",", "、": ",", "$": ".", "\u201c": "'", "\u201d": "'",
    '"': "'", "\u2018": "'", "\u2019": "'", "（": "'", "）": "'",
    "(": "'", ")": "'", "《": "'", "》": "'", "【": "'", "】": "'",
    "[": "'", "]": "'", "—": "-", "−": "-", "～": "-", "~": "-",
    "「": "'", "」": "'",
}

REPLACE_PUNCTUATION_PATTERN = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
TOKEN_SPLIT_RE = re.compile(r"([,;.\?\!\s+])")
SPACE_RE = re.compile(r"\s+")
PUNCT_SPACE_RE = re.compile(r"([,;.\?\!])([\w])")
LETTER_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")
COMPLEX_WETEXT_RE = re.compile(
    r"[^\x00-\x7F]|(?:\d{1,4}[-/:]\d)|(?:\b[A-Za-z]\.){2,}|[%#/@+=*&]"
)
PREPROCESS_TRANSLATION = str.maketrans({
    "–": "-", "—": "-", "～": "-", "，": ",", "。": ".", "：": ":",
    "；": ";", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
})

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"),
        ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
        ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"),
        ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
        ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    ]
]


def get_normalizer():
    global _normalizer
    if _normalizer is None:
        os.makedirs(EN_TN_CACHE_DIR, exist_ok=True)
        from tn.english.normalizer import Normalizer as tn_normalizer
        _normalizer = tn_normalizer(cache_dir=EN_TN_CACHE_DIR)
    return _normalizer


def preprocess_text(text):
    return text.translate(PREPROCESS_TRANSLATION)


def wetext_post_replace(text):
    text = re.sub(r"([0-9]+)\.([0-9]+)", r"\1 point \2", text)
    text = re.sub(r"([0-9]+),([0-9]+)", r"\1\2", text)
    return text


def should_use_wetext(text):
    if EN_TN_MODE == "always":
        return True
    if EN_TN_MODE == "never":
        return False
    return bool(COMPLEX_WETEXT_RE.search(text))


def replace_punctuation(text):
    return REPLACE_PUNCTUATION_PATTERN.sub(lambda x: rep_map[x.group()], text)


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


ACRONYM_SPLIT_RE = re.compile(r'\b([A-Z]{2,})\b')
KNOWN_ACRONYMS = {
    'chatgpt': 'chat GPT',
    'openai': 'open AI',
    'github': 'git hub',
    'youtube': 'you tube',
    'linkedin': 'linked in',
    'ios': 'I OS',
}


def _split_acronym(m):
    """Split consecutive uppercase letters with spaces: GPT → G P T."""
    return " ".join(m.group(0))


def text_normalize(text):
    text = preprocess_text(text)
    for name, replacement in KNOWN_ACRONYMS.items():
        text = re.sub(rf"\b{re.escape(name)}\b", replacement, text, flags=re.I)
    # Split acronyms before normalization so G2P reads them letter-by-letter
    text = ACRONYM_SPLIT_RE.sub(_split_acronym, text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = replace_punctuation(text)
    text = PUNCT_SPACE_RE.sub(r"\1 \2", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def text_normalize_without_numbers(text):
    """Normalize an English fragment without expanding Arabic numerals."""
    text = preprocess_text(text)
    for name, replacement in KNOWN_ACRONYMS.items():
        text = re.sub(rf"\b{re.escape(name)}\b", replacement, text, flags=re.I)
    text = ACRONYM_SPLIT_RE.sub(_split_acronym, text)
    text = expand_abbreviations(text)
    text = replace_punctuation(text)
    text = PUNCT_SPACE_RE.sub(r"\1 \2", text)
    return SPACE_RE.sub(" ", text).strip()


def text_to_words(text):
    return [w for w in TOKEN_SPLIT_RE.split(text) if w.strip()]


def refine_ph(phn):
    tone = 0
    if phn[-1].isdigit():
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    else:
        tone = 3
    return phn.lower(), tone


def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for phn in phn_list:
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


def read_dict():
    g2p_dict = {}
    with open(CMU_DICT_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            word_list = line.strip().split("  ")
            if len(word_list) < 2:
                continue
            word = word_list[0]
            syllable_list = word_list[1].split(" - ")
            g2p_dict[word] = [syl.split() for syl in syllable_list]
    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(g2p_dict, f)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)
        return g2p_dict


eng_dict = get_dict()


def g2p_token(token):
    if token in punctuation:
        return (token,), (0,)
    # Uppercase A is a spelled letter (for example, OA), not the article "a".
    if token == "A":
        return ("ey",), (2,)
    if token.upper() in eng_dict:
        phns, tns = refine_syllables(eng_dict[token.upper()])
        return tuple(post_replace_ph(i) for i in phns), tuple(tns)
    phone_list = [p for p in _g2p(token) if p != " "]
    phns, tns = [], []
    for ph in phone_list:
        if ph in arpa:
            ph, tn = refine_ph(ph)
            phns.append(ph)
            tns.append(tn)
        else:
            phns.append(ph)
            tns.append(0)
    return tuple(post_replace_ph(i) for i in phns), tuple(tns)


def g2p(text):
    phones = ["_"]
    tones = [0]
    word2ph = [1]
    for token in text_to_words(text):
        token_phones, token_tones = g2p_token(token)
        phones.extend(token_phones)
        tones.extend(token_tones)
        word2ph.append(len(token_phones))
    phones.append("_")
    tones.append(0)
    word2ph.append(1)
    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text
    return phones, tones, word2ph


def get_bert_feature(text, word2ph):
    return np.zeros((1024, sum(word2ph)), dtype=np.float32)
