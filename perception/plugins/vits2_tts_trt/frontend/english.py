import pickle
import os
import re
import unicodedata
import inflect
import numpy as np

from .symbols import punctuation, symbols
from .lexicon_rules import normalize_technical_lexemes
from .release_paths import frontend_data_dir, nltk_data_dir
from .external_lexicon import lookup_unified_lexicon, bind_external_lexicon_cmu

current_file_path = os.path.dirname(__file__)

# Configure the verified release NLTK data path before importing g2p_en.
NLTK_DATA_DIR = str(nltk_data_dir())
import nltk
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

from g2p_en import G2p

from .wetext_compat import ensure_wetext_compat

EN_TN_MODE = os.getenv("EN_TN_MODE", "auto").lower()

_frontend_data_dir = frontend_data_dir()
CMU_DICT_PATH = str(_frontend_data_dir / "cmudict.rep")
CACHE_PATH = str(_frontend_data_dir / "cmudict_cache.pickle")
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
    "\n": ".", "\r": ".", "\v": ".", "\f": ".",
    "\x85": ".", "\u2028": ".", "\u2029": ".", "．": ".", "…": "...", "···": "...", "・・・": "...",
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
        ensure_wetext_compat()
        from wetext import Normalizer as tn_normalizer
        _normalizer = tn_normalizer(lang="en", operator="tn")
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
    'openai': 'open A I',
    'github': 'git hub',
    'youtube': 'you tube',
    'linkedin': 'linked in',
    'ios': 'I OS',
}
# These uppercase abbreviations share CMU keys with ordinary lowercase words.
# Expand only complete uppercase tokens, before FST can lowercase them (IP).
_EXPLICIT_UPPERCASE_ACRONYMS = frozenset({"US", "IT", "ID", "IP", "AI"})


def _spell_explicit_uppercase_acronyms(text):
    def spell(match):
        if match.group() not in _EXPLICIT_UPPERCASE_ACRONYMS:
            return match.group()
        # CJK adjacency is allowed, but an accented Latin prefix/suffix or
        # combining mark still belongs to the same word (e.g. caféAI/AIé).
        neighbors = text[max(0, match.start() - 1):match.start()] + text[match.end():match.end() + 1]
        if any(is_latin_letter(char) or unicodedata.category(char).startswith("M")
               for char in neighbors):
            return match.group()
        return " ".join(match.group())

    return _EN_LEXICON_WORD_RE.sub(spell, text)


def _split_acronym(m):
    """Keep a known pronunciation; spell unknown uppercase acronyms."""
    word = m.group(0)
    if english_pronunciation(word) is not None:
        return word
    return " ".join(m.group(0))


def _split_unknown_acronyms(text):
    """Keep known dictionary words, including hyphens and apostrophes, intact."""
    text = _spell_explicit_uppercase_acronyms(text)
    protected = [
        match.span()
        for match in _EN_LEXICON_WORD_RE.finditer(text)
        if english_pronunciation(match.group()) is not None
    ]

    def replace(match):
        if any(start <= match.start() and match.end() <= end
               for start, end in protected):
            return match.group()
        return _split_acronym(match)

    return ACRONYM_SPLIT_RE.sub(replace, text)


def text_normalize(text):
    text = preprocess_text(text)
    text = normalize_technical_lexemes(text)
    for name, replacement in KNOWN_ACRONYMS.items():
        text = re.sub(rf"\b{re.escape(name)}\b", replacement, text, flags=re.I)
    # Split acronyms before normalization so G2P reads them letter-by-letter
    text = _split_unknown_acronyms(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = replace_punctuation(text)
    text = PUNCT_SPACE_RE.sub(r"\1 \2", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def text_normalize_without_numbers(text):
    """Normalize an English fragment without expanding Arabic numerals."""
    text = preprocess_text(text)
    text = normalize_technical_lexemes(text)
    for name, replacement in KNOWN_ACRONYMS.items():
        text = re.sub(rf"\b{re.escape(name)}\b", replacement, text, flags=re.I)
    text = _split_unknown_acronyms(text)
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
            if not line.strip() or line.lstrip().startswith("##"):
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
# Startup verifies the external resource before any request can claim to use
# the expanded dictionary. A missing/corrupt configured file is not a fallback.
_external_lexicon_status = bind_external_lexicon_cmu(eng_dict)


CUSTOM_EN_PRONUNCIATIONS = {
    "CUDA": [["K", "UW1", "D", "AH0"]],
    "HAMI": [["HH", "AE1", "M", "IY0"]],
    "IELTS": [["AY1"], ["EH0", "L", "T", "S"]],
}


# Narrow corrections from the IPA retained in the external lexicon IPA reference section.
# Keep them separate from Liaoqianqian's original three CUSTOM entries.
VERIFIED_EN_PRONUNCIATIONS = {
    # wordlist_37, i-wangtianshuo: /əˈdʒɛntɪk/; g2p_en adds a final T.
    "AGENTIC": [["AH0", "JH", "EH1", "N", "T", "IH0", "K"]],
    # user_terms_397, zhangjinghong: /ɛnˈvɪdiə/, stressed on the second syllable.
    "NVIDIA": [["EH0", "N", "V", "IH1", "D", "IY0", "AH0"]],
}


def english_pronunciation(token):
    """CUSTOM/verified terms, protected CMU and the unified Gold index.

    Return None for G2P fallback. The optimized external dictionary never
    overrides ordinary CMU words, and its exact-case rows retain ticker scope.
    """
    word = token.upper()
    if word in CUSTOM_EN_PRONUNCIATIONS:
        return CUSTOM_EN_PRONUNCIATIONS[word]
    if word in VERIFIED_EN_PRONUNCIATIONS:
        return VERIFIED_EN_PRONUNCIATIONS[word]
    return lookup_unified_lexicon(token, has_cmu=word in eng_dict) or eng_dict.get(word)


_EN_LEXICON_WORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*(?:['.-][A-Za-z0-9_]+)*(?![A-Za-z0-9_])"
)
_EN_TN_CONTEXT_WORDS = frozenset({
    "USD", "EUR", "GBP", "CNY", "RMB", "DR", "MR", "MRS", "MS", "ST",
})
# The released Chinese FST splits CAR and rewrites the internal "log" in
# kilogram(s) as 对数. Protect complete words only, including unit words
# already expanded by the language layer; pronunciation still uses CMU/G2P.
_EN_TN_PRESERVED_WORDS = frozenset({"CAR", "KILOGRAM", "KILOGRAMS"})
# Lexeme rules already spell initialisms such as AI/API as A I/A P I.
# Keep that explicit spelling together so sentence-final I. is not interpreted
# as a Roman list number. A standalone I./II./IV. still belongs to the FST.
_SPELLED_LETTERS_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?<![A-Za-z0-9_]['.-])"
    r"[A-Z](?:[ \t]+[A-Z])+(?![A-Za-z0-9_]|['.-][A-Za-z0-9_])"
)


def normalize_preserving_english_pronunciations(text, normalize_plain):
    """Preserve explicit letter spellings and selected complete words in TN.

    Called only on ordinary spans after the existing address, number, unit and
    identifier handlers. Explicit lexeme rewrites retain priority. Versioned
    model names are deliberately not split here: their digits and surrounding
    context remain owned by the existing frontend.

    Custom/verified terms and known FST exceptions keep their spelling. Other
    words follow the existing text normalizer and CMU/G2P path.
    """
    text = _spell_explicit_uppercase_acronyms(text)
    protected = list(_SPELLED_LETTERS_RE.finditer(text))
    for match in _EN_LEXICON_WORD_RE.finditer(text):
        word = match.group().upper()
        if (len(word) >= 3 and word not in _EN_TN_CONTEXT_WORDS
                and (word in CUSTOM_EN_PRONUNCIATIONS
                     or word in VERIFIED_EN_PRONUNCIATIONS
                     or word in _EN_TN_PRESERVED_WORDS
                     or lookup_unified_lexicon(match.group(), has_cmu=word in eng_dict) is not None)):
            protected.append(match)
    if not protected:
        return normalize_plain(text)
    if re.search(r"[\r\n]", text):
        # A protected token can leave a newline at the edge of an FST span,
        # where it is stripped. Keep paragraph boundaries for punctuation TN.
        return "\n".join(
            normalize_preserving_english_pronunciations(line, normalize_plain)
            for line in re.split(r"\r\n|\r|\n", text)
        )
    output, cursor = [], 0
    for match in sorted(protected, key=lambda item: item.start()):
        if cursor < match.start():
            output.append(normalize_plain(text[cursor:match.start()]))
        output.append(match.group())
        cursor = match.end()
    if cursor < len(text):
        output.append(normalize_plain(text[cursor:]))
    return " ".join(part.strip() for part in output if part.strip())



def is_latin_letter(char):
    """Recognize Latin letters without treating CJK, Greek or math as English."""
    return char.isalpha() and "LATIN" in unicodedata.name(char, "")


def iter_latin_words(text):
    """Yield complete Latin words, retaining attached marks and identifier tails."""
    index = 0
    while index < len(text):
        if not is_latin_letter(text[index]):
            index += 1
            continue
        start = index
        index += 1
        while index < len(text):
            char = text[index]
            if is_latin_letter(char) or unicodedata.category(char).startswith("M") or char.isascii() and (char.isdigit() or char == "_"):
                index += 1
            elif char in "'’-" and index + 1 < len(text) and (is_latin_letter(text[index + 1]) or text[index + 1].isascii() and text[index + 1].isdigit()):
                index += 1
            else:
                break
        yield start, index


def normalize_preserving_external_tokens(text, normalize_plain):
    """Keep eligible complete hyphenated lexemes through final punctuation TN.

    Structured formulas and addresses are resolved by the caller first. This
    operates only on remaining known external words, never on substring hits,
    arbitrary minus signs, or CMU words whose existing behavior is unchanged.
    """
    spans = []
    for start, end in iter_latin_words(text):
        word = text[start:end]
        if "-" not in word:
            continue
        if (start and text[start - 1].isascii()
                and (text[start - 1].isalnum() or text[start - 1] == "_")):
            continue
        if lookup_unified_lexicon(word, has_cmu=word.upper() in eng_dict) is not None:
            spans.append((start, end))
    if not spans:
        return normalize_plain(text)
    parts, cursor = [], 0
    for start, end in spans:
        if cursor < start:
            parts.append(normalize_plain(text[cursor:start]))
        parts.append(text[start:end])
        cursor = end
    if cursor < len(text):
        parts.append(normalize_plain(text[cursor:]))
    return "".join(parts)


def normalize_ordinary_latin_text(text, normalize_plain):
    """Keep ordinary English prose out of the Chinese FST.

    Structured quantities, addresses and identifiers are resolved by the caller
    first. Remaining alphanumeric identifiers stay on the original TN path.
    Latin words use the existing number-free English cleanup and shared G2P.
    Line boundaries remain explicit until punctuation conversion.
    """
    if re.search(r"[\r\n\v\f\x85\u2028\u2029]", text):
        return "\n".join(
            normalize_ordinary_latin_text(line, normalize_plain)
            for line in re.split(r"\r\n|[\r\n\v\f\x85\u2028\u2029]", text)
        )
    # A title before a capitalized name, including CJK-adjacent titles. Bare
    # DR and road abbreviations retain CMU; accented Latin prefixes are still
    # part of a word and cannot create a title boundary.
    def doctor_title(match):
        previous = text[match.start() - 1] if match.start() else ""
        if previous and (is_latin_letter(previous)
                         or unicodedata.category(previous).startswith("M")):
            return match.group()
        return "doctor"

    text = re.sub(r"(?<![A-Za-z0-9_])(?i:dr)\.(?=[ \t]+[A-ZÀ-ÖØ-Þ])",
                  doctor_title, text)
    # Keep standalone line-initial Roman list markers on the legacy FST route,
    # while a sentence-final pronoun in "You and I." remains an English word.
    romans = [match.span() for match in re.finditer(
        r"(?<![A-Za-z])(?:[IVXLCDM]{2,}\.|^[ \t]*[IVXLCDM]\.)(?=$|[ \t]+\S)", text
    )]
    pieces, cursor = [], 0
    for start, end in iter_latin_words(text):
        word = text[start:end]
        if (any(c.isdigit() for c in word)
                or start and (text[start - 1].isascii() and (text[start - 1].isalnum() or text[start - 1] == "_"))
                or any(a <= start < b for a, b in romans)):
            continue
        if cursor < start:
            pieces.append(normalize_plain(text[cursor:start]))
        if "_" in word:
            # The release FST also contains product rewrites that ignore
            # underscore identifier boundaries. Preserve nonnumeric identifier
            # pieces here; final punctuation cleanup still voices each piece
            # separately. Keep explicit uppercase letter spellings, but do not
            # apply product aliases inside the identifier.
            pieces.append("_".join(_spell_explicit_uppercase_acronyms(part)
                                   for part in word.split("_")))
        else:
            pieces.append(text_normalize_without_numbers(unicodedata.normalize("NFC", word)))
        cursor = end
    if cursor < len(text):
        pieces.append(normalize_plain(text[cursor:]))
    normalized = " ".join(piece.strip() for piece in pieces if piece.strip())
    return re.sub(r" +([,.!?;:])", r"\1", normalized)


def g2p_token(token):
    token = unicodedata.normalize("NFC", token)
    if token in punctuation:
        return (token,), (0,)
    # Uppercase A is a spelled letter (for example, OA), not the article "a".
    if token == "A":
        return ("ey",), (2,)
    syllables = english_pronunciation(token)
    if syllables is not None:
        phns, tns = refine_syllables(syllables)
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
