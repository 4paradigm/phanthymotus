"""Deterministic lexeme rules for the ZH/EN mixed VITS2 frontend.

Adds: `Qwen` -> 千问 and Chinese readings for accelerator model digits.

Applied to the raw text **before** the WeText FST normalizer, because the graph
otherwise rewrites English tokens it does not own:

* ``IDC`` -> ``I don't care`` (semantic corruption)
* ``Tier 1`` -> ``Tier 一``  (digits in an English context read as Chinese)

The rules also pin the reading of acronyms and camel-case product names that the
graph leaves untouched and that ``g2p_en`` then guesses wrongly, e.g.
``DCU`` -> ``d uw k`` ("duke"), ``AutoML`` -> ``ao t ah m`` (ML lost).

Separators:

* ``CNCF`` is the one acronym that uses dots between letters (``C.N.C.F``).
  With a plain space its four letters render as a single continuous 770 ms block
  with no measurable gap, which is heard as one blur; the dot yields 40-70 ms
  micro-gaps between letters. Every other acronym keeps the space-separated
  spelling it already had.
* A phrase that needs a longer audible gap gets a comma, because ``unified_g2p``
  maps whitespace to a zero-phone boundary and a comma is a real pause symbol.

Boundaries use ASCII lookarounds instead of ``\\b``: CJK characters are ``\\w``
in Python's Unicode regex, so ``\\b`` never matches between 中文 and an ASCII
word (``\\bdcu\\b`` fails on ``这是DCU的测试``).

Every entry is deliberately narrow — one release-controlled rewrite per lexeme —
so training and deployed runtime keep producing identical phonemes.
"""

import re
import unicodedata
from .lexicon_rules_generated import GENERATED_REPLACEMENTS
from .math_units import _english_quantity
from .special_text import verbalize_chinese_number


def _edge(pattern: str) -> re.Pattern:
    """Match ``pattern`` as a standalone ASCII token, CJK included."""
    return re.compile(rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])", re.IGNORECASE)


def _sub_lexeme(pattern, replacement, text):
    """Keep complete Latin words while still matching names next to CJK."""
    def replace(match):
        neighbors = text[max(0, match.start() - 1):match.start()] + text[match.end():match.end() + 1]
        if any(unicodedata.category(char).startswith("M")
               or char.isalpha() and "LATIN" in unicodedata.name(char, "")
               for char in neighbors):
            return match.group()
        return replacement(match) if callable(replacement) else match.expand(replacement)
    return pattern.sub(replace, text)


# 0) Phrase-level rules. These must run before the per-token rules below,
#    otherwise "Sage AIOS" is already split into letters and never matches.
_PHRASE_RULES = [
    # Keep a short pause between the brand and the acronym it introduces.
    (_edge(r"sage\s+aios"), "Sage, A I O S"),
]

# 1) Digits that belong to an English phrase must not be verbalized as Chinese.
_NUMBER_RULES = [
    (_edge(r"tier\s*1"), "Tier one"),
]

# 2) Tokens the English TN graph mangles before G2P ever sees them.
_PROTECTED_TOKENS = [
    (_edge(r"idc"), "I D C"),
]

# 3) Acronyms that are read letter by letter. CNCF uses dots: space-separated
#    letters come out as one continuous block (770 ms, no gap).
_LETTER_SPELLED = [
    (re.compile(r"(?<![A-Za-z0-9_])waic(?![A-Za-z0-9_])", re.I), "W A I C"),
    (_edge(r"aios"), "A I O S"),
    (_edge(r"dcu"), "D C U"),
    (_edge(r"mlu"), "M L U"),
    (_edge(r"cncf"), "C.N.C.F"),
    (_edge(r"xc"), "X C"),
    (_edge(r"vgpu"), "V G P U"),
    (_edge(r"gpu"), "G P U"),
    (_edge(r"npu"), "N P U"),
    (_edge(r"api"), "A P I"),
    (_edge(r"ai"), "A I"),
]

# 4) Mixed readings: the word part is spelled, the trailing letter is read.
_MIXED_READINGS = [
    (_edge(r"automl"), "Auto M L"),
    (_edge(r"enginex"), "Engine X"),
]

# 5) Camel-case / fused product names that must become two words.
_FUSED_NAMES = [
    # Restore the product readings from the integrated 2026-10-14 release.
    # Underscores belong to identifiers, so do not rewrite embedded names.
    (re.compile(r"(?<![A-Za-z0-9_])openmldb(?![A-Za-z0-9_])", re.I), "Open M L D B"),
    (re.compile(r"(?<![A-Za-z0-9_])featinsight(?![A-Za-z0-9_])", re.I), "Feat Insight"),
    (_edge(r"modelhub"), "Model Hub"),
    (_edge(r"phanthymotus"), "Phanthy Motus"),
    (_edge(r"phanthymovie"), "Phanthy Movie"),
    (_edge(r"phanthycloud"), "Phanthy Cloud"),
    (_edge(r"phancycloud"), "Phancy Cloud"),
    # Split so that "Router" resolves through CMUdict: /ˈruːtər/, not the
    # g2p_en guess /ˈraʊtər/ the single fused token produced.
    (_edge(r"phanrouter"), "Phan Router"),
    (_edge(r"phanclaw"), "Phan Claw"),
]


# 6) Chinese readings: brand names spoken as Chinese, and the digits of
#    accelerator model numbers read as Chinese numerals (A100 -> A一百,
#    MI250X -> M I二五零X). Round hundreds keep the natural form (一百/八百),
#    other three-digit numbers are read digit by digit (九一零/三七零).
_CN_NAMES = [
    (_edge(r"qwen"), "千问"),
    # The generated source contains a literal "+" used as an annotation
    # separator. Keep its data intact, but never send that sign to TTS.
    (_edge(r"01\.AI"), "零一 点 A I"),
]


def _ascend_model(match) -> str:
    """Keep model digit readings only with an explicit Ascend/昇腾 prefix."""
    model = match.group("model")
    digits = "三一零" if model.startswith("310") else "九一零"
    return match.group("brand") + " " + digits + model[3:]


_CONTEXT_MODEL_DIGITS = [
    (_edge(r"(?P<brand>Ascend|昇腾)[ \t]*(?:[-‑–—][ \t]*)?"
           r"(?P<model>310P?|910[ABC]?)"), _ascend_model),
]

_MODEL_DIGITS = [
    (_edge(r"MI250X"), "M I二五零X"),
    (_edge(r"MI300A"), "M I三百A"),
    (_edge(r"MI300X"), "M I三百X"),
    (_edge(r"MI325X"), "M I三二五X"),
    (_edge(r"MI355X"), "M I三五五X"),
    (_edge(r"MLU100"), "M L U一百"),
    (_edge(r"MLU270"), "M L U二七零"),
    (_edge(r"MLU290"), "M L U二九零"),
    (_edge(r"MLU370"), "M L U三七零"),
    (_edge(r"MLU590"), "M L U五九零"),
    (_edge(r"GB200"), "G B二百"),
    (_edge(r"GB300"), "G B三百"),
    (_edge(r"MI210"), "M I二一零"),
    (_edge(r"MI250"), "M I二五零"),
    (_edge(r"MI350"), "M I三五零"),
    (_edge(r"950PR"), "九五零PR"),
    (_edge(r"950DT"), "九五零DT"),
    (_edge(r"WSE-2"), "W S E二"),
    (_edge(r"WSE-3"), "W S E三"),
    (_edge(r"SN40L"), "S N四零L"),
    (_edge(r"V100"), "V一百"),
    (_edge(r"A100"), "A一百"),
    (_edge(r"A800"), "A八百"),
    (_edge(r"H100"), "H一百"),
    (_edge(r"H800"), "H八百"),
    (_edge(r"H200"), "H二百"),
    (_edge(r"B100"), "B一百"),
    (_edge(r"B200"), "B二百"),
    (_edge(r"B300"), "B三百"),
    (_edge(r"L40S"), "L四零S"),
    (_edge(r"310P"), "三一零P"),
    (_edge(r"910A"), "九一零A"),
    (_edge(r"910B"), "九一零B"),
    (_edge(r"910C"), "九一零C"),
    (_edge(r"AI2"), "A I二"),
    (_edge(r"A3C"), "A三C"),
    (_edge(r"A2C"), "A二C"),
    (_edge(r"TD3"), "T D三"),
    (_edge(r"H20"), "H二零"),
    (_edge(r"L40"), "L四零"),
    (_edge(r"v5e"), "v五e"),
    (_edge(r"v5p"), "v五p"),
    (_edge(r"v6e"), "v六e"),
    (_edge(r"C3"), "C三"),
    (_edge(r"T5"), "T五"),
    (_edge(r"F1"), "F一"),
    (_edge(r"L4"), "L四"),
    (_edge(r"T4"), "T四"),
    (_edge(r"v4"), "v四"),
    (_edge(r"v7"), "v七"),
    (_edge(r"D1"), "D一"),
]

# Order matters: phrase rules first, then longer alternatives before shorter ones
# (``aios`` before ``ai``, ``vgpu`` before ``gpu``).

RULES = (
    _PHRASE_RULES
    + _CN_NAMES
    + _CONTEXT_MODEL_DIGITS
    + _MODEL_DIGITS
    + _NUMBER_RULES
    + _PROTECTED_TOKENS
    + _LETTER_SPELLED
    + _MIXED_READINGS
    + _FUSED_NAMES
)




# 7) Generated lexeme rules from frontend_res. The complete plain-text
#    mapping is retained in lexicon_rules_generated.py; only entries
#    whose spoken form differs from the source term are compiled.
def _generated_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _generated_pattern(value: str) -> str:
    parts = re.split(r"\s+", value.strip())
    # A line break separates utterances/items, not words of a company name.
    return r"[^\S\r\n\v\f\x85\u2028\u2029]+".join(re.escape(part) for part in parts)


_GENERATED_NUMBER = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?=$|\s|[A-Z](?![A-Za-z0-9_]))")


def _generated_spoken(spoken, language):
    """Verbalize numbers introduced by a reviewed mapping before the FST.

    The original structured-number pass already ran before lexeme rewriting.
    A generated ``Velo 3D`` must therefore separate D and resolve 3 here rather
    than sending it back through an unqualified Chinese FST. Only numbers in
    matched, generated spoken forms are handled; arbitrary input is untouched.
    """
    def replace(match):
        if language == "EN":
            number = _english_quantity(match.group())
        else:
            number = verbalize_chinese_number(match.group())
        return " " + number + " "
    return re.sub(r" +", " ", _GENERATED_NUMBER.sub(replace, spoken)).strip()


def _pattern_literal_key(pattern) -> str:
    source = pattern.pattern
    for prefix in (r"(?<![A-Za-z0-9])", r"(?<![A-Za-z0-9_])"):
        if source.startswith(prefix):
            source = source[len(prefix):]
    for suffix in (r"(?![A-Za-z0-9])", r"(?![A-Za-z0-9_])"):
        if source.endswith(suffix):
            source = source[:-len(suffix)]
    source = source.replace(r"\s+", " ").replace(r"\s*", " ")
    source = re.sub(r"\\(.)", r"\1", source)
    return re.sub(r"\s+", " ", source).strip().casefold()


_CURATED_KEYS = {_pattern_literal_key(pattern) for pattern, _ in RULES}
_GENERATED_RULES = []
for _term, _spoken in GENERATED_REPLACEMENTS.items():
    _key = _generated_key(_term)
    # Bare numerals are quantities, dates, identifiers or model numbers only
    # in context. Leave them to the numeric frontend instead of globally
    # applying the generated company/name entry "111" -> "one one one".
    if _term.strip().isdecimal():
        continue
    if len(_term.strip()) < 2 or _term.casefold() == _spoken.casefold():
        continue
    if _key in _CURATED_KEYS:
        continue
    _pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + _generated_pattern(_term)
        + r")(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    _GENERATED_RULES.append((_pattern, _spoken, len(_term)))
_GENERATED_RULES.sort(key=lambda item: item[2], reverse=True)
_GENERATED_RULES = [(pattern, spoken) for pattern, spoken, _ in _GENERATED_RULES]
_NUMERIC_GENERATED_RULES = [
    (pattern, spoken) for pattern, spoken in _GENERATED_RULES
    if _GENERATED_NUMBER.search(spoken)
]
_STYLE_TTS_VERSION = _edge(r"StyleTTS2")


def normalize_numeric_lexemes(text, language="MIX"):
    """Resolve only known lexical numbers, preserving a caller's preference.

    Fixed-language callers may run this before clause-local routing. The normal
    lexeme pass also uses it for automatic EN/MIX context. Fully spoken output
    is idempotent and ordinary quantities/identifiers remain untouched.
    """
    for pattern, spoken in _NUMERIC_GENERATED_RULES:
        text = _sub_lexeme(pattern, lambda match: _generated_spoken(spoken, language), text)
    return _sub_lexeme(
        _STYLE_TTS_VERSION, lambda match: _generated_spoken("style T T S 2", language), text,
    )


_QWEN_VERSION_RE = re.compile(
    # A typed space between the brand and version does not change its reading.
    # Limit separators to horizontal whitespace: a newline starts a new item.
    r"(?<![A-Za-z0-9_])qwen[ \t\u00a0\u3000]*(\d+(?:\.\d+)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CN_DIGITS = "零一二三四五六七八九"


def _qwen_version(match) -> str:
    """Qwen3 / Qwen 3 -> 千问三; Qwen 2.5 -> 千问二点五."""
    digits = match.group(1)
    spoken = "".join(
        _CN_DIGITS[int(ch)] if ch.isdigit() else "点" for ch in digits
    )
    return "千问" + spoken


def has_chinese_name_reading(token: str) -> bool:
    """Whether one complete name has a curated, entirely Chinese reading.

    Used for standalone names in a Chinese list, not to translate English
    clauses or arbitrary substrings. Numbered Qwen names keep the same bounds
    and version grammar as the existing pronunciation rule.
    """
    if _QWEN_VERSION_RE.fullmatch(token):
        return True
    return any(
        pattern.fullmatch(token)
        and re.search(r"[\u4e00-\u9fff]", spoken)
        and not re.search(r"[A-Za-z]", spoken)
        for pattern, spoken in _CN_NAMES
    )


def apply_lexeme_rules(text: str, language: str = "MIX") -> str:
    """Rewrite known English lexemes into their intended spoken form."""
    if not text:
        return text
    text = normalize_numeric_lexemes(text, language)

    # Protect generated frontend_res matches from the older broad rules.
    protected = []
    # Choose a prefix absent from user input, and encode the index entirely in
    # private-use characters. Neither input resembling old markers nor later
    # ASCII lexeme rules can collide with these opaque placeholders.
    prefix = "\ue000"
    while prefix in text:
        prefix += "\ue000"
    for pattern, replacement in _GENERATED_RULES:
        if language == "EN" and re.search(r"[\u4e00-\u9fff]", replacement):
            continue
        def _protect(match):
            encoded = "".join(chr(0xE010 + int(digit, 16)) for digit in format(len(protected), "x"))
            marker = prefix + "\ue001" + encoded + "\ue002"
            protected.append((marker, _generated_spoken(replacement, language)))
            return marker
        text = _sub_lexeme(pattern, _protect, text)

    if language != "EN":
        text = _sub_lexeme(_QWEN_VERSION_RE, _qwen_version, text)
    for pattern, replacement in RULES:
        if language == "EN":
            def _english_replacement(match):
                spoken = replacement(match) if callable(replacement) else match.expand(replacement)
                return match.group() if re.search(r"[\u4e00-\u9fff]", spoken) else spoken
            text = _sub_lexeme(pattern, _english_replacement, text)
        else:
            text = _sub_lexeme(pattern, replacement, text)
    for marker, replacement in protected:
        text = text.replace(marker, replacement)
    return text
