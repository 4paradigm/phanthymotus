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

from functools import lru_cache
import json
import re
import unicodedata
from .release_paths import frontend_lexicon_path
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


def _ascend_model(match) -> str:
    """Keep model digit readings only with an explicit Ascend/昇腾 prefix."""
    model = match.group("model")
    digits = "三一零" if model.startswith("310") else "九一零"
    return match.group("brand") + " " + digits + model[3:]



# 7) Generated lexeme rules from frontend_res. The complete plain-text
#    mapping is retained in the external lexicon's text_replacements section;
#    only entries
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


def _compile_rules(rows):
    """Decode pinned, ordered rules without executing code from the resource."""
    if not isinstance(rows, list) or not rows or len(rows) > 4096:
        raise ValueError("VITS2 frontend rules require a nonempty bounded list")
    result = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("pattern"), str)
                or not row["pattern"] or type(row.get("flags")) is not int
                or row["flags"] & ~(re.IGNORECASE | re.UNICODE)):
            raise ValueError("Invalid VITS2 frontend pattern")
        if set(row) == {"pattern", "flags", "replacement"} and isinstance(row["replacement"], str):
            replacement = row["replacement"]
        elif set(row) == {"pattern", "flags", "handler"} and row["handler"] == "ascend_model":
            replacement = _ascend_model
        else:
            raise ValueError("Invalid VITS2 frontend replacement")
        result.append((re.compile(row["pattern"], row["flags"]), replacement))
    return tuple(result)


@lru_cache(maxsize=4)
def _generated_tables(path):
    """Compile verified external data once, after adapter resource preparation.

    Preserve insertion order for equal-length terms. Importing the language
    policy never opens a file or starts a network request.
    """
    with open(path, "rb") as source:
        data = source.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("VITS2 frontend resource exceeds 4 MiB")
    resource = json.loads(data)
    replacements = resource.get("text_replacements") if isinstance(resource, dict) else None
    if (not isinstance(replacements, dict) or not replacements
            or any(not isinstance(term, str) or not isinstance(spoken, str)
                   for term, spoken in replacements.items())):
        raise ValueError("VITS2 text replacements require a nonempty string mapping")
    frontend_rules = resource.get("frontend_rules")
    if not isinstance(frontend_rules, dict):
        raise ValueError("VITS2 frontend rules are missing")
    tables = {name: _compile_rules(frontend_rules.get(name))
              for name in ("before_tn", "after_tn", "numeric_names")}
    curated_keys = {_pattern_literal_key(pattern) for pattern, _ in tables["before_tn"]}
    rules = []
    for term, spoken in replacements.items():
        # Bare numbers belong to the contextual numeric frontend.
        if (term.strip().isdecimal() or len(term.strip()) < 2
                or term.casefold() == spoken.casefold()
                or _generated_key(term) in curated_keys):
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9_])(?:" + _generated_pattern(term)
                             + r")(?![A-Za-z0-9_])", re.IGNORECASE)
        rules.append((pattern, spoken, len(term)))
    rules.sort(key=lambda item: item[2], reverse=True)
    generated = tuple((pattern, spoken) for pattern, spoken, _ in rules)
    numeric = tuple((pattern, spoken) for pattern, spoken in generated
                    if _GENERATED_NUMBER.search(spoken))
    tables.update(generated=generated, numeric=numeric)
    return tables


def _frontend_tables():
    return _generated_tables(str(frontend_lexicon_path()))


def _generated_rules():
    tables = _frontend_tables()
    return tables["generated"], tables["numeric"]


def normalize_technical_lexemes(text: str) -> str:
    """Apply the release's post-TN product spellings at their original stage."""
    for pattern, replacement in _frontend_tables()["after_tn"]:
        text = _sub_lexeme(pattern, replacement, text)
    return text


def normalize_numeric_lexemes(text, language="MIX"):
    """Resolve only known lexical numbers, preserving a caller's preference.

    Fixed-language callers may run this before clause-local routing. The normal
    lexeme pass also uses it for automatic EN/MIX context. Fully spoken output
    is idempotent and ordinary quantities/identifiers remain untouched.
    """
    tables = _frontend_tables()
    for pattern, spoken in tables["numeric"] + tables["numeric_names"]:
        text = _sub_lexeme(pattern, lambda match: _generated_spoken(spoken, language), text)
    return text


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
    return bool(re.fullmatch("qwen", token, re.IGNORECASE))


def normalize_chinese_name(token: str) -> str:
    """Known standalone names used by the language API before model readiness."""
    token = _sub_lexeme(_QWEN_VERSION_RE, _qwen_version, token)
    return _sub_lexeme(_edge("qwen"), "千问", token)


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
    for pattern, replacement in _generated_rules()[0]:
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
    for pattern, replacement in _frontend_tables()["before_tn"]:
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
