"""Choose the reading language of one utterance for the zh/en frontend.

Words are always spoken in the script they are written in; what the language
decides is how notation with no script of its own is read — numbers, dates,
units, symbols, model names such as Qwen3. ``auto`` infers it from the
sentence around them; ``zh``/``en`` fix it.

This only selects readings. It never translates, never rejects text, and holds
no state: the caller passes the previous utterance's language for ``auto`` to
fall back on when a sentence carries no evidence of its own (``2026 10/20``).
"""

import re
import unicodedata

from .mixed_context import context_diagnostics, protected_spans
from .name_text import _english_version_names, normalize_standalone_name
from .neutral_text import is_neutral_text, normalize_mixed_text, normalize_neutral_text

LANGUAGES = ("auto", "zh", "en")


def _scripts(text):
    scripts = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
            scripts.add("zh")
        elif name.startswith("LATIN "):
            scripts.add("en")
        else:
            scripts.add("other")
    return scripts



def _language_evidence_text(text):
    """Ignore addresses, numbers and embedded citations for language selection.

    This copy is only evidence for the current user's outer prose. The actual
    request, conversation instruction and synthesized text are not rewritten.
    """
    chars = list(text)
    for start, end, _ in protected_spans(text):
        chars[start:end] = " " * (end - start)
    masked = "".join(chars)
    pairs = {"(": ")", "（": "）", "[": "]", "【": "】", "“": "”",
             "‘": "’", "「": "」", "『": "』", "《": "》", "〈": "〉",
             '"': '"', "'": "'", "`": "`"}
    stack = []
    for index, char in enumerate(masked):
        if (char in "'’" and index and masked[index - 1].isascii()
                and masked[index - 1].isalpha()
                and (not stack or char != stack[-1][1]
                     or (index + 1 < len(masked) and masked[index + 1].isascii()
                         and masked[index + 1].isalpha()))):
            continue  # Contractions and possessives are not quotations.
        if stack and char == stack[-1][1]:
            start, _ = stack.pop()
            chars[start:index + 1] = " " * (index + 1 - start)
        elif char in pairs:
            stack.append((index, pairs[char]))
    for start, _ in stack:
        chars[start:] = " " * (len(chars) - start)
    return "".join(chars)


# These Chinese prose cues prevent an English title/function-word list from
# outweighing an unquoted Chinese request. Ambiguous bilingual clauses fall
# back to Chinese; this deliberately does not claim semantic understanding.
_ZH_PROSE_CUE = re.compile(
    r"我们|你们|他们|请|需要|允许|然后|但是|已经|正在|可以|应该|"
    r"设为|设置|检查|确认|读取|查看|使用|运行|打开|关闭|"
    r"介绍|推荐|讲讲|说说|解释|谈谈|比较|翻译|总结|怎么样|如何|是什么|怎么|"
    r"(?:我|你|他|她|它|大家)(?:想|要|能|会|在|觉得|认为|希望|打算)"
)
_EN_REQUEST_FRAME = re.compile(
    r"(?:^|[.!?;]\s*)\s*(?:please\s+(?:explain|describe|check|read|say|show|tell|"
    r"compare|translate|summarize|open|close|use|help|find|look)\b|"
    r"(?:tell|show|give)\s+(?:me|us)\b)", re.IGNORECASE,
)
_EN_PREDICATE = re.compile(
    r"\b(?:am|is|are|was|were|has|have|had|can|could|will|would|should|"
    r"must|does|do|did|may|might|shall)\b", re.IGNORECASE,
)


def _is_language_neutral_input(text):
    """Only recognized notation and complete curated names may inherit."""
    return is_neutral_text(text) or (
        _scripts(text) == {"en"}
        and _scripts(normalize_standalone_name(text, "zh")) == {"zh"}
    )


def detect_language(text):
    # A standalone curated name is not evidence of English prose. Treat its
    # spaced and joined version forms equally; explicit/ASR tags still win.
    if _is_language_neutral_input(text):
        return None
    prose = _language_evidence_text(text)
    scripts = _scripts(prose)
    if len(scripts) == 1:
        return next(iter(scripts))
    if scripts != {"zh", "en"}:
        return None
    # Short imperatives have few function words; recognize their grammatical
    # frame without comparing the length of an inserted proper name. This
    # selects the input language, not the requested answer language or intent.
    chinese_prose = bool(_ZH_PROSE_CUE.search(prose))
    if _EN_REQUEST_FRAME.search(prose) and not chinese_prose:
        return "en"
    diagnostic = context_diagnostics(prose)
    reason = diagnostic["overall_reason"]
    if reason == "chinese-grammar-around-terms":
        return "zh"
    first_letter = next((char for char in prose if char.isalpha()), "")
    if (_scripts(first_letter) == {"zh"}
            and not _EN_PREDICATE.search(prose)
            and not _EN_REQUEST_FRAME.search(prose)):
        # A Chinese lead-in plus a title/noun phrase is not English grammar
        # merely because the title repeats "the/of/the". Require an English
        # predicate or request frame before accepting that route. This keeps
        # "北京 has/is ..." eligible without enumerating Chinese request verbs.
        return None
    if (reason in {"english-grammar-around-insertion", "english-grammar-around-short-name"}
            and not chinese_prose):
        return "en"
    # Do not accept the frontend's word/character-count fallbacks as a
    # conversation language decision. A name list alone is not a sentence.
    return None


def resolve(text, preference="auto", previous=None):
    """Return the zh/en language this utterance's notation is read in."""
    if preference in ("zh", "en"):
        return preference
    detected = detect_language(text)
    if detected is not None:
        return detected
    if previous in ("zh", "en") and _is_language_neutral_input(text):
        return previous
    # A bare English name list ("OpenAI, GPT") is not English prose, but read
    # in Chinese it would gain Chinese number readings it has no reason for.
    if not is_neutral_text(text) and _scripts(normalize_standalone_name(text, "zh")) == {"en"}:
        return "en"
    return "zh"


def prepare(text, preference="auto", previous=None):
    """Return ``(frontend_text, reading_language, language)`` for one utterance.

    ``reading_language`` is what ``normalize_text_mix`` takes: empty under
    ``auto``, so a mixed sentence keeps reading each clause by its own context,
    and the fixed language otherwise. ``language`` is the resolved zh/en, which
    the caller keeps as ``previous`` for the next ``auto`` utterance.

    Anything these readers cannot handle is passed through unchanged for the
    regular frontend to read — falling back is always better than silence.
    """
    if preference not in LANGUAGES:
        raise ValueError("language must be auto, zh or en")
    language = resolve(text, preference, previous)
    reading = preference if preference != "auto" else ""
    try:
        if is_neutral_text(text):
            return normalize_neutral_text(text, language), reading, language
        prepared = text
        if reading == "en":
            prepared = _english_version_names(prepared)
        elif not reading or _scripts(prepared) == {"en"}:
            prepared = normalize_standalone_name(prepared, reading or language)
        if reading:
            prepared = normalize_mixed_text(prepared, reading)
        return prepared, reading, language
    except (ValueError, ArithmeticError):
        return text, reading, language
