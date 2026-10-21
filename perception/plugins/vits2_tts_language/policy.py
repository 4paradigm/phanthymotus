"""Session language selection without translation, model imports or text rewriting.

This policy serves an explicitly Chinese/English product. Script checks cannot
identify arbitrary languages that share a writing system. A caller-provided
reply preference or ASR label takes precedence over conservative text grammar
rules. Unclear input defaults to Chinese, regardless of previous turns.
"""

from dataclasses import dataclass
import re
import threading
import unicodedata
import uuid

from ..vits2_tts_trt.frontend.mixed_context import context_diagnostics, protected_spans
from .name_text import normalize_standalone_name
from .neutral_text import is_neutral_numeric


class LanguagePolicyError(ValueError):
    """A public error whose message never contains the user's utterance."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    language: str
    turn_id: str


def _mode(value):
    if value not in ("auto", "zh", "en"):
        raise LanguagePolicyError(
            "invalid_reply_language", "reply_language must be auto, zh or en"
        )
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise LanguagePolicyError("empty_text", "A nonempty text string is required")
    return value


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


def _detect_text_language(text):
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


def validate_reply_script(
    text, language, *, allow_neutral=False, frontend=False, require_primary=True,
):
    """Check raw or normalized speech without changing any of its text.

    Chinese/English mixing is permitted. Complete replies must retain their
    selected primary script; individual frontend chunks can use either script.
    This checks script presence, not whether most words use that language.
    Neutral chunks may separate meaningful speech. The language-aware enqueue
    path can explicitly normalize a numeric reply using its resolved language;
    normalized complete speech still has to contain the selected primary script.
    """
    scripts = _scripts(text)
    code = "frontend_language_mismatch" if frontend else "reply_language_mismatch"
    if scripts - {"zh", "en"}:
        raise LanguagePolicyError(
            code,
            "Speech supports Chinese and Latin scripts only",
        )
    if scripts and require_primary and language not in scripts:
        raise LanguagePolicyError(
            code,
            "The reply must include its selected primary language",
        )
    if not scripts and not allow_neutral:
        raise LanguagePolicyError(
            "frontend_language_mismatch" if frontend else "ambiguous_reply_language",
            "Spell out numeric or symbol-only replies in the selected reply language",
        )


class LanguagePolicy:
    """One policy per conversation instance; never infer language from a reply."""

    def __init__(self, reply_language="auto"):
        self.lock = threading.RLock()
        self._mode = _mode(reply_language)
        self._resolved_language = None
        self._turn_id = None
        self._last_error = None

    def configure(self, reply_language):
        validated = _mode(reply_language)
        with self.lock:
            if validated != self._mode:
                self._mode = validated
                # A pending LLM answer no longer belongs to the new setting.
                self._turn_id = None
                self._resolved_language = None
            self._last_error = None
            return self.snapshot()

    def begin_turn(self, text, input_language=None, requested_reply_language=None):
        text = _text(text)
        if input_language is not None and input_language not in ("zh", "en"):
            raise LanguagePolicyError(
                "invalid_input_language", "input_language must be zh or en"
            )
        if requested_reply_language is not None and requested_reply_language not in ("zh", "en"):
            raise LanguagePolicyError(
                "invalid_requested_reply_language", "requested_reply_language must be zh or en"
            )
        scripts = _scripts(text)
        if "other" in scripts:
            raise LanguagePolicyError(
                "unsupported_script", "Only Chinese and Latin scripts are supported"
            )
        with self.lock:
            detected = input_language or _detect_text_language(text)
            if requested_reply_language is not None:
                resolved, source = requested_reply_language, "requested"
            elif self._mode != "auto":
                resolved, source = self._mode, "fixed"
            elif input_language is not None:
                resolved, source = input_language, "asr"
            elif detected is not None:
                resolved, source = detected, "text"
            else:
                # Do not report fallback as detection or inherit a prior turn.
                resolved, source = "zh", "default"
            # All validation precedes mutation so a rejected turn cannot cancel
            # the last accepted turn. UUIDs also prevent stop/restart token reuse.
            self._resolved_language = resolved
            self._turn_id = uuid.uuid4().hex
            self._last_error = None
            instruction = (
                "Reply mainly in Chinese. Chinese-English mixing is allowed; "
                "include English words or passages where appropriate, including "
                "names and technical terms."
                if resolved == "zh"
                else "Reply mainly in English. Chinese-English mixing is allowed; "
                "include Chinese words or passages where appropriate, including "
                "names and technical terms."
            )
            return {
                "reply_language": resolved,
                "reply_language_mode": self._mode,
                "detected_language": detected,
                "language_source": source,
                "turn_id": self._turn_id,
                "instruction": instruction,
            }

    def prepare(self, text, turn_id):
        text = _text(text)
        with self.lock:
            if not self._turn_id:
                raise LanguagePolicyError(
                    "language_turn_required", "Call action=language before speaking"
                )
            if not isinstance(turn_id, str) or turn_id != self._turn_id:
                raise LanguagePolicyError(
                    "stale_turn", "A current turn_id from action=language is required"
                )
            validate_reply_script(
                normalize_standalone_name(text, self._resolved_language),
                self._resolved_language, allow_neutral=is_neutral_numeric(text),
            )
            return SpeechRequest(text, self._resolved_language, self._turn_id)

    def record_error(self, error):
        with self.lock:
            if isinstance(error, LanguagePolicyError):
                self._last_error = {"code": error.code, "message": str(error)}
            else:
                self._last_error = {
                    "code": "invalid_request",
                    "message": "Invalid language-aware TTS request",
                }

    def clear_error(self):
        with self.lock:
            self._last_error = None

    def snapshot(self):
        with self.lock:
            return {
                "reply_language": self._mode,
                "resolved_language": self._resolved_language,
                "turn_id": self._turn_id,
                "last_language_error": dict(self._last_error) if self._last_error else None,
            }
