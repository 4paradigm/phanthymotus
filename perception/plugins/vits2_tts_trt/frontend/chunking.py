"""Engine-agnostic frontend text chunking for VITS2 (shared across backends).

Pure text-processing helpers only — no engine dependency. Both the TensorRT and
ONNX Runtime backends split text the same way so that chunk boundaries (and thus
synthesized audio) are identical regardless of backend.
"""

from __future__ import annotations

import re

import jieba

from .copula_context import find_copula_spans


_ZH_NUMBER_UNIT_RE = re.compile(
    r"[零〇一二两三四五六七八九十百千万亿点]+"
    r"(?:K\s*B|M\s*B|G\s*B|T\s*B|P\s*B)(?:每[A-Za-z])?",
    re.IGNORECASE,
)
_SPLIT_PUNCTUATION = "，,、。．.｡！!？?；;：:…⋯—–~～\n\r"
_CLOSING_PUNCTUATION = "”’\"'」』》〉】〕〗〙〛）)]｝}"
_PUNCTUATION_UNIT_RE = re.compile(
    rf".*?[{re.escape(_SPLIT_PUNCTUATION)}]+"
    rf"[{re.escape(_CLOSING_PUNCTUATION)}]*|.+$",
    flags=re.DOTALL,
)


def _language_kind(char: str) -> str | None:
    if "一" <= char <= "鿿":
        return "ZH"
    if char.isascii() and char.isalnum():
        return "EN"
    return None


def split_positions(text: str, protected_spans=None) -> tuple[list[int], list[int], list[int]]:
    """Return (language, word, fallback) safe split positions using jieba.

    ``protected`` spans (e.g. ``2KB``) are never split across.
    """
    protected = [match.span() for match in _ZH_NUMBER_UNIT_RE.finditer(text)]
    if protected_spans is None:
        protected_spans = [(span.start, span.end) for span in find_copula_spans(text)]
    protected.extend(protected_spans)

    def is_safe(index: int) -> bool:
        return not any(start < index < end for start, end in protected)

    boundaries = []
    previous_kind = None
    for index, char in enumerate(text):
        kind = _language_kind(char)
        if kind is None:
            continue
        if previous_kind is not None and kind != previous_kind:
            if is_safe(index):
                boundaries.append(index)
        previous_kind = kind
    usable = [index for index in boundaries if 1 < index < len(text) - 1]
    word_boundaries = []
    offset = 0
    for word in jieba.cut(text):
        offset += len(word)
        if 1 < offset < len(text) - 1 and is_safe(offset):
            word_boundaries.append(offset)
    fallback = [index for index in range(1, len(text)) if is_safe(index)]
    if not fallback:
        raise ValueError("Unable to split protected number-unit expression")
    return usable, word_boundaries, fallback


def split_punctuation_units(text: str) -> list[str]:
    """Split ``text`` into punctuation-bounded units (closing punct kept)."""
    return _PUNCTUATION_UNIT_RE.findall(text)


def iter_text_chunks(
    text: str,
    token_count,
    max_tokens: int,
) -> "generator[str, None, None]":
    """Yield text chunks that each fit within ``max_tokens``.

    Shared by every backend (TensorRT / ONNX) so that chunk boundaries — and
    therefore the synthesized audio — are identical regardless of backend.
    ``token_count`` must be a callable mapping ``text -> token count``; the
    packing uses the same punctuation-unit + jieba boundary strategy as the
    rest of the frontend.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    # Resolve on the complete normalized request, before punctuation/language
    # splitting. Each final chunk retains enough evidence to recompute the
    # same immutable decision without a request-global pronunciation cache.
    contexts = find_copula_spans(text)
    protected = [(span.start, span.end) for span in contexts]

    def _iter_unit_chunks(unit, offset):
        local = [(start - offset, end - offset) for start, end in protected
                 if offset <= start and end <= offset + len(unit)]
        for start, end in local:
            if token_count(unit[start:end]) > max_tokens:
                raise ValueError("Protected copula expression exceeds the text token limit")
        if token_count(unit) <= max_tokens:
            yield unit
            return
        if len(unit) <= 1:
            raise ValueError("Unable to split text within the token limit")
        language_boundaries, word_boundaries, fallback = split_positions(unit, protected_spans=local)

        def longest_fitting(positions):
            low, high = 0, len(positions) - 1
            best = None
            while low <= high:
                middle = (low + high) // 2
                position = positions[middle]
                if token_count(unit[:position]) <= max_tokens:
                    best = position
                    low = middle + 1
                else:
                    high = middle - 1
            return best

        split_at = longest_fitting(language_boundaries)
        if split_at is None:
            split_at = longest_fitting(word_boundaries)
        if split_at is None:
            split_at = longest_fitting(fallback)
        if split_at is None:
            raise ValueError("Unable to split text within the token limit")
        left, right = unit[:split_at], unit[split_at:]
        if not left.strip() or not right.strip():
            raise ValueError("Unable to split text within the token limit")
        yield from _iter_unit_chunks(left, offset)
        yield from _iter_unit_chunks(right, offset + split_at)

    # 1) Every natural punctuation unit fits on its own first.
    units = []
    start = 0
    for match in _PUNCTUATION_UNIT_RE.finditer(text):
        end = match.end()
        if any(a < end < b for a, b in protected):
            continue
        unit = text[start:end]
        if unit.strip():
            units.append((unit, start))
        start = end
    chunks = []
    for unit, offset in units:
        chunks.extend(_iter_unit_chunks(unit, offset))

    # 2) Greedy-join already-valid units; never re-split a merged unit.
    pending = ""
    for chunk in chunks:
        if not pending:
            pending = chunk
            continue
        if token_count(pending + chunk) <= max_tokens:
            pending += chunk
            continue
        yield pending
        pending = chunk
    if pending:
        yield pending
