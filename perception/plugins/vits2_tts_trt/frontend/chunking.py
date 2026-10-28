"""Engine-agnostic frontend text chunking for VITS2 (shared across backends).

Pure text-processing helpers only — no engine dependency. Both the TensorRT and
ONNX Runtime backends split text the same way so that chunk boundaries (and thus
synthesized audio) are identical regardless of backend.
"""

from __future__ import annotations

import re
import unicodedata

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


def _connected_identifier_char(char):
    """Whether an adjacent character can make a match part of an identifier."""
    if not char:
        return False
    if char in "_-'’/@\\‐‑‒–—﹣－−" or unicodedata.category(char).startswith("M"):
        return True
    # Chinese directly adjoining an English product name is an ordinary word
    # boundary. Other letters/digits may be part of the same identifier.
    return char.isalnum() and not ("\u4e00" <= char <= "\u9fff")


def find_isolated_plan(original_text, entries):
    """Find one unambiguous, ordinary-text occurrence of an optional term.

    Addresses and identifier-shaped terms must be protected *before* text
    normalization. Ambiguous or repeated occurrences deliberately use the
    original synthesis path; this function never rewrites input text.
    """
    if not isinstance(original_text, str) or not entries:
        return None
    from .special_text import _ADDRESSES

    addresses = [match.span() for match in _ADDRESSES.finditer(original_text)]
    matches = []
    for term, mode, parts in entries:
        flags = re.ASCII | (re.IGNORECASE if mode == "fold" else 0)
        for match in re.finditer(re.escape(term), original_text, flags):
            start, end = match.span()
            left = original_text[start - 1] if start else ""
            right = original_text[end] if end < len(original_text) else ""
            if (_connected_identifier_char(left) or _connected_identifier_char(right)
                    or any(start < b and end > a for a, b in addresses)):
                return None
            # Distinguish sentence-final punctuation from dotted names such as
            # a file extension, hostname, or identifier component.
            if (left == "." and start >= 2 and _connected_identifier_char(original_text[start - 2])):
                return None
            if (right == "." and end + 1 < len(original_text)
                    and _connected_identifier_char(original_text[end + 1])):
                return None
            matches.append((term, mode, parts))
            if len(matches) > 1:
                return None
    return matches[0] if len(matches) == 1 else None


def map_isolated_term(normalized_chunk, prepared_ids, term, mode, add_blank):
    """Map a complete normalized token to its left blank in prepared IDs.

    Recompute and compare all model IDs before trusting word2ph. The returned
    blank marks a predicted model allocation, not a forced audio alignment.
    """
    if (not add_blank or not isinstance(normalized_chunk, str)
            or not isinstance(term, str) or mode not in ("exact", "fold")):
        return None
    from . import cleaned_text_to_sequence_mix
    from .cleaner import g2p_normalized_text_mix
    from .english import TOKEN_SPLIT_RE, text_to_words
    from .unified_g2p import _mixed_tokens

    # Construct the exact frontend word2ph units with source offsets. For MIX,
    # Chinese runs supply one unit per character; whitespace is still a unit.
    units = []
    if any("\u4e00" <= c <= "\u9fff" for c in normalized_chunk):
        offset = 0
        for kind, token in _mixed_tokens(normalized_chunk):
            if kind == "ZH":
                units.extend((offset + i, offset + i + 1, char)
                             for i, char in enumerate(token))
            else:
                units.append((offset, offset + len(token), token))
            offset += len(token)
        if offset != len(normalized_chunk):
            return None
    else:
        cursor = 0
        for match in TOKEN_SPLIT_RE.finditer(normalized_chunk):
            if normalized_chunk[cursor:match.start()].strip():
                units.append((cursor, match.start(), normalized_chunk[cursor:match.start()]))
            if match.group().strip():
                units.append((match.start(), match.end(), match.group()))
            cursor = match.end()
        if normalized_chunk[cursor:].strip():
            units.append((cursor, len(normalized_chunk), normalized_chunk[cursor:]))
        if [token for _, _, token in units] != text_to_words(normalized_chunk):
            return None

    candidates = [i for i, (_, _, token) in enumerate(units)
                  if (token == term if mode == "exact" else token.casefold() == term.casefold())]
    if len(candidates) != 1:
        return None
    try:
        phones, tones, langs, word2ph = g2p_normalized_text_mix(normalized_chunk)
        if len(units) != len(word2ph) - 2 or sum(word2ph) != len(phones):
            return None
        sequence = cleaned_text_to_sequence_mix(phones, tones, langs)
        interspersed = []
        for values in sequence:
            padded = [0] * (2 * len(values) + 1)
            padded[1::2] = values
            interspersed.append(tuple(padded))
        if tuple(interspersed) != tuple(tuple(values) for values in prepared_ids):
            return None
        index = candidates[0]
        first = sum(word2ph[:index + 1])
        last = first + word2ph[index + 1]
        if first == last or any(lang != "EN" for lang in langs[first:last]):
            return None
        left_blank = 2 * first
        if left_blank >= len(interspersed[0]) or interspersed[0][left_blank] != 0:
            return None
        start, end, _ = units[index]
        return start, end, left_blank
    except (AssertionError, IndexError, TypeError, ValueError):
        # The already-prepared request remains valid even if optional mapping
        # cannot be established. Do not invent a boundary on uncertainty.
        return None


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
