"""Xiaoyou text-to-phoneme frontend with complete normalized-span context.

One implementation owns initialization, explicit language normalization and
Misaki conversion. Text/phoneme/token ABI and scoped rules match the reviewed
runtime; model configuration and term paths remain externally injectable.
"""
from __future__ import annotations

from hashlib import sha256
from contextvars import ContextVar
from contextlib import contextmanager
from concurrent.futures import CancelledError
from bisect import bisect_right
from functools import lru_cache
from importlib.metadata import version
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import regex

from plugins.vits2_tts_trt.frontend.special_text import read_numeric_literal

from . import numbers as numeric
from ..runtime.model_bundle import (
    FRONTEND_FILE, NORMALIZATION_MODULE_SHAS, TERMS_FILE, detect_jetpack_major, unpack_jp6_english_model,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_LATIN_WORD = r"(?:[0-9]+(?=[A-Za-z]))?[A-Za-z][A-Za-z0-9_'’\-]*"
_LATIN = re.compile(_LATIN_WORD + r"(?:[ \t]+" + _LATIN_WORD + r")*")
_HAN = re.compile(r'[\u3400-\u9fff]')
_EMOJI = regex.compile(
    r'(?P<key>[0-9#*])\ufe0f?\u20e3'
    r'|(?=\p{Emoji_Presentation}|\p{Extended_Pictographic}(?:[\ufe0f\u200d]|\p{Emoji_Modifier}))\X'
    r'|[♥☀☺☹]\ufe0e?'
)

_VERSION_NUMBER = r'[0-9]+(?:\.[0-9]+){0,3}'
_VERSION_LEFT = r'(?<![A-Za-z0-9_./\\-])'
_VERSION_RIGHT = r'(?![A-Za-z0-9_/\\-]|\.[A-Za-z0-9_.])'
_VERSION = re.compile(
    r'(?:' + _VERSION_LEFT + r'(?P<v>[vV])(?P<marked>' + _VERSION_NUMBER + r')|'
    r'(?P<label>' + _VERSION_LEFT + r'version(?:[ \t]+(?:is|was))?[ \t]*[:=]?[ \t]*|'
    r'版本(?:号)?[ \t]*(?:[为是:：=][ \t]*)?)(?P<labelled>' + _VERSION_NUMBER + r')|'
    # Known software followed immediately by its version is an explicit
    # identity too. The guards still exclude files, paths and suffixed IDs.
    + _VERSION_LEFT + r'(?:PyTorch|CUDA|TensorRT|cuDNN)[ \t]*(?:v[ \t]*)?'
    r'(?P<software_number>' + _VERSION_NUMBER + r'))'
    + _VERSION_RIGHT, re.I)
_VERSION_URL = re.compile(r'(?:https?://|ftp://|www\.)[^\s<>，。；！？]+', re.I)

def _version_spans(normalized, start, end, addresses=()):
    excluded = [m.span() for m in _VERSION_URL.finditer(normalized)]
    excluded.extend(addresses)
    result = []
    for match in _VERSION.finditer(normalized):
        if any(a < match.end() and match.start() < b for a, b in excluded):
            continue
        group = next(name for name in ('marked', 'labelled', 'software_number')
                     if match[name] is not None)
        a, b = match.span(group)
        if a < start < b or a < end < b:
            raise ValueError('Semantic span splits an explicit version component sequence')
        if start <= a < b <= end:
            result.append((a, b, match[group]))
    return result

def _version_language(normalized, position, language, auto_fallback=None):
    if language in ('zh', 'en'):
        return language
    numeric._validate_auto_fallback(auto_fallback)
    cursor = 0
    fallback = auto_fallback
    for clause in numeric._clauses(normalized):
        if fallback is not None:
            fallback, _ = numeric.auto_context_language(clause, fallback)
        if cursor <= position < cursor + len(clause):
            return numeric._language(clause, language, 'zh') if fallback is None else fallback
        cursor += len(clause)
    raise ValueError('Version position is outside the complete normalized request')

def _version_reading(value, language):
    components = value.split('.')
    readings = [numeric.speak_number(part, language, len(part) > 1 and part.startswith('0'))
                for part in components]
    return ('点' if language == 'zh' else ' point ').join(readings)

def _filename_reading(value, language):
    """Speak literal structure explicitly, with digit identities preserved."""
    parts = re.findall(r'[A-Za-z]+|[0-9]+|[_\.\-]', value)
    if ''.join(parts) != value:
        raise ValueError('Unsupported filename literal')
    separators = {'zh': {'_': '下划线', '.': '点', '-': '连字符'},
                  'en': {'_': 'underscore', '.': 'dot', '-': 'hyphen'}}[language]
    rendered = []
    for part in parts:
        if part.isdigit():
            rendered.append(read_numeric_literal(part, language))
        else:
            rendered.append(separators.get(part, part))
    result = ' '.join(rendered)
    return re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])', '', result)

@lru_cache(maxsize=32)
def _protected_term_spans(normalized):
    """Use this frontend's literal masks to guard aliases inside full context."""
    _, mapping = numeric._protect(normalized)
    spans = []
    for literal in mapping.values():
        literal = numeric._restore(literal, mapping)
        if not literal:
            continue
        start = 0
        while True:
            start = normalized.find(literal, start)
            if start < 0:
                break
            spans.append((start, start + len(literal)))
            start += len(literal)
    return tuple(sorted(set(spans)))


def _touches_identifier(text, start, end):
    separators = ('_', '/', '\\', '@', '`')
    return (text[max(0, start - 1):start] in separators
            or text[end:end + 1] in separators
            or bool(re.match(r'\.[A-Za-z0-9_]', text[end:end + 2])))


def source_hashes():
    """Return package-relative source identities without hashing this mapping."""
    names = (FRONTEND_FILE, *NORMALIZATION_MODULE_SHAS)
    return {name: sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
            for name in names}


_SOURCE_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_ENTRY_WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*\Z")
_BUSINESS_ATOM = (r"(?:[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*(?:\([A-Za-z0-9]+\)[A-Za-z0-9]*)*"
                  r"|\([A-Za-z0-9]+\)[A-Za-z0-9]*)")
_BUSINESS_WORD = re.compile(_BUSINESS_ATOM + r'|&')
_BUSINESS_PHRASE = re.compile(_BUSINESS_ATOM + r'(?:[ \t]+(?:&[ \t]+)?' + _BUSINESS_ATOM + r')*\Z')
_PROTECTED = re.compile(
    r"`[^`\n]*`"
    r"|\[[^\]\n]*\]\([^\)\n]*\)"
    r"|(?<![A-Za-z0-9+.-])(?:[A-Za-z][A-Za-z0-9+.-]*://|www\.)[^\s<>，。；！？]+"
    r"|(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
    r"|(?:[A-Za-z]:[\\/]|\.{1,2}/|/)[A-Za-z0-9_.~+\-/\\]+"
    r"|(?<![A-Za-z0-9_])[A-Za-z0-9_][A-Za-z0-9_.\-]*\.[A-Za-z0-9_][A-Za-z0-9_.\-]*"
)
_FUNCTION_TAGS = frozenset(('PRP', 'PRP$', 'WP', 'WP$', 'WDT', 'WRB', 'DT',
                            'PDT', 'IN', 'CC', 'TO', 'EX', 'MD'))
_GRAMMAR_WORDS = frozenset(('A', 'AN', 'AND', 'AS', 'AT', 'BE', 'BUT', 'BY', 'FOR',
    'HE', 'HER', 'HIS', 'HOW', 'I', 'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'NOR',
    'NOT', 'OF', 'ON', 'OR', 'OUR', 'SO', 'THAN', 'THAT', 'THE', 'THEIR', 'THEM',
    'THEN', 'THESE', 'THEY', 'THIS', 'THOSE', 'TO', 'US', 'WE', 'WHAT', 'WHEN',
    'WHERE', 'WHICH', 'WHO', 'WHOM', 'WHOSE', 'WHY', 'WILL', 'WITH', 'YOU', 'YOUR'))
_LEXICON_NO_STRESS = str.maketrans('', '', 'ˈˌ \t\r\n')


def _merged_spans(spans):
    result = []
    for start, end in sorted(spans):
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError('Invalid protected source offset')
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _complete_word(source, start, end):
    def ascii_word(char):
        return bool(char) and char.isascii() and (char.isalnum() or char == '_')
    before, after = source[start - 1:start] if start else '', source[end:end + 1]
    if ascii_word(before) or ascii_word(after) or before in ('_', '.', '/', '\\', '@', '-') or after in ('_', '/', '\\', '@', '-'):
        return False
    if before in ("'", '’') and start > 1 and ascii_word(source[start - 2:start - 1]):
        return False
    if after in ("'", '’') and ascii_word(source[end + 1:end + 2]):
        return False
    # A sentence period is allowed; a filename/domain suffix is not.
    if after == '.' and ascii_word(source[end + 1:end + 2]):
        return False
    return True


class EnglishLexiconExtension:
    """Load compact terms['english_lexicon'] once, then use whole-word hashes.

    Integration: bind_native(g2p.lexicon, default_context) after G2P construction; wrap every
    G2P.__call__ in scope(raw_text, existing_source_span); call apply_overrides
    after existing CUDA/SageAIOS/case-canonical handling in tokenize; install
    wrap_fallback(original_espeak) as G2P.fallback. Every missing/unsafe offset
    delegates to the original native/fallback path.
    """

    def __init__(self, resource, vocab, *, reserved_casefold=()):
        if resource is None:
            resource = {}
        if not isinstance(resource, dict) or resource.get('schema_version', 1) != 1:
            raise ValueError('Unsupported English lexicon schema')
        self._vocab = frozenset(vocab)
        native_delegates = resource.get('native_delegates', [])
        if (not isinstance(native_delegates, list) or len(native_delegates) > 1024
                or any(not isinstance(word, str) or len(word) > 256
                       or not _BUSINESS_PHRASE.fullmatch(word) for word in native_delegates)):
            raise ValueError('Invalid native English delegate inventory')
        self._reserved = frozenset(word.casefold() for word in (*reserved_casefold, *native_delegates))
        self._context_words = frozenset()
        self._native_reading = None
        self._scope = ContextVar('xiaoyou_english_lexicon_' + str(id(self)), default=None)
        self._layers = []
        self._phrases = {}
        self._phrase_max_words = 0
        phrase_keys = 0
        indexed = 0
        reviewed = resource.get('reviewed_pronunciations', {})
        business = resource.get('business_entries', [])
        entries = resource.get('entries', {})
        if not isinstance(reviewed, dict) or not isinstance(business, list) or not isinstance(entries, dict):
            raise ValueError('Invalid English override collections')
        groups = (
            (reviewed.items(), 'reviewed_override'),
            (((row.get('text'), row) for row in business if isinstance(row, dict)), 'business_override'),
            (entries.items(), 'override'),
        )
        if any(not isinstance(row, dict) for row in business):
            raise ValueError('Invalid business English entry')
        deferred = 0
        for priority, (rows, action) in enumerate(groups):
            exact, folded = {}, {}
            for surface, row in rows:
                if not isinstance(row, dict):
                    raise ValueError('Invalid English override row')
                if row.get('action') == 'defer_cmu' and row.get('mode') == 'cmu':
                    deferred += 1
                    continue
                if row.get('action') != action or row.get('mode') not in ('exact', 'fold'):
                    raise ValueError('Invalid English override mode/action')
                if not isinstance(surface, str) or len(surface) > 256:
                    raise ValueError('English override must be a bounded complete Latin spelling')
                phones = self._phones(row.get('phonemes'))
                if not _ENTRY_WORD.fullmatch(surface):
                    # Only the explicitly reviewed/business inventory can
                    # introduce multiword or digit-bearing brand identities.
                    # The broad name/CMU indexes remain whole Latin words.
                    if (action == 'override' or not _BUSINESS_PHRASE.fullmatch(surface)
                            or not re.search('[A-Za-z]', surface)):
                        raise ValueError('Invalid complete business phrase')
                    words = tuple(surface.split())
                    if len(words) > 32:
                        raise ValueError('Business phrase exceeds the bounded lookup depth')
                    node = self._phrases
                    for word in words:
                        node = node.setdefault(word.casefold(), {})
                    key = (priority, row['mode'], ' '.join(words) if row['mode'] == 'exact' else ' '.join(words).casefold())
                    terminal = node.setdefault(None, {})
                    if key in terminal and terminal[key] != phones:
                        raise ValueError('Conflicting business phrases in one priority layer')
                    if key not in terminal:
                        phrase_keys += 1
                    terminal[key] = phones
                    self._phrase_max_words = max(self._phrase_max_words, len(words))
                    continue
                index, key = (exact, surface) if row['mode'] == 'exact' else (folded, surface.casefold())
                if key in index and index[key] != phones:
                    raise ValueError('Conflicting English entries in one priority layer')
                index[key] = phones
            self._layers.append((exact, folded))
            indexed += len(exact) + len(folded)
        # Reuse the bounded phrase trie for native-only complete names. None
        # marks an exemption, including its component words and CMU fallback.
        for surface in native_delegates:
            words = tuple(surface.split())
            if len(words) < 2:
                continue
            node = self._phrases
            for word in words:
                node = node.setdefault(word.casefold(), {})
            node.setdefault(None, {})[(-1, 'fold', ' '.join(words).casefold())] = None
            self._phrase_max_words = max(self._phrase_max_words, len(words))
        cmu = resource.get('cmu_fallback', {})
        if not isinstance(cmu, dict):
            raise ValueError('Invalid CMU fallback mapping')
        self._cmu = {}
        for surface, row in cmu.items():
            if not isinstance(surface, str):
                raise ValueError('Invalid CMU fallback key')
            phones = self._phones(row.get('phonemes') if isinstance(row, dict) else row)
            # Numeric/symbol CMU keys remain in source audit, not runtime scope.
            if len(surface) <= 256 and _ENTRY_WORD.fullmatch(surface):
                key = surface.upper()
                if key in self._cmu and self._cmu[key] != phones:
                    raise ValueError('Conflicting CMU fallback base spellings')
                self._cmu[key] = phones
        self.metadata = {'schema_version': 1, 'indexed_override_keys': indexed + phrase_keys,
            'indexed_phrase_keys': phrase_keys, 'phrase_max_words': self._phrase_max_words,
            'deferred_cmu_audit_rows': deferred, 'indexed_cmu_keys': len(self._cmu),
            'lookup': 'complete source word hashes and bounded business phrase trie; no per-entry request regex',
            'cmu_priority': 'after all native lexical/POS resolution, before original eSpeak',
            'general_name_priority': 'native first; whole single-token lexical misses only',
            'native_delegate_spellings': len(native_delegates),
            'global_dictionary_modified': False}

    def _phones(self, phones):
        if not isinstance(phones, str) or not phones or set(phones) - self._vocab:
            raise ValueError('English lexicon phonemes are outside the model vocabulary')
        return phones

    def bind_native(self, lexicon, default_context):
        # Preserve all native context/POS dictionaries, even explicit broad
        # source entries such as AA/BA/CIS/ISO that would otherwise shadow them.
        gold = getattr(lexicon, 'golds', None)
        if not isinstance(gold, dict):
            raise TypeError('Expected native Misaki English Lexicon.golds')
        self._context_words = frozenset(word.casefold() for word, value in gold.items()
                                         if isinstance(value, dict))
        # The probe checks availability or equivalence only. The actual G2P
        # pass supplies the sentence context when producing native readings.
        self._native_reading = (lambda token: lexicon(token, default_context)[0]) if callable(lexicon) else None

    @contextmanager
    def scope(self, text, source_span=None):
        if not isinstance(text, str):
            raise TypeError('English source text must be a string')
        source, start, end, inherited = text, 0, len(text), ()
        if source_span is not None:
            source, start, end, inherited = source_span
            if (not isinstance(source, str) or type(start) is not int or type(end) is not int
                    or not 0 <= start <= end <= len(source)):
                raise ValueError('Invalid complete English source span')
        spans = [(m.start(), m.end()) for m in _PROTECTED.finditer(source)]
        spans.extend((a, b) for a, b, phones in self._phrase_matches(source)
                     if phones is None and _complete_word(source, a, b))
        spans.extend(inherited)
        spans = _merged_spans(spans)
        frame = {'source': source, 'start': start, 'end': end, 'input': text,
                 'spans': spans, 'span_starts': [a for a, _ in spans],
                 'literal': source[start:end] != text, 'cmu_safe': {}}
        token = self._scope.set(frame)
        try:
            yield self
        finally:
            self._scope.reset(token)

    @staticmethod
    def _safe_span(frame, start, end):
        source = frame['source']
        if not _complete_word(source, start, end):
            return False
        index = bisect_right(frame['span_starts'], end - 1) - 1
        return index < 0 or frame['spans'][index][1] <= start

    @staticmethod
    def _token_offsets(text, tokens):
        starts, ends, cursor = {}, {}, 0
        for index, token in enumerate(tokens):
            value, whitespace = token.text, token.whitespace
            if not isinstance(value, str) or not value or not isinstance(whitespace, str):
                return None
            if not text.startswith(value, cursor):
                return None
            starts[cursor] = index
            cursor += len(value)
            ends[cursor] = index
            if not text.startswith(whitespace, cursor):
                return None
            cursor += len(whitespace)
        return (starts, ends) if cursor == len(text) else None

    @staticmethod
    def _grammar_word(surface, tokens):
        return surface.upper() in _GRAMMAR_WORDS and any(token.tag in _FUNCTION_TAGS for token in tokens)

    def _lookup(self, surface, *, include_general=True):
        folded = surface.casefold()
        if folded in self._reserved or folded in self._context_words:
            return None
        layers = self._layers if include_general else self._layers[:2]
        for exact, folds in layers:
            if surface in exact:
                return exact[surface]
            if folded in folds:
                return folds[folded]
        return None

    def _matches_native_reading(self, tokens, phones):
        if self._native_reading is None:
            return False
        native = []
        for token in tokens:
            value = self._native_reading(token)
            if value is None:
                return False
            native.append(value)
        return ''.join(native).translate(_LEXICON_NO_STRESS) == phones.translate(_LEXICON_NO_STRESS)

    def _phrase_matches(self, text):
        """Find longest complete business spans without retokenizing English.

        The trie is compiled once from the small reviewed/business layer. A
        request visits at most 32 components per source word, regardless of the
        size of the broad English or CMU dictionaries. Source punctuation and
        newlines are boundaries, not interchangeable with phrase spaces.
        """
        if not self._phrases:
            return
        words = list(_BUSINESS_WORD.finditer(text))
        cursor = 0
        while cursor < len(words):
            node, best = self._phrases, None
            for end in range(cursor, min(len(words), cursor + self._phrase_max_words)):
                if end > cursor and not re.fullmatch(r'[ \t]+', text[words[end - 1].end():words[end].start()]):
                    break
                node = node.get(words[end].group().casefold())
                if node is None:
                    break
                terminal = node.get(None)
                if terminal:
                    surface = ' '.join(word.group() for word in words[cursor:end + 1])
                    candidates = [(priority, mode != 'exact', phones) for (priority, mode, key), phones in terminal.items()
                                  if key == (surface if mode == 'exact' else surface.casefold())]
                    if candidates:
                        best = (end, min(candidates)[2])
            if best is None:
                cursor += 1
                continue
            end, phones = best
            yield words[cursor].start(), words[end].end(), phones
            cursor = end + 1

    def apply_overrides(self, text, tokens, *, automatic_aliases=()):
        frame = self._scope.get()
        if frame is None or frame['literal']:
            return tokens
        raw = frame['input']
        stripped = raw.lstrip()
        # Misaki preprocess strips leading whitespace and user markup. No
        # guessed .find offsets across a changed source are ever authorized.
        if text != stripped:
            return tokens
        offsets = self._token_offsets(text, tokens)
        if offsets is None:
            return tokens
        starts, ends = offsets
        base = frame['start'] + len(raw) - len(stripped)
        # Record CMU safety from the untouched full source before applying any
        # overrides. Duplicate protected occurrences still close fallback.
        for match in _SOURCE_WORD.finditer(text):
            safe = self._safe_span(frame, base + match.start(), base + match.end())
            surface = match.group()
            frame['cmu_safe'][surface] = frame['cmu_safe'].get(surface, True) and safe
        for start, end, phones in self._phrase_matches(text):
            if phones is None:
                continue
            first, last = starts.get(start), ends.get(end)
            if (not self._safe_span(frame, base + start, base + end)
                    or first is None or last is None or first > last):
                continue
            group = tokens[first:last + 1]
            if any(token.phonemes is not None and first + index not in automatic_aliases
                   for index, token in enumerate(group)):
                continue
            if self._matches_native_reading(group, phones):
                continue  # Preserve native phrase rhythm when the sounds match.
            for index, token in enumerate(group):
                token.phonemes = phones if index == 0 else ''
                token._.rating = 5
                if index:
                    # Misaki fold_left merges the exact source tokens after
                    # native POS tagging, keeping only the final whitespace.
                    # This avoids extra silent spaces or duplicated words.
                    token._.is_head = False
        for match in _SOURCE_WORD.finditer(text):
            surface = match.group()
            safe = self._safe_span(frame, base + match.start(), base + match.end())
            # Fallback tokens lack reliable native source offsets after Misaki
            # retokenizes. Authorize only full source spellings whose EVERY
            # occurrence here is safe; ambiguous duplicates conservatively use
            # the original fallback, avoiding a protected-substring rewrite.
            first, last = starts.get(match.start()), ends.get(match.end())
            if not safe or first is None or last is None or first > last:
                continue
            group = tokens[first:last + 1]
            if any(token.phonemes is not None for token in group) or self._grammar_word(surface, group):
                continue
            phones = self._lookup(surface, include_general=False)
            if phones is None:
                # A spelling from another model's broad inventory is evidence
                # of a name, not evidence that Kokoro pronounces it wrongly.
                # Use it only for a proved complete native lexical miss.
                if len(group) != 1 or self._native_reading is None:
                    continue
                if self._native_reading(group[0]) is not None:
                    continue
                phones = self._lookup(surface)
            if phones is None:
                continue
            if self._matches_native_reading(group, phones):
                continue
            for index, token in enumerate(group):
                token.phonemes = phones if index == 0 else ''
                token._.rating = 5
        return tokens

    def wrap_fallback(self, original):
        if not callable(original):
            raise TypeError('Expected the original English fallback callable')

        def fallback(token):
            frame = self._scope.get()
            surface = token.text
            if (frame is not None and frame['cmu_safe'].get(surface, False)
                    and surface.casefold() not in self._reserved
                    and surface.casefold() not in self._context_words
                    and not self._grammar_word(surface, [token])):
                phones = self._cmu.get(surface.upper())
                if phones is not None:
                    return phones, 3
            return original(token)

        return fallback

# Contextual Chinese pronunciation on the same Misaki frontend.

_SEGMENTS = re.compile(r'[\u4E00-\u9FFF]+|[^\u4E00-\u9FFF]+')

RULE_ID = 'xiaoyou-v6-relisten-explicit-audio-object-v1'

_RELISTEN = re.compile(
    r'重听'
    r'(?:(?:一下|一遍|两遍|三遍|几遍|一次|两次|几次))?'
    r'(?:(?:(?:这|那|某|哪)一?)|一|两|几)?'
    r'(?:个|段|条)?'
    r'(?P<audio_object>片段|音频|录音)'
)

FOUR_WORD_RULE_ID = 'xiaoyou-v6-four-lexical-contexts-20261110-v1'

_WORD_READINGS = {'重来': 'chong2', '重试': 'chong2', '数清': 'shu3', '长椅': 'chang2'}

def _native_segments(zh, text):
    """Yield Misaki's unmodified Han words and their exact input offsets."""
    for match in _SEGMENTS.finditer(text):
        segment = match.group()
        words = []
        if '\u4E00' <= segment[0] <= '\u9FFF':
            tokens = zh.jieba.lcut(segment, cut_all=False)
            if ''.join(tokens) != segment:
                raise RuntimeError('Native jieba words do not reproduce the full Chinese segment')
            cursor = match.start()
            for word in tokens:
                end = cursor + len(word)
                words.append((word, cursor, end))
                cursor = end
        yield segment, words


def _render_native_zh(zh, prepared, fixes):
    """Replay the native word renderer, replacing only authorized syllables."""
    rendered = []
    applied = set()
    for segment, words in _native_segments(zh, prepared):
        if not words:
            rendered.append(segment)
            continue
        outputs = []
        for word, start, end in words:
            positions = [position for position in fixes if start <= position < end]
            if not positions:
                outputs.append(zh.ZHG2P.word2ipa(word))
                continue
            pinyins = zh.lazy_pinyin(word, style=zh.Style.TONE3, neutral_tone_with_five=True)
            if len(pinyins) != len(word):
                raise RuntimeError('Cannot establish a one-syllable-per-character word offset')
            for position in positions:
                pinyins[position - start] = fixes[position]
                applied.add(position)
            outputs.append(''.join(zh.ZHG2P.py2ipa(py) for py in pinyins))
        rendered.append(' '.join(outputs))
    if applied != set(fixes):
        raise RuntimeError('Not all word-scoped overrides were applied')
    return ''.join(rendered).replace(chr(815), '')


def _find_existing_relisten_interventions(text: str, tokens: list[dict]) -> list[dict]:
    """Return exact code-point offsets in the supplied complete Chinese span.

    Offsets refer to the original two-character word, not a fuzzy match or a
    phone-string search.  Audio semantics across an English split, arbitrary
    intervening words and other polyphones are outside this rule's scope.
    """
    matches = list(_RELISTEN.finditer(text))
    if not matches:
        return []
    eligible_words = {(token['start'], token['end']) for token in tokens
                      if token['surface'] == '重听'}
    return [
        {
            'rule_id': RULE_ID,
            'start': m.start(), 'end': m.start() + 2,
            'surface': '重听',
            'context_start': m.start(), 'context_end': m.end(),
            'context_surface': m.group(),
            'audio_object': m.group('audio_object'),
            'pinyin_override_at_start': 'chong2',
        }
        for m in matches
        if (m.start(), m.start() + 2) in eligible_words
    ]

def _native_tokens(text):
    """Full unchanged native words with raw-text offsets and Chinese-span IDs."""
    from misaki import zh
    return [{'surface': word, 'start': start, 'end': end, 'segment': index}
            for index, (_, words) in enumerate(_native_segments(zh, text))
            for word, start, end in words]

def find_context_interventions(text: str) -> list[dict]:
    if not re.search(r'重听|重来|重试|数清|长椅', text):
        return []
    tokens = _native_tokens(text)
    interventions = _find_existing_relisten_interventions(text, tokens)

    def add(surface, position, consumed, pattern):
        if text[position:position + len(surface)] != surface:
            raise RuntimeError('The lexical target does not match its exact character offsets')
        context_start, context_end = consumed[0]['start'], consumed[-1]['end']
        if any(token['segment'] != consumed[0]['segment'] for token in consumed):
            raise RuntimeError('A lexical intervention must not cross Chinese spans')
        interventions.append({
            'rule_id': FOUR_WORD_RULE_ID, 'start': position, 'end': position + len(surface),
            'surface': surface, 'context_start': context_start, 'context_end': context_end,
            'context_surface': text[context_start:context_end],
            'native_token_pattern': pattern,
            'native_tokens': [dict(token) for token in consumed],
            'pinyin_override_at_start': _WORD_READINGS[surface],
        })

    for index, token in enumerate(tokens):
        word = token['surface']
        if word in _WORD_READINGS:
            add(word, token['start'], [token], 'exact_complete_native_word')
        if index + 1 == len(tokens):
            continue
        following = tokens[index + 1]
        if token['end'] != following['start'] or token['segment'] != following['segment']:
            continue
        pair = (word, following['surface'])
        if pair == ('重', '来'):
            add('重来', token['start'], [token, following], 'two_complete_native_words_重_来')
        elif pair == ('先数', '清'):
            # Keep先数 and清 as the native tokenizer produced them. Only数's
            # syllable inside the complete先数 token changes;先 is untouched.
            add('数清', token['start'] + 1, [token, following], 'limited_action_fusion_先数_清')
    starts = [item['start'] for item in interventions]
    if len(set(starts)) != len(starts):
        raise RuntimeError('Multiple context rules select the same syllable offset')
    return sorted(interventions, key=lambda item: item['start'])

class ContextualZHG2P:
    """Upstream-call compatible instance wrapper; non-targets delegate exactly."""

    def __init__(self, upstream):
        if not callable(upstream):
            raise TypeError('A native Misaki ZHG2P callable is required')
        if getattr(upstream, 'version', None) is not None or getattr(upstream, 'frontend', None) is not None:
            raise ValueError('This candidate supports only the verified Misaki legacy version=None path')
        from misaki import zh
        self.upstream = upstream
        self._native = zh
        self.metadata = {
            'project_version': 'xiaoyou_v6',
            'kind': 'scoped_context_pronunciation_candidate',
            'rule_id': RULE_ID,
            'scope': 'Existing重听 explicit audio-object rule plus only4 lexical targets at native exact words/重+来/先数+清; no global character rules',
            'four_word_rule_id': FOUR_WORD_RULE_ID,
            'candidate_module_path': str(Path(__file__).resolve()),
            'candidate_module_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'four_word_readings': dict(_WORD_READINGS),
            'limited_cross_token_patterns': [['重', '来'], ['先数', '清']],
            'upstream_version': None,
            'normal_non_target_direct_delegate': True,
            'native_replay_must_match_before_intervention': True,
            'raw_text_substitution': False,
            'pypinyin_or_jieba_global_dictionary_modified': False,
            'legacy_V4_V5_rules_used': False,
            'other_polyphone_coverage_claimed': False,
        }

    def __getattr__(self, name):
        return getattr(self.upstream, name)

    def _render(self, prepared: str, fixes: dict[int, str]) -> str:
        authorized = {item['start']: item['pinyin_override_at_start']
                      for item in find_context_interventions(prepared)} if fixes else {}
        if any(position not in authorized or authorized[position] != pinyin
               or prepared[position] not in ('重', '数', '长') for position, pinyin in fixes.items()):
            raise RuntimeError('An override is not authorized at that exact native context offset')
        return _render_native_zh(self._native, prepared, fixes)

    def __call__(self, text, en_callable=None):
        # Normal input takes the untouched upstream implementation directly.
        if not find_context_interventions(text):
            return self.upstream(text, en_callable=en_callable)
        original = self.upstream(text, en_callable=en_callable)
        zh = self._native
        prepared = zh.ZHG2P.map_punctuation(zh.cn2an.transform(text, 'an2cn'))
        interventions = find_context_interventions(prepared)
        if not interventions:
            return original
        if self._render(prepared, {}) != original[0]:
            raise RuntimeError('Context candidate native replay differs from the complete upstream phonemes')
        fixes = {x['start']: x['pinyin_override_at_start'] for x in interventions}
        return self._render(prepared, fixes), original[1]

_GROWTH_OBJECT = r'(?:一些|一点儿?|点)?(?:见识|知识|本事|个儿|个子|个头)'
_GROWTH_SUFFIX = re.compile(r'(?:长?' + _GROWTH_OBJECT + r'|得|高|胖|壮|大|出|成)')


def _tower_is_object_of_hiding(text, start):
    left = text[max(0, start - 18):start]
    right = text[start + len('藏塔'):]
    if (left.endswith(('想', '打算', '试图', '准备', '决定'))
            and re.match(r'(?=$|[，。！？；,.!?;]|之后(?:再|就|才)|了[，。！？]|起来)', right)):
        return True
    return (re.search(r'(?:把|将)[\u4e00-\u9fff]{1,16}$', left) is not None
            and re.match(r'(?:旁边|前面|后面|下面|里面|底下)', right) is not None)

_SYNTAX_RULES = (
    (re.compile(r'(?:我们|你们|他们|她们|咱们|大家|我|你|您|他|她)(?P<target>得)'
                r'(?:先|把|去|赶紧|尽快|注意|确保)'),
     '得', 'dei3', 'modal_subject_得_action'),
    (re.compile(r'(?:解释|核对|整理|处理|检查|说|讲|读|做)(?P<target>得)'
                r'(?:非常|十分|特别|比较|这么|那么|很|更|太)?'
                r'(?:清楚|仔细|自然|流畅|准确)'),
     '得', 'de5', 'verb_得_optional_degree_quality'),
    (re.compile(r'(?:幸福快乐|有条不紊|实实在在|长时间|一点点|有目的|'
                r'安静|认真|仔细|耐心|温柔|轻轻|慢慢|快乐|高兴|开心|'
                r'静静|大声|小声|紧张|飞快|不停|更快|快速)(?P<target>地)'
                r'(?:长大成人|长得更高|长大|成长|说道|说话|说着|笑着|找到|'
                r'解释|介绍|处理|核对|检查|完成|学习|坐着|等待|准备|帮助|'
                r'放下|跑回家|收集|改变|听|说|讲|读|做|把|走|笑|喊|跑)'),
     '地', 'de5', 'manner_地_action'),
    (re.compile(r'(?:[零一二两三四五六七八九十百千万点]+)'
                r'(?:毫米|厘米|分米|千米|公里|米)(?P<target>长)'
                r'(?![高大出成])'),
     '长', 'chang2', 'length_after_measurement'),
)

_PHRASE_END = r'(?=[ \t]*(?:$|[\n，。！？；、：,.!?;:]))'
_LINE_COUNT = r'第?[零一二两三四五六七八九十百千万几]+'
_TIED_OBJECT = r'(?:安全带|鞋带|领带|救生衣|绳子|固定带)'
_DEBT_CONTEXT = r'(?:借款|贷款|债务|欠款|这笔钱)[^，。！？；,.!?;]{0,10}?'
_CAN_REPAY = r'上(?=$|[\s，。！？；、：,.!?;:吗呢吧了就才])'
_DIMENSION = r'[零〇一二两三四五六七八九十百千万点0-9]+(?:\.[0-9]+)?\s*(?:毫米|厘米|分米|千米|公里|米)'

# A contextual target can be inside a native token such as 请系 or 重装系统.
# Match the surrounding words before splitting, then verify the syllable offset.
_CONTEXT_READINGS = (
    (re.compile(r'(?P<target>还)(?:钱|款|车|书|货|衣服)'
                r'(?=[ \t]*(?:$|[\n，。！？；、：,.!?;:]|完|之后|以前|以后|之前|后|时|的时候|'
                r'期[限间]|日期|手续|周期|[了吧吗呢]))'),
     'huan2', 'return_object'),
    (re.compile(r'(?P<target>还)书不离手'), 'hai2', 'also_before_nominal_clause'),
    (re.compile(r'(?:把|将)[\u4e00-\u9fff]{1,16}?(?P<target>还)(?:给)?[我你您他她]'
                + _PHRASE_END), 'huan2', 'return_to_person'),
    (re.compile(r'(?:欠[^，。！？；,.!?;]{0,12}钱|(?:这本|那本|借来的|借的)书)'
                r'[^。！？；.!?;]{0,18}?(?:没|未|要|会|该|须|愿)(?P<target>还)' + _PHRASE_END),
     'huan2', 'return_referenced_item'),
    (re.compile(_DEBT_CONTEXT + r'(?P<target>还)得' + _CAN_REPAY), 'huan2', 'can_repay_debt'),
    (re.compile(_DEBT_CONTEXT + r'还(?P<target>得)' + _CAN_REPAY), 'de5', 'repay_potential_particle'),
    (re.compile(r'(?:(?<=集体)|(?<![尊慎看器倚贵加侧注负着举体郑]))(?P<target>重)(?:'
                r'装\s*(?:系统|操作系统|驱动|软件|应用|客户端|(?:CUDA|Python)(?![A-Za-z0-9_]))|'
                r'做(?:一次|一遍|测试|实验|审核|整道题)|跑(?:一次|一遍|流程|任务|测试|脚本|实验)|'
                r'听(?:一次|一遍|一下)?(?:(?:刚才|刚刚|之前|昨天)的?)?(?:这|那)?(?:一?段|一?遍)?'
                r'(?:音频|录音|片段|原声))'),
     'chong2', 'repeat_action_with_object'),
    (re.compile(r'(?:音节|元音|重音|字音)(?:都)?(?:要|应|应该|应当|需要|必须|可以)?'
                r'(?P<target>重)读(?![一两三几这那])'),
     'zhong4', 'stress_in_pronunciation'),
    (re.compile(_LINE_COUNT + r'(?P<target>行)\s*(?:[A-Za-z][A-Za-z0-9_+.-]*[ \t]*)?'
                r'(?:代码|日志|文本|小字|指令|注释|命令|字)'),
     'hang2', 'quantity_of_written_lines'),
    (re.compile(r'(?:(?:代码|日志|文本|表格|注释)[^。！？；.!?;]{0,22}?|'
                r'往[上下]看|向[上下]翻|查看|跳到|删除|填写|填入|站成|排成)' + _LINE_COUNT + r'(?P<target>行)'
                r'(?=[ \t]*(?:$|[\n，。！？；、：,.!?;:上下里起]))'),
     'hang2', 'line_in_written_context'),
    (re.compile(_LINE_COUNT + r'(?P<target>行)是(?:日志|输出|代码|标题|注释)'),
     'hang2', 'line_content'),
    (re.compile(r'(?:一路|一起|结伴|(?:与|和|跟)\s*(?:家人|同事|朋友|导游|[我你他她](?:们)?|'
                r'[A-Za-z][A-Za-z0-9_-]*\s*(?!\s*同行\s*(?:交流|分享|合作|讨论|学习|座谈))))'
                r'同(?P<target>行)(?!业|列|政)'),
     'xing2', 'travel_together'),
    (re.compile(r'(?:^|(?<=[，。！？；,.!?;])\s*|(?:请|先|快|正在|开始|记得|忘了)|(?=系[好紧牢上着]))'
                r'(?P<target>系)(?:好|紧|牢|上|着)?\s*(?:[A-Za-z][A-Za-z0-9_-]*\s*的\s*)?'
                + _TIED_OBJECT + r'(?!损坏|失效|过期|故障|断裂|松动|系统)'),
     'ji4', 'fasten_object'),
    (re.compile(_TIED_OBJECT + r'(?:上|没|未|已|已经)?(?P<target>系)(?:紧|牢|好|在|着)(?!统)'),
     'ji4', 'fastened_object'),
    (re.compile(r'(?:^|(?<=[，。！？；,.!?;])\s*|[我你您他她](?:们)?|咱们|大家|孩子|'
                r'还|也|就|才|总|必须|一定|之前|以后)(?P<target>得)'
                r'(?:先|把|去|赶紧|尽快|注意|确保|检查|吃饭|上[班学课岗线]|按照|按时|记得|重新|'
                r'当心|确认|休息|等|做|出门|提交|安排|考虑|核对|准备(?!金)|想|跟|小心)'),
     'dei3', 'modal_before_action'),
    (re.compile(r'[我你您他她](?:们)?说(?P<target>得)等(?=到|天亮|天黑|上一会|几分钟)'),
     'dei3', 'reported_need_to_wait'),
    (re.compile(r'(?:跑|走|跳|唱|写|读|说|讲|做|拿|解释|处理|检查|执行|急)(?P<target>得)'
                r'(?:非常|十分|特别|比较|这么|那么|越来越|很|更|太|真)?'
                r'(?:清楚|仔细|自然|流畅|准确|过去|有道理|流利|直跺脚|直发抖|好|快|慢|稳|起来)'),
     'de5', 'degree_or_result_after_verb'),
    (re.compile(r'(?:快快乐乐|痛痛快快|默默|缓慢|稳定|稳稳|迅速|耐心|安静|轻轻|踏实|彻底|礼貌|小心)(?P<target>地)'
                r'(?:工作|执行|等|承担|前进|长大|说|学习|处理|完成|震撼|问|运行|往前走)'),
     'de5', 'manner_before_action'),
    (re.compile(r'(?P<target>长)(?=长?' + _GROWTH_OBJECT + r')'), 'zhang3', 'gain_experience'),
    (re.compile(r'以[^，。！？；,.!?;]{1,16}?见(?P<target>长)' + _PHRASE_END), 'chang2', 'specialty'),
    (re.compile(r'(?:(?:这|那|[一二两三四五六七八九十几]+)[个条段部份篇])?'
                r'(?P<target>长)(?:文件名|链接|网址|文本|段落|音频|句子|视频|路径)'),
     'chang2', 'long_content'),
    (re.compile(r'(?:板子?|布|纸|线|电缆|绳子?|路|桥|隧道|钢管|零件|车身|机身|(?<!身)体|'
                r'[宽高]\s*' + _DIMENSION + r'\s*[，,]\s*)'
                r'\s*(?:约)?(?P<target>长)\s*(?:约|达|是)?\s*' + _DIMENSION),
     'chang2', 'measured_length'),
    (re.compile(r'乡(?P<target>长)'), 'zhang3', 'township_head'),
    (re.compile(r'(?:[先再请]连|连接|接上|插上)(?P<target>长)线(?![路性包程])'),
     'chang2', 'long_cable'),
    (re.compile(r'(?:请|大家|门口|窗口|已经|开始|正在|经常)(?:前|外|先|都)?排(?P<target>长)龙'
                r'(?=$|[了啦呢吗吧，。！？、；,.!?;])'),
     'chang2', 'long_queue'),
    (re.compile(r'(?P<target>藏)医(?!生)'), 'zang4', 'tibetan_medicine'),
    (re.compile(r'[川青滇新](?P<target>藏)(?:线|公路|铁路)'), 'zang4', 'tibetan_route'),
    (re.compile(r'(?:入|进|赴|援)(?P<target>藏)'
                r'(?=[ \t]*(?:$|[\n，。！？；、：,.!?;:]|[了前后时]|旅游|之前|之后|路线))'),
     'zang4', 'tibet_travel'),
    (re.compile(r'(?<![无多几少复质整负分实余双奇偶])(?P<target>数)'
                r'(?:钱|票|星星|羊|人数|人头)'),
     'shu3', 'count_objects'),
    (re.compile(r'(?<![人位次本票岁份系函参变总指倍计字代乘无多几少质整负息步分小底约基尾])(?P<target>数)'
                r'(?:到\s*[零〇一二两三四五六七八九十百0-9]|一数(?!二)|(?:了)?[一两三几]遍|一下(?!子)|'
                r'清(?!单|除|晰|洗)|完(?!整|全|善)|错(?!误率)|来数去)'),
     'shu3', 'count_progress'),
    (re.compile(r'(?:(?<=数来)(?=数去)|(?<=数一)(?=数(?!二)))(?P<target>数)'),
     'shu3', 'repeat_counting'),
    (re.compile(r'(?P<target>数)(?=数?(?:手里的|桌上的)(?:零钱|硬币|钱))'), 'shu3', 'count_visible_money'),
    (re.compile(r'(?:自己|大家|我们|请|再|先)(?P<target>数)(?:吧|一下)' + _PHRASE_END),
     'shu3', 'counting_request'),
    (re.compile(r'(?:睡|找|摸|够|碰|瞧|看|闻|猜|买|吃|喝|赶|追)(?:得|不)(?P<target>着)'),
     'zhao2', 'result_of_action'),
    (re.compile(r'穿(?P<target>着)(?:很|比较|十分|非常|挺)?(?:朴素|得体|讲究|时髦|正式|随意|整齐)(?!的)'),
     'zhuo2', 'description_of_attire'),
    (re.compile(r'(?:棋盘|棋局|棋手|对弈|下棋)[^，。！？；,.!?;]{0,12}?(?:这|那|一|妙)(?P<target>着)'
                r'(?=$|[，。！？；,.!?;]|[真很够好妙险稳])'),
     'zhao1', 'board_game_move'),
    (re.compile(r'(?:请|先|快|赶紧)(?P<target>给)水(?:和|以及|、)(?:面包|食物|饮料|饼干|牛奶)'),
     'gei3', 'give_food_and_water'),
    (re.compile(r'(?<![请指宗佛基传])(?P<target>教)(?:会了?[我你他她]|'
                r'(?:[我你他她](?:们)?|孩子|小朋友|学生)(?=写|做|读|唱|算|说|叠|学|用|游泳|练|骑|编|打|弹|画)|'
                r'数学|英语|语文|编程|游泳|钢琴)'),
     'jiao1', 'teach_skill'),
    (re.compile(r'(?:愿意|可以|想|先)(?P<target>教)[我你他她](?:们)?' + _PHRASE_END),
     'jiao1', 'offer_to_teach'),
    (re.compile(r'(?<![抽外派借统])(?P<target>调)(?:参(?!加|会|赛|观|与|照|考|展|审)|'
                r'优(?=化|模型|参数|配置|结果|[，。！？；,.!?;]|$))'),
     'tiao2', 'tune_model'),
    (re.compile(r'(?:请勿|严禁|禁止|不许|不可|不要|别|不得|不准)外(?P<target>传)'),
     'chuan2', 'distribute_outside'),
)


_CN_VALUE = r'(?:正负|正|负)?[零〇一二三四五六七八九十百千万亿兆两幺]+'

_ARABIC_VALUE = r'[+−-]?[0-9]+(?:\.[0-9]+)?'

_EN_NUMBER = (r'(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
              r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|'
              r'forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b')

_EN_VALUE = r'(?:(?:minus|negative|plus|positive)[ \t]+)?' + _EN_NUMBER

_NUMERIC_VALUE = r'(?:' + _CN_VALUE + r'|' + _ARABIC_VALUE + r'|' + _EN_VALUE + r')'

_VERSION_VALUE = r'v[0-9]+(?:\.[0-9]+){0,3}(?![A-Za-z0-9_+\-/]|\.[A-Za-z0-9_.])'

_WEI_RESULT = (
    r'基础(?!设施|岗位|课程|员工)|核心(?!团队|客户|用户|员工|成员)|中心(?!城区|城市|医院)|'
    r'主要(?:依据|目标|参考|指标|对象)|重点(?!客户|用户|企业|项目)|'
    r'衡量标准|标准|准入限制|准绳|准(?!客户|用户|会员|入)|例(?!外)|参考|依据|前提|目标|上限|下限|起点|入口|界|单位|'
    r'主(?!要|办|体|人|席|导|管|机)|重(?!点|要)|限|荣|耻|贵|由|名|线索'
)
_WEI_FRAME = re.compile(
    r'(?<![所足难得予加何之])以(?!及|前|后|来|往|内|外|上|下|为)'
    r'(?P<object>[\u4e00-\u9fffA-Za-z0-9_ \t%-]{1,64}?(?:'
    r'\.[0-9][\u4e00-\u9fffA-Za-z0-9_ \t%.-]{0,48}?)?)'
    r'(?P<target>为)[ \t]*(?:' + _WEI_RESULT + r')'
    r'(?=[ \t]*(?:$|[，。！？；：、,.!?;:)）〕”」]|的|'
    r'(?:来|再|先|继续|重新)?(?:说明|讲解|解释|开展|制定(?!者|人|机构|部门|单位)|校验|执行|办理|进行|提供|部署|完成|实现|'
    r'分析|评估|训练|开发|构建|建设|建立|搭建|设计|衡量|理解|推进|优化|处理|计算|申请|操作)))'
)
_WEI_ASSIGNMENT = re.compile(
    r'(?:算力(?=为基础(?!设施|岗位|课程|员工))|设定|设置|设|定|修改|更改|改|调|调整|切换|切成|变更|变|更新|转换|转化|换算|'
    r'升级|配置|定义|声明|指定|命名|标记|标注|标|视|当|识别|判定|误读|译|'
    r'归结|划分|分|合并|拆分|推举|推|选举|改写|转写|登记|聘|任命|结)'
    r'(?P<target>为)(?=[ \t]*[\u4e00-\u9fffA-Za-z0-9《“(])'
)
_WEI_FIELDS = (
    r'能量|功率|电量|容量|体积|面积|电压|电流|带宽|速率|质量|温度(?:范围)?|范围|'
    r'延迟|速度|时长|耗时|数量|样本数|记录量|值|结果|版本(?:号)?|编号|代码|端口(?:号)?|'
    r'状态|模式|语言|格式|级别|编码|型号|角色|权限|时间|日期|人数|次数|评分|期限|'
    r'长度|重量|尺寸|分辨率|帧率|转速|浓度|频率|金额|成本|误差|阈值|上限|下限|'
    r'指标|比例|占用率|利用率|通过率|准确率|失败率|成功率|增长率'
)
_WEI_ENUM_VALUES = (
    r'在线|离线|就绪|空闲|完成|已完成|已通过|通过|未通过|失败|已取消|待复核|异常|进行中|严重|'
    r'正常|只读|读写|管理员|手动|自动|隐藏|可见|开启|关闭|中文|英文|立体声|宽屏|全屏|'
    r'(?:true|false|stable|latest|debug|info|warn(?:ing)?|error|gpu|cpu|json|xml|utf-8|'
    r'wav|mp3|flac|opus|pcm|auto|manual|active|inactive|on|off)(?![A-Za-z0-9_])|'
    r'[A-Za-z]{1,6}[-_][0-9]+[A-Za-z]?(?![A-Za-z0-9_])'
)
_WEI_TYPED_VALUE = r'[ \t]*(?:' + _NUMERIC_VALUE + '|' + _VERSION_VALUE + '|' + _WEI_ENUM_VALUES + r')'
_WEI_FIELD = re.compile(r'(?:' + _WEI_FIELDS + r')(?:分别)?(?P<target>为)(?=' + _WEI_TYPED_VALUE + r')', re.I)
_WEI_TYPED_VALUE_PATTERN = re.compile(_WEI_TYPED_VALUE, re.I)
_WEI_ASSIGNMENT_CUE = re.compile(
    r'(?:^|[，。！？；,.!?;])[ \t]*$|(?:请|把|将|已|应|需|务必|手动|自动|重新)'
    r'[\u4e00-\u9fffA-Za-z0-9 \t_.-]{0,24}$'
)
_WEI_NAMED_FIELD = re.compile(
    r'(?:用户名|账号名|主机名|文件名|服务地址|服务路径)(?P<target>为)(?=[ \t]*[\u4e00-\u9fffA-Za-z0-9/])'
)
_WEI_GOAL_FIELD = re.compile(
    r'(?:业务|项目|设计|运营|主要)目标(?P<target>为)(?=[ \t]*(?:减少|降低|提高|增加|缩短|改善|实现|达到))'
)
_WEI_BENEFICIARY = re.compile(
    r'[ \t]*[\u4e00-\u9fffA-Za-z0-9 \t]{0,16}?'
    r'(?:用户|客户|买家|卖家|商户|经销商|主办方|团队|部门|机构|班级|学校|公司|国家|学员|听众|读者|居民|'
    r'同事|学生|员工|游客|孩子|观众|客人|宾客|来宾|工程师|人民|获奖者|参与者|志愿者|'
    r'单位|供应商|合作方|申请人|您|你们?|我们?|他们?|她们?|大家|自己)'
    r'(?:本人|们)?[ \t]*(?:提供|服务|办理|制作|发放|准备|安排|保留|推送|发出|发送|'
    r'划拨|分配|开放|开通|定制|设计|启动|完成|生成|争取|谈判|扩容|介绍|解释|说明|演示|节省|减少|带来|'
    r'着想|高兴|担心|庆祝|感到)'
)
_WEI_PURPOSE = re.compile(
    r'[ \t]*(?:了|着|(?:提高|降低|解决|避免|防止|防灾|保障|保护|维护|实现|确保|改善|提升|满足)'
    r'[\u4e00-\u9fffA-Za-z0-9 \t]{0,24}?而)'
)
_WEI_PASSIVE = re.compile(
    r'(?P<target>为)(?!了|此|何|什么|所)(?:华为|行为|(?!为)[\u4e00-\u9fffA-Za-z0-9 \t]){1,32}?所'
    r'(?:熟知|知晓|认识|知道|了解|爱戴|接受|认可|采用|采纳|遵守|引用|使用|容忍|'
    r'禁止|制止|损坏|阻挡|困扰|误解|误会|核对|批准|打动|感动|吸引|知)'
    r'了?(?=[ \t]*(?:[的，。！？；,.!?;]|$))'
)
_WEI_ATTRIBUTIVE_LEFT = re.compile(r'(?:一种|一个|一项|一条|一套|一份)[ \t]*$')
_WEI_DEGREE = re.compile(
    r'(?:广|极|尤|较|略|最|甚|颇|深|大)(?P<target>为)'
    r'(?=流传|引用|人知|明显|相似|关键|重要|复杂|不满|不安|有趣|便利|合适|困难|'
    r'出色|丰富|突出|危险|担忧|看好|遗憾|惊讶|感动|高兴|简单|稳定|高效)'
)
_WEI_FIXED = (
    re.compile(r'(?:有|无)(?P<target>为)(?=青年|而治)'),
    re.compile(r'(?P<target>为)人(?=(?:很|十分|特别|一向)?(?:热心|正直|诚恳|坦率|处世|忠厚|随和|诚实|谦逊|善良|可靠|老实|谨慎|宽厚))'),
    re.compile(r'(?<=的)(?P<target>为)人(?=很|如何|值得|[，。！？；,.!?;]|$)'),
    re.compile(r'(?:施|肇事者|何人|谁|[他她你我])所(?P<target>为)(?=之|皆|[，。！？；,.!?;]|$)'),
    re.compile(r'(?:何人|谁)(?P<target>为)之(?=[，。！？；,.!?;]|$)'),
    re.compile(r'不(?P<target>为)所动'),
    re.compile(r'(?P<target>为)(?:后人|世人|人们|业界)(?:广为|广泛)(?=引用|使用|接受|采纳|认可|传播)'),
    re.compile(r'拜[\u4e00-\u9fff]{1,16}?(?P<target>为)师'),
    re.compile(r'(?:公推|推选|选举|提名)[\u4e00-\u9fff]{1,12}?(?P<target>为)(?=代表|负责人|组长|队长|主席)'),
    re.compile(r'(?:此人|该人|他|她)(?P<target>为)'
               r'(?=(?:(?:本次|本届|此次)(?:比赛|活动|项目)的|服务|项目|技术)?'
               r'(?:负责人|主裁判|裁判|召集人|主持人|队长|代表)(?:[，。！？；,.!?;]|$))'),
    re.compile(r'(?:原|现)(?P<target>为)(?=学校|工厂|医院|仓库|技术负责人|项目负责人|管理员)'),
    re.compile(r'(?:变废|聚沙|化险|转危|化干戈)(?P<target>为)(?=宝|塔|夷|安|玉帛)'),
    re.compile(r'化(?:失败|压力|悲痛|危机)(?P<target>为)(?=动力|机会|力量|行动|转机)'),
    re.compile(r'(?:何以|未)(?P<target>为)(?=家|晚)'),
    re.compile(r'我(?P<target>为)鱼肉'),
)


def _wei_targets(text, words, protected_spans=()):
    """Select wéi on the full normalized source before splitting Han and Latin."""
    if '为' not in text:
        return []
    targets = {}
    starts = {start for _, start, _ in words}
    wei_positions = {
        position for word, start, _ in words
        for position, character in enumerate(word, start) if character == '为'
    }

    def add(position, context, rule):
        if position not in wei_positions:
            return
        if any(start <= position < end for start, end in protected_spans):
            return
        targets[position] = {'position': position, 'character': '为', 'pinyin': 'wei2',
                             'context': context, 'rule': rule}

    for word, start, _ in words:
        if word == '华为':
            add(start + 1, word, 'complete_native_word_华为')
    for pattern, rule in (
        (_WEI_FRAME, 'based_on_result'), (_WEI_ASSIGNMENT, 'assignment'),
        (_WEI_FIELD, 'field_value'), (_WEI_NAMED_FIELD, 'field_identifier'),
        (_WEI_GOAL_FIELD, 'field_goal'),
    ):
        for match in pattern.finditer(text):
            position = match.start('target')
            if pattern is _WEI_FRAME:
                object_text = match['object']
                if not re.search(r'[\u4e00-\u9fffA-Za-z0-9]', object_text):
                    continue
                if '.' in re.sub(r'(?<=[0-9])\.(?=[0-9])', '', object_text):
                    continue
            if pattern is _WEI_ASSIGNMENT:
                action = text[match.start():position]
                if len(action) == 1 and match.start() not in starts:
                    continue
                if (action in ('设置', '配置', '调整', '变更', '更新')
                        and not _WEI_TYPED_VALUE_PATTERN.match(text, position + 1)
                        and not _WEI_ASSIGNMENT_CUE.search(text[max(0, match.start()-28):match.start()])):
                    continue
            if not (_WEI_BENEFICIARY.match(text, position + 1) or _WEI_PURPOSE.match(text, position + 1)):
                add(position, match.group(), rule)
    for pattern in (_WEI_PASSIVE, _WEI_DEGREE, *_WEI_FIXED):
        for match in pattern.finditer(text):
            if pattern is _WEI_DEGREE and match.start() not in starts:
                continue
            if (pattern is _WEI_PASSIVE and text[match.end():].lstrip().startswith('的')
                    and not _WEI_ATTRIBUTIVE_LEFT.search(text[max(0, match.start()-8):match.start()])):
                continue
            add(match.start('target'), match.group(),
                'passive' if pattern is _WEI_PASSIVE else 'fixed_context')
    return [targets[position] for position in sorted(targets)]

def _syntax_targets(text, words, *, protected_spans=None):
    """Select local grammar readings and optional full-request overrides."""
    result=[]
    starts={start for _,start,_ in words};ends={end for _,_,end in words}
    for pattern,character,pinyin,rule in _SYNTAX_RULES:
        cursor = 0
        while (match := pattern.search(text, cursor)) is not None:
            if match.start() not in starts or match.end() not in ends:
                # A shorter context can start at a native boundary inside an
                # invalid longer match, as in “祝你幸福/快乐/地/长大成人”.
                cursor = match.start() + 1 if character == '地' else match.end()
                continue
            cursor = match.end()
            tokens=[word for word,start,end in words if match.start()<=start and end<=match.end()]
            if ''.join(tokens)!=match.group():
                raise RuntimeError('Grammar native tokens do not reconstruct exact context')
            result.append({'position':match.start('target'),'character':character,
                           'pinyin':pinyin,'context':match.group(),'rule':rule,
                           'native_word_boundary_verified':True,'native_context_tokens':tokens})
    if protected_spans is not None:
        positions = {target['position'] for target in result}
        for pattern, pinyin, rule in _CONTEXT_READINGS:
            for match in pattern.finditer(text):
                position = match.start('target')
                # These lexical modifiers must not match inside 厂长 or 收藏.
                if (rule in ('long_content', 'tibetan_medicine')
                        and position not in starts and match.start() not in starts):
                    continue
                if rule == 'township_head' and (match.start() not in starts or match.end() not in ends):
                    continue
                if position in positions or _touches_identifier(text, match.start(), match.end()) or any(
                        start < match.end() and match.start() < end for start, end in protected_spans):
                    continue
                result.append({'position': position, 'character': match['target'],
                               'pinyin': pinyin, 'context': match.group(), 'rule': rule})
                positions.add(position)
    return result

class PolyphoneContextG2P:
    def __init__(self,upstream,phrase_readings=None):
        if not callable(upstream) or getattr(upstream,'version',None)is not None or getattr(upstream,'frontend',None)is not None:
            raise TypeError('Only the audited legacy Misaki Chinese G2P ABI is supported')
        from misaki import zh
        self.upstream=upstream;self._native=zh;self.last_interventions=[]
        self._context=ContextVar('polyphone_context_'+str(id(self)),default=None)
        if phrase_readings is None:phrase_readings={}
        if not isinstance(phrase_readings,dict) or len(phrase_readings)>512:
            raise ValueError('Chinese phrase readings require a bounded mapping')
        self._phrase_trie={};self._phrase_count=len(phrase_readings)
        for phrase,record in phrase_readings.items():
            if (not isinstance(phrase,str) or not re.fullmatch(r'[\u4E00-\u9FFF]{2,32}',phrase)
                    or not isinstance(record,dict) or set(record)!={'offset','pinyin'}
                    or type(record['offset'])is not int or not 0<=record['offset']<len(phrase)
                    or not isinstance(record['pinyin'],str) or not re.fullmatch(r'[a-z]+[1-5]',record['pinyin'])):
                raise ValueError('Chinese phrase reading requires a complete Han phrase and one syllable offset')
            # Validate the native pinyin-to-IPA alphabet before accepting data.
            zh.ZHG2P.py2ipa(record['pinyin'])
            node=self._phrase_trie
            for character in phrase:node=node.setdefault(character,{})
            node[None]={'phrase':phrase,**record}
        self.metadata={'kind':'isolated_polyphone_candidate_NOT_active_profile','source_path':str(Path(__file__).resolve()),'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'scope':['complete_native_word_华为_second_syllable_wei2','full_context_selected_wei2_complements_values_and_passives','native_word_bounded_subject_得_action_dei3','native_word_bounded_verb_得_degree_quality_de5','native_word_bounded_manner_地_action_de5'],'bound_full_context_scope':'verified_grammar_and_phrase_targets_with_exact_offsets','global_dictionary_modified':False,'other_non_targets_delegate':True,'quality_accepted':False}

        self.metadata.update(phrase_reading_count=self._phrase_count,
            phrase_reading_scope='one reviewed syllable in complete consecutive native Han words; bound literal spans excluded; no global dictionaries',
            phrase_lookup='constructor_compiled_trie')

    def __getattr__(self,name):return getattr(self.upstream,name)

    @contextmanager
    def full_context(self,normalized,*,protected_spans=()):
        """Bind one already normalized article to this instance/task only."""
        if not isinstance(normalized,str) or not normalized:
            raise TypeError('A nonempty complete normalized string is required')
        if not isinstance(protected_spans,(list,tuple)) or any(
                not isinstance(item,(list,tuple)) or len(item)!=2
                or any(type(value)is not int for value in item)
                or not 0<=item[0]<item[1]<=len(normalized) for item in protected_spans):
            raise ValueError('Protected Chinese source spans require exact normalized offsets')
        words=self._words(normalized)
        targets = [
            {**item, 'global_position': item['position']}
            for item in (
                *_wei_targets(normalized, words, protected_spans),
                *_syntax_targets(normalized, words, protected_spans=protected_spans),
                *self._phrase_targets(normalized, words, protected_spans),
            )
        ]
        token=self._context.set({'text':normalized,'targets':targets})
        try:yield self
        finally:self._context.reset(token)

    def _verified_prepared_position(self, span, raw_position, converted, prepared, character):
        """Prove one exact authorized character offset across native preparation."""
        if not 0<=raw_position<len(span) or span[raw_position]!=character:
            raise RuntimeError('The bound source character is not '+character)
        zh=self._native
        left=zh.cn2an.transform(span[:raw_position],'an2cn')
        right=zh.cn2an.transform(span[raw_position+1:],'an2cn')
        if left+character+right!=converted:
            raise RuntimeError('cn2an source offset cannot be proven across this split')
        marker='\ue000POLYPHONE_TARGET\ue001'
        if marker in span or marker in converted:raise RuntimeError('Offset marker collision')
        marked=zh.ZHG2P.map_punctuation(left+marker+right)
        if marked.count(marker)!=1 or marked.replace(marker,character,1)!=prepared:
            raise RuntimeError('Punctuation mapping does not preserve the exact target offset')
        position=marked.index(marker)
        if prepared[position]!=character:raise RuntimeError('Prepared offset does not point to '+character)
        return position

    def call_span(self,normalized,start,end,en_callable=None):
        """Call one exact Chinese gap from the currently bound full article."""
        bound=self._context.get()
        if bound is None:raise RuntimeError('call_span requires an active full_context scope')
        if normalized!=bound['text']:raise ValueError('Span source differs from bound complete normalized text')
        if type(start)is not int or type(end)is not int or not 0<=start<end<=len(normalized):
            raise ValueError('Invalid exact normalized character span')
        span=normalized[start:end]
        if re.search(r'[A-Za-z]',span):raise ValueError('English spans must use the unchanged English frontend')
        zh = self._native
        converted = zh.cn2an.transform(span, 'an2cn')
        prepared = zh.ZHG2P.map_punctuation(converted)
        targets = []
        for item in bound['targets']:
            source=item['global_position']
            if start<=source<end:
                character=item.get('character','为')
                position = self._verified_prepared_position(
                    span, source - start, converted, prepared, character)
                targets.append({**item,'position':position,'character':character,'source_span':[start,end],
                                'source_span_position':source-start,'offset_proven':True})
        return self._call(span,en_callable=en_callable,bound_targets=targets,expected_prepared=prepared)

    def _words(self,prepared):
        return [word for _, words in _native_segments(self._native, prepared) for word in words]

    def _phrase_targets(self,text,words,protected_spans=()):
        """Match only complete consecutive native words using the compiled trie."""
        if not self._phrase_trie:return []
        ends={end for _,_,end in words};targets={}
        for _,start,_ in words:
            node=self._phrase_trie;cursor=start
            while cursor<len(text) and text[cursor]in node:
                node=node[text[cursor]];cursor+=1
                record=node.get(None)
                if record is None or cursor not in ends:continue
                if any(a<cursor and start<b for a,b in protected_spans):continue
                phrase = record['phrase']
                if phrase.startswith('藏塔') and _tower_is_object_of_hiding(text, start):
                    continue
                if phrase in ('很长', '更长', '太长', '多长') and _GROWTH_SUFFIX.match(text, cursor):
                    continue
                # Han-bearing filenames/paths may lie outside the ASCII-only
                # literal reader. Their separators still disqualify a name;
                # a sentence-final period remains ordinary punctuation.
                if _touches_identifier(text, start, cursor):continue
                position=start+record['offset'];pinyin=record['pinyin']
                if position in targets and targets[position]['pinyin']!=pinyin:
                    raise RuntimeError('Conflicting complete-phrase Chinese readings')
                targets[position]={'position':position,'character':text[position],'pinyin':pinyin,
                    'context':record['phrase'],'rule':'reviewed_complete_native_phrase_syllable',
                    'native_word_boundary_verified':True,'phrase_source_span':[start,cursor]}
        return [targets[key]for key in sorted(targets)]

    def _targets(self,prepared,words):
        targets={}
        def add(position,context,rule,character='为',pinyin='wei2',**proof):
            if prepared[position]!=character:raise RuntimeError('Context target is not the exact '+character+' syllable')
            if not any(start<=position<end for _,start,end in words):raise RuntimeError('No native word contains the target')
            targets[position]={'position':position,'character':character,'pinyin':pinyin,'context':context,'rule':rule,**proof}
        protected_spans = None
        if self._context.get() is None:
            protected_spans = _protected_term_spans(prepared)
            for item in _wei_targets(prepared, words, protected_spans):
                add(**item)
        for item in _syntax_targets(prepared, words, protected_spans=protected_spans):
            add(**item)
        return [targets[k]for k in sorted(targets)]

    def _existing_fixes(self,prepared):
        renderer=getattr(self.upstream,'_render',None)
        if renderer is None:return {}
        function=getattr(renderer,'__func__',None);finder=getattr(function,'__globals__',{}).get('find_context_interventions')
        if not callable(finder):raise RuntimeError('Unknown upstream context renderer; refuse to bypass existing rules')
        return {item['start']:item['pinyin_override_at_start']for item in finder(prepared)}

    def _render(self,prepared,fixes):
        return _render_native_zh(self._native, prepared, fixes)

    def __call__(self,text,en_callable=None):
        return self._call(text,en_callable=en_callable)

    def _call(self,text,en_callable=None,bound_targets=(),expected_prepared=None):
        original=self.upstream(text,en_callable=en_callable)
        if not(isinstance(original,tuple)and len(original)==2 and isinstance(original[0],str)):
            raise TypeError('Expected unchanged Misaki (phonemes, tokens) tuple ABI')
        zh=self._native;prepared=zh.ZHG2P.map_punctuation(zh.cn2an.transform(text,'an2cn'))
        if expected_prepared is not None and prepared!=expected_prepared:raise RuntimeError('Per-span preparation changed after offset proof')
        words=self._words(prepared);by_position={item['position']:item for item in self._targets(prepared,words)}
        # A bound article owns its verified phrase targets and literal guards.
        # Without that proof, accept only ordinary Chinese prose; a Chinese gap
        # inside a URL, code literal or mixed source must not acquire a rule.
        if self._context.get()is None and re.fullmatch(r'[\u4E00-\u9FFF\s，。！？；：、（）“”‘’《》「」『』,.!?;:]+',text):
            for item in self._phrase_targets(prepared,words):
                position=item['position']
                if position in by_position and by_position[position]['pinyin']!=item['pinyin']:
                    raise RuntimeError('Phrase reading conflicts with an existing context rule')
                by_position[position]=item
        for item in bound_targets:
            position=item['position']
            if prepared[position]!=item.get('character','为') or not any(start<=position<end for _,start,end in words):
                raise RuntimeError('Bound target is not the exact native Chinese syllable')
            if position in by_position and by_position[position]['pinyin']!=item['pinyin']:
                raise RuntimeError('Bound phrase reading conflicts with an existing context rule')
            by_position[position]=item
        targets=[by_position[k]for k in sorted(by_position)];self.last_interventions=targets
        if not targets:return original
        existing=self._existing_fixes(prepared)
        if self._render(prepared,existing)!=original[0]:raise RuntimeError('Full upstream replay differs; candidate must not overwrite unrelated rules')
        fixes=dict(existing)
        for target in targets:
            position=target['position']
            pinyin=target['pinyin']
            if position in fixes and fixes[position]!=pinyin:raise RuntimeError('Conflicting existing pronunciation rule')
            fixes[position]=pinyin
        return self._render(prepared,fixes),original[1]


class Frontend:
    """Keep the native (phonemes, ids) ABI; record actual interventions separately.

    Explicit zh/en selects the normalization language. G2P still follows script
    when the input contains both Han and Latin, so selecting English number
    readings never sends Chinese words into the English grapheme converter."""

    def __init__(self, config_path=None, terms_path=None):
        if config_path is None:
            config_path = Path(os.environ.get('XIAOYOU_V6_MODEL_DIR', '/models/01yuewen')) / 'config.json'
        self.config_path = Path(config_path)
        if terms_path is None:
            terms_path = self.config_path.parent / TERMS_FILE
        from misaki import en, espeak, zh
        if version('misaki') != '0.9.4':
            raise ValueError('V6 frontend requires the verified Misaki 0.9.4')
        self.terms_path = Path(terms_path)
        self.vocab = json.loads(self.config_path.read_text())['vocab']
        terms_data = self.terms_path.read_bytes()
        specification = json.loads(terms_data)
        terms = specification['terms']
        required = {'CUDA': 'kˈudə', 'cuda': 'kˈudə', 'Cuda': 'kˈudə', 'SageAIOS': 'sˈAʤ ˈA ˈI ˈO ˈɛs'}
        aliases = specification.get('case_aliases', {})
        if (not isinstance(aliases, dict) or set(terms) != set(required) | set(aliases)
                or any(terms.get(word) != phones for word, phones in required.items())):
            raise ValueError('Required native Xiaoyou readings or exact alias inventory differ')
        for alias, record in aliases.items():
            canonical = record.get('canonical', '')
            if (not canonical or alias == canonical or alias.casefold() != canonical.casefold()
                    or terms[alias] != record.get('phonemes')
                    or record.get('source') != 'current_xiaoyou_Misaki_canonical_spelling'):
                raise ValueError('Only explicitly reviewed complete-token case aliases are allowed')
        for word, phones in terms.items():
            unknown = set(phones) - set(self.vocab)
            if unknown:
                raise ValueError(f'Invalid V6 phonemes for {word}: {sorted(unknown)}')
        case_terms = {record['canonical'].casefold(): record for record in aliases.values()}
        case_terms['cuda'] = {'canonical': 'CUDA', 'phonemes': required['CUDA']}
        extension = EnglishLexiconExtension(
            specification.get('english_lexicon'), self.vocab,
            reserved_casefold=set(case_terms) | {word.casefold() for word in required})
        class ExactTermG2P(en.G2P):
            _source_span = ContextVar('xiaoyou_english_source_span', default=None)
            _spell_letters = ContextVar('xiaoyou_english_spell_letters', default=False)

            def __call__(self, text, *args, **kwargs):
                with extension.scope(text, self._source_span.get()):
                    return super().__call__(text, *args, **kwargs)

            def call_span(self, normalized, start, end):
                source_token = self._source_span.set((normalized, start, end, _protected_term_spans(normalized)))
                try:
                    return self(normalized[start:end])
                finally:
                    self._source_span.reset(source_token)

            def call_literal(self, normalized, start, end, reading, *, spell_letters=False):
                # Keep the ORIGINAL literal scope while speaking its generated
                # structure. Existing source-span guards reject casefold term
                # aliases inside it; canonical required readings stay native.
                source_token = self._source_span.set((normalized, start, end, _protected_term_spans(normalized)))
                spelling_token = self._spell_letters.set(spell_letters)
                try:
                    return self(reading)
                finally:
                    self._spell_letters.reset(spelling_token)
                    self._source_span.reset(source_token)

            def tokenize(self, text, tokens, features):
                native_tokens = super().tokenize(text, tokens, features)
                automatic_aliases = set()
                # Mark only complete spaCy MTokens before Misaki subtokenizes.
                # Explicit user phonemes remain authoritative.
                cursor = 0
                source_span = self._source_span.get()
                spell_letters = self._spell_letters.get()
                for index, token in enumerate(native_tokens):
                    local_start = text.find(token.text, cursor)
                    local_end = local_start + len(token.text)
                    cursor = max(cursor, local_end)
                    # Keep the four original user/native readings unchanged.
                    phones = required.get(token.text)
                    if spell_letters and len(token.text) == 1 and 'A' <= token.text <= 'Z':
                        phones = self.lexicon.golds[token.text]
                    record = case_terms.get(token.text.casefold()) if token.text.isascii() else None
                    if phones is None and record is not None and token.text != record['canonical'] and local_start >= 0:
                        source, a, b = text, local_start, local_end
                        if source_span is not None:
                            full, start, end, protected_spans = source_span
                            if full[start:end] != text:
                                continue  # Preprocessed markup has different offsets.
                            source, a, b = full, start + a, start + b
                            if any(lo < b and a < hi for lo, hi in protected_spans):
                                continue
                        before = source[a-1:a] if a else ''
                        after = source[b:b+1]
                        blocked_left = bool(before) and (before.isascii() and before.isalnum() or before in '_./\\@-')
                        blocked_right = bool(after) and (after.isascii() and after.isalnum() or after in '_/\\@-')
                        file_suffix = after == '.' and bool(source[b+1:b+2]) and source[b+1:b+2].isascii() and (source[b+1:b+2].isalnum() or source[b+1:b+2] == '_')
                        if not (blocked_left or blocked_right or file_suffix):
                            phones = record['phonemes']
                    if phones is not None and token.phonemes is None:
                        token.phonemes = phones
                        token._.rating = 5
                        if record is not None and record['canonical'] not in required:
                            automatic_aliases.add(index)
                return extension.apply_overrides(text, native_tokens, automatic_aliases=automatic_aliases)

        self.en = ExactTermG2P(trf=False, british=False,
                              fallback=extension.wrap_fallback(espeak.EspeakFallback(british=False)), unk='❓')
        extension.bind_native(self.en.lexicon, en.TokenContext())
        self.english_lexicon = extension
        self.zh = PolyphoneContextG2P(
            ContextualZHG2P(zh.ZHG2P(version=None, unk='❓')),
            phrase_readings=specification.get('chinese_phrase_readings'))
        self.term_phonemes = dict(terms)
        self.last_plan = None
        sources = source_hashes()
        self.metadata = {
            'project_version': 'xiaoyou_v6_business_lexicon_candidate_20261117',
            'misaki_version': '0.9.4',
            'frontend_path': str(Path(__file__).resolve()),
            'frontend_sha256': sources[FRONTEND_FILE],
            'config_path': str(self.config_path.resolve()),
            'config_sha256': sha256(self.config_path.read_bytes()).hexdigest(),
            'terms_path': str(self.terms_path.resolve()),
            'terms_sha256': sha256(terms_data).hexdigest(),
            'term_overrides': dict(terms),
            'english_lexicon': dict(extension.metadata),
            'english_lexicon_source_adapted': True,
            'legacy_rule_reuse': False,
            'raw_text_substitution': True,
            'normalization_order': ['protected_identifiers', 'clause_carrier_language', 'typed_contexts', 'pinned_general', 'pinned_math', 'NFKC_and_whitespace', 'restore_identifiers', 'native_G2P_with_scoped_tones'],
            'normalization_module_sha256': {name: sources[name] for name in NORMALIZATION_MODULE_SHAS},
            'context_pronunciation': self.zh.metadata,
            'override_scope': 'approved_complete_native_spacy_token_casefold_with_complete_source_span_identifier_guards',
            'lexicon_golds_modified': False,
            'candidate_only': True,
            'active_profile_changed': False,
            'candidate_sources': sources,
            'script_routing': 'Han to native Chinese; Latin to native English; no persistent language state',
            'quality_accepted': False,
        }

    @staticmethod
    def normalize_text(text, language='auto', *, auto_fallback=None):
        # Keep explicit version identities whole through numeric normalization;
        # otherwise e.g. version 2.11.0 adjacent to Han can lose a separator.
        # Their numbers are rendered only in phonemize_normalized, with the
        # complete original normalized offsets retained for scoped Chinese G2P.
        if not isinstance(text,str):return numeric.normalize_context(text,language,auto_fallback=auto_fallback)
        # Strip decorative graphemes before calculating any protected spans.
        # A separating space keeps adjacent words and numbers distinct.
        text = _EMOJI.sub(lambda match: match['key'] or ' ', text)
        addresses=tuple(numeric.address_ranges(text,include_absolute_paths=True))
        versions=_version_spans(text,0,len(text),addresses)
        if not versions:return numeric.normalize_context(text,language,auto_fallback=auto_fallback)
        parts=[];mapping=[];cursor=0
        for a,b,value in versions:
            index=1000000+len(mapping);token=numeric._token(index)
            while token in text or any(token==item[0]for item in mapping):
                index+=1;token=numeric._token(index)
            parts.extend([text[cursor:a],token]);mapping.append((token,value));cursor=b
        parts.append(text[cursor:]);normalized=numeric.normalize_context(''.join(parts),language,auto_fallback=auto_fallback)
        for token,value in mapping:
            if normalized.count(token)!=1:raise ValueError('Explicit version placeholder was not preserved')
            normalized=normalized.replace(token,value)
        return normalized

    def _english_spans(self, normalized, start, end):
        """Extend script routing only across an approved whole business name.

        Generic punctuation keeps its existing routing. A reviewed name such
        as HAMi(RISE)vGPU or Frost & Sullivan stays in one native English call
        even in Han text, retaining its whole phrase match and source guards.
        """
        spans = [match.span() for match in _LATIN.finditer(normalized, start, end)]
        protected = list(_protected_term_spans(normalized))
        protected.extend(match.span() for match in _PROTECTED.finditer(normalized))
        for a, b, _ in self.english_lexicon._phrase_matches(normalized[start:end]):
            a, b = start + a, start + b
            if (_complete_word(normalized, a, b)
                    and not any(lo < b and a < hi for lo, hi in protected)):
                spans.append((a, b))
        # Overlapping original Latin spans become one source interval. Never
        # glue disjoint clauses or insert text while grouping a business name.
        result = []
        for a, b in sorted(spans):
            if result and a < result[-1][1]:
                result[-1] = (result[-1][0], max(b, result[-1][1]))
            else:
                result.append((a, b))
        return result

    def phonemize_normalized(self, normalized, language='auto', *, start=0, end=None, normalization_language=None, auto_fallback=None):
        """G2P an exact semantic span while retaining the complete request context.

        Input has ALREADY passed normalize_text exactly once. This method never
        calls the general/math/typed normalizers again.
        """
        if language not in ('auto','mixed','zh','en'):
            raise ValueError('Unsupported candidate phoneme language')
        self.last_plan=None
        if not isinstance(normalized,str):raise TypeError('Normalized text must be a string')
        end=len(normalized) if end is None else end
        if type(start)is not int or type(end)is not int or not 0<=start<end<=len(normalized):
            raise ValueError('Invalid exact normalized semantic span')
        ascii_word=lambda c:c.isascii() and (c.isalnum() or c=='_')
        if start and ascii_word(normalized[start-1]) and ascii_word(normalized[start]):
            raise ValueError('Semantic span starts inside a protected English token')
        if end<len(normalized) and ascii_word(normalized[end-1]) and ascii_word(normalized[end]):
            raise ValueError('Semantic span ends inside a protected English token')
        span=normalized[start:end];interventions=[];version_readings=[]
        version_language=language if normalization_language is None else normalization_language
        if version_language not in ('auto','mixed','zh','en'):raise ValueError('Unsupported version language')
        numeric._validate_auto_fallback(auto_fallback)
        addresses=tuple(numeric.address_ranges(normalized,include_absolute_paths=True))
        events=[]
        for a,b in addresses:
            if a<start<b or a<end<b:
                raise ValueError('Semantic span splits a complete address')
            if start<=a<b<=end:events.append((a,b,normalized[a:b],'address'))
        filenames=[]
        for a,b in numeric.filename_spans(normalized):
            if any(lo<b and a<hi for lo,hi in addresses):
                continue
            if a<start<b or a<end<b:
                raise ValueError('Semantic span splits a complete filename')
            if start<=a<b<=end:filenames.append((a,b,normalized[a:b]))
        versions=[item for item in _version_spans(normalized,start,end,addresses)
                  if not any(a<item[1] and item[0]<b for a,b,_ in filenames)]
        filename_readings=[];address_readings=[]
        events.extend((a,b,value,'version') for a,b,value in versions)
        events.extend((a,b,value,'filename') for a,b,value in filenames)
        english_only = not _HAN.search(span) and language in ('auto', 'mixed', 'en')
        if english_only and not events:
            phonemes = self.en.call_span(normalized, start, end)[0]
        else:
            result = []

            def plain(a, b):
                if not normalized[a:b].strip():
                    return
                if english_only:
                    result.append(self.en.call_span(normalized, a, b)[0])
                    return
                cursor = a
                for latin_start, latin_end in self._english_spans(normalized, a, b):
                    if normalized[cursor:latin_start].strip():
                        result.append(self.zh.call_span(normalized, cursor, latin_start)[0])
                        interventions.extend(self.zh.last_interventions)
                    result.append(self.en.call_span(normalized, latin_start, latin_end)[0])
                    cursor = latin_end
                if normalized[cursor:b].strip():
                    result.append(self.zh.call_span(normalized, cursor, b)[0])
                    interventions.extend(self.zh.last_interventions)

            with self.zh.full_context(normalized, protected_spans=_protected_term_spans(normalized)):
                cursor=start
                for a,b,value,kind in sorted(events):
                    plain(cursor,a)
                    fallback=(auto_fallback or 'zh') if kind=='address' else auto_fallback
                    lang=_version_language(normalized,a,version_language,fallback)
                    if kind=='version':
                        reading=_version_reading(value,lang)
                        result.append(self.zh(reading)[0] if lang=='zh' else self.en(reading)[0])
                        version_readings.append({'source_span':[a,b],'source_text':value,'components':value.split('.'),'language':lang,'reading':reading,'rule':'explicit_version_components_preserve_dots_and_leading_zeros'})
                    else:
                        is_address=kind=='address'
                        reading=numeric.read_address(value,lang.upper()) if is_address else _filename_reading(value,lang)
                        if lang=='en' and not _HAN.search(reading):
                            result.append(self.en.call_literal(normalized,a,b,reading,spell_letters=is_address)[0])
                        else:
                            # Generated words get their own exact Chinese scope;
                            # the article and every original offset stay intact.
                            with self.zh.full_context(reading):
                                local=0
                                for match in _LATIN.finditer(reading):
                                    if local<match.start() and reading[local:match.start()].strip():
                                        result.append(self.zh.call_span(reading,local,match.start())[0])
                                    result.append(self.en.call_literal(normalized,a,b,match.group(),spell_letters=is_address)[0])
                                    local=match.end()
                                if local<len(reading) and reading[local:].strip():
                                    result.append(self.zh.call_span(reading,local,len(reading))[0])
                        record={'source_span':[a,b],'source_text':value,'language':lang,'reading':reading}
                        if is_address:
                            address_readings.append(record)
                        else:
                            filename_readings.append(dict(record,rule='whole_filename_digit_identity_and_explicit_separators'))
                    cursor=b
                plain(cursor,end)
            phonemes=' '.join(result)
        phonemes=re.sub(r'\s+',' ',phonemes).strip();unknown=sorted(set(phonemes)-set(self.vocab))
        if unknown:raise ValueError(f'Unknown phoneme symbols: {unknown!r}')
        ids=[self.vocab[p]for p in phonemes]
        if not ids:raise ValueError('Empty phoneme sequence')
        self.last_plan={'text':span,'normalization_language':version_language,'normalized_text':normalized,
                        'normalized_span':[start,end],'phonemes':phonemes,'token_ids':ids,
                        'interventions':interventions,'version_readings':version_readings,'filename_readings':filename_readings,'auto_fallback':auto_fallback,'normalization_already_applied':True}
        if address_readings:
            self.last_plan['address_readings']=address_readings
        return phonemes,ids

    def __call__(self,text,language='auto',*,normalization_language=None,auto_fallback=None):
        if language not in ('auto','mixed','zh','en'):raise ValueError('Unsupported candidate phoneme language')
        self.last_plan=None
        norm_language=language if normalization_language is None else normalization_language
        normalized=self.normalize_text(text,norm_language,auto_fallback=auto_fallback)
        result=self.phonemize_normalized(normalized,language,normalization_language=norm_language,auto_fallback=auto_fallback)
        self.last_plan.update(text=text,normalization_language=norm_language)
        return result


# Process boundary for pinned G2P dependencies on both JetPack platforms.

PYTHON = '/opt/xiaoyou-tts/bin/python3'


def require_runtime():
    """Check local prerequisites before loading a model or stopping another engine."""
    if platform.machine() != 'aarch64':
        raise RuntimeError('xiaoyou-zh-en requires JetPack on aarch64')
    major = detect_jetpack_major()
    if major is None:
        raise RuntimeError('xiaoyou-zh-en requires JetPack 5 or 6 (L4T R35 or R36)')
    if not Path(PYTHON).is_file() or not os.access(PYTHON, os.X_OK):
        raise RuntimeError('The Xiaoyou frontend environment is not installed in this image')
    return major


class FrontendWorker:
    INITIAL_TIMEOUT = 60
    REQUEST_TIMEOUT = 15

    def __init__(self, model_dir):
        self._model_dir = model_dir
        self._connection = None
        self._process = None
        self._closed = False
        self._failure = None
        self._english_model = None
        try:
            if detect_jetpack_major() == 6:
                self._english_model = TemporaryDirectory(prefix='xiaoyou-english-model-')
                unpack_jp6_english_model(model_dir, self._english_model.name)
            self.metadata, self.legal_ids, self.dictionary_seconds = self._start(self.INITIAL_TIMEOUT)
        except BaseException:
            self.close()
            raise

    @property
    def failure_reason(self):
        process = self._process
        if self._failure is not None:
            return self._failure
        if process is not None and process.poll() is not None:
            return 'Xiaoyou frontend worker stopped'
        return None

    def _start(self, timeout, *, cancelled=None):
        env = os.environ.copy()
        for name in ('PYTHONHOME', 'PYTHONPATH'):
            env.pop(name, None)
        if self._english_model is not None:
            env['PYTHONPATH'] = self._english_model.name
        self._connection, child = Pipe()
        try:
            with child:
                self._process = subprocess.Popen(
                    [PYTHON, '-Bs', '-m', __name__, str(child.fileno()), str(self._model_dir)],
                    pass_fds=(child.fileno(),), env=env, cwd=Path(__file__).resolve().parents[3],
                )
            hello = self._receive(timeout, cancelled=cancelled)
            if hasattr(self, 'metadata') and hello[:2] != (self.metadata, self.legal_ids):
                raise RuntimeError('Xiaoyou frontend worker restarted with different metadata or token IDs')
        except BaseException as error:
            self._retire(str(error))
            raise
        self._failure = None
        return hello

    def _receive(self, timeout, *, cancelled=None):
        try:
            deadline = time.monotonic() + timeout
            while True:
                if cancelled is not None and cancelled():
                    raise CancelledError('Xiaoyou frontend startup cancelled')
                remaining = max(0.0, deadline - time.monotonic())
                if self._connection.poll(remaining if cancelled is None else min(remaining, .1)):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Xiaoyou frontend worker timed out')
            status, value = self._connection.recv()
        except EOFError as error:
            raise BrokenPipeError('Xiaoyou frontend worker stopped') from error
        if status == 'ok':
            return value
        if status == 'invalid':
            raise ValueError(value)
        raise RuntimeError(value)

    def normalize_text(self, *args, cancelled=None, **options):
        return self._call('normalize', args, options, cancelled=cancelled)

    def __call__(self, *args, cancelled=None, **options):
        return self._call('phonemize', args, options, cancelled=cancelled)

    def phonemize_normalized(self, *args, cancelled=None, **options):
        return self._call('normalized_span', args, options, cancelled=cancelled)

    def _call(self, operation, args, options, *, cancelled=None):
        # The backend serializes RPC and close. Recover before sending a new
        # operation so an interrupted or partial reply is never replayed.
        if self._closed:
            raise RuntimeError('Xiaoyou frontend worker is closed')
        if cancelled is not None and cancelled():
            raise CancelledError('Xiaoyou frontend request cancelled')
        process = self._process
        if process is None or process.poll() is not None:
            self._retire(self._failure or 'Xiaoyou frontend worker stopped')
            self._start(self.INITIAL_TIMEOUT, cancelled=cancelled)
        if cancelled is not None and cancelled():
            raise CancelledError('Xiaoyou frontend request cancelled')
        deadline = time.monotonic() + self.REQUEST_TIMEOUT
        try:
            self._connection.send((operation, args, options))
            return self._receive(max(0.0, deadline - time.monotonic()))
        except TimeoutError as error:
            self._retire(str(error))
            raise
        except OSError as error:
            message = 'Xiaoyou frontend worker stopped'
            self._retire(message)
            raise RuntimeError(message) from error

    def _retire(self, reason=None):
        self._failure = reason
        connection, self._connection = self._connection, None
        process, self._process = self._process, None
        if connection is not None:
            connection.close()
        if process is None:
            return
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self):
        self._closed = True
        try:
            self._retire()
        finally:
            if self._english_model is not None:
                self._english_model.cleanup()
                self._english_model = None


def run(connection, model_dir):
    # Thinc supports a NumPy-only frontend if Torch is unavailable. Keep this
    # import policy inside the dedicated process.
    sys.modules['torch'] = None
    try:
        import jieba

        if version('en-core-web-sm') != '3.8.0':
            raise RuntimeError('Xiaoyou requires the pinned spaCy English model 3.8.0')
        started = time.perf_counter()
        jieba.initialize()
        if not jieba.dt.initialized:
            raise RuntimeError('Xiaoyou Jieba dictionary initialization did not complete')
        dictionary_seconds = time.perf_counter() - started
        frontend = Frontend(config_path=model_dir / 'config.json', terms_path=model_dir / TERMS_FILE)
        metadata = {key: frontend.metadata[key] for key in (
            'frontend_sha256', 'terms_sha256', 'normalization_module_sha256',
        )}
        connection.send(('ok', (metadata, set(frontend.vocab.values()) | {0}, dictionary_seconds)))
    except Exception as error:
        connection.send(('error', f'{type(error).__name__}: {error}'))
        return

    methods = {'normalize': frontend.normalize_text, 'phonemize': frontend,
               'normalized_span': frontend.phonemize_normalized}
    while True:
        try:
            operation, args, options = connection.recv()
        except EOFError:
            return
        try:
            value = methods[operation](*args, **options)
        except ValueError as error:
            connection.send(('invalid', str(error)))
        except Exception as error:
            connection.send(('error', f'{type(error).__name__}: {error}'))
        else:
            connection.send(('ok', value))


if __name__ == '__main__':
    with Connection(int(sys.argv[1])) as channel:
        run(channel, Path(sys.argv[2]))
