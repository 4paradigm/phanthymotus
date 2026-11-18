"""Xiaoyou text-to-phoneme frontend with complete normalized-span context.

One implementation owns initialization, explicit language normalization and
Misaki conversion. Text/phoneme/token ABI and scoped rules match the reviewed
runtime; model configuration and term paths remain externally injectable.
"""
from hashlib import sha256
from contextvars import ContextVar
from contextlib import contextmanager
from bisect import bisect_right
from functools import lru_cache
from importlib.metadata import version
import json
import os
from pathlib import Path
import re

from . import normalization as general
from . import numbers as numeric
from .pronunciation import ContextualZHG2P, PolyphoneContextG2P
from ..runtime.model_bundle import FRONTEND_FILE, NORMALIZATION_MODULE_SHAS, TERMS_FILE

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_LATIN_WORD = r"(?:[0-9]+(?=[A-Za-z]))?[A-Za-z][A-Za-z0-9_'’\-]*"
_LATIN = re.compile(_LATIN_WORD + r"(?:[ \t]+" + _LATIN_WORD + r")*")
_HAN = re.compile(r'[\u3400-\u9fff]')

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

def _version_spans(normalized, start, end):
    excluded = [m.span() for m in _VERSION_URL.finditer(normalized)]
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
    readings = [general.speak_number(part, language, len(part) > 1 and part.startswith('0'))
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
            rendered.append(('' if language == 'zh' else ' ').join(
                general.speak_number(digit, language, True) for digit in part))
        else:
            rendered.append(separators.get(part, part))
    result = ' '.join(rendered)
    return re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])', '', result)

@lru_cache(maxsize=32)
def _protected_term_spans(normalized):
    """Use this frontend's literal masks to guard aliases inside full context."""
    _, mapping = numeric._protect(normalized, general)
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

    Integration: bind_native(g2p.lexicon) after G2P construction; wrap every
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
        self._native_lexicon = None
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

    def bind_native(self, lexicon):
        # Preserve all native context/POS dictionaries, even explicit broad
        # source entries such as AA/BA/CIS/ISO that would otherwise shadow them.
        gold = getattr(lexicon, 'golds', None)
        if not isinstance(gold, dict):
            raise TypeError('Expected native Misaki English Lexicon.golds')
        self._context_words = frozenset(word.casefold() for word, value in gold.items()
                                         if isinstance(value, dict))
        self._native_lexicon = lexicon if callable(lexicon) else None

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
        if self._native_lexicon is None:
            return False
        native = []
        for token in tokens:
            value, _ = self._native_lexicon(token, None)
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
                if len(group) != 1 or self._native_lexicon is None:
                    continue
                native, _ = self._native_lexicon(group[0], None)
                if native is not None:
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

            def __call__(self, text, *args, **kwargs):
                with extension.scope(text, self._source_span.get()):
                    return super().__call__(text, *args, **kwargs)

            def call_span(self, normalized, start, end):
                source_token = self._source_span.set((normalized, start, end, _protected_term_spans(normalized)))
                try:
                    return self(normalized[start:end])
                finally:
                    self._source_span.reset(source_token)

            def call_literal(self, normalized, start, end, reading):
                # Keep the ORIGINAL filename scope while speaking its generated
                # structure. Existing source-span guards reject casefold term
                # aliases inside it; canonical required readings stay native.
                source_token = self._source_span.set((normalized, start, end, _protected_term_spans(normalized)))
                try:
                    return self(reading)
                finally:
                    self._source_span.reset(source_token)

            def tokenize(self, text, tokens, features):
                native_tokens = super().tokenize(text, tokens, features)
                automatic_aliases = set()
                # Mark only complete spaCy MTokens before Misaki subtokenizes.
                # Explicit user phonemes remain authoritative.
                cursor = 0
                source_span = self._source_span.get()
                for index, token in enumerate(native_tokens):
                    local_start = text.find(token.text, cursor)
                    local_end = local_start + len(token.text)
                    cursor = max(cursor, local_end)
                    # Keep the four original user/native readings unchanged.
                    phones = required.get(token.text)
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
        extension.bind_native(self.en.lexicon)
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
        versions=_version_spans(text,0,len(text))
        if not versions:return numeric.normalize_context(text,language,auto_fallback=auto_fallback)
        parts=[];mapping=[];cursor=0
        for a,b,value in versions:
            index=1000000+len(mapping);token=general._token(index)
            while token in text or any(token==item[0]for item in mapping):
                index+=1;token=general._token(index)
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
        filenames=[]
        for a,b in numeric.filename_spans(normalized):
            if a<start<b or a<end<b:
                raise ValueError('Semantic span splits a complete filename')
            if start<=a<b<=end:filenames.append((a,b,normalized[a:b]))
        versions=[item for item in _version_spans(normalized,start,end)
                  if not any(a<item[1] and item[0]<b for a,b,_ in filenames)]
        filename_readings=[]
        events=sorted([(a,b,value,'version')for a,b,value in versions]+
                      [(a,b,value,'filename')for a,b,value in filenames])
        if events:
            result=[]
            with self.zh.full_context(normalized, protected_spans=_protected_term_spans(normalized)):
                def plain(a,b):
                    if a==b or not normalized[a:b].strip():return
                    if not _HAN.search(span) and language in ('auto','mixed','en'):
                        result.append(self.en.call_span(normalized,a,b)[0]);return
                    cursor=a
                    for latin_start,latin_end in self._english_spans(normalized,a,b):
                        if cursor<latin_start and normalized[cursor:latin_start].strip():
                            result.append(self.zh.call_span(normalized,cursor,latin_start)[0]);interventions.extend(self.zh.last_interventions)
                        result.append(self.en.call_span(normalized,latin_start,latin_end)[0]);cursor=latin_end
                    if cursor<b and normalized[cursor:b].strip():
                        result.append(self.zh.call_span(normalized,cursor,b)[0]);interventions.extend(self.zh.last_interventions)
                cursor=start
                for a,b,value,kind in events:
                    plain(cursor,a)
                    lang=_version_language(normalized,a,version_language,auto_fallback)
                    if kind=='version':
                        reading=_version_reading(value,lang)
                        result.append(self.zh(reading)[0] if lang=='zh' else self.en(reading)[0])
                        version_readings.append({'source_span':[a,b],'source_text':value,'components':value.split('.'),'language':lang,'reading':reading,'rule':'explicit_version_components_preserve_dots_and_leading_zeros'})
                    else:
                        reading=_filename_reading(value,lang)
                        if lang=='en':
                            result.append(self.en.call_literal(normalized,a,b,reading)[0])
                        else:
                            # Generated words get their own exact Chinese scope;
                            # the article and every original offset stay intact.
                            with self.zh.full_context(reading):
                                local=0
                                for match in _LATIN.finditer(reading):
                                    if local<match.start() and reading[local:match.start()].strip():
                                        result.append(self.zh.call_span(reading,local,match.start())[0])
                                    result.append(self.en.call_literal(normalized,a,b,match.group())[0])
                                    local=match.end()
                                if local<len(reading) and reading[local:].strip():
                                    result.append(self.zh.call_span(reading,local,len(reading))[0])
                        filename_readings.append({'source_span':[a,b],'source_text':value,'language':lang,
                            'reading':reading,'rule':'whole_filename_digit_identity_and_explicit_separators'})
                    cursor=b
                plain(cursor,end)
            phonemes=' '.join(result)
        elif not _HAN.search(span) and language in ('auto','mixed','en'):
            phonemes=self.en.call_span(normalized,start,end)[0]
        else:
            result=[]
            with self.zh.full_context(normalized, protected_spans=_protected_term_spans(normalized)):
                cursor=start
                for latin_start,latin_end in self._english_spans(normalized,start,end):
                    if cursor<latin_start and normalized[cursor:latin_start].strip():
                        result.append(self.zh.call_span(normalized,cursor,latin_start)[0]);interventions.extend(self.zh.last_interventions)
                    result.append(self.en.call_span(normalized,latin_start,latin_end)[0]);cursor=latin_end
                if cursor<end and normalized[cursor:end].strip():
                    result.append(self.zh.call_span(normalized,cursor,end)[0]);interventions.extend(self.zh.last_interventions)
            phonemes=' '.join(result)
        phonemes=re.sub(r'\s+',' ',phonemes).strip();unknown=sorted(set(phonemes)-set(self.vocab))
        if unknown:raise ValueError(f'Unknown phoneme symbols: {unknown!r}')
        ids=[self.vocab[p]for p in phonemes]
        if not ids:raise ValueError('Empty phoneme sequence')
        self.last_plan={'text':span,'normalization_language':version_language,'normalized_text':normalized,
                        'normalized_span':[start,end],'phonemes':phonemes,'token_ids':ids,
                        'interventions':interventions,'version_readings':version_readings,'filename_readings':filename_readings,'auto_fallback':auto_fallback,'normalization_already_applied':True}
        return phonemes,ids

    def __call__(self,text,language='auto',*,normalization_language=None,auto_fallback=None):
        if language not in ('auto','mixed','zh','en'):raise ValueError('Unsupported candidate phoneme language')
        self.last_plan=None
        norm_language=language if normalization_language is None else normalization_language
        normalized=self.normalize_text(text,norm_language,auto_fallback=auto_fallback)
        result=self.phonemize_normalized(normalized,language,normalization_language=norm_language,auto_fallback=auto_fallback)
        self.last_plan.update(text=text,normalization_language=norm_language)
        return result
