"""CPU-only long-text segmentation candidate derived from pinned V6 runtime.

No normalization, pronunciation, waveform, speed or model changes. Call safe_split
only after the existing verified profile rejects a clause. The first-output plan
also admits safe colons and enumeration boundaries after two complete items.
Concatenation preserves source strings including punctuation and whitespace.
"""
import re

_ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs'}
# Sentence-boundary hints only. These never select pronunciations or replace text.
_JOINED_SENTENCE_STARTERS = frozenset(('I You We They He She It The A An This That These Those '
    'Today Tomorrow Yesterday What Why When Where Who How Please Let Thank Then Next First Second '
    'Finally However Meanwhile Instead For In On At After Before If As Our Your Their Here There Now').split())
# Preserve common domain/file uses even when a suffix happens to be a sentence hint.
_DOMAIN_SUFFIXES = frozenset('com org net io ai dev app edu gov co uk cn hk xyz info biz me tech today cloud online site'.split())


def _math_patterns():
    """Recognize existing raw/spoken math atoms for boundaries, never render them."""
    gap = r'[ \t]*'
    number = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+'
    variable = r'[A-Za-z](?:[₀₁₂₃₄₅₆₇₈₉]+)?'
    raw_op = r'(?:>=|<=|!=|==|[=≠≥≤<>+\-−*×/÷^])'
    implicit = rf'(?:{number}){gap}{variable}'
    raw_core = rf'(?:{implicit}|(?:{number})(?:{gap}[%％])?|{variable}|π)'
    raw_simple = rf'(?:[+\-−]{gap})?{raw_core}(?:[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ]+)?'
    raw_body = rf'{gap}{raw_simple}(?:{gap}{raw_op}{gap}{raw_simple})*{gap}'
    raw_group = rf'(?:\({raw_body}\)|\|{raw_body}\|)'
    raw_atom = rf'(?:[+\-−]{gap})?(?:(?:sin|sqrt){gap}{raw_group}|√{gap}{raw_simple}|{raw_group}|{raw_simple})'

    zh_digit = r'[零〇○一二两三四五六七八九]'
    zh_number = rf'[零〇○一二两三四五六七八九十百千万亿]+(?:{gap}点{gap}{zh_digit}(?:{gap}{zh_digit})*)?'
    zh_sign = rf'(?:[正负]{gap})?'
    zh_var = rf'{variable}(?:{gap}下标{gap}{zh_sign}{zh_number})?'
    zh_core = rf'(?:百分之{gap}{zh_sign}{zh_number}|{zh_number}|{zh_var}|派)'
    zh_simple = zh_sign + zh_core
    zh_op = r'(?:大于等于|小于等于|不等于|等于|乘以|除以|大于|小于|加|减|乘)'
    zh_group = rf'左括号{gap}{zh_simple}(?:{gap}{zh_op}{gap}{zh_simple})*{gap}右括号'
    zh_power = rf'(?:{gap}的(?:平方|立方|绝对值)|{gap}的{gap}{zh_simple}{gap}次方)'
    zh_atom = rf'{zh_sign}(?:(?:根号|正弦|平方根){gap}{zh_simple}|{zh_group}|{zh_core})(?:{zh_power})*'

    digit = r'(?:zero|one|two|three|four|five|six|seven|eight|nine)'
    word = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|quadrillion)'
    en_number = rf'{word}(?:[ \t]+{word})*(?:[ \t]+point[ \t]+{digit}(?:[ \t]+{digit})*)?'
    en_sign = r'(?:(?:negative|positive)[ \t]+)?'
    en_var = rf'{variable}(?:[ \t]+subscript[ \t]+{en_sign}{en_number})?'
    en_core = rf'(?:{en_number}(?:[ \t]+percent)?|{en_var}|pi)'
    en_simple = en_sign + en_core
    en_op = r'(?:greater[ \t]+than[ \t]+or[ \t]+equal[ \t]+to|less[ \t]+than[ \t]+or[ \t]+equal[ \t]+to|does[ \t]+not[ \t]+equal|greater[ \t]+than|less[ \t]+than|divided[ \t]+by|equals|plus|minus|times)'
    en_group = rf'open[ \t]+parenthesis[ \t]+{en_simple}(?:[ \t]+{en_op}[ \t]+{en_simple})*[ \t]+close[ \t]+parenthesis'
    en_power = rf'(?:[ \t]+(?:squared|cubed)|[ \t]+to[ \t]+the[ \t]+power[ \t]+of[ \t]+{en_simple})'
    en_atom = rf'{en_sign}(?:(?:square[ \t]+root[ \t]+of|sine[ \t]+of|the[ \t]+absolute[ \t]+value[ \t]+of)[ \t]+{en_simple}|{en_group}|{en_core})(?:{en_power})*'
    # Single-letter variables only; complete operators and numeric-word atoms
    # prevent ordinary phrases such as "business goal equals customer value"
    # from swallowing prose. Existing multi-letter identifiers are separate.
    left, right = r'(?<![A-Za-z0-9_])', r'(?![A-Za-z0-9_])'
    expressions = [rf'{raw_atom}(?:{gap}{raw_op}{gap}{raw_atom})+',
                   rf'{zh_atom}(?:{gap}{zh_op}{gap}{zh_atom})+',
                   rf'{en_atom}(?:[ \t]+{en_op}[ \t]+{en_atom})+',
                   # Standalone rendered powers/roots/groups and spoken numbers
                   # remain indivisible when no comparison follows them.
                   rf'(?:(?:sin|sqrt){gap}{raw_group}|√{gap}{raw_simple}|{raw_group}|{raw_core}[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ]+|{implicit})',
                   rf'(?:(?:根号|正弦|平方根){gap}{zh_simple}|{zh_group}|{zh_core}(?:{zh_power})+|{variable}{gap}下标{gap}{zh_simple}|{zh_sign}{zh_number})',
                   rf'(?:(?:square[ \t]+root[ \t]+of|sine[ \t]+of|the[ \t]+absolute[ \t]+value[ \t]+of)[ \t]+{en_simple}|{en_group}|{en_core}(?:{en_power})+|{variable}[ \t]+subscript[ \t]+{en_simple}|{en_sign}{en_number})']
    return tuple(re.compile(left + expression + right, re.IGNORECASE) for expression in expressions)


_MATH_PATTERNS = _math_patterns()


def _math_expression_spans(text):
    found = set()
    view = text
    # Recognized inner atoms become a same-length variable in this temporary
    # recognition view only. This admits nested existing groups/powers without
    # changing a source character, reading, punctuation or returned substring.
    depth = text.count('(') + text.count('左括号') + len(re.findall(r'open[ \t]+parenthesis', text, re.I))
    for _ in range(depth + 2):
        matches = []
        for pattern in _MATH_PATTERNS:
            for match in pattern.finditer(view):
                start = match.start()
                # Include only an immediately adjacent explicit math label,
                # never the rest of a clause. Unlabelled math is valid too.
                label = re.search(r'(?:公式|(?<![A-Za-z0-9_])formula)[ \t]*$', text[:start], re.IGNORECASE)
                if label:
                    start = label.start()
                matches.append((start, match.end()))
        found.update(matches)
        merged = []
        for start, stop in sorted(matches):
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        following = list(view)
        for start, stop in merged:
            following[start:stop] = 'x' + ' ' * (stop - start - 1)
        following = ''.join(following)
        if following == view:
            break
        view = following
    yield from sorted(found)


def _joined_period_boundary(text, dot):
    """Conservative no-space sentence ending, e.g. today.You; not any dotted ID."""
    if dot < 1 or dot + 1 >= len(text) or text[dot] != '.':
        return False
    left = re.search(r'([A-Za-z]+)$', text[:dot])
    right = re.match(r'([A-Z][a-z]+|I)(?![A-Za-z0-9_])', text[dot+1:])
    if not left or not right or not left.group(1).islower():
        return False
    if left.group(1) in _ABBREVIATIONS or len(left.group(1)) == 1:
        return False
    if right.group(1) not in _JOINED_SENTENCE_STARTERS:
        return False
    if right.group(1).lower() in _DOMAIN_SUFFIXES:
        # Today is also a domain suffix. Admit its explicit prose use only when
        # the following separate word is a subject, e.g. everyone.Today we begin.
        following = text[dot+1+len(right.group(1)):]
        if right.group(1) != 'Today' or not re.match(r'\s+(?:I|we|you|he|she|they|our|your)\b', following):
            return False
    # Do not reinterpret URL/path/email/version/compound ASCII tokens as prose.
    token_start = dot
    while token_start and re.match(r"[A-Za-z0-9_./:'’+@\\-]", text[token_start-1]):
        token_start -= 1
    token_end = dot + 1
    while token_end < len(text) and re.match(r"[A-Za-z0-9_./:'’+@\\-]", text[token_end]):
        token_end += 1
    token = text[token_start:token_end]
    if any(c in token for c in '/:_+@\\-'):
        return False
    # A final full stop may follow the joined next word, but repeated interior dots
    # indicate an abbreviation/domain/version and stay protected.
    if token.rstrip('.').count('.') != 1:
        return False
    return True


def _english_calendar_clock_patterns():
    # Match the exact spoken calendar/clock grammar produced by the current
    # formatter. Only atom boundaries are guarded; surrounding prose stays free.
    small='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    tens='zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
    def number(n):
        if n<20:return small[n]
        return tens[n//10]+(' '+small[n%10]if n%10 else'')
    def choices(items):return '(?:'+'|'.join(re.escape(x).replace(r'\ ',r'[ \t]+')for x in sorted(set(items),key=len,reverse=True))+')'
    below100=choices(number(n)for n in range(1,100))
    digit=choices(small[1:10])
    below1000=rf'(?:{digit}[ \t]+hundred(?:[ \t]+{below100})?|{below100})'
    year=rf'(?:{digit}[ \t]+thousand(?:[ \t]+{below1000})?|{below1000})'
    ordinal=choices([
        'first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','eleventh','twelfth','thirteenth','fourteenth','fifteenth','sixteenth','seventeenth','eighteenth','nineteenth','twentieth',
        *('twenty '+x for x in('first','second','third','fourth','fifth','sixth','seventh','eighth','ninth')),'thirtieth','thirty first'])
    month=choices('January February March April May June July August September October November December'.split())
    calendar=rf'{month}[ \t]+(?:{ordinal}[ \t]+)?{year}'
    hour=choices(number(n)for n in range(24))
    minute=choices(number(n)for n in range(10,60))
    seconds=choices(number(n)for n in range(60))
    clock=rf'{hour}[ \t]+(?:oh[ \t]+{choices(small[1:10])}|{minute}|o[ \t]+clock)(?:[ \t]+and[ \t]+{seconds}[ \t]+seconds)?(?:[ \t]+[ap][ \t]+m)?'
    return tuple(re.compile(r'(?<![A-Za-z0-9_])'+p+r'(?![A-Za-z0-9_])',re.I)for p in(calendar,clock))

_ENGLISH_CALENDAR_CLOCK_PATTERNS=_english_calendar_clock_patterns()


def protected_spans(text):
    """V6 segmentation guards only: no pronunciation replacement/legacy rules."""
    spans = []
    # These spans protect intact source dates/times only. They do not rewrite
    # text, select a reading, or import any pronunciation lexicon.
    digits = r'[0-9零〇○一二两三四五六七八九十百千]+'
    daypart = r'(?:凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|夜间)'
    clock = (rf'(?:{daypart}[ \t]*)?(?:'
             rf'{digits}(?:点|时)(?:(?:{digits}(?:分)?|半|一刻|三刻))?(?:{digits}秒)?'
             r'|\d{1,2}:\d{2}(?::\d{2})?)')
    calendar = (rf'(?:{digits}年{digits}月(?:{digits}(?:日|号))?'
                rf'|{digits}月{digits}(?:日|号))')
    patterns = [
        rf'{calendar}(?:[ \t]*{clock})?',
        clock,
        r"[A-Za-z][A-Za-z0-9_]*(?:[-./:'’+][A-Za-z0-9_]+)*",
        r'(?<![A-Za-z0-9])(?:[+−-]?\d+(?:[,.]\d+)*)(?:年|月|日|点|分|秒|%|GB|MB|GHz|MHz)?',
        r'(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)',
        r'(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)',
        r'\b[A-Za-z][A-Za-z0-9_./-]*(?:[ \t]+(?:Runtime|Toolkit|XC|Driver|Operator))?[ \t]+(?:version[ \t]+|v)?\d+(?:\.\d+){0,3}\b',
        r'\b\d+(?:\.\d+)*[ \t]+(?:GB|MB|GHz|MHz|GPU|CPU|tokens|ms|milliseconds|seconds)\b',
    ]
    phrases = ['CUDA Runtime', 'CUDA Toolkit', 'ModelHub XC', 'RISE vGPU', 'GPU Operator',
               '第四范式', '小优', '下一步', 'SageAIOS', 'PhancyGroup', 'PhanthyMotus', '4Paradigm', 'xiaoyou01']
    for pattern in patterns:
        spans.extend((m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE))
    for phrase in phrases:
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE))
    # The old ASCII guard treats today.You as one identifier. Release only the
    # explicitly recognized prose boundaries; all other original guards remain.
    released = [m.end() for m in re.finditer(r'\.', text) if _joined_period_boundary(text, m.start())]
    refined = []
    for start, stop in spans:
        cuts = [start, *[end for end in released if start < end < stop], stop]
        refined.extend(zip(cuts, cuts[1:]))
    # Math spans are atomic even if a lower-priority word/whitespace/jieba
    # boundary exists. A too-long complete atom therefore fails closed in
    # safe_split instead of truncating or emitting half an expression.
    return refined + list(_math_expression_spans(text)) + [(m.start(),m.end())for p in _ENGLISH_CALENDAR_CLOCK_PATTERNS for m in p.finditer(text)]


def safe_boundary(text, end):
    return not any(start < end < stop for start, stop in protected_spans(text))


class ProfileLimitError(ValueError):
    """Text is intact, but the verified model profile cannot contain this clause."""


def natural_clauses(text):
    """Natural V6 text slices, protecting numbers/versions and basic initials."""
    boundaries = {m.end() for m in re.finditer(r'[。！？!?；;\n]+|(?<!\d)[，,：:]|[，,：:](?!\d)', text)}
    abbreviations = _ABBREVIATIONS
    for found in re.finditer(r'\.(?=\s+[A-Z]|[A-Z][a-z]+|I(?:\b)|$)', text):
        if found.end() < len(text) and not text[found.end()].isspace() and not _joined_period_boundary(text, found.start()):
            continue
        before = text[:found.start()]
        word = re.search(r'([A-Za-z]+)$', before)
        # These are V6 sentence boundaries, not a pronunciation dictionary.
        if word and (word.group(1).lower() in abbreviations or len(word.group(1)) == 1):
            continue
        if re.search(r'(?:\b[A-Za-z]\.)+[A-Za-z]$', before):
            continue
        boundaries.add(found.end())
    pieces, start = [], 0
    for end in sorted(boundaries):
        if not safe_boundary(text, end):
            continue
        piece = text[start:end]
        if any(c.isalnum() for c in piece):
            pieces.append(piece)
            start = end
    tail = text[start:]
    if any(c.isalnum() for c in tail):
        pieces.append(tail)
    elif pieces:
        pieces[-1] += tail
    elif tail:
        pieces.append(tail)
    if ''.join(pieces) != text:
        raise RuntimeError('Candidate segmentation lost or duplicated source text')
    yield from pieces


def safe_split(text):
    """Split only at source punctuation/word gaps; no phoneme or audio truncation."""
    choices = [m.end() for m in re.finditer(r'[。！？!?；;，,、：:]|(?<!\d)\.(?!\d)|\s+', text)
               if 0 < m.end() < len(text) and text[:m.end()].strip() and text[m.end():].strip()]
    # A comma/colon surrounded by digits is not a source word boundary.
    choices = [x for x in choices if not (x >= 2 and x < len(text)
                and text[x-1] in ',，:：' and text[x-2].isdigit() and text[x].isdigit())]
    choices = [x for x in choices if safe_boundary(text, x)]
    if not choices and re.search(r'[\u3400-\u9fff]', text):
        # V6-native text word boundaries, not a pronunciation override/legacy lexicon.
        # Never cut inside an ASCII identifier, date, version or numeric run.
        try:
            import jieba
            choices = [end for _, _, end in jieba.tokenize(text)
                       if 0 < end < len(text) and text[:end].strip() and text[end:].strip()
                       and not (text[end-1].isascii() and text[end].isascii()
                                and re.match(r"[A-Za-z0-9_./:'’+-]", text[end-1])
                                and re.match(r"[A-Za-z0-9_./:'’+-]", text[end]))]
        except ImportError:
            choices = []
        choices = [x for x in choices if safe_boundary(text, x)]
    if not choices:
        raise ProfileLimitError('V6 clause exceeds its verified profile without a safe text boundary; nothing was truncated')
    cut = min(choices, key=lambda x: abs(x - len(text) / 2))
    return text[:cut], text[cut:]



# Exact first-sentence boundary policy used by the accepted audition.
_FIRST_SENTENCE_ABBREVIATIONS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'st.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'no.', 'fig.', 'inc.', 'ltd.', 'dept.', 'approx.', 'a.m.', 'p.m.'}
_FIRST_SENTENCE_CLOSERS = '\"\'”’》」』)]}）】'


def first_complete_sentence_spans(full):
    """Conservative real punctuation boundary, never token/count based."""
    opens = {'(': ')', '[': ']', '{': '}', '（': '）', '【': '】'}
    stack = []
    for i, char in enumerate(full):
        if char in opens:
            stack.append(opens[char]); continue
        if stack and char == stack[-1]:
            stack.pop(); continue
        if stack or char not in '。！？!?.': continue
        left, right = full[i-1:i], full[i+1:i+2]
        token_start = i
        while token_start and not full[token_start-1].isspace(): token_start -= 1
        token_end = i+1
        while token_end < len(full) and not full[token_end].isspace(): token_end += 1
        token = full[token_start:token_end]
        if char in '.?!' and ('://' in token or token.lower().startswith('www.')): continue
        if char == '.':
            if left == '.' or right == '.' or (left.isalnum() and right.isalnum()): continue
            # Known abbreviations, dotted initials and one-letter personal initials.
            found = re.search(r'[A-Za-z][A-Za-z.]*\.$', full[:i+1])
            word = found.group(0) if found else ''
            if word.lower() in _FIRST_SENTENCE_ABBREVIATIONS or re.fullmatch(r'(?:[A-Za-z]\.){2,}', word) or re.fullmatch(r'[A-Z]\.', word): continue
        elif char == '!' and right == '=': continue
        end = i+1
        while end < len(full) and full[end] in '。！？!?': end += 1
        while end < len(full) and full[end] in _FIRST_SENTENCE_CLOSERS: end += 1
        # A quoted sub-sentence followed immediately by prose is not a safe cut.
        if end > i+1 and end < len(full) and full[end].isalnum() and not full[end].isspace(): continue
        if char == '.' and end < len(full) and not full[end].isspace(): continue
        if full[:end].strip() and full[end:].strip(): return [(0,end),(end,len(full))]
        return [(0,len(full))]
    return [(0,len(full))]


# First-clause planning: original complete-sentence fallback remains unchanged.
# Conservative initial thresholds; quality and latency require separate evidence.
_FC_KEEP_SENTENCE_UNITS = 24
_FC_MIN_FIRST_CLAUSE_UNITS = 12
_FC_ASCII_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|[0-9]+(?:[.,][0-9]+)*")
_FC_MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?'
_FC_WEEKDAY = r'(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)\.?'
_FC_DAY = r'(?:[0-9]{1,2}(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty[ -](?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)|thirtieth|thirty[ -]first)'
_FC_WEEKDAY_END = re.compile(r'\b' + _FC_WEEKDAY + r'\s*$', re.I)
_FC_DATE_NEXT = re.compile(r'\s*(?:' + _FC_MONTH + r'\b|[0-9]{1,2}\s+' + _FC_MONTH + r'\b)', re.I)
_FC_MONTH_DAY_END = re.compile(r'\b' + _FC_MONTH + r'\s+' + _FC_DAY + r'\s*$', re.I)
_FC_MONTH_END = re.compile(r'\b' + _FC_MONTH + r'\s*$', re.I)
_FC_YEAR_NEXT = re.compile(r'\s*(?:[0-9]{4}\b|(?:nineteen|twenty|two\s+thousand)\b)', re.I)
_FC_CJK_NUMBER = r'[0-9零〇○一二两三四五六七八九十百千]+'
_FC_CJK_DATE_COMMAS = (
    (re.compile(_FC_CJK_NUMBER + r'年\s*$'), re.compile(r'\s*' + _FC_CJK_NUMBER + r'月')),
    (re.compile(_FC_CJK_NUMBER + r'月\s*$'), re.compile(r'\s*' + _FC_CJK_NUMBER + r'(?:日|号)')),
    (re.compile(_FC_CJK_NUMBER + r'(?:日|号)\s*$'),
     re.compile(r'\s*(?:(?:凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|夜间)|' + _FC_CJK_NUMBER + r'(?:点|时)|[0-9]{1,2}:[0-9]{2})')),
)
_FC_MATH_OPERATOR = r'(?:==|!=|>=|<=|=|≠|≥|≤|等于|大于|小于)'
_FC_ASSIGNMENT_END = re.compile(r'[A-Za-z][A-Za-z0-9_]*\s*' + _FC_MATH_OPERATOR + r'[^,，;；。]*$')
_FC_ASSIGNMENT_NEXT = re.compile(r'\s*[A-Za-z][A-Za-z0-9_]*\s*' + _FC_MATH_OPERATOR)
_FC_COMPANY_SUFFIX = re.compile(r'\s*(?:Inc|Ltd|Corp|Co|LLC|LLP)\.?\b', re.I)
_FC_IDENTIFIER_CHARS = frozenset('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./:@?=+#%&~\\,-')
_FC_OPEN = {'(': ')', '[': ']', '{': '}', '（': '）', '【': '】', '《': '》', '〈': '〉'}
_FC_QUOTES = {'"': '"', "'": "'", '“': '”', '‘': '’', '「': '」', '『': '』'}
_FC_CLOSE = set(_FC_OPEN.values())
_FC_QUOTE_CLOSE = set(_FC_QUOTES.values()) - {'"', "'"}


def _fc_spoken_units(text):
    """Count Han ideographs + twice ASCII words/numeric tokens; punctuation=0."""
    cjk = sum(0x3400 <= ord(c) <= 0x4DBF or 0x4E00 <= ord(c) <= 0x9FFF
              or 0xF900 <= ord(c) <= 0xFAFF or 0x20000 <= ord(c) <= 0x323AF
              for c in text)
    return cjk + 2 * len(_FC_ASCII_TOKEN.findall(text))


def _fc_escaped(text, index):
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == '\\':
        backslashes += 1
        index -= 1
    return bool(backslashes % 2)


def _fc_apostrophe(text, index):
    before, after = text[index - 1:index], text[index + 1:index + 2]
    # Contractions and English possessives are not opening quotation marks.
    return bool(before and before.isascii() and before.isalnum()
                and (not after or after.isspace() or after in ',，.;:!?' or after.isascii() and after.isalnum()))


def _fc_top_level_commas(text, delimiters=',，'):
    """Return eligible lexical positions, or None for malformed nesting/quotes."""
    stack = []
    commas = []
    for index, char in enumerate(text):
        if stack and stack[-1][1]:
            if char == stack[-1][0] and not _fc_escaped(text, index):
                stack.pop()
            continue
        if char in _FC_QUOTES:
            if char in "'‘’" and _fc_apostrophe(text, index):
                continue
            stack.append((_FC_QUOTES[char], True))
        elif char in _FC_QUOTE_CLOSE:
            if char == '’' and _fc_apostrophe(text, index):
                continue
            return None
        elif char in _FC_OPEN:
            stack.append((_FC_OPEN[char], False))
        elif char in _FC_CLOSE:
            if not stack or stack[-1][0] != char:
                return None
            stack.pop()
        elif char in delimiters and not stack:
            commas.append(index)
    return None if stack else commas


def _fc_unsafe_comma(full, index):
    before, after = full[:index], full[index + 1:]
    left, right = before.rstrip(), after.lstrip()
    if not left or not right:
        return True
    # A comma between numeric components is not a semantic clause boundary.
    if left[-1].isascii() and left[-1].isdigit() and right[0].isascii() and right[0].isdigit():
        return True
    if _FC_WEEKDAY_END.search(before) and _FC_DATE_NEXT.match(after):
        return True
    if (_FC_MONTH_DAY_END.search(before) or _FC_MONTH_END.search(before)) and _FC_YEAR_NEXT.match(after):
        return True
    if any(left_pattern.search(before) and right_pattern.match(after) for left_pattern, right_pattern in _FC_CJK_DATE_COMMAS):
        return True
    if _FC_ASSIGNMENT_END.search(before) and _FC_ASSIGNMENT_NEXT.match(after):
        return True
    if _FC_COMPANY_SUFFIX.match(after):
        return True
    word = re.search(r'([A-Za-z][A-Za-z.]*)\s*$', before)
    if word:
        token = word.group(1).lower()
        abbreviations = {str(x).lower().rstrip('.') for x in _ABBREVIATIONS}
        abbreviations.update(str(x).lower().rstrip('.') for x in _FIRST_SENTENCE_ABBREVIATIONS)
        if token.rstrip('.') in abbreviations or re.fullmatch(r'(?:[a-z]\.)+', token):
            return True
    # Protect embedded comma in a contiguous filename, URL or identifier. A
    # punctuation comma after a complete identifier is not inside that atom.
    start, stop = index, index + 1
    while start and full[start - 1] in _FC_IDENTIFIER_CHARS:
        start -= 1
    while stop < len(full) and full[stop] in _FC_IDENTIFIER_CHARS:
        stop += 1
    if start < index and stop > index + 1:
        token = full[start:stop]
        if any(char in token for char in '._/:@?=+#%&~\\'):
            return True
    return not safe_boundary(full, index) or not safe_boundary(full, index + 1)


def _fc_extra_boundary_safe(full, index, earlier):
    """Guard list items and colon atoms without changing comma decisions."""
    mark = full[index]
    if mark == '、':
        # The immediately preceding top-level delimiter must be another list
        # separator, with two complete nonempty items before this candidate.
        if not earlier or full[earlier[-1]] != '、':
            return False
        previous = earlier[-1]
        start = earlier[-2] + 1 if len(earlier) > 1 else 0
        return bool(_fc_spoken_units(full[start:previous])
                    and _fc_spoken_units(full[previous + 1:index]))
    if mark in ':：':
        left, right = full[:index].rstrip(), full[index + 1:].lstrip()
        # Ratios and type/function mappings with spaces remain one atom too.
        if re.search(r'(?<![A-Za-z0-9_])[A-Za-z]$', left) and re.match(r'[A-Za-z](?![A-Za-z0-9_])', right):
            return False
        if left and right and left[-1] in '零〇○一二两三四五六七八九十百千' and right[0] in '零〇○一二两三四五六七八九十百千':
            return False
    return True


def first_clause_spans(full):
    """Plan one safe punctuation cut; otherwise preserve baseline spans.

    This neither normalizes nor pronounces text. Callers must retain their
    original whole-text frontend context; the plan provides source offsets only.
    """
    if not isinstance(full, str):
        raise TypeError('full must be a string')
    original = first_complete_sentence_spans(full)
    if not original:
        raise ValueError('Baseline returned no sentence span')
    if any(not isinstance(span, (tuple, list)) or len(span) != 2 for span in original):
        raise ValueError('Invalid baseline sentence spans')
    if original[0][0] != 0 or original[-1][1] != len(full) or any(
            left[1] != right[0] for left, right in zip(original, original[1:])):
        raise ValueError('Baseline sentence spans do not preserve full text')
    first_end = original[0][1]
    first = full[:first_end]
    if _fc_spoken_units(first) <= _FC_KEEP_SENTENCE_UNITS:
        return original
    commas = _fc_top_level_commas(first, ',，:：、')
    if commas is None:
        return original
    for position, index in enumerate(commas):
        if not _fc_extra_boundary_safe(full, index, commas[:position]):
            continue
        end = index + 1
        if _fc_spoken_units(full[:end]) < _FC_MIN_FIRST_CLAUSE_UNITS:
            continue
        if not _fc_spoken_units(full[end:first_end]) or _fc_unsafe_comma(full, index):
            continue
        return [(0, end), (end, len(full))]
    return original
