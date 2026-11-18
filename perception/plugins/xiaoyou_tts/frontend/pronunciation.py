"""Instance-local Chinese readings with native-word and exact-offset proofs.

The lexical rules run before the full-request grammar rules. Both original
native replays and conflict checks remain; no global dictionary is modified.
"""
from __future__ import annotations
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from hashlib import sha256

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

def _find_existing_relisten_interventions(text: str) -> list[dict]:
    """Return exact code-point offsets in the supplied complete Chinese span.

    Offsets refer to the original two-character word, not a fuzzy match or a
    phone-string search.  Audio semantics across an English split, arbitrary
    intervening words and other polyphones are outside this rule's scope.
    """
    matches = list(_RELISTEN.finditer(text))
    if not matches:
        return []
    from misaki import zh
    eligible_words = set()
    for segment_match in _SEGMENTS.finditer(text):
        segment = segment_match.group()
        if not '\u4E00' <= segment[0] <= '\u9FFF':
            continue
        words = zh.jieba.lcut(segment, cut_all=False)
        if ''.join(words) != segment:
            raise RuntimeError('Cannot establish native word boundaries')
        cursor = segment_match.start()
        for word in words:
            if word == '重听':
                eligible_words.add((cursor, cursor + len(word)))
            cursor += len(word)
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
    tokens = []
    for segment_index, segment_match in enumerate(_SEGMENTS.finditer(text)):
        segment = segment_match.group()
        if not '\u4E00' <= segment[0] <= '\u9FFF':
            continue
        words = zh.jieba.lcut(segment, cut_all=False)
        if ''.join(words) != segment:
            raise RuntimeError('Cannot establish full native token offsets')
        cursor = segment_match.start()
        for word in words:
            tokens.append({'surface': word, 'start': cursor, 'end': cursor + len(word),
                           'segment': segment_index})
            cursor += len(word)
    return tokens

def find_context_interventions(text: str) -> list[dict]:
    if not re.search(r'重听|重来|重试|数清|长椅', text):
        return []
    interventions = _find_existing_relisten_interventions(text)
    tokens = _native_tokens(text)

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
        zh = self._native
        rendered = []
        applied = set()
        authorized = {item['start']: item['pinyin_override_at_start']
                      for item in find_context_interventions(prepared)} if fixes else {}
        if any(authorized.get(position) != pinyin for position, pinyin in fixes.items()):
            raise RuntimeError('An override is not authorized at that exact native context offset')
        for segment_match in _SEGMENTS.finditer(prepared):
            segment = segment_match.group()
            if not '\u4E00' <= segment[0] <= '\u9FFF':
                rendered.append(segment)
                continue
            # Same tokenizer, cut mode, word order and inter-word spaces as
            # Misaki 0.9.4 ZHG2P.legacy_call.  No add_word/load_dict calls.
            words = zh.jieba.lcut(segment, cut_all=False)
            if ''.join(words) != segment:
                raise RuntimeError('Native jieba offsets do not reproduce the full Chinese segment')
            word_outputs = []
            cursor = segment_match.start()
            for word in words:
                positions = [p for p in fixes if cursor <= p < cursor + len(word)]
                if not positions:
                    word_outputs.append(zh.ZHG2P.word2ipa(word))
                else:
                    if any(not cursor <= position < cursor + len(word) for position in positions):
                        raise RuntimeError('The override is outside the complete native token')
                    # Identical per-word pinyin API/arguments used by
                    # ZHG2P.word2ipa; replace only the context-bound syllable.
                    pinyins = zh.lazy_pinyin(word, style=zh.Style.TONE3,
                                            neutral_tone_with_five=True)
                    if len(pinyins) != len(word):
                        raise RuntimeError('Cannot establish a one-syllable-per-character word offset')
                    for position in positions:
                        if prepared[position] not in ('重', '数', '长'):
                            raise RuntimeError('Context override target character changed')
                        pinyins[position - cursor] = fixes[position]
                        applied.add(position)
                    word_outputs.append(''.join(zh.ZHG2P.py2ipa(py) for py in pinyins))
                cursor += len(word)
            rendered.append(' '.join(word_outputs))
        if applied != set(fixes):
            raise RuntimeError('Not all word-scoped overrides were applied')
        return ''.join(rendered).replace(chr(815), '')

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

_BASIS=re.compile(r'以[\u4E00-\u9FFF]{1,48}?为基础')

_VERIFIED_SPLIT_BASIS=re.compile(r'算力为基础')

_FULL_BASIS=re.compile(r'以(?P<object>[\u4E00-\u9FFFA-Za-z0-9_ \t\-.]{1,64}?)为基础')

_GROWTH_SUFFIX = re.compile(r'(?:见识|本事|个儿|个子|个头|得|高|胖|壮|大|出|成)')


def _tower_is_object_of_hiding(text, start, end):
    """Keep cáng in the narrow verb/object pattern “他想藏塔。”"""
    following = text[end:end + 1]
    if following and following not in '，。！？；,.!?;':
        return False
    return text[max(0, start - 2):start].endswith(('想', '打算', '试图'))

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

_CN_VALUE = r'(?:正负|正|负)?[零〇一二三四五六七八九十百千万亿兆两幺]+'

_ARABIC_VALUE = r'[+−-]?[0-9]+(?:\.[0-9]+)?'

_EN_NUMBER = (r'(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
              r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|'
              r'forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b')

_EN_VALUE = r'(?:(?:minus|negative|plus|positive)[ \t]+)?' + _EN_NUMBER

_NUMERIC_VALUE = r'(?:' + _CN_VALUE + r'|' + _ARABIC_VALUE + r'|' + _EN_VALUE + r')'

_VALUE_COPULA = re.compile(r'(?:能量|功率|电量|容量|体积|面积|电压|电流|带宽|速率|质量|温度范围|范围|延迟|速度|时长|数量|温度|值|结果|版本号|版本|编号|代码|端口号|端口)'
                           r'(?:分别)?(?P<target>为)(?=[ \t]*' + _NUMERIC_VALUE + r')', re.I)

_VERSION_VALUE = r'v[0-9]+(?:\.[0-9]+){0,3}(?![A-Za-z0-9_+\-/]|\.[A-Za-z0-9_.])'

_VERSION_COPULA = re.compile(r'(?:版本号|版本)(?P<target>为)(?=[ \t]*' + _VERSION_VALUE + r')', re.I)

def _syntax_targets(text,words):
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
    for pattern in (_VALUE_COPULA, _VERSION_COPULA):
        for match in pattern.finditer(text):
            result.append({'position':match.start('target'),'character':'为','pinyin':'wei2',
                           'context':match.group(),'rule':'numerical_value_copula_named_business_field'})
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
        self.metadata={'kind':'isolated_polyphone_candidate_NOT_active_profile','source_path':str(Path(__file__).resolve()),'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'scope':['complete_native_word_华为_second_syllable_wei2','same_Chinese_span_以_1to48ChineseChars_为基础_target_wei2','verified_English_split_suffix_算力为基础_target_wei2','native_word_bounded_subject_得_action_dei3','native_word_bounded_verb_得_degree_quality_de5','native_word_bounded_manner_地_action_de5','numerical_value_copula_named_business_fields_wei2'],'bound_full_context_scope':'explicit_以…为基础_and_verified_grammar_phrases_and_numerical_value_copula_with_exact_offsets','global_dictionary_modified':False,'other_non_targets_delegate':True,'quality_accepted':False}

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
        targets=[]
        for match in _FULL_BASIS.finditer(normalized):
            if not re.search(r'[\u4E00-\u9FFFA-Za-z0-9]',match.group('object')):continue
            # Only numeric-internal version dots; sentence periods are not object characters.
            if '.' in re.sub(r'(?<=[0-9])\.(?=[0-9])','',match.group('object')):continue
            position=match.end()-3
            if normalized[position]!='为':raise RuntimeError('Full-context target offset mismatch')
            targets.append({'global_position':position,'context':match.group(),'rule':'bound_full_normalized_以…为基础','pinyin':'wei2'})
        for item in _syntax_targets(normalized,words):
            targets.append({**item,'global_position':item['position']})
        for item in self._phrase_targets(normalized,words,protected_spans):
            targets.append({**item,'global_position':item['position']})
        token=self._context.set({'text':normalized,'targets':targets})
        try:yield self
        finally:self._context.reset(token)

    def _verified_prepared_position(self,span,raw_position,prepared,character='为'):
        """Prove one exact authorized character offset across native preparation."""
        if not 0<=raw_position<len(span) or span[raw_position]!=character:
            raise RuntimeError('The bound source character is not '+character)
        zh=self._native
        left=zh.cn2an.transform(span[:raw_position],'an2cn')
        right=zh.cn2an.transform(span[raw_position+1:],'an2cn')
        whole=zh.cn2an.transform(span,'an2cn')
        if left+character+right!=whole:
            raise RuntimeError('cn2an source offset cannot be proven across this split')
        marker='\ue000POLYPHONE_TARGET\ue001'
        if marker in span or marker in whole:raise RuntimeError('Offset marker collision')
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
        zh=self._native;prepared=zh.ZHG2P.map_punctuation(zh.cn2an.transform(span,'an2cn'));targets=[]
        for item in bound['targets']:
            source=item['global_position']
            if start<=source<end:
                character=item.get('character','为')
                position=self._verified_prepared_position(span,source-start,prepared,character)
                targets.append({**item,'position':position,'character':character,'source_span':[start,end],
                                'source_span_position':source-start,'offset_proven':True})
        return self._call(span,en_callable=en_callable,bound_targets=targets,expected_prepared=prepared)

    def _words(self,prepared):
        words=[]
        for segment in _SEGMENTS.finditer(prepared):
            if not '\u4E00'<=segment.group()[0]<='\u9FFF':continue
            tokens=self._native.jieba.lcut(segment.group(),cut_all=False)
            if ''.join(tokens)!=segment.group():raise RuntimeError('Native token offsets do not reproduce the Chinese span')
            cursor=segment.start()
            for word in tokens:words.append((word,cursor,cursor+len(word)));cursor+=len(word)
        return words

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
                if phrase == '藏塔' and _tower_is_object_of_hiding(text, start, cursor):
                    continue
                if phrase in ('很长', '更长', '太长', '多长') and _GROWTH_SUFFIX.match(text, cursor):
                    continue
                # Han-bearing filenames/paths may lie outside the ASCII-only
                # literal reader. Their separators still disqualify a name;
                # a sentence-final period remains ordinary punctuation.
                before=text[start-1:start]if start else'';after=text[cursor:cursor+1]
                if before in ('_','/','\\','@','`') or after in ('_','/','\\','@','`'):continue
                if after=='.' and re.match(r'[A-Za-z0-9_]',text[cursor+1:cursor+2]):continue
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
        for word,start,end in words:
            if word=='华为':add(start+1,word,'complete_native_word_华为')
        for match in _BASIS.finditer(prepared):add(match.end()-3,match.group(),'explicit_以…为基础_same_Chinese_span')
        # V6 splits an English GPU span before invoking its Chinese G2P.
        # Only this already verified exact Chinese suffix is additionally allowed.
        for match in _VERIFIED_SPLIT_BASIS.finditer(prepared):add(match.start()+2,match.group(),'verified_exact_算力为基础_suffix')
        for item in _syntax_targets(prepared,words):add(**item)
        return [targets[k]for k in sorted(targets)]

    def _existing_fixes(self,prepared):
        renderer=getattr(self.upstream,'_render',None)
        if renderer is None:return {}
        function=getattr(renderer,'__func__',None);finder=getattr(function,'__globals__',{}).get('find_context_interventions')
        if not callable(finder):raise RuntimeError('Unknown upstream context renderer; refuse to bypass existing rules')
        return {item['start']:item['pinyin_override_at_start']for item in finder(prepared)}

    def _render(self,prepared,fixes):
        zh=self._native;rendered=[];applied=set()
        for segment in _SEGMENTS.finditer(prepared):
            value=segment.group()
            if not '\u4E00'<=value[0]<='\u9FFF':rendered.append(value);continue
            words=zh.jieba.lcut(value,cut_all=False)
            if ''.join(words)!=value:raise RuntimeError('Cannot reproduce native word boundaries')
            outputs=[];cursor=segment.start()
            for word in words:
                positions=[p for p in fixes if cursor<=p<cursor+len(word)]
                if not positions:outputs.append(zh.ZHG2P.word2ipa(word))
                else:
                    pinyins=zh.lazy_pinyin(word,style=zh.Style.TONE3,neutral_tone_with_five=True)
                    if len(pinyins)!=len(word):raise RuntimeError('Cannot establish exact one-syllable character offsets')
                    for position in positions:pinyins[position-cursor]=fixes[position];applied.add(position)
                    outputs.append(''.join(zh.ZHG2P.py2ipa(py)for py in pinyins))
                cursor+=len(word)
            rendered.append(' '.join(outputs))
        if applied!=set(fixes):raise RuntimeError('Not all exact syllable offsets were applied')
        return ''.join(rendered).replace(chr(815),'')

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
