"""Xiaoyou text normalization, including typed fields, formulas, and boundary planning."""

from collections import namedtuple
from datetime import date
from decimal import Decimal
import ipaddress
import re
import unicodedata

from plugins.vits2_tts_trt.frontend.math_units import _OPERATORS as SHARED_OPERATORS, _UNITS as SHARED_UNITS
from plugins.vits2_tts_trt.frontend.number_context import find_calendar_spans


class NormalizationError(ValueError):
    pass

_DIGITS_ZH = '零一二三四五六七八九'
_SMALL_EN = ('zero one two three four five six seven eight nine ten eleven twelve '
             'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
_TENS_EN = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
_MONTHS = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
           'September', 'October', 'November', 'December']
_DAY_ORDINAL = {1:'first',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth',7:'seventh',
 8:'eighth',9:'ninth',10:'tenth',11:'eleventh',12:'twelfth',13:'thirteenth',14:'fourteenth',
 15:'fifteenth',16:'sixteenth',17:'seventeenth',18:'eighteenth',19:'nineteenth',20:'twentieth',
 21:'twenty first',22:'twenty second',23:'twenty third',24:'twenty fourth',25:'twenty fifth',
 26:'twenty sixth',27:'twenty seventh',28:'twenty eighth',29:'twenty ninth',30:'thirtieth',31:'thirty first'}
_PLAIN_NUMBER = r'[+\-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)'
_PLAIN_LEFT = r'(?<![A-Za-z0-9_./+=\-])'
_PLAIN_RIGHT = r'(?![A-Za-z0-9_]|\.\d|,\d)'
_HTTP = r'(?<![A-Za-z0-9_])HTTP(?:/(?:1\.0|1\.1|2|3))?'
_HTTP_LABELS = (
    _HTTP + r'[ \t]*(?:(?:接口|请求|响应)[ \t]*)?(?:返回|报错|提示|显示)(?:[ \t]*(?:状态码|错误码|错误代码))?',
    _HTTP + r'[ \t]+(?:status(?:[ \t]+code)?|error[ \t]+code|response[ \t]+code)(?:[ \t]+(?:is|was|of))?',
    _HTTP + r'[ \t]+(?:returns?|returned|reports?|reported)(?:[ \t]+(?:status(?:[ \t]+code)?|error[ \t]+code))?',
    _HTTP + r'[ \t]*(?:状态码|状态|响应码|错误码|错误代码|错误|响应)',
    r'(?<![A-Za-z0-9_])(?:status[ \t]+code|error[ \t]+code)(?:[ \t]+(?:is|was|of))?',
    r'(?:状态码|响应码|错误码|错误代码)',
    _HTTP,
)
_HTTP_QUANTITY = (
    r'(?:个|次|元|条|份|项|页(?!面)|年|月|日|小时|分钟|秒|毫秒|千克|克|米|'
    r'字节|摄氏度|毫升|升|items?\b|requests?\b|records?\b|bytes?\b|rows?\b|times?\b|'
    r'dollars?\b|yuan\b|euros?\b)'
)
_HTTP_CODE = re.compile(
    r'(?P<label>(?:' + '|'.join(_HTTP_LABELS) + r')[ \t]*(?:(?:[:：#]|为|是|等于)[ \t]*)?)'
    r'(?P<code>[1-5][0-9]{2})(?![A-Za-z0-9_]|[./][0-9]|[/\\]|[ \t]*' + _HTTP_QUANTITY + r')',
    re.I,
)


def _zh_group(n):
    out = []; zero = False
    for power, suffix in ((1000,'千'),(100,'百'),(10,'十'),(1,'')):
        digit, n = divmod(n, power)
        if digit:
            if zero and out: out.append('零')
            if not (digit == 1 and power == 10 and not out): out.append(_DIGITS_ZH[digit])
            out.append(suffix); zero = False
        elif out and n: zero = True
    return ''.join(out)


def _zh_int(n):
    if n == 0: return '零'
    if n < 0: return '负' + _zh_int(-n)
    if n >= 10**16: raise NormalizationError('Typed numeric value exceeds supported16 digits')
    parts = []; previous = False; zero = False
    for base, suffix in ((10**12,'兆'),(10**8,'亿'),(10**4,'万'),(1,'')):
        group, n = divmod(n, base)
        if group:
            if previous and (zero or group < 1000): parts.append('零')
            chunk = _zh_group(group)
            if previous and 10 <= group < 20: chunk = '一' + chunk
            parts.append(chunk + suffix); previous = True; zero = False
        elif previous and n: zero = True
    return ''.join(parts)


def _en_int(n):
    if n < 0: return 'minus ' + _en_int(-n)
    if n < 20: return _SMALL_EN[n]
    if n < 100: return _TENS_EN[n//10] + (' ' + _SMALL_EN[n%10] if n%10 else '')
    if n < 1000: return _SMALL_EN[n//100] + ' hundred' + (' ' + _en_int(n%100) if n%100 else '')
    for base, suffix in ((10**15,'quadrillion'),(10**12,'trillion'),(10**9,'billion'),(10**6,'million'),(1000,'thousand')):
        if n >= base:
            if n >= 10**18: raise NormalizationError('Typed numeric value exceeds supported18 digits')
            return _en_int(n//base) + ' ' + suffix + (' ' + _en_int(n%base) if n%base else '')
    raise NormalizationError('Unsupported number')


def _parse_number(value):
    value = str(value).strip().replace('−','-')
    sign = ''
    if value[:1] in ('+','-'): sign, value = value[0], value[1:]
    if not re.fullmatch(r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)', value):
        raise NormalizationError('Invalid typed numeric literal: ' + value)
    integer, dot, fraction = value.partition('.')
    if ',' in integer and not re.fullmatch(r'\d{1,3}(?:,\d{3})+', integer):
        raise NormalizationError('Invalid thousands grouping: ' + value)
    integer = integer.replace(',','') or '0'
    if len(integer) > 18: raise NormalizationError('Typed numeric value too long')
    return sign, integer, fraction if dot else None


def speak_number(value, language='zh', digit_by_digit=False):
    """Read a parsed number only; callers must first identify its typed context."""
    language = 'en' if language == 'en' else 'zh'
    sign, integer, fraction = _parse_number(value)
    words = _SMALL_EN if language == 'en' else _DIGITS_ZH
    if digit_by_digit:
        whole = (' ' if language == 'en' else '').join(words[int(ch)] for ch in integer)
    else: whole = _en_int(int(integer)) if language == 'en' else _zh_int(int(integer))
    if fraction is not None:
        tail = (' ' if language == 'en' else '').join(words[int(ch)] for ch in fraction)
        whole += (' point ' if language == 'en' else '点') + tail
    if sign: whole = ({'en':{'+':'plus ','-':'minus '},'zh':{'+':'正','-':'负'}}[language][sign]) + whole
    return whole


def context_language(text, start, end, language='auto'):
    """Explicit language wins; auto/mixed chooses nearby clause, then whole text."""
    if language in ('zh','en'): return language
    if language not in ('auto','mixed'): raise NormalizationError('Unsupported language: ' + str(language))
    left = re.split(r'[。！？!?；;，,\n]', text[max(0,start-100):start])[-1]
    right = re.split(r'[。！？!?；;，,\n]', text[end:end+100])[0]
    for nearby in (left,right):
        if re.search(r'[\u3400-\u9fff]', nearby): return 'zh'
        if re.search(r'[A-Za-z]', nearby): return 'en'
    return 'zh' if re.search(r'[\u3400-\u9fff]',text) or not re.search('[A-Za-z]',text) else 'en'


def _token(index):
    chars = ''
    while True:
        index, digit = divmod(index,26); chars = chr(65+digit) + chars
        if not index: return '\ue000GNR' + chars + '\ue001'
        index -= 1


def protect_identifiers(text):
    """Mask own nonlinguistic spans; existing PUA placeholders remain untouched."""
    spans = []
    patterns = [
        # Protect complete identifier=quoted-literal spans, not natural quotations.
        '(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_][A-Za-z0-9_]*)*\\s*=\\s*(?:"(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\')',
        r'`[^`\n]*`',
        r'\b(?:https?://|www\.)[^\s<>，。；！？]+',
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        r'(?<![A-Za-z0-9_])(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?',
        r'(?<![A-Za-z0-9_])(?:[A-Za-z]:\\|\.{1,2}/|/)[A-Za-z0-9_./\\\-]+',
        r'\b(?:version|ver|v)\s*\d+(?:\.\d+){1,5}\b',
        r'\b(?:JetPack|TensorRT|CUDA(?:\s+Runtime)?|Python|Kokoro|HTTP)\s*\d+(?:\.\d+){1,5}\b',
        r'\b[A-Za-z_][A-Za-z0-9_.\-]*_[A-Za-z0-9_.\-]+\b',
        r'\b(?:xiaoyou\d+|C\+\+|C#)\b',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern,text,flags=re.I): spans.append(match.span())
    # Alphanumeric version/model identifiers, excluding adjacent currency codes.
    for match in re.finditer(r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_.\-]*[A-Za-z0-9_]',text):
        word = match.group()
        if any(ch.isdigit() for ch in word) and not re.match(r'(?i)^(?:CNY|RMB|USD|EUR)\d|^UTC-\d',word) and not(word=='cm2' and re.search(r'\d\s*$',text[:match.start()])):
            spans.append(match.span())
    # Preserve valid bare IPv6 addresses; ordinary clock triples are not IPv6.
    for match in re.finditer(r'(?<![\w])(?:[0-9A-Fa-f]*:){2,}[0-9A-Fa-f:.]+',text):
        try: ipaddress.ip_address(match.group())
        except ValueError: continue
        spans.append(match.span())
    spans.sort(); merged=[]
    for start,end in spans:
        if merged and start <= merged[-1][1]: merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else: merged.append((start,end))
    output=[];mapping={};previous=0;counter=0
    for start,end in merged:
        output.append(text[previous:start]);token=_token(counter)
        while token in text or token in mapping: counter+=1;token=_token(counter)
        mapping[token]=text[start:end];output.append(token);previous=end;counter+=1
    output.append(text[previous:])
    return ''.join(output),mapping


def restore_identifiers(text,mapping):
    for token,original in mapping.items(): text=text.replace(token,original)
    return text


def _read_date(year,month,day,language):
    year,month,day = int(year),int(month),int(day)
    try: date(year,month,day)
    except ValueError as error: raise NormalizationError('Invalid calendar date') from error
    if language == 'en': return _MONTHS[month] + ' ' + _DAY_ORDINAL[day] + ' ' + _en_int(year)
    return speak_number(str(year),'zh',True) + '年' + _zh_int(month) + '月' + _zh_int(day) + '日'


def _math_adjacent(text,start,end):
    before=text[:start].rstrip()[-1:] or ' '
    after=text[end:].lstrip()[:1] or ' '
    return before in '=+×÷*/^≥≤≠<>|' or after in '=+×÷*/^≥≤≠<>|'


def _read_money(value,code,language):
    code={'￥':'CNY','¥':'CNY','人民币':'CNY','RMB':'CNY','$':'USD','美元':'USD','€':'EUR','欧元':'EUR'}.get(code,code.upper())
    sign,integer,fraction=_parse_number(value)
    prefix=({'en':{'+':'plus ','-':'minus '},'zh':{'+':'正','-':'负'}}[language].get(sign,''))
    number=int(integer)
    if language=='en':
        major={'CNY':'Chinese yuan','USD':'US dollar','EUR':'euro'}[code]
        if code!='CNY' and number!=1: major+='s'
        if fraction is None or set(fraction)<={'0'}: return prefix+_en_int(number)+' '+major
        if len(fraction)<=2:
            cents=int(fraction.ljust(2,'0'));minor='cent'if cents==1 else'cents'
            if code=='CNY':minor='fen'
            return prefix+_en_int(number)+' '+major+' and '+_en_int(cents)+' '+minor
        return speak_number(value,'en')+' '+major
    major={'CNY':'元','USD':'美元','EUR':'欧元'}[code]
    leader='人民币'if code=='CNY'else''
    if fraction is None or set(fraction)<={'0'}:return prefix+leader+_zh_int(number)+major
    if len(fraction)>2:return leader+speak_number(value,'zh')+major
    padded=fraction.ljust(2,'0');cents=int(padded)
    if code=='CNY':
        jiao,fen=divmod(cents,10)
        tail=(_zh_int(jiao)+'角'if jiao else'零')+(_zh_int(fen)+'分'if fen else'')
    else:tail=_zh_int(cents)+('美分'if code=='USD'else'欧分')
    return prefix+leader+_zh_int(number)+major+tail


_UNITS={
 **{name: SHARED_UNITS[name] for name in ('kg', 'm', 'cm', 'm/s', 'A', 'µA', 'μA', 'mV', 'kV', 'cm²', 'm²', '°C')},

 'ms':('毫秒','millisecond','milliseconds'),

 'Hz':('赫兹','hertz','hertz'), 'GHz':('吉赫兹','gigahertz','gigahertz'),
 'MHz':('兆赫兹','megahertz','megahertz'), 'kHz':('千赫兹','kilohertz','kilohertz'),
 'MB':('兆字节','megabyte','megabytes'), 'mA':('毫安','milliamp','milliamps'),
 'mAh':('毫安时','milliampere hour','milliampere hours'),
 'V':('伏','volt','volts'),



}


def _normalize_protected_http_codes(protected, mapping, language):
    # Unmask a complete HTTP404 token while retaining URL, path and literal masks.
    for token, original in mapping.items():
        if re.fullmatch(r'HTTP[1-5][0-9]{2}', original, re.I):
            protected = protected.replace(token, original)

    def read(match):
        lang = context_language(match.string, match.start(), match.end(), language)
        label, code = match['label'], match['code']
        if lang == 'en':
            if label and label[-1].isalnum():
                label += ' '
            return label + ' '.join(_SMALL_EN[int(digit)] for digit in code)
        return label + ''.join(_DIGITS_ZH[int(digit)] for digit in code)

    return _HTTP_CODE.sub(read, protected)


def normalize_general(text,language='auto'):
    if not isinstance(text,str):raise TypeError('text must be str')
    if language not in ('auto','mixed','zh','en'):raise NormalizationError('Unsupported language')
    protected,mapping=protect_identifiers(text)
    protected=_normalize_protected_http_codes(protected,mapping,language)
    def lang(match):return context_language(match.string,match.start(),match.end(),language)
    # Full calendar/date spans precede any arithmetic or generic NFKC handling.
    iso=re.compile(_PLAIN_LEFT+r'(?P<y>\d{4})(?P<sep>[-/])(?P<m>\d{1,2})(?P=sep)(?P<d>\d{1,2})(?![A-Za-z0-9_/\-])')
    protected=iso.sub(lambda m:_read_date(m['y'],m['m'],m['d'],lang(m)),protected)
    chinese=re.compile(_PLAIN_LEFT+r'(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*[日号]')
    protected=chinese.sub(lambda m:_read_date(m['y'],m['m'],m['d'],lang(m)),protected)
    # UTC offsets are typed timezones, not arithmetic or ordinary clocks.
    timezone=re.compile(r'(?<![A-Za-z0-9_])UTC\s*(?P<sign>[+\-])\s*(?P<h>\d{1,2})(?::(?P<m>\d{2}))?(?![A-Za-z0-9_:])',re.I)
    def timezone_read(m):
        h,minute=int(m['h']),int(m['m']or'0')
        if h>14 or minute>59 or(h==14 and minute):raise NormalizationError('Invalid UTC offset')
        l=lang(m)
        if l=='en':return 'U T C '+('plus 'if m['sign']=='+'else'minus ')+_en_int(h)+' hours'+(' '+_en_int(minute)+' minutes'if minute else'')
        return 'U T C '+('加'if m['sign']=='+'else'减')+_zh_int(h)+'小时'+(_zh_int(minute)+'分'if minute else'')
    protected=timezone.sub(timezone_read,protected)
    clock=re.compile(_PLAIN_LEFT+r'(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?(?:\s*(?P<ap>[ap]\.?m\.?))?(?![A-Za-z0-9_:])',re.I)
    def clock_read(m):
        # The same digits may denote ratios or scores. Only an adjacent explicit
        # label blocks clock reading; a sentence merely mentioning a score and
        # later recording its time remains an ordinary clock.
        before=m.string[max(0,m.start()-70):m.start()]
        after=m.string[m.end():m.end()+35]
        ratio_before=re.search(r'(?:比例|配比|比值|比分|得分|ratio|score)\s*(?:为|是|等于|is|of|=|:)?\s*$',before,re.I)
        ratio_after=re.match(r'\s*(?:(?:的|这个|该)\s*(?:比例|配比|比值|比分)|(?:ratio|score)\b)',after,re.I)
        if ratio_before or ratio_after:return m.group()
        h,minute,second=int(m['h']),int(m['m']),int(m['s']or'0');ap=m['ap']
        if minute>59 or second>59 or h>23 or(ap and not 1<=h<=12):raise NormalizationError('Invalid clock time')
        l=lang(m);prefix=m.string[max(0,m.start()-45):m.start()]
        duration=bool(re.search(r'(?:倒计时|计时长度|持续时间|countdown|timer|duration)[^。！？!?；;\n]{0,24}$',prefix,re.I))
        if duration:
            if ap:raise NormalizationError('Duration cannot have AM/PM')
            if l=='en':return _en_int(h)+' hours '+_en_int(minute)+' minutes'+(' '+_en_int(second)+' seconds'if m['s']is not None else'')
            return _zh_int(h)+'小时'+_zh_int(minute)+'分'+(_zh_int(second)+'秒'if m['s']is not None else'')
        if l=='en':
            result=_en_int(h)+((' oh '+_en_int(minute))if 0<minute<10 else(' '+_en_int(minute)if minute else" o clock"))
            if m['s']is not None:result+=' and '+_en_int(second)+' seconds'
            if ap:result+=' '+('a m'if ap.lower().startswith('a')else'p m')
            if ap and ap.endswith('.') and not m.string[m.end():].strip():result+='.'
            return result
        result=('上午'if ap and ap.lower().startswith('a')else'下午'if ap else'')+_zh_int(h)+'点'
        if m['s']is None and minute==0:return result+'整'
        result+=('零'if 0<minute<10 else'')+_zh_int(minute)+'分'
        if m['s']is not None:result+=('零'if 0<second<10 else'')+_zh_int(second)+'秒'
        return result
    protected=clock.sub(clock_read,protected)
    currencies=r'(?:CNY|RMB|USD|EUR|人民币|美元|欧元|￥|¥|\$|€)'
    money_prefix=re.compile(_PLAIN_LEFT+r'(?P<code>'+currencies+r')\s*(?P<num>'+_PLAIN_NUMBER+r')'+_PLAIN_RIGHT,re.I)
    protected=money_prefix.sub(lambda m:_read_money(m['num'],m['code'],lang(m)),protected)
    money_suffix=re.compile(_PLAIN_LEFT+r'(?P<num>'+_PLAIN_NUMBER+r')\s*(?P<code>CNY|RMB|USD|EUR|人民币|美元|欧元)'+_PLAIN_RIGHT,re.I)
    protected=money_suffix.sub(lambda m:_read_money(m['num'],m['code'],lang(m)),protected)
    percent=re.compile(_PLAIN_LEFT+r'(?P<num>'+_PLAIN_NUMBER+r')\s*[%％]'+_PLAIN_RIGHT)
    def percent_read(m):
        if _math_adjacent(m.string,m.start(),m.end()):return m.group()
        l=lang(m);value=speak_number(m['num'],l)
        return value+' percent'if l=='en'else'百分之'+value
    protected=percent.sub(percent_read,protected)
    tensor=re.compile(r'(?<![A-Za-z0-9_])\[\s*(\d+(?:\s*,\s*\d+){1,7})\s*\]')
    def tensor_read(m):
        context=m.string[max(0,m.start()-60):m.start()]
        if re.search(r'[A-Za-z_][A-Za-z0-9_]*\s*=\s*$',context):return m.group()
        if not re.search(r'(?:张量|形状|维度|尺寸|tensor|shape|dimension)',context,re.I) and m.string.strip()!=m.group():return m.group()
        l=lang(m);dimensions=[speak_number(value.strip(),l)for value in m[1].split(',')]
        return(' by 'if l=='en'else'乘').join(dimensions)
    protected=tensor.sub(tensor_read,protected)
    # Longest/squared units first, with strict case and identifier boundaries.
    unit_names=r'(?:cm(?:²|\^2|2)|m(?:²|\^2)|m/s|mAh|GHz|MHz|kHz|kg|cm|ms|mA|mV|MB|Hz|V|m|°\s*C)'
    units=re.compile(_PLAIN_LEFT+r'(?P<pm>±\s*)?(?P<num>'+_PLAIN_NUMBER+r')\s*(?P<unit>'+unit_names+r')(?![A-Za-z0-9_²³^])')
    def unit_read(m):
        if _math_adjacent(m.string,m.start(),m.end()):return m.group()
        l=lang(m);unit=re.sub(r'\s+','',m['unit']);unit={'cm2':'cm²','cm^2':'cm²','m^2':'m²'}.get(unit,unit)
        if unit not in _UNITS:raise NormalizationError('Unsupported typed unit')
        value=speak_number(m['num'],l)
        sign,integer,fraction=_parse_number(m['num']);single=Decimal(integer+('.'+fraction if fraction is not None else''))==1
        label=_UNITS[unit][0]if l=='zh'else _UNITS[unit][1 if single else 2]
        if m['pm']:value=('正负'if l=='zh'else'plus or minus ')+value
        return value+label if l=='zh'else value+' '+label
    protected=units.sub(unit_read,protected)
    # A bare tolerance ± literal is handled only in an explicit error/tolerance
    # context; x=±2 and other mathematical uses are left for the math parser.
    tolerance=re.compile(_PLAIN_LEFT+r'±\s*(?P<num>'+_PLAIN_NUMBER+r')'+_PLAIN_RIGHT)
    def tolerance_read(m):
        if not re.search(r'(?:误差|容差|tolerance)[^。！？!?；;\n]{0,24}$',m.string[max(0,m.start()-45):m.start()],re.I):return m.group()
        if _math_adjacent(m.string,m.start(),m.end()):return m.group()
        l=lang(m);return('正负'if l=='zh'else'plus or minus ')+speak_number(m['num'],l)
    protected=tolerance.sub(tolerance_read,protected)
    return restore_identifiers(protected,mapping)


# Context-sensitive syntax on top of the baseline formatter.


class NumericNormalizationError(ValueError):
    pass

# A prose comma is not part of the number; internal commas remain validated.
_NUMBER=r'[+\-−]?(?:[0-9](?:[0-9,]*[0-9])?(?:\.[0-9]+)?|\.[0-9]+)'
_LEFT=r'(?<![A-Za-z0-9_./+\-=])'
_RIGHT=r'(?![A-Za-z0-9_]|\.[0-9]|,[0-9])'
_DATE=re.compile(_LEFT+r'(?P<y>[0-9]{4})(?P<sep>[-/])(?P<m>[0-9]{1,2})(?P=sep)(?P<d>[0-9]{1,2})(?![A-Za-z0-9_/\-])')
_ISO_DATETIME = re.compile(
    _LEFT + r'(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})T(?P<clock>[0-9]{2}:[0-9]{2})'
    r'(?::(?P<seconds>[0-9]{2})(?P<fraction>\.[0-9]{1,9})?)?'
    r'(?:(?P<utc>Z)|(?P<offset>[+-][0-9]{2}):?(?P<minutes>[0-9]{2}))?'
    r'(?![A-Za-z0-9_/+:\-]|\.[A-Za-z0-9_])', re.I)
_PERCENTILE = r'[Pp](?P<percentile>100|[0-9]{1,2}(?:\.[0-9]+)?)'
_METRIC_COMPARISON = re.compile(
    _LEFT + r'(?:' + _PERCENTILE + r'|(?P<metric>QPS|TPS|RPS|TTFT|RTF))'
    r'[ \t]*(?P<op>==|!=|>=|<=|[=<>≤≥≠])[ \t]*(?=' + _NUMBER + r')', re.I)
_ZHDATE=re.compile(_LEFT+r'(?P<y>[0-9]{4})\s*年\s*(?P<m>[0-9]{1,2})\s*月\s*(?P<d>[0-9]{1,2})\s*[日号]')
_YM=re.compile(_LEFT+r'(?P<y>[0-9]{4})\s*年\s*(?P<m>[0-9]{1,2})\s*月(?!\s*[0-9]{1,2}\s*[日号])')
_YEAR=re.compile(_LEFT+r'(?P<y>[0-9]{4})\s*年(?!\s*[0-9]{1,2}\s*月)')
_EXTRA_UNITS={
 'TB':('太字节','terabyte','terabytes'),'GB':('吉字节','gigabyte','gigabytes'),
 'Gb':('吉比特','gigabit','gigabits'),'MB':('兆字节','megabyte','megabytes'),
 'Mb':('兆比特','megabit','megabits'),'kB':('千字节','kilobyte','kilobytes'),
 'GB/s':('吉字节每秒','gigabyte per second','gigabytes per second'),
 'Gb/s':('吉比特每秒','gigabit per second','gigabits per second'),
 'MB/s':('兆字节每秒','megabyte per second','megabytes per second'),
 'Mb/s':('兆比特每秒','megabit per second','megabits per second'),
 'MBps':('兆字节每秒','megabyte per second','megabytes per second'),
 'TBps':('太字节每秒','terabyte per second','terabytes per second'),
 'Mbps':('兆比特每秒','megabit per second','megabits per second'),
 'Gbps':('吉比特每秒','gigabit per second','gigabits per second'),
 'ms/s':('毫秒每秒','millisecond per second','milliseconds per second'),
 's':('秒','second','seconds'),'min':('分钟','minute','minutes'),
 'h':('小时','hour','hours'),'℃':('摄氏度','degree Celsius','degrees Celsius')}
# Case is semantic: milli m and mega M, byte B and bit b remain distinct.
# Only numeric quantities admit the explicit nonstandard kJ spelling aliases.
_EXTRA_UNITS.update({
 **{name: SHARED_UNITS[name] for name in ('mW', 'kW', '°F', 'µs', 'μs', 'mm', 'km', 'm³', 'km/h')},
 'J':('焦耳','joule','joules'), 'mJ':('毫焦','millijoule','millijoules'),
 'kJ':('千焦','kilojoule','kilojoules'), 'MJ':('兆焦','megajoule','megajoules'),
 'GJ':('吉焦','gigajoule','gigajoules'),
 'W':('瓦','watt','watts'),
 'MW':('兆瓦','megawatt','megawatts'),
 'GW':('吉瓦','gigawatt','gigawatts'),
 'Wh':('瓦时','watt hour','watt hours'),
 'kWh':('千瓦时','kilowatt hour','kilowatt hours'),
 'kWh/m²':('千瓦时每平方米','kilowatt hour per square meter','kilowatt hours per square meter'),
 'MWh':('兆瓦时','megawatt hour','megawatt hours'),

 '℉':('华氏度','degree Fahrenheit','degrees Fahrenheit'),


 'us':('微秒','microsecond','microseconds'),
 'ns':('纳秒','nanosecond','nanoseconds'),



 'cm³':('立方厘米','cubic centimeter','cubic centimeters'),

 'cm/s':('厘米每秒','centimeter per second','centimeters per second'),
 'Pa':('帕','pascal','pascals'), 'kPa':('千帕','kilopascal','kilopascals'),
 'MPa':('兆帕','megapascal','megapascals'),
 'L':('升','liter','liters'), 'mL':('毫升','milliliter','milliliters'),
 'KB':('千字节','kilobyte','kilobytes'),
 'kb':('千比特','kilobit','kilobits'),
})
_UNIT_ALIASES={'kj':'kJ','Kj':'kJ','KJ':'kJ'}
for _alias,_canonical in _UNIT_ALIASES.items():
 _EXTRA_UNITS[_alias]=_EXTRA_UNITS[_canonical]

_UNIT_NAMES = '|'.join(re.escape(name) for name in sorted(_UNITS.keys() | _EXTRA_UNITS.keys(), key=lambda name: (-len(name), name)))
_METRIC_FIELD = re.compile(
    _METRIC_COMPARISON.pattern + _NUMBER + r'(?:[ \t]*(?:' + _UNIT_NAMES + r')(?![A-Za-z0-9_]))?', re.I)

_QUANT=r'(?:个|条|台|份|项|次|人|件|张|元|页|套|本|组|items?\b|records?\b|requests?\b|devices?\b|people\b|times?\b|rows?\b)'
_QUANTITY=re.compile(_LEFT+r'(?P<n>'+_NUMBER+r')(?P<space>[ \t]*)(?P<q>'+_QUANT+r')',re.I)
_QUANTITY_RANGE = re.compile(
    _LEFT + r'(?P<a>' + _NUMBER + r')[ \t]*(?:-|–|—|~|～|至|到|to\b)[ \t]*'
    r'(?P<b>' + _NUMBER + r')(?P<space>[ \t]*)(?P<q>' + _QUANT + r'|天(?!气)|days?\b|workers?\b)', re.I)
_STOCK_LABEL=r'(?:股票代码|证券代码|股票编号|港股代码|股票代号|股份代号|stock[ \t]+code|ticker[ \t]+code)'
_STOCK=re.compile(r'(?P<label>'+_STOCK_LABEL+r'[ \t]*(?:(?:为|是|[:：#]|is|was)[ \t]*)?)(?P<code>[0-9]{1,6})(?P<market>\.?(?:HK|SH|SZ))?(?![A-Za-z0-9_]|\.[0-9]|[/\\]|[ \t]*'+_QUANT+r')',re.I)
_BARE_STOCK=re.compile(r'(?<![A-Za-z0-9_./\\])(?P<code>[0-9]{1,6})(?P<market>\.?(?:HK|SH|SZ))(?![A-Za-z0-9_./\\])',re.I)
_CODE=re.compile(r'(?P<label>(?<![A-Za-z0-9_])(?:错误码|错误代码|返回码|响应码|状态码|error[ \t]+code|status[ \t]+code|response[ \t]+code)[ \t]*(?:(?:为|是|[:：#]|is|was)[ \t]*)?)(?P<code>[0-9]{1,16})(?![A-Za-z0-9_]|\.[0-9]|[/\\]|[ \t]*'+_QUANT+r')',re.I)
_PORT=re.compile(r'(?P<label>(?:端口(?:号)?|(?<![A-Za-z0-9_])port(?:[ \t]+number)?\b)[ \t]*(?:(?:为|是|[:：#]|is|was)[ \t]*)?)(?P<code>[0-9]{1,5})(?![A-Za-z0-9_]|\.[0-9]|[/\\]|[ \t]*'+_QUANT+r')',re.I)
_IDENTIFIER=re.compile(r'(?P<label>编号[ \t]*(?:(?:为|是|[:：#])[ \t]*)?)(?P<code>[0-9]{1,16})(?![A-Za-z0-9_]|\.[0-9]|[/\\]|[ \t]*'+_QUANT+r')')


# Model numbers are identifiers, not counts. Limit this rule to four-digit
# RTX/GTX families; complete code, path, URL and filename masks still win.
_GPU_MODEL=re.compile(r'(?<![A-Za-z0-9_./\\-])(?P<family>RTX|GTX)[ \t]*(?P<model>[1-9][0-9]{3})(?P<suffix>[ \t]*(?:Ti|SUPER))?(?![A-Za-z0-9_/\\-]|\.[A-Za-z0-9_.])',re.I)
_MD_RIGHT=r'(?![A-Za-z0-9_/\\-]|\.[A-Za-z0-9_.])'
_MD_FIELD=re.compile(r'(?P<label>日期[ \t]*(?:(?:为|是|[:：])[ \t]*)?)(?P<m>[0-9]{1,2})/(?P<d>[0-9]{1,2})'+_MD_RIGHT)
_MD_EVENT=re.compile(r'(?P<label>(?:改到|定于|安排在|定在)[ \t]*)(?P<m>[0-9]{1,2})/(?P<d>[0-9]{1,2})'+_MD_RIGHT+r'(?=[ \t]*(?:举行|召开|开幕|开始|进行))')
_EN_MD_CALENDAR=re.compile(r'(?<![A-Za-z0-9_])(?:date[ \t]*(?:(?:is|was|[:=])[ \t]*)?|(?:scheduled[ \t]+(?:for|on)|(?:meeting|event|appointment)[ \t]+(?:is[ \t]+)?on)[ \t]+)[0-9]{1,2}/[0-9]{1,2}'+_MD_RIGHT,re.I)

# Keep model-specific unit aliases ahead of calendar inference.
_CALENDAR_UNITS = set(_EXTRA_UNITS)
for _readings in _EXTRA_UNITS.values():
    _CALENDAR_UNITS.update(_readings)
_CALENDAR_EXTRA_QUANTITY = re.compile(
    r'[ \t]*(?:' + '|'.join(map(re.escape, sorted(_CALENDAR_UNITS))) + r')(?![A-Za-z0-9_])', re.I)


def _read_calendar_fields(text, language):
    pieces = []
    cursor = 0
    for start, end, spoken in find_calendar_spans(
            text, language, lambda value, lang: speak_number(value, lang.lower()),
            extra_quantity=_CALENDAR_EXTRA_QUANTITY):
        pieces.extend((text[cursor:start], spoken))
        cursor = end
    return ''.join((*pieces, text[cursor:])) if pieces else text


# Only standalone ASCII filenames with reviewed file suffixes enter the literal
# reader. Domains, paths, URLs, mail and source-code quotations keep their own
# existing semantics; an arbitrary dotted expression is not a filename.
_FILENAME_SUFFIXES=frozenset(('json jsonl yaml yml toml ini cfg conf config env txt md rst csv tsv '
 'xml html htm css js jsx ts tsx py pyc sh bash zsh c h cpp cc hpp java go rs sql '
 'pdf doc docx xls xlsx ppt pptx png jpg jpeg gif svg webp bmp ico '
 'wav mp3 flac ogg m4a mp4 mov mkv webm pth pt ckpt onnx plan engine '
 'npy npz pkl pickle bin safetensors zip tar gz bz2 xz log').split())
_FILENAME=re.compile(r'(?<![A-Za-z0-9_./\\@+\-])(?:[A-Za-z0-9_][A-Za-z0-9_\-]*\.)+[A-Za-z][A-Za-z0-9_\-]*(?![A-Za-z0-9_/\\@+\-]|\.[A-Za-z0-9_])')
_FILENAME_EXCLUDED=re.compile(
 r'(?:https?://|ftp://|www\.)[^\s<>，。；！？]+'
 r'|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'
 r'|`[^`\n]*`'
 r'|(?<![A-Za-z0-9_])(?:[A-Za-z]:\\|\.{1,2}/|/)[A-Za-z0-9_./\\\-]+'
 r'|(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_.]*\s*=\s*(?:"(?:\\.|[^"\\\n])*"|\x27(?:\\.|[^\x27\\\n])*\x27)',re.I)

def filename_spans(text):
 """Return whole bare filename spans; never reinterpret a path/URL/email."""
 excluded=[m.span()for m in _FILENAME_EXCLUDED.finditer(text)]
 return tuple(m.span()for m in _FILENAME.finditer(text)
  if m.group().rsplit('.',1)[1].casefold()in _FILENAME_SUFFIXES
  and not any(a<m.end()and m.start()<b for a,b in excluded))

_RELATIVE_FILE_PATH=re.compile(r'(?<![A-Za-z0-9_./\\])(?:[A-Za-z0-9_.\-]+[/\\])+[A-Za-z0-9_][A-Za-z0-9_.\-]*[A-Za-z0-9_\-](?![A-Za-z0-9_./\\])')

def _relative_file_path_spans(text):
 return tuple(m.span()for m in _RELATIVE_FILE_PATH.finditer(text)
  if m.group().rsplit('.',1)[-1].casefold()in _FILENAME_SUFFIXES)

def _validate_auto_fallback(auto_fallback):
 if auto_fallback not in (None,'zh','en'):
  raise NumericNormalizationError('auto_fallback must be None/zh/en')

def _language(chunk,language,fallback):
 if language in ('zh','en'):return language
 # Script in date/unit notation is not the carrier language of an English clause.
 probe=_ZHDATE.sub(' ',chunk);probe=_YM.sub(' ',probe);probe=_YEAR.sub(' ',probe)
 probe=re.sub(r'[0-9:.+\-−]+\s*(?:年|月|日|号|毫秒|秒|分钟|小时|元|个|条)',' ',probe)
 if re.search('[\u3400-\u9fff]',probe):return 'zh'
 if re.search('[A-Za-z]',probe):return 'en'
 return fallback

def _natural_language(chunk):
 # Request-local prose carrier only: variables, formulas and identifier masks
 # must not change a Chinese request into English at the next comma.
 chunk = _METRIC_FIELD.sub(' ', chunk)
 spans=[(ts[0].start,ts[-1].end)for ts in _tokens_and_runs(chunk)if _is_math(ts)]
 for a,b in reversed(spans):chunk=chunk[:a]+' '*(b-a)+chunk[b:]
 chunk=re.sub(r'\ue000[A-Z]+\ue001',' ',chunk)
 chunk=_ZHDATE.sub(' ',chunk);chunk=_YM.sub(' ',chunk);chunk=_YEAR.sub(' ',chunk)
 chunk=_DATE.sub(' ',chunk)
 chunk=re.sub(r'[0-9:.+\-−]+\s*(?:年|月|日|号|毫秒|秒|分钟|小时|元|个|条)',' ',chunk)
 if re.search('[\u3400-\u9fff]',chunk):return 'zh'
 # Lower-case prose or sentence-capitalized words; a lone x/API/SageAIOS is
 # not a natural English sentence carrier. No language state escapes this call.
 if re.search(r'(?<![A-Za-z0-9_])[A-Z]?[a-z]{2,}(?![A-Za-z0-9_])',chunk):return 'en'
 return None

def auto_context_language(text,fallback='zh'):
 """Return (last natural carrier or fallback, has_carrier), without state.

 Identifiers and formulas cannot change the caller's session language. Natural
 English/Chinese prose does override it, including within the same request.
 """
 if not isinstance(text,str):raise TypeError('text must be str')
 if fallback not in ('zh','en'):raise NumericNormalizationError('fallback must be zh/en')
 if len(text)>20000:raise NumericNormalizationError('Input exceeds bounded20000 characters')
 protected,_=_protect(text);found=False
 for clause in _clauses(protected):
  carrier=_natural_language(clause)
  if carrier is not None:fallback=carrier;found=True
 return fallback,found

def _restore(text,mapping):
 # Later outer masks can contain an earlier URL mask, e.g. a URL in backticks.
 for token,value in reversed(list(mapping.items())):text=text.replace(token,value)
 return text

def _protect(text):
 mapping={}
 def stash_value(value):
  i=len(mapping)+1000;token=_token(i)
  while token in text or token in mapping:i+=1;token=_token(i)
  mapping[token]=value;return token
 def stash(match):return stash_value(match.group())
 # Whole URLs must precede the base's partial path/identifier masking.
 for pattern in [r'(?<![A-Za-z0-9_./\\])[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:\.[A-Za-z0-9_][A-Za-z0-9_.-]*)+',r'(?:https?://|ftp://|www\.)[^\s<>，。；！？]+',r'(?<![A-Za-z0-9_/])[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}(?![A-Za-z0-9_/])',_ISO_DATETIME.pattern,r'(?<![A-Za-z0-9_])[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.+Z\-]+']:
  text=re.sub(pattern,stash,text,flags=re.I)
 # Preserve an entire unimplemented compound unit, never expand just its
 # numerator. Known complete rates continue through the typed-unit path.
 atom_names='|'.join(re.escape(x)for x in sorted({**_UNITS,**_EXTRA_UNITS},key=len,reverse=True)if '/'not in x)
 text=re.sub(_LEFT+r'(?:'+_NUMBER+r')(?:'+atom_names+r')-[A-Za-z0-9_][A-Za-z0-9_.-]*',stash,text)
 range_value=_NUMBER+r'(?:[ \t]*(?:–|—|~|～|至|到|to\b|-)[ \t]*'+_NUMBER+r')?'
 denominator=r'(?:[0-9]*[A-Za-zµμ°]+(?:[²³]|\^[+\-]?[0-9]+)?|\([A-Za-zµμ°]+[²³]?\))'
 compounds=re.compile(_LEFT+r'(?P<n>'+range_value+r')[ \t]*(?P<u>(?:'+atom_names+r')(?:[ \t]*/[ \t]*'+denominator+r')+)(?![A-Za-z0-9_^/])')
 def compound(m):
  unit=re.sub(r'[ \t]+','',m['u'])
  if unit in {**_UNITS,**_EXTRA_UNITS}:return m['n']+unit
  return stash_value(m.group())
 text=compounds.sub(compound,text)
 # Protect simple report.json as well as digit/underscore identifiers. The
 # original whole literal is restored; its speaking form belongs to G2P.
 for a,b in reversed(_relative_file_path_spans(text)):
  text=text[:a]+stash_value(text[a:b])+text[b:]
 for a,b in reversed(filename_spans(text)):
  text=text[:a]+stash_value(text[a:b])+text[b:]
 text,native=protect_identifiers(text)
 for token,value in list(native.items()):
  if value not in ('/h','/s'):continue
  position=text.find(token)
  family='km'if value=='/h'else'cm'
  if position>=0 and re.search(_LEFT+r'(?:'+_NUMBER+r')[ \t]*'+family+r'$',text[:position]):
   text=text.replace(token,value);del native[token]
 if set(mapping)&set(native):raise NumericNormalizationError('Identifier placeholder collision')
 mapping.update(native)
 # Release only recognized literals to their typed readers.
 for token,value in list(mapping.items()):
  if token in text and (re.fullmatch(r'HTTP[1-5][0-9]{2}', value, re.I) or _GPU_MODEL.fullmatch(value)
                        or _ISO_DATETIME.fullmatch(value) or re.fullmatch(_PERCENTILE, value)):
   text=text.replace(token,value);del mapping[token]
 timer=re.compile(r'(?P<label>(?:倒计时|计时长度|持续时间|countdown|timer|duration)[^。！？!?；;\n0-9:]{0,16}?)(?P<value>[0-9]{1,2}:[0-9]{2})(?!:[0-9]|[0-9])',re.I)
 text=timer.sub(lambda m:m['label']+stash_value(m['value']),text)
 return text,mapping


def _clauses(text):
 thousands=[m.span()for m in re.finditer(r'[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?',text)]
 previous=0
 for m in re.finditer(r'[。！？!?；;，,\n]+|\.(?=\s|[\u3400-\u9fff]|$)',text):
  if any(a<=m.start()<b for a,b in thousands)or(text[m.start():m.end()]==','and m.start()>0 and m.end()<len(text)and text[m.start()-1].isdigit()and text[m.end()].isdigit()):continue
  yield text[previous:m.end()];previous=m.end()
 if previous<len(text):yield text[previous:]

def _number(value,lang,digits=False):
 # A code-like leading zero must never disappear via integer conversion.
 unsigned=value.lstrip('+-−').replace(',','').split('.')[0]
 return speak_number(value,lang,digits or(len(unsigned)>1 and unsigned.startswith('0')))

def _normalize_clause(chunk,lang):
 def iso_datetime(match):
  clock = match['clock']
  seconds = match['seconds']
  if match['fraction']:
   if int(seconds) > 59:
    raise NumericNormalizationError('Invalid clock time')
   value = _number(str(int(seconds)) + match['fraction'], lang)
   clock += f' and {value} seconds' if lang == 'en' else value + '秒'
  elif seconds and seconds != '00':
   clock += ':' + seconds
  timezone = ''
  if match['offset']:
   timezone = f" UTC{match['offset']}:{match['minutes']}"
  elif match['utc']:
   timezone = ' U T C'
  return f"{match['date']} {clock}{timezone}"
 chunk = _ISO_DATETIME.sub(iso_datetime, chunk)
 def metric_comparison(match):
  if match['percentile'] is not None:
   name = f"P {_number(match['percentile'], lang)}"
  else:
   name = ' '.join(match['metric'].upper())
  operator = SHARED_OPERATORS['=' if match['op'] == '==' else match['op']][lang == 'en']
  return f"{name} {operator} "
 chunk = _METRIC_COMPARISON.sub(metric_comparison, chunk)
 chunk=_read_calendar_fields(chunk,lang)
 currencies=r'(?:CNY|RMB|USD|EUR|人民币|美元|欧元|￥|¥|\$|€)'
 money_prefix=re.compile(_LEFT+r'(?P<code>'+currencies+r')[ \t]*(?P<n>'+_NUMBER+r')'+_RIGHT,re.I)
 money_suffix=re.compile(_LEFT+r'(?P<n>'+_NUMBER+r')[ \t]*(?P<code>CNY|RMB|USD|EUR|人民币|美元|欧元)'+_RIGHT,re.I)
 chunk=money_prefix.sub(lambda m:_read_money(m['n'],m['code'],lang),chunk)
 chunk=money_suffix.sub(lambda m:_read_money(m['n'],m['code'],lang),chunk)
 def word_boundary(m,value):
  # English typed replacements must not merge with an adjacent raw clock/token.
  if lang=='en':
   if m.start() and m.string[m.start()-1].isascii() and m.string[m.start()-1].isalnum():value=' '+value
   if m.end()<len(m.string) and m.string[m.end()].isascii() and m.string[m.end()].isalnum():value+=' '
  return value
 # These Chinese event/field structures establish month/day semantics.
 # English normalization has no MM/DD versus DD/MM locale contract.
 if _EN_MD_CALENDAR.search(chunk):
  raise NumericNormalizationError('Ambiguous slash calendar date: use an ISO date or an explicit month name')
 def month_day(m):
  if lang!='zh':raise NumericNormalizationError('Slash calendar date requires an explicit English month/day order')
  month,day=int(m['m']),int(m['d']);date(2000,month,day)
  return m['label']+_zh_int(month)+'月'+_zh_int(day)+'日'
 chunk=_MD_FIELD.sub(month_day,chunk);chunk=_MD_EVENT.sub(month_day,chunk)
 def gpu_model(m):
  code=m['model'];suffix=(' '+m['suffix'].strip())if m['suffix']else''
  if lang=='zh':reading=_number(code,lang,True)
  elif code.endswith('00'):reading=_en_int(int(code))
  else:reading=_en_int(int(code[:2]))+' '+_en_int(int(code[2:]))
  return m['family']+' '+reading+suffix
 chunk=_GPU_MODEL.sub(gpu_model,chunk)
 def full_date(m):return word_boundary(m,_read_date(m['y'],m['m'],m['d'],lang))
 chunk=_DATE.sub(full_date,chunk);chunk=_ZHDATE.sub(full_date,chunk)
 def ym(m):
  y,month=int(m['y']),int(m['m']);date(y,month,1)
  value=_MONTHS[month]+' '+_en_int(y)if lang=='en'else speak_number(m['y'],'zh',True)+'年'+_zh_int(month)+'月'
  return word_boundary(m,value)
 chunk=_YM.sub(ym,chunk)
 chunk=_YEAR.sub(lambda m:word_boundary(m,_en_int(int(m['y'])))if lang=='en'else speak_number(m['y'],'zh',True)+'年',chunk)
 # Explicit ongoing duration, not a deadline such as 持续到09:05. Require H:M:S.
 elapsed=re.compile(r'(?P<label>(?:预计[ \t]*)?持续[ \t]*(?:(?:约|大约|为|是|[:：])[ \t]*)?)(?P<h>[0-9]{1,2}):(?P<m>[0-9]{2}):(?P<s>[0-9]{2})(?![A-Za-z0-9_:])')
 def elapsed_read(m):
  h,minute,second=int(m['h']),int(m['m']),int(m['s'])
  if minute>59 or second>59:raise NumericNormalizationError('Invalid explicit elapsed time')
  if lang=='en':value=_en_int(h)+' hours '+_en_int(minute)+' minutes '+_en_int(second)+' seconds'
  else:value=_zh_int(h)+'小时'+_zh_int(minute)+'分'+_zh_int(second)+'秒'
  return m['label']+word_boundary(m,value)
 chunk=elapsed.sub(elapsed_read,chunk)
 # 12 AM is midnight and 12 PM is noon. Render only explicit AM/PM
 # clocks here; the pinned general formatter retains all other clock policy.
 clock12=re.compile(r'(?<![A-Za-z0-9_./+\-=:])(?P<h>[0-9]{1,2}):(?P<m>[0-9]{2})(?::(?P<s>[0-9]{2}))?[ \t]*(?P<ap>[ap]\.?m\.?)(?![A-Za-z0-9_:])',re.I)
 def clock12_read(m):
  h,minute,second=int(m['h']),int(m['m']or'0'),int(m['s']or'0')
  if not 1<=h<=12 or minute>59 or second>59:raise NumericNormalizationError('Invalid12-hour clock time')
  if not(lang=='zh'and h==12):return m.group()
  prefix=m.string[max(0,m.start()-45):m.start()]
  if re.search(r'(?:倒计时|计时长度|持续时间|countdown|timer|duration)[^。！？!?；;\n]{0,24}$',prefix,re.I):raise NumericNormalizationError('Duration cannot have AM/PM')
  part=('凌晨'if m['ap'].lower().startswith('a')else'中午')if h==12 else('上午'if m['ap'].lower().startswith('a')else'下午')
  hour=0 if h==12 and m['ap'].lower().startswith('a')else h
  result=part+_zh_int(hour)+'点'
  if m['s']is None and minute==0:result+='整'
  else:
   result+=('零'if 0<minute<10 else'')+_zh_int(minute)+'分'
   if m['s']is not None:result+=('零'if 0<second<10 else'')+_zh_int(second)+'秒'
  if m['ap'].endswith('.')and not m.string[m.end():].strip():result+='.'
  return result
 chunk=clock12.sub(clock12_read,chunk)
 def code(m):
  label=m['label'];value=m['code']
  if re.search(r'状态码|status[ \t]+code|响应码|response[ \t]+code',label,re.I)or re.search(r'\bHTTP\b',chunk,re.I):
   if not(re.fullmatch('[1-5][0-9]{2}',value)):return m.group()
  return label+(' 'if lang=='en'and label[-1:].isalnum()else'')+_number(value,lang,True)
 chunk=_STOCK.sub(lambda m:m['label']+(' 'if lang=='en'and m['label'][-1:].isalnum()else'')+_number(m['code'],lang,True)+(' '+m['market'].lstrip('.').upper()if m['market']else''),chunk)
 chunk=_BARE_STOCK.sub(lambda m:_number(m['code'],lang,True)+' '+m['market'].lstrip('.').upper(),chunk)
 def port(m):
  if int(m['code'])>65535:raise NumericNormalizationError('Explicit port outside0..65535')
  return m['label']+(' 'if lang=='en'and m['label'][-1:].isalnum()else'')+_number(m['code'],lang,True)
 chunk=_PORT.sub(port,chunk);chunk=_CODE.sub(code,chunk)
 chunk=_IDENTIFIER.sub(lambda m:m['label']+(' 'if lang=='en'and m['label'][-1:].isascii()and m['label'][-1:].isalnum()else'')+_number(m['code'],lang,True),chunk)
 # A slash before an explicit cup/portion unit is a quantity fraction, not a date.
 fractions=re.compile(_LEFT+r'(?P<a>[0-9]{1,16})[ \t]*/[ \t]*(?P<b>[0-9]{1,16})(?P<space>[ \t]*)(?P<u>杯|份|cups?\b|portions?\b)')
 def fraction_read(m):
  denominator=int(m['b']);numerator=int(m['a'])
  if denominator==0:raise NumericNormalizationError('Quantity fraction denominator is zero')
  if lang=='zh':return _zh_int(denominator)+'分之'+_zh_int(numerator)+m['space']+m['u']
  names={2:('half','halves'),3:('third','thirds'),4:('quarter','quarters'),5:('fifth','fifths'),6:('sixth','sixths'),8:('eighth','eighths'),10:('tenth','tenths')}
  value=(_en_int(numerator)+' '+names[denominator][numerator!=1])if denominator in names else _en_int(numerator)+' over '+_en_int(denominator)
  if re.match('[A-Za-z]',m['u']):return value+' of a '+m['u'].rstrip('s')
  return value+m['space']+m['u']
 chunk=fractions.sub(fraction_read,chunk)
 labels={**_UNITS,**_EXTRA_UNITS};names=_UNIT_NAMES
 def unit_label(u,value,range_value=False):
  singular=not range_value and abs(Decimal(value.replace('−','-').replace(',','')))==1
  return labels[u][0 if lang=='zh'else 1 if singular else 2]
 ranges=re.compile(_LEFT+r'(?P<a>'+_NUMBER+r')\s*(?:–|—|~|～|至|到|to\b|-)\s*(?P<b>'+_NUMBER+r')\s*(?P<u>'+names+r'|[%％])(?![A-Za-z0-9_²³^/\\\ue000]|\.[A-Za-z0-9_\ue000]|-[A-Za-z0-9_\ue000])')
 def range_read(m):
  if _math_adjacent(m.string,m.start(),m.end()):return m.group()
  a,b=_number(m['a'],lang),_number(m['b'],lang);u=m['u']
  if u in ['%','％']:return a+' to '+b+' percent'if lang=='en'else'百分之'+a+'到百分之'+b
  return a+(' to 'if lang=='en'else'到')+b+(' 'if lang=='en'else'')+unit_label(u,m['b'],True)
 chunk=ranges.sub(range_read,chunk)
 # A bare hyphen stays subtraction unless an explicit range/interval label disambiguates it.
 bare_range=re.compile(r'(?P<label>(?:范围|区间|(?<![A-Za-z0-9_])(?:range|interval)\b)[ \t]*(?:(?:为|是|[:：]|is|of|from)[ \t]*)?)(?P<a>'+_NUMBER+r')[ \t]*(?:–|—|~|～|至|到|to\b|-)[ \t]*(?P<b>'+_NUMBER+r')'+_RIGHT,re.I)
 chunk=bare_range.sub(lambda m:m['label']+_number(m['a'],lang)+(' to 'if lang=='en'else'到')+_number(m['b'],lang),chunk)
 units=re.compile(_LEFT+r'(?P<pm>±\s*)?(?P<n>'+_NUMBER+r')\s*(?P<u>'+names+r')(?![A-Za-z0-9_²³^/\\\ue000]|\.[A-Za-z0-9_\ue000]|-[A-Za-z0-9_\ue000])')
 def unit_read(m):
  if _math_adjacent(m.string,m.start(),m.end()):return m.group()
  value=_number(m['n'],lang);value=(('plus or minus 'if lang=='en'else'正负')if m['pm']else'')+value
  return value+(' 'if lang=='en'else'')+unit_label(m['u'],m['n'])
 chunk=units.sub(unit_read,chunk)
 def quantity_range_read(match):
  if _math_adjacent(match.string, match.start(), match.end()):
   return match.group()
  first, last = (_number(match[bound], lang) for bound in ('a', 'b'))
  connector = ' to ' if lang == 'en' else '到'
  space = match['space'] or (' ' if match['q'][0].isascii() else '')
  return first + connector + last + space + match['q']
 chunk = _QUANTITY_RANGE.sub(quantity_range_read, chunk)
 def quantity_read(m):
  if _math_adjacent(m.string,m.start(),m.end()):return m.group()
  return _number(m['n'],lang)+(m['space']or(' 'if re.match('[A-Za-z]',m['q'])else''))+m['q']
 chunk=_QUANTITY.sub(quantity_read,chunk)
 # Explicit English normalization of a named Chinese scalar field. Do not
 # consume operands of formulas/ranges or change unlabelled integer/ID policy.
 if lang=='en':
  fields=r'(?:温度范围|数量|结果|温度|范围|延迟|速度|时长|值)'
  field_integer=re.compile(r'(?P<label>'+fields+r'[ \t]*(?:为|是|[:：])[ \t]*)(?P<n>[+\-−]?[0-9](?:[0-9,]*[0-9])?)'+_RIGHT)
  def named_integer(m):
   after=m.string[m.end('n'):].lstrip()[:1]
   if after in ('-','−','%','％') or (after and after in '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ₀₁₂₃₄₅₆₇₈₉₊₋') or _math_adjacent(m.string,m.start('n'),m.end('n')):return m.group()
   return m['label']+word_boundary(m,_number(m['n'],lang))
  chunk=field_integer.sub(named_integer,chunk)
 # Typed bare decimal/signed values; formulas/tolerances/percentages stay with V6.
 formula_spans=[]
 for ts in _tokens_and_runs(chunk):
  if not _is_math(ts):continue
  # Ordinary standalone signed scalar readings keep their historical minus/负
  # policy. A grouped scalar participating in a formula is not standalone.
  scalar=(len(ts)==1 and ts[0].kind=='NUMBER')or(len(ts)==2 and ts[0].kind=='OP'and ts[0].value in('+','-','−')and ts[1].kind=='NUMBER')
  if not scalar:formula_spans.append((ts[0].start,ts[-1].end))
 numeric=re.compile(_LEFT+r'(?P<n>'+_NUMBER+r')'+_RIGHT)
 def simple(m):
  if any(a<=m.start()and m.end()<=b for a,b in formula_spans):return m.group()
  n=m['n'];before=m.string[:m.start()].rstrip()[-1:];after=m.string[m.end():].lstrip()[:1]
  # Preserve the complete raw subtraction before invoking the pinned parser.
  # The base adjacency set deliberately omits minus for ordinary signed values.
  arithmetic_minus=before in ('-','−') or after in ('-','−')
  signed_binary=n.startswith(('+','-','−')) and bool(before) and (before.isdigit() or before in (')',']'))
  if not('.'in n or n.startswith(('+','-','−')))or before=='±'or after in ['%','％','²','³']or arithmetic_minus or signed_binary or _math_adjacent(m.string,m.start(),m.end()):return m.group()
  return _number(n,lang)
 chunk=numeric.sub(simple,chunk)
 return normalize_math(normalize_general(chunk,lang),lang)

def normalize_context(text,language='auto',*,auto_fallback=None):
 """Apply typed, math and Unicode normalization before restoring masked literals."""
 if not isinstance(text,str):raise TypeError('text must be str')
 if language not in ['auto','mixed','zh','en']:raise NumericNormalizationError('language must be auto/mixed/zh/en')
 if len(text)>20000:raise NumericNormalizationError('Input exceeds bounded20000 characters')
 _validate_auto_fallback(auto_fallback)
 protected,mapping=_protect(text)
 fallback=_language(text,language,'zh')if auto_fallback is None else auto_fallback;parts=[]
 try:
  for clause in _clauses(protected):
   if language in ('zh','en'):lang=language
   else:
    carrier=_natural_language(clause);lang=carrier or fallback
    if carrier is not None:fallback=carrier
   parts.append(_normalize_clause(clause,lang))
 except ValueError as e:raise NumericNormalizationError(str(e))from e
 # Finish the current V6 static-normalization contract exactly once.
 result=unicodedata.normalize('NFKC',''.join(parts)).strip();result=re.sub(r'\s+',' ',result);result=result.replace('·',' ').replace('–','—')
 if re.search(r'[À-ÖØ-öø-ÿĀ-ſ]',result):raise NumericNormalizationError('Unsupported accented Latin; preserve/review explicitly')
 return _restore(result,mapping)


# Formula tokenization, parsing and speech rendering.


class MathNormalizationError(ValueError):
    """A recognized mathematical span cannot be rendered safely."""


Token = namedtuple('Token', 'kind value start end')
SUPERSCRIPTS = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ', '0123456789+-n'))
SUBSCRIPTS = dict(zip('₀₁₂₃₄₅₆₇₈₉₊₋', '0123456789+-'))
TOKEN_RE = re.compile(
    r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?:\d+(?:\.\d+)?|\.\d+)'
    r'|[A-Za-z][A-Za-z0-9_]*|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ]+|[₀₁₂₃₄₅₆₇₈₉₊₋]+'
    r'|>=|<=|!=|==|[π√+\-−*×÷/^=≥≤≠<>|()@∑∫≈∈%％\[\]{}]')

# These denote source text, identifiers or typed units, not math formulas.
# a/b remains mathematical; long/multi-component paths and m/s remain literal.
PROTECTED_PATTERNS = [
    re.compile(r'\ue000[A-Z]+\ue001'),
    re.compile(r'```[\s\S]*?```|`[^`\n]*`'),
    re.compile(r'(?:https?|ftp)://[^\s，。；;!]+', re.I),
    re.compile(r'\b[A-Za-z]:[\\/][^\s，。；;!?]+'),
    re.compile(r'(?<![\w)])(?:\.{0,2}/)(?:[A-Za-z0-9_.-]+/)*[A-Za-z_.][A-Za-z0-9_.-]*'),
    re.compile(r'\bC\+\+(?:\d+)?(?![A-Za-z0-9_])|\bC#(?![A-Za-z0-9_])', re.I),
    re.compile(r'\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b'),
    re.compile(r'\b(?:[A-Za-z][A-Za-z0-9_]+-[A-Za-z0-9_-]+|[A-Za-z]-[A-Za-z][A-Za-z0-9_-]+)\b'),
    re.compile(r'\b[A-Za-z_][A-Za-z0-9_]{1,}\s*(?:==|=)[^，。；;\n]*'),
    re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
    re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b'),
    re.compile(r'\b(?:[vV]\d+(?:\.\d+)+|\d+(?:\.\d+){2,})\b'),
    re.compile(r'\b\d+(?:\.\d+)?[eE][+-]?\d+\b'),
    re.compile(r'\b(?:kg/m/s|kg/m(?:[²³]|\^[23])?|m/s(?:[²³]|\^[23])?|km/h|cm/s|mg/mL)\b'),
    re.compile(r'\{[^{}\n]*["\'][^{}\n]*\}'),
]


def _protected_spans(text):
    spans = [(m.start(), m.end()) for pattern in PROTECTED_PATTERNS for m in pattern.finditer(text)]
    # Bare paths require a multi-letter component or more than one slash.
    for m in re.finditer(r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)+(?![A-Za-z0-9_])', text):
        parts = m.group().split('/')
        if len(parts) > 2 or any(len(p) > 1 for p in parts):
            spans.append((m.start(), m.end()))
    return sorted(spans)


def _tokens_and_runs(text):
    protected = _protected_spans(text)
    runs, current = [], []
    previous_end = 0
    def finish():
        if current:
            runs.append(tuple(current))
            current.clear()
    for m in TOKEN_RE.finditer(text):
        if text[previous_end:m.start()].strip():
            finish()
        previous_end = m.end()
        if any(a < m.end() and b > m.start() for a, b in protected):
            finish()
            continue
        v = m.group()
        if v[0] in SUPERSCRIPTS:
            kind = 'SUP'
        elif v[0] in SUBSCRIPTS:
            kind = 'SUB'
        elif v[0].isdigit() or v.startswith('.'):
            kind = 'NUMBER'
        elif re.fullmatch('[A-Za-z][A-Za-z0-9_]*', v):
            if len(v) == 1:
                kind = 'VAR'
            elif v in ('sin', 'sqrt', 'cos', 'tan', 'log', 'exp'):
                kind = 'FUNCTION'
            elif re.match(r'\s*[*×]\s*[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)\s*[%％](?![A-Za-z0-9_])', text[m.end():]):
                # A named operand is admitted only for unambiguous multiplication
                # by a numerical percentage, never a general code assignment.
                kind = 'VAR'
            elif re.match(r'\s*\(', text[m.end():]):
                kind = 'FUNCTION'  # Explicit unknown math functions fail closed.
            else:
                finish()
                continue
        elif v == 'π':
            kind = 'PI'
        elif v == '√':
            kind = 'ROOT'
        elif v == '(':
            kind = 'LPAREN'
        elif v == ')':
            kind = 'RPAREN'
        elif v == '|':
            kind = 'ABS'
        elif v in ('%', '％'):
            kind = 'PERCENT'
        elif v in ('@', '∑', '∫', '≈', '∈', '[', ']', '{', '}'):
            kind = 'UNSUPPORTED'
        else:
            kind = 'OP'
        token = Token(kind, v, m.start(), m.end())
        # Do not absorb an English article/pronoun into the next expression.
        # Adjacent 2x and spaced 2 x remain supported implicit multiplication.
        if current and text[current[-1].end:token.start].isspace():
            prev = current[-1]
            if prev.kind in ('VAR', 'NUMBER', 'SUP', 'SUB', 'RPAREN') and token.kind in ('VAR', 'NUMBER', 'FUNCTION', 'PI', 'ROOT'):
                if not (prev.kind == 'NUMBER' and token.kind == 'VAR'):
                    finish()
        current.append(token)
    finish()
    return runs


def _is_math(tokens):
    if not any(t.kind in ('NUMBER', 'VAR', 'PI') for t in tokens):
        return False
    if any(t.kind in ('SUP', 'SUB', 'ROOT', 'PI') for t in tokens):
        return True
    if any(t.kind == 'FUNCTION' and i+1 < len(tokens) and tokens[i+1].kind == 'LPAREN'
           for i, t in enumerate(tokens)):
        return True
    if any(t.kind == 'OP' for t in tokens):
        return True
    # Percentages belong to general formatting. A binary modulo is unsupported.
    return any(t.kind in ('UNSUPPORTED', 'PERCENT') and 0 < i < len(tokens)-1 for i, t in enumerate(tokens))


class _Parser:
    def __init__(self, tokens, source):
        if len(tokens) > 256:
            raise MathNormalizationError('Mathematical span exceeds 256 lexical tokens')
        self.ts, self.i, self.depth, self.source = tokens, 0, 0, source

    def peek(self, *values):
        return self.i < len(self.ts) and self.ts[self.i].value in values

    def take(self):
        if self.i >= len(self.ts):
            raise MathNormalizationError('Incomplete mathematical expression')
        t = self.ts[self.i]
        self.i += 1
        return t

    def expect(self, value):
        if not self.peek(value):
            raise MathNormalizationError('Missing matching '+value+' in mathematical expression')
        return self.take()

    def parse(self):
        if any(t.kind == 'UNSUPPORTED' for t in self.ts):
            raise MathNormalizationError('Unsupported mathematical operator')
        node = self.comparison()
        if self.i != len(self.ts):
            raise MathNormalizationError('Unparsed mathematical token: '+self.ts[self.i].value)
        return node

    def comparison(self):
        node = self.sum()
        while self.peek('=', '==', '!=', '≠', '<', '>', '<=', '>=', '≤', '≥'):
            op = self.take().value
            node = ('binary', op, node, self.sum())
        return node

    def sum(self):
        node = self.product()
        while self.peek('+', '-', '−'):
            node = ('binary', self.take().value, node, self.product())
        return node

    def product(self):
        node = self.unary()
        while self.i < len(self.ts):
            if self.peek('*', '×', '/', '÷'):
                node = ('binary', self.take().value, node, self.unary())
            else:
                t, prev = self.ts[self.i], self.ts[self.i-1]
                implicit = t.kind in ('VAR', 'PI', 'LPAREN', 'FUNCTION', 'ROOT') and (
                    t.start == prev.end or (prev.kind == 'NUMBER' and t.kind == 'VAR'))
                if not implicit:
                    break
                node = ('binary', '*', node, self.unary())
        return node

    def unary(self):
        if self.peek('+', '-', '−'):
            op = self.take().value
            return ('unary', op, self.unary())
        return self.power()

    def power(self):
        node = self.atom()
        if self.peek('%', '％'):
            if node[0] != 'number':
                raise MathNormalizationError('Percent postfix requires a numeric literal')
            self.take()
            node = ('percent', node)
        if self.i < len(self.ts) and self.ts[self.i].kind == 'SUP':
            raw = self.take().value
            exponent = ''.join(SUPERSCRIPTS[c] for c in raw)
            if re.fullmatch(r'[+-]?\d+', exponent):
                value = ('number', exponent.lstrip('+-'))
                if exponent[0] in '+-':
                    value = ('unary', exponent[0], value)
            elif exponent == 'n':
                value = ('var', 'n')
            else:
                raise MathNormalizationError('Unsupported superscript sequence')
            node = ('power', node, value)
        elif self.peek('^'):
            self.take()
            node = ('power', node, self.unary())
        return node

    def atom(self):
        token = self.take()
        if token.kind == 'NUMBER':
            node = ('number', token.value.replace(',', ''))
        elif token.kind == 'VAR':
            node = ('var', token.value)
        elif token.kind == 'PI':
            node = ('pi',)
        elif token.kind == 'ROOT':
            node = ('root', self.power())
        elif token.kind == 'FUNCTION':
            if token.value not in ('sin', 'sqrt'):
                raise MathNormalizationError('Unsupported mathematical function: '+token.value)
            self.expect('(')
            self.depth += 1
            if self.depth > 24:
                raise MathNormalizationError('Mathematical grouping is too deep')
            inner = self.comparison()
            self.expect(')')
            self.depth -= 1
            node = ('function', token.value, ('group', inner))
        elif token.kind in ('LPAREN', 'ABS'):
            self.depth += 1
            if self.depth > 24:
                raise MathNormalizationError('Mathematical grouping is too deep')
            inner = self.comparison()
            self.expect(')' if token.kind == 'LPAREN' else '|')
            self.depth -= 1
            node = ('group' if token.kind == 'LPAREN' else 'absolute', inner)
        else:
            raise MathNormalizationError('Expected mathematical operand, got: '+token.value)
        if self.i < len(self.ts) and self.ts[self.i].kind == 'SUB':
            if node[0] != 'var':
                raise MathNormalizationError('Only variable digit subscripts are supported')
            sub = ''.join(SUBSCRIPTS[c] for c in self.take().value)
            if not re.fullmatch(r'[+-]?\d+', sub):
                raise MathNormalizationError('Unsupported subscript sequence')
            subnode = ('number', sub.lstrip('+-'))
            if sub[0] in '+-':
                subnode = ('unary', sub[0], subnode)
            node = ('subscript', node, subnode)
        return node


def _formula_zh_int(n):
    digits = '零一二三四五六七八九'
    if n == 0:
        return '零'
    for value, unit in ((1000000000000, '万亿'), (100000000, '亿'), (10000, '万')):
        if n >= value:
            q, r = divmod(n, value)
            return _formula_zh_int(q)+unit+(('零' if r < value//10 else '')+_formula_zh_int(r) if r else '')
    result, pending = '', False
    for value, unit in ((1000,'千'),(100,'百'),(10,'十'),(1,'')):
        q, n = divmod(n, value)
        if q:
            if pending:
                result += '零'
            result += digits[q]+unit
            pending = False
        elif result and n:
            pending = True
    return result[1:] if result.startswith('一十') else result


def _formula_number(value, language):
    if not re.fullmatch(r'(?:\d+(?:\.\d+)?|\.\d+)', value) or len(value) > 32:
        raise MathNormalizationError('Unsupported mathematical numeric literal')
    whole, dot, frac = value.partition('.')
    whole = whole or '0'
    if len(whole) > 18:
        raise MathNormalizationError('Mathematical integer is longer than 18 digits')
    digits = list('零一二三四五六七八九') if language == 'zh' else 'zero one two three four five six seven eight nine'.split()
    if len(whole) > 1 and whole.startswith('0'):
        spoken = ' '.join(digits[int(x)] for x in whole)
    else:
        spoken = _formula_zh_int(int(whole)) if language == 'zh' else _en_int(int(whole))
    if dot:
        spoken += (' 点 ' if language == 'zh' else ' point ')+' '.join(digits[int(x)] for x in frac)
    return spoken


_BINARY_OPERATORS = ('+', '-', '−', '*', '×', '/', '÷', '=', '==', '!=', '≠', '>', '<', '>=', '≥', '<=', '≤')
_OPERATOR_ALIASES = {'−': '-', '==': '='}
ZH_OPS = {operator: SHARED_OPERATORS[_OPERATOR_ALIASES.get(operator, operator)][0] for operator in _BINARY_OPERATORS}
EN_OPS = {operator: SHARED_OPERATORS[_OPERATOR_ALIASES.get(operator, operator)][1] for operator in _BINARY_OPERATORS}
EN_OPS.update({'!=': 'does not equal', '≠': 'does not equal'})


def _render(node, language):
    kind = node[0]
    zh = language == 'zh'
    if kind == 'number':
        return _formula_number(node[1], language)
    if kind == 'percent':
        value = _render(node[1], language)
        return '百分之 '+value if zh else value+' percent'
    if kind == 'var':
        return node[1]
    if kind == 'pi':
        return '派' if zh else 'pi'
    if kind == 'group':
        return ('左括号 ' if zh else 'open parenthesis ')+_render(node[1],language)+(' 右括号' if zh else ' close parenthesis')
    if kind == 'absolute':
        target = node[1]
        value = _render(target if target[0] in ('var','number','subscript') else ('group',target),language)
        return value+' 的绝对值' if zh else 'the absolute value of '+value
    if kind == 'subscript':
        return _render(node[1],language)+(' 下标 ' if zh else ' subscript ')+_render(node[2],language)
    if kind == 'unary':
        target = node[2]
        if zh and target[0] == 'percent':
            sign = '正 ' if node[1] == '+' else '负 '
            return '百分之 '+sign+_render(target[1], language)
        value = _render(target, language)
        if target[0] in ('binary','power'):
            value = _render(('group',target),language)
        return (('负 ' if zh else 'negative ') if node[1] != '+' else ('正 ' if zh else 'positive '))+value
    if kind == 'binary':
        ops = ZH_OPS if zh else EN_OPS
        return _render(node[2],language)+' '+ops[node[1]]+' '+_render(node[3],language)
    if kind == 'power':
        base, exponent = _render(node[1],language), node[2]
        if exponent == ('number','2'):
            return base+(' 的平方' if zh else ' squared')
        if exponent == ('number','3'):
            return base+(' 的立方' if zh else ' cubed')
        if exponent[0] in ('binary','power'):
            exponent = ('group',exponent)
        return base+(' 的 ' if zh else ' to the power of ')+_render(exponent,language)+(' 次方' if zh else '')
    if kind == 'root':
        target = node[1]
        if target[0] in ('binary','power'):
            target = ('group',target)
        return ('根号 ' if zh else 'square root of ')+_render(target,language)
    if kind == 'function':
        name = ('正弦' if node[1]=='sin' else '平方根') if zh else ('sine of' if node[1]=='sin' else 'square root of')
        return name+' '+_render(node[2],language)
    raise MathNormalizationError('Unsupported mathematical AST node')


def normalize_math(text, language='auto'):
    """Read supported mathematical spans in raw text; preserve all other bytes.

    language auto/mixed uses Chinese math words when the complete request contains
    Han characters, otherwise English. Explicit zh/en uses that carrier language.
    Not a symbolic calculator or general program/code parser. Incomplete or
    unsupported recognized expressions fail closed with the original span.
    """
    if not isinstance(text, str):
        raise TypeError('text must be str')
    if language not in ('auto','mixed','zh','en'):
        raise ValueError('language must be auto, mixed, zh or en')
    if len(text) > 20000:
        raise MathNormalizationError('Text is too long for bounded mathematical parsing')
    carrier = 'zh' if language == 'zh' or (language in ('auto','mixed') and re.search(r'[\u3400-\u9fff]',text)) else 'en'
    edits = []
    for ts in _tokens_and_runs(text):
        if not _is_math(ts):
            continue
        start, end = ts[0].start, ts[-1].end
        try:
            ast = _Parser(ts, text).parse()
            spoken = _render(ast, carrier)
        except MathNormalizationError as exc:
            raise MathNormalizationError(str(exc)+'; span='+repr(text[start:end])) from exc
        edits.append((start,end,spoken))
    for start,end,spoken in reversed(edits):
        # Spaces keep English variables and source-neighbor words separate.
        left = ' ' if start and text[start-1].isascii() and text[start-1].isalnum() else ''
        right = ' ' if end<len(text) and text[end].isascii() and text[end].isalnum() else ''
        text = text[:start]+left+spoken+right+text[end:]
    return text


# Text boundaries for acoustic profiles and streaming.

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
