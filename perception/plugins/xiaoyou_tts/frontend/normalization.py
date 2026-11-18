"""Verbalize typed identifiers, dates, times, amounts and units before math and G2P."""
from datetime import date
from decimal import Decimal
import ipaddress
import re

DEPENDENCIES = ('Python standard library',)

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
_NUMBER = r'[+\-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)'
_LEFT = r'(?<![A-Za-z0-9_./+=\-])'
_RIGHT = r'(?![A-Za-z0-9_]|\.\d|,\d)'
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
 'kg':('千克','kilogram','kilograms'), 'm':('米','meter','meters'),
 'cm':('厘米','centimeter','centimeters'), 'ms':('毫秒','millisecond','milliseconds'),
 'm/s':('米每秒','meter per second','meters per second'),
 'Hz':('赫兹','hertz','hertz'), 'GHz':('吉赫兹','gigahertz','gigahertz'),
 'MHz':('兆赫兹','megahertz','megahertz'), 'kHz':('千赫兹','kilohertz','kilohertz'),
 'MB':('兆字节','megabyte','megabytes'), 'mA':('毫安','milliamp','milliamps'),
 'mAh':('毫安时','milliampere hour','milliampere hours'),
 'V':('伏','volt','volts'), 'mV':('毫伏','millivolt','millivolts'),
 'cm²':('平方厘米','square centimeter','square centimeters'),
 'm²':('平方米','square meter','square meters'),
 '°C':('摄氏度','degree Celsius','degrees Celsius'),
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
    iso=re.compile(_LEFT+r'(?P<y>\d{4})(?P<sep>[-/])(?P<m>\d{1,2})(?P=sep)(?P<d>\d{1,2})(?![A-Za-z0-9_/\-])')
    protected=iso.sub(lambda m:_read_date(m['y'],m['m'],m['d'],lang(m)),protected)
    chinese=re.compile(_LEFT+r'(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*[日号]')
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
    clock=re.compile(_LEFT+r'(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?(?:\s*(?P<ap>[ap]\.?m\.?))?(?![A-Za-z0-9_:])',re.I)
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
    money_prefix=re.compile(_LEFT+r'(?P<code>'+currencies+r')\s*(?P<num>'+_NUMBER+r')'+_RIGHT,re.I)
    protected=money_prefix.sub(lambda m:_read_money(m['num'],m['code'],lang(m)),protected)
    money_suffix=re.compile(_LEFT+r'(?P<num>'+_NUMBER+r')\s*(?P<code>CNY|RMB|USD|EUR|人民币|美元|欧元)'+_RIGHT,re.I)
    protected=money_suffix.sub(lambda m:_read_money(m['num'],m['code'],lang(m)),protected)
    percent=re.compile(_LEFT+r'(?P<num>'+_NUMBER+r')\s*[%％]'+_RIGHT)
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
    units=re.compile(_LEFT+r'(?P<pm>±\s*)?(?P<num>'+_NUMBER+r')\s*(?P<unit>'+unit_names+r')(?![A-Za-z0-9_²³^])')
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
    tolerance=re.compile(_LEFT+r'±\s*(?P<num>'+_NUMBER+r')'+_RIGHT)
    def tolerance_read(m):
        if not re.search(r'(?:误差|容差|tolerance)[^。！？!?；;\n]{0,24}$',m.string[max(0,m.start()-45):m.start()],re.I):return m.group()
        if _math_adjacent(m.string,m.start(),m.end()):return m.group()
        l=lang(m);return('正负'if l=='zh'else'plus or minus ')+speak_number(m['num'],l)
    protected=tolerance.sub(tolerance_read,protected)
    return restore_identifiers(protected,mapping)
