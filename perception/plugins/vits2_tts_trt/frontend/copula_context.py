"""Immutable reading context for 为 in normalized numerical predicates.

This recognizes complete spoken values, not arbitrary Chinese semantics. It
never rewrites text or depends on jieba, a previous request, or a global cache.
The spans also keep the evidence needed for this decision together in chunks.
"""
from dataclasses import dataclass
import re

from .math_units import _UNITS


@dataclass(frozen=True)
class CopulaSpan:
    start: int
    end: int
    wei: int
    tone2: bool


_HWS = r"[^\S\r\n\v\f\x85\u2028\u2029]*"
_WORD = r"[A-Za-z]+(?:[-'][A-Za-z]+)*"
_ZH_DIGITS = r"[零〇一二两三四五六七八九十百千万亿]+"
_ZH_CORE = rf"{_ZH_DIGITS}(?:点[零〇一二两三四五六七八九]+)?"
_ZH_NUMBER = rf"(?:正负|负|正)?{_ZH_CORE}(?:分之{_ZH_CORE})?"
_EN_DIGIT = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_EN_CARDINAL_WORD = (
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion)"
)
_EN_CORE = rf"{_EN_CARDINAL_WORD}(?:[ -]+(?:and[ ]+)?{_EN_CARDINAL_WORD})*"
_EN_FRACTION = r"(?:halves|half|thirds?|quarters?|fourths?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?)"
_EN_NUMBER = (
    rf"(?:(?:plus or minus|minus|negative|plus|positive)[ ]+)?{_EN_CORE}"
    rf"(?:[ ]+point(?:[ ]+{_EN_DIGIT})+|[ ]+{_EN_FRACTION})?"
    rf"(?:[ ]+times[ ]+ten[ ]+to[ ]+the[ ]+power[ ]+of[ ]+(?:minus[ ]+|plus[ ]+)?{_EN_CORE})?"
    r"(?![A-Za-z0-9_])"
)
_ARABIC = r"[+\-±]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9_]|\.\d)"
_SCALAR = re.compile(rf"(?:{_ZH_NUMBER}|{_EN_NUMBER}|{_ARABIC})", re.IGNORECASE)
_spoken_units = {form for forms in _UNITS.values() for form in forms}
_spoken_units.update({
    '元', '美元', '欧元', '英镑', '人民币', 'yuan', 'dollar', 'dollars',
    'euro', 'euros', 'pound', 'pounds', 'cent', 'cents',
    '秒', '毫秒', '微秒', '分钟', '小时', 'second', 'seconds', 'minute', 'minutes', 'hour', 'hours',
})
_UNIT = re.compile(
    rf"{_HWS}(?:" + '|'.join(re.escape(u).replace(r'\ ', _HWS) for u in sorted(_spoken_units, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])", re.IGNORECASE,
)
_RANGE = re.compile(rf"{_HWS}(?:到|至|to\b){_HWS}", re.IGNORECASE)
_MONEY_TAIL = re.compile(rf"[ ]+and[ ]+{_EN_CORE}[ ]+cents?\b", re.IGNORECASE)
_CLAUSE_END = re.compile(r"[，,。.!！?？;；:：\n\r\v\f\x85\u2028\u2029]")
_VALUE_END = re.compile(r"(?:左右|以上|以下|之间|时|则|并|且|然后|而|\b(?:and|or|then|so|while|but|before|after|when)\b)", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"(?:设置|设定|指定|定义|调整|设|改)$")
_FIELD_END = re.compile(r"(?:参数|数值|学习率|值|率|模式|语言|配置|选项|名称)$")
# Semantic field suffixes, not jieba words or sentence-specific pinyin entries.
# Suffix matching covers modifiers such as 额定电流 and 项目编号 without
# changing segmentation. A numeric object alone does not make a preposition a copula.
_MEASUREMENT_END = re.compile(
    r"(?:电流|电压|阈值|误差|残差|范围|售价|价格|金额|预算|结果|温度|编号|"
    r"数值|比值|值|比例|概率|数量|长度|宽度|高度|速度|频率|功率|电阻|率)(?:不)?$"
)


def _value_end(text, start):
    match = _SCALAR.match(text, start)
    if not match:
        return None
    end = match.end()
    unit = _UNIT.match(text, end)
    if unit:
        end = unit.end()
        cents = _MONEY_TAIL.match(text, end)
        if cents:
            end = cents.end()
    separator = _RANGE.match(text, end)
    if separator:
        other = _SCALAR.match(text, separator.end())
        if other:
            end = other.end()
            unit = _UNIT.match(text, end)
            if unit:
                end = unit.end()
    return end


def _assignment_start(text, wei):
    """Accept explicit set-to syntax, not the nominal 设置/人设 in prose."""
    left = text[:wei]
    verb = _ASSIGNMENT.search(left)
    if not verb:
        return None
    clause_start = max((m.end() for m in _CLAUSE_END.finditer(left)), default=0)
    prefix = left[clause_start:verb.start()]
    if prefix.endswith('请'):
        return verb.start() - 1
    obj = re.search(r'(?:把|将)[^，,。.!！?？;；:：\n\r]+$', prefix)
    if obj:
        return clause_start + obj.start()
    field = _FIELD_END.search(prefix)
    if field:
        return clause_start + field.start()
    return None


def find_copula_spans(text):
    """Return context/guard spans; only ``tone2`` spans change readings.

    A rejected numeric prefix keeps its next word as a guard too: splitting
    ``为 One | Piece`` must not turn a name into a scalar in the next G2P call.
    """
    if '为' not in text:
        return ()
    spans = []
    for match in re.finditer('为', text):
        wei = match.start()
        start = max(0, wei - 1)  # retain 因 / 不 immediately before 为
        right = re.match(_HWS, text[wei + 1:]).end() + wei + 1
        end = _value_end(text, right)
        if end is not None:
            tail = re.match(_HWS, text[end:]).end() + end
            complete = tail == len(text) or bool(_CLAUSE_END.match(text, tail) or _VALUE_END.match(text, tail))
            field = _MEASUREMENT_END.search(text[:wei])
            assignment_start = _assignment_start(text, wei)
            tone2 = complete and (field is not None or assignment_start is not None)
            if field is not None:
                start = field.start()
            elif assignment_start is not None:
                start = assignment_start
            if not complete:
                # Keep the disqualifying continuation with its numeric prefix.
                guard = re.match(_WORD + r'|[\u4e00-\u9fff]', text[tail:])
                if guard:
                    end = tail + guard.end()
            spans.append(CopulaSpan(start, end, wei, tone2))
            continue
        assignment_start = _assignment_start(text, wei)
        if assignment_start is None or right == len(text):
            continue
        value = re.match(_WORD + r'|[\u4e00-\u9fff]', text[right:])
        if value:
            spans.append(CopulaSpan(assignment_start, right + value.end(), wei, True))
    return tuple(spans)


def wei2_positions(text):
    return frozenset(span.wei for span in find_copula_spans(text) if span.tone2)
