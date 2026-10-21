"""Local language scope for bilingual TN, without acoustic cuts or translation.

Written entities are protected before splitting clauses and quoted/parenthetical
spans. Distributed grammar evidence chooses local number language; diagnostics
retain raw offsets. Pure-language requests keep the released frontend behavior.
"""
import re
from .lexicon_rules import apply_lexeme_rules, has_chinese_name_reading
from .special_text import _ADDRESSES, normalize_special_text
from .number_context import _DATE, _TIME, _PHONE, _NUMBERED_ID, _GPU_MODEL

_CJK = re.compile(r"[\u4e00-\u9fff]")
_EN_WORD = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
_ASCII_TOKEN = _EN_WORD
_EN_FUNCTION = frozenset(
    "a an the this that these those i you he she it we they me us them "
    "my your his her its our their is are was were be been being am "
    "has have had do does did can could may might must shall should will would "
    "of in on at to from for with without by as than into over under after before "
    "and or but if because while when where which who whose what how not".split()
)
_ZH_GRAMMAR = re.compile(r"我们|你们|他们|请|需要|允许|然后|但是|已经|正在|可以|应该|"
                         r"设为|设置|检查|确认|读取|查看|使用|运行|打开|关闭|"
                         r"[的了是为在把被将与及]")
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "no", "st", "vs", "etc", "e.g", "i.e"})
_DECIMAL = re.compile(r"(?<![A-Za-z0-9_])[+\-±]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+\-−‑]?\d+)?")
_MODEL = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:[.\-_‑][A-Za-z0-9]+)*(?![A-Za-z0-9_])")
_MEASURE = re.compile(r"(?<![A-Za-z0-9_.,])[+\-±]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:[kKMGTmun]?B(?:/s)?|[kKMm]?Hz|[kKmM]?V|[mµu]?A|[kKm]?W|dB|℃|°[CF]|%)(?![A-Za-z])")
_ZH_FIELD = re.compile(r"(?:内存(?:阈值)?|信噪比|电流|电压|功率|温度|误差|学习率|"
                       r"日期|时间|下单时间|订单(?:号|编号|\s*ID)?|编号|验证码|尾号|"
                       r"最后(?:四|4)位|后(?:四|4)位|客服电话|电话|容量|阈值)"
                       r"\s*(?:[A-Za-z]{1,5}\s*)?(?:设为|设置为|是|为|等于)?\s*[:：=≥≤<>]*\s*$", re.I)
_EN_FIELD = re.compile(r"\b(?:memory(?:\s+(?:limit|threshold))?|current|voltage|power|"
                       r"temperature|error|learning\s+rate|date|time|order\s+(?:id|number)|"
                       r"serial\s+number|verification\s+code|last\s+four\s+digits|phone|"
                       r"capacity|threshold)\s*(?:is|are|equals|set\s+to)?\s*[:=<>]*\s*$", re.I)


def _trim_address(text):
    value = text.rstrip(".,;!?")
    for close, opening in ((")", "("), ("]", "[")):
        while value.endswith(close) and value.count(close) > value.count(opening):
            value = value[:-1]
    return value


def protected_spans(text):
    """Nonoverlapping entities, always in original input coordinates."""
    found = []
    def add(a, b, kind):
        if a < b and not any(a < y and b > x for x, y, _ in found):
            found.append((a, b, kind))
    for match in _ADDRESSES.finditer(text):
        add(match.start(), match.start() + len(_trim_address(match.group())), "address")
    for kind, pattern in (("date", _DATE), ("time", _TIME), ("phone", _PHONE),
                          ("identifier", _NUMBERED_ID), ("model", _GPU_MODEL),
                          ("measurement", _MEASURE), ("number", _DECIMAL)):
        for match in pattern.finditer(text):
            add(match.start(), match.end(), kind)
    for match in _MODEL.finditer(text):
        if any(c.isdigit() for c in match.group()) or any(c in match.group() for c in "._-"):
            add(match.start(), match.end(), "identifier")
    return sorted(found)


def _protected(position, entities):
    return any(start <= position < end for start, end, _ in entities)


def _is_period_boundary(text, position):
    if position and position + 1 < len(text) and text[position-1].isdigit() and text[position+1].isdigit():
        return False
    previous = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", text[:position])
    if previous and (previous.group().lower() in _ABBREVIATIONS or len(previous.group()) == 1):
        return False
    return position + 1 == len(text) or not text[position+1].isascii() or not text[position+1].isalnum()


def _strong_spans(text, entities):
    start = 0
    for i, char in enumerate(text):
        if _protected(i, entities):
            continue
        if char in "。！？!?；;\n" or (char == "." and _is_period_boundary(text, i)):
            yield start, i + 1
            start = i + 1
    if start < len(text):
        yield start, len(text)


def _local_spans(text, start, end, entities):
    cursor = start
    for i in range(start, end):
        char = text[i]
        if _protected(i, entities):
            continue
        apostrophe = char in "'’" and i > 0 and i+1 < len(text) and text[i-1].isascii() and text[i-1].isalpha() and text[i+1].isascii() and text[i+1].isalpha()
        if char in ",，、":
            yield cursor, i + 1
            cursor = i + 1
        elif char in "()（）[]【】\"“”‘’「」『』'" and not apostrophe:
            if cursor < i:
                yield cursor, i
            yield i, i + 1
            cursor = i + 1
    if cursor < end:
        yield cursor, end


def _language(text, inherited="ZH"):
    chars = list(text)
    for start, end, kind in protected_spans(text):
        chars[start:end] = " " * (end-start)
    prose = "".join(chars)
    zh_count = len(_CJK.findall(prose))
    words = _EN_WORD.findall(prose)
    if not zh_count and not words:
        return inherited, "inherit-numeric-or-punctuation", {"zh": 0, "en": 0}
    if not zh_count:
        return "EN", "english-local-span", {"zh": 0, "en": len(words)}
    if not words:
        return "ZH", "chinese-local-span", {"zh": zh_count, "en": 0}
    grammatical = sum(w.lower() in _EN_FUNCTION for w in words)
    ordinary = sum(not (w.isupper() and len(w)>1) and not any(c.isupper() for c in w[1:]) for w in words)
    zh_grammar = len(_ZH_GRAMMAR.findall(prose))
    facts = {"zh": zh_count, "en": len(words), "en_grammar": grammatical,
             "en_ordinary": ordinary, "zh_grammar": zh_grammar}
    if grammatical >= 2 and ordinary >= 3 and grammatical >= zh_grammar:
        return "EN", "english-grammar-around-insertion", facts
    if zh_grammar >= 1 and zh_grammar >= grammatical:
        return "ZH", "chinese-grammar-around-terms", facts
    if ordinary >= 3 and (zh_count <= 4 or ordinary >= zh_count):
        return "EN", "english-prose-around-short-name", facts
    if grammatical and ordinary >= 2 and zh_count <= 4:
        return "EN", "english-grammar-around-short-name", facts
    if zh_count >= 4:
        return "ZH", "chinese-prose-with-english-terms", facts
    return inherited, "ambiguous-inherit-local-context", facts


def _embedded_ranges(text, entities):
    """Quoted/parenthesized ranges retain their existing local language.

    Ignore address punctuation and apostrophes inside English words. An
    unfinished quote remains protected to the end of the input.
    """
    pairs = {"(": ")", "（": "）", "[": "]", "【": "】", "“": "”",
             "‘": "’", "「": "」", "『": "』", '"': '"', "'": "'"}
    stack, ranges = [], []
    for index, char in enumerate(text):
        if _protected(index, entities):
            continue
        if (char in "'’" and index > 0
                and text[index - 1].isascii() and text[index - 1].isalpha()
                and ((index + 1 < len(text) and text[index + 1].isascii()
                      and text[index + 1].isalpha())
                     or not stack or char != stack[-1][1])):
            # English possessives such as users' cannot open a quotation.
            continue
        if stack and char == stack[-1][1]:
            start, _ = stack.pop()
            ranges.append((start, index + 1))
        elif char in pairs:
            stack.append((index, pairs[char]))
    ranges.extend((start, len(text)) for start, _ in stack)
    return ranges


def context_diagnostics(text):
    """Exact spans/evidence for review; no input text is rewritten here."""
    entities = protected_spans(text)
    embedded = _embedded_ranges(text, entities)
    overall, overall_reason, _ = _language(text)
    pieces = []
    for start, end in _strong_spans(text, entities):
        parent, parent_reason, _ = _language(text[start:end], overall)
        for a, b in _local_spans(text, start, end, entities):
            raw = text[a:b]
            language, reason, evidence = _language(raw, parent)
            # A Chinese enumeration mark keeps a product-list item in its
            # parent context. Only curated, complete names inherit the Chinese
            # parent here; real English clauses and embedded quotations keep
            # their local route. Offsets still refer to the untouched input.
            name = raw.strip(" \t\r\n,，、;；。.!?！？:：")
            in_list = raw.rstrip().endswith("、") or text[:a].rstrip().endswith("、")
            if (parent == "ZH" and in_list and has_chinese_name_reading(name)
                    and not any(a < end and b > start for start, end in embedded)
                    and not any(kind == "address" and a < end and b > start
                                for start, end, kind in entities)):
                language, reason = "ZH", "chinese-name-in-list"
            pieces.append({"start": a, "end": b, "text": raw, "language": language,
                           "reason": reason, "evidence": evidence,
                           "parent_language": parent, "parent_reason": parent_reason})
    if "".join(piece["text"] for piece in pieces) != text:
        raise ValueError("Mixed-language routing changed or omitted raw input")
    return {"overall_language": overall, "overall_reason": overall_reason,
            "protected": [{"start": a, "end": b, "text": text[a:b], "kind": k} for a,b,k in entities],
            "segments": pieces}


def _number_language(text, start, end, default):
    left = text[:start][-80:]
    zh, en = _ZH_FIELD.search(left), _EN_FIELD.search(left)
    if zh and (not en or zh.start() > en.start()):
        return "ZH"
    return "EN" if en else default


def normalize_mixed_context(text, normalize_original, normalize_plain, verbalize_number):
    if not (_CJK.search(text) and _EN_WORD.search(text)):
        return normalize_original(text)
    diagnostic = context_diagnostics(text)
    if not any(p["language"] == "EN" and _EN_WORD.search(p["text"]) for p in diagnostic["segments"]):
        return normalize_original(text)

    def english_ordinary(ordinary):
        ordinary = apply_lexeme_rules(ordinary, language="EN")
        parts, cursor = [], 0
        for match in _ASCII_TOKEN.finditer(ordinary):
            parts.append(normalize_plain(ordinary[cursor:match.start()]))
            token = match.group()
            spoken = normalize_plain(token)
            parts.append(token if _CJK.search(spoken) else spoken)
            cursor = match.end()
        parts.append(normalize_plain(ordinary[cursor:]))
        return " ".join(p.strip() for p in parts if p.strip())

    output = []
    for piece in diagnostic["segments"]:
        raw, language = piece["text"], piece["language"]
        if language == "EN":
            output.append(normalize_special_text(
                raw, english_ordinary, verbalize_number, language="EN",
                language_resolver=lambda a,b,default: _number_language(raw,a,b,default),
            ))
        else:
            output.append(normalize_original(raw))
    normalized = " ".join(part.strip() for part in output if part.strip())
    return re.sub(r"(?<=[\u4e00-\u9fff，。！？；：、,.!?;:]) +"
                  r"(?=[\u4e00-\u9fff，。！？；：、,.!?;:])", "", normalized)
