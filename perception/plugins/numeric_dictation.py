"""Render standalone long numeric dictation without rewriting prose or names."""

import re

_SMALL = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20),
))
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
_SCALES = {"thousand": 1000, "million": 10**6, "billion": 10**9, "trillion": 10**12}
_DIGITS = {k: str(v) for k, v in _SMALL.items() if v < 10}
_DIGITS.update(oh="0", o="0")


def _digit_sequence(words):
    result = []
    index = 0
    while index < len(words):
        count = {"double": 2, "triple": 3}.get(words[index], 1)
        if count > 1:
            index += 1
        if index == len(words) or words[index] not in _DIGITS:
            return None
        result.append(_DIGITS[words[index]] * count)
        index += 1
    return "".join(result)


def _under_thousand(words):
    if not words:
        return None
    value = 0
    if len(words) >= 2 and words[0] in _SMALL and 1 <= _SMALL[words[0]] <= 9 and words[1] == "hundred":
        value = _SMALL[words[0]] * 100
        words = words[2:]
        if words and words[0] == "and":
            words = words[1:]
            if not words:
                return None
        if not words:
            return value
    if len(words) == 1 and words[0] in _SMALL:
        return value + _SMALL[words[0]]
    if words[0] in _TENS and len(words) <= 2:
        if len(words) == 1:
            return value + _TENS[words[0]]
        if words[1] in _SMALL and 1 <= _SMALL[words[1]] <= 9:
            return value + _TENS[words[0]] + _SMALL[words[1]]
    return None


def _cardinal(words):
    total, begin, previous_scale = 0, 0, float("inf")
    for index, word in enumerate(words):
        if word not in _SCALES:
            continue
        scale = _SCALES[word]
        group = _under_thousand(words[begin:index])
        if group is None or group == 0 or scale >= previous_scale:
            return None
        total += group * scale
        begin, previous_scale = index + 1, scale
    tail = words[begin:]
    if total and tail and tail[0] == "and":
        tail = tail[1:]
        if not tail:
            return None
    remainder = _under_thousand(tail) if tail else 0
    return str(total + remainder) if remainder is not None else None


def normalize_long_numeric_dictation(text, language):
    """Convert numeric-only utterances of at least seven digits.

    These are long-number/phone dictation, where written digits are useful.
    Ordinary sentences, short answers and alphanumeric model names stay intact.
    """
    core = text.strip().rstrip(".!?。！？").strip().lower()
    value = None
    if language == "en" and re.fullmatch(r"[a-z\s,\-]+", core):
        words = re.findall(r"[a-z]+", core)
        if words:
            value = _digit_sequence(words)
            if value is None:
                value = _cardinal(words)
    elif language == "zh":
        digits = dict(zip("零一二三四五六七八九", "0123456789"))
        digits.update({"幺": "1", "〇": "0"})
        sequence = re.sub(r"[\s,，、]", "", core)
        if sequence and all(char in digits or char in "0123456789" for char in sequence):
            value = "".join(digits.get(char, char) for char in sequence)
    return value if value is not None and len(value) >= 7 else text
