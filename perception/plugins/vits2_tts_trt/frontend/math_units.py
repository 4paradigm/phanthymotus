"""Spoken forms for explicit math/units; ordinary words and numbers stay intact."""

import re
from decimal import Decimal
from functools import lru_cache


# key: Chinese, singular English, plural English. Long units match first.
_UNITS = {
    "kg/m³": ("千克每立方米", "kilogram per cubic meter", "kilograms per cubic meter"),
    "km/h": ("千米每小时", "kilometer per hour", "kilometers per hour"),
    "m/s": ("米每秒", "meter per second", "meters per second"),
    "m/s²": ("米每二次方秒", "meter per second squared", "meters per second squared"),
    "m²": ("平方米", "square meter", "square meters"),
    "cm²": ("平方厘米", "square centimeter", "square centimeters"),
    "m³": ("立方米", "cubic meter", "cubic meters"),
    "kg": ("千克", "kilogram", "kilograms"),
    "g": ("克", "gram", "grams"),
    "mg": ("毫克", "milligram", "milligrams"),
    "km": ("千米", "kilometer", "kilometers"),
    "m": ("米", "meter", "meters"),
    "cm": ("厘米", "centimeter", "centimeters"),
    "mm": ("毫米", "millimeter", "millimeters"),
    "°C": ("摄氏度", "degree Celsius", "degrees Celsius"),
    "°F": ("华氏度", "degree Fahrenheit", "degrees Fahrenheit"),
    "%": ("百分比", "percent", "percent"),
    "GB": ("G B", "gigabyte", "gigabytes"),
    "MB": ("M B", "megabyte", "megabytes"),
    "KB": ("K B", "kilobyte", "kilobytes"),
    "A": ("安培", "ampere", "amperes"),
    "mA": ("毫安", "milliampere", "milliamperes"),
    "V": ("伏特", "volt", "volts"),
    "mV": ("毫伏", "millivolt", "millivolts"),
    "kV": ("千伏", "kilovolt", "kilovolts"),
    "W": ("瓦特", "watt", "watts"),
    "mW": ("毫瓦", "milliwatt", "milliwatts"),
    "kW": ("千瓦", "kilowatt", "kilowatts"),
    "dB": ("分贝", "decibel", "decibels"),
}
_NUMBER = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_DASH = r"\-‐‑−﹣－"
_NORMALIZE_DASH = str.maketrans({char: "-" for char in "‐‑−﹣－"})
_UNIT_FORMS = sorted(_UNITS, key=len, reverse=True)
_UNIT_PATTERN = "|".join(re.escape(unit).replace(r"/", r"\s*/\s*") for unit in _UNIT_FORMS)
_UNIT_EXPR = rf"(?:{_UNIT_PATTERN})|℃|℉|(?i:degrees?\s+(?:Celsius|Fahrenheit))"
_UNIT_END = r"(?![A-Za-z0-9_/²³^]|\.\d)"
_COMPARATOR_PATTERN = r"<=|>=|!=|[=<>≤≥≠]"
_MEASURE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<uncertainty>±\s*)?(?P<number>[+-]?{_NUMBER})\s*"
    rf"(?P<unit>{_UNIT_EXPR}){_UNIT_END}"
)
_SCIENTIFIC = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<mantissa>[+{_DASH}]?{_NUMBER})[eE]"
    rf"(?P<exponent>[+{_DASH}]?\d+)(?![0-9_]|\.\d)"
    rf"(?:\s*(?P<unit>{_UNIT_EXPR}){_UNIT_END}|(?![A-Za-z0-9_/²³^]))"
)
_SCIENTIFIC_CORE = re.compile(
    rf"(?<![A-Za-z0-9_.])[+-]?{_NUMBER}[eE][+-]?\d+(?![0-9_]|\.\d)"
)
_TEMPERATURE_RANGE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<low>[+-]?{_NUMBER})\s*(?P<low_unit>℃|℉|°C|°F)"
    rf"\s*(?:～|~|–|—|-|(?i:to))\s*"
    rf"(?P<high>[+-]?{_NUMBER})\s*(?P<high_unit>℃|℉|°C|°F){_UNIT_END}"
)
_FRACTION = re.compile(
    r"(?<![A-Za-z0-9_./])(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)"
    r"(?![A-Za-z0-9_/]|\.\d)"
)
_COEFFICIENT = rf"{_NUMBER}[A-Za-z]"
_ATOM = rf"(?<![A-Za-z0-9_.])(?:{_COEFFICIENT}|{_NUMBER}|[A-Za-z])(?![A-Za-z0-9_]|\.\d)|π"
_TOKEN = rf"(?:{_ATOM}|<=|>=|!=|[+\-*×/÷^=<>≤≥≠²³()])"
_CANDIDATE = re.compile(rf"{_TOKEN}(?:\s*{_TOKEN})*")
_TOKENIZER = re.compile(_TOKEN)
_PRESERVE = re.compile(
    rf"(?<![A-Za-z0-9_])\d{{4}}[{_DASH}/]\d{{1,2}}[{_DASH}/]\d{{1,2}}(?![A-Za-z0-9_])|"
    r"\b\d[\d-]*-[A-Za-z]{2}[A-Za-z0-9.-]*"
)
_OPERATORS = {
    "+": ("加", "plus"), "-": ("减", "minus"),
    "*": ("乘以", "times"), "×": ("乘以", "times"),
    "/": ("除以", "divided by"), "÷": ("除以", "divided by"),
    "=": ("等于", "equals"), "<": ("小于", "less than"),
    ">": ("大于", "greater than"), "<=": ("小于等于", "less than or equal to"),
    ">=": ("大于等于", "greater than or equal to"),
    "≤": ("小于等于", "less than or equal to"),
    "≥": ("大于等于", "greater than or equal to"),
    "≠": ("不等于", "not equal to"), "!=": ("不等于", "not equal to"),
    "(": ("左括号", "open parenthesis"), ")": ("右括号", "close parenthesis"),
    "π": ("派", "pi"), "²": ("的平方", "squared"), "³": ("的立方", "cubed"),
}


@lru_cache(maxsize=1)
def _english_cardinals():
    # inflect is already part of the released English frontend environment.
    import inflect
    return inflect.engine()


def _english_quantity(value):
    """Read explicit measurements/math as quantities, never as calendar years."""
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = ("plus " if value[0] == "+" else "minus "), value[1:]
    whole, dot, fraction = value.partition(".")
    spoken = _english_cardinals().number_to_words(int(whole or "0"), andword="").replace(",", "")
    if dot:
        digits = "zero one two three four five six seven eight nine".split()
        spoken += " point " + " ".join(digits[int(char)] for char in fraction)
    return sign + spoken


def _english_fraction(numerator, denominator):
    """Use natural fraction nouns; arithmetic slash is handled separately."""
    count, divisor = int(numerator), int(denominator)
    number = _english_quantity(numerator)
    if divisor < 2:
        return f"{number} over {_english_quantity(denominator)}"
    if divisor == 2:
        noun = "half" if count == 1 else "halves"
    elif divisor == 4:
        noun = "quarter" if count == 1 else "quarters"
    else:
        engine = _english_cardinals()
        noun = engine.number_to_words(engine.ordinal(divisor), andword="").replace(",", "")
        if re.fullmatch(r"10+", str(divisor)) and noun.startswith("one "):
            noun = noun[4:]  # one hundredth, two hundredths (not two one hundredths)
        if count != 1:
            noun = engine.plural(noun)
    return f"{number} {noun}"


def _valid_math(tokens):
    """Validate token order, returning implicit multiplication positions."""
    expect_operand, depth, explicit = True, 0, False
    implicit = set()
    for index, token in enumerate(tokens):
        value = token.group()
        atom = bool(re.fullmatch(rf"{_COEFFICIENT}|{_NUMBER}|[A-Za-zπ]", value))
        if expect_operand:
            if value == "(":
                depth += 1
            elif value in {"+", "-"}:
                pass  # Unary signs, spoken distinctly from binary subtraction.
            elif atom:
                expect_operand = False
            else:
                return None
        elif value == ")":
            depth -= 1
            if depth < 0:
                return None
        elif value in {"²", "³"}:
            explicit = True
        elif value in _OPERATORS and value not in {"(", "π"} or value == "^":
            explicit, expect_operand = True, True
        elif value == "(" or (
            atom and index and tokens[index - 1].end() == token.start()
            and (tokens[index - 1].group() == "π" or value == "π")
        ):
            implicit.add(index)
            explicit = True
            if value == "(":
                depth += 1
                expect_operand = True
        else:
            return None
    return implicit if explicit and not expect_operand and depth == 0 else None


def _speak_math(tokens, implicit, language, verbalize_number):
    zh = language.lower().startswith("zh")
    words, index, expect_operand = [], 0, True
    while index < len(tokens):
        value = tokens[index].group()
        if index in implicit:
            words.append("乘以" if zh else "times")
        if value == "^":
            exponent = tokens[index + 1].group()
            if exponent in {"2", "3"}:
                words.append(_OPERATORS["²" if exponent == "2" else "³"][not zh])
                index += 2
                continue
            end = index + 2
            if exponent == "(":
                depth = 1
                while depth:
                    depth += (tokens[end].group() == "(") - (tokens[end].group() == ")")
                    end += 1
            elif exponent in {"+", "-"}:
                end += 1
            part = tokens[index + 1:end]
            spoken = _speak_math(part, set(), language, verbalize_number)
            words.append(f"的 {spoken} 次方" if zh else f"to the power of {spoken}")
            index = end
            continue
        if re.fullmatch(_NUMBER, value):
            words.append(verbalize_number(value, language))
        elif re.fullmatch(_COEFFICIENT, value):
            words.extend((verbalize_number(value[:-1], language),
                          "乘以" if zh else "times", value[-1]))
        elif value in {"+", "-"} and expect_operand:
            words.append(("正" if value == "+" else "负") if zh else
                         ("positive" if value == "+" else "negative"))
        else:
            words.append(_OPERATORS[value][not zh] if value in _OPERATORS else value)
        expect_operand = value in _OPERATORS and value not in {"π", "²", "³", ")"}
        index += 1
    return " ".join(words)


def _canonical_unit(unit):
    unit = re.sub(r"\s+", "", unit)
    if unit in {"℃", "℉"} or "celsius" in unit.lower() or "fahrenheit" in unit.lower():
        return "°C" if unit == "℃" or "celsius" in unit.lower() else "°F"
    return unit


def _unit_label(unit, value, zh):
    return _UNITS[_canonical_unit(unit)][0 if zh else (1 if abs(Decimal(value.replace(",", ""))) == 1 else 2)]


def _measure_spoken(value, unit, language, verbalize_number):
    zh = language.lower().startswith("zh")
    number = verbalize_number(value, language)
    label = _unit_label(unit, value, zh)
    return f"百分之 {number}" if zh and unit == "%" else f"{number} {label}"


def _comparison_prefix(text, start, spoken, zh):
    """Include a directly preceding comparison without consuming its label."""
    prefix = re.search(rf"(?P<op>{_COMPARATOR_PATTERN})\s*$", text[:start])
    if prefix:
        return prefix.start(), _OPERATORS[prefix["op"]][not zh] + " " + spoken
    return start, spoken


def _fraction_context(text, match):
    """Numeric a/b is a fraction unless a local arithmetic context is explicit."""
    left = re.split(r"[。！？!?，,；;\n]", text[:match.start()])[-1][-48:]
    right = text[match.end():]
    if re.search(r"算式|计算|除法|除以|(?i:\b(?:calculate|compute|divide|division|equation)\b)", left):
        return False
    # A slash within an equation or a compound expression remains an operator.
    if re.search(r"[=+\-*×÷^<>≤≥≠(]\s*$", left) or re.match(r"\s*[=+*×÷^<>≤≥≠)]", right):
        return False
    if re.search(r"\d\s*-\s*$", left) or re.match(r"\s*-\s*\d", right):
        return False
    return True


def find_spans(text, language, verbalize_number):
    """Return nonoverlapping ``(start, end, spoken)`` spans; URLs are excluded upstream.

    ``verbalize_number(value, language)`` receives the original numeric spelling,
    including decimal zeroes. Language can be ``ZH``/``EN`` or ``zh``/``en``.
    """
    # Retain original offsets, including grouping commas, while passing a
    # canonical numeric value to the existing language-specific number reader.
    original_number_reader = verbalize_number
    def verbalize_number(value, number_language):
        value = value.replace(",", "")
        if number_language.lower().startswith("en"):
            return _english_quantity(value)
        return original_number_reader(value, number_language)

    zh = language.lower().startswith("zh")
    # Translation is length preserving, so offsets still refer to the input.
    text = text.translate(_NORMALIZE_DASH)
    spans = []
    preserved = [(match.start(), match.end()) for match in _PRESERVE.finditer(text)]
    scientific_cores = [(match.start(), match.end()) for match in _SCIENTIFIC_CORE.finditer(text)]

    def overlaps(start, end):
        return any(start < b and end > a for a, b in preserved) or any(
            start < b and end > a for a, b, _ in spans)

    for match in _TEMPERATURE_RANGE.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        low = _measure_spoken(match["low"], match["low_unit"], language, verbalize_number)
        high = _measure_spoken(match["high"], match["high_unit"], language, verbalize_number)
        spoken = f"{low} {'到' if zh else 'to'} {high}"
        spans.append((match.start(), match.end(), spoken))

    for match in _SCIENTIFIC.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        mantissa, exponent = match["mantissa"], match["exponent"]
        base = verbalize_number(mantissa, language)
        exponent_reading = verbalize_number(exponent, language)
        spoken = (f"{base} 乘以 十的 {exponent_reading} 次方" if zh else
                  f"{base} times ten to the power of {exponent_reading}")
        if match["unit"]:
            if zh and match["unit"] == "%":
                spoken = "百分之 " + spoken
            else:
                spoken += " " + _unit_label(match["unit"], f"{mantissa}e{exponent}", zh)
        start, spoken = _comparison_prefix(text, match.start(), spoken, zh)
        if not overlaps(start, match.end()):
            spans.append((start, match.end(), spoken))

    for match in _MEASURE.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        value = match["number"]
        spoken = _measure_spoken(value, match["unit"], language, verbalize_number)
        if match["uncertainty"]:
            spoken = ("正负 " if zh else "plus or minus ") + spoken
        start, spoken = _comparison_prefix(text, match.start(), spoken, zh)
        if not overlaps(start, match.end()):
            spans.append((start, match.end(), spoken))

    for match in _FRACTION.finditer(text):
        if overlaps(match.start(), match.end()) or not _fraction_context(text, match):
            continue
        numerator = verbalize_number(match["numerator"], language)
        denominator = verbalize_number(match["denominator"], language)
        spoken = (f"{denominator} 分之 {numerator}" if zh else
                  _english_fraction(match["numerator"], match["denominator"]))
        spans.append((match.start(), match.end(), spoken))
    for candidate in _CANDIDATE.finditer(text):
        if any(candidate.start() < end and candidate.end() > start for start, end in preserved):
            continue
        # Never reinterpret an unsupported scientific quantity such as
        # 1e-4kg/day as the incomplete formula "1 times e minus 4".
        if any(candidate.start() < end and candidate.end() > start
               for start, end in scientific_cores):
            continue
        if re.fullmatch(r"[A-Za-z]\s*/\s*[A-Za-z]", candidate.group().strip()):
            continue  # I/O and similar notation are not unambiguous division.
        tokens = list(_TOKENIZER.finditer(text, candidate.start(), candidate.end()))
        # Leave unusually long expressions to the existing frontend. Bounding
        # the parser prevents quadratic/cubic prefix searches on long input.
        if len(tokens) > 64:
            continue
        start = 0
        while start < len(tokens):
            for end in range(len(tokens), start + 1, -1):
                part = tokens[start:end]
                left, right = part[0].start(), part[-1].end()
                if any(left < b and right > a for a, b, _ in spans):
                    continue
                implicit = _valid_math(part)
                if implicit is not None:
                    spans.append((left, right, _speak_math(part, implicit, language, verbalize_number)))
                    start = end
                    break
            else:
                start += 1
    return sorted(spans)
