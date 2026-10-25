"""Read an unsupported numeric shape without repairing or dropping symbols."""

_DIGITS = {"ZH": tuple("零一二三四五六七八九"),
           "EN": "zero one two three four five six seven eight nine".split()}
_SYMBOLS = {
    "ZH": {"/": "斜杠", ".": "点", ",": "逗号", "-": "连字符", "+": "加号",
           "−": "减号", "$": "美元符号", "£": "英镑符号", "€": "欧元符号",
           "¥": "元符号", "￥": "元符号"},
    "EN": {"/": "slash", ".": "dot", ",": "comma", "-": "hyphen", "+": "plus sign",
           "−": "minus sign", "$": "dollar sign", "£": "pound sign", "€": "euro sign",
           "¥": "yuan sign", "￥": "yuan sign"},
}
_WIDTH = {code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)}
_WIDTH[0x3000] = ord(" ")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–﹣－"), "-")


def fold_numeric_width(text):
    """Fold ASCII width one-to-one so diagnostic offsets stay unchanged."""
    return text.translate(_WIDTH)


def read_numeric_literal(text, language):
    """Preserve every written digit and internal separator in the chosen language."""
    language = "ZH" if language.lower().startswith("zh") else "EN"
    words = []
    for char in fold_numeric_width(text).translate(_DASHES):
        if char.isspace():
            continue
        words.append(_DIGITS[language][int(char)] if char.isdecimal()
                     else _SYMBOLS[language].get(char, char))
    return ("" if language == "ZH" else " ").join(words)
