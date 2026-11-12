"""Output-only number rendering with small, lazily loaded OpenFst grammars."""

from collections import OrderedDict
import json
import logging
from pathlib import Path
import re
import unicodedata

from plugins.numeric_dictation import normalize_long_numeric_dictation

log = logging.getLogger(__name__)
_EN_DIGITS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))


class NumericOutputNormalizer:
    def __init__(self, grammar_dir, model_names=()):
        self.root = Path(grammar_dir)
        self.model_names = tuple(model_names)
        self._lexicon = None
        self._graphs = OrderedDict()
        self._patterns = {}
        self._cache = OrderedDict()
        self._warned = set()

    def _warn(self, language, error):
        if language not in self._warned:
            log.warning("[asr] numeric normalization unavailable for %s: %s; preserving text", language, error)
            self._warned.add(language)

    def _load(self):
        if self._lexicon is None:
            self._lexicon = json.loads((self.root / "lexicon.json").read_text())
        return self._lexicon

    def _rewrite(self, text, language):
        import kaldifst

        if language not in self._graphs:
            folder = self.root / language
            self._graphs[language] = tuple(kaldifst.StdVectorFst.read(str(folder / (name + ".fst")))
                                           for name in ("tagger", "verbalizer"))
            while len(self._graphs) > 2:
                self._graphs.popitem(last=False)
        self._graphs.move_to_end(language)
        result = text.lower()
        for graph in self._graphs[language]:
            path = kaldifst.compose(kaldifst.make_linear_acceptor(list(result.encode("utf-8"))), graph)
            path = kaldifst.shortest_path(path)
            ok, _, labels, _ = kaldifst.get_linear_symbol_sequence(path)
            if not ok or not labels:
                return text
            result = bytes(labels).decode("utf-8").strip()
        return text if "tokens {" in result or result.lower() == text.lower() else result

    @staticmethod
    def _name_pattern(name):
        chars = [c for c in name if not c.isspace() and c not in "-_"]
        return r"(?<![A-Za-z0-9])" + r"[\s_-]*".join(re.escape(c) for c in chars) + r"(?![A-Za-z0-9])"

    def _protect(self, text):
        protected = []

        def save(match):
            protected.append(match.group())
            number = len(protected)
            suffix = ""
            while number:
                number, digit = divmod(number - 1, 26)
                suffix = chr(97 + digit) + suffix
            return "zzasrprotected" + suffix + "zz"

        if self.model_names:
            pattern = "|".join(self._name_pattern(n) for n in sorted(self.model_names, key=len, reverse=True))
            text = re.sub(pattern, save, text, flags=re.I)
        text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]+\b|https?://\S+|"
                      r"\b(?=[A-Za-z0-9_.+-]*[0-9])(?=[A-Za-z0-9_.+-]*[A-Za-z])[A-Za-z0-9_][A-Za-z0-9_.+-]*\b",
                      save, text)
        return text, protected

    def _model_numbers(self, text):
        variants = []
        for name in self.model_names:
            if not any(c.isdigit() for c in name):
                continue
            spoken = re.sub(r"(?<=\d)\.(?=\d)", " point ", name)
            spoken = re.sub(r"\d", lambda m: " " + _EN_DIGITS[m.group()] + " ", spoken)
            spoken = " ".join(spoken.replace("-", " ").split())
            variants.append((spoken, name))
        for spoken, name in sorted(variants, key=lambda x: len(x[0]), reverse=True):
            text = re.sub(self._name_pattern(spoken), lambda _: name, text, flags=re.I)
        return text

    def _digit_runs(self, text, language):
        aliases = self._load()["digits"][language]
        mapping = {word: str(i) for i, words in enumerate(aliases) for word in words.split()}
        repeats = {"en": {"double": 2, "triple": 3}, "fr": {"double": 2, "triple": 3},
                   "de": {"doppelt": 2, "dreifach": 3}, "es": {"doble": 2, "triple": 3},
                   "ru": {"дважды": 2, "трижды": 3}}
        for prefix, count in repeats.get(language, {}).items():
            for word, value in list(mapping.items()):
                if len(value) == 1:
                    mapping[prefix + " " + word] = value * count
        atom = "(?:" + "|".join(re.escape(x) for x in sorted(mapping, key=len, reverse=True)) + r"|[0-9])"
        boundary = language not in ("zh", "ja", "th")
        sep = r"[\s,，、;；\-–—]*" if not boundary or language == "de" else r"[\s,，;；\-–—]+"
        pattern = (r"(?<!\w)" if boundary else "") + atom + "(?:" + sep + atom + "){1,}" + (r"(?!\w)" if boundary else "")

        def replace(match):
            tokens = re.findall(atom, match.group(), flags=re.I)
            values = [mapping.get(t.lower(), t) for t in tokens]
            if all(t.isdigit() for t in tokens):
                return match.group()
            result = "".join(values)
            return result if result.isdigit() and len(result) >= 7 else match.group()

        return re.sub(pattern, replace, text, flags=re.I)

    def _western(self, text, language):
        data = self._load()
        words = set(data["words"][language])
        numeric = words - set(data["extra"][language].split())
        if language not in self._patterns:
            atom = "(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + ")"
            if language == "de":
                atom = "(?:" + atom + ")+"
            atom = "(?:" + atom + r"|\d+(?:[.,]\d+)*)"
            self._patterns[language] = re.compile(r"(?<!\w)" + atom + r"(?:[\s\-]+" + atom + r")*(?!\w)", re.I)

        def replace(match):
            value = match.group()
            parts = value.split()
            before, after = [], []
            connectors = data["connectors"].get(language, [])
            while parts and parts[0].lower() in connectors:
                before.append(parts.pop(0))
            while parts and parts[-1].lower() in connectors:
                after.insert(0, parts.pop())
            core = " ".join(parts)
            if not core or not re.search(r"[^\W\d_]", core):
                return value
            if language != "de" and not any(w.lower() in numeric for w in re.findall(r"[^\W\d_]+", core)):
                return value
            if language == "fr":
                # French ASR often omits the hyphens expected by the grammar.
                # Join valid compounds, preserving lists such as "trois et cinq".
                core = re.sub(r"\bquatre\s+vingts?\b", "quatre-vingt", core, flags=re.I)
                core = re.sub(r"\bdix\s+(sept|huit|neuf)\b", r"dix-\1", core, flags=re.I)
                core = re.sub(r"\b(soixante|quatre-vingt)\s+(dix(?:-[a-z]+)?|onze|douze|treize|quatorze|quinze|seize)\b", r"\1-\2", core, flags=re.I)
                core = re.sub(r"\b(vingt|trente|quarante|cinquante|soixante|quatre-vingt|soixante-dix|quatre-vingt-dix)\s+(deux|trois|quatre|cinq|six|sept|huit|neuf)\b", r"\1-\2", core, flags=re.I)
            result = self._rewrite(core.replace("dreissig", "dreißig") if language == "de" else core, language)
            return " ".join(before + [result] + after)

        return self._patterns[language].sub(replace, text)

    def _cjk(self, text, language):
        chars = "零〇○一二三四五六七八九十百千万萬亿億兆两兩幺壹贰貳叁肆伍陆陸柒捌玖点點"
        digits = {char: str(i) for i, aliases in enumerate(
            ("零〇○", "一幺壹", "二两兩贰貳", "三叁", "四肆", "五伍", "六陆陸", "七柒", "八捌", "九玖"))
                  for char in aliases}
        atom = "[" + chars + "0-9]"
        pattern = r"(?:负|負|マイナス)?(?:百分之)?" + atom + r"+(?:\." + atom + r"+)?(?:パーセント|%)?"

        def replace(match):
            value = match.group()
            original = value
            if not any(c in chars for c in value):
                return value
            if len(value) == 1 and ((match.start() and text[match.start() - 1].isalnum()) or
                                    (match.end() < len(text) and text[match.end()].isalnum())):
                return value
            if not any(c in digits or c.isdecimal() for c in value):
                # Bare scales in expressions such as 数百万 are approximate words.
                return value
            sign = ""
            for prefix in ("负", "負", "マイナス"):
                if value.startswith(prefix):
                    sign, value = "-", value[len(prefix):]
                    break
            percent = value.startswith("百分之") or value.endswith(("パーセント", "%"))
            if value.startswith("百分之"):
                value = value[3:]
            value = re.sub(r"(?:パーセント|%)$", "", value)
            if value.startswith(("万", "萬", "亿", "億", "兆")):
                # Large units require a coefficient; 万一 is an ordinary word.
                return original
            if all(c in digits or c.isdecimal() for c in value):
                if (1 < len(value) < 7 and not any(c.isdecimal() or c in "零〇○" for c in value)
                        and not re.match(r"(?:年|月|日|号|號|番)", text[match.end():])):
                    # 一二线 and 百分之四五 are not unambiguously 12 and 45%.
                    return original
                result = "".join(digits.get(c, c) for c in value)
            else:
                whole = re.split(r"[点點.]", value, maxsplit=1)[0]
                nonzero = "".join(c for c, number in digits.items() if number != "0")
                if re.search("[" + nonzero + "]{2,}", whole):
                    # 一五一十 / 十三五 are not canonical cardinal numbers.
                    return original
                if language == "zh" and (
                        whole.startswith(("百", "千")) or
                        re.search("[百千万萬亿億兆][" + nonzero + "1-9]$", whole)):
                    # Compact spoken amounts such as 一千一 are ambiguous.
                    return original
                result = self._rewrite(value, language)
            return sign + result + ("%" if percent else "")

        return re.sub(pattern, replace, text)

    def _thai(self, text):
        digits = {a: str(i) for i, aliases in enumerate(self._load()["digits"]["th"]) for a in aliases.split()}
        scales = {"สิบ": 10, "ร้อย": 100, "พัน": 1000, "หมื่น": 10000, "แสน": 100000}
        components = sorted(set(digits) | set(scales) | {"ล้าน", "จุด"}, key=len, reverse=True)
        atom = "(?:" + "|".join(map(re.escape, components)) + r"|[0-9])"

        def integer(value):
            value = re.sub(r"\s+", "", value)
            if value.isdigit():
                return value
            tokens = re.findall(atom, value)
            if not tokens or "".join(tokens) != value:
                raise ValueError("Not a Thai number")
            if all(t in digits for t in tokens):
                return "".join(digits[t] for t in tokens)
            total, group, digit, previous = 0, 0, None, 1000000
            for token in tokens:
                if token in digits or token.isdigit():
                    if digit is not None:
                        raise ValueError("Ambiguous adjacent Thai digits")
                    digit = int(digits.get(token, token))
                elif token == "ล้าน":
                    if digit == 0 and not group and not total:
                        raise ValueError("Ambiguous zero before Thai scale")
                    coefficient = total + group + (digit or 0)
                    total = (coefficient or 1) * 1000000
                    group, digit, previous = 0, None, 1000000
                elif token in scales:
                    scale = scales[token]
                    if scale >= previous or digit == 0:
                        raise ValueError("Unordered Thai number scales")
                    group += (digit if digit is not None else 1) * scale
                    digit, previous = None, scale
                else:
                    raise ValueError("Not a Thai integer")
            return str(total + group + (digit or 0))

        def number(value):
            if value.count("จุด") == 1:
                whole, fraction = value.split("จุด")
                fraction = re.sub("|".join(map(re.escape, sorted(digits, key=len, reverse=True))), lambda m: digits[m.group()], fraction)
                fraction = re.sub(r"\s+", "", fraction)
                if not fraction.isdigit():
                    return value
                return integer(whole.strip()) + "." + fraction
            return integer(value)

        def replace(match):
            value = match.group()
            if value in digits or value in scales or value == "ล้าน":
                return value
            try:
                return number(value)
            except ValueError:
                return value

        text = re.sub(r"ร้อยละ\s*(" + atom + r"+(?:\s+" + atom + r"+)*)", lambda m: number(m.group(1)) + "%", text)
        return re.sub(atom + r"+(?:\s+" + atom + r"+)*", replace, text)

    @staticmethod
    def _literal(value, language):
        sign = value[0] if value[:1] in ("+", "-") else ""
        if sign:
            value = value[1:]
        if "," in value and "." in value:
            decimal = "," if value.rfind(",") > value.rfind(".") else "."
            grouping = "." if decimal == "," else ","
            value = value.replace(grouping, "").replace(decimal, ".")
        elif "," in value:
            if re.fullmatch(r"\d{1,3}(?:,\d{3})+", value) and (language in ("en", "zh", "ja", "th") or value.count(",") > 1):
                value = value.replace(",", "")
            else:
                value = value.replace(",", ".")
        elif "." in value and language in ("de", "es") and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value):
            value = value.replace(".", "")
        # Keep decimal precision: version 3.10 must not become version 3.1.
        return sign + value

    def normalize(self, text, language):
        key = (text, language)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if language not in ("en", "zh", "ja", "fr", "de", "es", "ru", "th") or not text:
            return text
        original = text
        try:
            data = self._load()
            text = normalize_long_numeric_dictation(text, language)
            if language != "zh":
                # Mixed-language dictation may spell digits in Chinese even
                # when the detected language is different.
                text = normalize_long_numeric_dictation(text, "zh")
            # Preserve prose orthography (notably Thai); normalize digits only.
            text = text.replace("−", "-")
            text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
            text = self._model_numbers(text)
            text, protected = self._protect(text)
            text = self._digit_runs(text, language)
            if language != "en":
                text = self._digit_runs(text, "en")
            text = self._cjk(text, language) if language in ("zh", "ja") else self._thai(text) if language == "th" else self._western(text, language)
            if language != "en":
                text = self._western(text, "en")
            text = re.sub(r"(?<![A-Za-z0-9_])[-+]?\d+(?:[.,]\d+)*(?![A-Za-z0-9_])", lambda m: self._literal(m.group(), language), text)
            percent = data["percent"][language]
            text = re.sub(r"(?<=\d)\s*" + percent + (r"\b" if language in ("en", "fr", "de", "es", "ru") else ""), "%", text, flags=re.I)
            text = re.sub(r"(?<=\d)\s+%", "%", text)
            for index, value in enumerate(protected, 1):
                suffix, number = "", index
                while number:
                    number, digit = divmod(number - 1, 26)
                    suffix = chr(97 + digit) + suffix
                text = text.replace("zzasrprotected" + suffix + "zz", value)
            core = text.strip().rstrip(".!?。！？").strip()
            if core.isdigit() and len(core) >= 7:
                text = core
        except (ImportError, OSError, RuntimeError, ValueError, UnicodeError, AttributeError) as error:
            self._warn(language, error)
            text = normalize_long_numeric_dictation(original, language)
        self._cache[key] = text
        while len(self._cache) > 256:
            self._cache.popitem(last=False)
        return text
