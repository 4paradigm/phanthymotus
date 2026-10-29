"""Validated, cached external pronunciation dictionary shared by both G2P paths."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

from .release_paths import frontend_lexicon_path
from .symbols import en_symbols
from .hotwords import bundled_hotwords_source, deployment_configuration_active, lookup_hotword

_MAX_BYTES = 4 * 1024 * 1024
_MAX_ENTRIES = 50000
_KEY = re.compile(r"[A-Z]+(?:['-][A-Z]+)*\Z")
_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ISOLATED_TERM = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:['-][A-Za-z0-9]+)*\Z")
_ISOLATED_PART = re.compile(r"[A-Za-z0-9]+(?:[ '-][A-Za-z0-9]+)*\Z")
_MAX_ISOLATED_ENTRIES = 256
_VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_CONSONANTS = set("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
_MODEL_PHONES = set(en_symbols)
_cmu_binding = None


def _status_copy(status):
    return {**{key: value for key, value in status.items() if not key.startswith("_")},
            "mode_counts": dict(status["mode_counts"])}


class _CMUBinding:
    """One immutable-release CMU identity, shared by cached resource loads."""

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.entries = len(dictionary)
        canonical = json.dumps(dictionary, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True).encode("utf-8")
        self.sha256 = hashlib.sha256(canonical).hexdigest()


def _check_cmu(entries, status, binding):
    if (binding.entries != status["cmu_entries"]
            or binding.sha256 != status["cmu_canonical_sha256"]):
        raise ValueError("External lexicon was prepared for another CMU dictionary")
    for key, (mode, phones) in entries.items():
        if key in binding.dictionary:
            released = tuple(phone for syllable in binding.dictionary[key] for phone in syllable)
            if mode != "cmu" or released != phones:
                raise ValueError("External lexicon CMU conflict repair does not match the release")
        elif mode == "cmu":
            raise ValueError("External lexicon declares a missing CMU entry")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("External lexicon contains duplicate JSON keys")
        value[key] = item
    return value


def _phones(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("External lexicon requires 1..64 flat ARPAbet phones")
    for phone in value:
        match = re.fullmatch(r"([A-Z]+)([012])?", phone) if isinstance(phone, str) else None
        if match is None:
            raise ValueError("External lexicon has an invalid ARPAbet phone")
        base, stress = match.groups()
        model_phone = "V" if base == "V" else base.lower()
        if (not ((base in _VOWELS and stress is not None)
                 or (base in _CONSONANTS and stress is None))
                or model_phone not in _MODEL_PHONES):
            raise ValueError("External lexicon phone is incompatible with the model")
    return tuple(value)


def _isolated_synthesis(value):
    """Validate optional term-level synthesis plans without changing words.

    The parts are text to synthesize independently, not phone replacements.
    Requiring the original spelling to be preserved prevents a malformed
    resource from silently changing or dropping user-supplied text.
    """
    if not isinstance(value, list) or len(value) > _MAX_ISOLATED_ENTRIES:
        raise ValueError("External lexicon isolated synthesis requires a bounded list")
    result, seen = [], set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"term", "mode", "parts"}:
            raise ValueError("External lexicon isolated synthesis entry is invalid")
        term, mode, parts = item["term"], item["mode"], item["parts"]
        if (not isinstance(term, str) or len(term) > 64 or not _ISOLATED_TERM.fullmatch(term)
                or mode not in ("exact", "fold") or not isinstance(parts, list)
                or not 1 <= len(parts) <= 8):
            raise ValueError("External lexicon isolated synthesis entry is invalid")
        if any(not isinstance(part, str) or len(part) > 128
               or not _ISOLATED_PART.fullmatch(part) for part in parts):
            raise ValueError("External lexicon isolated synthesis must preserve the term")
        joined = "".join(part.replace(" ", "") for part in parts)
        if (joined != term if mode == "exact" else joined.casefold() != term.casefold()):
            raise ValueError("External lexicon isolated synthesis must preserve the term")
        folded = term.casefold()
        if folded in seen:
            raise ValueError("External lexicon isolated synthesis contains duplicate terms")
        seen.add(folded)
        result.append((term, mode, tuple(parts)))
    return tuple(result)


@lru_cache(maxsize=8)
def _resolved_path(value):
    return str(Path(value).expanduser().resolve())


def _configuration():
    configured = os.environ.get("VITS2_LEXICON_PATH", "").strip()
    path = Path(configured).expanduser() if configured else frontend_lexicon_path()
    checksum = os.environ.get("VITS2_LEXICON_SHA256", "").strip().lower()
    if checksum and not _SHA.fullmatch(checksum):
        raise ValueError("VITS2_LEXICON_SHA256 requires 64 hexadecimal digits")
    return _resolved_path(str(path)), checksum


@lru_cache(maxsize=4)
def _load(path, checksum, binding=None):
    # Dictionary reads/validation happen once, never once per speech token.
    with open(path, "rb") as stream:
        payload = stream.read(_MAX_BYTES + 1)
    if len(payload) > _MAX_BYTES:
        raise ValueError("External lexicon exceeds 4 MiB")
    actual_sha = hashlib.sha256(payload).hexdigest()
    if checksum and actual_sha != checksum:
        raise ValueError("External lexicon SHA256 mismatch")
    data = json.loads(payload, object_pairs_hook=_unique_object)
    if (not isinstance(data, dict) or data.get("version") != 1
            or data.get("format") != "vits2-external-pronunciations"):
        raise ValueError("External lexicon has an unsupported format/version")
    entries, source, cmu = data.get("entries"), data.get("source"), data.get("cmu")
    if not isinstance(entries, dict) or len(entries) > _MAX_ENTRIES:
        raise ValueError("External lexicon requires at most 50000 entries")
    for name, metadata in (("source", source), ("cmu", cmu)):
        if (not isinstance(metadata, dict) or not _SHA.fullmatch(str(metadata.get("sha256", "")))
                or type(metadata.get("entries")) is not int or metadata["entries"] < 0):
            raise ValueError("External lexicon requires valid " + name + " provenance")
    if (cmu.get("canonical_layout") != "nested_syllables"
            or not _SHA.fullmatch(str(cmu.get("canonical_sha256", "")))):
        raise ValueError("External lexicon requires nested CMU canonical provenance")
    if source["entries"] != len(entries):
        raise ValueError("External lexicon source entry count mismatch")
    parsed, modes, supplement_keys = {}, {"cmu": 0, "exact": 0, "fold": 0}, set()
    for key, row in entries.items():
        if not isinstance(key, str) or not _KEY.fullmatch(key) or not isinstance(row, dict):
            raise ValueError("External lexicon requires complete uppercase Latin keys")
        mode = row.get("mode")
        if mode not in modes:
            raise ValueError("External lexicon has an unsupported matching mode")
        phones = _phones(row.get("arpa"))
        if "original_arpa" in row:
            _phones(row["original_arpa"])
        if "origin" in row:
            if row["origin"] != "supplement":
                raise ValueError("External lexicon has an unsupported entry origin")
            supplement_keys.add(key)
        parsed[key] = (mode, phones)
        modes[mode] += 1
    status = {"loaded": True, "path": path, "sha256": actual_sha,
              "source_sha256": source["sha256"], "cmu_sha256": cmu["sha256"],
              "cmu_canonical_sha256": cmu["canonical_sha256"],
              "cmu_entries": cmu["entries"], "entry_count": len(parsed),
              "mode_counts": modes, "cmu_verified": binding is not None,
              "_isolated_synthesis": _isolated_synthesis(data.get("isolated_synthesis", []))}
    supplements, base_source = data.get("supplements"), data.get("base_source")
    if supplements is not None or base_source is not None or supplement_keys:
        if (not isinstance(supplements, dict) or not isinstance(base_source, dict)
                or not _SHA.fullmatch(str(supplements.get("sha256", "")))
                or not _SHA.fullmatch(str(base_source.get("canonical_sha256", "")))
                or not _SHA.fullmatch(str(base_source.get("sha256", "")))
                or type(base_source.get("entries")) is not int
                or base_source["entries"] + len(supplement_keys) != len(entries)
                or type(supplements.get("added_entries")) is not int
                or supplements["added_entries"] != len(supplement_keys)
                or not isinstance(supplements.get("added_keys"), list)
                or len(supplements["added_keys"]) != len(supplement_keys)
                or any(not isinstance(key, str) for key in supplements["added_keys"])
                or set(supplements["added_keys"]) != supplement_keys):
            raise ValueError("Merged lexicon has invalid supplement provenance")
        maintained = {key: row.get("original_arpa", row["arpa"]) for key, row in entries.items()}
        base = {key: phones for key, phones in maintained.items() if key not in supplement_keys}
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=True).encode("utf-8")
        if (hashlib.sha256(canonical(base)).hexdigest() != base_source["canonical_sha256"]
                or hashlib.sha256(canonical(maintained)).hexdigest() != source.get("canonical_sha256")
                or hashlib.sha256(canonical(maintained) + b"\n").hexdigest() != source["sha256"]):
            raise ValueError("Merged lexicon does not match its maintained Gold source")
        status.update(supplements_sha256=supplements["sha256"],
                      base_entry_count=base_source["entries"],
                      supplement_entry_count=len(supplement_keys),
                      _supplement_keys=frozenset(supplement_keys))
    if binding is not None:
        _check_cmu(parsed, status, binding)
    return parsed, status


def reload_external_lexicon(path=None, checksum=""):
    """Reload against the active CMU; reject an incompatible replacement.

    Before English startup binds its CMU, this only clears preparation caches.
    A rejected active file keeps raising on lookup until corrected and reloaded;
    it must never silently become a G2P fallback. Resource preparation can pass
    its output explicitly before the default release path has been configured.
    """
    _load.cache_clear()
    _resolved_path.cache_clear()
    if _cmu_binding is not None:
        if path is None:
            selected, pin = _configuration()
            _load(selected, pin, _cmu_binding)
        else:
            validate_external_lexicon_file(path, checksum)


def external_lexicon_status():
    """Validate the configured resource and return caller-owned status metadata."""
    path, checksum = _configuration()
    _, status = _load(path, checksum, _cmu_binding)
    return _status_copy(status)


def isolated_synthesis_entries():
    """Return validated immutable (term, case mode, parts) plans, if present."""
    if not os.environ.get("VITS2_LEXICON_PATH", "").strip():
        from . import release_paths
        if release_paths._release_root is None:
            # Injected engines may run without a configured frontend release.
            # Only this absence disables the optional feature; a configured
            # missing or corrupt resource must still raise normally.
            return ()
    path, checksum = _configuration()
    _, status = _load(path, checksum, _cmu_binding)
    return status["_isolated_synthesis"]


def validate_external_lexicon_file(path, checksum=""):
    """Validate a supplied resource without changing process configuration."""
    checksum = str(checksum).strip().lower()
    if checksum and not _SHA.fullmatch(checksum):
        raise ValueError("External lexicon SHA256 requires 64 hexadecimal digits")
    _, status = _load(_resolved_path(str(path)), checksum, _cmu_binding)
    return _status_copy(status)


def validate_external_lexicon_cmu(cmu):
    """Check supplied CMU without changing the active frontend binding."""
    path, checksum = _configuration()
    _, status = _load(path, checksum, _CMUBinding(cmu))
    return _status_copy(status)


def bind_external_lexicon_cmu(cmu):
    """Bind English startup's actual eng_dict before any expanded lookup."""
    global _cmu_binding
    binding = _CMUBinding(cmu)
    path, checksum = _configuration()
    _, status = _load(path, checksum, binding)
    # A failed validation must not replace a previously usable CMU identity.
    _cmu_binding = binding
    return _status_copy(status)


def _lookup_entry(entries, status, token, has_cmu=False, base_only=False):
    if has_cmu or base_only and token.upper() in status.get("_supplement_keys", ()):
        return None
    row = entries.get(token.upper())
    if row is None:
        return None
    mode, phones = row
    if mode == "cmu" or mode == "exact" and token != token.upper():
        return None
    return [list(phones)]


def lookup_external_lexicon(token, *, has_cmu=False):
    """Return original-model-shaped phones; ordinary CMU words keep priority."""
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        return None
    path, checksum = _configuration()
    entries, status = _load(path, checksum, _cmu_binding)
    return _lookup_entry(entries, status, token, has_cmu)


def lookup_unified_lexicon(token, *, has_cmu=False):
    """One default Gold index; retain explicit-file/domain override semantics.

    The normal deployment already compiled reviewed additions into Gold. It
    does not read or scan their maintenance manifest while speaking. Older
    explicit dictionaries and opt-in domain overrides use indexed candidates,
    preserving their existing priority and CMU-override contracts.
    """
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        return None
    special = deployment_configuration_active()
    if has_cmu and not special:
        return None
    path, checksum = _configuration()
    entries, status = _load(path, checksum, _cmu_binding)
    if not special and status.get("supplements_sha256") == bundled_hotwords_source()[1]:
        return _lookup_entry(entries, status, token, has_cmu)
    selected = lookup_hotword(token, has_cmu=has_cmu)
    if selected is not None:
        return selected
    return _lookup_entry(entries, status, token, has_cmu, base_only=True)
