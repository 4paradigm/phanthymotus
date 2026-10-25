"""Validated, cached external pronunciation dictionary shared by both G2P paths."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

from .release_paths import frontend_data_dir
from .symbols import en_symbols

_MAX_BYTES = 4 * 1024 * 1024
_MAX_ENTRIES = 50000
_KEY = re.compile(r"[A-Z]+(?:['-][A-Z]+)*\Z")
_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_CONSONANTS = set("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
_MODEL_PHONES = set(en_symbols)
_cmu_binding = None


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


@lru_cache(maxsize=8)
def _resolved_path(value):
    return str(Path(value).expanduser().resolve())


def _configuration():
    configured = os.environ.get("VITS2_LEXICON_PATH", "").strip()
    path = Path(configured).expanduser() if configured else frontend_data_dir() / "external_lexicon.json"
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
    parsed, modes = {}, {"cmu": 0, "exact": 0, "fold": 0}
    for key, row in entries.items():
        if not isinstance(key, str) or not _KEY.fullmatch(key) or not isinstance(row, dict):
            raise ValueError("External lexicon requires complete uppercase Latin keys")
        mode = row.get("mode")
        if mode not in modes:
            raise ValueError("External lexicon has an unsupported matching mode")
        phones = _phones(row.get("arpa"))
        if "original_arpa" in row:
            _phones(row["original_arpa"])
        parsed[key] = (mode, phones)
        modes[mode] += 1
    status = {"loaded": True, "path": path, "sha256": actual_sha,
              "source_sha256": source["sha256"], "cmu_sha256": cmu["sha256"],
              "cmu_canonical_sha256": cmu["canonical_sha256"],
              "cmu_entries": cmu["entries"], "entry_count": len(parsed),
              "mode_counts": modes, "cmu_verified": binding is not None}
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
    return {**status, "mode_counts": dict(status["mode_counts"])}


def validate_external_lexicon_file(path, checksum=""):
    """Validate a supplied resource without changing process configuration."""
    checksum = str(checksum).strip().lower()
    if checksum and not _SHA.fullmatch(checksum):
        raise ValueError("External lexicon SHA256 requires 64 hexadecimal digits")
    _, status = _load(_resolved_path(str(path)), checksum, _cmu_binding)
    return {**status, "mode_counts": dict(status["mode_counts"])}


def validate_external_lexicon_cmu(cmu):
    """Check supplied CMU without changing the active frontend binding."""
    path, checksum = _configuration()
    _, status = _load(path, checksum, _CMUBinding(cmu))
    return {**status, "mode_counts": dict(status["mode_counts"])}


def bind_external_lexicon_cmu(cmu):
    """Bind English startup's actual eng_dict before any expanded lookup."""
    global _cmu_binding
    binding = _CMUBinding(cmu)
    path, checksum = _configuration()
    _, status = _load(path, checksum, binding)
    # A failed validation must not replace a previously usable CMU identity.
    _cmu_binding = binding
    return {**status, "mode_counts": dict(status["mode_counts"])}


def lookup_external_lexicon(token, *, has_cmu=False):
    """Return original-model-shaped phones; ordinary CMU words keep priority."""
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        return None
    path, checksum = _configuration()
    entries, _ = _load(path, checksum, _cmu_binding)
    row = entries.get(token.upper())
    if row is None or has_cmu:
        return None
    mode, phones = row
    if mode == "cmu" or mode == "exact" and token != token.upper():
        return None
    return [list(phones)]
