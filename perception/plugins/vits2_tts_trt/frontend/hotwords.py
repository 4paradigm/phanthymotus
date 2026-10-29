"""External reviewed business supplements plus optional deployment overrides.

The verified ModelScope lexicon includes the reviewed business source records.
VITS2_HOTWORDS_PATH adds a local manifest; VITS2_HOTWORDS_SHA256 is optional.
The pinned bundle allows 128 KiB of provenance; deployment manifests keep the
64 KiB limit. Their combined active vocabulary still cannot exceed 128 entries.
VITS2_HOTWORD_DOMAIN selects a deployment domain, defaulting to ``general``.
Manifests stay fixed until the path/SHA changes or reload_hotwords() is called.
English tokens in Chinese speech still use language="en". This does not infer
the sentence's language, translate text, or match substrings/whole sentences.
"""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

from .symbols import en_symbols
from .release_paths import frontend_lexicon_path

_MAX_BYTES = 64 * 1024
_BUSINESS_MAX_BYTES = 128 * 1024
_RESOURCE_MAX_BYTES = 4 * 1024 * 1024
_MAX_ENTRIES = 128
_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*\Z")
_VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_CONSONANTS = set("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
_MODEL_PHONES = set(en_symbols)
_BUSINESS_SHA256 = "b3f20b1692ee107d119bd778690b74b65ae246f8ea80ec541ca932d2d650d6e6"


def bundled_hotwords_source():
    """Return the resource path and canonical business-section SHA; no I/O."""
    return str(frontend_lexicon_path()), _BUSINESS_SHA256


def _entry(row):
    """Validate to an immutable tuple; disabled candidates never become active."""
    if not isinstance(row, dict) or type(row.get("enabled")) is not bool or type(row.get("reviewed")) is not bool:
        raise ValueError("Hotword entries require explicit enabled and reviewed booleans")
    if not row["enabled"]:
        return None
    if not row["reviewed"]:
        raise ValueError("Enabled hotwords must be reviewed")
    text, case = row.get("text"), row.get("case")
    languages, domain = row.get("languages"), row.get("domain")
    priority, override = row.get("priority"), row.get("override_cmu", False)
    if not isinstance(text, str) or not _TOKEN.fullmatch(text) or case not in ("exact", "fold"):
        raise ValueError("Hotwords require a complete Latin token and case=exact or fold")
    if not isinstance(languages, list) or not languages or any(x != "en" for x in languages):
        raise ValueError("Hotword language is the English token language: languages=['en']")
    if not isinstance(domain, str) or not re.fullmatch(r"\*|[a-z][a-z0-9_-]{0,31}", domain):
        raise ValueError("Hotwords require an explicit domain")
    if type(priority) is not int or not -1000 <= priority <= 1000 or type(override) is not bool:
        raise ValueError("Hotwords require an integer priority and boolean override_cmu")
    if override and (case != "exact" or domain in ("*", "general")):
        raise ValueError("CMU overrides require exact case and a specific non-general domain")
    if not isinstance(row.get("source"), str) or not row["source"].strip() or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_sha256", ""))):
        raise ValueError("Hotwords require source and source_sha256 provenance")
    arpa = row.get("arpa")
    if not isinstance(arpa, list) or not 1 <= len(arpa) <= 64:
        raise ValueError("Hotword arpa must be a nonempty list of at most 64 phones")
    for phone in arpa:
        match = re.fullmatch(r"([A-Z]+)([012])?", phone) if isinstance(phone, str) else None
        if not match:
            raise ValueError("Invalid hotword ARPAbet phone")
        base, stress = match.groups()
        model_phone = "V" if base == "V" else base.lower()
        if not ((base in _VOWELS and stress is not None) or (base in _CONSONANTS and stress is None)) or model_phone not in _MODEL_PHONES:
            raise ValueError("Hotword ARPAbet phone is incompatible with the model")
    return text, case, tuple(languages), domain, priority, override, tuple(arpa)


def _check_overlaps(entries):
    for index, left in enumerate(entries):
        for right in entries[index + 1:]:
            shared_case = left[0] == right[0] if left[1] == right[1] == "exact" else left[0].upper() == right[0].upper()
            shared_domain = left[3] == right[3] or "*" in (left[3], right[3])
            if shared_case and shared_domain and set(left[2]) & set(right[2]) and left[4] == right[4]:
                raise ValueError("Overlapping hotwords require different explicit priorities")


def _check_source(row, sources):
    """Verify phones against the pinned manifest's retained source records.

    Original file hashes and references support offline source audits. Runtime
    needs only these small records; it never opens private handoff directories.
    """
    source = sources.get(row["source"]) if isinstance(sources, dict) else None
    terms = row.get("source_terms")
    if (not isinstance(source, dict) or source.get("sha256") != row["source_sha256"]
            or not isinstance(source.get("reference"), str) or not source["reference"].strip()
            or not isinstance(source.get("records"), dict)
            or not isinstance(terms, list) or not 1 <= len(terms) <= 64
            or any(not isinstance(term, str) or not term for term in terms)):
        raise ValueError("Hotword source provenance does not match its source records")
    records = [source["records"].get(term) for term in terms]
    if (any(not isinstance(record, list) or not record for record in records)
            or [phone for record in records for phone in record] != row["arpa"]):
        raise ValueError("Hotword phones do not match their source provenance")
    if "cmu_components" in row:
        cmu = sources.get("released_cmu", {})
        components = row["cmu_components"]
        records = cmu.get("records", {}) if isinstance(cmu, dict) else {}
        if (not isinstance(components, list) or not 1 <= len(components) <= 64 or not isinstance(records, dict)
                or any(not isinstance(term, str) or not isinstance(records.get(term), list) for term in components)
                or [phone for term in components for phone in records[term]] != row["arpa"]):
            raise ValueError("Hotword CMU component provenance does not match its phones")


@lru_cache(maxsize=8)
def _load_external(path, checksum, *, max_bytes=_MAX_BYTES):
    # Read once per explicit configuration, avoiding disk I/O for every token.
    with open(path, "rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"Hotword manifest exceeds {max_bytes // 1024} KiB; raw Gold is not a manifest")
    data = json.loads(payload)
    if isinstance(data, dict) and data.get("format") == "vits2-external-pronunciations":
        data = data.get("business_lexicon")
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("utf-8") + b"\n"
        if len(payload) > _BUSINESS_MAX_BYTES:
            raise ValueError("Embedded business vocabulary exceeds 128 KiB")
    if checksum and hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("Hotword manifest SHA256 mismatch")
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("entries"), list):
        raise ValueError("Hotword manifest requires version=1 and entries; raw Gold is not supported")
    if len(data["entries"]) > _MAX_ENTRIES:
        raise ValueError("Hotword manifest exceeds 128 reviewed entries")
    entries = []
    for row in data["entries"]:
        parsed = _entry(row)
        if parsed is not None:
            if "sources" in data:
                _check_source(row, data["sources"])
            entries.append(parsed)
    entries = tuple(entries)
    _check_overlaps(entries)
    return entries


@lru_cache(maxsize=8)
def _load_manifests(path, checksum):
    business_path, business_sha256 = bundled_hotwords_source()
    entries = _load_external(business_path, business_sha256, max_bytes=_RESOURCE_MAX_BYTES)
    if path:
        entries += _load_external(path, checksum)
        if len(entries) > _MAX_ENTRIES:
            raise ValueError("Combined hotword manifests exceed 128 reviewed entries")
        _check_overlaps(entries)
    return entries


@lru_cache(maxsize=8)
def _manifest_index(path, checksum):
    """Index once per resource identity, not once per spoken token."""
    exact, folded = {}, {}
    for row in _load_manifests(path, checksum):
        target = exact if row[1] == "exact" else folded
        key = row[0] if row[1] == "exact" else row[0].upper()
        target.setdefault(key, []).append(row)
    return ({key: tuple(rows) for key, rows in exact.items()},
            {key: tuple(rows) for key, rows in folded.items()})


@lru_cache(maxsize=8)
def _resolved_manifest(path):
    return str(Path(path).expanduser().resolve())


def deployment_configuration_active():
    """Whether lookup needs the optional deployment/domain compatibility path."""
    return bool(os.environ.get("VITS2_HOTWORDS_PATH", "")
                or os.environ.get("VITS2_HOTWORDS_SHA256", "")
                or os.environ.get("VITS2_HOTWORD_DOMAIN", "general") != "general")


def reload_hotwords():
    """Explicitly reload an edited local manifest on the next lookup."""
    _load_external.cache_clear()
    _load_manifests.cache_clear()
    _manifest_index.cache_clear()
    _resolved_manifest.cache_clear()


def lookup_hotword(token, *, language="en", domain=None, has_cmu=False):
    """Return a fresh CMU-shaped pronunciation, or None to retain normal lookup.

    Caller keeps CUSTOM/VERIFIED first and passes has_cmu for the CMU guard.
    Explicit domain overrides can precede CMU; general supplements cannot.
    """
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        return None
    selected_domain = domain if domain is not None else os.environ.get("VITS2_HOTWORD_DOMAIN", "general")
    path = os.environ.get("VITS2_HOTWORDS_PATH", "")
    checksum = os.environ.get("VITS2_HOTWORDS_SHA256", "").strip().lower()
    if checksum and (not path or not re.fullmatch(r"[0-9a-f]{64}", checksum)):
        raise ValueError("Hotword SHA256 requires a manifest path and 64 hexadecimal digits")
    source = _resolved_manifest(path) if path else ""
    exact, folded = _manifest_index(source, checksum)
    candidates = exact.get(token, ()) + folded.get(token.upper(), ())
    matches = [row for row in candidates
               if language in row[2] and row[3] in ("*", selected_domain)
               and (token == row[0] if row[1] == "exact" else token.upper() == row[0].upper())
               and (not has_cmu or row[5])]
    if not matches:
        return None
    chosen = max(matches, key=lambda row: row[4])
    return [list(chosen[6])]
