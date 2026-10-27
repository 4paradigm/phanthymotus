#!/usr/bin/env python3
"""Compile a pinned Gold source into an external, CMU-safe VITS2 lexicon.

No model execution, downloads, source edits, or guessed pronunciation labels.
The large output belongs in the model/data cache, never in the Git repository.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


FORMAT = "vits2-external-pronunciations"
DEFAULT_CASE_POLICIES = None
_KEY = re.compile(r"[A-Z]+(?:['-][A-Z]+)*\Z")
_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SUPPLEMENTS_MAX_BYTES = 128 * 1024
_VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_CONSONANTS = set("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
# Alphabet names define spelling detection, not business-word pronunciation.
_LETTER_ARPA = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (
    "EY1", "B IY1", "S IY1", "D IY1", "IY1", "EH1 F", "JH IY1",
    "EY1 CH", "AY1", "JH EY1", "K EY1", "EH1 L", "EH1 M", "EH1 N",
    "OW1", "P IY1", "K Y UW1", "AA1 R", "EH1 S", "T IY1", "Y UW1",
    "V IY1", "D AH1 B AH0 L Y UW0", "EH1 K S", "W AY1", "Z IY1")))


def canonical_json(value):
    """Stable semantic encoding shared with the runtime CMU verification."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate lexicon JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_unique_object)


def parse_cmu_rep(path):
    """Match the release reader's last-definition-wins text parsing.

    The original cmudict.rep contains duplicate words such as BING. Retaining
    its last definition reproduces the verified runtime cache exactly.
    """
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.strip().split("  ")
        if len(fields) < 2 or fields[0].startswith("##"):
            continue
        result[fields[0]] = [part.split() for part in fields[1].split(" - ")]
    if not result:
        raise ValueError("CMU source is empty")
    return result


def model_phone_inventory():
    path = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/frontend/symbols.py"
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "en_symbols" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise ValueError("Model English phone inventory was not found")


def validate_arpa(phones, inventory):
    if not isinstance(phones, list) or not 1 <= len(phones) <= 128:
        raise ValueError("Pronunciation must be a nonempty flat list with at most 128 phones")
    for phone in phones:
        match = re.fullmatch(r"([A-Z]+)([012])?", phone) if isinstance(phone, str) else None
        if match is None:
            raise ValueError("Invalid ARPAbet phone: " + str(phone))
        base, stress = match.groups()
        if not ((base in _VOWELS and stress is not None) or (base in _CONSONANTS and stress is None)):
            raise ValueError("Invalid ARPAbet stress: " + phone)
        if ("V" if base == "V" else base.lower()) not in inventory:
            raise ValueError("ARPAbet phone is outside the model inventory: " + phone)


def _without_stress(phones):
    return [re.sub(r"[012]$", "", phone) for phone in phones]


def spells_letters(word, phones):
    # Hyphen/apostrophe spelling is still spelling, e.g. D-ID or GPT-J.
    # The full phone-sequence comparison prevents treating a normal compound
    # as an initialism merely because its written form contains a separator.
    word = word.replace("-", "").replace("'", "")
    if not word or any(letter not in _LETTER_ARPA for letter in word):
        return False
    spelling = [phone for letter in word for phone in _LETTER_ARPA[letter].split()]
    return _without_stress(phones) == _without_stress(spelling)


def _read_supplements(path, checksum, inventory):
    """Validate a pinned general-English manifest without loading the frontend.

    Deployment domain overrides remain a separate runtime facility. The
    compiler accepts only reviewed, unconditional fold entries and verifies
    their phones against the manifest's retained source records.
    """
    if not _SHA.fullmatch(str(checksum)):
        raise ValueError("Supplements require a pinned SHA256")
    with Path(path).open("rb") as stream:
        raw = stream.read(_SUPPLEMENTS_MAX_BYTES + 1)
    if len(raw) > _SUPPLEMENTS_MAX_BYTES:
        raise ValueError("Supplements exceed 128 KiB")
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("Pinned supplements SHA256 mismatch")
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or not isinstance(manifest.get("entries"), list)
            or len(manifest["entries"]) > 128):
        raise ValueError("Supplements require version=1 and at most 128 entries")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("Supplements require retained source provenance")
    entries, source_counts = {}, Counter()
    for row in manifest["entries"]:
        if (not isinstance(row, dict) or type(row.get("enabled")) is not bool
                or type(row.get("reviewed")) is not bool):
            raise ValueError("Supplement entries require enabled/reviewed booleans")
        if not row["enabled"]:
            continue
        text = row.get("text")
        if (not row["reviewed"] or not isinstance(text, str) or not _TOKEN.fullmatch(text)
                or row.get("case") != "fold" or row.get("languages") != ["en"]
                or row.get("domain") != "general" or type(row.get("priority")) is not int
                or row["priority"] != 0 or row.get("override_cmu") is not False):
            raise ValueError("Supplements require reviewed fold/general/en entries without overrides")
        phones = row.get("arpa")
        validate_arpa(phones, inventory)
        if len(phones) > 64:
            raise ValueError("Supplement pronunciations require at most 64 phones")
        source_id, terms = row.get("source"), row.get("source_terms")
        source = sources.get(source_id) if isinstance(source_id, str) else None
        if (not isinstance(source_id, str) or not source_id.strip()
                or not isinstance(source, dict) or not _SHA.fullmatch(str(source.get("sha256", "")))
                or source.get("sha256") != row.get("source_sha256")
                or not isinstance(source.get("reference"), str) or not source["reference"].strip()
                or not isinstance(source.get("records"), dict)
                or not isinstance(terms, list) or not 1 <= len(terms) <= 64
                or any(not isinstance(term, str) or not term for term in terms)):
            raise ValueError("Supplement source provenance does not match retained records")
        records = [source["records"].get(term) for term in terms]
        if (any(not isinstance(record, list) or not record for record in records)
                or [phone for record in records for phone in record] != phones):
            raise ValueError("Supplement phones do not match source records")
        if "cmu_components" in row:
            components = row["cmu_components"]
            cmu_source = sources.get("released_cmu", {})
            cmu_records = cmu_source.get("records", {}) if isinstance(cmu_source, dict) else {}
            if (not isinstance(components, list) or not 1 <= len(components) <= 64
                    or not isinstance(cmu_records, dict)
                    or any(not isinstance(term, str) or not isinstance(cmu_records.get(term), list)
                           for term in components)
                    or [phone for term in components for phone in cmu_records[term]] != phones):
                raise ValueError("Supplement CMU components do not match phones")
        key = text.upper()
        if key in entries:
            raise ValueError("Duplicate folded supplement key: " + key)
        entries[key] = list(phones)
        source_counts[source_id] += 1
    provenance = {
        "name": Path(path).name, "sha256": checksum, "bytes": len(raw),
        "entries": len(manifest["entries"]), "enabled_entries": len(entries),
        "source_counts": dict(source_counts),
        "sources": {key: {"reference": sources[key]["reference"], "sha256": sources[key]["sha256"]}
                    for key in sorted(source_counts)},
    }
    return entries, provenance


def compile_lexicon(source_path, cmu, *, expected_source_sha256=None,
                    expected_source_canonical_sha256=None, case_policies=None,
                    cmu_source_sha256=None, supplements_path=None,
                    supplements_sha256=None):
    """Return (runtime payload, audit); require a byte or semantic source pin.

    cmu is the release's actual nested-syllable mapping. Overlapping entries
    retain its default pronunciation; original Gold annotations stay auditable.
    Case policies are external data, not runtime token-specific conditions.
    """
    if not expected_source_sha256 and not expected_source_canonical_sha256:
        raise ValueError("A pinned source SHA256 or canonical SHA256 is required")
    raw = Path(source_path).read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    source = json.loads(raw, object_pairs_hook=_unique_object)
    semantic_hash = canonical_sha256(source)
    for expected, actual in ((expected_source_sha256, source_hash),
                             (expected_source_canonical_sha256, semantic_hash)):
        if expected is not None and (not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or expected != actual):
            raise ValueError("Pinned Gold source SHA256 mismatch")
    if not isinstance(source, dict) or not source or len(source) > 50000:
        raise ValueError("Gold source must contain 1..50000 entries")
    if not isinstance(cmu, dict) or not cmu:
        raise ValueError("A nonempty release CMU mapping is required")
    if cmu_source_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", str(cmu_source_sha256)):
        raise ValueError("CMU source SHA256 must contain 64 lowercase hexadecimal digits")
    if case_policies is not None and not isinstance(case_policies, dict):
        raise ValueError("Case policies must be an object")
    policies = dict(case_policies or {})
    if any(key not in source or value not in {"exact", "fold"} for key, value in policies.items()):
        raise ValueError("Case policies require existing source keys and exact/fold modes")
    inventory = model_phone_inventory()
    base_source = {"name": "fanyi/en_gold_pronunciations.json", "sha256": source_hash,
                   "canonical_sha256": semantic_hash, "entries": len(source)}
    base_keys = set(source)
    supplement_keys = set()
    supplements = None
    if supplements_path is None:
        if supplements_sha256 is not None:
            raise ValueError("Supplements SHA256 requires a supplements path")
    else:
        additions, supplements = _read_supplements(supplements_path, supplements_sha256, inventory)
        source = dict(source)
        added, deduplicated = [], []
        for key, phones in additions.items():
            if key in source:
                if source[key] != phones:
                    raise ValueError("Supplement conflicts with Gold pronunciation: " + key)
                deduplicated.append(key)
            else:
                source[key] = phones
                policies[key] = "fold"
                added.append(key)
                supplement_keys.add(key)
        if len(source) > 50000:
            raise ValueError("Merged Gold source exceeds 50000 entries")
        supplements.update(added_entries=len(added), deduplicated_entries=len(deduplicated),
                           added_keys=sorted(added), deduplicated_keys=sorted(deduplicated))
        semantic_hash = canonical_sha256(source)
        source_hash = hashlib.sha256(canonical_json(source) + b"\n").hexdigest()
    entries, counts, conflicts = {}, Counter(), []
    for key in sorted(source):
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Gold keys must be uppercase whole Latin tokens: " + str(key))
        original = source[key]
        validate_arpa(original, inventory)
        spelling = spells_letters(key, original)
        counts["source_letter_spellings"] += int(spelling)
        if spelling:
            counts["source_separated_letter_spellings" if "-" in key or "'" in key
                   else "source_plain_letter_spellings"] += 1
        if key in cmu:
            syllables = cmu[key]
            if not isinstance(syllables, list) or not syllables or any(not isinstance(row, list) or not row for row in syllables):
                raise ValueError("CMU pronunciations require nested syllable lists")
            phones = [phone for syllable in syllables for phone in syllable]
            validate_arpa(phones, inventory)
            row = {"arpa": phones, "mode": "cmu", "original_arpa": list(original)}
            if phones != original:
                conflicts.append({"key": key, "source_arpa": original, "default_arpa": phones,
                                  "source_spells_letters": spelling})
        else:
            row = {"arpa": list(original), "mode": policies.get(key, "exact" if spelling else "fold")}
        if key in supplement_keys:
            row["origin"] = "supplement"
        counts[row["mode"]] += 1
        entries[key] = row
    payload = {
        "format": FORMAT, "version": 1,
        "source": {"name": "merged/en_gold_pronunciations.json" if supplements is not None
                            else "fanyi/en_gold_pronunciations.json", "sha256": source_hash,
                   "canonical_sha256": semantic_hash, "entries": len(source)},
        "cmu": {"sha256": cmu_source_sha256, "canonical_sha256": canonical_sha256(cmu),
                "canonical_layout": "nested_syllables", "entries": len(cmu)},
        "policy": {"cmu_overlap": "retain_release_cmu_default", "letter_spellings": "exact_uppercase",
                   "other_oov": "casefold", "case_overrides": policies,
                   "case_overrides_sha256": canonical_sha256(policies),
                   "semantic_pronunciation_review": "not established by structural validation"},
        "entries": entries,
    }
    if supplements is not None:
        payload["source"].update(canonical_layout="uppercase_tokens_arpabet",
                                 serialization="canonical_json_utf8_lf")
        payload.update(base_source=base_source, supplements=supplements)
    audit = {"source": payload["source"], "cmu": payload["cmu"], "counts": dict(counts),
             "source_entries_preserved": set(entries) == set(source),
             "base_source_entries_preserved": base_keys <= set(entries),
             "pronunciation_conflicts_defaulted_to_cmu": len(conflicts), "cmu_conflicts": conflicts,
             "case_overrides": policies,
             "risk_notes": ["A source spelling can be appropriate for a ticker and inappropriate for a person's name.",
                            "OOV fold entries retain source phonemes; case folding is not proof of semantic pronunciation.",
                            "No acoustic or device-listening acceptance is implied by a valid phone sequence."],
             "acoustic_acceptance": False, "model_inference": False, "source_modified": False}
    if supplements is not None:
        audit.update(base_source=base_source, supplements=supplements)
    return payload, audit


def encode_payload(payload):
    return canonical_json(payload) + b"\n"


def maintenance_payloads(payload):
    """Export reproducible Gold and matching policy copies beside the runtime file."""
    gold = {key: row.get("original_arpa", row["arpa"])
            for key, row in payload["entries"].items()}
    return {"en_gold_pronunciations.json": canonical_json(gold) + b"\n",
            "lexicon_case_policies.json": canonical_json(payload["policy"]["case_overrides"]) + b"\n"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--source-canonical-sha256")
    parser.add_argument("--cmu", required=True, type=Path, help="Original cmudict.rep text; no pickle execution")
    parser.add_argument("--case-policies", type=Path, default=DEFAULT_CASE_POLICIES,
                        help="External matching-policy JSON for the original Gold source")
    parser.add_argument("--supplements", type=Path,
                        help="External reviewed business manifest to merge into the original Gold source")
    parser.add_argument("--supplements-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory: " + str(args.output_dir))
    cmu = parse_cmu_rep(args.cmu)
    policies = read_json(args.case_policies) if args.case_policies else {}
    if args.supplements is None:
        if args.supplements_sha256 is not None:
            raise ValueError("--supplements-sha256 requires --supplements")
        supplements_path, supplements_sha256 = None, None
    else:
        supplements_path, supplements_sha256 = args.supplements, args.supplements_sha256
    payload, audit = compile_lexicon(args.source, cmu, expected_source_sha256=args.source_sha256,
        expected_source_canonical_sha256=args.source_canonical_sha256, case_policies=policies,
        cmu_source_sha256=hashlib.sha256(args.cmu.read_bytes()).hexdigest(),
        supplements_path=supplements_path, supplements_sha256=supplements_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "external_lexicon.json"
    output.write_bytes(encode_payload(payload))
    maintained = maintenance_payloads(payload)
    for filename, data in maintained.items():
        (args.output_dir / filename).write_bytes(data)
    audit["maintenance_outputs"] = {
        filename: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for filename, data in maintained.items()}
    audit.update(output={"file": output.name, "bytes": output.stat().st_size,
                         "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
                 tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "entries": len(payload["entries"]),
                      "counts": audit["counts"], **audit["output"]}))


if __name__ == "__main__":
    main()
