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
DEFAULT_CASE_POLICIES = Path(__file__).resolve().parent / "frontend/lexicon_case_policies.json"
_KEY = re.compile(r"[A-Z]+(?:['-][A-Z]+)*\Z")
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
    path = Path(__file__).resolve().parent / "frontend/symbols.py"
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


def compile_lexicon(source_path, cmu, *, expected_source_sha256=None,
                    expected_source_canonical_sha256=None, case_policies=None,
                    cmu_source_sha256=None):
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
        counts[row["mode"]] += 1
        entries[key] = row
    payload = {
        "format": FORMAT, "version": 1,
        "source": {"name": "fanyi/en_gold_pronunciations.json", "sha256": source_hash,
                   "canonical_sha256": semantic_hash, "entries": len(source)},
        "cmu": {"sha256": cmu_source_sha256, "canonical_sha256": canonical_sha256(cmu),
                "canonical_layout": "nested_syllables", "entries": len(cmu)},
        "policy": {"cmu_overlap": "retain_release_cmu_default", "letter_spellings": "exact_uppercase",
                   "other_oov": "casefold", "case_overrides": policies,
                   "case_overrides_sha256": canonical_sha256(policies),
                   "semantic_pronunciation_review": "not established by structural validation"},
        "entries": entries,
    }
    audit = {"source": payload["source"], "cmu": payload["cmu"], "counts": dict(counts),
             "source_entries_preserved": set(entries) == set(source),
             "pronunciation_conflicts_defaulted_to_cmu": len(conflicts), "cmu_conflicts": conflicts,
             "case_overrides": policies,
             "risk_notes": ["A source spelling can be appropriate for a ticker and inappropriate for a person's name.",
                            "OOV fold entries retain source phonemes; case folding is not proof of semantic pronunciation.",
                            "No acoustic or device-listening acceptance is implied by a valid phone sequence."],
             "acoustic_acceptance": False, "model_inference": False, "source_modified": False}
    return payload, audit


def encode_payload(payload):
    return canonical_json(payload) + b"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--source-canonical-sha256")
    parser.add_argument("--cmu", required=True, type=Path, help="Original cmudict.rep text; no pickle execution")
    parser.add_argument("--case-policies", type=Path, default=DEFAULT_CASE_POLICIES)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory: " + str(args.output_dir))
    cmu = parse_cmu_rep(args.cmu)
    policies = read_json(args.case_policies) if args.case_policies else {}
    payload, audit = compile_lexicon(args.source, cmu, expected_source_sha256=args.source_sha256,
        expected_source_canonical_sha256=args.source_canonical_sha256, case_policies=policies,
        cmu_source_sha256=hashlib.sha256(args.cmu.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "external_lexicon.json"
    output.write_bytes(encode_payload(payload))
    audit.update(output={"file": output.name, "bytes": output.stat().st_size,
                         "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
                 tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "entries": len(payload["entries"]),
                      "counts": audit["counts"], **audit["output"]}))


if __name__ == "__main__":
    main()
