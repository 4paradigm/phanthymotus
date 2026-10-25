#!/usr/bin/env python3
"""CLI for the plugin-owned, standard-library external lexicon compiler."""
from pathlib import Path
import sys

PERCEPTION = Path(__file__).resolve().parents[1]
if str(PERCEPTION) not in sys.path:
    sys.path.insert(0, str(PERCEPTION))

from plugins.vits2_tts_trt.lexicon_compiler import (
    DEFAULT_CASE_POLICIES, FORMAT, canonical_json, canonical_sha256,
    compile_lexicon, encode_payload, main, model_phone_inventory,
    parse_cmu_rep, read_json, spells_letters, validate_arpa,
)

if __name__ == "__main__":
    main()
