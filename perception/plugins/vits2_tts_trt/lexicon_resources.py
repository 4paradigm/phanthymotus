"""Validate the release's pronunciation dictionary before acoustic startup.

The dictionary is compiled offline (tools/vits2_prepare_external_lexicon.py)
and shipped inside the VITS2 release archive, next to the CMU dictionary it was
compiled against. There is no separate download: the archive's own size and
SHA256 pin cover it, and a robot that has the model has its dictionary.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def load_external_lexicon(root) -> Path:
    """Activate the release dictionary (or VITS2_LEXICON_PATH) and check it.

    Call after ``configure_release_paths(root)``. Raises if the dictionary is
    missing or was compiled for another CMU source — a dictionary that does not
    match its CMU data would silently mispronounce, which is worse than failing
    the load where ``info`` shows it.
    """
    from .frontend.external_lexicon import (
        reload_external_lexicon, validate_external_lexicon_file,
    )
    from .frontend.hotwords import reload_hotwords
    from .frontend.lexicon_rules import _generated_tables
    from .frontend.release_paths import frontend_lexicon_path

    root = Path(root).resolve()
    cmu_path = root / "frontend_data/cmudict.rep"
    if not cmu_path.is_file():
        raise FileNotFoundError("VITS2 CMU source is missing: " + str(cmu_path))
    explicit = os.environ.get("VITS2_LEXICON_PATH", "").strip()
    checksum = os.environ.get("VITS2_LEXICON_SHA256", "").strip().lower()
    selected = Path(explicit).expanduser().resolve() if explicit else frontend_lexicon_path()
    if not selected.is_file():
        raise FileNotFoundError("VITS2 pronunciation dictionary is missing: " + str(selected))

    reload_external_lexicon(path=selected, checksum=checksum)
    status = validate_external_lexicon_file(selected, checksum)
    if hashlib.sha256(cmu_path.read_bytes()).hexdigest() != status["cmu_sha256"]:
        raise ValueError("External lexicon was prepared for another CMU source")
    reload_hotwords()
    _generated_tables.cache_clear()
    return selected
