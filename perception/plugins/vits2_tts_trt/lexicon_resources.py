"""Install verified external frontend data before acoustic startup.

The Gold dictionary is compiled when its resource version is published, not
on a robot. Vocabulary changes therefore update one pinned ModelScope JSON;
no generated vocabulary or model artifact is part of the Python source tree.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import re


log = logging.getLogger(__name__)
_RESOURCE_REVISION = "275f343e89d9cef520c829e9fff3e950354b9b12"
_SOURCE_BASE = (
    "https://www.modelscope.cn/api/v1/models/lanxijiang/TTSV4_Ch_Eng/repo"
    "?Revision=" + _RESOURCE_REVISION + "&FilePath=frontend/lexicon/20261027-v4-7c0e657e/{file}"
)
_SOURCE_FILES = {
    "external_lexicon.json": {
        "size": 1418015,
        "sha256": "7c0e657e9b3eae25d3aa12461e2db1c4f0386009d60a32d36f847088addbd68a",
    },
}

_SHA = re.compile(r"[0-9a-f]{64}\Z")


def ensure_external_lexicon(root, *, progress_cb=None, stage_cb=None):
    """Prepare one external resource and its single runtime lookup index.

    An explicitly configured pronunciation index is validated before any
    download, and is never replaced. The shared text and maintenance sections
    remain part of frontend preparation even with an explicit index.
    """
    from utils.model_downloader import ensure_verified_bundle
    from .frontend.external_lexicon import (
        reload_external_lexicon, validate_external_lexicon_file,
    )
    from .frontend.release_paths import LEXICON_RESOURCE_DIRECTORY

    def stage(name):
        if stage_cb is not None:
            try:
                stage_cb(name)
            except Exception:
                log.debug("[vits2] external lexicon status callback failed", exc_info=True)

    stage("verify")
    root = Path(root).resolve()
    explicit = os.environ.get("VITS2_LEXICON_PATH", "").strip()
    checksum = os.environ.get("VITS2_LEXICON_SHA256", "").strip().lower()
    if checksum and not _SHA.fullmatch(checksum):
        raise ValueError("VITS2_LEXICON_SHA256 requires 64 hexadecimal digits")
    selected = Path(explicit).expanduser().resolve() if explicit else None
    if selected is not None:
        validate_external_lexicon_file(selected, checksum)
    cmu_path = root / "frontend_data/cmudict.rep"
    if not cmu_path.is_file():
        raise FileNotFoundError("VITS2 CMU source is missing: " + str(cmu_path))

    directory = root / "frontend_data" / LEXICON_RESOURCE_DIRECTORY
    resource = directory / "external_lexicon.json"
    if selected == resource.resolve():
        expected = _SOURCE_FILES["external_lexicon.json"]
        if (selected.stat().st_size != expected["size"]
                or hashlib.sha256(selected.read_bytes()).hexdigest() != expected["sha256"]):
            raise ValueError("Explicit lexicon conflicts with the shared resource cache; use a separate path")

    stage("download")
    paths = ensure_verified_bundle(
        "vits2-frontend-20261027-v4-7c0e657e", str(directory),
        _SOURCE_BASE, _SOURCE_FILES, progress_cb=progress_cb,
    )
    if selected is None:
        selected = Path(paths["external_lexicon.json"])
    # Resource identity must match the installed model even before English
    # binds its full CMU mapping. Its existing startup binding also verifies
    # each overlapping entry and the nested-syllable canonical identity.
    reload_external_lexicon(path=selected, checksum=checksum)
    status = validate_external_lexicon_file(selected, checksum)
    if hashlib.sha256(cmu_path.read_bytes()).hexdigest() != status["cmu_sha256"]:
        raise ValueError("External lexicon was prepared for another CMU source")
    from .frontend.hotwords import reload_hotwords
    from .frontend.lexicon_rules import _generated_tables
    reload_hotwords()
    _generated_tables.cache_clear()
    stage("verify")
    return selected
