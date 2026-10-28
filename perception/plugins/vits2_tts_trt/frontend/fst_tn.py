"""Checksum-verified Chinese TN execution with the Kaldifst runtime."""

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Union

from .wetext_compat import ensure_wetext_compat

ensure_wetext_compat()

import kaldifst
from wetext import token_parser


class FstReleaseError(ValueError):
    """Raised when a TN release is incomplete or fails verification."""


class FstTextNormalizer:
    """Execute the release-controlled Chinese tagger/verbalizer FST pair."""

    _token_parser_lock = threading.RLock()

    def __init__(self, release_dir: Union[str, Path]):
        root = Path(release_dir).resolve()
        manifest_path = root / "tn_manifest.json"
        if not manifest_path.is_file():
            raise FstReleaseError(f"missing TN manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FstReleaseError(f"invalid TN manifest: {exc}") from exc
        graphs = manifest.get("fst")
        if manifest.get("schema_version") != 1 or not isinstance(graphs, dict):
            raise FstReleaseError("unsupported TN manifest schema")

        self._tagger_path = self._verified_path(root, graphs, "zh_tn_tagger.fst")
        self._verbalizer_path = self._verified_path(root, graphs, "zh_tn_verbalizer.fst")

    @staticmethod
    def _verified_path(root: Path, graphs: dict, filename: str) -> str:
        metadata = graphs.get(filename)
        if not isinstance(metadata, dict):
            raise FstReleaseError(f"TN release does not declare {filename}")
        expected_hash = metadata.get("sha256")
        expected_size = metadata.get("bytes")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise FstReleaseError(f"invalid checksum for {filename}")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise FstReleaseError(f"invalid size for {filename}")
        path = (root / filename).resolve()
        if root not in path.parents or not path.is_file():
            raise FstReleaseError(f"missing TN graph: {filename}")
        if path.stat().st_size != expected_size:
            raise FstReleaseError(f"TN graph size mismatch: {filename}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise FstReleaseError(f"TN graph checksum mismatch: {filename}")
        return str(path)

    @property
    @lru_cache(maxsize=1)
    def _tagger(self):
        return kaldifst.TextNormalizer(self._tagger_path)

    @property
    @lru_cache(maxsize=1)
    def _verbalizer(self):
        return kaldifst.TextNormalizer(self._verbalizer_path)

    def normalize(self, text: str) -> str:
        """Run the release graphs while retaining literal 啊 character tokens."""
        if not text:
            return text
        tagged = self._tagger(text)
        segments = []
        with self._token_parser_lock:
            original_escape = token_parser.escape_value
            try:
                token_parser.escape_value = lambda value: value
                parser = token_parser.TokenParser("zh", "tn")
                reordered = parser.reorder(tagged)
                # The released verbalizer deletes 啊, even in "你好啊". Preserve
                # only this literal char token after the complete input has been
                # classified, so dates/numbers keep their original context.
                # Other interjections and structured tokens retain graph behavior.
                if "啊" in tagged:
                    pending = []
                    for token in parser.tokens:
                        if token.name == "char" and token.members == {"value": "啊"}:
                            if pending:
                                segments.append((False, " ".join(pending)))
                                pending = []
                            segments.append((True, "啊"))
                        else:
                            pending.append(token.string(parser.orders))
                    if segments and pending:
                        segments.append((False, " ".join(pending)))
            finally:
                token_parser.escape_value = original_escape
        if segments:
            return "".join(value if literal else self._verbalizer(value)
                           for literal, value in segments)
        return self._verbalizer(reordered)
