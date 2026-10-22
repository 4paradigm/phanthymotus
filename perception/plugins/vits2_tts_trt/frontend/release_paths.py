"""Release-controlled frontend asset locations."""

from __future__ import annotations

from pathlib import Path


_release_root: Path | None = None


def configure_release_paths(root: Path) -> None:
    """Configure frontend assets from the verified VITS2 release root."""
    global _release_root
    _release_root = root.resolve()


def _root() -> Path:
    if _release_root is None:
        raise RuntimeError("VITS2 frontend release paths are not configured")
    return _release_root


def frontend_data_dir() -> Path:
    """Return the verified release frontend-data directory."""
    return _root() / "frontend_data"


def nltk_data_dir() -> Path:
    """Return the verified release NLTK-data directory."""
    return _root() / "nltk_data"


def tn_cache_dir() -> Path:
    """Return the verified release TN graph directory."""
    return _root() / "tn_cache"
