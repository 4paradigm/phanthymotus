"""Compatibility helpers for the public WeText runtime."""

import sys


def ensure_wetext_compat() -> None:
    """Provide ``importlib.resources.files`` for Python 3.8 before WeText loads."""
    if sys.version_info >= (3, 9):
        return
    import importlib.resources as resources
    from importlib_resources import files

    if not hasattr(resources, "files"):
        resources.files = files
