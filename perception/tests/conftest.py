"""Shared pytest wiring for perception tests (stubs + node-registry reset)."""

import os
import sys

import pytest

import vision_stubs  # noqa: F401  (importing installs the ROS stubs)
from vision_stubs import _FakeNode


def _assert_plugins_is_ours() -> None:
    """`plugins` must resolve to perception/plugins, not actucore/plugins.

    The mirror of the guard in actucore/tests/conftest.py — both directories
    hold a top-level `plugins` package, so the two suites cannot share one
    pytest process. Whichever is collected first wins and the other reports a
    pile of ImportErrors for modules that plainly exist. Say why once instead.
    """
    plugins = sys.modules.get("plugins")
    if plugins is None:
        return
    ours = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "plugins"))
    paths = sorted({os.path.realpath(p) for p in getattr(plugins, "__path__", [])})
    if ours in paths:
        return
    # pytest.exit rather than raise: an exception from a hook is reported as
    # INTERNALERROR with a traceback, which reads like the test tooling broke.
    pytest.exit(
        f"`plugins` is bound to {', '.join(paths) or plugins!r}, not {ours}. "
        f"perception/tests and actucore/tests both define a top-level "
        f"`plugins` package, so they cannot share one pytest process. Run "
        f"them as two separate commands.",
        returncode=pytest.ExitCode.USAGE_ERROR,
    )


def pytest_collectstart(collector):
    # See the same hook in actucore/tests/conftest.py: every conftest in the
    # tree is imported before any test module, so at conftest import time
    # `plugins` is still unbound and the check would see nothing. This fires
    # immediately before each module here is imported.
    if isinstance(collector, pytest.Module):
        _assert_plugins_is_ours()


@pytest.fixture(autouse=True)
def _reset_nodes():
    _FakeNode.instances.clear()
    yield
    _FakeNode.instances.clear()
