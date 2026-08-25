"""Shared pytest wiring for perception tests (stubs + node-registry reset)."""

import pytest

import vision_stubs  # noqa: F401  (importing installs the ROS stubs)
from vision_stubs import _FakeNode


@pytest.fixture(autouse=True)
def _reset_nodes():
    _FakeNode.instances.clear()
    yield
    _FakeNode.instances.clear()
