"""
tests/test_bundle_dispatch.py — PerceptionBundle tool-name resolution.

`main.py` maps an incoming `tools/call` name onto a plugin by PREFIX. It used
to split on the first underscore, which made any PREFIX containing one
undispatchable: `face_recognition` resolved to a plugin called `face`, matched
nothing, and the tool was reported as unknown. This pins the longest-prefix
behaviour and that `dispatch` and `owns` agree, which `owns`'s docstring
promises.

Importing main.py needs the ROS stubs, hence vision_stubs.
"""

from __future__ import annotations

import sys
import types

import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (installs stubs, sets sys.path)


@pytest.fixture(scope="module")
def bundle_cls():
    """Import main.py without its optional heavy dependencies."""
    for name in ("yaml",):
        if name not in sys.modules:
            pytest.importorskip(name)
    import main
    return main.PerceptionBundle


class _StubPlugin:
    def __init__(self, prefix: str, tools: list[str] | None = None):
        self.PREFIX = prefix
        self._tools = tools or [prefix]
        self.calls: list[tuple[str, dict]] = []

    def get_tools(self):
        return [{"name": name} for name in self._tools]

    def dispatch(self, name, args):
        self.calls.append((name, args))
        return {"plugin": self.PREFIX, "name": name}


def _bundle(bundle_cls, *plugins):
    """A bundle with no config, wired to stub plugins directly."""
    bundle = bundle_cls.__new__(bundle_cls)
    bundle._plugins = list(plugins)
    return bundle


def test_single_word_prefix_still_resolves(bundle_cls):
    ocr = _StubPlugin("ocr")
    bundle = _bundle(bundle_cls, ocr)
    assert bundle.dispatch("ocr", {"action": "info"}) == {"plugin": "ocr", "name": "ocr"}
    assert ocr.calls[0][0] == "ocr"
    assert bundle.owns("ocr") is True


def test_underscored_prefix_resolves(bundle_cls):
    """The regression this resolver exists for."""
    face = _StubPlugin("face_recognition")
    bundle = _bundle(bundle_cls, face)
    result = bundle.dispatch("face_recognition", {"action": "info"})
    assert result == {"plugin": "face_recognition", "name": "face_recognition"}
    assert face.calls[0][0] == "face_recognition"
    assert bundle.owns("face_recognition") is True


def test_sub_tool_name_is_stripped_of_its_prefix(bundle_cls):
    asr = _StubPlugin("asr", ["asr", "asr_start"])
    bundle = _bundle(bundle_cls, asr)
    bundle.dispatch("asr_start", {})
    assert asr.calls[0][0] == "start"


def test_underscored_prefix_sub_tool(bundle_cls):
    face = _StubPlugin("face_recognition", ["face_recognition", "register"])
    bundle = _bundle(bundle_cls, face)
    bundle.dispatch("face_recognition_register", {})
    assert face.calls[0][0] == "register"


def test_longest_prefix_wins(bundle_cls):
    """A shorter prefix that is a prefix of a longer one must not shadow it."""
    face = _StubPlugin("face")
    face_recognition = _StubPlugin("face_recognition")
    for order in ((face, face_recognition), (face_recognition, face)):
        bundle = _bundle(bundle_cls, *order)
        assert bundle.dispatch("face_recognition", {})["plugin"] == "face_recognition"
        assert bundle.dispatch("face", {})["plugin"] == "face"


def test_unknown_tool_is_not_owned_and_dispatches_to_nothing(bundle_cls):
    bundle = _bundle(bundle_cls, _StubPlugin("ocr"))
    assert bundle.dispatch("lidar_scan", {}) is None
    assert bundle.owns("lidar_scan") is False


def test_partial_prefix_match_is_not_enough(bundle_cls):
    """`facelift` must not be routed to the `face` plugin."""
    bundle = _bundle(bundle_cls, _StubPlugin("face"))
    assert bundle.owns("facelift") is False
    assert bundle.dispatch("facelift", {}) is None


def test_owns_and_dispatch_never_disagree(bundle_cls):
    bundle = _bundle(
        bundle_cls,
        _StubPlugin("asr"), _StubPlugin("tts"), _StubPlugin("ocr"),
        _StubPlugin("vop"), _StubPlugin("face_recognition"),
    )
    names = [
        "asr", "asr_start", "tts", "tts_speak", "ocr", "vop",
        "face_recognition", "face_recognition_register_user_photo",
        "face", "facelift", "unknown_tool", "",
    ]
    for name in names:
        owned = bundle.owns(name)
        dispatched = bundle.dispatch(name, {}) is not None
        assert owned == dispatched, name


def test_get_all_tools_prefixes_sub_tools_only(bundle_cls):
    """A tool named exactly PREFIX keeps its name; others get PREFIX_ prepended.

    So a plugin declares sub-tools by their bare name (`start`, not
    `asr_start`), and every emitted name must route back to that plugin.
    """
    bundle = _bundle(
        bundle_cls,
        _StubPlugin("face_recognition", ["face_recognition"]),
        _StubPlugin("asr", ["asr", "start"]),
    )
    names = {tool["name"] for tool in bundle.get_all_tools()}
    assert names == {"face_recognition", "asr", "asr_start"}
    for name in names:
        assert bundle.owns(name), name
