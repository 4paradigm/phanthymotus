"""
tests/test_barcode_plugin.py — barcode card lifecycle and scan dispatch.

ROS stubs come from vision_stubs (installed by conftest before collection).
The decoder itself is stubbed: everything version-sensitive about zxing-cpp
lives in plugins/barcode_runtime.py and is tested in
tests/test_barcode_runtime.py, so this file never needs the library present.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import threading
import time

import pytest

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _wait_until,
    frame_bytes,
)

from utils.qos import CAMERA_QOS  # noqa: E402
import plugins.barcode as barcode_plugin  # noqa: E402
from plugins.barcode_runtime import DecodeError, DecoderUnavailable  # noqa: E402


class _DecoderProbe:
    """Stands in for barcode_runtime.decode_codes; records every call."""

    def __init__(self, codes=None, error=None, per_frame=None):
        self.codes = codes if codes is not None else [
            {"format": "QRCode", "text": "hello", "valid": True}
        ]
        self.error = error
        # {frame_bytes: codes} — lets a test make only the newest frame decode.
        self.per_frame = per_frame or {}
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, image_bytes, opts):
        with self.lock:
            self.calls.append({"bytes": image_bytes, "opts": dict(opts)})
        if self.error is not None:
            raise self.error
        codes = self.per_frame.get(image_bytes, self.codes) if self.per_frame else self.codes
        return {
            "codes": list(codes),
            "image": {"width": 200, "height": 100},
            "elapsed_ms": 1,
        }


def _make_plugin(monkeypatch, decoder=None, cfg=None, decoder_status=None):
    decoder = decoder or _DecoderProbe()
    monkeypatch.setattr(barcode_plugin, "decode_codes", decoder)
    monkeypatch.setattr(
        barcode_plugin, "decoder_status",
        lambda: decoder_status or {"available": True, "library": "zxing-cpp",
                                   "version": "fake", "formats": ["QRCode"]},
    )
    executor = _FakeExecutor()
    plugin = barcode_plugin.BarcodeScanPlugin(dict(cfg or {}), executor)
    return plugin, executor, decoder


@pytest.fixture
def card(monkeypatch, tmp_path):
    # image_roots points at tmp_path for the same reason every other
    # *_by_photo test in this suite does it: load_image_bytes confines paths
    # to the configured roots, and a pytest tmp dir is under none of them.
    plugin, executor, decoder = _make_plugin(
        monkeypatch, cfg={"image_roots": [str(tmp_path)]})
    yield plugin, executor, decoder
    plugin.dispatch("barcode", {"action": "stop"})


def _photo(tmp_path, name="label.jpg", payload=None):
    path = tmp_path / name
    path.write_bytes(payload if payload is not None else frame_bytes())
    return str(path)


def _start(plugin, executor, topic="", instance_id="", count=1):
    args = {"action": "start"}
    if topic:
        args["input_topic"] = topic
    if instance_id:
        args["instance_id"] = instance_id
    result = plugin.dispatch("barcode", args)
    assert _wait_until(
        lambda: len(executor.nodes) >= count
        and all(node.state == "running" for node in executor.nodes)
    ), "node never came up"
    return result


def _feed(node, payloads):
    callback = node.subscriptions[0].callback
    for payload in payloads:
        callback(_FakeCompressedImage(payload))


# ── the card's declared shape ────────────────────────────────────────────────

def test_card_is_an_actuator_with_an_input_and_no_output():
    """canvas.js derives the port layout from exactly this.

    An output port would promise a stream of scan results, which is the thing
    this card deliberately does not produce (see the module docstring).
    """
    tool = barcode_plugin.TOOLS[0]
    assert tool["type"] == "actuator"
    assert tool["topic_in"] == [
        {"format": "image/jpeg", "desc": "camera image input (optional)"}
    ]
    assert "topic_out" not in tool


def test_every_action_is_documented_in_x_action_params():
    tool = barcode_plugin.TOOLS[0]
    actions = set(tool["inputSchema"]["properties"]["action"]["enum"])
    documented = set(tool["inputSchema"]["x-action-params"])
    assert actions == documented


def test_description_warns_that_payloads_are_untrusted():
    """A decoded payload is attacker-chosen text going into an LLM prompt."""
    assert "untrusted" in barcode_plugin.TOOLS[0]["description"]


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_start_subscribes_with_the_shared_camera_qos(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    assert executor.nodes[0].subscriptions[0].qos is CAMERA_QOS


def test_start_without_a_topic_subscribes_to_nothing(card):
    plugin, executor, _ = card
    result = _start(plugin, executor)
    assert executor.nodes[0].subscriptions == []
    assert result["state"] == "running"
    assert result["topic_in"] == []


def test_the_card_publishes_nothing(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    assert executor.nodes[0].publishers == []


def test_restart_on_a_new_topic_disposes_the_old_node(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a", instance_id="x")
    old = executor.nodes[0]
    plugin.dispatch("barcode", {"action": "start", "input_topic": "/cam/b",
                                "instance_id": "x"})
    assert _wait_until(
        lambda: executor.nodes and executor.nodes[-1].subscriptions
        and executor.nodes[-1].subscriptions[0].topic == "/cam/b"
    )
    assert old not in executor.nodes and old.destroyed
    assert len(executor.nodes) == 1


def test_restart_on_the_same_topic_is_idempotent(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    plugin.dispatch("barcode", {"action": "start", "input_topic": "/cam/a"})
    assert executor.nodes == [node]
    assert len(node.subscriptions) == 1


def test_stop_disposes_the_node(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    plugin.dispatch("barcode", {"action": "stop"})
    assert executor.nodes == [] and node.destroyed


def test_stop_drops_the_window(card):
    """Frames from before a stop must not answer a scan after the next start."""
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    _feed(node, [frame_bytes()])
    assert node.recent_frames(10.0)
    node.stop()
    assert node.recent_frames(10.0) == []


def test_concurrent_start_stop_leaves_nothing_running(monkeypatch):
    """The canvas does config→start→stop→config→start within seconds.

    What this guards is the orphan window the README describes: a node that
    got started but never registered keeps a live subscription that nothing
    can reach, and it survives until the process exits.
    """
    plugin, executor, _ = _make_plugin(monkeypatch)
    errors = []

    def hammer(action):
        for _ in range(20):
            try:
                plugin.dispatch("barcode", {"action": action,
                                            "input_topic": "/cam/a"})
            except Exception as error:  # noqa: BLE001
                errors.append(error)

    threads = [threading.Thread(target=hammer, args=(action,))
               for action in ("start", "stop", "start", "stop")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors, errors
    plugin.dispatch("barcode", {"action": "stop"})
    assert executor.nodes == []
    assert all(node.destroyed for node in _FakeNode.instances)


# ── info ─────────────────────────────────────────────────────────────────────

def test_info_reports_the_decoder_and_formats(card):
    plugin, _, _ = card
    info = plugin.dispatch("barcode", {"action": "info"})
    assert info["decoder"]["available"] is True
    assert info["decoder"]["formats"] == ["QRCode"]
    assert "AprilTag" in info["markers"]       # says so rather than staying silent


def test_info_without_a_decoder_is_an_error_state_with_the_reason(monkeypatch):
    plugin, _, _ = _make_plugin(
        monkeypatch,
        decoder_status={"available": False, "detail": "zxing-cpp is not installed"},
    )
    info = plugin.dispatch("barcode", {"action": "info"})
    assert info["state"] == "error"
    assert "not installed" in info["error"]
    assert "not installed" in info["desc"]


def test_info_lists_instances(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a", instance_id="a")
    _start(plugin, executor, "/cam/b", instance_id="b", count=2)
    info = plugin.dispatch("barcode", {"action": "info"})
    assert set(info["instances"]) == {"a", "b"}
    assert {entry["topic"] for entry in info["topic_in"]} == {"/cam/a", "/cam/b"}


# ── scan_by_stream ───────────────────────────────────────────────────────────

def test_scan_by_stream_reads_the_newest_frame(card):
    plugin, executor, decoder = card
    _start(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    _feed(node, [frame_bytes(width=201), frame_bytes(width=202), frame_bytes(width=203)])

    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["ok"] is True and result["count"] == 1
    assert result["source"] == "stream:/cam/a"
    assert result["frames_examined"] == 1
    # Newest first, and it stopped as soon as one frame decoded.
    assert decoder.calls[0]["bytes"] == frame_bytes(width=203)
    assert len(decoder.calls) == 1


def test_scan_by_stream_walks_back_until_a_frame_decodes(monkeypatch):
    """Motion blur is why more than one frame is examined at all."""
    newest, middle, oldest = (frame_bytes(width=w) for w in (203, 202, 201))
    decoder = _DecoderProbe(
        codes=[],
        per_frame={
            newest: [],
            middle: [],
            oldest: [{"format": "QRCode", "text": "found", "valid": True}],
        },
    )
    plugin, executor, _ = _make_plugin(monkeypatch, decoder)
    _start(plugin, executor, "/cam/a")
    _feed(executor.nodes[0], [oldest, middle, newest])

    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["count"] == 1 and result["codes"][0]["text"] == "found"
    assert result["frames_examined"] == 3
    assert [call["bytes"] for call in decoder.calls] == [newest, middle, oldest]
    plugin.dispatch("barcode", {"action": "stop"})


def test_scan_by_stream_honours_max_frames_scanned(monkeypatch):
    decoder = _DecoderProbe(codes=[])
    plugin, executor, _ = _make_plugin(
        monkeypatch, decoder, cfg={"max_frames_scanned": 2})
    _start(plugin, executor, "/cam/a")
    _feed(executor.nodes[0], [frame_bytes(width=200 + n) for n in range(5)])
    plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert len(decoder.calls) == 2
    plugin.dispatch("barcode", {"action": "stop"})


def test_nothing_found_is_a_success_with_count_zero(monkeypatch):
    """`ok: false` would make an LLM retry a scan that worked fine."""
    plugin, executor, _ = _make_plugin(monkeypatch, _DecoderProbe(codes=[]))
    _start(plugin, executor, "/cam/a")
    _feed(executor.nodes[0], [frame_bytes()])
    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["ok"] is True and result["count"] == 0 and result["codes"] == []
    assert "no barcode found" in result["detail"]
    plugin.dispatch("barcode", {"action": "stop"})


def test_scan_by_stream_without_frames_says_nothing_is_publishing(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["ok"] is False
    assert result["reason"] == barcode_plugin.REASON_NO_FRAMES
    assert "/cam/a" in result["detail"]


def test_stale_frames_are_outside_the_window(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    with node._window_lock:
        node._window.append((frame_bytes(), time.time() - 30.0))
    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["reason"] == barcode_plugin.REASON_NO_FRAMES


def test_window_s_is_capped_by_the_card_config(card):
    """A caller asking for an hour of history gets the configured bound."""
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    with node._window_lock:
        node._window.append((frame_bytes(), time.time() - 5.0))
    result = plugin.dispatch(
        "barcode", {"action": "scan_by_stream", "window_s": 3600})
    assert result["window_s"] == barcode_plugin.DEFAULT_WINDOW_S
    assert result["reason"] == barcode_plugin.REASON_NO_FRAMES


def test_scan_by_stream_on_a_topicless_card_points_at_the_photo_actions(card):
    plugin, executor, _ = card
    _start(plugin, executor)
    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["reason"] == barcode_plugin.REASON_NO_INPUT_TOPIC
    assert "scan_by_photo" in result["detail"]


def test_scan_by_stream_with_no_instance_running(card):
    plugin, _, _ = card
    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["ok"] is False
    assert "start the card" in result["detail"]


def test_several_instances_require_an_instance_id(card):
    """Guessing would confidently answer about the wrong camera."""
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a", instance_id="a")
    _start(plugin, executor, "/cam/b", instance_id="b", count=2)
    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["ok"] is False
    assert result["instances"] == ["a", "b"]
    assert "instance_id" in result["detail"]


def test_undecodable_frames_report_the_decode_failure(monkeypatch):
    plugin, executor, _ = _make_plugin(
        monkeypatch, _DecoderProbe(error=DecodeError("not a picture")))
    _start(plugin, executor, "/cam/a")
    _feed(executor.nodes[0], [frame_bytes()])
    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["ok"] is False and result["reason"] == "decode_failed"
    assert result["frames_examined"] == 1
    plugin.dispatch("barcode", {"action": "stop"})


def test_missing_decoder_surfaces_on_a_stream_scan(monkeypatch):
    plugin, executor, _ = _make_plugin(
        monkeypatch, _DecoderProbe(error=DecoderUnavailable("no zxing here")))
    _start(plugin, executor, "/cam/a")
    _feed(executor.nodes[0], [frame_bytes()])
    result = plugin.dispatch("barcode", {"action": "scan_by_stream"})
    assert result["reason"] == "decoder_unavailable"
    plugin.dispatch("barcode", {"action": "stop"})


# ── scan_by_photo / scan_by_url ──────────────────────────────────────────────

def test_scan_by_photo_needs_no_running_instance(card, tmp_path):
    """One-shot scanning is useful with no camera wired at all."""
    plugin, executor, decoder = card
    path = _photo(tmp_path)
    result = plugin.dispatch("barcode", {
        "action": "scan_by_photo", "image_path": path,
    })
    assert executor.nodes == []        # nothing was started
    assert result["ok"] is True and result["count"] == 1
    assert result["source"] == path
    assert result["image"] == {"width": 200, "height": 100}


def test_scan_by_photo_outside_the_roots_is_refused(card):
    plugin, _, _ = card
    result = plugin.dispatch("barcode", {
        "action": "scan_by_photo", "image_path": "/etc/shadow",
    })
    assert result["ok"] is False and result["reason"] == "bad_input"
    # The shared rejection names the way in, and the right action to use.
    assert "file/upload" in result["detail"]
    assert "scan_by_url" in result["detail"]


def test_scan_by_photo_without_an_argument(card):
    plugin, _, _ = card
    result = plugin.dispatch("barcode", {"action": "scan_by_photo"})
    assert result["ok"] is False and result["reason"] == "bad_input"


def test_scan_by_url_rejects_a_non_http_scheme(card):
    plugin, _, _ = card
    result = plugin.dispatch("barcode", {
        "action": "scan_by_url", "url": "file:///etc/shadow",
    })
    assert result["ok"] is False and result["reason"] == "bad_input"


def test_base64_input_is_refused_with_the_alternative(card):
    plugin, _, _ = card
    result = plugin.dispatch("barcode", {
        "action": "scan_by_photo", "image_b64": "AAAA",
    })
    assert result["reason"] == "bad_input" and "image_b64" in result["detail"]


def test_photo_scan_with_no_code_is_a_success(monkeypatch, tmp_path):
    plugin, _, _ = _make_plugin(monkeypatch, _DecoderProbe(codes=[]),
                                cfg={"image_roots": [str(tmp_path)]})
    result = plugin.dispatch("barcode", {
        "action": "scan_by_photo", "image_path": _photo(tmp_path, "blank.jpg")})
    assert result["ok"] is True and result["count"] == 0


def test_hits_carry_the_untrusted_note(card, tmp_path):
    """The description is read once at registration; this travels with the
    payload it is about."""
    plugin, _, _ = card
    result = plugin.dispatch("barcode", {
        "action": "scan_by_photo", "image_path": _photo(tmp_path)})
    assert "untrusted" in result["note"]


# ── options plumbing ─────────────────────────────────────────────────────────

def test_yaml_only_knobs_still_reach_the_decoder(monkeypatch, tmp_path):
    """try_rotate and max_text_chars are off the card on purpose, but a
    config.yaml that sets them must still be honoured."""
    decoder = _DecoderProbe()
    plugin, _, _ = _make_plugin(
        monkeypatch, decoder,
        cfg={"formats": ["QRCode"], "try_rotate": False, "max_text_chars": 32,
             "image_roots": [str(tmp_path)]},
    )
    plugin.dispatch("barcode", {"action": "scan_by_photo",
                                "image_path": _photo(tmp_path)})
    opts = decoder.calls[0]["opts"]
    assert opts["formats"] == ["QRCode"]
    assert opts["try_rotate"] is False
    assert opts["max_text_chars"] == 32


def test_per_call_scan_mode_overrides_the_card(monkeypatch, tmp_path):
    decoder = _DecoderProbe()
    plugin, _, _ = _make_plugin(
        monkeypatch, decoder,
        cfg={"scan_mode": "1d", "image_roots": [str(tmp_path)]})
    monkeypatch.setattr(barcode_plugin, "resolve_scan_mode",
                        lambda mode: {"1d": ["Code128"], "qr": ["QRCode"]}[mode])
    plugin.dispatch("barcode", {"action": "scan_by_photo",
                                "image_path": _photo(tmp_path),
                                "scan_mode": "qr"})
    assert decoder.calls[0]["opts"]["formats"] == ["QRCode"]


def test_an_explicit_yaml_format_list_wins_over_the_mode(monkeypatch, tmp_path):
    """`formats` is the config.yaml escape hatch; somebody who wrote one meant it."""
    decoder = _DecoderProbe()
    plugin, _, _ = _make_plugin(
        monkeypatch, decoder,
        cfg={"formats": ["DataMatrix"], "scan_mode": "qr",
             "image_roots": [str(tmp_path)]})
    plugin.dispatch("barcode", {"action": "scan_by_photo",
                                "image_path": _photo(tmp_path)})
    assert decoder.calls[0]["opts"]["formats"] == ["DataMatrix"]


def test_scan_mode_is_resolved_against_this_builds_names(monkeypatch, tmp_path):
    """The point of the mode: the two zxing builds spell the aggregates
    differently, so the card stores `2d` and the runtime picks the name that
    exists here."""
    decoder = _DecoderProbe()
    plugin, _, _ = _make_plugin(monkeypatch, decoder,
                                cfg={"image_roots": [str(tmp_path)]})
    seen = []
    monkeypatch.setattr(barcode_plugin, "resolve_scan_mode",
                        lambda mode: seen.append(mode) or ["MatrixCodes"])
    plugin.dispatch("barcode", {"action": "scan_by_photo",
                                "image_path": _photo(tmp_path),
                                "scan_mode": "2d"})
    assert seen == ["2d"]
    assert decoder.calls[0]["opts"]["formats"] == ["MatrixCodes"]


def test_config_changes_the_live_window(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    plugin.dispatch("barcode", {"action": "config", "window_s": 2.5})
    assert node._window_seconds == 2.5


def test_instance_config_is_merged_per_instance(card):
    plugin, executor, _ = card
    _start(plugin, executor, "/cam/a", instance_id="a")
    plugin.dispatch("barcode", {"action": "config", "instance_id": "a",
                                "window_s": 0.5})
    assert executor.nodes[0]._window_seconds == 0.5
    result = plugin.dispatch("barcode", {"action": "info", "instance_id": "a"})
    assert result["state"] == "running"


# ── MCP contract ─────────────────────────────────────────────────────────────

def test_dispatch_returns_plain_dicts(card, tmp_path):
    """Never a pre-wrapped [{"type": "text", ...}] — the HTTP handler wraps."""
    plugin, executor, _ = card
    path = _photo(tmp_path)
    for args in (
        {"action": "info"},
        {"action": "start", "input_topic": "/cam/a"},
        {"action": "config", "window_s": 1.0},
        {"action": "scan_by_photo", "image_path": path},
        {"action": "scan_by_stream"},
        {"action": "stop"},
    ):
        result = plugin.dispatch("barcode", args)
        assert isinstance(result, dict), args


def test_unknown_action_returns_none(card):
    assert card[0].dispatch("barcode", {"action": "levitate"}) is None


def test_action_style_tool_names_dispatch(card, tmp_path):
    """`barcode_scan_by_photo` arrives as name="scan_by_photo", no action key.

    main.py strips the PREFIX and hands the remainder over, so a registry that
    split this card into one tool per action still reaches the same code.
    """
    plugin, _, _ = card
    result = plugin.dispatch("scan_by_photo", {"image_path": _photo(tmp_path)})
    assert result["ok"] is True
    assert plugin.dispatch("info", {})["decoder"]["available"] is True


# ── the config panel ─────────────────────────────────────────────────────────
#
# web/js/sidebar.js renders this; the three rules below came from reading the
# rendered form rather than the schema.

def test_config_offers_only_questions_an_operator_can_answer():
    """Expert knobs stay in config.yaml. A field nobody knows how to fill —
    "max symbols per frame" — is worse than no field."""
    props = barcode_plugin.TOOLS[0]["configSchema"]["properties"]
    assert set(props) == {"scan_mode", "window_s"}


def test_config_labels_and_help_are_english():
    """The config panel is English; the LLM-facing descriptions are not bound
    by this."""
    props = barcode_plugin.TOOLS[0]["configSchema"]["properties"]
    for key, field in props.items():
        text = f"{field.get('title', '')} {field.get('description', '')}"
        for option in field.get("oneOf", []):
            text += " " + option.get("title", "")
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in text), key


def test_code_type_choice_is_a_dropdown_not_a_free_text_list():
    """An array of strings renders as a comma-separated text box (with vop's
    "person, chair" placeholder, no less) — there is no multi-select widget.
    A mode with `oneOf` titles is a real dropdown, and it also keeps the
    unportable zxing format names out of a saved card."""
    field = barcode_plugin.TOOLS[0]["configSchema"]["properties"]["scan_mode"]
    assert field["type"] == "string"
    assert [entry["const"] for entry in field["oneOf"]] == ["all", "qr", "2d", "1d"]
    assert all(entry.get("title") for entry in field["oneOf"])


def test_every_config_field_has_a_short_title():
    """Without `title` the whole description becomes the label and the form
    turns into an essay."""
    props = barcode_plugin.TOOLS[0]["configSchema"]["properties"]
    for key, field in props.items():
        assert field.get("title"), key
        assert len(field["title"]) <= 40, key
