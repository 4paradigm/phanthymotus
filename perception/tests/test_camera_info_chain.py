"""motus.camera/1 through a processor — the middle of the chain.

A camera card only emits a declaration; a navigation card only reads one. A
perception processor does both, and the interesting part is the middle: it must
rewrite the parts its own processing changed, **and only those**.

`visual_depth` is the concrete case. It resizes 1280x720 into 640x480 before
publishing, so the declared size changes, `K` has to be rescaled, and
`half_fov_rad` must **not** change — `cv2.resize` stretches the frame rather than
cropping it, so the same angular extent is still in the picture.

Why it matters: that field of view used to be typed into navi's config by hand.
On r1_sz it read 0.55 rad against a lens measuring 0.888, and the metric
avoidance corridor came out 1.86 m wide — wider than any door. Every doorframe
read as dead ahead and the robot turned away from openings it fitted through,
while the depth map reported clear.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plugins.camera_info import (SCHEMA, camera_id, for_topic,  # noqa: E402
                                 inherit)

CAM_TOPIC = "/ubuntu/camera/main"


def _upstream(**over):
    out = {"schema": SCHEMA, "topic": CAM_TOPIC, "format": "image/jpeg",
           "id": "unitree/r1/camera_main", "width": 1280, "height": 720,
           "distortion_model": "unknown", "D": None, "K": None,
           "half_fov_rad": 0.888, "half_fov_v_rad": None,
           "source": "measured", "measured_on": "r1_sz, 2026-09-23",
           "pipeline": ["unitree/r1/camera_main"], "vendor": {"note": "超广角"}}
    out.update(over)
    return out


def _depth_entry(**over):
    entries = inherit(_upstream(**over), topic=f"{CAM_TOPIC}/visual_depth",
                      fmt="image/depth-zlib", stage="perception/visual_depth",
                      width=640, height=480)
    return entries[0]


# ── what a resize changes, and what it must not ──────────────────────────────

def test_the_angle_survives_a_resize_but_the_size_does_not():
    """`cv2.resize` stretches the whole frame, so the same angular extent is
    still in the picture. The declaration describes the image *this* port
    publishes, which is the resized one."""
    entry = _depth_entry()
    assert entry["half_fov_rad"] == 0.888
    assert (entry["width"], entry["height"]) == (640, 480)


def test_the_identity_survives_so_downstream_can_still_join_on_it():
    """Three stages later, navi compares the id on its two inputs to check both
    are the same lens. That only works if every stage carries it through."""
    entry = _depth_entry()
    assert camera_id(entry) == "unitree/r1/camera_main"


def test_the_pipeline_records_who_touched_it():
    entry = _depth_entry()
    assert entry["pipeline"] == ["unitree/r1/camera_main", "perception/visual_depth"]


def test_the_source_becomes_inherited_rather_than_still_claiming_measured():
    """This stage did not measure anything. `measured_on` is kept because it is
    about the original measurement and stays true."""
    entry = _depth_entry()
    assert entry["source"] == "inherited"
    assert entry["measured_on"] == "r1_sz, 2026-09-23"


def test_an_unknown_upstream_is_not_relabelled_as_inherited():
    """"Nobody knows" passed down must keep looking like "nobody knows", or it
    reads as a number having been handed over."""
    entry = _depth_entry(half_fov_rad=None, source="unknown")
    assert entry["half_fov_rad"] is None
    assert entry["source"] == "unknown"


# ── K in pixels, and the trap that comes with it ─────────────────────────────

def test_K_is_rescaled_so_the_derived_angle_comes_out_the_same():
    """The trap. `fx` is in pixels and a consumer derives the angle as
    `atan((width/2)/fx)`. Halve the width and pass `K` through untouched and both
    fields stay individually plausible while their ratio is wrong by exactly the
    resize factor — the same silent, confidently wrong geometry this format was
    written against.
    """
    import math

    fx = fy = 700.0
    up = _upstream(K=[fx, 0.0, 640.0, 0.0, fy, 360.0, 0.0, 0.0, 1.0])
    before = math.atan((up["width"] / 2) / fx)

    entry = inherit(up, topic="/d", fmt="image/depth-zlib",
                    stage="perception/visual_depth", width=640, height=480)[0]
    after = math.atan((entry["width"] / 2) / entry["K"][0])
    assert after == pytest.approx(before), "resize 之后 K 和尺寸对不上了"


def test_the_two_axes_scale_independently():
    """1280x720 into 640x480 is not a uniform scale, which is exactly why `fx`
    and `fy` are separate entries."""
    up = _upstream(K=[700.0, 0.0, 640.0, 0.0, 700.0, 360.0, 0.0, 0.0, 1.0])
    K = inherit(up, topic="/d", fmt="image/depth-zlib", stage="s",
                width=640, height=480)[0]["K"]
    assert K[0] == pytest.approx(350.0)          # sx = 0.5
    assert K[4] == pytest.approx(700.0 * 480 / 720)
    assert K[8] == 1.0, "底行不是像素量，不该缩放"


def test_D_survives_unchanged_because_it_is_dimensionless():
    up = _upstream(D=[-0.31, 0.11, 0.0, 0.0])
    assert _depth_entry_from(up)["D"] == [-0.31, 0.11, 0.0, 0.0]


def _depth_entry_from(upstream):
    return inherit(upstream, topic="/d", fmt="image/depth-zlib", stage="s",
                   width=640, height=480)[0]


def test_no_resize_leaves_K_alone():
    """vop does not touch the picture's geometry, so nothing should be recomputed
    for it — recomputation is where rounding and mistakes live."""
    K = [700.0, 0.0, 640.0, 0.0, 700.0, 360.0, 0.0, 0.0, 1.0]
    entry = inherit(_upstream(K=K), topic="/o", fmt="data/json",
                    stage="perception/vop")[0]
    assert entry["K"] == K
    assert (entry["width"], entry["height"]) == (1280, 720)


# ── nothing upstream means nothing downstream, not an invented identity ──────

def test_no_upstream_declaration_produces_no_declaration():
    """The format requires a non-empty `id` and this stage cannot invent one — a
    camera's identity is not derivable from a topic name. A consumer seeing no
    entry falls back conservatively and says so, which is right; a consumer
    seeing a made-up id would compare it against another made-up one and
    conclude two different lenses are the same.
    """
    assert inherit({}, topic="/d", fmt="x", stage="s") == []
    assert inherit(None, topic="/d", fmt="x", stage="s") == []


def test_a_foreign_schema_is_not_passed_on_as_ours():
    assert inherit(_upstream(schema="motus.camera/2"), topic="/d", fmt="x",
                   stage="s") == []


def test_an_upstream_without_an_id_is_not_passed_on():
    assert inherit(_upstream(id=""), topic="/d", fmt="x", stage="s") == []


# ── joining on topic ────────────────────────────────────────────────────────

def test_the_input_topic_selects_which_upstream_declaration_applies():
    declarations = {CAM_TOPIC: _upstream(),
                    "/ubuntu/camera/side": _upstream(id="unitree/r1/camera_left")}
    assert camera_id(for_topic(declarations, CAM_TOPIC)) == "unitree/r1/camera_main"
    assert camera_id(for_topic(declarations, "/ubuntu/camera/side")) == \
        "unitree/r1/camera_left"
    assert for_topic(declarations, "/nothing") == {}


def test_a_raw_info_list_is_accepted_too():
    assert camera_id(for_topic([_upstream()], CAM_TOPIC)) == "unitree/r1/camera_main"


# ── the card ────────────────────────────────────────────────────────────────

def _plugin():
    from plugins import visual_depth as depth_plugin
    return depth_plugin.VideoDepthPerceptionPlugin({}, "ubuntu", None)


def test_the_card_passes_it_on_for_both_output_ports():
    """The depth map and the summary are the same picture, so both carry the same
    optics — a consumer wired only to the summary should not have to guess."""
    plugin = _plugin()
    plugin._upstream_camera[CAM_TOPIC] = _upstream()
    out, note = plugin._camera_info(None, CAM_TOPIC, {},
                                    f"{CAM_TOPIC}/visual_depth",
                                    f"{CAM_TOPIC}/visual_depth_summary")
    assert note == ""
    assert [e["topic"] for e in out] == [f"{CAM_TOPIC}/visual_depth",
                                        f"{CAM_TOPIC}/visual_depth_summary"]
    assert all(e["half_fov_rad"] == 0.888 for e in out)
    assert all((e["width"], e["height"]) == (640, 480) for e in out)


def test_the_card_says_whose_problem_a_missing_declaration_is():
    """"The camera declared nothing" and "this card dropped it" look identical
    from downstream, and only the first is somebody else's to fix."""
    plugin = _plugin()
    out, note = plugin._camera_info(None, CAM_TOPIC, {}, "/d", "/s")
    assert out == []
    assert "上游相机没有声明" in note


# ── vop: the other half of what navi needs ───────────────────────────────────

def _vop():
    from plugins import vop as vop_plugin
    plugin = vop_plugin.VideoObjectPerceptionPlugin({}, "ubuntu", None)
    plugin._vocabulary = ["person"]
    return plugin, vop_plugin


def test_vop_passes_the_optics_through_without_touching_the_geometry():
    """vop reads the frame and publishes text — nothing about the picture's
    geometry changes — so only `pipeline` grows. `width`/`height` and `K` must
    come through exactly as they arrived, because recomputing what did not change
    is where mistakes live.
    """
    plugin, mod = _vop()
    K = [700.0, 0.0, 640.0, 0.0, 700.0, 360.0, 0.0, 0.0, 1.0]
    plugin._upstream_camera[CAM_TOPIC] = _upstream(K=K)

    info = plugin.dispatch("vop", {"action": "info", "input_topic": CAM_TOPIC})
    entry = info["camera_info"][0]

    assert entry["topic"] == mod.output_topic_for(CAM_TOPIC)
    assert (entry["width"], entry["height"]) == (1280, 720)
    assert entry["K"] == K
    assert entry["half_fov_rad"] == 0.888
    assert entry["pipeline"] == ["unitree/r1/camera_main", "perception/vop"]


def test_vop_and_the_depth_map_carry_the_same_camera_identity():
    """What makes navi's same-camera check possible at all.

    vop reports a *normalised* lateral offset and the depth map is a grid of
    distances; navi turns both into metres with one field of view, which is only
    correct if both come from the same lens. Inputs are bound by what they carry
    — deliberately — so camera A's vop paired with camera B's depth has always
    been wirable, and would produce confidently wrong distances with nothing in
    any log. Both sides declaring the same `id` is what turns that into a
    comparison.
    """
    plugin, _ = _vop()
    plugin._upstream_camera[CAM_TOPIC] = _upstream()
    objects = plugin.dispatch("vop", {"action": "info",
                                      "input_topic": CAM_TOPIC})["camera_info"][0]
    assert camera_id(objects) == camera_id(_depth_entry())


def test_vop_says_why_a_downstream_angle_will_be_missing():
    """Without a field of view, `position` is a dimensionless number to everyone
    downstream — and the avoidance corridor's width is computed from exactly that
    angle. Saying so here is cheaper than diagnosing it at a doorway."""
    plugin, _ = _vop()
    info = plugin.dispatch("vop", {"action": "info", "input_topic": CAM_TOPIC})
    assert "camera_info" not in info
    assert "归一化" in info["camera_info_note"]


def test_a_retired_node_takes_its_camera_with_it():
    """A re-wired card answering with the optics of a camera it is no longer fed
    by is worse than answering with nothing, because downstream cannot tell the
    two apart."""
    plugin, _ = _vop()
    plugin._upstream_camera[CAM_TOPIC] = _upstream()
    plugin._retire_node(CAM_TOPIC)
    assert CAM_TOPIC not in plugin._upstream_camera


# ── 加载中也要答得出声明 ─────────────────────────────────────────────────────

def test_vop_declares_while_its_engine_is_still_loading():
    """The bug this was found by, on a *cold* r1_sz.

    perception's vision cards answer `start` with `loading` while a TensorRT
    engine comes up in the background, and agent-core carries on and records
    whatever `info()` returns. That reply used to be a bare stub — no
    `topic_out`, no `camera_info` — so a consumer that started in that window
    got no camera declaration at all, and nothing said so. navi lost both its
    box path and the corridor's real geometry.

    It only ever reproduced on a cold container: a warm one starts immediately
    and never passes through the loading reply at all.

    The declaration is recorded at `start`, before loading begins, so it was
    always available — the early return simply dropped it.
    """
    plugin, _ = _vop()
    plugin._upstream_camera[CAM_TOPIC] = _upstream()
    plugin._model_loading = True

    info = plugin.dispatch("vop", {"action": "info", "input_topic": CAM_TOPIC})
    assert info["state"] == "loading"
    assert camera_id(info["camera_info"][0]) == "unitree/r1/camera_main"


def test_visual_depth_declares_while_its_engine_is_still_loading():
    plugin = _plugin()
    plugin._upstream_camera[CAM_TOPIC] = _upstream()
    plugin._model_loading = True

    info = plugin.dispatch("visual_depth", {"action": "info",
                                            "input_topic": CAM_TOPIC})
    assert info["state"] == "loading"
    assert [e["topic"] for e in info["camera_info"]] == [
        f"{CAM_TOPIC}/visual_depth", f"{CAM_TOPIC}/visual_depth_summary"]


def test_a_loading_card_with_no_upstream_declares_nothing_rather_than_a_stub():
    plugin, _ = _vop()
    plugin._model_loading = True
    info = plugin.dispatch("vop", {"action": "info", "input_topic": CAM_TOPIC})
    assert "camera_info" not in info
