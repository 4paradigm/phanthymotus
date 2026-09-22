"""The navi card's contract with agent-core and with the driver below it.

The behaviour is tested in test_navi_policy.py; this covers the card itself —
what it refuses, what it declares, and what it tells an operator when it is
running at reduced precision. No ROS: everything here stops before the card
opens a publisher, which is exactly the set of paths where a mistake produces
"the canvas looks fine and the robot does nothing".
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navi import NaviPlugin  # noqa: E402
from plugins.navi import plugin as navi_plugin  # noqa: E402


def _descriptor(mode="twist", dof=6, **over):
    out = {
        "control_interface": "motus.control/1",
        "mode": mode,
        "dof": dof,
        "joint_names": ["vx", "vy", "vz", "wx", "wy", "wz"][:dof],
        "units": {"linear": "m/s", "angular": "rad/s", "time": "s"},
        "limits": {"lower": [-1.0] * dof, "upper": [1.0] * dof},
        "rate": {"max_hz": 20, "expected_hz": 10, "watchdog_ms": 300},
        "force_torque": None,
    }
    out.update(over)
    return out


def _card(**cfg):
    return NaviPlugin(cfg, executor=None)


# ── refusing to start ────────────────────────────────────────────────────────

def test_no_downstream_card_is_refused_with_the_wiring_to_fix():
    result = _card().dispatch("navi", {"action": "start"})
    assert result["state"] == "error"
    assert "control/velocity" in result["message"]


def test_a_joint_card_downstream_is_refused():
    """This card produces chassis velocities. Wired to an arm, the dimensions
    might even agree — `mode` is what catches it, and refusing at start beats
    discovering it one command at a time at 10 Hz."""
    result = _card().dispatch("navi", {
        "action": "start",
        "control_interface": _descriptor(mode="joint_position"),
    })
    assert result["state"] == "error"
    assert "twist" in result["message"]


def test_a_downstream_of_the_wrong_width_is_refused():
    result = _card().dispatch("navi", {
        "action": "start", "control_interface": _descriptor(dof=4),
    })
    assert result["state"] == "error"


def test_the_refusal_says_strategy_not_model():
    """`negotiate` was written for the vla card and says 「模型」 by default.
    Telling a navigation operator that "模型输出 6 维" sends them looking for a
    checkpoint that does not exist."""
    result = _card().dispatch("navi", {
        "action": "start", "control_interface": _descriptor(dof=4),
    })
    assert "策略" in result["message"]
    assert "模型" not in result["message"]


def test_navigate_to_without_a_target_is_refused():
    assert _card().dispatch("navi", {"action": "navigate_to"})["state"] == "error"


def test_navigate_to_on_a_stopped_card_says_so():
    result = _card().dispatch("navi", {"action": "navigate_to", "target": "chair"})
    assert result["state"] == "idle"


# ── what it declares ─────────────────────────────────────────────────────────

def _tool():
    return _card().get_tools()[0]


def test_it_declares_its_output_topic_before_starting():
    """A card that names no topic until it runs leaves the card downstream with
    "连线缺少 topic" the moment this one fails to start — which reads as a
    wiring problem on a canvas that is wired correctly."""
    out = _tool()["topic_out"][0]
    assert out["topic"] == navi_plugin.DEFAULT_TOPIC
    assert out["format"] == "control/velocity"


def test_the_capabilities_declare_a_control_mode():
    """`negotiate` refuses anything that does not — matching dimensions do not
    mean matching action spaces."""
    assert _card()._capabilities()["control_mode"] == "twist"
    assert _card()._capabilities()["action_dim"] == 6


def test_start_and_stop_are_not_offered_to_the_model():
    params = _tool()["inputSchema"]["x-action-params"]
    assert "start" not in params and "stop" not in params
    assert "navigate_to" in params


def test_info_is_in_the_action_enum():
    """agent-core probes liveness by calling the tools whose enum contains it;
    a card without it stays permanently offline."""
    assert "info" in _tool()["inputSchema"]["properties"]["action"]["enum"]


def test_navigate_to_declares_completion():
    """Unlike the vla card. A policy runs until stopped; arriving somewhere is
    a finite action with an answer."""
    completion = _tool()["inputSchema"]["x-completion"]
    assert completion["actions"] == ["navigate_to"]


def test_the_interrupt_hooks_stop_navigation():
    hooks = _tool()["inputSchema"]["x-hooks"]
    assert hooks["on_interrupt_all"]["action"] == "stop_navigation"
    assert hooks["on_interrupt_motion"]["action"] == "stop_navigation"


def test_it_claims_the_base_resource():
    assert _tool()["inputSchema"]["x-resource"] == ["base"]


def test_the_prefix_has_no_underscore():
    assert "_" not in NaviPlugin.PREFIX


def test_it_declares_all_three_input_formats():
    formats = {entry["format"] for entry in _tool()["topic_in"]}
    assert formats == {"data/json", "image/depth-zlib", "state/odom"}


# ── degradation has to be visible ────────────────────────────────────────────

def test_a_card_with_no_odometry_says_it_has_no_stuck_protection():
    """Otherwise it looks identical to one that has it, and the difference only
    shows up as a robot leaning into a wall."""
    card = _card()
    card._binding = {"objects": "/cam/objects", "depth_map": "/cam/visual_depth",
                     "odom": ""}
    assert any("卡死" in note for note in card._degradations())


def test_a_card_with_only_the_summary_says_its_distances_are_coarse():
    card = _card()
    card._binding = {"objects": "/cam/objects", "depth_summary": "/cam/sum",
                     "depth_map": "", "odom": "/r1/state/odom"}
    assert any("深度摘要" in note for note in card._degradations())


def test_a_fully_wired_card_reports_no_degradation():
    card = _card()
    card._binding = {"objects": "/cam/objects", "depth_map": "/cam/visual_depth",
                     "odom": "/r1/state/odom"}
    assert card._degradations() == []


def test_info_carries_the_degradations_and_the_target():
    card = _card()
    card._binding = {"objects": "/o", "depth_map": "", "odom": ""}
    info = card.dispatch("navi", {"action": "info"})
    assert info["state"] == "idle"
    assert len(info["degraded"]) == 2


# ── config ───────────────────────────────────────────────────────────────────

def test_config_from_yaml_reaches_the_policy():
    card = _card(stop_distance_m=2.5, vx_max=0.2)
    assert card._config.stop_distance_m == 2.5
    assert card._config.vx_max == 0.2


def test_unknown_config_keys_are_ignored_rather_than_crashing():
    """`topic`, `priority` and friends live in the same dict and are not policy
    parameters."""
    card = _card(topic="/x", priority=10, resource="base", enabled=True)
    assert card._topic == "/x"


def test_the_config_action_updates_a_live_card():
    card = _card()
    card.dispatch("navi", {"action": "config", "vx_max": 0.15})
    assert card._config.vx_max == 0.15
