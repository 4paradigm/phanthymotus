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

# Minimal ROS message stubs. `_bind_inputs` imports these only to hand a type to
# create_subscription, and the point of these tests is the role assignment that
# happens before that — installing the two names is cheaper, and far less
# fragile, than requiring a ROS install to test a dictionary lookup.
import types as _types  # noqa: E402

for _mod, _name in (("sensor_msgs.msg", "CompressedImage"),
                    ("std_msgs.msg", "String")):
    _pkg = _mod.split(".")[0]
    sys.modules.setdefault(_pkg, _types.ModuleType(_pkg))
    _m = sys.modules.setdefault(_mod, _types.ModuleType(_mod))
    setattr(_m, _name, type(_name, (), {}))
    setattr(sys.modules[_pkg], "msg", _m)

# rclpy.qos, for `_sensor_qos`. Stubbed rather than skipped because the profile
# it builds is the whole point: a reliable subscriber silently receives nothing
# from these best-effort publishers, so "which reliability" is worth asserting.
_rclpy = sys.modules.setdefault("rclpy", _types.ModuleType("rclpy"))
_qos = sys.modules.setdefault("rclpy.qos", _types.ModuleType("rclpy.qos"))
for _enum in ("ReliabilityPolicy", "HistoryPolicy", "DurabilityPolicy"):
    setattr(_qos, _enum, type(_enum, (), {"BEST_EFFORT": "best_effort",
                                          "RELIABLE": "reliable",
                                          "KEEP_LAST": "keep_last",
                                          "VOLATILE": "volatile"}))
setattr(_qos, "QoSProfile", lambda **kw: dict(kw))
setattr(_rclpy, "qos", _qos)

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

def test_no_downstream_card_is_not_a_reason_to_refuse():
    """Superseded a check that demanded a chassis before the card would run.
    See test_it_starts_without_a_downstream_card for what it does instead."""
    result = _card().dispatch("navi", {"action": "start"})
    assert "没有拿到下游动作空间" not in (result.get("message") or "")


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


# ── input binding ────────────────────────────────────────────────────────────
#
# These are the tests that would have caught the real-machine failure: on r1_sz
# the canvas was wired correctly and navi still refused to start, because
# visual_depth was still loading its TensorRT engine and its topic had no
# publisher yet. The next log line was `visual_depth settled: running`.

class _FakeNode:
    """A node that knows nothing — which is the state during upstream loading."""

    def __init__(self, graph=None):
        self.graph = graph or {}
        self.subscriptions = []

    def get_topic_names_and_types(self):
        return list(self.graph.items())

    def create_subscription(self, msg_type, topic, cb, depth):
        self.subscriptions.append(topic)


def _bind(topics, graph=None):
    card = _card()
    return card._bind_inputs(_FakeNode(graph), topics)


def test_roles_come_from_topic_names_not_from_live_publishers():
    """The fix. An empty ROS graph must still bind correctly — perception
    names its outputs deterministically, and loading is a normal transient."""
    bound, problem = _bind(["/ubuntu/camera/main/objects",
                            "/ubuntu/camera/main/visual_depth"])
    assert problem == ""
    assert bound["objects"] == "/ubuntu/camera/main/objects"
    assert bound["depth_map"] == "/ubuntu/camera/main/visual_depth"


def test_the_summary_suffix_is_not_swallowed_by_the_depth_map_suffix():
    """`/visual_depth_summary` starts with `/visual_depth`; matched in the
    wrong order the summary binds as a depth map and every distance is wrong."""
    bound, problem = _bind(["/cam/objects", "/cam/visual_depth_summary"])
    assert problem == ""
    assert bound["depth_summary"] == "/cam/visual_depth_summary"
    assert bound["depth_map"] == ""


def test_odom_binds_by_name_too():
    bound, _ = _bind(["/cam/objects", "/cam/visual_depth", "/ubuntu/state/odom"])
    assert bound["odom"] == "/ubuntu/state/odom"


def test_a_summary_only_wiring_is_accepted():
    _, problem = _bind(["/cam/objects", "/cam/visual_depth_summary"])
    assert problem == ""


def test_missing_depth_is_still_refused():
    """The check has to keep working — this is not a licence to bind nothing."""
    _, problem = _bind(["/cam/objects"])
    assert "深度" in problem


def test_missing_detections_is_still_refused():
    _, problem = _bind(["/cam/visual_depth"])
    assert "vop" in problem


def test_an_unrecognised_name_falls_back_to_the_graph():
    bound, problem = _bind(
        ["/cam/objects", "/weird/topic"],
        graph={"/weird/topic": ["sensor_msgs/msg/CompressedImage"]})
    assert problem == ""
    assert bound["depth_map"] == "/weird/topic"


def test_an_unrecognised_string_topic_is_not_guessed():
    """Three different payloads ride on String. Mistaking a depth summary for
    odometry makes the robot act on entirely the wrong numbers, so an
    unrecognisable one is left unbound rather than assigned a role."""
    bound, problem = _bind(["/cam/objects", "/cam/visual_depth", "/weird/topic"],
                           graph={"/weird/topic": ["std_msgs/msg/String"]})
    # Required inputs are all present, so this does not block the start...
    assert problem == ""
    assert "/weird/topic" not in bound.values()
    # ...but an ignored wire must not be invisible: the operator drew it.
    assert any("/weird/topic" in h for h in bound["unknown"])


def test_an_ignored_wire_shows_up_in_the_degradations():
    card = _card()
    card._binding = {"objects": "/o", "depth_map": "/d", "odom": "/r1/state/odom",
                     "unknown": ["/weird/topic（String，但话题名不符合…）"]}
    assert any("没有被使用" in note for note in card._degradations())


def test_an_unrecognised_name_is_refused_when_a_required_input_is_missing():
    """Then it is the likely cause, and naming it is the whole point."""
    _, problem = _bind(["/cam/objects", "/weird/topic"],
                       graph={"/weird/topic": ["std_msgs/msg/String"]})
    assert "深度" in problem and "/weird/topic" in problem


def test_the_first_topic_of_a_role_wins():
    bound, _ = _bind(["/a/objects", "/b/objects", "/cam/visual_depth"])
    assert bound["objects"] == "/a/objects"


def test_subscriptions_are_actually_created():
    card = _card()
    node = _FakeNode()
    bound, problem = card._bind_inputs(
        node, ["/cam/objects", "/cam/visual_depth", "/r1/state/odom"])
    assert problem == ""
    assert set(node.subscriptions) == {"/cam/objects", "/cam/visual_depth",
                                       "/r1/state/odom"}


def test_subscriptions_use_best_effort_qos():
    """A reliable subscriber never matches a best-effort publisher, and ROS2
    reports that as nothing at all — no error, no warning, and `ros2 topic info`
    still shows both ends. Every producer this card reads (vop, visual_depth,
    loco_state) publishes best-effort on purpose.
    """
    from plugins.navi.plugin import _sensor_qos

    profile = _sensor_qos()
    assert profile["reliability"] == "best_effort"
    assert profile["depth"] == 2      # only the newest sample is ever read


def test_navigate_to_says_at_once_when_the_target_is_not_in_view():
    """Not a refusal — "turn around and find the chair" is legitimate. But the
    caller learns it now rather than from info().last sixteen seconds later,
    which is when a mistyped key used to surface."""
    card = _card()
    card._running = True
    card._recent.extend([{"objects": [{"name": "person", "confidence": 0.9,
                                       "position": [0, 0],
                                       "color": "dim muted green"}]}] * 10)
    out = card.dispatch("navi", {"action": "navigate_to", "target": "person#'green"})
    assert out["state"] == "running"          # still starts
    assert "不在当前可见列表" in out["warning"]
    assert out["visible_now"] == ["person#green"]


def test_a_target_that_is_in_view_gets_no_warning():
    card = _card()
    card._running = True
    card._recent.extend([{"objects": [{"name": "person", "confidence": 0.9,
                                       "position": [0, 0],
                                       "color": "dim muted green"}]}] * 10)
    out = card.dispatch("navi", {"action": "navigate_to", "target": "person#green"})
    assert "warning" not in out


def test_a_bare_name_matching_a_keyed_object_is_not_warned_about():
    card = _card()
    card._running = True
    card._recent.extend([{"objects": [{"name": "person", "confidence": 0.9,
                                       "position": [0, 0],
                                       "color": "dim muted green"}]}] * 10)
    assert "warning" not in card.dispatch(
        "navi", {"action": "navigate_to", "target": "person"})


def test_navigate_to_returns_an_action_id():
    """The ACP contract runs tool → agent-core, not the other way round:
    `mcp_client` parses `action_id` out of the *reply* to register the pending
    action. A reply without one leaves the completion referring to an id nobody
    is waiting on — which is indistinguishable, from the outside, from a card
    that never reports failure at all."""
    card = _card()
    card._running = True
    out = card.dispatch("navi", {"action": "navigate_to", "target": "chair"})
    assert out["action_id"].startswith("navi_")
    assert card._acp_action_id == out["action_id"]


def test_each_navigate_to_gets_a_fresh_action_id():
    card = _card()
    card._running = True
    first = card.dispatch("navi", {"action": "navigate_to", "target": "chair"})
    second = card.dispatch("navi", {"action": "navigate_to", "target": "tv"})
    assert first["action_id"] != second["action_id"]


def test_the_visible_list_stays_small_under_full_colour():
    """`publish_color: full` carries twelve numbers per object, and this reply
    goes into LLM context whole, every time it is called. On r1_sz it reached
    7356 characters — none of which the caller can use, since choosing a target
    needs only the key and a description."""
    import json

    card = _card()
    full = {"rgb_mean": [79.6, 74.4, 76.3], "rgb_var": [1551.5, 1417.7, 1811.9],
            "hsv_mean": [59.9, 41, 85.3], "hsv_var": [4327.9, 1200.6, 1571.8],
            "dominant_hue": "green", "dominant_saturation": "muted",
            "dominant_brightness": "dim", "color_name": "green"}
    card._recent.extend([{"objects": [
        {"name": f"thing{i}", "confidence": 0.9, "position": [0.1 * i, 0],
         "color": full} for i in range(8)]}] * 10)

    out = card.dispatch("navi", {"action": "list_visible_objects"})
    assert out["count"] == 8
    text = json.dumps(out, ensure_ascii=False)
    assert "rgb_var" not in text
    assert len(text) < 2000, f"reply is {len(text)} chars"


# ── a downstream is not required to start ────────────────────────────────────

def test_it_starts_without_a_downstream_card():
    """"Don't wire the chassis yet, I just want to see what it computes" is a
    legitimate — and the most common — way to bring this card up. Refusing
    demanded a robot that can move before its decisions could be observed."""
    card = _card()
    node = _FakeNode()
    card._executor = object()
    card._open_publisher = lambda: setattr(card, "_node", node)
    out = card.dispatch("navi", {"action": "start",
                                 "input_topics": ["/cam/objects",
                                                  "/cam/visual_depth"]})
    assert out["state"] == "running"
    assert any("不会驱动任何硬件" in note for note in out["degraded"])


def test_an_unbound_card_says_so_in_info():
    """A card that looks running but drives nothing must be distinguishable
    from one that is actually connected to a chassis."""
    card = _card()
    card._running = True
    card._descriptor = {}
    card._binding = {"objects": "/o", "depth_map": "/d", "odom": "/r1/state/odom"}
    assert any("不会驱动任何硬件" in n for n in card._degradations())


def test_a_connected_card_does_not_claim_to_be_unbound():
    card = _card()
    card._running = True
    card._descriptor = _descriptor()
    card._binding = {"objects": "/o", "depth_map": "/d", "odom": "/r1/state/odom"}
    assert card._degradations() == []


def test_negotiation_is_still_strict_when_a_downstream_is_present():
    """Loosening the no-downstream case must not loosen the real one: that is
    where hardware moves, and a mismatch has to be refused at start."""
    out = _card().dispatch("navi", {"action": "start",
                                    "control_interface": _descriptor(mode="joint_position")})
    assert out["state"] == "error" and "twist" in out["message"]
