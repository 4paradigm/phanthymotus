"""The navi card's contract with agent-core and with the driver below it.

The behaviour is tested in test_navi_policy.py; this covers the card itself —
what it refuses, what it declares, and what it tells an operator when it is
running at reduced precision. No ROS: everything here stops before the card
opens a publisher, which is exactly the set of paths where a mistake produces
"the canvas looks fine and the robot does nothing".
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ROS stubs live in conftest.py so every module in the suite shares one set —
# two files installing their own let whichever is collected last win.
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
    card._running = True
    card._binding = {"objects": "/cam/objects", "depth_map": "/cam/visual_depth",
                     "odom": ""}
    assert any("卡死" in note for note in card._degradations())


def test_a_card_with_only_the_summary_says_its_distances_are_coarse():
    card = _card()
    card._running = True
    card._binding = {"objects": "/cam/objects", "depth_summary": "/cam/sum",
                     "depth_map": "", "odom": "/r1/state/odom"}
    assert any("深度摘要" in note for note in card._degradations())


def test_a_fully_wired_card_reports_no_degradation():
    card = _card()
    card._running = True
    card._descriptor = _descriptor()          # a chassis is wired downstream
    card._binding = {"objects": "/cam/objects", "depth_map": "/cam/visual_depth",
                     "odom": "/r1/state/odom"}
    # Fully wired includes the camera having said what resolution its boxes are
    # in: without it vop's pixel boxes cannot be normalised and the target's
    # distance silently drops to a centre patch.
    card._objects_frame = (1280, 720)
    assert card._degradations() == []


def test_info_carries_the_degradations_and_the_target():
    card = _card()
    card._running = True
    card._descriptor = _descriptor()
    card._binding = {"objects": "/o", "depth_map": "", "odom": ""}
    card._objects_frame = (1280, 720)
    info = card.dispatch("navi", {"action": "info"})
    assert info["state"] == "running"
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
    card._running = True
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
    card._objects_frame = (1280, 720)
    assert card._degradations() == []


def test_negotiation_is_still_strict_when_a_downstream_is_present():
    """Loosening the no-downstream case must not loosen the real one: that is
    where hardware moves, and a mismatch has to be refused at start."""
    out = _card().dispatch("navi", {"action": "start",
                                    "control_interface": _descriptor(mode="joint_position")})
    assert out["state"] == "error" and "twist" in out["message"]


def test_info_returns_topic_out_so_the_dashboard_can_subscribe():
    """agent-core registers monitored topics from `info()['topic_out']`, not
    from the tool schema (`api/config.py`, after starting each card). Without
    it the card runs correctly and the messages are genuinely on the bus, while
    the canvas data-flow panel stays empty for ever — which reads as the card
    producing nothing."""
    out = _card().dispatch("navi", {"action": "info"})["topic_out"]
    assert out == [{"topic": navi_plugin.DEFAULT_TOPIC,
                    "format": "control/velocity"},
                   # The human-only overlay is a port like any other: a card
                   # that publishes it without declaring it is invisible in the
                   # panel while being perfectly healthy everywhere else.
                   {"topic": "/actucore/navi/view", "format": "image/jpeg"}]


def test_info_and_the_schema_agree_about_the_output_topic():
    """Two places name it; they must not drift."""
    card = _card()
    assert (card.dispatch("navi", {"action": "info"})["topic_out"][0]["topic"]
            == card.get_tools()[0]["topic_out"][0]["topic"])


def test_an_idle_card_blames_itself_rather_than_vop():
    """The card knows its own state; pointing at the upstream sends someone to
    investigate something that is working. Seen on r1_sz after a container
    restart: vop was publishing fine and the message said to go check it."""
    out = _card().dispatch("navi", {"action": "list_visible_objects"})
    assert "未在运行" in out["message"] and "vop" not in out["message"]


def test_a_running_card_with_no_frames_does_point_at_vop():
    card = _card()
    card._running = True
    card._binding = {"objects": "/cam/objects"}
    assert "/cam/objects" in card.dispatch(
        "navi", {"action": "list_visible_objects"})["message"]


def test_an_idle_card_reports_no_degradations():
    """"Running, but missing something" is what a degradation means. An idle
    card has nothing bound because it has not started, and reporting "only the
    depth summary is wired" there is inventing a fact."""
    assert _card()._degradations() == []


# ── the stream describes itself ──────────────────────────────────────────────

def _first_command(card):
    card._running = True
    card._state = navi_plugin.policy_mod.State(target="chair")
    card._objects = {"objects": [{"name": "chair", "confidence": 0.9,
                                  "position": [0.5, 0.0]}]}
    card._objects_ms = 10 ** 13
    card._depth_bands = {"left": 5.0, "center": 5.0, "right": 5.0}
    card._depth_ms = 10 ** 13
    card._config.max_obs_age_ms = 10 ** 13
    # The policy will not command anything from a track it has not confirmed,
    # so the card has to tick a few times before there is a first command at
    # all. Dropping the unconfirmed ticks keeps this about the descriptor echo.
    for _ in range(card._config.confirm_hits):
        message = card.next_command()
        if message is not None:
            return message
    return card.next_command()


def test_the_first_command_names_its_axes_even_with_no_downstream():
    """A twist producer knows what its six numbers are called: the order is
    fixed by the protocol. Leaving them unnamed pushes "call them joint1..6"
    onto the renderer, and on a chassis that labels the yaw rate `joint6`."""
    message = _first_command(_card())
    assert message["control_interface"]["joint_names"] == \
        ["vx", "vy", "vz", "wx", "wy", "wz"]


def test_a_downstream_descriptor_is_echoed_in_preference():
    """The driver's own descriptor carries real limits, which is what gives the
    dashboard's range bars a scale instead of inferring one from extremes."""
    card = _card()
    card._descriptor = _descriptor()
    assert _first_command(card)["control_interface"] is card._descriptor


def test_the_descriptor_is_not_echoed_on_every_command():
    """It is a constant with a hundred numbers in it; at 10 Hz echoing it every
    time multiplies the command stream for no new information."""
    card = _card()
    first = _first_command(card)
    assert "control_interface" in first
    assert first["seq"] == 1, "the echo is keyed on the sequence number"
    assert "control_interface" not in card.next_command()


def test_navigate_to_tells_the_model_it_searches_on_its_own():
    """Otherwise the model tries to help: it sees an empty
    `list_visible_objects`, concludes the target is absent, and either gives up
    or starts issuing its own turn commands — fighting a card that was about to
    turn anyway, over a chassis they share."""
    text = _tool()["inputSchema"]["x-action-params"]["navigate_to"]["description"]
    assert "不必现在就看得见" in text and "转身搜索" in text


def test_list_visible_objects_does_not_read_as_an_existence_check():
    """It reports what is in frame right now. A model reading an empty list as
    "there is no chair here" refuses a navigation that a quarter turn would
    have satisfied."""
    text = _tool()["inputSchema"]["x-action-params"]["list_visible_objects"]["description"]
    assert "不代表目标不存在" in text


def test_a_clear_goal_does_not_have_to_list_first():
    """"Go to the sofa" needs no disambiguation. Telling the model to list
    first costs a round trip on every navigation and invites it to treat an
    empty list as a reason not to go."""
    params = _tool()["inputSchema"]["x-action-params"]
    assert "不必先" in params["navigate_to"]["description"]
    assert "甄别" in params["list_visible_objects"]["description"]


# ── the tracker needs the whole twist, not just forward speed ────────────────

def test_odometry_is_read_on_all_three_axes_the_policy_uses():
    """`vx` alone was enough for the stuck detector, which is all there used to
    be. The tracker compensates its prediction for the robot's whole twist, and
    **yaw matters most**: on a legged chassis turning is what moves a target
    across the frame fastest, and a yaw rate read as zero puts the prediction
    the full rotation out within a few ticks."""
    card = _card()

    class _Message:
        data = json.dumps({"schema": "motus.odom/1", "frame": "body",
                           "stamp_ms": 1, "twist": [0.4, None, None, None, None, 1.0]})

    card._on_string("odom", _Message())
    assert card._odom == {"vx": 0.4, "vy": None, "wz": 1.0}


def test_an_unmeasured_axis_stays_none_rather_than_zero():
    """The whole reason motus.odom/1 forbids reporting 0.0 for an axis nobody
    measured. A robot that cannot answer "am I turning" must not have its own
    rotation assumed away — the tracker would then predict a stationary world
    while the chassis spins."""
    card = _card()

    class _Message:
        data = json.dumps({"schema": "motus.odom/1", "frame": "body",
                           "stamp_ms": 1, "twist": [None] * 6})

    card._on_string("odom", _Message())
    assert card._odom == {"vx": None, "vy": None, "wz": None}


def test_the_listing_bar_and_the_chasing_bar_agree():
    """These used to disagree by an order of magnitude — ten frames to appear in
    `list_visible_objects`, one frame to start driving a chassis — and the
    *listing* was the strict one."""
    card = _card()
    ratio = card._config.confirm_hits / card._config.confirm_window
    assert card._stability_bar(10) == round(10 * ratio)
    assert card._stability_bar(1) == 1, "never ask for more frames than exist"


def test_a_lost_target_tells_the_caller_to_look_before_retrying():
    """A detector's class for one object is not stable: the same fire
    extinguisher on r1_sz was reported 373 times as `fire extinguisher` in one
    recording and as `bottle` twenty minutes later. So "not found" far more
    often means the name does not match than that the thing is absent, and a
    caller told only "lost" retries the same wrong name or gives up."""
    text = _tool()["inputSchema"]["x-action-params"]["navigate_to"]["description"]
    assert "list_visible_objects" in text
    assert "类别并不稳定" in text


# ── the config dialog is three knobs, and its defaults are load-bearing ──────

def test_the_dialog_offers_only_what_an_operator_can_judge():
    """Twenty-two fields, most of them things like `release_frac` and
    `bearingless_std_factor`, gave an operator no way to judge an answer and
    buried the three that matter. The rest live in config.yaml with the
    paragraph of explanation they need, and stay reachable via the `config`
    action."""
    keys = set(_tool()["configSchema"]["properties"])
    assert keys == {"rate_hz", "stop_distance_m", "obstacle_stop_m"}


def test_every_schema_default_matches_the_file_default():
    """**agent-core sends every schema field on every config call, defaults
    included**, so a key here silently overrides the same key in config.yaml.
    On r1_sz the file said `vx_max: 0.6` and the card reported 0.4 — the schema
    default won, and the only trace was a degraded note that read like the file
    had never been edited.

    A field may live in the schema or in the file; if it lives in both, the two
    defaults have to agree or the file is decoration."""
    import os
    import yaml

    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "config.yaml")) as handle:
        navi_cfg = yaml.safe_load(handle)["plugins"]["navi"]

    for key, spec in _tool()["configSchema"]["properties"].items():
        if key in navi_cfg:
            assert navi_cfg[key] == spec["default"], (
                f"{key}: config.yaml 是 {navi_cfg[key]}，schema 默认是 "
                f"{spec['default']} —— 画布会用后者覆盖前者")


def test_a_chassis_that_swallows_commands_is_reported_as_degraded():
    """A policy whose commands are being dropped looks exactly like one that is
    working: same verdicts, same counters, same silence. On r1_sz a deploy reset
    the driver to its image defaults (`dry_run: true`) and the robot stood still
    while every layer reported success."""
    card = _card()
    card._running = True
    card._binding = {"objects": "o", "depth_map": "d", "odom": "s"}
    card._descriptor = {"mode": "twist", "dry_run": True}
    assert any("dry_run" in note for note in card._degradations())


# ── 相机几何沿连线传下来（motus.camera/1）──────────────────────────────────────

_DEPTH_TOPIC = "/ubuntu/camera/main/visual_depth"
_OBJECTS_TOPIC = "/ubuntu/camera/main/objects"


def _decl(topic, **over):
    out = {"schema": "motus.camera/1", "topic": topic,
           "id": "unitree/r1/camera_main", "width": 640, "height": 480,
           "half_fov_rad": 0.888, "source": "inherited"}
    out.update(over)
    return out


def _binding(**over):
    out = {"objects": _OBJECTS_TOPIC, "depth_map": _DEPTH_TOPIC,
           "depth_summary": "", "odom": ""}
    out.update(over)
    return out


def test_the_card_adopts_the_geometry_of_whichever_topic_carries_the_depth():
    """Joined on the topic this card bound, not on list position and not on the
    upstream card's name — inputs are dispatched by what they carry, on purpose,
    so a different depth source has to keep working."""
    card = _card(half_fov_rad=0.55)
    assert card._adopt_camera({_DEPTH_TOPIC: _decl(_DEPTH_TOPIC)}, _binding()) == ""
    assert card._config.half_fov_rad == 0.888
    assert any("0.888" in note for note in card._camera_notes)


def test_a_declaration_for_some_other_topic_is_not_used_for_the_depth():
    """The failure this keying prevents: one lens's geometry silently applied to
    another lens's picture."""
    card = _card(half_fov_rad=0.55)
    card._adopt_camera({"/somewhere/else": _decl("/somewhere/else")}, _binding())
    assert card._config.half_fov_rad == 0.55
    assert card._camera_notes, "用了兜底值就得说出来"


def test_the_note_reaches_degraded_where_an_operator_will_see_it():
    card = _card(half_fov_rad=0.55)
    card._running = True
    card._adopt_camera(None, _binding())
    card._binding = _binding()
    assert any("camera_info" in note for note in card._degradations())


def test_only_a_summary_is_enough_to_carry_the_geometry():
    """Depth-summary-only is an existing degraded tier. It is still a camera, so
    it still has a field of view."""
    card = _card(half_fov_rad=0.55)
    summary = "/ubuntu/camera/main/visual_depth/summary"
    card._adopt_camera({summary: _decl(summary)},
                       _binding(depth_map="", depth_summary=summary))
    assert card._config.half_fov_rad == 0.888


def test_two_different_cameras_refuse_to_start():
    """vop reports a normalised offset, the depth map is a grid of distances, and
    this card turns both into metres with **one** field of view — which is only
    correct if they are the same lens.

    Nothing has ever enforced it: inputs bind by what they carry, so camera A's
    vop plus camera B's depth has always been wirable and would produce
    confidently wrong distances with nothing in any log. The original navi plan
    promised this check and never implemented it; both sides declaring an `id`
    makes it a comparison.
    """
    card = _card()
    problem = card._adopt_camera({
        _DEPTH_TOPIC: _decl(_DEPTH_TOPIC, id="unitree/r1/camera_main"),
        _OBJECTS_TOPIC: _decl(_OBJECTS_TOPIC, id="unitree/r1/camera_left"),
    }, _binding())
    assert "不同的相机" in problem
    assert "camera_left" in problem and "camera_main" in problem


def test_one_side_declaring_nothing_is_not_treated_as_a_mismatch():
    """Half the cards in the repo declare nothing. Refusing on a missing
    declaration would fail every canvas that has not been updated yet."""
    card = _card()
    assert card._adopt_camera({_DEPTH_TOPIC: _decl(_DEPTH_TOPIC)}, _binding()) == ""
    assert card._adopt_camera({}, _binding()) == ""


def test_the_same_camera_on_both_inputs_is_accepted():
    card = _card()
    assert card._adopt_camera({
        _DEPTH_TOPIC: _decl(_DEPTH_TOPIC),
        _OBJECTS_TOPIC: _decl(_OBJECTS_TOPIC),
    }, _binding()) == ""


def test_a_camera_that_never_said_its_resolution_is_reported():
    """Not a cosmetic loss. Without the frame size vop's pixel boxes cannot be
    normalised, and `target_distance` silently drops from "percentile over the
    whole target" to "one patch at its centre" — the degradation this card
    documents as the cost of running vop without `publish_bbox`, reached by a
    different route and, until now, reported by nothing."""
    card = _card()
    card._running = True
    card._descriptor = _descriptor()
    card._binding = {"objects": "/o", "depth_map": "/d", "odom": "/r1/state/odom"}
    assert any("像素框无法归一化" in n for n in card._degradations())


def test_the_box_counters_localise_a_missing_overlay_box():
    """Four things have to line up for the overlay to draw a box, and "no box"
    used to be one symptom for all four. These counters say which."""
    card = _card()
    card._objects_frame = (1000, 500)
    card._state.target = "person"
    card._objects = card._normalise_boxes({"objects": [
        {"name": "person", "bbox": [100, 50, 300, 450], "position": [0.0, 0.0]},
        {"name": "chair", "bbox": [0, 0, 10, 10], "position": [0.5, 0.0]},
        {"name": "person", "position": [0.2, 0.0]},        # detector gave no box
    ]})
    stats = card._box_stats()
    assert stats == {"detections": 3, "with_bbox": 2, "with_bbox_norm": 2,
                     "matching_target": 2, "target_has_box": 1}


def test_a_pixel_box_becomes_a_normalised_one():
    card = _card()
    card._objects_frame = (1000, 500)
    out = card._normalise_boxes({"objects": [{"name": "person",
                                              "bbox": [100, 50, 300, 450]}]})
    assert out["objects"][0]["bbox_norm"] == [0.1, 0.1, 0.3, 0.9]


def test_without_a_declaration_no_box_is_invented():
    """Inferring the resolution from a box that happens to be large is exactly
    the plausible guess this card keeps being bitten by."""
    card = _card()
    out = card._normalise_boxes({"objects": [{"name": "person",
                                              "bbox": [100, 50, 300, 450]}]})
    assert "bbox_norm" not in out["objects"][0]
