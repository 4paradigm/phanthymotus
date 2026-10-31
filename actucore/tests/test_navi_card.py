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


def test_it_declares_every_input_format_it_accepts():
    formats = {entry["format"] for entry in _tool()["topic_in"]}
    assert formats == {"data/json", "image/depth-zlib",
                       "state/odom",
                       # The raw camera, optional and only for the overlay.
                       "image/jpeg"}


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
        self.subscription_details = []
        self.timers = []
        self.destroyed = False

    def get_topic_names_and_types(self):
        return list(self.graph.items())

    def create_subscription(self, msg_type, topic, cb, depth):
        self.subscriptions.append(topic)
        self.subscription_details.append((msg_type, topic, cb, depth))

    def create_timer(self, period, callback):
        self.timers.append(callback)
        return callback

    def destroy_timer(self, timer):
        if timer in self.timers:
            self.timers.remove(timer)

    def destroy_node(self):
        self.destroyed = True
        self.timers.clear()


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


def test_a_driver_camera_depth_topic_binds_as_metric_depth():
    """A registered hardware depth stream is already the depth map; requiring
    it to be renamed to perception's `/visual_depth` would force Bumi to run a
    less accurate monocular model despite having aligned RealSense depth."""
    bound, problem = _bind(["/ubuntu/camera/color/objects",
                            "/ubuntu/camera/depth"])
    assert problem == ""
    assert bound["depth_map"] == "/ubuntu/camera/depth"



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


def test_an_image_topic_with_no_known_suffix_is_the_camera_not_the_depth_map():
    """perception derives its depth topic from the camera's
    (`<camera>/visual_depth`), so a CompressedImage whose name does not end in
    that suffix cannot be the depth map — `_role_of` would already have caught
    it. Binding it as depth instead would compute distances from JPEG bytes;
    getting it wrong this way costs only the loud "depth is not connected"
    refusal at start.
    """
    bound, problem = _bind(
        ["/cam/objects", "/cam/visual_depth", "/ubuntu/camera/main"],
        graph={"/ubuntu/camera/main": ["sensor_msgs/msg/CompressedImage"]})
    assert problem == ""
    assert bound["image"] == "/ubuntu/camera/main"
    assert bound["depth_map"] == "/cam/visual_depth"


def test_the_camera_is_optional():
    """Every canvas drawn before this input existed must keep working."""
    bound, problem = _bind(["/cam/objects", "/cam/visual_depth"])
    assert problem == "" and bound["image"] == ""


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


# ── operator tuning fields and startup defaults ─────────────────────────────

def test_the_dialog_exposes_follow_gains_search_and_velocity_limits():
    keys = set(_tool()["configSchema"]["properties"])
    assert keys == {"rate_hz", "stop_distance_m", "stop_tolerance_m", "obstacle_stop_m",
                    "k_yaw", "k_fwd", "search_rate", "vx_max", "vy_max",
                    "wz_max", "use_lateral"}


def test_tuning_fields_are_available_to_the_config_action_with_the_same_schema():
    tool = _tool()
    params = tool["inputSchema"]["x-action-params"]["config"]["params"]
    assert set(params) == {"stop_distance_m", "stop_tolerance_m", "k_yaw", "k_fwd", "search_rate", "vx_max", "vy_max",
                           "wz_max", "use_lateral"}
    for key in params:
        dialog = tool["configSchema"]["properties"][key]
        assert dialog["scope"] == "instance"
        assert tool["inputSchema"]["properties"][key] == {
            name: value for name, value in dialog.items() if name != "scope"}


def test_tuning_defaults_follow_yaml_not_effective_chassis_limits():
    configured = {"k_yaw": 0.8, "k_fwd": 0.4, "search_rate": 0.3,
                  "vx_max": 0.2, "vy_max": 0.0, "wz_max": 0.6,
                  "use_lateral": False}
    card = _card(**configured)
    # adopt_limits changes this effective Config when a chassis is connected.
    card._config.wz_max = 0.1
    schema = card.get_tools()[0]["configSchema"]["properties"]
    assert {key: schema[key]["default"] for key in configured} == configured


def test_all_exposed_tuning_fields_update_and_can_be_read_back():
    card = _card()
    configured = {"k_yaw": 0.7, "k_fwd": 0.3, "search_rate": 0.2,
                  "vx_max": 0.1, "vy_max": 0, "wz_max": 0.5,
                  "use_lateral": False}
    result = card.dispatch("navi", {"action": "config", **configured})
    assert result["config"] == configured
    assert card._info()["config"] == {**configured, "stop_distance_m": 0.8,
                                      "stop_tolerance_m": 0.0, "arrival_distance_m": 0.8}
    schema = card.get_tools()[0]["configSchema"]["properties"]
    assert {key: schema[key]["default"] for key in configured} == configured
    from plugins.navi import policy
    restored = policy.Config(**{key: value for key, value in card._cfg.items()
                                if key in policy.Config.__dataclass_fields__})
    assert {key: getattr(restored, key) for key in configured} == configured


def _running_tuning_card(**cfg):
    card = _card(**cfg)
    card._open_publisher = lambda: setattr(card, "_node", _FakeNode({
        "/camera/depth": ["sensor_msgs/msg/CompressedImage"],
    }))
    result = card._start({
        "control_interface": _descriptor(limits={
            "lower": [-0.3, 0, 0, 0, 0, -1.0],
            "upper": [0.3, 0, 0, 0, 0, 1.0],
            "min_magnitude": [0, 0, 0, 0, 0, 0],
        }, footprint={"half_width": 0.22, "source": "measured"}),
        "input_topics": ["/cam/objects", "/camera/depth"],
    })
    assert result["state"] == "running", result
    return card


def test_live_tuning_reapplies_chassis_limits_without_losing_requested_values():
    card = _running_tuning_card()
    configured = {"vx_max": 99, "vy_max": 99, "wz_max": 99,
                  "search_rate": 99, "use_lateral": True}
    result = card.dispatch("navi", {"action": "config", **configured})
    assert result["config"] == configured
    assert result["effective_config"] == {
        "vx_max": 0.3, "vy_max": 0.0, "wz_max": 1.0,
        "search_rate": 1.0, "use_lateral": True,
    }
    effective = card._info()["config"]
    assert effective["vx_max"] == 0.3 and effective["vy_max"] == 0.0
    assert effective["wz_max"] == effective["search_rate"] == 1.0
    schema = card.get_tools()[0]["configSchema"]["properties"]
    assert {key: schema[key]["default"] for key in configured} == configured
    # A later gain edit must not forget why the other requested limits differ
    # from their effective values, nor quietly replace the requested settings.
    card._config_action({"k_yaw": 0.8})
    assert card._config.vx_max == 0.3 and card._cfg["vx_max"] == 99
    assert any("vx_max" in note for note in card._limit_notes)
    assert any("search_rate" in note for note in card._limit_notes)
    card._stop()


def test_live_tuning_preserves_camera_geometry_and_current_tracking_session():
    card = _running_tuning_card(clearance_margin_m=0.23)
    card._config.half_fov_rad = 0.55  # camera geometry adopted during start
    card._state.target = "person#azure"
    state, tracker = card._state, card._state.tracker
    result = card._config_action({"k_fwd": 0.4, "wz_max": 0.5})
    assert result["status"] == "configured"
    assert card._config.half_fov_rad == 0.55
    assert card._config.half_width_m == 0.22
    assert card._config.clearance_margin_m == 0.23
    assert card._state is state and card._state.tracker is tracker
    assert card._state.target == "person#azure"
    card._stop()


def test_live_search_can_be_disabled_and_deadbands_are_still_adopted():
    card = _running_tuning_card()
    card._descriptor["limits"]["min_magnitude"] = [0.1, 0, 0, 0, 0, 0.2]
    result = card._config_action({"vx_max": 0.01, "wz_max": 0.1,
                                  "search_rate": 0.1})
    assert result["effective_config"]["vx_max"] == pytest.approx(0.15)
    assert result["effective_config"]["wz_max"] == pytest.approx(0.3)
    assert result["effective_config"]["search_rate"] == 0
    assert card._config.floor_vx == 0.1 and card._config.floor_wz == 0.2
    assert card._config_action({"search_rate": 0})["effective_config"] == {"search_rate": 0}
    assert card._config.search_rate == card._cfg["search_rate"] == 0
    card._stop()


@pytest.mark.parametrize("key", ["stop_distance_m", "stop_tolerance_m",
                                "k_yaw", "k_fwd", "search_rate", "vx_max",
                                "vy_max", "wz_max"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True, "bad"])
def test_invalid_tuning_rejects_the_entire_request_atomically(key, value):
    card = _running_tuning_card()
    effective, requested = vars(card._config).copy(), card._cfg.copy()
    notes = list(card._limit_notes)
    changed = {"use_lateral": False, key: value}
    result = card._config_action(changed)
    assert result["state"] == "error" and key in result["message"]
    assert vars(card._config) == effective and card._cfg == requested
    assert card._limit_notes == notes
    card._stop()


@pytest.mark.parametrize("value", [0, 1, "false", "true", [], {}])
def test_use_lateral_requires_a_boolean_without_partial_gain_update(value):
    card = _card(k_yaw=0.7)
    result = card._config_action({"k_yaw": 0.9, "use_lateral": value})
    assert result["state"] == "error" and "use_lateral" in result["message"]
    assert card._config.k_yaw == card._cfg["k_yaw"] == 0.7


def test_stopped_tuning_accepts_numeric_strings_without_using_an_old_chassis():
    card = _running_tuning_card()
    card._stop()
    result = card._config_action({"vx_max": "0.8", "search_rate": "1.2"})
    assert result["config"] == result["effective_config"] == {
        "vx_max": 0.8, "search_rate": 1.2,
    }


def test_stop_tolerance_is_exposed_with_unchanged_zero_default():
    tool = _tool()
    spec = tool["configSchema"]["properties"]["stop_tolerance_m"]
    assert spec["type"] == "number" and spec["minimum"] == 0
    assert spec["default"] == 0.0 and spec["scope"] == "instance"
    assert tool["inputSchema"]["x-action-params"]["navigate_to"]["params"] == [
        "target", "stop_distance_m", "stop_tolerance_m"]
    card = _card(stop_distance_m=1.2, stop_tolerance_m=0.1)
    schema = card.get_tools()[0]["configSchema"]["properties"]
    assert schema["stop_distance_m"]["default"] == 1.2
    assert schema["stop_tolerance_m"]["default"] == 0.1
    assert card._info()["config"]["arrival_distance_m"] == pytest.approx(1.3)


def test_stop_tolerance_config_updates_requested_and_effective_readback():
    card = _running_tuning_card()
    requested = {"stop_distance_m": 1.2, "stop_tolerance_m": 0.1}
    result = card._config_action(requested)
    assert result["status"] == "configured"
    assert result["config"] == result["effective_config"] == requested
    assert {key: card._info()["config"][key] for key in requested} == requested
    assert card._info()["config"]["arrival_distance_m"] == pytest.approx(1.3)
    assert {key: card._cfg[key] for key in requested} == requested
    card._stop()


def test_navigation_distance_override_is_atomic_and_omission_keeps_current_tolerance():
    card = _running_tuning_card(stop_distance_m=0.8, stop_tolerance_m=0.05)
    result = card._navigate_to({"target": "person", "stop_distance_m": 1.2,
                                "stop_tolerance_m": 0.1})
    assert result["state"] == "running"
    assert result["stop_distance_m"] == 1.2 and result["stop_tolerance_m"] == 0.1
    assert result["arrival_distance_m"] == pytest.approx(1.3)
    # Per-navigation overrides do not rewrite the persistent requested config.
    assert card._cfg["stop_distance_m"] == 0.8 and card._cfg["stop_tolerance_m"] == 0.05
    card._config_action({"k_yaw": 0.8})
    second = card._navigate_to({"target": "chair"})
    assert second["stop_distance_m"] == 1.2 and second["stop_tolerance_m"] == 0.1
    assert second["arrival_distance_m"] == pytest.approx(1.3)
    assert card._info()["config"]["arrival_distance_m"] == pytest.approx(1.3)
    card._stop()


@pytest.mark.parametrize("key", ["stop_distance_m", "stop_tolerance_m"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True, "bad"])
def test_invalid_navigation_distance_preserves_current_action_and_pause(key, value):
    card = _running_tuning_card()
    card._navigate_to({"target": "person", "stop_distance_m": 1.2,
                       "stop_tolerance_m": 0.1})
    card._paused = True
    state, config, action_id = card._state, card._config, card._acp_action_id
    requested = card._cfg.copy()
    args = {"target": "chair", "stop_distance_m": 2.0, "stop_tolerance_m": 0.2,
            key: value}
    result = card._navigate_to(args)
    assert result["state"] == "error" and key in result["message"]
    assert card._state is state and state.target == "person"
    assert card._config is config and card._cfg == requested
    assert card._acp_action_id == action_id and card._paused is True
    card._stop()


@pytest.mark.parametrize("action", ["config", "navigate_to"])
def test_distance_threshold_overflow_does_not_apply_any_configuration(action):
    card = _running_tuning_card()
    state, config, action_id = card._state, card._config, card._acp_action_id
    requested = card._cfg.copy()
    result = card.dispatch("navi", {"action": action, "target": "chair",
                                    "stop_distance_m": 1e308, "stop_tolerance_m": 1e308})
    assert result["state"] == "error"
    assert card._state is state and card._config is config and card._cfg == requested
    assert card._acp_action_id == action_id
    card._stop()


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


# ── 渲染不许占用喂底盘的那条线程 ─────────────────────────────────────────────

def test_the_tick_hands_the_frame_off_instead_of_rendering_it():
    """Rendering a frame costs **27 ms measured on Orin 5** — a quarter of a
    10 Hz command period. Doing it inline stalls the stream that feeds the
    chassis every time the overlay ticks.

    The module said a picture for a human must not be able to stop a robot;
    that was written about exceptions, and only exceptions were guarded.
    Latency is the same promise.
    """
    import queue as _queue

    card = _card()
    card._running = True
    card._view_queue = _queue.Queue(maxsize=1)
    card._view_publisher = object()
    card._state.target = "person"

    card._publish_view()
    assert card._view_queue.qsize() == 1, "tick 没有把帧交出去"
    snapshot = card._view_queue.get_nowait()
    # Everything the worker needs, and nothing that has to be recomputed on the
    # tick to get it.
    assert set(snapshot) >= {"depth_m", "objects", "jpeg", "decision", "track",
                             "bearing", "target", "clearance", "coverage"}


def test_a_backed_up_queue_keeps_the_newest_frame():
    """One slot, newest wins. An overlay frame from two seconds ago is worse
    than a dropped one — the same rule the camera frames upstream follow."""
    import queue as _queue

    card = _card()
    card._running = True
    card._view_queue = _queue.Queue(maxsize=1)
    card._view_publisher = object()
    card._view_queue.put_nowait({"stale": True})

    card._view_next_at = 0.0
    card._publish_view()
    assert "stale" not in card._view_queue.get_nowait()


def test_the_tick_does_nothing_when_the_overlay_is_off():
    card = _card(view_hz=0)
    card._running = True
    assert card._publish_view() is None


def test_the_bearing_is_snapshotted_as_a_number_not_as_the_track():
    """The Track is mutated in place by the tick; handing the object to another
    thread would let the renderer read it mid-update. A float cannot tear."""
    import queue as _queue

    card = _card()
    card._running = True
    card._view_queue = _queue.Queue(maxsize=1)
    card._view_publisher = object()
    card._publish_view()
    snapshot = card._view_queue.get_nowait()
    assert snapshot["bearing"] is None or isinstance(snapshot["bearing"], float)


def test_depth_loading_defers_one_type_until_publisher_is_known():
    card = _card(); node = _FakeNode()
    card._running = True
    card._node = node
    bound, problem = card._bind_inputs(node, ['/cam/objects', '/q5/camera/depth'])
    assert problem == '' and bound['depth_map'] == '/q5/camera/depth'
    assert not any(t[1] == '/q5/camera/depth' for t in node.subscription_details)
    # A CompressedImage publisher appears and the deferred bind completes.
    # It used to be `sensor_msgs/msg/Image` here: `image/depth-z16` is retired,
    # because every driver that published it already produced exactly 640x480
    # little-endian 16UC1 and now compresses in the driver
    # (phanthymotus-driver#419). A depth topic still carrying a raw Image is a
    # stale driver image, and is refused by type rather than decoded.
    node.graph['/q5/camera/depth'] = ['sensor_msgs/msg/CompressedImage']
    node.timers[0]()
    assert [t[0].__name__ for t in node.subscription_details
            if t[1] == '/q5/camera/depth'] == ['CompressedImage']
    assert node.timers == []


def test_bumi_zlib_and_q5_raw_declarations_each_bind_one_type_without_graph():
    for fmt, name in [('image/depth-zlib', 'CompressedImage')]:
        card = _card(); node = _FakeNode()
        _, problem = card._bind_inputs(node, ['/cam/objects', '/camera/depth'], {'/camera/depth': {'format': fmt}})
        assert problem == ''
        assert [t[0].__name__ for t in node.subscription_details if t[1] == '/camera/depth'] == [name]


def test_jpeg_preview_never_becomes_metric_depth_and_type_conflicts_refuse():
    card = _card(); node = _FakeNode({'/camera/depth_preview': ['sensor_msgs/msg/CompressedImage']})
    _, problem = card._bind_inputs(node, ['/cam/objects', '/camera/depth_preview'])
    assert problem
    for types, declaration in [(['sensor_msgs/msg/Image','sensor_msgs/msg/CompressedImage'],None),
                               (['sensor_msgs/msg/Image'], {'/camera/depth':{'format':'image/depth-zlib'}})]:
        node = _FakeNode({'/camera/depth':types})
        _, problem = card._bind_inputs(node, ['/cam/objects','/camera/depth'],declaration)
        assert problem
        assert not any(t[1] == '/camera/depth' for t in node.subscription_details)


def test_bumi_compressed_depth_creates_only_the_compressed_subscription():
    card = _card()
    node = _FakeNode({'/ubuntu/camera/depth': ['sensor_msgs/msg/CompressedImage']})
    _, problem = card._bind_inputs(node, ['/cam/objects', '/ubuntu/camera/depth'])
    assert problem == ''
    details = [entry for entry in node.subscription_details
               if entry[1] == '/ubuntu/camera/depth']
    assert [entry[0].__name__ for entry in details] == ['CompressedImage']
    assert details[0][2] == card._on_depth_map


def test_explicit_input_binding_selects_depth_type_while_ros_graph_is_empty():
    card = _card()
    node = _FakeNode()
    _, problem = card._bind_inputs(
        node, ['/cam/objects', '/camera/depth'], None,
        {'bindings': [{'topic': '/camera/depth', 'format': 'image/depth-zlib',
                       'port_index': 2}]},
    )
    assert problem == ''
    assert [entry[0].__name__ for entry in node.subscription_details
            if entry[1] == '/camera/depth'] == ['CompressedImage']
    assert node.timers == []


def test_type_conflict_is_reported_before_any_depth_subscription_is_created():
    card = _card()
    node = _FakeNode({'/camera/depth': ['sensor_msgs/msg/Image']})
    _, problem = card._bind_inputs(
        node, ['/cam/objects', '/camera/depth'], None,
        {'/camera/depth': {'format': 'image/depth-zlib', 'port_index': 2}},
    )
    # Reported as a type conflict rather than a declaration mismatch: with z16
    # retired, `sensor_msgs/msg/Image` is not a supported depth type at all, so
    # an Image-only topic fails the observed-type check before its declaration
    # is ever compared. Either way nothing is subscribed, which is the part
    # that matters.
    assert '类型冲突' in problem
    assert not [entry for entry in node.subscription_details
                if entry[1] == '/camera/depth']


def test_unknown_depth_type_has_a_bounded_diagnostic_and_cancels_its_retry(monkeypatch):
    card = _card()
    node = _FakeNode()
    card._running = True
    card._node = node
    _, problem = card._bind_inputs(node, ['/cam/objects', '/camera/depth'])
    assert problem == '' and len(node.timers) == 1
    monkeypatch.setattr(navi_plugin.time, 'monotonic',
                        lambda: card._depth_bind_deadline + 0.01)
    node.timers[0]()
    assert len(node.timers) == 1
    assert "深度话题" in card._last_error

    # Publisher appears after the initial deadline: retry must recover.
    node.graph["/camera/depth"] = ["sensor_msgs/msg/CompressedImage"]
    node.timers[0]()
    assert node.timers == []
    assert any(
        msg_type.__name__ == "CompressedImage" and topic == "/camera/depth"
        for msg_type, topic, _, _ in node.subscription_details
    )
    assert card._last_error == ""


def test_start_stop_start_destroys_old_node_and_does_not_duplicate_depth_subscriptions():
    class _Executor:
        def remove_node(self, node):
            self.removed.append(node)

        def __init__(self):
            self.removed = []

    card = _card()
    executor = _Executor()
    card._executor = executor
    first, second = _FakeNode({'/camera/depth': ['sensor_msgs/msg/CompressedImage']}), \
                    _FakeNode({'/camera/depth': ['sensor_msgs/msg/CompressedImage']})
    nodes = iter((first, second))
    card._open_publisher = lambda: setattr(card, '_node', next(nodes))
    args = {'action': 'start', 'input_topics': ['/cam/objects', '/camera/depth']}
    assert card.dispatch('navi', args)['state'] == 'running'
    assert card.dispatch('navi', {'action': 'stop'})['state'] == 'stopped'
    assert first.destroyed and executor.removed == [first]
    assert card.dispatch('navi', args)['state'] == 'running'
    assert [item[0].__name__ for item in second.subscription_details
            if item[1] == '/camera/depth'] == ['CompressedImage']


@pytest.mark.parametrize("first_yaw_limit,expected_first", [
    (0.83, 0.83),  # Bumi -> R1
    (0.0, 0.0),    # No yaw -> R1
])
def test_search_rate_recovers_after_switching_chassis(
        monkeypatch, first_yaw_limit, expected_first):
    card = _card(search_rate=1.0)

    class Executor:
        def remove_node(self, node):
            pass

    card._executor = Executor()
    graph = {
        "/camera/depth": ["sensor_msgs/msg/CompressedImage"],
    }
    nodes = iter((_FakeNode(graph.copy()), _FakeNode(graph.copy())))
    card._open_publisher = lambda: setattr(card, "_node", next(nodes))

    # This test focuses on per-start limit negotiation.
    monkeypatch.setattr(navi_plugin.negotiate, "check",
                        lambda *args, **kwargs: [])

    def descriptor(yaw):
        return {
            "mode": card._capabilities()["control_mode"],
            "limits": {
                "lower": [-1.0, -1.0, 0, 0, 0, -yaw],
                "upper": [1.0, 1.0, 0, 0, 0, yaw],
                "min_magnitude": [0, 0, 0, 0, 0, 0.1],
            },
        }

    first = card._start({
        "control_interface": descriptor(first_yaw_limit),
        "input_topics": ["/cam/objects", "/camera/depth"],
    })
    assert first["state"] == "running", first
    assert card._config.search_rate == pytest.approx(expected_first)

    card._stop()

    second = card._start({
        "control_interface": descriptor(2.0),
        "input_topics": ["/cam/objects", "/camera/depth"],
    })
    assert second["state"] == "running", second
    assert card._config.search_rate == pytest.approx(1.0)
    card._stop()


def test_user_search_rate_setting_survives_next_start():
    card = _card(search_rate=1.0)
    result = card._config_action({"search_rate": 0.7})

    assert result["config"]["search_rate"] == pytest.approx(0.7)
    assert card._cfg["search_rate"] == pytest.approx(0.7)

    # Same reconstruction that happens before each start.
    restored = navi_plugin.policy_mod.Config(**{
        k: v for k, v in card._cfg.items()
        if k in navi_plugin.policy_mod.Config.__dataclass_fields__
    })
    assert restored.search_rate == pytest.approx(0.7)


def test_tianyi_head_depth_topic_binds_as_metric_depth():
    """Tianyi publishes metric depth as camera/head/depth."""
    bound, problem = _bind([
        "/nvidia_desktop/camera/head/objects",
        "/nvidia_desktop/camera/head/depth",
    ])

    assert problem == ""
    assert bound["depth_map"] == "/nvidia_desktop/camera/head/depth"


