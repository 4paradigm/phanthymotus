"""Exercise the visual harness's EEF transport and measured action outcomes without ROS."""
import json
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.autonomous_task import eef


def descriptor(sides=("left", "right")):
    # Use different grippers, a non-TRON2 frame, and arbitrary group order.
    return {"control_interface": "motus.control/1", "mode": "eef_pose", "frame": "world",
            "dof": 8 * len(sides), "tip_frames": {side: f"{side}_tcp" for side in sides},
            "state": {"schema": "motus.eef/1", "field": "eef"},
            "limits": {"lower": ([-1.] * 7 + [0.]) * len(sides),
                       "upper": [v for side in sides for v in [1.] * 7 + [.085 if side == "left" else .065]]},
            "rate": {"expected_hz": 10, "watchdog_ms": 200, "max_obs_age_ms": 750},
            "groups": [{"name": f"{kind}_{side[0]}", "offset": 8*i+offset, "count": size, "unit": unit}
                       for i, side in enumerate(sides) for kind, offset, size, unit in
                       (("eef", 0, 7, "m+quat_xyzw"), ("gripper", 7, 1, "m"))]}


def robot(sides=("left", "right"), **fields):
    node = Mock()
    pub = node.create_publisher.return_value
    pub.get_subscription_count.return_value = 1
    spec = eef.EefInterface({**descriptor(sides), **fields})
    endpoint = eef.EefEndpoint(node, "/state", "/control", spec)
    return endpoint, node, pub


def full():
    return [.1, .23, .34, 0., 0., 0., 1., .008, -.11, -.22, .33, 0., 0., 0., 1., 0.]


def feed(endpoint, stamp, measured, **overrides):
    packet = {"schema": "motus.eef/1", "frame": endpoint.interface.frame,
              "stamp_ms": stamp, "eef": measured, **overrides}
    endpoint._receive(SimpleNamespace(data=json.dumps(packet)))


def messages(pub):
    return [json.loads(call.args[0].data) for call in pub.publish.call_args_list]


def drive(endpoint, name, args, start, samples, drift=None):
    # Deliver a real state callback on each simulated 5 Hz capture. Only the waiting
    # is replaced. Every idle, active and hold message uses the actual encoder.
    clock = endpoint._stream.sample["stamp_ms"] if endpoint._stream.sample else eef._now_ms()
    endpoint._stream._now_ms = lambda: clock + 3
    samples = iter([start, start if drift is None else drift, *samples])

    def next_sample(after, cancel=None, window_ms=0):
        nonlocal clock
        if cancel and cancel.is_set():
            raise InterruptedError("cancelled")
        try:
            values = next(samples)
        except StopIteration:
            clock += 2000
            raise TimeoutError("feedback lost") from None
        clock = max(clock + 200, after + 1)
        feed(endpoint, clock, values)
        return clock, values, [values]

    endpoint._latest = next_sample
    state = endpoint.observe([name.split("_")[0]])
    result = endpoint.execute(name, args, threading.Event())
    return state, result


def test_group_layout_validated_and_frame_from_driver():
    endpoint, node, _ = robot(("right", "left"))
    spec = endpoint.interface
    assert spec.offsets["left", "eef"] == slice(8, 15)
    assert node.create_subscription.call_args.args[-1].reliability == "best_effort"
    assert node.create_publisher.call_args.args[-1].durability == "volatile"
    assert eef.EefInterface(descriptor(("right",))).dof == 8
    for change in ({"offset": 8}, {"unit": "mm"}, {"count": 6}, {"offset": -1}):
        desc = descriptor()
        desc["groups"][0].update(change)
        with pytest.raises(ValueError):
            eef.EefInterface(desc)
    for change in ({"groups": []}, {"dof": 15}, {"control_interface": "test.eef/1"}, {"frame": ""}):
        with pytest.raises(ValueError):
            eef.EefInterface(dict(descriptor(), **change))


def test_state_drops_invalid_and_expired_measurements_echoes_equal_capture_only_once_in_history():
    endpoint, _, pub = robot()
    stamp = eef._now_ms() - 10
    base = full()
    feed(endpoint, stamp - 1000, base)
    feed(endpoint, stamp, base[:10])
    feed(endpoint, stamp, [math.nan] + base[1:])
    feed(endpoint, stamp, base[:6] + [2.] + base[7:])
    feed(endpoint, stamp, base, schema="motus.control/1", values=base)
    assert not endpoint._samples and not pub.publish.called
    feed(endpoint, stamp, base)
    changed = [base[0] + .001, *base[1:]]
    feed(endpoint, stamp, changed)
    feed(endpoint, stamp - 1, base)
    assert len(endpoint._samples) == 1 and endpoint._latest(0)[1] == changed
    first, second = messages(pub)
    assert first["values"] == base and second["values"] == changed
    assert second["seq"] == first["seq"] + 1 and second["obs_stamp_ms"] == first["obs_stamp_ms"] == stamp
    assert second["source"] == first["source"] and first["source"].startswith("mcp__actucore__autonomous_task/")
    endpoint.STATE_TIMEOUT = .01
    with pytest.raises(TimeoutError, match="advancing"):
        endpoint._latest(stamp)
    with pytest.raises(ValueError):
        endpoint.observe(["other"])
    endpoint.close()
    feed(endpoint, stamp + 1, base)
    assert len(messages(pub)) == 2
    with pytest.raises(InterruptedError):
        endpoint.observe(["left"])


def test_observed_tcp_axes_and_arm_specific_openings():
    endpoint, _, _ = robot()
    pose = full()
    sine = math.sqrt(.5)
    pose[3:7] = [0., 0., sine, sine]
    endpoint._latest = Mock(return_value=(eef._now_ms(), pose, [pose]))
    obs = endpoint.observe(["left", "right"])
    assert obs["opening_range_mm_by_arm"] == {"left": [0., 85.], "right": [0., 65.]}
    measured = obs["arms"]["left"]["eef"]
    assert measured["frame"] == "world" and measured["eef_frame"] == "left_tcp"
    axes = measured["axes_in_reference"]
    assert axes["x"] == pytest.approx([0., 1., 0.])
    assert axes["y"] == pytest.approx([-1., 0., 0.])
    assert axes["z"] == pytest.approx([0., 0., 1.])


def test_relative_rotation_uses_base_axes_and_latest_both_arm_measurements():
    endpoint, _, pub = robot()
    start = full()
    a = math.sqrt(.5)
    start[3:7] = [a, 0., 0., a]
    current = start.copy()
    current[0] += .0008
    current[8] += .015
    theta = .3
    s, c = math.sin(theta/2), math.cos(theta/2)
    goal = current.copy()
    goal[1] += .05
    goal[3:7] = [c*a, s*a, s*a, c*a]
    _, result = drive(endpoint, "left_move_by", {"translation_m": [0., .05, 0.],
                                                 "rotation_vector_rad": [0., 0., theta]}, start, [goal]*7, current)
    cmd = result["command"]
    assert cmd["values"] == pytest.approx(goal)
    assert cmd["schema"] == "motus.control/1" and cmd["frame"] == "world" and cmd["dof"] == 16
    assert cmd["stamp_ms"] - cmd["obs_stamp_ms"] == 3 and 0 < cmd["ttl_ms"] <= 200
    assert result["target_reached"] and result["displacement_observed"]
    assert result["measured_translation_m"] == pytest.approx([0., .05, 0.])
    assert result["actual"]["eef_frame"] == "left_tcp"
    # The model emitted one tool call. The feedback clock maintains the selected target.
    sent = messages(pub)
    assert len(sent) > 3 and cmd == sent[2]
    assert [m["seq"] for m in sent] == list(range(sent[0]["seq"], sent[-1]["seq"] + 1))
    assert all(m["values"] == pytest.approx(goal) for m in sent[2:])


@pytest.mark.parametrize("field,position_error,angle_error,reached", [
    ({}, .0009, .009, True), ({}, .0011, .009, False), ({}, .0009, .015, False),
    ({"pose_tolerance": {"position_m": .002, "orientation_rad": .03}}, .0015, .025, True),
    ({"pose_tolerance": {"position_m": .002, "orientation_rad": .03}}, .0021, .025, False),
    ({"pose_tolerance": {"position_m": .002, "orientation_rad": .03}}, .0015, .031, False),
])
def test_shortfall_returns_measured_motion_for_next_model_turn(field, position_error, angle_error, reached):
    endpoint, _, _ = robot(**field)
    start = full()
    actual = start.copy()
    actual[0] += .02 - position_error
    actual[3:7] = [0., math.sin(angle_error / 2), 0., math.cos(angle_error / 2)]
    _, result = drive(endpoint, "left_move_by", {"translation_m": [.02, 0, 0], "rotation_vector_rad": [0, 0, 0]},
                      start, [actual] * 30)
    assert result["status"] == "completed" and result["target_reached"] is reached and result["displacement_observed"]
    assert result["position_error_m"] == pytest.approx(position_error)
    assert result["angle_error_rad"] == pytest.approx(angle_error)


@pytest.mark.parametrize("tolerance", [None, {"position_m": .001}, {"position_m": True, "orientation_rad": .01},
                         {"position_m": .001, "orientation_rad": float("nan")}, {"position_m": -.01, "orientation_rad": .01}])
def test_malformed_pose_tolerance_fails_at_wiring(tolerance):
    with pytest.raises(ValueError, match="pose_tolerance"):
        robot(pose_tolerance=tolerance)


def test_contact_then_pose_echoes_measured_opening_like_manual_eef():
    endpoint, _, _ = robot()
    start = full()
    start[7] = .08
    contact = start.copy()
    contact[7] = .04
    _, closed = drive(endpoint, "left_set_gripper", {"opening_mm": 0.}, start, [start]*12 + [contact]*30)
    assert closed["status"] == "completed" and not closed["target_reached"]
    assert closed["actual"]["opening_mm"] == 40. and closed["requested_opening_mm"] == 0.
    goal = contact.copy()
    goal[0] += .02
    _, moved = drive(endpoint, "left_move_by", {"translation_m": [.02, 0, 0], "rotation_vector_rad": [0, 0, 0]},
                     contact, [goal]*7)
    assert moved["command"]["values"] == pytest.approx(goal)
    assert moved["target_reached"]


def test_gripper_stays_still_then_cancellation_holds_latest_state():
    endpoint, _, pub = robot()
    start = full()
    _, result = drive(endpoint, "right_set_gripper", {"opening_mm": 3.}, start, [start]*30)
    assert result["status"] == "no_motion_observed" and not result["target_reached"]
    assert result["requested_opening_mm"] == 3 and result["actual"]["opening_mm"] == 0
    assert result["feedback_frames"] >= 20
    endpoint.cancel()
    assert messages(pub)[-1]["values"] == start


def test_feedback_loss_latches_next_measurement_and_never_resumes_unknown_target():
    endpoint, _, pub = robot()
    start = full()
    obs, result = drive(endpoint, "right_set_gripper", {"opening_mm": 4.}, start, [])
    assert set(obs["arms"]) == {"right"} and result["status"] == "outcome_unknown" and result["motion_sent"]
    assert result["command"] == messages(pub)[-1]
    now = endpoint._stream._now_ms()
    resumed = start.copy()
    resumed[15] = .002
    feed(endpoint, now, resumed)
    assert messages(pub)[-1]["values"] == resumed
    drift = resumed.copy()
    drift[8] += .001
    drift[15] += .001
    feed(endpoint, now + 1, drift)
    assert messages(pub)[-1]["values"] == drift[:15] + [.002]
    # A second interrupt (or the worker retiring the same task) cannot relatch the drift.
    endpoint.cancel()
    feed(endpoint, now + 2, drift)
    assert messages(pub)[-1]["values"] == drift[:15] + [.002]


def test_other_arm_drift_stops_the_task_stream_across_cancel_and_new_submissions():
    endpoint, _, pub = robot()
    initial = full()
    goal = initial.copy()
    goal[7] = .015
    _, result = drive(endpoint, "left_set_gripper", {"opening_mm": 15.}, initial, [goal]*8)
    assert result["target_reached"]
    del endpoint._latest  # Use the real wait path for the fault and recovery checks.
    clock = [endpoint._stream.sample["stamp_ms"]]
    endpoint._stream._now_ms = lambda: clock[0]

    def advance(distance):
        clock[0] += 100
        actual = goal.copy()
        actual[8] += distance
        feed(endpoint, clock[0], actual)

    for distance in (.004, .008, .012):
        advance(distance)
        assert not endpoint.fault
        assert messages(pub)[-1]["values"][8:] == initial[8:]
    published = pub.publish.call_count
    advance(.016)
    assert "right" in endpoint.fault
    with pytest.raises(eef.EefStreamFault):
        endpoint._latest(0)
    endpoint.cancel()
    with pytest.raises(eef.EefStreamFault):
        endpoint.begin_task()
    with pytest.raises(eef.EefStreamFault):
        endpoint.execute("right_set_gripper", {"opening_mm": 10.}, threading.Event())
    advance(0.)
    assert pub.publish.call_count == published


def test_preflight_requires_continuous_samples_after_feedback_gap():
    endpoint, _, _ = robot()
    now = eef._now_ms()
    for offset in (-730, -720, -40, -10):
        feed(endpoint, now + offset, full())
    endpoint.STATE_TIMEOUT = .01
    with pytest.raises(TimeoutError):
        endpoint._latest(0, window_ms=250)
    # Advance the wall clock alongside the hardware clock.
    endpoint._stream._now_ms = lambda: now + 240
    feed(endpoint, now + 240, full())
    assert len(endpoint._latest(0, window_ms=250)[2]) >= 3


def test_clipping_is_direction_preserving_and_applies_only_to_selected_group():
    desc = descriptor()
    desc["limits"]["upper"][0] = .12
    endpoint, _, _ = robot(**desc)
    start = full()
    start[15] = .09
    goal = start.copy()
    goal[0] = .12
    goal[1] += .01
    _, moved = drive(endpoint, "left_move_by", {"translation_m": [.04, .02, 0.], "rotation_vector_rad": [0,0,0]},
                     start, [goal]*10)
    assert moved["command"]["values"] == pytest.approx(goal)
    target = goal.copy()
    target[15] = .065
    _, jaw = drive(endpoint, "right_set_gripper", {"opening_mm": 80.}, goal, [target]*10)
    assert jaw["command"]["values"] == target


def test_selected_group_drift_or_dropped_subscriber_prevents_a_target():
    endpoint, _, pub = robot()
    start = full()
    drift = start.copy()
    drift[7] += .015
    _, result = drive(endpoint, "left_set_gripper", {"opening_mm": 3}, start, [], drift)
    assert result["status"] == "rejected" and not result["motion_sent"]
    assert messages(pub)[-1]["values"] == drift
    pub.get_subscription_count.return_value = 0
    with pytest.raises(RuntimeError, match="subscriber"):
        drive(endpoint, "left_set_gripper", {"opening_mm": 3}, start, [])
    assert messages(pub)[-1]["values"] == start


@pytest.mark.parametrize("tool,args", [
    ("left_move_by", {"translation_m": [.11, 0, 0], "rotation_vector_rad": [0, 0, 0]}),
    ("left_move_by", {"translation_m": [0, 0, 0], "rotation_vector_rad": [0, 0, .55]}),
    ("left_move_by", {"position_m": [.11, .23, .34], "quaternion_xyzw": [0, 0, 0, 1]}),
    ("left_set_gripper", {"opening_mm": float("nan")})])
def test_invalid_tool_target_can_only_publish_idle_echo(tool, args):
    endpoint, _, pub = robot()
    _, result = drive(endpoint, tool, args, full(), [])
    assert result["status"] == "rejected" and not result["motion_sent"]
    assert all(m["values"] == full() for m in messages(pub))


@pytest.mark.parametrize("side", ["left", "right"])
def test_retired_absolute_tool_cannot_publish_a_target(side):
    endpoint, _, pub = robot()
    start = full()
    position = start[endpoint.interface.offsets[side, "eef"]][:3]
    position[0] += .01
    with pytest.raises(RuntimeError):
        drive(endpoint, f"{side}_move_to", {"position_m": position, "quaternion_xyzw": [0, 0, 0, 1]}, start, [])
    assert all(command["values"] == start for command in messages(pub))


def test_settled_short_pose_with_jitter_returns_to_visual_loop():
    # Initial and final XYZ from tape-01's third approach. Oscillation is synthetic:
    # its amplitude exceeds the old 0.35 mm count-from-anchor threshold.
    start = full()
    start[:3] = [.437779, .294633, -.275202]
    poses = []
    for i in range(30):
        pose = start.copy()
        jitter = .00055 if i % 2 else -.00055
        pose[:3] = [.463766 + jitter, .294760, -.290190 - jitter/2.]
        angle = .010 if i % 2 else -.010
        q = [0., 0., math.sin(angle/2.), math.cos(angle/2.)]
        pose[3:7] = q if i % 2 else [-v for v in q]
        poses.append(pose)
    endpoint, _, _ = robot()
    _, out = drive(endpoint, "left_move_by", {"translation_m": [.03, 0, -.01],
                                               "rotation_vector_rad": [0, 0, 0]}, start, poses)
    assert out["status"] == "completed" and not out["target_reached"]
    assert out["completion_basis"] == "settled_measured_eef" and out["feedback_frames"] < 15
    assert .005 < out["position_error_m"] < .008


@pytest.mark.parametrize("after_hold", ["stopped", "drifting", "feedback_lost"])
def test_execution_deadline_verifies_motion_using_only_new_hold_measurements(after_hold):
    endpoint, _, pub = robot()
    endpoint.MOTION_TIMEOUT = 2.
    start = full()
    moving = []
    for i in range(12):
        pose = start.copy()
        pose[0] += .002 * (i + 1)
        moving.append(pose)
    # The simulated 5 Hz feedback first passes the 2 s command budget on sample 11.
    if after_hold == "stopped":
        feedback = moving + [moving[-1]] * 12
    elif after_hold == "drifting":
        # The speed is low enough for each 200 ms increment to be < 0.35 mm.
        # Over the whole rolling window its steady directed drift is detectable.
        feedback = moving.copy()
        for i in range(30):
            pose = moving[-1].copy()
            pose[0] += .00033 * (i + 1)
            feedback.append(pose)
    else:
        feedback = moving[:11]
    _, out = drive(endpoint, "left_move_by", {"translation_m": [.05, 0, 0], "rotation_vector_rad": [0, 0, 0]},
                   start, feedback)
    goal = start[0] + .05
    # The previously latched target has been replaced by a hold at a measured position.
    assert out["command"]["values"][0] == pytest.approx(goal)
    assert messages(pub)[-1]["values"][0] < goal - .01
    if after_hold == "stopped":
        assert out["status"] == "completed" and not out["target_reached"]
        assert out["completion_basis"] == "settled_after_measured_hold"
        assert out["state_stamp_ms"] - out["hold_command_stamp_ms"] >= endpoint.SETTLED_MS
        assert out["actual"]["position_m"] == moving[-1][:3]
    else:
        assert out["status"] == "outcome_unknown" and out["motion_sent"]
        assert ("motion continued" if after_hold == "drifting" else "feedback lost") in out["reason"]


def test_small_noisy_short_stroke_waits_for_late_motion_start():
    endpoint, _, _ = robot()
    start = full()
    feedback = []
    for i in range(10):
        pose = start.copy()
        pose[0] += .0006 if i % 2 else -.0006
        feedback.append(pose)
    actual = start.copy()
    actual[0] += .0025
    _, out = drive(endpoint, "left_move_by", {"translation_m": [.004, 0, 0], "rotation_vector_rad": [0, 0, 0]},
                   start, feedback + [actual] * 10)
    assert out["feedback_frames"] > 10 and out["displacement_observed"] and not out["target_reached"]
    assert out["measured_translation_m"][0] == pytest.approx(.0025)
