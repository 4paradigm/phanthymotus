"""Exercise the wired EEF protocol with measured packets and a recording ROS node."""
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
    return {"control_interface": "test.eef/1", "mode": "eef_pose", "dof": 8*len(sides), "frame": "world",
            "tip_frames": {side: f"{side}_tcp" for side in sides}, "opening_m": [0., .085],
            "groups": [{"name": f"{kind}_{side[0]}", "offset": 8*i+offset, "count": size, "unit": unit}
                       for i, side in enumerate(sides) for kind, offset, size, unit in
                       (("eef", 0, 7, "m+quat_xyzw"), ("gripper", 7, 1, "m"))]}


def robot(sides=("left", "right")):
    node = Mock()
    pub = node.create_publisher.return_value
    pub.get_subscription_count.return_value = 1
    endpoint = eef.EefEndpoint(node, "/state", "/control", eef.EefInterface(descriptor(sides)))
    return endpoint, node, pub


def full():
    return [.1, .23, .34, 0., 0., 0., 1., .008, -.11, -.22, .33, 0., 0., 0., 1., 0.]


def feed(endpoint, stamp, values):
    packet = {"schema": endpoint.interface.schema, "frame": endpoint.interface.frame,
              "stamp_ms": stamp, "values": values}
    endpoint._receive(SimpleNamespace(data=json.dumps(packet)))


def drive(endpoint, name, args, start, samples, drift=None):
    # Synthetic generation times advance 200 ms independently of test wall time.
    after = eef._now_ms()
    current = start if drift is None else drift
    endpoint._latest = Mock(side_effect=[(after, start, [start]), (after+1, current, [current])] +
                            [(after+200*i, values, [values]) for i, values in enumerate(samples, 1)])
    endpoint.observe([name.split("_")[0]])
    return endpoint.execute(name, args, threading.Event())


def test_group_layout_validated_and_frame_from_driver():
    endpoint, node, _ = robot(("right", "left"))
    interface = endpoint.interface
    assert interface == eef.EefInterface(descriptor(("right", "left")))
    assert interface.offsets["left", "eef"] == slice(8, 15)
    assert node.create_subscription.call_args.args[-1].reliability == "best_effort"
    assert node.create_publisher.call_args.args[-1].durability == "volatile"
    assert eef.EefInterface(descriptor(("right",))).dof == 8
    for change in ({"offset": 8}, {"unit": "mm"}, {"count": 6}, {"offset": -1}):
        desc = descriptor()
        desc["groups"][0].update(change)
        with pytest.raises(ValueError):
            eef.EefInterface(desc)
    for change in ({"groups": []}, {"dof": 15}, {"opening_m": [1., 0]}, {"frame": ""}):
        with pytest.raises(ValueError):
            eef.EefInterface(dict(descriptor(), **change))


def test_state_drops_invalid_duplicate_and_expired_measurements():
    endpoint, _, _ = robot()
    stamp = eef._now_ms() - 10
    base = full()
    feed(endpoint, stamp - 1500, base)
    feed(endpoint, stamp, base[:10])
    feed(endpoint, stamp, [math.nan] + base[1:])
    feed(endpoint, stamp, base[:6] + [2.] + base[7:])
    assert not endpoint._samples
    feed(endpoint, stamp, base)
    feed(endpoint, stamp, [123.] + base[1:])
    feed(endpoint, stamp - 1, [123.] + base[1:])
    assert endpoint._latest(0)[1] == base
    endpoint.STATE_TIMEOUT = .02
    with pytest.raises(TimeoutError, match="advancing"):
        endpoint._latest(stamp)
    with pytest.raises(ValueError):
        endpoint.observe(["other"])
    endpoint.close()
    with pytest.raises(InterruptedError):
        endpoint.observe(["left"])


def test_relative_rotation_about_tcp_uses_base_axes_and_measured_other_arm():
    endpoint, _, pub = robot()
    start = full()
    a = math.sqrt(.5)
    start[3:7] = [a, 0., 0., a]
    theta = .3
    s, c = math.sin(theta/2), math.cos(theta/2)
    goal = list(start)
    goal[1] += .05
    goal[3:7] = [c*a, s*a, s*a, c*a]
    latest = list(start)
    latest[8] += .0008  # Use the most recent opposite-arm measurement.
    result = drive(endpoint, "left_move_by", {"translation_m": [0., .05, 0.],
                                              "rotation_vector_rad": [0., 0., theta]}, start, [goal]*7, latest)
    message = json.loads(pub.publish.call_args.args[0].data)
    assert message["values"][:7] == pytest.approx(goal[:7])
    assert message["values"][7:] == latest[7:]
    assert message["schema"] == "test.eef/1" and message["frame"] == "world"
    assert result["command"] == message
    assert result["target_reached"] and result["displacement_observed"]
    assert result["measured_translation_m"] == pytest.approx([0., .05, 0.])
    assert result["actual"]["eef_frame"] == "left_tcp"


def test_gripper_no_motion_is_not_success_and_error_is_reported():
    endpoint, _, pub = robot()
    start = full()
    result = drive(endpoint, "right_set_gripper", {"opening_mm": 3}, start, [start]*30)
    assert pub.publish.call_count == 1
    assert json.loads(pub.publish.call_args.args[0].data)["values"] == start[:15] + [.003]
    assert result["status"] == "no_motion_observed"
    assert not result["target_reached"] and not result["displacement_observed"]
    assert result["requested_opening_mm"] == 3 and result["actual"]["opening_mm"] == 0
    assert result["feedback_frames"] >= 20


def test_slow_start_and_short_close_keep_clamping_force_on_next_step():
    endpoint, _, pub = robot()
    start = full()
    start[7] = .08
    contact = list(start)
    contact[7] = .04
    result = drive(endpoint, "left_set_gripper", {"opening_mm": 0.}, start, [start]*12 + [contact]*30)
    assert result["status"] == "completed" and not result["target_reached"]
    assert result["actual"]["opening_mm"] == 40.
    # Restating the closing target under load produces no further finger
    # travel. The next absolute frame must still carry that same closing target.
    repeated = drive(endpoint, "left_set_gripper", {"opening_mm": 0.}, contact, [contact]*30)
    assert repeated["status"] == "no_motion_observed"
    goal = list(contact)
    goal[0] += .02
    result = drive(endpoint, "left_move_to", {"position_m": goal[:3], "quaternion_xyzw": goal[3:7]},
                   contact, [goal]*7)
    messages = [json.loads(call.args[0].data) for call in pub.publish.call_args_list]
    assert messages[2]["values"][7] == 0. and messages[2]["values"][8:] == start[8:]
    assert messages[0]["stamp_ms"] < messages[1]["stamp_ms"] < messages[2]["stamp_ms"]
    assert result["actual"]["position_m"] == goal[:3]


def test_pending_or_unknown_other_motion_and_out_of_range_idle_measurement():
    endpoint, _, pub = robot()
    start = full()
    changed = list(start)
    changed[8] += .015
    args = {"opening_mm": 10}
    result = drive(endpoint, "left_set_gripper", args, start, [], changed)
    assert result["status"] == "rejected" and not pub.publish.called
    # Deliberately continuing our own in-flight absolute target on another group.
    pending = list(start[8:15])
    pending[1] += .03
    endpoint._goals["right", "eef"] = pending
    changed[15] = start[15] = .09
    feedback = list(changed)
    feedback[7] = .01
    drive(endpoint, "left_set_gripper", args, start, [feedback]*7, changed)
    sent = json.loads(pub.publish.call_args.args[0].data)["values"]
    assert sent[8:15] == pending and sent[15] == .085


def test_missing_feedback_does_not_replay_unknown_goal():
    endpoint, _, pub = robot()
    base = full()
    stamp = eef._now_ms()
    endpoint._latest = Mock(side_effect=[(stamp, base, [base]), (stamp+1, base, [base]),
                                        TimeoutError("no advancing feedback")])
    state = endpoint.observe(["right"])
    assert state["control_interface"]["opening_m"] == [0., .085] and set(state["arms"]) == {"right"}
    result = endpoint.execute("right_set_gripper", {"opening_mm": 3.}, threading.Event())
    assert result["status"] == "outcome_unknown" and result["motion_sent"]
    assert result["command"] == json.loads(pub.publish.call_args.args[0].data)
    assert result["command_stamp_ms"] == result["command"]["stamp_ms"]
    assert pub.publish.call_count == 1
    assert ("right", "gripper") not in endpoint._goals
    settled = list(base)
    settled[7] = .01
    drive(endpoint, "left_set_gripper", {"opening_mm": 10.}, base, [settled]*7)
    assert endpoint._latest.call_args_list[1].args[2] == endpoint.SETTLED_MS
    assert json.loads(pub.publish.call_args.args[0].data)["values"][15] == 0.


def test_preflight_requires_continuous_samples_after_feedback_gap():
    endpoint, _, _ = robot()
    base = full()
    now = eef._now_ms()
    for offset in (-995, -980, -200, -10):
        feed(endpoint, now+offset, base)
    endpoint.STATE_TIMEOUT = .01
    with pytest.raises(TimeoutError):
        endpoint._latest(0, window_ms=250)
    feed(endpoint, now+65, base)
    assert len(endpoint._latest(0, window_ms=250)[2]) >= 3


def test_dropped_connection_or_invalid_move_cannot_publish():
    endpoint, _, pub = robot()
    base = full()
    for name, args in (("left_set_gripper", {"opening_mm": 90}),
                       ("left_move_to", {"position_m": [.3, .23, .34], "quaternion_xyzw": [0.,0.,0.,1.]}),
                       ("left_move_by", {"translation_m": [0.,0.,0.], "rotation_vector_rad": [0.,0.,7.]})):
        stamp = eef._now_ms()
        endpoint._latest = Mock(return_value=(stamp, base, [base]))
        endpoint.observe(["left"])
        assert not endpoint.execute(name, args, threading.Event())["motion_sent"]
    pub.get_subscription_count.return_value = 0
    with pytest.raises(RuntimeError, match="subscriber"):
        drive(endpoint, "left_set_gripper", {"opening_mm": 2}, base, [])
    assert not pub.publish.called
