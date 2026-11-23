"""Exercise the EEF card through its registered tool and both ROS ports."""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.eef import plugin


POSE = {"x_m": .6, "y_m": .1, "z_m": -.2, "qx": .1, "qy": .2, "qz": .3, "qw": math.sqrt(.86)}


def measured():
    a = math.sqrt(.5)
    return [.54, .22, .31, a, 0., 0., a, .032,
            .48, -.27, .36, 0., 0., a, a, .046]


def descriptor():
    lower = [.25, -.21, -.67, -1., -1., -1., -1., 0.] + [.25, -.90, -.67, -1., -1., -1., -1., 0.]
    upper = [.732, .90, .5, 1., 1., 1., 1., .085] + [.732, .21, .5, 1., 1., 1., 1., .065]
    return {"control_interface": "motus.control/1", "mode": "eef_pose", "dof": 16, "frame": "base_Link",
            "state": {"schema": "motus.eef/1", "field": "eef"},
            "groups": [{"name": name, "offset": offset, "count": count, "unit": unit}
                       for name, offset, count, unit in
                       [("eef_l", 0, 7, "m+quat_xyzw"), ("gripper_l", 7, 1, "m"),
                        ("eef_r", 8, 7, "m+quat_xyzw"), ("gripper_r", 15, 1, "m")]],
            "limits": {"lower": lower, "upper": upper},
            "rate": {"expected_hz": 10, "watchdog_ms": 200, "max_obs_age_ms": 750}}


@pytest.fixture
def wire(monkeypatch):
    from plugins.eef import EefPlugin

    clock = [1700000000000]
    monkeypatch.setattr(plugin, "_now_ms", lambda: clock[0])
    factory = Mock()
    monkeypatch.setitem(sys.modules, "rclpy.node", SimpleNamespace(Node=factory))
    card = EefPlugin({}, Mock())

    def call(action, **args):
        return card.dispatch("eef", {"action": action, **args})

    def start(*, new_instance=False, interface=None):
        nonlocal card
        if new_instance:
            card = EefPlugin({}, Mock())
        node = Mock()
        factory.side_effect = lambda *_: node
        if interface is None:
            interface = descriptor()
        assert call("start", input_topics=["/bench/state/eef"], control_interface=interface)["state"] == "running"
        pub = node.create_publisher.return_value
        pub.get_subscription_count.return_value = 1
        sub = node.create_subscription.call_args.args
        assert sub[1] == "/bench/state/eef"
        assert sub[-1].reliability == "best_effort"
        assert node.create_publisher.call_args.args[1] == "/limx_tron2/control/eef"
        assert node.create_subscription.call_count == node.create_publisher.call_count == 1

        def feed(eef, age=0, field="eef", advance=100, schema="motus.eef/1", **extra):
            clock[0] += advance
            msg = String()
            msg.data = json.dumps({"schema": schema, "frame": interface["frame"],
                                   "stamp_ms": clock[0] - age, field: eef, **extra})
            sub[2](msg)
            return clock[0] - age

        return pub, feed

    pub, feed = start()
    yield SimpleNamespace(call=call, start=start, pub=pub, feed=feed, clock=clock)
    call("stop")


def commands(pub):
    result = []
    for published in pub.publish.call_args_list:
        command = json.loads(published.args[0].data)
        assert set(command) == {"schema", "frame", "stamp_ms", "values", "mode", "dof", "source", "seq",
                                "ttl_ms", "obs_stamp_ms", "priority"}
        assert (command["schema"], command["frame"], command["mode"]) == ("motus.control/1", "base_Link", "eef_pose")
        assert len(command["values"]) == command["dof"] == 16
        assert type(command["stamp_ms"]) is int
        assert command["source"].startswith("mcp__actucore__eef/")
        assert type(command["seq"]) is int and command["ttl_ms"] == 200
        if result:
            assert command["seq"] > result[-1]["seq"] and command["source"] == result[-1]["source"]
        result.append(command)
    return result


def move(wire, arm, **increments):
    frames = wire.pub.publish.call_count
    result = wire.call("move_by", arm=arm, **increments)
    assert result["status"] == "published", result
    assert wire.pub.publish.call_count == frames + 1
    return result["command"]["values"]


def test_bundle_registers_and_dispatches_single_eef_tool(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy.executors", ModuleType("rclpy.executors"))
    import importlib.util
    from unittest.mock import patch
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("eef_test_actucore_host", main_path)
    host = importlib.util.module_from_spec(spec)
    # Docker supplies yaml and installs a global stdout wrapper at import. This
    # test supplies config directly and only exercises bundle registration.
    with patch.dict(sys.modules, {"yaml": ModuleType("yaml"), "logsafe": SimpleNamespace(install=lambda: None)}):
        spec.loader.exec_module(host)

    bundle = host.ActuCoreBundle({"plugins": {"eef": {"enabled": True}}}, Mock())
    tools = bundle.get_all_tools()
    assert [t["name"] for t in tools] == ["eef"]
    assert [p["format"] for p in tools[0]["topic_in"]] == ["state/eef"]
    assert [p["format"] for p in tools[0]["topic_out"]] == ["control/eef"]
    schema = tools[0]["inputSchema"]
    assert "move_to" in schema["properties"]["action"]["enum"]
    assert schema["x-action-params"]["move_to"]["params"] == ["arm", *POSE, "opening_m"]
    assert all(schema["properties"][name]["type"] == "number" for name in (*POSE, "opening_m"))
    assert "cancel" in schema["properties"]["action"]["enum"]
    assert schema["x-action-params"]["cancel"]["params"] == []
    assert schema["x-hooks"]["on_interrupt_motion"] == {"action": "cancel"}
    assert schema["x-hooks"]["on_interrupt_all"] == {"action": "cancel"}
    assert bundle.dispatch("eef", {"action": "info"})["state"] == "idle"


def test_idle_output_follows_each_fresh_measurement(wire):
    wire.pub.publish.assert_not_called()
    samples = [measured() for _ in range(3)]
    samples[1][7] += .003
    samples[2][8] += .002
    stamps = [wire.feed(sample) for sample in samples]
    frames = commands(wire.pub)
    assert [f["values"] for f in frames] == samples
    assert [f["stamp_ms"] for f in frames] == stamps
    assert wire.call("cancel")["command"]["values"] == samples[-1]
    moved = samples[-1].copy()
    moved[8] += .001
    wire.feed(moved)
    assert commands(wire.pub)[-1]["values"] == moved


@pytest.mark.parametrize("arm,idx", [("left", 0), ("right", 8)])
def test_absolute_target_is_held_until_overwritten_from_measurement(wire, arm, idx):
    start = measured()
    wire.feed(start, age=47)
    first = move(wire, arm, dx_m=.1)
    expected = start.copy()
    expected[idx] += .1
    assert first == expected

    halfway = start.copy()
    halfway[idx] += .05
    halfway[idx+7] += .001
    replacement_observation = wire.feed(halfway, age=32)
    first_with_measured_jaws = halfway.copy()
    first_with_measured_jaws[idx] = first[idx]
    assert commands(wire.pub)[-1]["values"] == first_with_measured_jaws
    second = move(wire, arm, dx_m=.1)
    expected = halfway.copy()
    expected[idx] += .1
    assert second == expected
    assert second[idx] == pytest.approx(start[idx] + .15)

    wire.feed(halfway)
    wire.feed(expected)
    frames = commands(wire.pub)
    assert [f["values"] for f in frames] == [start, first, first_with_measured_jaws, second, second, second]
    stamps = [f["stamp_ms"] for f in frames]
    assert all(a <= b for a, b in zip(stamps, stamps[1:]))
    assert wire.call("info")["last"]["state_stamp_ms"] == replacement_observation


def test_switching_arms_replaces_old_target_and_uses_current_peer_pose(wire):
    current = measured()
    wire.feed(current)
    move(wire, "left", dx_m=-.01)
    current[0] -= .004
    wire.feed(current)
    target = move(wire, "right", dz_m=.009)
    assert target[:10] == current[:10]
    assert target[10] == pytest.approx(current[10] + .009)
    assert target[11:] == current[11:]
    shifted = current.copy()
    shifted[0] -= .002
    shifted[7] += .001
    wire.feed(shifted)
    assert commands(wire.pub)[-1]["values"][:8] == current[:8]
    assert move(wire, "right", opening_delta_m=.001)[:8] == current[:8]
    assert move(wire, "left", opening_delta_m=.001)[:7] == shifted[:7]


@pytest.mark.parametrize("arm,idx,peer", [("left", 0, 8), ("right", 8, 0)])
@pytest.mark.parametrize("opening", [None, .02])
def test_move_to_holds_absolute_pose_and_peer_until_cancel(wire, arm, idx, peer, opening):
    start = measured()
    wire.feed(start)
    expected = start.copy()
    expected[idx:idx + 7] = list(POSE.values())
    target = POSE.copy()
    if opening is not None:
        target["opening_m"] = expected[idx + 7] = opening
    result = wire.call("move_to", arm=arm, **target)
    assert wire.pub.publish.call_count == 2  # Feedback echo, then one command for the whole target.
    assert result["status"] == "published"
    assert result["command"]["values"] == commands(wire.pub)[-1]["values"] == expected

    current = start.copy()
    current[idx] += .01
    current[idx + 7] += .003
    current[peer] -= .002
    current[peer + 7] += .001
    wire.feed(current)
    if opening is None:
        expected[idx + 7] = current[idx + 7]
    assert commands(wire.pub)[-1]["values"] == expected
    expected[idx:idx + 8] = current[idx:idx + 8]
    assert wire.call("cancel")["command"]["values"] == expected


def test_move_to_zero_opening_holds_jaws_and_follows_measured_tcp(wire):
    start = measured()
    wire.feed(start)
    # A gripper-only target replaces any previous TCP target on the same arm.
    wire.call("move_to", arm="right", **POSE)
    expected = start.copy()
    expected[15] = 0.
    assert wire.call("move_to", arm="right", opening_m=0.)["command"]["values"] == expected
    current = start.copy()
    current[8] += .005
    current[0] += .004
    current[7] += .002
    wire.feed(current)
    expected[8] = current[8]
    assert commands(wire.pub)[-1]["values"] == expected


def test_move_to_requires_complete_valid_target_and_fresh_state(wire):
    valid = dict(arm="right", **POSE)
    assert wire.call("move_to", **valid)["state"] == "error"
    wire.pub.publish.assert_not_called()
    wire.feed(measured())
    active = move(wire, "left", dx_m=.01)
    before = wire.pub.publish.call_count
    assert wire.call("move_to", arm="right")["state"] == "error"
    assert wire.call("move_to", arm="right", x_m=.5, opening_m=.02)["state"] == "error"
    for opening in (None, False, "0.02", float("nan"), float("inf"), -.001, .07):
        assert wire.call("move_to", arm="right", opening_m=opening)["state"] == "error"
        assert wire.call("move_to", **valid, opening_m=opening)["state"] == "error"
    for key in valid:
        assert wire.call("move_to", **{k: v for k, v in valid.items() if k != key})["state"] == "error"
    for invalid in ({"arm": "both"}, {"x_m": True}, {"z_m": float("nan")}, {"qw": float("inf")},
                    {"qx": "0.1"}, {"qw": 0.}, {"x_m": .8}, {"y_m": .3}):
        assert wire.call("move_to", **{**valid, **invalid})["state"] == "error"
    wire.clock[0] += 2000
    assert wire.call("move_to", **valid)["state"] == "error"
    assert wire.pub.publish.call_count == before
    wire.feed(measured())
    assert commands(wire.pub)[-1]["values"] == active


def test_zero_increment_replaces_an_active_move_with_a_measured_hold(wire):
    start = measured()
    wire.feed(start)
    move(wire, "left", dx_m=.02)
    start[0] += .003
    wire.feed(start)
    assert move(wire, "left", dx_m=0.) == start
    wire.feed(start)
    assert commands(wire.pub)[-1]["values"] == start


def test_cancel_immediately_latches_measured_state_until_replaced(wire):
    start = measured()
    wire.feed(start)
    target = move(wire, "right", dz_m=.008, opening_delta_m=.01)
    halfway = start.copy()
    halfway[10] += .003
    halfway[15] += .002
    stamp = wire.feed(halfway)
    assert commands(wire.pub)[-1]["values"] == target

    frames = wire.pub.publish.call_count
    result = wire.call("cancel")
    assert result["status"] == "published"
    assert wire.pub.publish.call_count == frames + 1
    hold = commands(wire.pub)[-1]
    assert hold["values"] == halfway and hold["stamp_ms"] == stamp

    latest = halfway.copy()
    latest[0] -= .002
    latest[10] += .001
    latest[15] += .001
    latest_stamp = wire.feed(latest)
    assert wire.call("cancel")["status"] == "published"
    last_two = commands(wire.pub)[-2:]
    assert [f["values"] for f in last_two] == [start[:8] + halfway[8:], start[:8] + latest[8:]]
    assert [f["stamp_ms"] for f in last_two] == [latest_stamp, latest_stamp]
    latest[10] += .001
    wire.feed(latest)
    assert move(wire, "right", dz_m=-.002)[10] == pytest.approx(latest[10] - .002)


def test_cancel_without_fresh_feedback_never_resumes_the_old_target(wire):
    assert wire.call("cancel")["status"] == "waiting_for_state"
    wire.pub.publish.assert_not_called()
    start = measured()
    wire.feed(start)
    move(wire, "left", dx_m=.008)
    wire.clock[0] += 5000
    assert wire.call("cancel")["status"] == "waiting_for_state"
    wire.feed(start, age=5000)
    assert wire.pub.publish.call_count == 2
    latest = start.copy()
    latest[0] += .002
    wire.feed(latest)
    assert commands(wire.pub)[-1]["values"] == latest
    resumed_stamp = wire.feed(start)
    hold = commands(wire.pub)[-1]
    assert hold["values"] == latest and hold["stamp_ms"] == resumed_stamp
    wire.call("stop")
    assert wire.call("cancel")["state"] == "idle"
    assert wire.pub.publish.call_count == 4


def test_cancel_during_feedback_publish_replaces_in_flight_target_last(wire):
    start = measured()
    wire.feed(start)
    target = move(wire, "left", dx_m=-.008)
    halfway = start.copy()
    halfway[0] -= .003
    publishing, release, cancelling = Event(), Event(), Event()
    sent = []

    def publish(msg):
        eef = json.loads(msg.data)["values"]
        if eef == target:
            publishing.set()
            assert release.wait(2)
        sent.append(eef)

    def cancel():
        cancelling.set()
        return wire.call("cancel")

    wire.pub.publish.side_effect = publish
    with ThreadPoolExecutor(max_workers=2) as workers:
        feedback = workers.submit(wire.feed, halfway)
        try:
            assert publishing.wait(2)
            cancelled = workers.submit(cancel)
            assert cancelling.wait(2)
        finally:
            release.set()
        feedback.result(timeout=2)
        assert cancelled.result(timeout=2)["status"] == "published"
    assert sent == [target, halfway]


def test_rotation_uses_reference_axes_about_the_selected_tcp(wire):
    start = measured()
    wire.feed(start)
    angle = .7
    target = move(wire, "left", dy_m=.008, dz_m=-.006, rz_rad=angle)
    assert target[:3] == pytest.approx([start[0], start[1]+.008, start[2]-.006])
    # Rz(delta) * Rx(90 degrees). Reversing the multiplication flips y.
    a = math.sqrt(.5)
    s, c = math.sin(angle/2), math.cos(angle/2)
    assert target[3:7] == pytest.approx([c*a, s*a, s*a, c*a])
    assert target[7:] == start[7:]


@pytest.mark.parametrize("arm,index,delta", [("left", 7, .09), ("right", 15, .09), ("right", 15, -.09)])
def test_gripper_target_is_clipped_to_its_declared_limit(wire, arm, index, delta):
    start = measured()
    wire.feed(start)
    target = move(wire, arm, opening_delta_m=delta)
    assert target[index] == (descriptor()["limits"]["upper"][index] if delta > 0 else 0.)
    assert target[:index] + target[index+1:] == start[:index] + start[index+1:]
    wire.feed(start)
    assert commands(wire.pub)[-1]["values"] == target


def test_repeated_fresh_observation_is_a_heartbeat(wire):
    start = measured()
    stamp = wire.feed(start)
    wire.feed(start, age=100)
    target = move(wire, "left", dz_m=.008)
    wire.feed(start, age=200)
    frames = commands(wire.pub)
    assert [f["values"] for f in frames] == [start, start, target, target]
    assert [f["stamp_ms"] for f in frames] == [stamp, stamp + 100, stamp + 100, stamp + 200]
    assert [f["obs_stamp_ms"] for f in frames] == [stamp] * 4
    wire.feed(start, age=1500, advance=1300)
    assert wire.pub.publish.call_count == len(frames)


def test_long_running_gripper_target_keeps_peer_and_uses_live_selected_tcp(wire):
    start = measured()
    wire.feed(start)
    fixed = move(wire, "left", opening_delta_m=.01)[7]
    for step in range(1, 15):
        sample = start.copy()
        sample[0] += step * .0002
        sample[7] += step * .0001
        sample[10] -= step * .0003
        sample[15] = .0904  # A raw measured gap may exceed the commanded range.
        observed = wire.feed(sample, age=20)
        command = commands(wire.pub)[-1]
        expected = sample[:8] + start[8:]
        expected[7] = fixed
        assert command["values"] == expected
        assert command["obs_stamp_ms"] == observed and command["stamp_ms"] - observed == 20

    before = wire.pub.publish.call_count
    wire.clock[0] += 2000
    wire.feed(sample, age=2000)
    assert wire.pub.publish.call_count == before
    new_stamp = wire.feed(sample, age=37)
    resumed = commands(wire.pub)[-1]
    assert resumed["values"] == expected and resumed["stamp_ms"] == new_stamp + 37


@pytest.mark.parametrize("drift", ["translation", "rotation"])
def test_peer_drift_suspends_all_output_until_card_restart(wire, drift):
    start = measured()
    wire.feed(start)
    move(wire, "left", dx_m=.015)
    perturbed = start.copy()
    if drift == "translation":
        perturbed[9] += .013
    else:
        perturbed[11:15] = [0., 0., 0., 1.]
    wire.feed(perturbed)
    wire.feed(perturbed, advance=0)
    assert wire.call("info")["state"] == "running"
    equivalent = start.copy()
    equivalent[11:15] = [-v for v in start[11:15]]
    wire.feed(equivalent)
    wire.feed(perturbed)
    before = wire.pub.publish.call_count
    wire.feed(perturbed)
    fault = wire.call("info")["fault"]
    assert "Unselected right TCP" in fault and "mm" in fault and "degrees" in fault
    assert wire.pub.publish.call_count == before
    wire.feed(start)
    assert wire.call("info")["sample"]["eef"] == start
    assert wire.call("cancel") == {"state": "error", "message": fault}
    assert wire.call("move_by", arm="right", dx_m=.001) == {"state": "error", "message": fault}
    assert wire.pub.publish.call_count == before
    wire.call("stop")
    restarted, feed = wire.start()
    feed(perturbed)
    assert commands(restarted)[-1]["values"] == perturbed


def test_new_task_reads_new_peer_hold_on_first_action():
    from plugins.eef.stream import EefStream
    packets = []
    stream = EefStream(descriptor(), "test", packets.append, now_ms=lambda: 110)
    initial = measured()
    packet = {"schema": "motus.eef/1", "frame": "base_Link", "stamp_ms": 100, "eef": initial}
    stream.receive(packet)
    jaw = stream.spec.offsets["left", "gripper"]
    stream.set_target("left", jaw, [.01])
    stream.cancel()
    later = initial.copy()
    later[8] += .001
    later[15] += .002
    stream.begin_task()
    stream.receive(dict(packet, stamp_ms=101, eef=later))
    assert packets[-1]["values"][8:] == initial[8:]
    assert stream.set_target("left", jaw, [.02])["command"]["values"][8:] == later[8:]


def test_no_republish_from_stale_or_previous_session_callback(wire):
    start = measured()
    wire.feed(start)
    move(wire, "left", dx_m=-.01)
    conflict = start.copy()
    conflict[0] += .001
    wire.feed(conflict, age=20, advance=10)
    wire.feed(start, age=5000)
    assert wire.call("info")["sample"]["eef"] == start
    assert wire.pub.publish.call_count == 2
    wire.clock[0] += 5000
    assert wire.call("move_by", arm="left", dx_m=.005)["state"] == "error"

    assert wire.call("stop")["state"] == "idle"
    wire.feed(start)
    assert wire.pub.publish.call_count == 2
    next_pub, next_feed = wire.start()
    wire.feed(start)
    assert wire.call("move_by", arm="left", dx_m=.005)["state"] == "error"
    next_feed(start)
    next_commands = commands(next_pub)
    assert [f["values"] for f in next_commands] == [start]
    assert wire.call("move_by", arm="left", dx_m=.005)["status"] == "published"
    wire.feed(start)
    assert next_pub.publish.call_count == wire.pub.publish.call_count == 2


def test_move_and_cancel_use_latest_pose_when_gripper_sample_keeps_timestamp(wire):
    start = measured()
    stamp = wire.feed(start)
    newer = start.copy()
    newer[0] += .003
    wire.feed(newer, age=50, advance=50)
    assert wire.call("info")["sample"] == {"schema": "motus.eef/1", "frame": "base_Link",
                                          "stamp_ms": stamp, "eef": newer}
    assert move(wire, "left", dx_m=.01)[0] == pytest.approx(newer[0] + .01)
    newer[0] += .004
    wire.feed(newer, age=75, advance=25)
    assert wire.call("cancel")["command"]["values"] == newer


def test_bad_input_never_generates_motion(wire):
    args = {"arm": "left", "dz_m": .01}
    assert wire.call("move_by", **args)["state"] == "error"
    wire.feed(measured(), field="values")
    wire.feed(measured(), schema="motus.control/1")
    wire.feed(measured(), schema="tron2.eef/1")
    wire.feed(measured(), unused=True)
    assert wire.call("move_by", **args)["state"] == "error"
    wire.pub.publish.assert_not_called()
    wire.feed(measured())
    assert wire.call("move_by", arm="both", dx_m=.01)["state"] == "error"
    assert wire.call("move_by", arm="right", rz_rad=float("nan"))["state"] == "error"
    assert [f["values"] for f in commands(wire.pub)] == [measured()]

def test_cartesian_projection_uses_first_face_and_latches_reachable_goal(wire):
    current = measured()
    wire.feed(current)
    goal = move(wire, "left", dx_m=1., dy_m=1., dz_m=-.5)
    fraction = descriptor()["limits"]["upper"][0] - current[0]
    assert goal[:3] == pytest.approx([.732, current[1] + fraction, current[2] - .5 * fraction])
    current[0] += .01
    wire.feed(current)
    assert commands(wire.pub)[-1]["values"] == goal


def test_start_requires_matching_descriptor_and_never_commands_another_layout(wire):
    wire.call("stop")
    assert wire.call("start", input_topic="/bench/state/eef")["state"] == "error"
    bad = descriptor()
    bad["groups"][0]["offset"] = 8
    assert wire.call("start", input_topic="/bench/state/eef", control_interface=bad)["state"] == "error"
    wrong_state = descriptor()
    wrong_state["state"]["schema"] = "other/1"
    assert wire.call("start", input_topic="/bench/state/eef", control_interface=wrong_state)["state"] == "error"
    assert wire.call("info")["state"] == "idle"
    wire.pub.publish.assert_not_called()


def test_producer_identity_and_replay_across_restarts(wire):
    wire.feed(measured())
    old = commands(wire.pub)[-1]
    wire.call("stop")
    pub, feed = wire.start()
    feed(measured(), advance=0)
    restarted = commands(pub)[-1]
    assert restarted["source"] == old["source"] and restarted["seq"] > old["seq"]
    wire.call("stop")
    pub, feed = wire.start(new_instance=True)
    feed(measured(), advance=0)
    assert commands(pub)[-1]["source"] != old["source"]


def test_outside_measured_peer_is_echoed_without_commanding_it_to_an_edge(wire):
    current = measured()
    current[8] = .24
    wire.feed(current)
    assert move(wire, "left", opening_delta_m=.02)[8:] == current[8:]
    before = wire.pub.publish.call_count
    assert wire.call("move_by", arm="right", dy_m=-.002)["state"] == "error"
    assert wire.pub.publish.call_count == before


@pytest.mark.parametrize("side,base", [("left", 0), ("right", 8)])
def test_stream_negotiates_single_arm_with_gripper_before_pose(side, base, wire):
    from plugins.eef.stream import EefStream

    desc = descriptor()
    desc.update(dof=8, frame="table", tip_frames={side: "tool0"})
    # Group order in the descriptor and order in the wire vector are independent.
    pose, jaw = [g for g in desc["groups"] if g["name"].endswith(f"_{side[0]}")]
    desc["groups"] = [dict(pose, offset=1), dict(jaw, offset=0)]
    for key, bounds in desc["limits"].items():
        desc["limits"][key] = bounds[base + 7:base + 8] + bounds[base:base + 7]
    current = measured()
    values = current[base + 7:base + 8] + current[base:base + 7]
    packets = []
    stream = EefStream(desc, "test-source", packets.append, now_ms=lambda: 101)
    sample = {"schema": "motus.eef/1", "stamp_ms": 100, "frame": "table", "eef": values}
    assert stream.receive(sample)
    assert packets[-1]["values"] == values
    target = stream.set_target(side, stream.spec.offsets[side, "gripper"], [.01])["command"]
    assert target["values"] == [.01] + values[1:]
    assert (target["dof"], target["frame"], target["obs_stamp_ms"]) == (8, "table", 100)
    peer = "right" if side == "left" else "left"
    with pytest.raises(ValueError, match="arm"):
        stream.set_target(peer, slice(0, 1), [.02])
    assert len(packets) == 2
    assert stream.cancel()["command"]["values"] == values

    wire.call("stop")
    pub, feed = wire.start(interface=desc)
    feed(values)
    assert wire.call("move_to", arm=peer, **POSE)["state"] == "error"
    assert pub.publish.call_count == 1
    moved = wire.call("move_to", arm=side, **POSE)["command"]
    assert (moved["frame"], moved["dof"], moved["values"]) == ("table", 8, [values[0], *POSE.values()])
    combined = wire.call("move_to", arm=side, **POSE, opening_m=.015)["command"]
    assert combined["values"] == [.015, *POSE.values()]


def test_advisory_actuator_group_is_rejected_before_start(wire):
    wire.call("stop")
    desc = descriptor()
    desc["groups"][1]["advisory"] = True
    result = wire.call("start", input_topic="/bench/state/eef", control_interface=desc)
    assert result["state"] == "error" and "gripper_l" in result["message"]
    wire.pub.publish.assert_not_called()
