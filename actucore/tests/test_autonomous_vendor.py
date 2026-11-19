"""Exercise the card's native WebSocket protocol with an in-memory TRON2."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.autonomous_task import vendor


class RobotSocket:
    serial = "DACH_TRON2A_379"

    def __init__(self):
        self.info = {"working_mode_valid": True, "working_mode": "developer_mode",
                     "collision_enable_valid": True, "collision_enable": True,
                     "compliant_enable_valid": True, "compliant_enable": False}
        self.sent = []
        self.inbox = []
        self.q = [0.] * 16
        self.dq = [0.] * 16
        self.claws = {"left_opening": 38, "left_speed": 0, "left_force": 0,
                      "right_opening": 3, "right_speed": 0, "right_force": 0}
        self.tick = 0
        self.step = 1
        self.after_command = lambda request: None
        self.names = []
        self.closed = 0
        self.ack_serial = self.serial

    async def ws_connect(self, url, **kwargs):
        assert url == "ws://fixture.invalid"
        self.inbox.append({"accid": self.serial, "title": "notify_robot_info", "data": self.info})
        return self

    async def close(self):
        self.closed += 1

    async def send_str(self, text):
        request = json.loads(text)
        self.sent.append(request)
        assert request["accid"] == self.serial and request["timestamp"] > 0
        assert isinstance(request["guid"], str)
        op = request["title"].removeprefix("request_")
        self.tick += self.step
        data = {"result": "success", "timestamp": self.tick}
        if op == "get_joint_state":
            data.update(q=list(self.q), dq=list(self.dq), names=self.names)
        elif op == "get_limx_2fclaw_state":
            data.update(self.claws)
        else:
            self.after_command(request)
        reply = {"accid": self.ack_serial, "title": "response_" + op, "guid": request["guid"], "data": data}
        self.inbox.extend([{**reply, "guid": "different-query", "data": {"result": "fail"}}, reply])

    async def receive_str(self):
        return json.dumps(self.inbox.pop(0), allow_nan=False)

    @property
    def motions(self):
        return [r for r in self.sent if r["title"] in ("request_movej", "request_set_limx_2fclaw_cmd")]


@pytest.fixture
def sdk(monkeypatch):
    robot = RobotSocket()
    class ClientSession:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return robot

        async def __aexit__(self, *exc):
            pass

    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientSession=ClientSession))
    monkeypatch.setattr(vendor, "_POLL_INTERVAL", .002)
    monkeypatch.setattr(vendor, "_SETTLE_INTERVAL", .006)
    monkeypatch.setattr(vendor, "_GRIPPER_START_WINDOW", .03)
    return vendor.Vendor("ws://fixture.invalid"), robot


def test_new_read_correlates_both_measurements_and_closes_socket(sdk):
    client, robot = sdk
    robot.names = list(reversed(vendor._JOINTS))
    robot.q = list(reversed(range(16)))
    snapshot = client.snapshot()
    assert snapshot["q"] == list(range(16))
    assert snapshot["grippers"]["left"]["opening_mm"] == pytest.approx(38 * 85 / 94)
    assert snapshot["grippers"]["right"]["vendor_timestamp_ms"] > 0
    assert robot.closed == 1 and not robot.motions
    robot.serial = "DACH_TRON2A_999"
    with pytest.raises(RuntimeError, match="identity"):
        client.snapshot()
    assert robot.closed == 2
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.snapshot()


@pytest.mark.parametrize("requested,raw", [(0, 0), (.001, 1), (60, 66), (85, 94)])
def test_writes_only_selected_gripper_and_returns_actual(sdk, requested, raw):
    client, robot = sdk
    robot.after_command = lambda _: robot.claws.update(left_opening=24)
    result = client.set_gripper("left", requested)
    assert len(robot.motions) == 1
    assert robot.motions[0]["data"] == {"left_opening": raw, "left_speed": 15, "left_force": 30}
    assert result["action_id"] == robot.motions[0]["guid"]
    assert result["grippers"]["left"]["opening_raw"] == 24
    assert result["displacement_observed"]


def test_contact_can_leave_opening_unchanged_after_observation_window(sdk):
    client, robot = sdk
    now = vendor.time.monotonic()
    result = client.set_gripper("left", 0)
    assert vendor.time.monotonic() - now >= vendor._GRIPPER_START_WINDOW + vendor._SETTLE_INTERVAL
    assert result["grippers"]["left"]["opening_raw"] == 38
    assert not result["displacement_observed"] and len(robot.motions) == 1


def test_movej_waits_full_duration_and_reports_unchanged_measured_state(sdk):
    client, robot = sdk
    target = [.05] + [0.] * 13
    now = vendor.time.monotonic()
    result = client.move_j(target, 1, "left")
    assert vendor.time.monotonic() - now >= 1
    assert result["q"] == [0.] * 16
    assert not result["displacement_observed"]
    assert [m["data"] for m in robot.motions] == [{"joint": target, "time": 1}]


def test_lost_ack_latches_unconfirmed_and_never_retransmits(sdk):
    client, robot = sdk
    def disconnect(_):
        raise ConnectionError("disconnected")
    robot.after_command = disconnect
    with pytest.raises(RuntimeError, match="unconfirmed"):
        client.set_gripper("left", 0)
    with pytest.raises(RuntimeError, match="Previous"):
        client.set_gripper("left", 0)
    assert len(robot.motions) == 1


def test_preflight_rejects_cancel_invalid_mode_and_moving_robot(sdk):
    client, robot = sdk
    client.cancel_event = threading.Event()
    client.cancel_event.set()
    with pytest.raises(InterruptedError):
        client.set_gripper("left", 0)
    client.cancel_event.clear()
    robot.info["collision_enable"] = False
    with pytest.raises(RuntimeError, match="collision protection"):
        client.move_j([0.] * 14, 1, "left")
    robot.info["collision_enable"] = True
    robot.dq[8] = .2
    with pytest.raises(RuntimeError, match="moving"):
        client.set_gripper("left", 0)
    assert not robot.motions


def test_rejects_wrong_robot_reply_and_oversized_targets(sdk):
    client, robot = sdk
    robot.ack_serial = "DACH_TRON2A_999"
    with pytest.raises(RuntimeError, match="another robot"):
        client.snapshot()
    with pytest.raises(ValueError):
        client.move_j([5.] * 14, 1, "left")
    with pytest.raises(ValueError):
        client.set_gripper("right", 86)
    assert not robot.motions


def test_other_arm_movement_after_ack_terminates(sdk):
    client, robot = sdk
    robot.after_command = lambda _: robot.q.__setitem__(8, .05)
    with pytest.raises(RuntimeError, match="Held arm moved"):
        client.move_j([.05] + [0.] * 13, 1, "left")
    assert len(robot.motions) == 1


def test_cancellation_after_send_waits_for_this_command_once(sdk):
    client, robot = sdk
    client.cancel_event = threading.Event()
    def accepted(_):
        client.cancel_event.set()
        robot.claws.update(right_opening=66)
    robot.after_command = accepted
    result = client.set_gripper("right", 60)
    assert result["grippers"]["right"]["opening_raw"] == 66
    with pytest.raises(InterruptedError):
        client.set_gripper("right", 60)
    assert len(robot.motions) == 1


def test_duplicate_source_timestamps_do_not_prove_completion(sdk):
    client, robot = sdk
    send = robot.send_str
    polls = 0
    async def send_with_stale_then_disconnected_feedback(text):
        nonlocal polls
        if robot.motions and json.loads(text)["title"] == "request_get_joint_state":
            polls += 1
            # This leaves time for ten apparent stationary samples. The source
            # timestamp never progresses after the command, so none is valid.
            if polls == 11:
                raise ConnectionError("no new telemetry")
        await send(text)
    robot.send_str = send_with_stale_then_disconnected_feedback
    def accepted(_):
        robot.step = 0
        robot.claws.update(left_opening=66)
    robot.after_command = accepted
    with pytest.raises(RuntimeError, match="no new telemetry"):
        client.set_gripper("left", 60)
    assert polls == 11 and len(robot.motions) == 1


def test_movej_holds_explicit_other_side_even_if_selected_motion_is_smaller(sdk):
    client, robot = sdk
    with pytest.raises(ValueError, match="hold the other arm"):
        client.move_j([.02] + [0.] * 13, 1, "right")
    assert not robot.motions


def test_movej_returns_loaded_jaw_change_to_observation(sdk):
    client, robot = sdk
    robot.after_command = lambda _: robot.claws.update(right_opening=21, right_speed=2)
    target = [0.] * 7 + [.05] + [0.] * 6
    result = client.move_j(target, 1, "right")
    assert result["grippers"]["right"]["opening_raw"] == 21
    assert result["grippers"]["right"]["speed"] == 2
    assert result["grippers"]["left"]["opening_raw"] == 38
    assert len(robot.motions) == 1
