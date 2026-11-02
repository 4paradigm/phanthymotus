"""ROS-free calibration action through the actual navi command/card path."""
from __future__ import annotations

import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navi import NaviPlugin  # noqa: E402
from plugins.navi import plugin as P  # noqa: E402


class _Clock:
    elapsed = 0.0

    def wall(self):
        return 1800000000.0 + self.elapsed

    def monotonic(self):
        return 1000.0 + self.elapsed

    def advance(self, seconds=0.1):
        self.elapsed = round(self.elapsed + seconds, 9)


class _Publisher:
    def __init__(self):
        self.commands = []
        self.fail = False

    def publish(self, payload):
        if self.fail:
            raise RuntimeError("simulated command transport failure")
        self.commands.append(json.loads(payload.data))


class _ImmediateThread:
    def __init__(self, *, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def _descriptor():
    return {"control_interface": "motus.control/1", "mode": "twist", "dof": 6,
            "joint_names": ["vx", "vy", "vz", "wx", "wy", "wz"],
            "frame": "base_link", "dry_run": False,
            "units": {"linear": "m/s", "angular": "rad/s", "time": "s"},
            "limits": {"lower": [-0.3, 0, 0, 0, 0, -1.0],
                       "upper": [0.3, 0, 0, 0, 0, 1.0]},
            "rate": {"max_hz": 20, "expected_hz": 10, "watchdog_ms": 300}}


@pytest.fixture
def rig(monkeypatch, tmp_path):
    clock = _Clock()
    monkeypatch.setattr(P.time, "time", clock.wall)
    monkeypatch.setattr(P.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(P.threading, "Thread", _ImmediateThread)

    def card(**cfg):
        instance = NaviPlugin({"calibration_dir": str(tmp_path), **cfg}, executor=None)
        instance._running = True
        instance._publisher = _Publisher()
        instance._descriptor = _descriptor()
        instance._binding = {"odom": "/robot/state/odom", "objects": "/robot/camera/objects",
                             "depth_map": "/robot/camera/depth"}
        instance._publish_view = lambda: None
        instance.events = []
        instance._post_acp = lambda *args: instance.events.append(args)
        feed(instance)
        return instance

    def feed(instance, wz=0.0, **over):
        payload = {"schema": "motus.odom/1", "frame": "body",
                   "units": {"linear": "m/s", "angular": "rad/s", "length": "m"},
                   "stamp_ms": int(clock.wall() * 1000),
                   "twist": [0.0, 0.0, None, None, None, wz]}
        payload.update(over)
        instance._on_string("odom", SimpleNamespace(data=json.dumps(payload)))

    return SimpleNamespace(clock=clock, card=card, feed=feed, directory=tmp_path)


def _begin(card):
    return card.dispatch("navi", {"action": "calibrate_yaw"})


def _complete(rig, card, *, begin=True):
    if begin:
        result = _begin(card)
        assert result["state"] == "running", result
    publisher = card._publisher
    card._tick()
    for _ in range(130):
        if card._calibration is None:
            break
        wz = publisher.commands[-1]["values"][5]
        rig.clock.advance()
        rig.feed(card, wz)
        card._tick()
    return card._info(), publisher.commands


def _camera_frame(rig, card):
    card._on_string("objects", SimpleNamespace(data=json.dumps({
        "timestamp": rig.clock.wall(), "objects": [
            {"name": "person", "position": [0.2, 0.0], "confidence": 0.9}]})))
    card._depth_bands = {"left": 5.0, "center": 5.0, "right": 5.0}
    card._depth_ms = int(rig.clock.wall() * 1000)


def test_calibration_uses_real_odom_and_guarded_command_stream_then_persists(rig):
    card = rig.card()
    result = _begin(card)
    info, commands = _complete(rig, card, begin=False)
    assert info["calibration"]["status"] == "completed", info["calibration"]
    assert commands[-1]["values"] == [0.0] * 6
    assert {command["values"][5] for command in commands} == {-0.5, 0.0, 0.5}
    turns = [command["values"][5] for command in commands]
    first_positive = next(index for index, value in enumerate(turns) if value > 0)
    first_negative = next(index for index, value in enumerate(turns) if value < 0)
    assert turns[first_negative - 1] == 0.5
    assert 0.0 not in turns[first_positive:first_negative]
    assert all(command["values"][:5] == [0.0] * 5 for command in commands)
    assert all(command["schema"] == "motus.control/1" for command in commands)
    assert [command["seq"] for command in commands] == list(range(1, len(commands) + 1))
    assert all(command["obs_stamp_ms"] == command["stamp_ms"] for command in commands)
    assert info["target"] == "" and card._calibration is None
    assert card.events[-1][0:2] == (result["action_id"], "completed")
    completion = card.events[-1][2]
    assert completion["p"] == completion["k_yaw"] == info["calibration"]["profile"]["k_yaw"]
    assert completion["k_fwd"] == 0.9
    assert completion["elapsed_s"] == pytest.approx(10.0)
    assert completion["samples"] > 0
    assert info["calibration"]["commanded_wz_rad_s"] == pytest.approx(0.5)
    assert info["gains"]["forward"]["base"] == 0.9
    assert info["gains"]["yaw"]["source"] == "calibrated"
    assert info["gains"]["yaw"]["base"] == info["calibration"]["profile"]["k_yaw"]
    assert info["calibration"]["profile"]["yaw_damping"] == 0.0
    assert "damping" not in info["gains"]["yaw"]
    assert list(rig.directory.glob("yaw-*.json"))


def test_same_connection_reloads_its_profile_without_resuming_a_goal(rig):
    first = rig.card()
    profile = _complete(rig, first)[0]["calibration"]["profile"]
    second = rig.card()
    second._load_yaw_profile()
    info = second._info()
    assert info["calibration"]["status"] == "loaded"
    assert info["gains"]["yaw"]["base"] == profile["k_yaw"]
    assert info["gains"]["forward"]["base"] == 0.9
    before = len(second._publisher.commands)
    second._tick()
    assert second._state.target == "" and len(second._publisher.commands) == before


def test_four_hz_odometry_with_ten_hz_command_ticks_calibrates_and_persists(rig):
    card = rig.card()
    assert _begin(card)["state"] == "running"
    card._tick()
    # Independent event clocks: an ODOM callback can arrive between two
    # command ticks. Re-reading its cached value never creates another sample.
    for event in range(1, 241):
        rig.clock.advance(0.05)
        if event % 5 == 0:
            rig.feed(card, card._publisher.commands[-1]["values"][5])
        if event % 2 == 0:
            card._tick()
        if card._calibration is None:
            break
    report = card._info()["calibration"]
    assert report["status"] == "completed", report
    assert report["elapsed_s"] == pytest.approx(10.0)
    assert report["samples"] == 41
    assert len(card._publisher.commands) == 101
    assert card._publisher.commands[-1]["values"] == [0.0] * 6
    assert list(rig.directory.glob("yaw-*.json"))


def test_stopping_a_normal_navigation_goal_clears_its_completion_lease_before_calibrating(rig):
    card = rig.card()
    navigation = card.dispatch("navi", {"action": "navigate_to", "target": "person"})
    assert navigation["state"] == "running" and card._acp_action_id
    assert _begin(card)["state"] == "error"
    card.dispatch("navi", {"action": "stop_navigation"})
    assert card._state.target == "" and card._acp_action_id == ""
    assert card.events[-1][0] == navigation["action_id"]
    assert card.events[-1][2]["status"] == "cancelled"
    assert _begin(card)["state"] == "running"


def test_normal_following_reports_actual_calibrated_yaw_gain_and_fixed_forward_gain(rig):
    card = rig.card()
    profile = _complete(rig, card)[0]["calibration"]["profile"]
    card.dispatch("navi", {"action": "navigate_to", "target": "person"})
    for _ in range(6):
        rig.clock.advance()
        rig.feed(card, 0.0)
        _camera_frame(rig, card)
        card._tick()
    gains = card._info()["gains"]
    assert gains["forward"] == {"base": 0.9, "effective": 0.9}
    assert gains["yaw"]["effective"] == profile["k_yaw"]
    assert "damping_effective" not in gains["yaw"]
    assert card._publisher.commands[-1]["values"][0] > 0.0
    assert card._publisher.commands[-1]["values"][5] < 0.0


def test_calibration_feedback_budget_does_not_relax_live_odom_freshness(rig):
    card = rig.card()
    _complete(rig, card)
    card.dispatch("navi", {"action": "navigate_to", "target": "person"})
    for _ in range(6):
        rig.clock.advance()
        rig.feed(card, 0.02)
        _camera_frame(rig, card)
        card._tick()
    rig.clock.advance(0.301)
    _camera_frame(rig, card)
    assert card._calibration_feedback()[3] == ""  # 301 ms fits calibration's 650 ms bound.
    assert card.observation()[2]["wz"] is None  # Live ego-motion still uses 300 ms.


@pytest.mark.parametrize("problem", ["offline", "paused", "goal", "dry_run", "no_descriptor",
                                     "no_odom", "low_rate", "unsupported_deadband",
                                     "narrow_limits", "configured_cap"])
def test_calibration_refuses_execution_contracts_that_cannot_support_a_safe_pulse(rig, problem):
    card = rig.card()
    if problem == "offline":
        card._running = False
    elif problem == "paused":
        card._paused = True
    elif problem == "goal":
        card._state.target = "person"
    elif problem == "dry_run":
        card._descriptor["dry_run"] = True
    elif problem == "no_descriptor":
        card._descriptor = {}
    elif problem == "no_odom":
        card._binding.pop("odom")
    elif problem == "low_rate":
        card._rate_hz = 5
    elif problem == "unsupported_deadband":
        card._descriptor["limits"]["min_magnitude"] = [0, 0, 0, 0, 0, 1.1]
    elif problem == "narrow_limits":
        card._descriptor["limits"]["upper"][5] = 0.1
    else:
        card._descriptor["limits"]["min_magnitude"] = [0, 0, 0, 0, 0, 1.0]
        card._config.wz_max = 0.9
    assert _begin(card)["state"] == "error"
    assert card._calibration is None and not card._publisher.commands


def test_calibration_uses_the_stationary_deadband_for_an_r1_style_chassis(rig):
    card = rig.card()
    card._descriptor["limits"].update(
        lower=[-1.0, -1.0, 0, 0, 0, -2.0],
        upper=[1.0, 1.0, 0, 0, 0, 2.0],
        min_magnitude=[0.4, 0.4, 0, 0, 0, 1.0],
        # Calibration is stationary and must not mistake this moving-gait
        # threshold for the speed that can start an in-place turn.
        min_magnitude_moving=[0.4, 0.4, 0, 0, 0, 0.05],
    )
    card._config.wz_max = 1.0

    started = _begin(card)

    assert started["state"] == "running", started
    assert card._calibration.speed == pytest.approx(1.0)
    assert started["calibration"]["commanded_wz_rad_s"] == pytest.approx(1.0)
    assert "±1rad/s" in started["message"]
    card._tick()
    assert card._publisher.commands[-1]["values"][5] == 0.0


def test_r1_style_chassis_completes_the_full_dynamic_speed_calibration(rig):
    card = rig.card()
    card._descriptor["limits"].update(
        lower=[-1.0, -1.0, 0, 0, 0, -2.0],
        upper=[1.0, 1.0, 0, 0, 0, 2.0],
        min_magnitude=[0.4, 0.4, 0, 0, 0, 1.0],
        min_magnitude_moving=[0.4, 0.4, 0, 0, 0, 0.05],
    )
    card._config.wz_max = 1.0

    info, commands = _complete(rig, card)

    report = info["calibration"]
    assert report["status"] == "completed", report
    assert report["commanded_wz_rad_s"] == pytest.approx(1.0)
    assert report["profile"]["response_gain"] == pytest.approx(1.0)
    assert {command["values"][5] for command in commands} == {-1.0, 0.0, 1.0}
    assert commands[-1]["values"] == [0.0] * 6


def test_calibration_without_a_declared_deadband_keeps_the_half_rad_per_second_pulse(rig):
    card = rig.card()
    started = _begin(card)
    assert started["state"] == "running", started
    assert card._calibration.speed == pytest.approx(0.5)
    assert started["calibration"]["commanded_wz_rad_s"] == pytest.approx(0.5)


@pytest.mark.parametrize("payload", [{"stamp_ms": None}, {"stamp_ms": 1},
                                     {"units": {"angular": "deg/s"}},
                                     {"twist": [0, 0, None, None, None, None]},
                                     {"twist": [0.1, 0, None, None, None, 0.0]}])
def test_calibration_requires_stamped_metric_yaw_and_a_stationary_chassis(rig, payload):
    card = rig.card()
    rig.clock.advance()
    rig.feed(card, **payload)
    assert _begin(card)["state"] == "error"
    assert card._calibration is None and not card._publisher.commands


@pytest.mark.parametrize("kind", ["stale", "duplicate", "command_gap"])
def test_feedback_loss_or_dispatch_gap_aborts_to_an_explicit_zero(rig, kind):
    card = rig.card()
    assert _begin(card)["state"] == "running"
    card._tick()
    if kind == "command_gap":
        rig.clock.advance(0.31)
        rig.feed(card)
    else:
        stamp = int(rig.clock.wall() * 1000)
        for _ in range(7):
            rig.clock.advance()
            if kind == "duplicate":
                rig.feed(card, stamp_ms=stamp)
            card._tick()
    card._tick()
    assert card._calibration is None
    assert card._info()["calibration"]["status"] == "failed"
    assert card._info()["calibration"]["phase"] == "failed"
    if kind == "stale":
        assert "角速度测量已过期" in card._info()["calibration"]["reason"]
    assert card._publisher.commands[-1]["values"] == [0.0] * 6
    assert card.events[-1][1] == "failed"
    assert card._state.target == ""


def test_calibration_failure_reports_a_receipt_gap_separately_from_source_age(rig):
    card = rig.card()
    assert _begin(card)["state"] == "running"
    card._tick()
    card._odom_feedback_received = rig.clock.monotonic() - 0.701
    card._tick()
    report = card._info()["calibration"]
    assert report["status"] == "failed" and report["phase"] == "failed"
    assert "角速度反馈断流" in report["reason"]
    assert "650ms" in report["reason"]
    assert card._publisher.commands[-1]["values"] == [0.0] * 6


def test_fit_diagnostics_reach_info_and_the_action_complete_result(rig):
    card = rig.card()
    started = _begin(card)
    diagnostics = {"commanded_wz_rad_s": 0.5,
                   "required_response_rad_s": 0.075,
                   "directions": {"left": {"samples": 4},
                                  "right": {"samples": 4}}}
    result = P.yaw_cal.CalibrationResult(
        [0.0] * 6, "failed", "left tail response unstable", "failed",
        10.0, 40, diagnostics=diagnostics)
    card._finish_calibration(result)
    report = card._info()["calibration"]
    assert report["diagnostics"] == diagnostics
    assert card.events[-1][0:2] == (started["action_id"], "failed")
    assert card.events[-1][2]["diagnostics"] == diagnostics


def test_translation_detected_during_calibration_aborts_to_zero(rig):
    card = rig.card()
    assert _begin(card)["state"] == "running"
    card._tick()
    rig.clock.advance()
    rig.feed(card, twist=[0.1, 0.0, None, None, None, 0.0])
    card._tick()
    assert card._calibration is None
    assert card._info()["calibration"]["status"] == "failed"
    assert "平移" in card._info()["calibration"]["reason"]
    assert card._publisher.commands[-1]["values"] == [0.0] * 6


@pytest.mark.parametrize("action", ["stop_navigation", "pause", "stop"])
def test_manual_stop_actions_cancel_calibration_with_zero_and_never_follow_automatically(rig, action):
    card = rig.card()
    assert _begin(card)["state"] == "running"
    publisher = card._publisher
    for _ in range(12):
        rig.clock.advance()
        rig.feed(card)
        card._tick()
    assert any(command["values"][5] > 0 for command in publisher.commands)
    card.dispatch("navi", {"action": action})
    assert publisher.commands[-1]["values"] == [0.0] * 6
    assert card._calibration is None and card._state.target == ""
    assert card._info()["calibration"]["status"] == "cancelled"
    assert card.events[-1][2]["status"] == "cancelled"
    count = len(publisher.commands)
    card._tick()
    assert len(publisher.commands) == count


def test_publisher_failure_cleans_up_calibration_even_if_the_final_zero_also_fails(rig):
    card = rig.card()
    assert _begin(card)["state"] == "running"
    card._publisher.fail = True
    card._tick()
    assert card._calibration is None and card._state.target == ""
    report = card._info()["calibration"]
    assert report["status"] == "cancelled"
    assert "simulated command transport failure" in report["reason"]
    assert card.events[-1][1] == "failed"


@pytest.mark.parametrize("action", ["stop_navigation", "pause", "stop"])
def test_stop_publish_errors_do_not_leave_the_calibration_session_or_goal_active(rig, action):
    card = rig.card()
    assert _begin(card)["state"] == "running"
    card._publisher.fail = True
    card.dispatch("navi", {"action": action})
    assert card._calibration is None and card._state.target == ""
    report = card._info()["calibration"]
    assert report["status"] == "cancelled"
    assert "零速发布失败" in report["reason"]
    assert card.events[-1][1] == "failed"


def test_other_actions_cannot_interleave_following_or_reconfigure_an_active_calibration(rig):
    card = rig.card()
    assert _begin(card)["state"] == "running"
    session = card._calibration
    for args in ({"action": "navigate_to", "target": "person"},
                 {"action": "config", "vx_max": 0.1},
                 {"action": "start"}, {"action": "resume"},
                 {"action": "calibrate_yaw"}):
        result = card.dispatch("navi", args)
        assert result["state"] == "error", result
        assert card._calibration is session and card._state.target == ""
    assert not card._publisher.commands


def test_failed_save_keeps_the_previous_profile_and_reports_failure(rig, monkeypatch):
    card = rig.card()
    old_profile = _complete(rig, card)[0]["calibration"]["profile"]
    old_profile = dict(old_profile, k_yaw=0.3)
    P.yaw_cal.save_profile(rig.directory, card._calibration_key(), old_profile)
    card._apply_yaw_profile(old_profile)
    previous_file = P.yaw_cal.profile_path(rig.directory, card._calibration_key()).read_bytes()

    def fail_save(*args):
        raise OSError("simulated read-only calibration volume")

    monkeypatch.setattr(P.yaw_cal, "save_profile", fail_save)
    rig.clock.advance()
    rig.feed(card)
    info, commands = _complete(rig, card)
    assert info["calibration"]["status"] == "failed"
    assert "未保存" in info["calibration"]["reason"]
    assert card._yaw_profile == old_profile
    assert info["gains"]["yaw"]["base"] == 0.3
    assert P.yaw_cal.profile_path(rig.directory, card._calibration_key()).read_bytes() == previous_file
    assert commands[-1]["values"] == [0.0] * 6


@pytest.mark.parametrize("changed", ["odom", "limits", "camera"])
def test_a_different_execution_connection_cannot_inherit_the_saved_profile(rig, changed):
    first = rig.card()
    _complete(rig, first)
    second = rig.card()
    if changed == "odom":
        second._binding["odom"] = "/other/state/odom"
    elif changed == "limits":
        second._descriptor["limits"]["upper"][5] = 0.8
    else:
        second._config.half_fov_rad = 0.5
    second._load_yaw_profile()
    assert second._yaw_profile is None
    assert second._info()["calibration"]["status"] == "not_calibrated"
    assert second._config.calibrated_yaw_gain is None


@pytest.mark.parametrize("corruption", ["identity", "nan"])
def test_corrupt_stored_profiles_are_reported_and_never_applied(rig, corruption):
    card = rig.card()
    profile = _complete(rig, card)[0]["calibration"]["profile"]
    key = card._calibration_key()
    damaged = dict(profile, **({"robot_key": "other"} if corruption == "identity"
                               else {"k_yaw": float("nan")}))
    P.yaw_cal.profile_path(rig.directory, key).write_text(json.dumps(damaged), encoding="utf-8")
    fresh = rig.card()
    fresh._load_yaw_profile()
    assert fresh._yaw_profile is None and fresh._config.calibrated_yaw_gain is None
    assert fresh._info()["calibration"]["status"] == "not_calibrated"
    assert fresh._info()["calibration"]["reason"]


def test_manual_gain_controls_stay_hidden_while_info_reports_both_readonly_gains(rig):
    card = rig.card()
    tool = card.get_tools()[0]
    hidden = {"k_yaw", "k_fwd", "adaptive_k_yaw", "adaptive_k_fwd", "yaw_damping"}
    assert not hidden.intersection(tool["inputSchema"]["properties"])
    assert not hidden.intersection(tool["configSchema"]["properties"])
    assert "calibrate_yaw" in tool["inputSchema"]["properties"]["action"]["enum"]
    before = copy.deepcopy(card._config)
    info = card.dispatch("navi", {"action": "info"})
    assert card._config == before and not card._publisher.commands
    assert info["gains"]["forward"]["base"] == 0.9
    assert "effective" in info["gains"]["yaw"]
    assert "damping_effective" not in info["gains"]["yaw"]


def test_publisher_open_during_startup_does_not_allow_calibration_or_a_second_start(rig, monkeypatch):
    card = rig.card()
    card._stop()
    opened = _Publisher()
    monkeypatch.setattr(card, "_open_publisher", lambda: setattr(card, "_publisher", opened))
    binding = dict(card._binding, odom="/new/state/odom")
    attempted = []

    def bind_while_starting(*args):
        assert card._starting and card._running and card._publisher is opened
        config = copy.deepcopy(card._config)
        descriptor = copy.deepcopy(card._descriptor)
        attempted.append(_begin(card))
        second = _descriptor()
        second["limits"]["upper"][5] = 0.25
        attempted.append(card.dispatch("navi", {"action": "start", "control_interface": second}))
        attempted.append(card.dispatch("navi", {"action": "config", "vx_max": 0.1}))
        attempted.append(card.dispatch("navi", {"action": "navigate_to", "target": "person"}))
        assert card._config == config and card._descriptor == descriptor
        assert card._calibration is None and not opened.commands
        return binding, ""

    monkeypatch.setattr(card, "_bind_inputs", bind_while_starting)
    result = card.dispatch("navi", {"action": "start", "control_interface": _descriptor()})
    assert result["state"] == "running"
    assert all(item["state"] == "error" for item in attempted)
    assert "already running" in attempted[1]["message"]
    assert card._binding == binding and not card._starting
    assert card._calibration is None and not opened.commands


def test_stop_during_input_binding_cancels_the_outer_start_and_cleans_open_publisher(rig, monkeypatch):
    card = rig.card()
    card._stop()
    opened = _Publisher()
    monkeypatch.setattr(card, "_open_publisher", lambda: setattr(card, "_publisher", opened))

    def stop_while_binding(*args):
        assert card._starting and card._publisher is opened
        card.dispatch("navi", {"action": "stop"})
        return dict(card._binding), ""

    monkeypatch.setattr(card, "_bind_inputs", stop_while_binding)
    result = card.dispatch("navi", {"action": "start", "control_interface": _descriptor()})
    assert result["state"] == "idle" and "取消" in result["message"]
    assert not card._running and not card._starting
    assert card._publisher is None and card._calibration is None
    card._tick()
    assert not opened.commands


def test_input_binding_failure_keeps_its_original_error_after_startup_cleanup(rig, monkeypatch):
    card = rig.card()
    card._stop()
    monkeypatch.setattr(card, "_open_publisher", lambda: setattr(card, "_publisher", _Publisher()))
    monkeypatch.setattr(card, "_bind_inputs", lambda *args: ({}, "specific depth binding failure"))
    result = card.dispatch("navi", {"action": "start", "control_interface": _descriptor()})
    assert result["state"] == "error" and result["message"] == "specific depth binding failure"
    assert not card._starting and not card._running
    assert card._publisher is None and card._calibration is None


def test_calibration_refuses_a_descriptor_that_does_not_declare_radians_per_second(rig):
    card = rig.card()
    card._descriptor["units"]["angular"] = "deg/s"
    assert _begin(card)["state"] == "error"
    assert card._calibration is None and not card._publisher.commands
