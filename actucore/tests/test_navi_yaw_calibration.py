"""Yaw calibration is bounded independently of ROS, SDK and physical hardware."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navi import yaw_calibration as C  # noqa: E402


def _run(*, left=1.0, right=1.0, delay=0.0, tau=0.0, override=None, dt=0.1):
    """A command/measurement loop with a simple delayed first-order plant."""
    session = C.YawCalibration(now=0.0, robot_key="robot-one")
    issued, results = [], []
    measured = 0.0
    for tick in range(100):
        now = round(tick * dt, 9)
        history = [command for at, command in issued if at <= now - delay + 1e-9]
        wanted = history[-1] if history else 0.0
        wanted *= left if wanted >= 0 else right
        measured += (wanted - measured) * (1.0 if tau == 0 else min(1.0, dt / tau))
        actual = override(now, measured) if override else measured
        result = session.step(now, actual, sample_key=tick, sample_stamp=now)
        results.append(result)
        issued.append((now, result.values[5]))
        if result.status != "running":
            return session, result, results
    pytest.fail("calibration exceeded the bounded test duration")


def _run_independent_feedback(sample_period):
    """Commands tick at 10 Hz; odometry has its own independent sample clock."""
    session = C.YawCalibration(now=0.0, robot_key="robot-one")
    issued = []
    next_sample = 0.0
    latest_stamp, latest_wz = None, 0.0
    for tick in range(80):
        now = round(tick * 0.1, 9)
        while next_sample <= now + 1e-9:
            latest_stamp = round(next_sample, 9)
            earlier_commands = [command for at, command in issued
                                if at <= latest_stamp + 1e-9]
            latest_wz = earlier_commands[-1] if earlier_commands else 0.0
            next_sample += sample_period
        result = session.step(now, latest_wz, latest_stamp, latest_stamp)
        issued.append((now, result.values[5]))
        if result.status != "running":
            return result, issued
    pytest.fail("calibration exceeded its bounded independent-feedback simulation")


def test_nominal_two_pulse_response_completes_with_zero_and_a_bounded_profile():
    session, result, history = _run()
    assert result.status == "completed", result.reason
    assert result.values == [0.0] * 6
    assert result.elapsed_s == pytest.approx(5.2)
    assert result.samples == 53
    profile = result.profile
    assert C.validate_profile(profile, "robot-one") == profile
    assert profile["response_gain"] == pytest.approx(1.0)
    assert 0.2 <= profile["k_yaw"] <= 1.2
    assert 0.1 <= profile["yaw_damping"] <= 0.5
    assert {sample.phase for sample in history} >= {
        "settle", "left", "right", "rest_right", "completed"}
    assert all(sample.values[:5] == [0.0] * 5 for sample in history)
    assert max(abs(sample.values[5]) for sample in history) == 0.2
    turns = [sample.values[5] for sample in history]
    first_positive = next(index for index, value in enumerate(turns) if value > 0)
    first_negative = next(index for index, value in enumerate(turns) if value < 0)
    assert turns[first_negative - 1] == 0.2
    assert 0.0 not in turns[first_positive:first_negative]
    assert session.step(100.0, 1.0, "ignored") == result


def test_faster_or_delayed_response_chooses_a_more_conservative_gain():
    nominal = _run()[1].profile
    fast = _run(left=1.1, right=1.1)[1].profile
    delayed = _run(delay=0.2)[1].profile
    assert fast["response_gain"] > nominal["response_gain"]
    assert fast["k_yaw"] < nominal["k_yaw"]
    assert delayed["delay_s"] > nominal["delay_s"]
    assert delayed["k_yaw"] < nominal["k_yaw"]
    assert delayed["yaw_damping"] > nominal["yaw_damping"]


def test_slow_but_measurable_response_is_not_tuned_above_the_nominal_gain():
    _, result, _ = _run(left=0.4, right=0.5, tau=0.15)
    assert result.status == "completed", result.reason
    assert 0.15 <= result.profile["response_gain"] < 1.0
    assert result.profile["k_yaw"] <= 1.2


@pytest.mark.parametrize("gain", [0.0, 0.05])
def test_no_reliable_response_never_produces_a_profile(gain):
    _, result, _ = _run(left=gain, right=gain)
    assert result.status == "failed"
    assert "response" in result.reason
    assert result.profile is None and result.values == [0.0] * 6


@pytest.mark.parametrize("dt", [0.1, 0.2, 0.25])
def test_four_five_and_ten_hz_samples_complete_with_the_longer_bounded_pulses(dt):
    _, result, _ = _run(dt=dt)
    assert result.status == "completed", result.reason
    assert result.values == [0.0] * 6
    assert 5.2 <= result.elapsed_s <= 5.2 + dt
    assert result.profile["response_gain"] == pytest.approx(1.0)


@pytest.mark.parametrize("sample_period", [0.1, 0.2, 0.25])
def test_slow_feedback_does_not_become_a_command_gap_or_duplicate_fit_evidence(sample_period):
    result, commands = _run_independent_feedback(sample_period)
    assert result.status == "completed", result.reason
    assert result.elapsed_s == pytest.approx(5.2)
    assert result.samples <= 1 + int(5.2 / sample_period)
    assert len(commands) == 53
    assert commands[-1][1] == 0.0
    assert result.profile["response_gain"] == pytest.approx(1.0)


def test_three_hz_feedback_is_too_sparse_to_claim_a_stable_calibration():
    result, commands = _run_independent_feedback(1 / 3)
    assert result.status == "failed" and "insufficient distinct stable" in result.reason
    assert result.profile is None and commands[-1][1] == 0.0


def test_a_fresh_sample_after_a_long_observation_gap_fails_even_with_continuous_commands():
    session = C.YawCalibration(now=0.0, robot_key="r")
    for tick in range(7):
        result = session.step(tick / 10, 0.0, "first", 0.0)
        assert result.status == "running" and result.values == [0.0] * 6
    result = session.step(0.66, 0.0, "second", 0.66)
    assert result.status == "failed" and "650 ms" in result.reason
    assert result.values == [0.0] * 6


def test_extreme_left_right_asymmetry_is_rejected():
    _, result, _ = _run(left=1.0, right=0.2)
    assert result.status == "failed" and "asymmetric" in result.reason
    assert result.profile is None


def test_wrong_direction_aborts_before_the_full_pulse():
    _, result, _ = _run(left=-1.0)
    assert result.status == "failed" and "direction" in result.reason
    assert result.elapsed_s < 1.3 and result.values == [0.0] * 6


def test_observed_rate_limit_aborts_immediately():
    _, result, _ = _run(override=lambda now, measured: 0.51 if now >= 0.6 else measured)
    assert result.status == "failed" and "0.5 rad/s" in result.reason
    assert result.values == [0.0] * 6


def test_observed_angle_limit_includes_motion_even_before_rest():
    _, result, _ = _run(override=lambda now, measured: 0.49 if 0.9 <= now < 3.0 else measured)
    assert result.status == "failed" and "angle limit" in result.reason
    assert result.elapsed_s < 3.0 and result.values == [0.0] * 6


def test_reversal_that_never_changes_measured_direction_is_rejected():
    _, result, history = _run(override=lambda now, measured:
                             0.05 if 2.7 <= now <= 3.0 else measured)
    assert result.status == "failed" and "direction" in result.reason
    assert any(sample.values[5] < 0 for sample in history)
    assert result.values == [0.0] * 6


def test_completion_requires_two_current_stationary_samples():
    _, result, _ = _run(override=lambda now, measured:
                       -0.04 if now >= 4.9 else measured)
    assert result.status == "failed" and "at the end" in result.reason
    assert result.profile is None and result.values == [0.0] * 6


def test_initial_motion_is_rejected_before_any_nonzero_command():
    _, result, history = _run(override=lambda now, measured: 0.04)
    assert result.status == "failed" and "already turning" in result.reason
    assert len(history) == 1 and result.values == [0.0] * 6


def test_unstable_response_cannot_be_saved_as_a_good_calibration():
    _, result, _ = _run(override=lambda now, measured:
                       measured * (0.3 if round(now * 10) % 2 else 1.5)
                       if 1.4 <= now < 2.1 else measured)
    assert result.status == "failed", result
    assert result.profile is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, True])
def test_invalid_odometry_aborts_to_zero(value):
    result = C.YawCalibration(now=0.0, robot_key="r").step(0.0, value, "frame")
    assert result.status == "failed" and "measured_wz" in result.reason
    assert result.values == [0.0] * 6


def test_duplicate_samples_never_count_as_new_observations_and_eventually_fail():
    session = C.YawCalibration(now=0.0, robot_key="r")
    first = session.step(0.0, 0.0, "same", 0.0)
    second = session.step(0.1, 0.0, "same", 0.1)
    third = session.step(0.2, 0.0, "same", 0.2)
    assert first.samples == second.samples == third.samples == 1
    session.step(0.4, 0.0, "same", 0.4)
    session.step(0.6, 0.0, "same", 0.6)
    result = session.step(0.66, 0.0, "same", 0.66)
    assert result.status == "failed" and "fresh yaw observation" in result.reason
    assert result.values == [0.0] * 6


def test_a_fresh_cached_prestart_sample_waits_at_zero_until_new_odometry_arrives():
    session = C.YawCalibration(now=1.0, robot_key="r")
    first = session.step(1.0, 0.0, "cached", 0.92)
    assert first.status == "running" and first.phase == "settle"
    assert first.values == [0.0] * 6 and first.samples == 0
    second = session.step(1.1, 0.0, "fresh", 1.02)
    assert second.status == "running" and second.samples == 1
    assert second.values == [0.0] * 6


def test_waiting_for_poststart_odometry_still_obeys_staleness_and_tick_guards():
    session = C.YawCalibration(now=1.0, robot_key="r")
    session.step(1.0, 0.0, "cached", 0.92)
    session.step(1.2, 0.0, "cached", 0.92)
    session.step(1.4, 0.0, "cached", 0.92)
    stale = session.step(1.58, 0.0, "cached", 0.92)
    assert stale.status == "failed" and "stale" in stale.reason
    session = C.YawCalibration(now=1.0, robot_key="r")
    session.step(1.0, 0.0, "cached", 0.92)
    gap = session.step(1.31, 0.0, "fresh", 1.31)
    assert gap.status == "failed" and "command gap" in gap.reason


@pytest.mark.parametrize("stamp", [-0.66, 0.1])
def test_stale_or_future_timestamp_is_rejected(stamp):
    result = C.YawCalibration(now=0.0, robot_key="r").step(0.0, 0.0, "frame", stamp)
    assert result.status == "failed" and "stale or future" in result.reason


def test_distinct_keys_with_a_nonadvancing_sample_clock_are_rejected():
    session = C.YawCalibration(now=0.0, robot_key="r")
    session.step(0.0, 0.0, 1, 0.0)
    result = session.step(0.1, 0.0, 2, 0.0)
    assert result.status == "failed" and "timestamp did not advance" in result.reason


def test_command_gaps_abort_without_replaying_missed_motion_phases():
    session = C.YawCalibration(now=0.0, robot_key="r")
    result = session.step(0.31, 0.0, 1, 0.31)
    assert result.status == "failed" and "command gap" in result.reason
    assert result.values == [0.0] * 6
    session = C.YawCalibration(now=0.0, robot_key="r")
    session.step(0.0, 0.0, 1)
    result = session.step(1.0, 0.0, 2)
    assert result.status == "failed" and result.phase == "failed"


def test_backwards_controller_clock_cannot_extend_calibration():
    session = C.YawCalibration(now=1.0, robot_key="r")
    result = session.step(0.9, 0.0, 1)
    assert result.status == "failed" and "backwards" in result.reason


def test_cancellation_is_terminal_and_zero():
    session = C.YawCalibration(now=0.0, robot_key="r")
    session.step(0.0, 0.0, 1)
    cancelled = session.cancel("operator requested stop")
    assert cancelled.status == "cancelled" and cancelled.values == [0.0] * 6
    assert cancelled.reason == "operator requested stop"
    assert session.step(0.1, 0.0, 2) == cancelled


def test_profile_round_trip_is_per_robot_and_path_components_are_hashed(tmp_path):
    profile = _run()[1].profile
    assert C.load_profile(tmp_path, "robot-one") is None
    path = C.save_profile(tmp_path, "robot-one", profile)
    assert C.load_profile(tmp_path, "robot-one") == profile
    assert C.load_profile(tmp_path, "other-robot") is None
    assert path.parent == tmp_path and path.name.startswith("yaw-")
    assert C.profile_path(tmp_path, "../../secret").parent == tmp_path
    assert not list(tmp_path.glob("*.tmp"))
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("field,value", [
    ("k_yaw", 1.21), ("k_yaw", 0.19), ("yaw_damping", 0.51),
    ("response_gain", float("nan")), ("delay_s", -0.01),
    ("settling_s", 0.91), ("left_gain", True), ("created_at_ms", 0),
    ("schema", "other/1"), ("robot_key", "other-robot")])
def test_invalid_profiles_are_rejected_before_writing(tmp_path, field, value):
    profile = dict(_run()[1].profile, **{field: value})
    with pytest.raises(ValueError):
        C.save_profile(tmp_path, "robot-one", profile)
    assert not list(tmp_path.iterdir())


def test_invalid_json_and_wrong_identity_remain_reportable_load_errors(tmp_path):
    path = C.profile_path(tmp_path, "robot-one")
    path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError):
        C.load_profile(tmp_path, "robot-one")
    profile = _run()[1].profile
    path.write_text(json.dumps(dict(profile, robot_key="wrong")), encoding="utf-8")
    with pytest.raises(ValueError, match="different robot"):
        C.load_profile(tmp_path, "robot-one")


def test_a_failed_atomic_replace_preserves_previous_profile_and_cleans_temporary_file(tmp_path, monkeypatch):
    profile = _run()[1].profile
    path = C.save_profile(tmp_path, "robot-one", profile)
    previous = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(C.os, "replace", fail_replace)
    with pytest.raises(OSError, match="persistence failure"):
        C.save_profile(tmp_path, "robot-one", dict(profile, k_yaw=0.3))
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_directory_fsync_failure_after_commit_reports_success_with_a_durability_warning(tmp_path, monkeypatch, caplog):
    profile = _run()[1].profile
    C.save_profile(tmp_path, "robot-one", profile)
    original_fsync = C.os.fsync
    calls = []

    def fail_directory_fsync(descriptor):
        calls.append(descriptor)
        if len(calls) == 2:
            raise OSError("simulated directory durability failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(C.os, "fsync", fail_directory_fsync)
    replacement = dict(profile, k_yaw=0.3)
    path = C.save_profile(tmp_path, "robot-one", replacement)
    assert path.exists()
    assert C.load_profile(tmp_path, "robot-one") == replacement
    assert "committed" in caplog.text and "directory fsync failed" in caplog.text
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("speed", [0.0, -0.2, 0.21, float("nan"), True])
def test_constructor_rejects_unsafe_pulse_speeds(speed):
    with pytest.raises(ValueError):
        C.YawCalibration(now=0.0, robot_key="r", speed=speed)
