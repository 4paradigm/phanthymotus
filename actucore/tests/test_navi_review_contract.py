"""Runtime regressions for the driver / navi contract review."""

import math
import json
import time
from types import SimpleNamespace

import pytest

from test_navi_card import _card, _descriptor, _FakeNode
from plugins.navi import policy as P


def start_card(card, **descriptor_fields):
    card._open_publisher = lambda: setattr(card, "_node", _FakeNode({
        "/camera/depth": ["sensor_msgs/msg/CompressedImage"],
    }))
    result = card._start({
        "control_interface": _descriptor(**descriptor_fields),
        "input_topics": ["/cam/objects", "/camera/depth"],
    })
    assert result["state"] == "running", result


@pytest.mark.parametrize("rate,expected", [(20, 1000), (2, 1500)])
def test_odom_optional_delay_preserves_producer_stamp_and_rate_budget(monkeypatch, rate, expected):
    card = _card(max_obs_age_ms=1000)
    card._adopt_odom_interface({
        "schema": "motus.odom/1", "rate_hz": rate,
        "timing": {"stamp_source": "received"},
    }, bound_topic="/robot/state/odom")
    assert card._odom_interface["timing"]["stamp_source"] == "received"
    assert card._odom_max_age_ms == expected
    assert any("未声明测量延迟" in note for note in card._odom_notes)
    assert not any("无效" in note for note in card._odom_notes)
    monkeypatch.setattr(time, "time", lambda: 123.456)
    card._on_string("odom", SimpleNamespace(data=json.dumps({
        "schema": "motus.odom/1", "stamp_ms": 123000,
        "twist": [0.1, 0, None, None, None, 0.2],
    })))
    assert card._odom_ms == 123000


@pytest.mark.parametrize("delay", [None, True, -1, float("nan"), "late"])
def test_present_invalid_odom_delay_does_not_become_an_unknown_delay(delay):
    card = _card()
    card._adopt_odom_interface({
        "schema": "motus.odom/1", "rate_hz": 20,
        "timing": {"stamp_source": "received", "max_measurement_delay_ms": delay},
    }, bound_topic="/robot/state/odom")
    assert card._odom_interface == {}
    assert card._odom_max_age_ms == 500


@pytest.mark.parametrize("rate", [True, False, "2", "20", None, [], {}, 0, -1,
                                 float("nan"), float("inf")])
def test_invalid_odom_rate_keeps_receipt_time_fallback(monkeypatch, rate):
    card = _card()
    card._adopt_odom_interface({
        "schema": "motus.odom/1", "rate_hz": rate,
        "timing": {"stamp_source": "received"},
    }, bound_topic="/robot/state/odom")
    assert card._odom_interface == {}
    assert card._odom_max_age_ms == 500
    monkeypatch.setattr(time, "time", lambda: 123.456)
    card._on_string("odom", SimpleNamespace(data=json.dumps({
        "schema": "motus.odom/1", "stamp_ms": 123000,
        "twist": [0.1, 0, None, None, None, 0.2],
    })))
    assert card._odom_ms == 123456


@pytest.mark.parametrize("interface,notice", [
    ({"schema": "motus.odom/1", "rate_hz": 20,
      "timing": {"stamp_source": "received"}}, "未声明测量延迟"),
    ({"schema": "wrong"}, "声明无效"),
])
@pytest.mark.parametrize("response", [None, {"wz": "invalid"}])
def test_runtime_config_preserves_odom_and_response_notices(interface, notice, response):
    card = _card()
    try:
        start_card(card, response=response)
        card._adopt_odom_interface(interface, bound_topic="/robot/state/odom")
        before = card._info()["degraded"]
        assert any(notice in note for note in before)
        for distance in (1.0, 1.2):
            assert card._config_action({"stop_distance_m": distance})["status"] == "configured"
            after = card._info()["degraded"]
            assert any(notice in note for note in after)
            if response:
                assert any("response.wz 声明无效" in note for note in after)
            assert len(after) == len(set(after))
    finally:
        card._stop()


@pytest.mark.parametrize("configured,declared,age,fresh", [
    (1000, 600, 600, True), (1000, 600, 601, False),
    (400, 600, 401, False), (500, None, 501, False),
])
def test_visual_age_obeys_both_config_and_driver(monkeypatch, configured, declared, age, fresh):
    card = _card(max_obs_age_ms=configured)
    rate = {"max_hz": 20, "expected_hz": 10, "watchdog_ms": 500}
    if declared is not None:
        rate["max_obs_age_ms"] = declared
    try:
        start_card(card, rate=rate)
        monkeypatch.setattr(time, "time", lambda: 123.456)
        card._objects = {"objects": []}
        card._depth_bands = {"center": 2.0}
        card._objects_ms = card._depth_ms = 123456 - age
        objects, depth, _ = card.observation()
        assert (objects is not None) is fresh
        assert (depth is not None) is fresh
    finally:
        card._stop()


def test_observation_budget_recovers_after_switching_driver():
    card = _card(max_obs_age_ms=1000)
    try:
        start_card(card, rate={"max_hz": 20, "max_obs_age_ms": 600})
        assert card._config.max_obs_age_ms == 600
        card._stop()
        start_card(card)
        assert card._config.max_obs_age_ms == 1000
    finally:
        card._stop()


@pytest.mark.parametrize("floor,cap,requested,expected", [
    (1.0, 2.0, 0.6, 1.0), (1.4, 2.0, 0.6, 1.4),
    (1.4, 1.0, 0.6, 0.0), (1.0, 2.0, 0.0, 0.0),
])
def test_search_rate_is_executable_and_sweep_uses_emitted_speed(floor, cap, requested, expected):
    config = P.Config(search_rate=requested)
    limits = {"lower": [-1, -1, 0, 0, 0, -cap],
              "upper": [1, 1, 0, 0, 0, cap],
              "min_magnitude": [0, 0, 0, 0, 0, floor]}
    P.adopt_limits(config, _descriptor(limits=limits))
    state = P.State(target="person", missing_frames=10)
    decision = P._search(config, state, 0.1)
    if expected == 0:
        assert decision.status == P.FAILED and decision.values is None
    else:
        output = P.apply_deadband(decision.values, limits["min_magnitude"])
        assert abs(output[5]) == expected
        assert state.searched_rad == pytest.approx(expected * 0.1)


@pytest.mark.parametrize("requested,pinned,expected", [
    (False, False, False), (True, False, True),
    (False, True, False), (True, True, False),
])
def test_lateral_hardware_capability_cannot_enable_disabled_motion(requested, pinned, expected):
    config = P.Config(use_lateral=requested)
    cap = 0 if pinned else 1
    P.adopt_limits(config, _descriptor(limits={
        "lower": [-1, -cap, 0, 0, 0, -2],
        "upper": [1, cap, 0, 0, 0, 2],
    }))
    assert config.use_lateral is expected


def test_runtime_bounds_remain_configurable_and_driver_clamped():
    card = _card()
    try:
        start_card(card, limits={
            "lower": [-0.5, -0.5, 0, 0, 0, -1],
            "upper": [0.5, 0.5, 0, 0, 0, 1],
        })
        result = card._config_action({"vx_max": 0.2, "vy_max": 0.3,
                                     "wz_max": 2.0, "search_rate": 0.4,
                                     "slow_distance_m": 2.0,
                                     "align_full_stop": 0.7, "use_lateral": False})
        assert result["effective_config"] == {
            "vx_max": 0.2, "vy_max": 0.3, "wz_max": 1.0,
            "search_rate": 0.4, "slow_distance_m": 2.0,
            "align_full_stop": 0.7, "use_lateral": False}
        card._config_action({"stop_distance_m": 1.0})
        assert card._config.vx_max == 0.2
        assert card._config.use_lateral is False
    finally:
        card._stop()


@pytest.mark.parametrize("args", [
    {"wz_max": True}, {"search_rate": -0.1}, {"vx_max": float("nan")},
    {"use_lateral": "false"},
])
def test_invalid_runtime_bounds_leave_configuration_unchanged(args):
    card = _card()
    config = card._config
    result = card._config_action(args)
    assert result["state"] == "error"
    assert card._config is config


def test_gap_turn_uses_driver_response_without_target_gain_schedule(monkeypatch):
    config = P.Config(k_yaw=1.2, response_yaw_gain=0.4, adaptive_k_yaw=True,
                      use_lateral=False, floor_wz=0)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs:
                        (5.0, 1.0) if kwargs.get("lateral_offset_m") else (0.4, 1.0))
    result = P._avoid(0.2, 0, 0, {}, config, "approaching", "target")
    assert abs(result[2]) == pytest.approx(0.4 * math.atan2(P._keepout_m(config), 0.4))
