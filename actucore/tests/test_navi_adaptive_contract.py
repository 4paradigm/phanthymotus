"""Exercise adaptive forward P through the card's emitted velocity contract."""

import time

import pytest

from test_navi_card import _card
from test_navi_policy import _detections, _obj, _solid_depth


def feed(card, distance):
    now = time.time()
    card._objects = _detections(_obj(bbox=(0.4, 0.4, 0.6, 0.6)))
    card._depth_map = _solid_depth(distance)
    card._depth_bands = {name: distance for name in ("left", "center", "right")}
    card._objects_ms = card._depth_ms = int(now * 1000)
    card._last_tick = now - 0.1
    return card.next_command()


def test_enabled_adaptive_gain_is_reflected_in_emitted_commands_and_actual_readback():
    card = _card(adaptive_k_fwd=True, k_fwd=0.2, vx_max=0.3,
                 use_lateral=False, stop_distance_m=0.8, stop_tolerance_m=0.05)
    card._running = True
    card._state.target = "chair"
    for _ in range(18):
        command = feed(card, 1.0)
        if command is not None:
            assert command["schema"] == "motus.control/1"
            assert 0 <= command["values"][0] <= 0.3
            assert command["values"][1] == 0
    info = card.dispatch("navi", {"action": "info"})
    gain = info["last"]["k_fwd_effective"]
    assert info["config"]["adaptive_k_fwd"] is True
    assert info["config"]["k_fwd"] == 0.2
    assert 0.2 < gain <= 0.6
    assert gain == pytest.approx(card._last_decision.k_fwd_effective)
    assert command is not None and command["values"][0] > 0


def test_adaptive_mode_does_not_hide_the_reported_depth_bias_or_override_obstacle_stop():
    card = _card(adaptive_k_fwd=True, k_fwd=0.4, vx_max=0.3,
                 use_lateral=False, stop_distance_m=0.4, stop_tolerance_m=0.05,
                 obstacle_stop_m=0.6)
    card._running = True
    card._state.target = "chair"
    for _ in range(18):
        command = feed(card, 0.6)
        if command is not None:
            assert command["values"][:2] == [0.0, 0.0]
    info = card.dispatch("navi", {"action": "info"})
    assert info["config"]["obstacle_stop_m"] == 0.6
    assert info["config"]["arrival_distance_m"] == pytest.approx(0.45)
    assert info["last"]["distance_m"] == pytest.approx(0.6, abs=0.001)
    assert info["last"]["status"] != "arrived"
    assert any("0.6" in note and "0.45" in note for note in info["degraded"])
