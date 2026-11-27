"""The public card lifecycle enables native DWA for new and saved instances."""
import json
from types import SimpleNamespace

import pytest

from plugins.navi import plugin, policy
from test_navi_card import _DEPTH_TOPIC, _OBJECTS_TOPIC, _FakeNode, _decl, _descriptor, _first_command
from test_navi_dwa_integration import observation_case, observe_selector


@pytest.mark.parametrize("saved_controller", [None, "legacy", "dwa"],
                         ids=["new-card", "saved-legacy", "saved-dwa"])
@pytest.mark.parametrize("body", [True, False], ids=["body", "front-camera"])
def test_default_and_saved_cards_publish_native_dwa_after_start_and_restart(
        monkeypatch, saved_controller, body):
    saved = dict(use_lateral=False, vx_max=.30, max_depth_age_ms=800, view_hz=0.)
    if saved_controller is not None:
        saved["local_controller"] = saved_controller
    card = plugin.NaviPlugin(json.loads(json.dumps(saved)), executor=None)
    packets = []

    def open_publisher():
        card._node = _FakeNode({_DEPTH_TOPIC: ["sensor_msgs/msg/CompressedImage"]})
        card._publisher = SimpleNamespace(publish=lambda payload: packets.append(json.loads(payload.data)))

    monkeypatch.setattr(card, "_open_publisher", open_publisher)
    monkeypatch.setattr(plugin.time, "time", lambda: 1.)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: 1.)
    declaration = _decl(_DEPTH_TOPIC, distortion_model="pinhole",
                        K=[420., 0., 320., 0., 420., 240., 0., 0., 1.])
    if body:
        declaration["vendor"] = {"body_from_optical": [
            [0., 0., 1., 0.], [-1., 0., 0., 0.],
            [0., -1., 0., 0.], [0., 0., 0., 1.]]}
    args = dict(action="start", input_topics=[_OBJECTS_TOPIC, _DEPTH_TOPIC, "/robot/state/odom"],
                control_interface=_descriptor(
                    footprint={"half_width": .20},
                    rate={"max_hz": 20, "expected_hz": 10, "watchdog_ms": 500,
                          "max_obs_age_ms": 800}),
                camera_info={_DEPTH_TOPIC: declaration})
    calls = observe_selector(monkeypatch)
    try:
        for _ in range(2):
            assert card.dispatch("navi", args)["state"] == "running"
            assert card._config.local_planning
            assert (card._config.scan_geometry is not None) == body
            assert card.dispatch("navi", {"action": "navigate_to", "target": "person"})["state"] == "running"
            # Reuse observations, never the fixture's explicitly chosen Config:
            # the real start path must supply this card's DWA configuration.
            _, state, depth, odom = observation_case(body=body)
            card._state = state
            monkeypatch.setattr(card, "_observation_snapshot", lambda **_: (None, depth, odom, 1000))
            calls.clear()
            packets.clear()
            message = card.next_command()
            accepted = [choice for _, choice in calls if choice is not None]
            assert accepted, "Every card must reach native DWA without an enable setting"
            assert state.dwa_selected, (card._last_decision, state.dwa_trace)
            twist = tuple(message["values"][i] for i in (0, 1, 5))
            assert twist == accepted[-1].command and twist[0] > 0.
            card._publish_command(message)
            assert len(packets) == 1
            assert tuple(packets[0]["values"][i] for i in (0, 1, 5)) == twist
            assert card._info()["config"]["local_controller"] == "dwa"
            card._stop()
    finally:
        card._stop()


@pytest.mark.parametrize("planning_ms", [20, 200, 201], ids=["timely", "at-expiry", "past-expiry"])
def test_unwired_card_accounts_for_motion_stopped_after_planning(monkeypatch, planning_ms):
    card = plugin.NaviPlugin({"idle_timeout_s": .45}, executor=None)
    clock = [123_456]
    monkeypatch.setattr(plugin.time, "time", lambda: clock[0] / 1000)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: clock[0] / 1000)
    assert _first_command(card) is not None
    card._config.local_planning = True
    card._config.max_obs_age_ms = 800
    card._descriptor = _descriptor(rate={"max_obs_age_ms": 500})
    packets = []
    card._publisher = SimpleNamespace(publish=lambda payload: packets.append(json.loads(payload.data)))
    step = policy.step
    planned = []

    def slow_step(**kwargs):
        decision = step(**kwargs)
        planned.append(decision.values.copy() if decision.values is not None else None)
        clock[0] += planning_ms
        return decision

    monkeypatch.setattr(policy, "step", slow_step)
    records = []
    for _ in range(8):
        clock[0] += 100
        card._objects_ms = clock[0] - 56
        card._depth_ms = clock[0] - 300
        message = card.next_command()
        assert message is not None
        card._publish_command(message)
        records.append((card._last_decision.status, card._state.idle_for_s))
        if card._last_decision.status == policy.FAILED:
            break

    assert card._config.local_controller == "dwa"
    assert not card._state.dwa_progress.requires_feedback
    assert len(planned) >= 2 and all(values and any(values) for values in planned)
    if planning_ms >= 200:
        # Real policy decisions reset command-based idle because they request
        # motion. Every actual packet is a hold; those holds must still time out.
        assert all(packet["values"] == [0.] * 6 for packet in packets)
        assert records[-1][0] == policy.FAILED, records
        assert records[0][1] > 0. and records[-1][1] >= card._config.idle_timeout_s
        assert "no motion command issued" in card._last_decision.reason
    else:
        assert len(records) == 8
        assert all(any(packet["values"]) for packet in packets)
        assert all(idle == 0. for _, idle in records)
        assert not card._state.failed_reason
