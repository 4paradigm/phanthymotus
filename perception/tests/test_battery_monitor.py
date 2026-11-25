"""Battery monitor: one canonical reading from three vendors' payloads.

The three payloads below are copied from the driver cards as they ship today —
Bumi (`noetix/bumi/device.py`), G1 (`unitree/g1/device.py`) and Tianyi 2.0
(`x-humanoid/tianyi2.0/device.py`). They are the reason normalization exists:
the same question ("how much battery is left") has three different shapes, and
one of them has no answer at all.

Runs on the shared ROS stubs — no rclpy, no robot.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from vision_stubs import _FakeExecutor, _FakeString

from plugins import battery_monitor
from plugins.battery_monitor import (
    SCHEMA,
    BatteryMonitorPlugin,
    normalize_reading,
)

BUMI = {"soc": 87, "soh": 99, "temperature": 33, "alarm": 0}
G1 = {
    "soc": 64,
    "soh": 97,
    "current": -3140,               # mA, per unitree/g1/device.py
    "voltage": [3940, 3937, 3942, 0],  # per-cell mV
    "cell_vol": [3940, 3937, 3942, 0],
    "temperature": [31, 34, 33, 0],    # °C per cell
    "cycle": 12,
}
TIANYI = {
    "master_voltage": 48.6,
    "master_current": -2.5,
    "master_power": -121.5,
    "little_voltage": 12.1,
    "little_current": -0.4,
    "little_power": -4.8,
    "battery_installed": True,
    "battery_working": True,
}


# ── normalization ────────────────────────────────────────────────────────────

def test_bumi_payload_maps_the_fields_whose_units_need_no_profile():
    assert normalize_reading(BUMI) == {"soc": 87.0, "soh": 99.0, "alarm": 0.0}


def test_auto_profile_leaves_ambiguous_bare_fields_unmapped():
    """`current`, `cell_vol` and `temperature` mean nothing without a unit.
    Adam and K1 also publish bare `current` with no unit stated, so `auto`
    refuses them all rather than publishing G1's mA as if it were A."""
    reading = normalize_reading(G1)
    assert reading == {"soc": 64.0, "soh": 97.0, "cycle": 12.0}


def test_g1_profile_reads_the_unitree_bms_units():
    reading = normalize_reading(G1, profile="g1")
    assert reading["soc"] == 64.0
    assert reading["current_ma"] == -3140.0
    # Zero-padded unused cell slots are not measurements.
    assert reading["temperature_c"] == 34.0
    assert reading["cell_voltage_max_mv"] == 3942.0
    # `voltage` on G1 is the package list in mV (49.0 V per pack on the office
    # robot), not a cell list.
    assert reading["pack_voltage_mv"] == 3942.0
    assert reading["cycle"] == 12.0


def test_g1_cell_list_never_becomes_pack_voltage():
    """A pack voltage cannot be derived from a cell list — topology is the
    model's, not the message's. Inventing one would be a confident wrong
    number, which is the failure this card exists to avoid."""
    assert "voltage_v" not in normalize_reading(G1, profile="g1")


def test_unknown_profile_falls_back_to_auto():
    assert normalize_reading(G1, profile="nonsense") == normalize_reading(G1)


def test_tianyi_reports_voltage_but_no_soc():
    reading = normalize_reading(TIANYI)
    assert reading["voltage_v"] == 48.6
    assert reading["current_a"] == -2.5
    assert reading["power_w"] == -121.5
    assert reading["installed"] is True
    assert reading["working"] is True
    # Absent, not zero: a renderer must be able to say "not reported".
    assert "soc" not in reading


def test_unknown_fields_are_not_guessed():
    assert normalize_reading({"some_vendor_field": 42}) == {}


def test_non_mapping_and_non_numeric_values_are_ignored():
    assert normalize_reading(None) == {}
    assert normalize_reading(["soc", 50]) == {}
    # A boolean SOC is a bug in the publisher, not 1%.
    assert normalize_reading({"soc": True, "soh": "99"}) == {}


def test_charging_accepts_common_encodings():
    assert normalize_reading({"charging": 1})["charging"] is True
    assert normalize_reading({"charge_state": "discharging"})["charging"] is False
    assert "charging" not in normalize_reading({"charging": "maybe"})


# ── node: message in, normalized topic out ───────────────────────────────────

def _node(**kwargs):
    return battery_monitor._BatteryNode(
        "/ubuntu/state/battery", "/perception/battery", kwargs.get("stale_after_s", 10.0)
    )


def _message(payload):
    msg = _FakeString()
    msg.data = json.dumps(payload)
    return msg


def test_message_is_republished_with_schema_and_freshness():
    node = _node()
    node._on_message(_message(BUMI))

    published = json.loads(node._publisher.messages[-1])
    assert published["schema"] == SCHEMA
    assert published["source_topic"] == "/ubuntu/state/battery"
    assert published["soc"] == 87.0
    assert published["stale"] is False
    assert published["age_s"] == 0.0


def test_unparseable_message_is_counted_not_published():
    node = _node()
    msg = _FakeString()
    msg.data = "not json"
    node._on_message(msg)

    assert node._publisher.messages == []
    assert node.statistics()["unparsed"] == 1


def test_reading_goes_stale_without_new_input():
    node = _node(stale_after_s=5.0)
    with patch.object(battery_monitor.time, "monotonic", return_value=100.0):
        node._on_message(_message(BUMI))
    with patch.object(battery_monitor.time, "monotonic", return_value=106.0):
        latest = node.latest()
    assert latest["stale"] is True
    assert latest["age_s"] == 6.0


def test_timer_republishes_so_a_dead_robot_greys_out():
    node = _node()
    with patch.object(battery_monitor.time, "monotonic", return_value=100.0):
        node._on_message(_message(BUMI))
    before = len(node._publisher.messages)
    with patch.object(battery_monitor.time, "monotonic", return_value=112.0):
        node._timer.callback()
    assert len(node._publisher.messages) == before + 1
    assert json.loads(node._publisher.messages[-1])["stale"] is True


def test_timer_without_any_input_publishes_nothing():
    node = _node()
    node._timer.callback()
    assert node._publisher.messages == []


# ── facade: canvas wiring and dispatch ───────────────────────────────────────

def _plugin(**cfg):
    executor = _FakeExecutor()
    return BatteryMonitorPlugin(cfg, executor), executor


def test_start_without_a_topic_is_an_error_not_a_silent_idle():
    plugin, executor = _plugin()
    result = plugin.dispatch("battery_monitor", {"action": "start"})
    assert result["state"] == "error"
    assert "input topic" in result["error"]
    assert executor.nodes == []


def test_start_takes_the_canvas_wiring_and_reports_its_topics():
    plugin, executor = _plugin()
    plugin.dispatch("battery_monitor", {"action": "start"})
    result = plugin.dispatch(
        "battery_monitor",
        {"action": "start", "input_topics": ["/ubuntu/state/battery"]},
    )
    assert result["state"] == "running"
    assert result["topic_in"][0]["topic"] == "/ubuntu/state/battery"
    assert result["topic_out"][0]["format"] == "sensor/battery"
    assert len(executor.nodes) == 1


def test_starting_twice_on_the_same_topic_does_not_stack_nodes():
    plugin, executor = _plugin()
    args = {"action": "start", "input_topics": ["/ubuntu/state/battery"]}
    plugin.dispatch("battery_monitor", args)
    plugin.dispatch("battery_monitor", args)
    assert len(executor.nodes) == 1


def test_status_is_the_skill_facing_query():
    plugin, _ = _plugin()
    plugin.dispatch("battery_monitor", {"action": "status"})
    plugin.dispatch(
        "battery_monitor",
        {"action": "start", "input_topics": ["/ubuntu/state/battery"]},
    )
    node = plugin._node
    node._on_message(_message(BUMI))

    status = plugin.dispatch("battery_monitor", {"action": "status"})
    assert status["state"] == "running"
    assert status["latest"]["soc"] == 87.0
    assert status["statistics"]["received"] == 1


def test_status_before_start_says_so():
    plugin, _ = _plugin()
    result = plugin.dispatch("battery_monitor", {"action": "status"})
    assert result["state"] == "idle"
    assert result["latest"] is None


def test_stop_disposes_the_node_and_removes_it_from_the_executor():
    plugin, executor = _plugin()
    plugin.dispatch(
        "battery_monitor",
        {"action": "start", "input_topics": ["/ubuntu/state/battery"]},
    )
    node = plugin._node
    result = plugin.dispatch("battery_monitor", {"action": "stop"})
    assert result["state"] == "idle"
    assert plugin._node is None
    assert node.destroyed is True
    assert executor.nodes == []


def test_unknown_action_is_declined_so_the_mcp_layer_can_say_why():
    plugin, _ = _plugin()
    assert plugin.dispatch("battery_monitor", {"action": "explode"}) is None


def test_config_reports_effective_values():
    plugin, _ = _plugin(stale_after_s=3)
    result = plugin.dispatch("battery_monitor", {"action": "config"})
    assert result["config"]["output_topic"] == "/perception/battery"
    assert result["config"]["stale_after_s"] == 3.0
    assert result["config"]["input_profile"] == "auto"
    assert "g1" in result["config"]["profiles"]


def test_config_from_the_canvas_dialog_is_applied():
    """A dialog that saves and changes nothing is worse than no dialog."""
    plugin, _ = _plugin()
    result = plugin.dispatch("battery_monitor",
                             {"action": "config", "stale_after_s": 3, "input_profile": "g1"})
    assert result["status"] == "configured"
    assert result["config"]["stale_after_s"] == 3.0
    assert result["config"]["input_profile"] == "g1"


def test_config_rejects_an_unknown_profile_and_keeps_the_old_one():
    plugin, _ = _plugin()
    result = plugin.dispatch("battery_monitor",
                             {"action": "config", "input_profile": "bumi-app"})
    assert result["status"] == "error"
    assert result["config"]["input_profile"] == "auto"


def test_config_rebuilds_the_node_when_its_output_topic_changes():
    plugin, executor = _plugin()
    plugin.dispatch("battery_monitor",
                    {"action": "start", "input_topics": ["/ubuntu/state/battery"]})
    old = plugin._node
    plugin.dispatch("battery_monitor",
                    {"action": "config", "output_topic": "/perception/battery2"})
    new = plugin._node
    assert new is not old
    assert old.destroyed is True
    assert new.output_topic == "/perception/battery2"
    assert new._publisher.topic == "/perception/battery2"
    assert [n.input_topic for n in executor.nodes] == ["/ubuntu/state/battery"]


def test_config_patches_the_staleness_window_without_a_restart():
    plugin, _ = _plugin()
    plugin.dispatch("battery_monitor",
                    {"action": "start", "input_topics": ["/ubuntu/state/battery"]})
    node = plugin._node
    plugin.dispatch("battery_monitor", {"action": "config", "stale_after_s": 2})
    assert plugin._node is node
    assert node.stale_after_s == 2.0


def test_unknown_profile_is_rejected_at_start():
    plugin, executor = _plugin(input_profile="bumi-app")
    result = plugin.dispatch(
        "battery_monitor",
        {"action": "start", "input_topics": ["/g1/state/battery"]},
    )
    assert result["state"] == "error"
    assert "input_profile" in result["error"]
    assert executor.nodes == []


def test_g1_profile_reaches_the_node_and_the_reading():
    plugin, _ = _plugin(input_profile="g1")
    plugin.dispatch(
        "battery_monitor",
        {"action": "start", "input_topics": ["/g1/state/battery"]},
    )
    plugin._node._on_message(_message(G1))

    status = plugin.dispatch("battery_monitor", {"action": "status"})
    assert status["profile"] == "g1"
    assert status["latest"]["current_ma"] == -3140.0
    assert status["latest"]["cell_voltage_max_mv"] == 3942.0


def test_tool_declaration_matches_what_the_card_publishes():
    from plugins.battery_monitor import TOOLS

    (tool,) = TOOLS
    assert tool["name"] == "battery_monitor"
    assert tool["multiInstance"] is False
    assert tool["topic_out"][0]["format"] == "sensor/battery"
    assert tool["topic_in"][0]["format"] == "data/json"
    assert "status" in tool["inputSchema"]["properties"]["action"]["enum"]


def test_enabled_in_config_registers_the_card_with_the_bundle():
    """The registration block in main.py is the only thing between the file and
    the canvas palette — a typo there means the card never appears."""
    import main as perception_main

    bundle = perception_main.PerceptionBundle(
        {"plugins": {"battery_monitor": {"enabled": True}}}, _FakeExecutor()
    )
    # echo_cancellation / audio_mixer default to enabled, so membership is the
    # claim here, not exclusivity.
    assert "battery_monitor" in [plugin.PREFIX for plugin in bundle._plugins]
    assert any(tool["name"] == "battery_monitor" for tool in bundle.get_all_tools())
