"""Exercise submission, interlocks and final visual verification without hardware."""
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.autonomous_task import plugin


def test_card_schema_and_optional_inputs():
    card = plugin.AutonomousTaskPlugin({"key": "private"}, None)
    schema, = card.get_tools()
    assert card.PREFIX + "_" + schema["name"] == "autonomous_task"
    assert [p["format"] for p in schema["topic_in"]] == ["image/jpeg"] * 3 + ["state/eef"]
    assert [p["format"] for p in schema["topic_out"]] == ["control/eef"]
    assert "private" not in repr(schema)
    assert "private" not in repr(card.dispatch("task", {"action": "info"}))
    assert card.dispatch("task", {"action": "submit", "instruction": "inspect"})["state"] == "error"


def test_query_recorded_task_after_card_restart(tmp_path):
    action_id = "autonomous_task_" + "a" * 32
    record = {"action_id": action_id, "state": "completed", "history": [{"step": 1, "tool": "done"}]}
    (tmp_path / action_id).mkdir()
    (tmp_path / action_id / "result.json").write_text(json.dumps(record))
    card = plugin.AutonomousTaskPlugin({"record_dir": str(tmp_path)}, None)
    args = {"action": "status", "action_id": action_id}
    assert card.dispatch("task", args) == {"action_id": action_id, "state": "completed", "last": record["history"][0]}
    assert card.dispatch("task", {**args, "include_history": True})["history"] == record["history"]
    assert card.dispatch("task", {**args, "action_id": "../" + action_id})["state"] == "error"


def test_decide_act_reobserve_and_review(monkeypatch):
    card = plugin.AutonomousTaskPlugin({}, None)
    run = {"action_id": "a", "instruction": "lift", "arm_mode": "left", "history": []}
    model, driver, cameras, cancel = Mock(), Mock(), Mock(), threading.Event()
    model.last_note, model.last_model = "observation", "test"
    model.decide.side_effect = [("left_set_gripper", {"opening_mm": 2}), ("done", {"reason": "held"}),
                                ("done", {"reason": "visible"})]
    model.review.side_effect = [("observe", {"reason": "occluded"}), ("done", {"reason": "verified"})]
    driver.execute.return_value = {"motion_sent": True, "status": "completed"}
    monkeypatch.setattr(cancel, "wait", lambda seconds: False)
    assert card._loop(run, model, driver, cameras, cancel, None) == ("completed", "verified")
    assert cameras.snapshot.call_count == driver.observe.call_count == 5
    assert driver.execute.call_count == 1
    assert [r["review"] for r in run["history"]] == [False, False, True, False, True]


def test_cancel_while_model_is_deciding():
    card = plugin.AutonomousTaskPlugin({}, None)
    model, driver, cancel = Mock(), Mock(), threading.Event()
    def decide(*_):
        cancel.set()
        return "left_set_gripper", {"opening_mm": 10}
    model.decide.side_effect = decide
    run = {"action_id": "a", "instruction": "lift", "arm_mode": "left", "history": []}
    with pytest.raises(InterruptedError):
        card._loop(run, model, driver, Mock(), cancel, None)
    driver.execute.assert_not_called()


def test_transient_inference_failure_reobserves_and_never_replays_motion(monkeypatch):
    card = plugin.AutonomousTaskPlugin({}, None)
    model, driver, cameras, cancel = Mock(), Mock(), Mock(), threading.Event()
    monkeypatch.setattr(cancel, "wait", lambda seconds: False)
    model.last_note, model.last_model = "seen", "vision"
    model.decide.side_effect = [("left_move_by", {}), plugin.TransientModelError("timeout"),
                                ("done", {"reason": "done"})]
    model.review.side_effect = [plugin.TransientModelError("503"), ("done", {"reason": "verified"})]
    driver.execute.return_value = {"status": "completed", "motion_sent": True}
    run = {"action_id": "r", "instruction": "move", "arm_mode": "left", "history": []}
    assert card._loop(run, model, driver, cameras, cancel, None) == ("completed", "verified")
    assert cameras.snapshot.call_count == driver.observe.call_count == 5
    assert driver.execute.call_count == 1
    assert [r["tool"] for r in run["history"]] == [
        "left_move_by", "model_unavailable", "done", "model_unavailable", "done"]


def test_unconfirmed_gripper_outcome_is_returned_for_visual_reassessment():
    card = plugin.AutonomousTaskPlugin({}, None)
    model, driver, cameras = Mock(), Mock(), Mock()
    model.last_note, model.last_model = "check the actual opening", "vision"
    model.decide.side_effect = [("left_set_gripper", {"opening_mm": 60}),
                                ("failed", {"reason": "jaw path occluded"})]
    driver.execute.return_value = {"status": "target_not_reached", "motion_sent": True}
    run = {"action_id": "r", "instruction": "move", "arm_mode": "left", "history": []}
    assert card._loop(run, model, driver, cameras, threading.Event(), None) == (
        "failed", "jaw path occluded")
    assert cameras.snapshot.call_count == 2 and driver.execute.call_count == 1
    assert run["history"][0]["feedback"] == driver.execute.return_value


def test_duplicate_and_busy_submit(monkeypatch):
    card = plugin.AutonomousTaskPlugin({"model_name": "vision", "base_url": "http://localhost",
                                        "key": "private"}, None)
    card._eef, card._cameras, card._binding = Mock(), Mock(), {"head": "/camera", "eef_state": "/state/eef"}
    card._interface = Mock(tip_frames={"left": "tool"})
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    args = {"action": "submit", "instruction": "inspect", "request_id": "one"}
    first = card.dispatch("task", args)
    assert first["state"] == "running"
    assert first == card.dispatch("task", args)
    assert card.dispatch("task", {**args, "request_id": "two"})["state"] == "error"
    assert card.dispatch("task", {**args, "instruction": "other"})["state"] == "error"
    assert card.dispatch("task", {"action": "cancel", "action_id": "unknown"})["state"] == "error"
    assert not card._cancel.is_set()
    assert card.dispatch("task", {"action": "cancel"})["state"] == "cancelling"
    assert card._cancel.is_set()


def test_stop_with_outstanding_task_keeps_endpoint_until_worker_finishes():
    card = plugin.AutonomousTaskPlugin({}, Mock())
    card._node, card._eef, card._cameras = Mock(), Mock(), Mock()
    card._active, card._cancel = "r", threading.Event()
    run = {"action_id": "r", "step": 0, "history": []}
    card._runs["r"] = run
    assert card.dispatch("task", {"action": "stop"}) == {"state": "stopping", "pending_task": "r"}
    assert card._cancel.is_set()
    card._eef.close.assert_not_called()
    with patch.object(plugin, "_completion"):
        card._loop = Mock(side_effect=InterruptedError("cancelled"))
        card._work(run, Mock(), card._eef, card._cameras, card._cancel)
    assert run["state"] == "cancelled" and card._node is None


def test_mcp_start_negotiates_wired_ports_and_arm_mode(monkeypatch):
    node, executor = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "rclpy.node", SimpleNamespace(Node=lambda _: node))
    cfg = {"control_topic": "/bench/command", "state_topic": "/bench/observed"}
    card = plugin.AutonomousTaskPlugin(cfg, executor)
    descriptor = {"control_interface": "bench.eef/1", "frame": "bench", "mode": "eef_pose",
                  "tip_frames": {"right": "tool"}, "opening_m": [0, .046], "dof": 8,
                  "groups": [{"name": "eef_r", "offset": 0, "count": 7, "unit": "m+quat_xyzw"},
                             {"name": "gripper_r", "offset": 7, "count": 1, "unit": "m"}]}
    args = {"action": "start", "instance_id": "test", "input_topics": ["/bench/observed", "/camera"],
            "camera_info": {"/camera": {"id": "right_wrist_camera"}}, "control_interface": descriptor}
    assert card.dispatch("task", {**args, "control_interface": {}})["state"] == "error"
    info = card.dispatch("task", args)
    assert info["state"] == "running" and info["inputs"] == {"right_wrist": "/camera", "eef_state": "/bench/observed"}
    assert [p.get("topic") for p in info["topic_in"]] == [None, None, "/camera", "/bench/observed"]
    assert info["topic_out"][0]["topic"] == "/bench/command"
    assert card.dispatch("task", args) == info
    executor.add_node.assert_called_once_with(node)
    assert card.dispatch("task", {"action": "submit", "instruction": "pick", "arm_mode": "left"})["state"] == "error"
    assert card.dispatch("task", {"action": "stop"})["state"] == "idle"
    assert card.dispatch("task", {**args, "input_topics": ["/bench/observed"]})["state"] == "running"
    assert card.dispatch("task", {"action": "submit", "instruction": "pick", "arm_mode": "right"})["state"] == "error"


def test_published_command_with_missing_feedback_ends_task_without_retry():
    card = plugin.AutonomousTaskPlugin({}, None)
    model, eef = Mock(), Mock()
    model.decide.return_value = ("left_move_by", {})
    result = {"status": "outcome_unknown", "motion_sent": True, "reason": "no feedback", "command_stamp_ms": 17}
    eef.execute.return_value = result
    run = {"action_id": "r", "instruction": "move", "arm_mode": "left", "history": []}
    assert card._loop(run, model, eef, Mock(), threading.Event(), None) == ("failed", "no feedback")
    assert run["history"][0]["feedback"] == result and eef.execute.call_count == 1


def test_invalid_text_decision_reobserves_without_actuation():
    card = plugin.AutonomousTaskPlugin({}, None)
    model, eef, cameras = Mock(), Mock(), Mock()
    model.decide.side_effect = [plugin.InvalidDecisionError("no status", "需要辨认目标")] * 3
    run = {"action_id": "r", "instruction": "move", "arm_mode": "left", "history": []}
    state, _ = card._loop(run, model, eef, cameras, threading.Event(), None)
    assert state == "failed" and cameras.snapshot.call_count == 3
    eef.execute.assert_not_called()
    assert all(record["note"] == "需要辨认目标" and not record["feedback"]["motion_sent"]
               for record in run["history"])
