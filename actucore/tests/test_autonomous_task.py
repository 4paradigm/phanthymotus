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
    settings = schema["configSchema"]["properties"]
    assert all(field["scope"] == "shared" for field in settings.values())
    assert {"state_topic", "control_topic"}.isdisjoint(settings)
    assert {"head_topic", "left_wrist_topic", "right_wrist_topic"} <= settings.keys()
    assert settings["arm_mode"]["enum"] == ["left", "right", "both"]
    assert settings["arm_mode"]["default"] == "left"
    assert list(settings)[:3] == ["base_url", "key", "model_name"]
    assert [field["description"] for field in list(settings.values())[:3]] == [
        "LLM API URL", "LLM API Key", "LLM 模型名称"]
    assert "arm_mode" not in schema["inputSchema"]["properties"]
    assert "arm_mode" not in schema["inputSchema"]["x-action-params"]["submit"]["params"]
    assert "private" not in repr(schema)
    assert "private" not in repr(card.dispatch("task", {"action": "info"}))
    assert card.dispatch("task", {"action": "submit", "instruction": "inspect"})["state"] == "error"


def test_key_setting_uses_the_same_password_field_before_and_after_saving():
    card = plugin.AutonomousTaskPlugin({}, None)
    field = card.get_tools()[0]["configSchema"]["properties"]["key"]
    assert field == {"type": "string", "scope": "shared", "description": "LLM API Key", "format": "password"}

    credential = "test-only-model-credential"
    result = card.dispatch("task", {"action": "config", "key": credential})
    assert result["state"] == "idle"
    schema, = card.get_tools()
    assert schema["configSchema"]["properties"]["key"] == field
    assert credential not in json.dumps([schema, result])


def test_mcp_has_no_action_to_list_or_read_model_logs(tmp_path):
    card = plugin.AutonomousTaskPlugin({"model_log_dir": str(tmp_path)}, None)
    schema = card.get_tools()[0]["inputSchema"]
    assert "get_model_log" not in schema["properties"]["action"]["enum"]
    assert "get_model_log" not in schema["x-action-params"]
    assert {"filename", "offset", "revision"}.isdisjoint(schema["properties"])

    action_id = "autonomous_task_" + "a" * 32
    session = tmp_path / action_id
    session.mkdir()
    filename = "0001-request.json"
    (session / filename).write_text('{"model": "test-vision"}')
    for fields in ({}, {"filename": filename, "offset": 0}):
        result = card.dispatch("task", {"action": "get_model_log", "action_id": action_id, **fields})
        assert result is None or result.get("state") == "error"


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


def test_duplicate_and_busy_submit(monkeypatch, tmp_path):
    card = plugin.AutonomousTaskPlugin({"model_name": "vision", "base_url": "http://localhost",
                                        "key": "private", "model_log_dir": str(tmp_path)}, None)
    card._eef, card._cameras, card._binding = Mock(), Mock(), {"head": "/camera", "eef_state": "/state/eef"}
    card._interface = Mock(tip_frames={"left": "tool"})
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    args = {"action": "submit", "instruction": "inspect", "request_id": "one"}
    first = card.dispatch("task", args)
    assert first["state"] == "running"
    assert first == card.dispatch("task", args)
    assert card.dispatch("task", {**args, "request_id": "two"})["state"] == "error"
    assert card.dispatch("task", {**args, "instruction": "other"})["state"] == "error"
    assert card.dispatch("task", {"action": "config", "arm_mode": "both"})["state"] == "error"
    session = tmp_path / first["action_id"]
    assert list(tmp_path.iterdir()) == [session]
    assert [p.name for p in session.iterdir()] == ["0000-session.json"]
    assert json.loads((session / "0000-session.json").read_bytes())["action_id"] == first["action_id"]
    card._eef.begin_task.assert_called_once_with()
    assert card.dispatch("task", {"action": "cancel", "action_id": "unknown"})["state"] == "error"
    assert not card._cancel.is_set()
    card._eef.cancel.assert_not_called()
    assert card.dispatch("task", {"action": "cancel"})["state"] == "cancelling"
    card._eef.cancel.assert_called_once_with()
    assert card._cancel.is_set()


def test_shared_arm_setting_controls_submits_and_replays(tmp_path, monkeypatch):
    card = plugin.AutonomousTaskPlugin({"model_name": "vision", "base_url": "http://localhost",
                                        "key": "private", "model_log_dir": str(tmp_path)}, None)
    card._binding = {"head": "/camera", "eef_state": "/state/eef"}
    card._eef, card._cameras = Mock(fault=None), Mock()
    card._interface = Mock(tip_frames={"left": "left_tcp", "right": "right_tcp"})
    monkeypatch.setattr(threading.Thread, "start", threading.Thread.run)
    monkeypatch.setattr(plugin, "_completion", Mock())
    monkeypatch.setattr(card, "_loop", Mock(return_value=("completed", "fixture")))

    setting = {"action": "config", "arm_mode": "right"}
    assert card.dispatch("task", {**setting, "instance_id": "a"})["state"] == "error"
    assert card.dispatch("task", {"action": "info", "instance_id": "b"})["arm_mode"] == "left"
    assert card.dispatch("task", setting)["arm_mode"] == "right"

    submit = {"action": "submit", "instruction": "inspect", "request_id": "first", "instance_id": "a"}
    assert card.dispatch("task", {**submit, "arm_mode": "left"})["state"] == "error"
    card._eef.begin_task.assert_not_called()
    first = card.dispatch("task", submit)
    assert (first["state"], first["arm_mode"]) == ("completed", "right")
    second = card.dispatch("task", {**submit, "request_id": "second", "instance_id": "b", "arm_mode": "right"})
    assert (second["state"], second["arm_mode"]) == ("completed", "right")
    assert first["action_id"] != second["action_id"]

    assert card.dispatch("task", {**setting, "arm_mode": "both"})["arm_mode"] == "both"
    assert card.dispatch("task", submit) == first
    third = card.dispatch("task", {**submit, "request_id": "third"})
    assert (third["state"], third["arm_mode"]) == ("completed", "both")
    assert card._eef.begin_task.call_count == 3


def test_stop_with_outstanding_task_keeps_endpoint_until_worker_finishes():
    card = plugin.AutonomousTaskPlugin({}, Mock())
    card._node, card._eef, card._cameras = Mock(), Mock(fault=None), Mock()
    card._active, card._cancel = "r", threading.Event()
    run = {"action_id": "r", "step": 0, "history": []}
    card._runs["r"] = run
    assert card.dispatch("task", {"action": "stop"}) == {"state": "stopping", "pending_task": "r"}
    assert card._cancel.is_set()
    card._eef.cancel.assert_called_once_with()
    card._eef.close.assert_not_called()
    with patch.object(plugin, "_completion"):
        card._loop = Mock(side_effect=InterruptedError("cancelled"))
        card._work(run, Mock(), card._eef, card._cameras, card._cancel)
    assert run["state"] == "cancelled" and card._node is None


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_terminal_task_retires_its_target_before_the_next_submit(outcome):
    card = plugin.AutonomousTaskPlugin({}, Mock())
    run = {"action_id": "r", "step": 2, "history": []}
    endpoint, event = Mock(fault=None), threading.Event()
    card._active, card._cancel = run["action_id"], event
    with patch.object(card, "_loop", return_value=(outcome, "reviewed")), patch.object(plugin, "_completion"):
        card._work(run, Mock(), endpoint, Mock(), event)
    endpoint.cancel.assert_called_once_with()
    assert card._active is None and run["state"] == outcome


def test_final_log_disk_failure_is_reported_and_releases_task():
    card = plugin.AutonomousTaskPlugin({}, None)
    model = Mock()
    model.model_log.append.side_effect = OSError("disk full")
    run = {"action_id": "r", "step": 0, "history": []}
    endpoint, event = Mock(fault=None), threading.Event()
    card._active, card._cancel = "r", event
    with patch.object(card, "_loop", return_value=("completed", "checked")), patch.object(plugin, "_completion") as notify:
        card._work(run, model, endpoint, Mock(), event)
    assert run["model_log_error"] == "disk full"
    assert run["state"] == "failed" and card._active is None
    endpoint.cancel.assert_called_once_with()
    notify.assert_called_once_with("r", "failed", run["reason"])


def test_stream_fault_during_inference_wins_over_a_late_model_action():
    card = plugin.AutonomousTaskPlugin({}, Mock())
    endpoint, model = Mock(fault=None), Mock()
    card._node, card._eef = Mock(), endpoint
    run = {"action_id": "r", "step": 0, "history": [], "instruction": "pick", "arm_mode": "left"}
    card._active, card._cancel = run["action_id"], threading.Event()
    card._runs["r"] = run
    reason = "right TCP drifted from its measured hold by 16 mm"

    def late_decision(*_):
        endpoint.fault = reason
        endpoint.check_fault.side_effect = RuntimeError(reason)
        return "left_set_gripper", {"opening_mm": 10}

    model.decide.side_effect = late_decision
    with patch.object(plugin, "_completion") as completion:
        card._work(run, model, endpoint, Mock(), card._cancel)
    endpoint.execute.assert_not_called()
    endpoint.cancel.assert_not_called()
    assert card._active is None and run["state"] == "failed"
    assert card.dispatch("task", {"action": "status", "action_id": "r"})["reason"] == reason
    info = card.dispatch("task", {"action": "info"})
    assert info["state"] == "error" and info["fault"] == reason
    completion.assert_called_once_with("r", "failed", reason)


def test_mcp_start_negotiates_wired_ports_and_arm_mode(monkeypatch):
    node, executor = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "rclpy.node", SimpleNamespace(Node=lambda _: node))
    cfg = {"control_topic": "/retired/command", "state_topic": "/retired/observed"}
    card = plugin.AutonomousTaskPlugin(cfg, executor)
    descriptor = {"control_interface": "motus.control/1", "frame": "bench", "mode": "eef_pose",
                  "tip_frames": {"right": "tool"}, "opening_m": [0, .046], "dof": 8,
                  "state": {"schema": "motus.eef/1", "field": "eef"},
                  "rate": {"expected_hz": 10, "watchdog_ms": 200, "max_obs_age_ms": 500},
                  "limits": {"lower": [-1.] * 7 + [0.], "upper": [1.] * 7 + [.046]},
                  "groups": [{"name": "eef_r", "offset": 0, "count": 7, "unit": "m+quat_xyzw"},
                             {"name": "gripper_r", "offset": 7, "count": 1, "unit": "m"}]}
    args = {"action": "start", "instance_id": "test", "input_topics": ["/bench/observed", "/camera"],
            "camera_info": {"/camera": {"id": "right_wrist_camera"}}, "control_interface": descriptor}
    assert card.dispatch("task", {**args, "control_interface": {}})["state"] == "error"
    info = card.dispatch("task", args)
    assert info["state"] == "running" and info["inputs"] == {"right_wrist": "/camera", "eef_state": "/bench/observed"}
    assert [p.get("topic") for p in info["topic_in"]] == [None, None, "/camera", "/bench/observed"]
    output = info["topic_out"][0]["topic"]
    assert output == "/actucore/autonomous_task/control/eef"
    assert node.create_publisher.call_args.args[1] == output
    assert any(call.args[1] == "/bench/observed" for call in node.create_subscription.call_args_list)
    assert card.dispatch("task", args) == info
    executor.add_node.assert_called_once_with(node)
    submit = {"action": "submit", "instruction": "pick"}
    assert "不支持 left" in card.dispatch("task", submit)["message"]
    assert card.dispatch("task", {"action": "stop"})["state"] == "idle"
    assert card.dispatch("task", {"action": "config", "arm_mode": "right"})["arm_mode"] == "right"
    assert card.dispatch("task", {**args, "input_topics": ["/bench/observed"]})["state"] == "running"
    assert "至少一路 RGB" in card.dispatch("task", submit)["message"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_loop_never_dispatches_a_retired_absolute_tool(side):
    card = plugin.AutonomousTaskPlugin({}, None)
    model, driver = Mock(), Mock()
    tool = f"{side}_move_to"
    model.decide.return_value = (tool, {"position_m": [.11, .23, .34], "quaternion_xyzw": [0, 0, 0, 1]})
    run = {"action_id": "r", "instruction": "move", "arm_mode": "both", "history": []}
    with pytest.raises(ValueError, match=tool):
        card._loop(run, model, driver, Mock(), threading.Event(), None)
    driver.execute.assert_not_called()


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
