"""Exercise submission, interlocks and final visual verification without hardware."""
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.autonomous_task import plugin


def test_card_schema_and_optional_inputs():
    card = plugin.AutonomousTaskPlugin({"key": "private"}, None)
    schema, = card.get_tools()
    assert card.PREFIX + "_" + schema["name"] == "autonomous_task"
    assert len(schema["topic_in"]) == 3 and len(schema["topic_out"]) == 2
    assert "private" not in repr(schema)
    assert "private" not in repr(card.dispatch("task", {"action": "info"}))
    assert card.dispatch("task", {"action": "submit", "instruction": "inspect"})["state"] == "error"


def test_decide_act_reobserve_and_review(monkeypatch):
    card = plugin.AutonomousTaskPlugin({}, None)
    run = {"action_id": "a", "instruction": "lift", "arm_mode": "left", "history": []}
    model, driver, cameras, cancel = Mock(), Mock(), Mock(), threading.Event()
    model.last_note, model.last_model = "observation", "test"
    model.decide.side_effect = [("left_set_gripper", {"opening_mm": 2}), ("done", {"reason": "held"}),
                                ("done", {"reason": "visible"})]
    model.review.side_effect = [("observe", {"reason": "occluded"}), ("done", {"reason": "verified"})]
    driver.execute.return_value = {"motion_sent": True, "status": "completed"}
    monkeypatch.setattr(card, "_publish", Mock())
    monkeypatch.setattr(cancel, "wait", lambda seconds: False)
    assert card._loop(run, model, driver, cameras, cancel, None) == ("completed", "verified")
    assert cameras.snapshot.call_count == driver.observe.call_count == 5
    assert driver.execute.call_count == card._publish.call_count == 1
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
    monkeypatch.setattr(card, "_publish", Mock())
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


def test_unconfirmed_gripper_outcome_is_observed_and_not_published(monkeypatch):
    card = plugin.AutonomousTaskPlugin({}, None)
    model, driver, cameras = Mock(), Mock(), Mock()
    monkeypatch.setattr(card, "_publish", Mock())
    model.last_note, model.last_model = "check the actual opening", "vision"
    model.decide.side_effect = [("left_set_gripper", {"opening_mm": 60}),
                                ("failed", {"reason": "jaw path occluded"})]
    driver.execute.return_value = {"status": "target_not_reached", "motion_sent": True}
    run = {"action_id": "r", "instruction": "move", "arm_mode": "left", "history": []}
    assert card._loop(run, model, driver, cameras, threading.Event(), None) == (
        "failed", "jaw path occluded")
    assert cameras.snapshot.call_count == 2 and driver.execute.call_count == 1
    assert run["history"][0]["feedback"] == driver.execute.return_value
    card._publish.assert_not_called()


def test_duplicate_and_busy_submit(monkeypatch):
    card = plugin.AutonomousTaskPlugin({"model_name": "vision", "base_url": "http://localhost",
                                        "key": "private"}, None)
    card._cameras, card._binding = Mock(), {"head": "/camera"}
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
