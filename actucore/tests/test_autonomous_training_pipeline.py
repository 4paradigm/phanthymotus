"""Submit through the real card and compare its persisted log to the HTTP exchange."""
import base64
import io
import json
from pathlib import Path
import queue
import sys
import threading
import urllib.request
from unittest.mock import Mock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.autonomous_task import plugin


def read_log(directory):
    return {path.name: json.loads(path.read_bytes()) for path in directory.glob("*.json")}


def jpeg(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, "JPEG")
    return buffer.getvalue()


def test_submits_publish_raw_multimodal_training_contexts(tmp_path, monkeypatch):
    cfg = {"base_url": "https://fixture.test", "model_name": "fixture-vision", "key": "test-only-secret",
           "model_log_dir": str(tmp_path)}
    card = plugin.AutonomousTaskPlugin(cfg, None)
    scenes = [{"head": jpeg(a), "left_wrist": jpeg(b)}
              for a, b in [("red", "blue"), ("green", "white"), ("yellow", "black")]]
    measured = {"opening_range_mm_by_arm": {"left": [0, 85]}, "arms": {"left": {"gripper": {"opening_mm": 42}}}}
    card._eef = Mock(fault=None)
    card._eef.observe.return_value = measured
    card._cameras = Mock()
    card._cameras.snapshot.side_effect = scenes
    card._binding = {"head": "/head", "left_wrist": "/left", "eef_state": "/eef_state"}
    card._interface = Mock(tip_frames={"left": "left_tcp"})
    received, release = threading.Event(), threading.Event()
    terminal = queue.Queue()
    monkeypatch.setattr(plugin, "_completion", lambda *args: terminal.put(args))
    requests, request_bytes, response_bytes = [], [], []

    def provider(request, timeout):
        assert request.full_url == "https://fixture.test/v1/responses" and request.method == "POST"
        request_bytes.append(request.data)
        requests.append(json.loads(request.data))
        if len(requests) == 1:
            received.set()
            assert release.wait(5)
        status = "done" if len(requests) < 3 else "failed"
        output = [{"type": "message", "content": [{"type": "output_text",
                   "text": json.dumps({"status": status, "reason": "fixture result"})}]}]
        result = {"id": f"resp_{len(requests)}", "model": cfg["model_name"], "status": "completed",
                  "output": output, "usage": {"input_tokens": 123}, "provider_extension": ["原始响应"]}
        body = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
        response_bytes.append(body)
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", provider)
    submit = {"action": "submit", "request_id": "one", "instruction": "看一眼胶带"}
    first_id = card.dispatch("task", submit)["action_id"]
    try:
        assert received.wait(5)
        assert card.dispatch("task", submit)["action_id"] == first_id
        pending = read_log(tmp_path / first_id)
        assert set(pending) == {"0000-session.json", "0001-request.json"}
        assert pending["0001-request.json"] == requests[0]
        assert card.dispatch("task", {"action": "status", "action_id": first_id})["state"] == "running"
    finally:
        release.set()
    assert terminal.get(timeout=10)[:2] == (first_id, "completed")
    first = read_log(tmp_path / first_id)

    second_id = card.dispatch("task", {**submit, "request_id": "two", "instruction": "检查左腕画面"})["action_id"]
    assert second_id != first_id and terminal.get(timeout=5)[:2] == (second_id, "failed")
    assert len(list(tmp_path.glob("*/0000-session.json"))) == 2
    assert read_log(tmp_path / first_id) == first
    second = read_log(tmp_path / second_id)
    assert [log["0000-session.json"]["request_id"] for log in (first, second)] == ["one", "two"]
    assert set(second) == {"0000-session.json", "0001-request.json", "0001-response.json", "0000-end.json"}
    assert set(first) == set(second) | {"0002-request.json", "0002-response.json"}
    for log, expected in [(first, "completed"), (second, "failed")]:
        end = log["0000-end.json"]
        assert end["status"] == expected and end["reason"] == "fixture result"
        action_id = log["0000-session.json"]["action_id"]
        assert card.dispatch("task", {"action": "status", "action_id": action_id})["state"] == expected
        assert cfg["key"] not in json.dumps(log)
    for i, (action_id, step) in enumerate(((first_id, 1), (first_id, 2), (second_id, 1))):
        assert (tmp_path / action_id / f"{step:04d}-request.json").read_bytes() == request_bytes[i]
        assert (tmp_path / action_id / f"{step:04d}-response.json").read_bytes() == response_bytes[i]
    assert first["0001-request.json"]["tools"] and first["0002-request.json"]["tools"] == []
    for request, visible_scenes in zip(requests, [[scenes[0]], scenes[:2], [scenes[2]]]):
        content = request["input"][0]["content"]
        raw_images = [part["image_url"].split(",", 1) for part in content if part["type"] == "input_image"]
        assert all(prefix == "data:image/jpeg;base64" for prefix, _ in raw_images)
        assert [base64.b64decode(data, validate=True) for _, data in raw_images] == [
            frame for scene in visible_scenes for frame in scene.values()]
        assert json.loads(content[0]["text"])["current_measured_state"] == measured
    initial = json.loads(requests[2]["input"][0]["content"][0]["text"])
    assert initial["instruction"] == second["0000-session.json"]["instruction"] and initial["actual_action_history"] == []
    card._eef.execute.assert_not_called()


@pytest.fixture
def tool_card(tmp_path, monkeypatch):
    card = plugin.AutonomousTaskPlugin({"base_url": "https://fixture.test", "model_name": "fixture",
                                        "key": "test-secret", "model_log_dir": str(tmp_path)}, None)
    card._interface = Mock(tip_frames={"left": "tcp"})
    card._binding = {"head": "/head", "eef_state": "/eef_state"}
    card._cameras = Mock()
    card._cameras.snapshot.return_value = {"head": jpeg("orange")}
    card._eef = Mock(fault=None)
    card._eef.observe.return_value = {"opening_range_mm_by_arm": {"left": [0, 85]}}
    finished = queue.Queue()
    monkeypatch.setattr(plugin, "_completion", lambda *args: finished.put(args))
    return card, finished


def motion_response(call_id, opening):
    return {"status": "completed", "model": "fixture", "output": [
        {"type": "function_call", "id": "item-" + call_id, "call_id": call_id, "name": "left_set_gripper",
         "arguments": json.dumps({"opening_mm": opening})}]}


@pytest.mark.parametrize("ending", ["outcome_unknown", "cancelled", "exception", "cancel_before_execute"])
def test_tool_result_persists_without_next_request(tool_card, tmp_path, monkeypatch, ending):
    card, finished = tool_card
    returned = {"status": "completed" if ending == "cancelled" else "outcome_unknown",
                "reason": "endpoint result", "motion_sent": True, "extension": {"samples": [21, 22]}}

    def reply(model, payload):
        if ending == "cancel_before_execute":
            card._cancel.set()
        return motion_response("call-last", 42)

    def execute(name, arguments, cancel):
        if ending == "exception":
            raise RuntimeError("endpoint offline")
        if ending == "cancelled":
            cancel.set()
        return returned

    monkeypatch.setattr(plugin.Model, "_post", reply)
    card._eef.execute.side_effect = execute
    action_id = card.dispatch("task", {"action": "submit", "instruction": "test tool"})["action_id"]
    outcome = "cancelled" if ending.startswith("cancel") else "failed"
    assert finished.get(timeout=5)[:2] == (action_id, outcome)
    files = read_log(tmp_path / action_id)
    ordinary = {"0000-session.json", "0001-request.json", "0001-response.json", "0000-end.json"}
    if ending == "cancel_before_execute":
        card._eef.execute.assert_not_called()
        assert set(files) == ordinary
        return
    card._eef.execute.assert_called_once()
    assert set(files) == ordinary | {"0001-tool-result.json"}
    result = files["0001-tool-result.json"]
    assert (result["step"], result["phase"], result["call_id"]) == (1, "decide", "call-last")
    assert (result["name"], result["arguments"]) == card._eef.execute.call_args.args[:2]
    if ending == "exception":
        assert result["error"] == {"type": "RuntimeError", "message": "endpoint offline"}
        assert "payload" not in result
    else:
        assert result["payload"] == returned
        assert "error" not in result


def test_returned_payload_remains_in_next_request_history(tool_card, tmp_path, monkeypatch):
    card, finished = tool_card
    rejected = {"status": "rejected", "motion_sent": False, "reason": "stale state", "extra": ["driver", 3]}
    final = {"status": "outcome_unknown", "reason": "missing telemetry", "motion_sent": True}
    card._eef.execute.side_effect = [rejected, final]
    requests = []

    def reply(model, payload):
        requests.append(payload)
        if len(requests) == 1:
            return motion_response("call-first", 12)
        return motion_response("call-second", 25)

    monkeypatch.setattr(plugin.Model, "_post", reply)
    action_id = card.dispatch("task", {"action": "submit", "instruction": "test tool"})["action_id"]
    assert finished.get(timeout=5)[:2] == (action_id, "failed")
    files = read_log(tmp_path / action_id)
    assert set(files) == {"0000-session.json", "0000-end.json",
                          "0001-request.json", "0001-response.json", "0001-tool-result.json",
                          "0002-request.json", "0002-response.json", "0002-tool-result.json"}
    results = [files["0001-tool-result.json"], files["0002-tool-result.json"]]
    assert [(e["step"], e["call_id"], e["arguments"], e["payload"]) for e in results] == [
        (1, "call-first", {"opening_mm": 12}, rejected), (2, "call-second", {"opening_mm": 25}, final)]
    assert [files["0001-request.json"], files["0002-request.json"]] == requests
    message, = requests[1]["input"]
    assert message["role"] == "user"
    history, = json.loads(message["content"][0]["text"])["actual_action_history"]
    assert (history["step"], history["tool"], history["feedback"]) == (1, "left_set_gripper", rejected)


def test_tool_log_failure_does_not_repeat_execution(tool_card, tmp_path, monkeypatch):
    card, finished = tool_card
    append = plugin.ModelLog.append

    def full_disk(log, kind, **event):
        if kind == "tool_result":
            raise OSError("disk full")
        return append(log, kind, **event)

    monkeypatch.setattr(plugin.ModelLog, "append", full_disk)
    monkeypatch.setattr(plugin.Model, "_post", lambda self, req: motion_response("call-once", 33))
    card._eef.execute.return_value = {"status": "completed", "motion_sent": True}
    action_id = card.dispatch("task", {"action": "submit", "instruction": "test tool"})["action_id"]
    ended_id, state, reason = finished.get(timeout=5)
    assert (ended_id, state) == (action_id, "failed") and "disk full" in reason
    card._eef.execute.assert_called_once()
    assert set(read_log(tmp_path / action_id)) == {"0000-session.json", "0000-end.json",
                                              "0001-request.json", "0001-response.json"}
