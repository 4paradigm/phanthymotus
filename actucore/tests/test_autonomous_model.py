"""Check the model boundary without a network or a ROS installation."""

import io
import json
import pathlib
import sys
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from plugins.autonomous_task.model import Model


def message(text):
    return {"type": "message", "status": "completed",
            "content": [{"type": "output_text", "text": text}]}


def call(name, arguments):
    return {"type": "function_call", "call_id": "call-1", "status": "completed",
            "name": name, "arguments": json.dumps(arguments)}


def reply(*items):
    return {"model": "azure-openai/gpt-6-astra", "status": "completed", "output": list(items)}


@pytest.fixture
def model():
    return Model("https://example.test/v1/", "test-key", "azure-openai/gpt-6-astra")


@pytest.mark.parametrize("arms", [("left",), ("right",), ("left", "right")])
def test_responses_request_uses_wired_rgb_state_history_and_selected_arm_tools(monkeypatch, model, arms):
    tool = arms[-1] + "_set_gripper"
    observation = {"left_wrist": b"jpeg"}
    state = {"opening_mm": 30}
    history = [{"action": "observe", "reason": "确认夹爪位置"}]
    def serve(request, timeout):
        assert request.full_url == "https://example.test/v1/responses"
        assert request.get_header("Authorization") == "Bearer test-key"
        assert timeout == 180
        body = json.loads(request.data)
        assert body["model"] == "azure-openai/gpt-6-astra"
        assert body["parallel_tool_calls"] is False
        assert {tool["name"] for tool in body["tools"]} == {
            f"{side}_{action}" for side in arms for action in ("move_to", "move_by", "set_gripper")}
        content = body["input"][0]["content"]
        context = json.loads(content[0]["text"])
        assert (context["actual_action_history"], context["current_measured_state"]) == (history, state)
        assert context["current_camera_names"] == ["left_wrist"]
        assert content[1]["text"] == "CURRENT observation 1: left_wrist RGB"
        assert content[2]["image_url"] == "data:image/jpeg;base64,anBlZw=="
        return io.BytesIO(json.dumps(reply(message("夹爪已对齐胶带边缘"), call(tool, {"opening_mm": 5}))).encode())
    monkeypatch.setattr(urllib.request, "urlopen", serve)
    assert model.decide("拿起胶带", arms, observation, state, history) == (tool, {"opening_mm": 5})
    assert model.last_note == "夹爪已对齐胶带边缘"


@pytest.mark.parametrize("bad", [
    reply(call("right_move_to", {})),
    reply(call("left_set_gripper", {"opening_mm": 0}), call("left_set_gripper", {"opening_mm": 85})),
    reply(call("left_move_by", {"translation_m": [.08, .08, 0], "rotation_vector_rad": [0, 0, 0], "duration_s": 4})),
    reply(call("left_move_by", {"translation_m": [0, 0, .01], "rotation_vector_rad": [.5, .5, 0], "duration_s": 4})),
    reply(call("left_move_to", {"position_m": [1, 0, 0], "quaternion_xyzw": [1, 1, 0, 0], "duration_s": 4})),
    reply(call("left_set_gripper", {"opening_mm": False})),
    reply(call("left_set_gripper", {"opening_mm": float("nan")})),
    reply(call("left_set_gripper", {"opening_mm": 86})),
    reply(call("left_set_gripper", {"opening_mm": 5, "force": 100})),
    reply(message('{"status":"done","reason":""}')),
    reply(message('{"status":"done","reason":"ok"}'), call("left_set_gripper", {"opening_mm": 30})),
    reply({"type": "message", "content": [{"type": "refusal", "refusal": "no"}]},
          call("left_set_gripper", {"opening_mm": 20})),
    {**reply(call("left_set_gripper", {"opening_mm": 20})), "status": "incomplete"},
])
def test_rejects_incomplete_ambiguous_or_out_of_contract_decisions(monkeypatch, model, bad):
    monkeypatch.setattr(model, "_post", lambda body: bad)
    with pytest.raises(ValueError):
        model.decide("lift", ("left",), {"head": b"now"}, {}, [])


@pytest.mark.parametrize("returned,expected", [("gpt-6-astra", True), ("different/gpt-6-astra", False), (None, False)])
def test_gateway_model_id_must_match_configured_name(monkeypatch, model, returned, expected):
    response = reply(call("left_set_gripper", {"opening_mm": 10}))
    response["model"] = returned
    monkeypatch.setattr(model, "_post", lambda body: response)
    if expected:
        assert model.decide("close", ("left",), {"head": b"now"}, {}, [])[0] == "left_set_gripper"
        assert model.last_model == returned
    else:
        with pytest.raises(ValueError, match="model_name"):
            model.decide("close", ("left",), {"head": b"now"}, {}, [])


def test_review_uses_current_and_historical_rgb_without_motion_tools(monkeypatch, model):
    captures = []
    def send(payload):
        captures.append(payload)
        return reply(message('{"status":"observe","reason":"胶带仍可能在桌面"}'))
    monkeypatch.setattr(model, "_post", send)
    for i in range(10):
        model.decide("lift", ("left",), {"head": str(i).encode()}, {}, [])
    assert model.review("lift", ("left",), {"head": b"now"}, {}, []) == ("observe", {"reason": "胶带仍可能在桌面"})
    body = captures[-1]
    assert body["tools"] == []
    captions = [part["text"] for part in body["input"][0]["content"][1:] if part["type"] == "input_text"]
    assert captions[0] == "HISTORICAL observation 1: head RGB"
    assert "HISTORICAL observation 2: head RGB" not in captions
    assert captions[-1] == "CURRENT observation 11: head RGB"
    monkeypatch.setattr(model, "_post", lambda payload: reply(call("left_set_gripper", {"opening_mm": 0})))
    with pytest.raises(ValueError):
        model.review("lift", ("left",), {"head": b"now"}, {}, [])


def test_http_error_does_not_expose_provider_body(monkeypatch, model):
    def failed(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "test-key", {}, io.BytesIO(b"test-key"))
    monkeypatch.setattr(urllib.request, "urlopen", failed)
    with pytest.raises(RuntimeError, match="HTTP 401") as exc:
        model.decide("lift", ("left",), {"head": b"now"}, {}, [])
    assert "test-key" not in str(exc.value)
