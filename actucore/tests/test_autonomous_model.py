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


def test_review_keeps_gripper_and_motion_evidence_after_recent_images_expire(monkeypatch, model):
    payloads = []
    def send(payload):
        payloads.append(payload)
        return reply(message('{"status":"observe","reason":"inspect"}'))
    monkeypatch.setattr(model, "_post", send)
    history = []
    xyz = [0., 0., 0.]
    width = 85
    actions = {
        2: ("set_gripper", {"opening_mm": 5}),
        3: ("move_by", {"position_m": [0., 0., .03]}),
        5: ("move_by", {"position_m": [0., .05, .03]}),
        7: ("set_gripper", {"opening_mm": 80}),
        8: ("move_by", {"position_m": [.02, .09, .09]}),  # post-opening withdrawal
    }
    for step in range(1, 21):
        state = {"arms": {"left": {"gripper": {"opening_mm": width}, "eef": {"position_m": xyz}}}}
        model.decide("relocate", ["left"], {"head": bytes([step])}, state, history)
        entry = {"step": step, "tool": "observe"}
        if step in actions:
            tool, measured = actions[step]
            entry.update(tool="left_" + tool, feedback={"actual": measured, "status": "completed",
                                                       "motion_sent": True})
            xyz = measured.get("position_m", xyz)
            width = measured.get("opening_mm", width)
        if step == 10:
            entry.update(tool="left_set_gripper", feedback={"actual": {"opening_mm": 0},
                         "status": "rejected", "motion_sent": False})
        history.append(entry)
    model.review("relocate", ["left"], {"head": b"fresh"}, state, history)
    parts = payloads[-1]["input"][0]["content"]
    events = json.loads(parts[0]["text"])["measured_event_observations"]
    actual = {e["measured_event"]: (e["before_observation"], e["after_observation"]) for e in events}
    assert actual == {"jaws_narrowed": (2, 3), "highest_measured_eef": (3, 4),
                      "furthest_measured_eef_xy": (5, 6), "jaws_widened": (7, 8)}
    labels = [p["text"] for p in parts[1:] if p["type"] == "input_text"]
    assert len(labels) <= 10
    for step in range(1, 9):
        assert f"HISTORICAL observation {step}: head RGB" in labels
    assert labels[-1] == "CURRENT observation 21: head RGB"


def test_observe_two_native_regions_with_fresh_full_views(monkeypatch, model):
    import base64
    import hashlib
    from PIL import Image
    from PIL import ImageChops

    def gradient_jpeg(color):
        source = Image.new("RGB", (80, 60))
        source.putdata([(x * 3, y * 4, color) for y in range(60) for x in range(80)])
        stream = io.BytesIO()
        source.save(stream, format="JPEG")
        return stream.getvalue()
    initial = {"head": gradient_jpeg(0), "left_wrist": gradient_jpeg(50)}
    fresh = {"head": gradient_jpeg(150), "left_wrist": gradient_jpeg(210)}
    crops = [{"camera": "head", "bbox": [.25, .25, .75, .75]},
             {"camera": "left_wrist", "bbox": [0., 0., .5, .5]}]
    request = {"status": "observe", "reason": "查看间隙与桌面净空", "regions": crops}
    replies = iter([reply(message(json.dumps(request))), reply(message('{"status":"observe","reason":"继续观察"}'))])
    payloads = []
    def post(payload):
        payloads.append(payload)
        return next(replies)
    monkeypatch.setattr(model, "_post", post)
    state = {"arms": {"left": {"gripper": {"opening_mm": 58.8}}}}
    decision, args = model.decide("move", ["left"], initial, state, [])
    assert decision == "observe" and args["reason"] == request["reason"]
    assert [m["bbox_pixels"] for m in args["crops"]] == [[20, 15, 60, 45], [0, 0, 40, 30]]
    model.decide("move", ["left"], fresh, state, [])
    for payload in payloads:
        assert {t["name"] for t in payload["tools"]} == {"left_move_to", "left_move_by", "left_set_gripper"}
    parts = payloads[-1]["input"][0]["content"]
    historical = [i for i, p in enumerate(parts) if p.get("text", "").startswith("HISTORICAL native crop")]
    assert len(historical) == 2
    for index, meta in zip(historical, args["crops"]):
        assert parts[index]["text"] == "HISTORICAL native crop " + json.dumps(meta)
        assert meta["source_observation"] == 1 and meta["source_dimensions"] == [80, 60]
        jpeg = initial[meta["camera"]]
        assert meta["source_sha256"] == hashlib.sha256(jpeg).hexdigest()
        mime, data = parts[index+1]["image_url"].split(",")
        assert mime == "data:image/png;base64"
        with Image.open(io.BytesIO(base64.b64decode(data))) as actual:
            expected = Image.open(io.BytesIO(jpeg)).crop(meta["bbox_pixels"])
            assert actual.size == expected.size
            assert ImageChops.difference(actual, expected).getbbox() is None
    for role, jpeg in fresh.items():
        i = next(i for i, p in enumerate(parts) if p.get("text") == f"CURRENT observation 2: {role} RGB")
        assert base64.b64decode(parts[i+1]["image_url"].split(",")[1]) == jpeg
    assert model._remaining_crop_requests == 3


def test_invalid_region_is_a_bounded_readonly_observation(monkeypatch, model):
    request = {"status": "observe", "reason": "检查夹口", "regions": [
        {"camera": "unwired", "bbox": [0, 0, 1, 1]}]}
    monkeypatch.setattr(model, "_post", lambda body: reply(message(json.dumps(request))))
    for step in range(5):
        name, args = model.decide("move", ["left"], {"left_wrist": b"frame"}, {}, [])
        assert name == "observe" and not model._pending_crops
        expected = "wired camera" if step < 4 else "budget spent"
        assert expected in args["crop_error"]


def test_transient_model_retry_keeps_crop_source_step_and_takes_new_images(monkeypatch, model):
    from PIL import Image
    from plugins.autonomous_task.model import TransientModelError

    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(stream, format="JPEG")
    request = {"status": "observe", "reason": "检查蓝色区域",
               "regions": [{"camera": "head", "bbox": [0, 0, 1, 1]}]}
    payloads = []
    def post(body):
        payloads.append(body)
        if len(payloads) == 1:
            return reply(message(json.dumps(request)))
        if len(payloads) == 2:
            raise TransientModelError("timeout")
        return reply(message('{"status":"observe","reason":"核查最新全景"}'))
    monkeypatch.setattr(model, "_post", post)
    model.decide("inspect", ["left"], {"head": stream.getvalue()}, {}, [])
    with pytest.raises(TransientModelError):
        model.decide("inspect", ["left"], {"head": b"second"}, {}, [])
    model.decide("inspect", ["left"], {"head": b"third"}, {}, [])
    for index, payload in enumerate(payloads[1:], start=2):
        texts = [p.get("text", "") for p in payload["input"][0]["content"]]
        detail, = [s for s in texts if s.startswith("HISTORICAL native crop ")]
        assert json.loads(detail.removeprefix("HISTORICAL native crop "))["source_observation"] == 1
        assert f"CURRENT observation {index}: head RGB" in texts


def test_settled_grip_short_of_nominal_opening_allows_visual_test_lift(monkeypatch, model):
    poses = {"position_m": [0.4, 0.2, -0.2]}
    initial = {"arms": {"left": {"gripper": {"opening_mm": 60.}, "eef": poses}}}
    after = {"arms": {"left": {"gripper": {"opening_mm": 8.}, "eef": poses}}}
    entry = {"step": 1, "tool": "left_set_gripper", "args": {"opening_mm": 0},
             "feedback": {"status": "completed", "motion_sent": True,
                          "actual": {"opening_mm": 8.}, "target_opening_mm": 0}}
    payloads = []
    def post(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return reply(call("left_set_gripper", {"opening_mm": 0}))
        return reply(call("left_move_by", {"translation_m": [0, 0, .01],
                                          "rotation_vector_rad": [0, 0, 0], "duration_s": 3}))
    monkeypatch.setattr(model, "_post", post)
    assert model.decide("grasp", ["left"], {"head": b"before"}, initial, [])[0] == "left_set_gripper"
    assert model.decide("grasp", ["left"], {"head": b"after"}, after, [entry])[0] == "left_move_by"
    content = payloads[-1]["input"][0]["content"]
    context = json.loads(content[0]["text"])
    assert context["current_measured_state"] == after and context["actual_action_history"] == [entry]
    assert content[-2]["text"] == "CURRENT observation 2: head RGB"
