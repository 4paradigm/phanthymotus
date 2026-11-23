"""Check the model boundary without a network or a ROS installation."""

import base64
import io
import json
import pathlib
import sys
import urllib.error
import urllib.request
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from plugins.autonomous_task.model import InvalidDecisionError, Model, TransientModelError


def message(text):
    return {"type": "message", "status": "completed",
            "content": [{"type": "output_text", "text": text}]}


def call(name, arguments):
    return {"type": "function_call", "call_id": "call-1", "status": "completed",
            "name": name, "arguments": json.dumps(arguments)}


def reply(*items):
    return {"model": "azure-openai/gpt-6-astra", "status": "completed", "output": list(items)}


def measured(**values):
    return {"control_interface": {"frame": "fixture_base",
                                  "tip_frames": {"left": "left_tool", "right": "right_tool"}},
            "opening_range_mm_by_arm": {"left": [0., 115.], "right": [0., 110.]}, **values}


@pytest.fixture
def model():
    return Model("https://example.test/v1/", "test-key", "azure-openai/gpt-6-astra")


@pytest.mark.parametrize("arms", [("left",), ("right",), ("left", "right")])
def test_responses_request_uses_wired_rgb_state_history_and_selected_arm_tools(monkeypatch, model, arms):
    tool = arms[-1] + "_set_gripper"
    observation = {"left_wrist": b"jpeg"}
    state = measured(arms={side: {"eef": {"frame": "fixture_base", "eef_frame": side + "_tool",
                                          "reference_label": "TCP", "position_m": [0.4, .2, .3],
                                          "quaternion_xyzw": [0., 0., 0., 1.]},
                                  "gripper": {"opening_mm": 30}} for side in arms},
                     frame_description="Up is negative X")
    history = [{"action": "observe", "reason": "确认夹爪位置"}]
    def serve(request, timeout):
        assert request.full_url == "https://example.test/v1/responses"
        assert request.get_header("Authorization") == "Bearer test-key"
        assert timeout == 180
        body = json.loads(request.data)
        assert body["model"] == "azure-openai/gpt-6-astra"
        assert body["parallel_tool_calls"] is False
        assert {tool["name"] for tool in body["tools"]} == {
            f"{side}_{action}" for side in arms for action in ("move_by", "set_gripper")}
        for spec in body["tools"]:
            props = spec["parameters"]["properties"]
            assert "duration_s" not in props
            if spec["name"].endswith("_set_gripper"):
                assert props["opening_mm"]["maximum"] == (115 if spec["name"].startswith("left_") else 110)
                assert props["opening_mm"]["minimum"] == 0
        content = body["input"][0]["content"]
        context = json.loads(content[0]["text"])
        assert (context["actual_action_history"], context["current_measured_state"]) == (history, state)
        assert context["current_camera_names"] == ["left_wrist"]
        assert content[1]["text"] == "CURRENT observation 1: left_wrist RGB"
        assert content[2]["image_url"] == "data:image/jpeg;base64,anBlZw=="
        return io.BytesIO(json.dumps(reply(message("夹爪已对齐胶带边缘"), call(tool, {"opening_mm": 100}))).encode())
    monkeypatch.setattr(urllib.request, "urlopen", serve)
    assert model.decide("拿起胶带", arms, observation, state, history) == (tool, {"opening_mm": 100})
    assert model.last_note == "夹爪已对齐胶带边缘"


@pytest.mark.parametrize("bad", [
    reply(call("right_move_by", {"translation_m": [.01, 0, 0], "rotation_vector_rad": [0, 0, 0]})),
    reply(call("left_set_gripper", {"opening_mm": 0}), call("left_set_gripper", {"opening_mm": 85})),
    reply(call("left_move_by", {"translation_m": [.08, .08, 0], "rotation_vector_rad": [0, 0, 0]})),
    reply(call("left_move_by", {"translation_m": [0, 0, .01], "rotation_vector_rad": [.5, .5, 0]})),
    reply(call("left_move_by", {"translation_m": [0, 0, .01], "rotation_vector_rad": [0, 0, 0], "duration_s": 4})),
    reply(call("left_set_gripper", {"opening_mm": False})),
    reply(call("left_set_gripper", {"opening_mm": float("nan")})),
    reply(call("left_set_gripper", {"opening_mm": 116})),
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
        model.decide("lift", ("left",), {"head": b"now"}, measured(), [])


@pytest.mark.parametrize("side", ["left", "right"])
def test_rejects_retired_absolute_tool_even_on_an_enabled_arm(monkeypatch, model, side):
    pose = {"position_m": [.11, .23, .34], "quaternion_xyzw": [0, 0, 0, 1]}
    monkeypatch.setattr(model, "_post", lambda body: reply(call(f"{side}_move_to", pose)))
    with pytest.raises(ValueError, match="tools offered"):
        model.decide("lift", ("left", "right"), {"head": b"now"}, measured(), [])


def test_truncated_tool_call_requires_reobservation_even_if_arguments_look_complete(monkeypatch, model):
    response = {**reply(call("left_set_gripper", {"opening_mm": 20})), "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"}}
    monkeypatch.setattr(model, "_post", lambda _: response)
    with pytest.raises(InvalidDecisionError, match="max_output_tokens"):
        model.decide("grasp", ["left"], {"head": b"now"}, measured(), [])
    response["incomplete_details"]["reason"] = "content_filter"
    with pytest.raises(ValueError, match="content_filter") as exc:
        model.decide("grasp", ["left"], {"head": b"after"}, measured(), [])
    assert not isinstance(exc.value, InvalidDecisionError)


@pytest.mark.parametrize("decoration", ["{}", "```json\n{}\n```", "```\n{}\n```"])
def test_single_json_status_can_be_bare_or_fenced(monkeypatch, model, decoration):
    status = {"status": "observe", "reason": "检查夹爪位置"}
    monkeypatch.setattr(model, "_post", lambda body: reply(message(decoration.format(json.dumps(status)))))
    assert model.decide("inspect", ["left"], {"head": b"now"}, measured(), []) == (
        "observe", {"reason": "检查夹爪位置"})


@pytest.mark.parametrize("text", [
    '先观察\n{"status":"done","reason":"ok"}',
    '```json\n{"status":"done","reason":"ok"}\n```\n补充说明',
    '```json\n{"status":"done","reason":"ok"}\n```\n```json\n{}\n```',
    '{"name":"left_set_gripper","arguments":{"opening_mm":30}}',
    '下一步我将执行 left_move_by(0.05, 0, 0)。' * 100,
])
def test_unparseable_decision_preserves_bounded_visible_text_and_reobserves(monkeypatch, model, text):
    payloads = []
    def send(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return reply(message(text))
        return reply(message('{"status":"observe","reason":"检查当前画面"}'))
    monkeypatch.setattr(model, "_post", send)
    with pytest.raises(InvalidDecisionError) as exc:
        model.decide("inspect", ["left"], {"head": b"before"}, measured(), [])
    assert exc.value.visible_output == text[:1200]
    history = [{"tool": "invalid_decision", "note": exc.value.visible_output}]
    model.decide("inspect", ["left"], {"head": b"after"}, measured(), history)
    assert "Your previous reply could not be parsed" in payloads[1]["instructions"]
    assert payloads[1]["input"][0]["content"][-2]["text"] == "CURRENT observation 2: head RGB"


def test_fenced_status_or_refusal_alongside_real_call_prevents_execution(monkeypatch, model):
    motion = call("left_set_gripper", {"opening_mm": 3})
    for companion, error in [
        (message('检查\n```json\n{"status":"observe","reason":"等候"}\n```'), "Ambiguous response"),
        ({"type": "message", "content": [{"type": "refusal", "refusal": "拒绝原因"}]}, "拒绝原因"),
    ]:
        monkeypatch.setattr(model, "_post", lambda body: reply(motion, companion))
        with pytest.raises(ValueError, match=error) as exc:
            model.decide("grasp", ["left"], {"head": b"now"}, measured(), [])
        assert not isinstance(exc.value, InvalidDecisionError)


@pytest.mark.parametrize("returned,expected", [("gpt-6-astra", True), ("different/gpt-6-astra", False), (None, False)])
def test_gateway_model_id_must_match_configured_name(monkeypatch, model, returned, expected):
    response = reply(call("left_set_gripper", {"opening_mm": 10}))
    response["model"] = returned
    monkeypatch.setattr(model, "_post", lambda body: response)
    if expected:
        assert model.decide("close", ("left",), {"head": b"now"}, measured(), [])[0] == "left_set_gripper"
        assert model.last_model == returned
    else:
        with pytest.raises(ValueError, match="model_name"):
            model.decide("close", ("left",), {"head": b"now"}, measured(), [])


def test_review_uses_current_and_historical_rgb_without_motion_tools(monkeypatch, model):
    captures = []
    def send(payload):
        captures.append(payload)
        return reply(message('{"status":"observe","reason":"胶带仍可能在桌面"}'))
    monkeypatch.setattr(model, "_post", send)
    for i in range(10):
        model.decide("lift", ("left",), {"head": str(i).encode()}, measured(), [])
    assert model.review("lift", ("left",), {"head": b"now"}, measured(), []) == ("observe", {"reason": "胶带仍可能在桌面"})
    body = captures[-1]
    assert body["tools"] == []
    captions = [part["text"] for part in body["input"][0]["content"][1:] if part["type"] == "input_text"]
    assert captions[0] == "HISTORICAL observation 1: head RGB"
    assert "HISTORICAL observation 2: head RGB" not in captions
    assert captions[-1] == "CURRENT observation 11: head RGB"
    monkeypatch.setattr(model, "_post", lambda payload: reply(call("left_set_gripper", {"opening_mm": 0})))
    with pytest.raises(ValueError):
        model.review("lift", ("left",), {"head": b"now"}, measured(), [])


def test_http_error_does_not_expose_provider_body(monkeypatch, model):
    def failed(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "test-key", {}, io.BytesIO(b"test-key"))
    monkeypatch.setattr(urllib.request, "urlopen", failed)
    model.model_log = Mock()
    with pytest.raises(RuntimeError, match="HTTP 401") as exc:
        model.decide("lift", ("left",), {"head": b"now"}, measured(), [])
    assert "test-key" not in str(exc.value)
    request_event, error_event = model.model_log.append.call_args_list
    assert b"test-key" not in request_event.kwargs["raw_body"]
    assert error_event.kwargs["provider_response"]["body"] == "[REDACTED]"


def test_model_log_matches_actual_multimodal_requests_and_unparsed_responses(monkeypatch, model):
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(stream, format="JPEG")
    jpeg = stream.getvalue()
    responses = [
        {**reply(call("left_set_gripper", {"opening_mm": 18})), "id": "resp_1",
         "usage": {"output_tokens": 42, "output_tokens_details": {"reasoning_tokens": 12}}},
        {**reply(message("原始回复")), "id": "resp_2", "model": "wrong/model",
         "provider_metadata": {"custom": ["完整保留", 3]}},
    ]
    model.model_log = Mock()
    wire_requests = []
    wire_responses = [("\n" + json.dumps(v, ensure_ascii=False, indent=2) + " \n").encode() for v in responses]
    wire_responses[0] = wire_responses[0].replace(b'"output_tokens": 42', b'"output_tokens": 42.0e0')
    def serve(request, timeout):
        index = len(wire_requests)
        body = json.loads(request.data)
        wire_requests.append(request.data)
        calls = model.model_log.append.call_args_list
        assert len(calls) == 2 * index + 1
        assert calls[-1].args == ("request",) and calls[-1].kwargs["payload"] == body
        assert calls[-1].kwargs["raw_body"] is request.data
        return io.BytesIO(wire_responses[index])
    monkeypatch.setattr(urllib.request, "urlopen", serve)
    assert model.decide("抓取", ["left"], {"head": jpeg}, measured(), [])[0] == "left_set_gripper"
    with pytest.raises(ValueError, match="model_name"):
        model.review("抓取", ["left"], {"left_wrist": jpeg}, measured(), [])
    for i, phase in enumerate(("decide", "review")):
        request_event, response_event = model.model_log.append.call_args_list[2*i:2*i+2]
        assert request_event.kwargs == {"step": i+1, "phase": phase, "payload": json.loads(wire_requests[i]),
                                        "raw_body": wire_requests[i]}
        assert response_event.args == ("response",)
        assert response_event.kwargs == {"step": i+1, "phase": phase, "payload": responses[i],
                                         "raw_body": wire_responses[i]}
        images = [part["image_url"] for part in json.loads(wire_requests[i])["input"][0]["content"]
                  if part["type"] == "input_image"]
        assert len(images) == i + 1  # Review contains both the initial and fresh observation.
        for image in images:
            prefix, data = image.split(",")
            assert prefix == "data:image/jpeg;base64" and base64.b64decode(data, validate=True) == jpeg
    assert len(model.model_log.append.call_args_list) == 4


@pytest.mark.parametrize("case", ["rate_limit", "invalid_json", "binary", "connection"])
def test_model_log_records_one_error_with_actual_http_body_when_available(monkeypatch, model, case):
    rate_body = b'{"error": {"code":"busy", "message":"retry"}}'
    error_body = b"<!doctype html>gateway unavailable"
    binary_body = b"\xff\xfe\x00"
    def serve(request, timeout):
        if case == "connection":
            raise urllib.error.URLError("unavailable")
        if case == "rate_limit":
            raise urllib.error.HTTPError(request.full_url, 429, "busy", {"X-Test": "private header"},
                                         io.BytesIO(rate_body))
        source = io.BytesIO(binary_body if case == "binary" else error_body)
        source.status = 200
        return source
    monkeypatch.setattr(urllib.request, "urlopen", serve)
    model.model_log = Mock()
    expected = TransientModelError if case in ("rate_limit", "connection") else ValueError
    with pytest.raises(expected) as exc:
        model.decide("inspect", ["left"], {"head": b"frame"}, measured(), [])
    request_event, *responses, error_event = model.model_log.append.call_args_list
    assert request_event.args == ("request",) and error_event.args == ("error",)
    fields = error_event.kwargs
    assert (fields["phase"], fields["step"]) == ("decide", 1)
    assert fields["error"] == {"type": type(exc.value).__name__, "message": str(exc.value)}
    if case == "rate_limit":
        response, = responses
        assert response.args == ("response",)
        assert response.kwargs == {"step": 1, "phase": "decide", "payload": json.loads(rate_body), "raw_body": rate_body}
        assert fields["provider_response"] == {"http_status": 429}
        return
    assert not responses
    if case == "connection":
        assert "provider_response" not in fields
    else:
        status, body, encoding = {
            "invalid_json": (200, error_body.decode(), "utf-8"),
            "binary": (200, base64.b64encode(binary_body).decode(), "base64"),
        }[case]
        assert fields["provider_response"] == {"http_status": status, "body": body, "body_encoding": encoding}


@pytest.mark.parametrize("body,headers,expected", [
    (b'{"error":"test\\u002dkey"}', {}, "redacted"),
    (b'{"retry":true}', {"Content-Length": "110"}, "partial"),
    (b'{}' + b' ' * 40, {}, "partial"),
    (b'{"invalid":NaN}', {}, "invalid"),
    (b'null', {}, "null"),
])
def test_http_error_response_file_requires_a_complete_json_body(monkeypatch, model, body, headers, expected):
    monkeypatch.setattr("plugins.autonomous_task.model._MAX_RESPONSE_BYTES", 32)
    def serve(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "busy", headers, io.BytesIO(body))
    monkeypatch.setattr(urllib.request, "urlopen", serve)
    model.model_log = Mock()
    with pytest.raises(TransientModelError):
        model.decide("inspect", ["left"], {"head": b"frame"}, measured(), [])
    _, *responses, error = model.model_log.append.call_args_list
    details = error.kwargs["provider_response"]
    if expected == "partial":
        assert not responses and details["truncated"] is True
    elif expected == "invalid":
        assert not responses and details["body"] == body.decode()
    else:
        response, = responses
        assert response.args == ("response",)
        if expected == "null":
            assert response.kwargs["payload"] is None and response.kwargs["raw_body"] == body
        else:
            assert details["credential_redacted"] is True
            assert json.loads(response.kwargs["raw_body"]) == response.kwargs["payload"] == {"error": "[REDACTED]"}


@pytest.mark.parametrize("retry_after_opening", [False, True])
def test_review_keeps_gripper_and_motion_evidence_after_recent_images_expire(monkeypatch, model, retry_after_opening):
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
        3: ("move_by", {"position_m": [-.03, 0., 0.]}),
        4: ("set_gripper", {"opening_mm": 3}),  # tighten during the same hold
        5: ("move_by", {"position_m": [-.03, .05, 0.]}),
        7: ("set_gripper", {"opening_mm": 80}),
        8: ("move_by", {"position_m": [.02, .09, .09]}),  # post-opening withdrawal
    }
    if retry_after_opening:
        actions.update({
            10: ("set_gripper", {"opening_mm": 12}),  # start a fresh grasp
            11: ("move_by", {"position_m": [.02, .09, .05]}),
            12: ("move_by", {"position_m": [.07, .09, .05]}),
            13: ("set_gripper", {"opening_mm": 70}),
        })
    for step in range(1, 21):
        state = measured(arms={"left": {"gripper": {"opening_mm": width}, "eef": {"position_m": xyz}}})
        model.decide("relocate", ["left"], {"head": bytes([step])}, state, history)
        entry = {"step": step, "tool": "observe"}
        if step in actions:
            tool, actual = actions[step]
            entry.update(tool="left_" + tool, feedback={"actual": actual, "status": "completed",
                                                       "motion_sent": True})
            xyz = actual.get("position_m", xyz)
            width = actual.get("opening_mm", width)
        if step == 16:
            entry.update(tool="left_set_gripper", feedback={"actual": {"opening_mm": 0},
                         "status": "rejected", "motion_sent": False})
        history.append(entry)
    model.review("relocate", ["left"], {"head": b"fresh"}, state, history)
    parts = payloads[-1]["input"][0]["content"]
    events = json.loads(parts[0]["text"])["measured_event_observations"]
    actual = {e["measured_event"]: (e["before_observation"], e["after_observation"]) for e in events}
    if retry_after_opening:
        expected = {"jaws_narrowed": (10, 11), "first_measured_tcp_translation": (11, 12),
                    "furthest_measured_tcp": (12, 13), "jaws_widened": (13, 14)}
    else:
        expected = {"jaws_narrowed": (2, 3), "first_measured_tcp_translation": (3, 4),
                    "furthest_measured_tcp": (5, 6), "jaws_widened": (7, 8)}
    assert actual == expected
    normal = payloads[-2]["input"][0]["content"]
    evidence = json.loads(normal[0]["text"])["measured_event_observations"]
    assert {e["measured_event"]: (e["before_observation"], e["after_observation"]) for e in evidence} == {
        name: expected[name] for name in ("jaws_narrowed", "first_measured_tcp_translation")}
    assert all(e["camera"] == "head" for e in evidence)
    for event in evidence:
        for key in ("before_observation", "after_observation"):
            assert {"type": "input_text", "text": f"HISTORICAL observation {event[key]}: head RGB"} in normal
    labels = [p["text"] for p in parts[1:] if p["type"] == "input_text"]
    assert len(labels) <= 10
    for step in {1} | {i for pair in expected.values() for i in pair}:
        assert f"HISTORICAL observation {step}: head RGB" in labels
    assert labels[-1] == "CURRENT observation 21: head RGB"


@pytest.mark.parametrize("cameras,expected_left,expected_right", [
    (("head", "left_wrist", "right_wrist"), "left_wrist", "right_wrist"),
    (("head", "left_wrist"), "left_wrist", "head"),
    (("right_wrist",), "right_wrist", "right_wrist"),
])
def test_ordinary_decision_pairs_grasp_images_with_their_actual_arm_and_camera(
        monkeypatch, model, cameras, expected_left, expected_right):
    import base64

    payloads = []
    def post(payload):
        payloads.append(payload)
        return reply(message('{"status":"observe","reason":"compare visible contact surfaces"}'))
    monkeypatch.setattr(model, "_post", post)
    arms = ("left", "right")
    history = []
    states = {a: {"gripper": {"opening_mm": 60.}, "eef": {"position_m": [0., 0., 0.]}} for a in arms}
    actions = {3: ("left", "set_gripper", {"opening_mm": 8.}),
               4: ("left", "move_by", {"position_m": [0., .015, 0.]}),
               6: ("right", "set_gripper", {"opening_mm": 10.}),
               8: ("right", "move_by", {"position_m": [.02, 0., 0.]})}
    for step in range(1, 16):
        views = {c: f"{step}-{c}".encode() for c in cameras}
        model.decide("inspect grasp", arms, views, measured(arms=states), history)
        item = {"step": step, "tool": "observe"}
        if step in actions:
            arm, tool, actual = actions[step]
            item.update(tool=f"{arm}_{tool}", feedback={"actual": actual, "status": "completed", "motion_sent": True})
            states = {**states, arm: {**states[arm], "gripper" if tool == "set_gripper" else "eef": actual}}
        history.append(item)

    parts = payloads[-1]["input"][0]["content"]
    evidence = json.loads(parts[0]["text"])["measured_event_observations"]
    assert [(e["arm"], e["measured_event"], e["camera"], e["before_observation"], e["after_observation"])
            for e in evidence] == [
        ("left", "jaws_narrowed", expected_left, 3, 4),
        ("left", "first_measured_tcp_translation", expected_left, 4, 5),
        ("right", "jaws_narrowed", expected_right, 6, 7),
        ("right", "first_measured_tcp_translation", expected_right, 8, 9)]
    captions = [p["text"] for p in parts[1:] if p["type"] == "input_text"]
    assert len(captions) == len(set(captions))
    assert sum(s.startswith("HISTORICAL") for s in captions) <= 3 * len(cameras) + 4 * len(arms)
    for label, image in zip(parts[1::2], parts[2::2]):
        _, _, number, camera, _ = label["text"].split()
        step = int(number.strip(":"))
        assert base64.b64decode(image["image_url"].split(",")[1]) == f"{step}-{camera}".encode()
        if 3 <= step <= 5:
            assert camera == expected_left
        elif 6 <= step <= 9:
            assert camera == expected_right
    assert f"HISTORICAL observation 4: {expected_left} RGB" in captions
    assert len([s for s in captions if s.startswith("CURRENT observation 15:")]) == len(cameras)


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
    state = measured(arms={"left": {"gripper": {"opening_mm": 58.8}}})
    decision, args = model.decide("move", ["left"], initial, state, [])
    assert decision == "observe" and args["reason"] == request["reason"]
    assert [m["bbox_pixels"] for m in args["crops"]] == [[20, 15, 60, 45], [0, 0, 40, 30]]
    model.decide("move", ["left"], fresh, state, [])
    for payload in payloads:
        assert {t["name"] for t in payload["tools"]} == {"left_move_by", "left_set_gripper"}
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
        name, args = model.decide("move", ["left"], {"left_wrist": b"frame"}, measured(), [])
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
    model.model_log = Mock()
    model.decide("inspect", ["left"], {"head": stream.getvalue()}, measured(), [])
    with pytest.raises(TransientModelError):
        model.decide("inspect", ["left"], {"head": b"second"}, measured(), [])
    model.decide("inspect", ["left"], {"head": b"third"}, measured(), [])
    events = model.model_log.append.call_args_list
    assert [(event.args[0], event.kwargs["step"]) for event in events] == [
        ("request", 1), ("response", 1), ("request", 2), ("error", 2), ("request", 3), ("response", 3)]
    assert [event.kwargs["payload"] for event in events[::2]] == payloads
    pngs = [part["image_url"] for part in events[4].kwargs["payload"]["input"][0]["content"]
            if part.get("image_url", "").startswith("data:image/png;base64,")]
    with Image.open(io.BytesIO(base64.b64decode(pngs[0].split(",")[1], validate=True))) as image:
        assert image.format == "PNG" and image.size == (32, 32)
    for index, payload in enumerate(payloads[1:], start=2):
        texts = [p.get("text", "") for p in payload["input"][0]["content"]]
        detail, = [s for s in texts if s.startswith("HISTORICAL native crop ")]
        assert json.loads(detail.removeprefix("HISTORICAL native crop "))["source_observation"] == 1
        assert f"CURRENT observation {index}: head RGB" in texts


def test_settled_grip_short_of_nominal_opening_allows_visual_test_lift(monkeypatch, model):
    poses = {"position_m": [0.4, 0.2, -0.2]}
    initial = measured(arms={"left": {"gripper": {"opening_mm": 60.}, "eef": poses}})
    after = measured(arms={"left": {"gripper": {"opening_mm": 8.}, "eef": poses}})
    entry = {"step": 1, "tool": "left_set_gripper", "args": {"opening_mm": 0},
             "feedback": {"status": "completed", "motion_sent": True,
                          "actual": {"opening_mm": 8.}, "target_opening_mm": 0}}
    payloads = []
    def post(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return reply(call("left_set_gripper", {"opening_mm": 0}))
        return reply(call("left_move_by", {"translation_m": [0, 0, .01],
                                          "rotation_vector_rad": [0, 0, 0]}))
    monkeypatch.setattr(model, "_post", post)
    assert model.decide("grasp", ["left"], {"head": b"before"}, initial, [])[0] == "left_set_gripper"
    assert model.decide("grasp", ["left"], {"head": b"after"}, after, [entry])[0] == "left_move_by"
    content = payloads[-1]["input"][0]["content"]
    context = json.loads(content[0]["text"])
    assert context["current_measured_state"] == after and context["actual_action_history"] == [entry]
    assert content[-2]["text"] == "CURRENT observation 2: head RGB"


def test_published_gripper_target_without_measured_motion_stays_unverified(monkeypatch, model):
    state = measured(arms={"right": {"eef": {"position_m": [.2, .1, .3]},
                                     "gripper": {"opening_mm": 0}}})
    history = [{"step": 1, "tool": "right_set_gripper", "args": {"opening_mm": 3},
                "feedback": {"status": "no_motion_observed", "motion_sent": True,
                             "target_reached": False, "displacement_observed": False,
                             "actual": {"opening_mm": 0}}}]
    sent = []
    def post(payload):
        sent.append(payload)
        if len(sent) == 1:
            return reply(call("right_set_gripper", {"opening_mm": 3}))
        return reply(message('{"status":"observe","reason":"图像及开口均未显示张开"}'))
    monkeypatch.setattr(model, "_post", post)
    model.decide("松开", ["right"], {"right_wrist": b"before"}, state, [])
    name, _ = model.review("松开", ["right"], {"right_wrist": b"after"}, state, history)
    context = json.loads(sent[-1]["input"][0]["content"][0]["text"])
    assert name == "observe" and context["actual_action_history"] == history
    assert context["measured_event_observations"] == []


def test_model_receives_unmoved_targets_and_can_select_a_new_approach_orientation(monkeypatch, model):
    tcp = {"position_m": [.4, .2, .1], "quaternion_xyzw": [0, 0, 0, 1]}
    state = measured(arms={"left": {"eef": tcp, "gripper": {"opening_mm": 30}}})
    history = [{"step": step, "tool": "left_move_by",
                "args": {"translation_m": delta, "rotation_vector_rad": rotation},
                "feedback": {"status": "no_motion_observed", "actual": tcp, "target_reached": False,
                             "motion_sent": True, "displacement_observed": False, "feedback_frames": 26,
                             "command_stamp_ms": 1000 * step, "state_stamp_ms": 1000 * step + 250}}
               for step, (delta, rotation) in enumerate([
                   ([.02, 0, 0], [0, .08, 0]), ([0, 0, 0], [0, .06, 0])], 1)]
    history.append({"step": 3, "tool": "observe"})  # More images, same physical outcome.
    new_pose = {"translation_m": [0, 0, 0], "rotation_vector_rad": [0, -.03, 0]}
    def send(payload):
        context = json.loads(payload["input"][0]["content"][0]["text"])
        assert context["actual_action_history"] == history
        assert context["current_measured_state"] == state
        for entry in history[:2]:
            assert f"Step {entry['step']} left_move_by: {json.dumps(entry['args'])}" in payload["instructions"]
        assert "opposite sign or a smaller" in payload["instructions"]
        assert "outcome_unknown" in payload["instructions"]
        return reply(call("left_move_by", new_pose))
    monkeypatch.setattr(model, "_post", send)
    assert model.decide("approach", ["left"], {"head": b"current"}, state, history) == ("left_move_by", new_pose)


@pytest.mark.parametrize("tool,update,review", [
    ("left_set_gripper", {}, False),
    ("left_move_by", {"status": "outcome_unknown"}, False),
    ("left_move_by", {"motion_sent": False}, False),
    ("left_move_by", {"feedback_frames": 1}, False),
    ("left_move_by", {"state_stamp_ms": 1000}, False),
    ("left_move_by", {"status": "completed", "displacement_observed": True}, False),
    ("left_move_by", {}, True),
])
def test_pose_recovery_hint_requires_a_verified_unmoved_pose_and_an_action_turn(monkeypatch, model, tool, update, review):
    feedback = {"status": "no_motion_observed", "target_reached": False, "motion_sent": True,
                "displacement_observed": False, "feedback_frames": 20, "command_stamp_ms": 1000,
                "state_stamp_ms": 5000}
    history = [{"step": 1, "tool": "left_move_by", "feedback": feedback},
               {"step": 2, "tool": tool, "feedback": {**feedback, **update}}]
    def send(payload):
        assert "opposite sign or a smaller" not in payload["instructions"]
        return reply(message('{"status":"observe","reason":"继续确认净空"}'))
    monkeypatch.setattr(model, "_post", send)
    decide = model.review if review else model.decide
    assert decide("inspect", ["left"], {"head": b"fresh"}, measured(), history)[0] == "observe"
