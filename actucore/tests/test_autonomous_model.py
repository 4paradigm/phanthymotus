"""Check the model boundary without a network or a ROS installation."""

import io
import json
import pathlib
import sys
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from plugins.autonomous_task.model import InvalidDecisionError, Model


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
                                  "tip_frames": {"left": "left_tool", "right": "right_tool"},
                                  "opening_m": [0., .115]}, **values}


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
            f"{side}_{action}" for side in arms for action in ("move_to", "move_by", "set_gripper")}
        for spec in body["tools"]:
            props = spec["parameters"]["properties"]
            assert "duration_s" not in props
            if spec["name"].endswith("_set_gripper"):
                assert props["opening_mm"]["maximum"] == 115
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
    reply(call("right_move_to", {})),
    reply(call("left_set_gripper", {"opening_mm": 0}), call("left_set_gripper", {"opening_mm": 85})),
    reply(call("left_move_by", {"translation_m": [.08, .08, 0], "rotation_vector_rad": [0, 0, 0]})),
    reply(call("left_move_by", {"translation_m": [0, 0, .01], "rotation_vector_rad": [.5, .5, 0]})),
    reply(call("left_move_to", {"position_m": [1, 0, 0], "quaternion_xyzw": [1, 1, 0, 0]})),
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
    with pytest.raises(RuntimeError, match="HTTP 401") as exc:
        model.decide("lift", ("left",), {"head": b"now"}, measured(), [])
    assert "test-key" not in str(exc.value)


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
    labels = [p["text"] for p in parts[1:] if p["type"] == "input_text"]
    assert len(labels) <= 10
    for step in {1} | {i for pair in expected.values() for i in pair}:
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
    state = measured(arms={"left": {"gripper": {"opening_mm": 58.8}}})
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
    model.decide("inspect", ["left"], {"head": stream.getvalue()}, measured(), [])
    with pytest.raises(TransientModelError):
        model.decide("inspect", ["left"], {"head": b"second"}, measured(), [])
    model.decide("inspect", ["left"], {"head": b"third"}, measured(), [])
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
                "args": {"translation_m": delta, "rotation_vector_rad": [0, 0, 0]},
                "feedback": {"status": "no_motion_observed", "actual": tcp, "target_reached": False,
                             "motion_sent": True, "displacement_observed": False}}
               for step, delta in enumerate(([.02, 0, 0], [0, .015, 0]), 1)]
    new_pose = {"translation_m": [0, 0, 0], "rotation_vector_rad": [0, .04, 0]}
    def send(payload):
        context = json.loads(payload["input"][0]["content"][0]["text"])
        assert context["actual_action_history"] == history
        assert context["current_measured_state"] == state
        assert "slightly change TCP orientation" in payload["instructions"]
        assert "outcome_unknown" in payload["instructions"]
        return reply(call("left_move_by", new_pose))
    monkeypatch.setattr(model, "_post", send)
    assert model.decide("approach", ["left"], {"head": b"current"}, state, history) == ("left_move_by", new_pose)
