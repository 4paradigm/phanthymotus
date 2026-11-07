"""Real Perception registry, HTTP/MCP, camera format and concurrency boundaries."""

from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from test_goal_verifier import config, observed
from plugins.goal_verifier.contracts import NAME, Refusal
from plugins.goal_verifier.plugin import GoalVerifierPlugin
from plugins.goal_verifier.schema import GOAL, OBS_SCHEMA, input_schema
from plugins.goal_verifier.server import handler, loads, rpc


def current(sequence, x=0.5):
    o = observed(sequence, x)
    o["captured_at"] = time.time()
    return o


def call(card, action, **args):
    return card.dispatch(NAME, dict(action=action, **args))


def task(card):
    out = call(
        card,
        "start_session",
        request_id=uuid4().hex,
        task_id="SYNTHETIC-TEST",
        source_id="cam",
        selector=dict(class_name="bottle", color="red"),
        goal=dict(
            units="normalized",
            frame_id="optical",
            shape="circle",
            center=[0.75, 0.5],
            radius=0.1,
            tolerance=0,
        ),
    )
    return {k: out[k] for k in ("session_id", "session_token")}


def test_official_perception_bundle_registers_only_opted_in_card():
    import main

    cfg = dict(
        plugins=dict(
            piper_object_goal_verifier={**config(), "enabled": True},
            echo_cancellation=dict(enabled=False),
            audio_mixer=dict(enabled=False),
        )
    )
    bundle = main.PerceptionBundle(cfg, None)
    assert bundle.owns(NAME) and bundle.get_all_tools()[0]["name"] == NAME
    assert bundle.dispatch(NAME, dict(action="info"))["robot_model"] == "agilex_piper"
    assert bundle.dispatch(NAME + "_move", {})["result"] == "INCONCLUSIVE"
    disabled = main.PerceptionBundle(
        dict(
            plugins=dict(
                echo_cancellation=dict(enabled=False), audio_mixer=dict(enabled=False)
            )
        ),
        None,
    )
    assert not disabled.owns(NAME)


def test_advertised_json_schema_and_structural_validation():
    for schema in (input_schema(), GOAL, OBS_SCHEMA):
        Draft202012Validator.check_schema(schema)
    card = GoalVerifierPlugin(config())
    result = call(
        card,
        "start_session",
        request_id="test",
        task_id="test",
        source_id="cam",
        selector={"class_name": "bottle"},
        goal={
            "units": "normalized",
            "frame_id": "optical",
            "shape": "circle",
            "tolerance": 0,
        },
    )
    assert result["result"] == "INCONCLUSIVE" and result["code"] == "ARGUMENT_SCHEMA"
    assert not card.engine.sessions


def test_real_http_mcp_initialization_discovery_errors_and_stop():
    card = GoalVerifierPlugin(config())
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(card))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    try:
        assert rpc(url, "initialize")["protocolVersion"] == "2024-11-05"
        assert rpc(url, "tools/list")["tools"][0]["name"] == NAME
        out = rpc(url, "tools/call", dict(name=NAME, arguments=dict(action="info")))
        assert not json.loads(out["content"][0]["text"])["motion_capability"]
        with pytest.raises(Refusal, match="UNKNOWN_TOOL"):
            rpc(url, "tools/call", dict(name="piper_joint", arguments={}))
        stopped = rpc(url, "tools/call", dict(name=NAME, arguments=dict(action="stop")))
        assert json.loads(stopped["content"][0]["text"])["sessions_retired"]
    finally:
        card.stop()
        server.shutdown()
        server.server_close()
        worker.join(1)


def test_perception_http_does_not_log_session_capabilities(monkeypatch, caplog):
    import logging
    import main

    card = GoalVerifierPlugin(config())
    bundle = main.PerceptionBundle.__new__(main.PerceptionBundle)
    bundle._plugins = [card]
    monkeypatch.setattr(main, "_bundle", bundle)
    caplog.set_level(logging.INFO)
    server = ThreadingHTTPServer(("127.0.0.1", 0), main.make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    try:
        started = rpc(
            url,
            "tools/call",
            dict(
                name=NAME,
                arguments=dict(
                    action="start_session",
                    request_id="start",
                    task_id="SYNTHETIC",
                    source_id="cam",
                    selector=dict(class_name="bottle"),
                    goal=dict(
                        units="normalized",
                        frame_id="optical",
                        shape="circle",
                        center=[0.5, 0.5],
                        radius=0.1,
                        tolerance=0,
                    ),
                ),
            ),
        )
        token = json.loads(started["content"][0]["text"])["session_token"]
        canary = "TEST_CANARY_NOT_A_REAL_SECRET"
        rpc(
            url,
            "tools/call",
            dict(
                name=NAME,
                arguments=dict(
                    action="get_result", session_id="unknown", session_token=canary
                ),
            ),
        )
        assert token not in caplog.text and canary not in caplog.text
        assert "payload omitted" in caplog.text
    finally:
        card.stop()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_duplicate_json_keys_and_nonfinite_are_rejected_before_dispatch():
    for value in (
        '{"camera_id":"a","camera_id":"b"}',
        '{"value":NaN}',
        '{"value":Infinity}',
    ):
        with pytest.raises(Refusal):
            loads(value)


def test_bootstrap_doctor_works_without_any_site_packages(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config()))
    output = tmp_path / "doctor.json"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "perception/tools/goal_verifier_cli.py"),
            "doctor",
            "--config",
            str(cfg),
            "--output",
            str(output),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(root / "perception"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(output.read_text())
    assert report["state"] == "BLOCKED" and report["checks"]["dependencies"] is False
    assert report["checks"]["configuration"] is True
    assert not report["real_visual_sample_checked"]


def test_failed_new_verification_does_not_leave_previous_pass_as_latest():
    card = GoalVerifierPlugin(config())
    s = task(card)
    call(card, "capture_before", **s, request_id="before", observation=current(0))
    marker = call(card, "mark_operation_complete", **s, request_id="mark")
    first = call(
        card,
        "verify_after",
        **s,
        request_id="one",
        marker_id=marker["marker_id"],
        observation=current(1, 0.75),
    )
    assert first["result"] == "PASS"
    bad = current(2)
    bad["captured_at"] -= 30
    failure = call(
        card,
        "verify_after",
        **s,
        request_id="two",
        marker_id=marker["marker_id"],
        observation=bad,
    )
    assert failure["result"] == "INCONCLUSIVE" and not failure["data_valid"]
    assert call(card, "get_result", **s)["result"] == "INCONCLUSIVE"
    evidence = call(card, "export_session", **s)
    assert "session_token" not in evidence and s["session_token"] not in json.dumps(
        evidence
    )
    assert evidence["before"]["observation_id"] == "frame-0"
    assert evidence["observations"][1]["observation_id"] == "frame-1"


def test_concurrent_same_request_is_one_verification():
    card = GoalVerifierPlugin(config())
    s = task(card)
    call(card, "capture_before", **s, request_id="before", observation=current(0))
    marker = call(card, "mark_operation_complete", **s, request_id="mark")
    args = dict(
        **s,
        request_id="one",
        marker_id=marker["marker_id"],
        observation=current(1, 0.75),
    )
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(call(card, "verify_after", **args))
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(1)
    assert len(results) == 4 and all(r == results[0] for r in results)
    assert len(call(card, "export_session", **s)["history"]) == 1


def test_conflicting_retry_preserves_original_verification_receipt():
    card = GoalVerifierPlugin(config())
    s = task(card)
    call(card, "capture_before", **s, request_id="before", observation=current(0))
    marker = call(card, "mark_operation_complete", **s, request_id="mark")
    args = dict(
        **s,
        request_id="after",
        marker_id=marker["marker_id"],
        observation=current(1, 0.75),
    )
    original = call(card, "verify_after", **args)
    assert original["result"] == "PASS"
    conflict = call(card, "verify_after", **{**args, "observation": current(2, 0.5)})
    assert (
        conflict["code"] == "REQUEST_ID_CONFLICT"
        and conflict["result"] == "INCONCLUSIVE"
    )
    assert call(card, "verify_after", **args) == original
    card.stop()


def test_new_source_error_during_detection_cannot_be_cleared_by_older_frame(
    monkeypatch,
):
    from plugins.goal_verifier import sources

    cfg = config()
    cfg["sources"]["cam"].update(
        kind="buffered_image",
        view_id="test-view",
        view_mode="fixed",
        fixed_view_record="SYNTHETIC",
        classes=[
            dict(
                class_name="bottle",
                color="red",
                hsv_min=[0, 150, 150],
                hsv_max=[10, 255, 255],
            )
        ],
    )
    card = GoalVerifierPlugin(cfg)
    card.sources.feed_image(
        "cam", np.zeros((80, 120, 3), np.uint8), time.time(), provenance="synthetic"
    )
    entered, release = threading.Event(), threading.Event()
    original = sources.color_objects

    def paused(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(sources, "color_objects", paused)
    out = []
    worker = threading.Thread(
        target=lambda: out.append(call(card, "observe", source_id="cam"))
    )
    worker.start()
    assert entered.wait(2)
    card.sources.reject("cam", ValueError("newer invalid camera frame"))
    release.set()
    worker.join(3)
    assert not worker.is_alive() and out[0]["code"] == "SOURCE_INPUT_ERROR"
    assert card.sources.errors["cam"] == "newer invalid camera frame"
    assert "cam" not in card.engine.latest
    card.stop()


def test_mid_acquisition_reset_retires_old_request_without_touching_new_session():
    card = GoalVerifierPlugin(config())
    s = task(card)
    call(card, "capture_before", **s, request_id="before", observation=current(0))
    marker = call(card, "mark_operation_complete", **s, request_id="mark")
    entered = threading.Event()
    original = card.sources.read

    def read(*a, **kw):
        entered.set()
        return original(*a, **kw)

    card.sources.read = read
    result = []
    t = threading.Thread(
        target=lambda: result.append(
            call(
                card,
                "verify_after",
                **s,
                request_id="after",
                marker_id=marker["marker_id"],
            )
        )
    )
    t.start()
    assert entered.wait(1)
    new = call(card, "reset_session", **s, request_id="reset")
    card.sources.push(current(1, 0.75))
    t.join(1)
    assert not t.is_alive() and result[0]["code"] == "RETIRED_SESSION"
    assert (
        call(
            card,
            "get_result",
            session_id=new["session_id"],
            session_token=new["session_token"],
        )["code"]
        == "NOT_VERIFIED"
    )


@pytest.mark.parametrize("kind", ["compressed", "raw"])
def test_ros_image_message_decoding_and_original_capture_stamp(kind):
    cfg = config()
    b = cfg["sources"]["cam"]
    b.update(
        kind="ros_image",
        provenance="synthetic",
        stamp_semantics="capture",
        capture_time_record="SYNTHETIC-STAMP-CONTRACT",
        message_type=kind,
        topic="/TEST/ONLY/image",
        domain_id=42,
        view_id="operator-declared-view",
        view_mode="fixed",
        fixed_view_record="SYNTHETIC-SETUP-TEST",
        classes=[
            dict(
                class_name="bottle",
                color="red",
                hsv_min=[0, 150, 150],
                hsv_max=[10, 255, 255],
            )
        ],
    )
    card = GoalVerifierPlugin(cfg)
    image = np.zeros((80, 120, 3), np.uint8)
    image[20:60, 30:70] = (0, 0, 255)
    stamp = time.time() - 0.1
    header = SimpleNamespace(
        frame_id="optical",
        stamp=SimpleNamespace(sec=int(stamp), nanosec=int((stamp % 1) * 1e9)),
    )
    if kind == "compressed":
        _, jpeg = cv2.imencode(".jpg", image)
        msg = SimpleNamespace(header=header, data=jpeg.tobytes(), format="jpeg")
    else:
        msg = SimpleNamespace(
            header=header,
            width=120,
            height=80,
            encoding="bgr8",
            step=360,
            data=image.tobytes(),
        )
    card.sources.image("cam", msg)
    o = card.sources.read("cam")
    assert (
        abs(o["captured_at"] - stamp) < 1e-6
        and o["image_size"] == [120, 80]
        and o["objects"]
        and o["provenance"] == "synthetic"
    )
    preview = call(
        card, "get_preview", source_id="cam", observation_id=o["observation_id"]
    )["preview"]
    import base64

    assert preview["historical_frame"] and preview["observation"] == o
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(preview["jpeg_base64"]), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == image.shape and decoded[40, 50, 2] > 200
    preview["observation"]["captured_at"] = 0
    assert (
        card.sources.preview("cam", o["observation_id"])["observation"]["captured_at"]
        == o["captured_at"]
    )
    assert (
        call(card, "get_preview", source_id="cam", observation_id="missing")["preview"]
        is None
    )
    card.sources.image("cam", msg)
    assert "image replay" in card.sources.errors["cam"]
    card.stop()
    assert not card.sources.previews


def test_moving_camera_does_not_get_a_fixed_view_id_without_evidence():
    cfg = config()
    cfg["sources"]["cam"].update(kind="buffered_image", view_id="UNPROVEN")
    with pytest.raises(Refusal):
        GoalVerifierPlugin(cfg)


def test_no_robot_packages_are_needed(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kw):
        if name.split(".")[0] in ("piper_sdk", "can", "agilex"):
            raise AssertionError("Robot/control import prohibited")
        return original(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    card = GoalVerifierPlugin(config())
    s = task(card)
    assert s["session_id"] and call(card, "info")["motion_capability"] is False
    card.stop()


def test_actual_vop_url_inference_and_ephemeral_image_route(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    import main
    from plugins.vop import VideoObjectPerceptionPlugin
    from plugins.vision_runtime import LetterboxMeta
    from plugins.goal_verifier.exchange import FrameExchange
    from vision_stubs import _FakeExecutor

    monkeypatch.setattr(
        VideoObjectPerceptionPlugin, "_prefetch_vocabulary", lambda self: None
    )

    class FakeEngine:
        input_size = (640, 640)

        def infer(self, image):
            return [np.array([[[10, 10, 40, 40, 0.95, 0]]], np.float32)], LetterboxMeta(
                1.0, 0, 0, image.shape[1], image.shape[0]
            )

    vop = VideoObjectPerceptionPlugin({}, "TEST", _FakeExecutor())
    vop._model = FakeEngine()
    vop._vocabulary = ["piece"]
    cfg = config()
    cfg["sources"]["cam"].update(
        kind="buffered_image",
        view_id="fixed-test-view",
        view_mode="fixed",
        fixed_view_record="SYNTHETIC-FIXTURE",
        detector="vop",
        vop_mcp_url="http://127.0.0.1:1/mcp",
    )
    goal = GoalVerifierPlugin(cfg)
    bundle = main.PerceptionBundle.__new__(main.PerceptionBundle)
    bundle._plugins = [goal, vop]
    monkeypatch.setattr(main, "_bundle", bundle)
    server = ThreadingHTTPServer(("127.0.0.1", 0), main.make_handler())
    base = f"http://127.0.0.1:{server.server_port}"
    goal.config["sources"]["cam"]["vop_mcp_url"] = base + "/mcp"
    goal.image_exchange = FrameExchange(base)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        stamp = time.time()
        goal.sources.feed_image(
            "cam", np.zeros((80, 120, 3), np.uint8), stamp, provenance="synthetic"
        )
        result = goal.sources.read("cam")
        assert result["objects"][0]["class_name"] == "piece" and result["objects"][0][
            "bbox"
        ] == [10, 10, 40, 40]
        assert result["captured_at"] == stamp and result["provenance"] == "synthetic"
        assert not goal.image_exchange.frames
        # Actual original URL decoder, image parsing and VOP postprocessor ran;
        # only its TensorRT inference object and ROS executor were replaced.
    finally:
        goal.stop()
        vop.dispatch("vop", {"action": "stop"})
        server.shutdown()
        server.server_close()
        worker.join(1)


def test_own_stage_manifest_and_modified_files_are_preserved(tmp_path):
    from plugins.goal_verifier.packaging import stage, rollback, sha

    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "own.py").write_text("TEST")
    (payload / "MANIFEST.json").write_text(
        json.dumps(
            dict(
                schema="motus.piper-goal-package/1",
                files={"own.py": sha(payload / "own.py")},
            )
        )
    )
    target = tmp_path / "code"
    record = stage(payload, target)
    (target / "own.py").write_text("USER CHANGE")
    with pytest.raises(Refusal, match="preserve modified"):
        rollback(target, record["run_id"], remove=True)
    assert target.exists()
    (target / "own.py").write_text("TEST")
    rollback(target, record["run_id"], remove=True)
    assert not target.exists()


def test_doctor_has_no_registration_or_visual_side_effects():
    from plugins.goal_verifier.diagnostics import check

    report = check(config())
    assert (
        report["state"] == "READY FOR ONSITE TEST"
        and not report["real_visual_sample_checked"]
    )
    assert not report["browser_test"] and not report["physical_result_verified"]


def test_optional_plane_fit_rejects_extrapolation_or_fake_measurements():
    from plugins.goal_verifier.calibration import fit

    record = dict(
        provenance="operator_measured",
        piper_id="TEST",
        camera_id="TEST",
        frame_id="piper/right/plane",
        view_id="TEST",
        image_size=[100, 100],
        distortion="rectified",
        created_at=1,
        expires_at=10,
        measurement_record="SYNTHETIC-TEST",
        object_plane_record="SYNTHETIC-TEST",
        class_name="piece",
        valid_pixel_bounds=[0, 0, 100, 100],
        fit_pixels=[[0, 0], [100, 0], [100, 100], [0, 100]],
        fit_mm=[[0, 0], [200, 0], [200, 200], [0, 200]],
        check_pixels=[[20, 30], [30, 70], [80, 60]],
        check_mm=[[40, 60], [60, 140], [160, 120]],
        max_reprojection_px=0.1,
        max_check_mm=0.1,
        measurement_uncertainty_mm=0.1,
    )
    assert np.allclose(
        fit(record)["pixel_to_plane_mm"], [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    )
    record["valid_pixel_bounds"][2] = 101
    with pytest.raises(Refusal, match="extrapolation"):
        fit(record)
    record["provenance"] = "unconfirmed"
    with pytest.raises(Refusal, match="measured calibration"):
        fit(record)


def test_mcp_cannot_invent_observations_for_a_private_camera_source():
    cfg = config()
    cfg["sources"]["cam"]["accept_external_observations"] = False
    card = GoalVerifierPlugin(cfg)
    result = call(card, "submit_observation", observation=current(0))
    assert result["code"] == "EXTERNAL_OBSERVATION_DISABLED"
    # Exact copies of already received observations are safe references.
    raw = current(0)
    card.sources.push(raw)
    assert call(card, "submit_observation", observation=raw)["state"] == "observed"


def test_latest_source_error_blocks_fallback_to_previous_good_cache():
    card = GoalVerifierPlugin(config())
    card.sources.push(current(0))
    card.sources.errors["cam"] = "wrong camera identity on latest frame"
    with pytest.raises(Refusal, match="wrong camera"):
        card.sources.read("cam")
    card.sources.push(current(1))
    assert card.sources.read("cam")["observation_id"] == "frame-1"
