"""Synthetic RGB functional tests; no camera, model or robot is accessed."""

import threading
import time
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest
from plugins.tabletop_roi_goal_verifier.contracts import NAME
from plugins.tabletop_roi_goal_verifier.plugin import TabletopROIGoalVerifierPlugin
from plugins.tabletop_roi_goal_verifier.roi import compare, settings

REGION = [0.5, 0.25, 0.9, 0.8]


def configuration():
    binding = {
        "source_id": "camera",
        "camera_id": "SYNTHETIC-CAMERA",
        "frame_id": "optical",
        "view_id": "fixed",
        "distortion": "raw",
        "kind": "buffered_image",
        "view_mode": "fixed",
        "fixed_view_record": "SYNTHETIC TEST ONLY",
        "allow_test_data": True,
        "provenance": "synthetic",
    }
    return {"verifier_id": "SYNTHETIC-ROI", "sources": {"camera": binding}}


def picture(occupied=False):
    image = np.full((240, 320, 3), 150, np.uint8)
    # Background texture allows the camera-movement diagnostic to be tested.
    rng = np.random.default_rng(42)
    image[:50] = rng.integers(60, 200, size=image[:50].shape, dtype=np.uint8)
    if occupied:
        image[100:150, 200:240] = (20, 40, 220)
    return image


def invoke(card, action, **args):
    return card.dispatch(NAME, dict(action=action, **args))


def task(card, **extra):
    session = invoke(
        card,
        "start_session",
        request_id=uuid4().hex,
        task_id="ROI TEST",
        source_id="camera",
        target_roi=REGION,
        **extra,
    )
    assert "session_id" in session, session
    return {k: session[k] for k in ("session_id", "session_token")}


def feed(card, image):
    card.sources.feed_image("camera", image, time.time(), provenance="synthetic")


def verify(card, identity, image):
    marker = invoke(card, "mark_operation_complete", request_id=uuid4().hex, **identity)
    halt = threading.Event()

    def producer():
        while not halt.wait(0.03):
            feed(card, image)

    thread = threading.Thread(target=producer)
    thread.start()
    try:
        return invoke(
            card,
            "verify_after",
            request_id=uuid4().hex,
            marker_id=marker["marker_id"],
            **identity,
        )
    finally:
        halt.set()
        thread.join(1)


@pytest.mark.parametrize("occupied,expected", [(True, "PASS"), (False, "FAIL")])
def test_roi_rgb_end_to_end_without_classes_or_detector(occupied, expected):
    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        identity = task(card)
        feed(card, picture())
        before = invoke(card, "capture_before", request_id=uuid4().hex, **identity)
        assert before["state"] == "before_captured"
        result = verify(card, identity, picture(occupied))
        assert result["result"] == expected, result
        assert len(set(result["after_observation_ids"])) == 2
        assert result["criterion"] == "roi_visual_change"
        assert result["physical_motion_confirmed"] is False
        assert result["identity_confirmed"] is False
        assert result["roi_evidence"]["before_target_jpeg_base64"]
        exported = invoke(card, "export_session", **identity)
        assert "session_token" not in exported and "objects" not in exported["before"]
        assert exported["before"]["image_sha256"]
        assert len(exported["observations"]) == 3
    finally:
        card.stop()


def test_vop_failure_does_not_affect_roi_preview_or_verification(monkeypatch):
    card = TabletopROIGoalVerifierPlugin(configuration())
    monkeypatch.setattr(
        card.sources,
        "vop_call",
        lambda *a: pytest.fail("ROI called VOP"),
        raising=False,
    )
    try:
        feed(card, picture())
        observed = invoke(card, "observe", source_id="camera")
        assert observed["observation"]["schema"] == "motus.roi-image-observation/1"
        identity = task(card)
        invoke(card, "capture_before", request_id=uuid4().hex, **identity)
        assert verify(card, identity, picture(True))["result"] == "PASS"
    finally:
        card.stop()


def test_compressed_ros_image_callback_to_roi_without_detector():
    cfg = configuration()
    cfg["sources"]["camera"].update(
        kind="ros_image",
        domain_id=42,
        topic="/synthetic-roi-test",
        message_type="compressed",
        stamp_semantics="capture",
        capture_time_record="SYNTHETIC header timestamp",
        provenance="synthetic",
    )
    card = TabletopROIGoalVerifierPlugin(
        cfg
    )  # Callback-only synthetic test: do not initialize ROS or DDS.
    identity = task(card, roi_options={"stable_frames": 2})

    def send(image):
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        stamp = time.time_ns()
        msg = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="optical",
                stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9),
            ),
            format="jpeg",
            data=encoded.tobytes(),
        )
        card.sources.image("camera", msg)
        return stamp / 1e9

    try:
        stamp = send(picture())
        invoke(card, "capture_before", request_id=uuid4().hex, **identity)
        assert card.engine.sessions[identity["session_id"]]["before"][
            "captured_at"
        ] == pytest.approx(stamp)
        marker = invoke(
            card, "mark_operation_complete", request_id=uuid4().hex, **identity
        )
        send(picture(True))
        halt = threading.Event()

        def stream():
            while not halt.wait(0.03):
                send(picture(True))

        producer = threading.Thread(target=stream)
        producer.start()
        try:
            result = invoke(
                card,
                "verify_after",
                request_id=uuid4().hex,
                marker_id=marker["marker_id"],
                **identity,
            )
        finally:
            halt.set()
            producer.join(1)
        assert result["result"] == "PASS" and result["provenance"] == "synthetic"
    finally:
        card.stop()


@pytest.mark.parametrize(
    "scenario", ["occlusion", "camera_move", "lighting", "unstable"]
)
def test_uncomparable_roi_images_do_not_pass(scenario):
    from plugins.tabletop_roi_goal_verifier.contracts import Refusal

    before = picture()
    after = picture(True)
    if scenario == "occlusion":
        after[60:192, 160:288] = 40
    elif scenario == "camera_move":
        after = np.roll(after, 18, axis=1)
    elif scenario == "lighting":
        after = np.clip(after.astype(int) + 65, 0, 255).astype(np.uint8)
    region, options = settings(REGION)
    images = [after, picture()] if scenario == "unstable" else [after, after]
    try:
        result = compare(before, images, region, options)
        assert result["result"] == "INCONCLUSIVE", result
    except Refusal as exc:
        assert exc.code.startswith("ROI_")


def test_stale_frame_missing_image_and_reset_do_not_reuse_baseline(monkeypatch):
    card = TabletopROIGoalVerifierPlugin(configuration())
    identity = task(card)
    try:
        # No frame can be substituted by a normalized object observation.
        monkeypatch.setattr(
            card.sources,
            "read_image",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("no image")),
        )
        out = invoke(card, "capture_before", request_id=uuid4().hex, **identity)
        assert out["result"] == "INCONCLUSIVE"
        monkeypatch.undo()
        feed(card, picture())
        packet = card.sources.read_image("camera")
        old = deepcopy(packet)
        old["frame"]["captured_at"] -= 20
        monkeypatch.setattr(card.sources, "read_image", lambda *a, **kw: old)
        assert (
            invoke(card, "capture_before", request_id=uuid4().hex, **identity)["code"]
            == "STALE_OR_FUTURE"
        )
        monkeypatch.undo()
        assert (
            invoke(card, "capture_before", request_id=uuid4().hex, **identity)["state"]
            == "before_captured"
        )
        new = invoke(card, "reset_session", request_id=uuid4().hex, **identity)
        assert new["session_token"] != identity["session_token"]
        assert not card.engine.roi_images
        assert invoke(card, "get_result", **identity)["code"] == "RETIRED_SESSION"
    finally:
        card.stop()


def test_repeated_post_frame_is_rejected(monkeypatch):
    card = TabletopROIGoalVerifierPlugin(configuration())
    identity = task(card)
    try:
        feed(card, picture())
        invoke(card, "capture_before", request_id=uuid4().hex, **identity)
        marker = invoke(
            card, "mark_operation_complete", request_id=uuid4().hex, **identity
        )
        feed(card, picture(True))
        packet = card.sources.read_image("camera")
        monkeypatch.setattr(
            card.sources, "read_image", lambda *a, **kw: deepcopy(packet)
        )
        result = invoke(
            card,
            "verify_after",
            request_id=uuid4().hex,
            marker_id=marker["marker_id"],
            **identity,
        )
        assert (
            result["result"] == "INCONCLUSIVE"
            and result["code"] == "REPLAYED_AFTER_OBSERVATION"
        )
    finally:
        card.stop()


@pytest.mark.parametrize("region", [[0, 0, 0, 1], [-0.1, 0, 0.5, 0.5], [0, 0, 2, 1]])
def test_invalid_region_is_rejected(region):
    card = TabletopROIGoalVerifierPlugin(configuration())
    out = invoke(
        card,
        "start_session",
        request_id=uuid4().hex,
        task_id="bad ROI",
        source_id="camera",
        target_roi=region,
    )
    assert out["result"] == "INCONCLUSIVE" and not card.engine.sessions


def test_pre_operation_frame_cannot_be_used_for_after(monkeypatch):
    card = TabletopROIGoalVerifierPlugin(configuration())
    identity = task(card)
    try:
        feed(card, picture())
        old = card.sources.read_image("camera")
        invoke(card, "capture_before", request_id=uuid4().hex, **identity)
        marker = invoke(
            card, "mark_operation_complete", request_id=uuid4().hex, **identity
        )
        monkeypatch.setattr(card.sources, "read_image", lambda *a, **kw: deepcopy(old))
        result = invoke(
            card,
            "verify_after",
            request_id=uuid4().hex,
            marker_id=marker["marker_id"],
            **identity,
        )
        assert (
            result["result"] == "INCONCLUSIVE"
            and result["code"] == "PRE_OPERATION_OBSERVATION"
        )
    finally:
        card.stop()


@pytest.mark.parametrize(
    "scenario,code",
    [
        ("occlusion", "ROI_OCCLUDED_OR_REPLACED"),
        ("lighting", "ROI_LIGHTING_CHANGED"),
        ("camera", "ROI_CAMERA_MOVED"),
    ],
)
def test_session_inconclusive_retains_images_and_metadata(scenario, code):
    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        identity = task(card)
        feed(card, picture())
        invoke(card, "capture_before", request_id=uuid4().hex, **identity)
        image = picture(True)
        if scenario == "occlusion":
            image[60:192, 160:288] = 40
        elif scenario == "lighting":
            image = np.clip(image.astype(int) + 65, 0, 255).astype(np.uint8)
        else:
            image = np.roll(image, 18, axis=1)
        out = verify(card, identity, image)
        assert out["result"] == "INCONCLUSIVE" and out["code"] == code, out
        assert out["roi_evidence"]["after_target_jpeg_base64"]
        assert len(out["evidence"]["after"]) == 2
    finally:
        card.stop()


def test_idempotence_and_conflicts_including_concurrent_capture_and_retired_tokens(
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor

    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        args = {
            "request_id": "start",
            "task_id": "test",
            "source_id": "camera",
            "target_roi": REGION,
        }
        first = invoke(card, "start_session", **args)
        assert invoke(card, "start_session", **args) == first
        assert (
            invoke(card, "start_session", **{**args, "task_id": "other"})["code"]
            == "REQUEST_ID_CONFLICT"
        )
        identity = {k: first[k] for k in ("session_id", "session_token")}
        feed(card, picture())
        count = 0
        read = card.sources.read_image

        def counted(*a, **kw):
            nonlocal count
            count += 1
            return read(*a, **kw)

        monkeypatch.setattr(card.sources, "read_image", counted)
        with ThreadPoolExecutor(2) as pool:
            outs = list(
                pool.map(
                    lambda _: invoke(
                        card, "capture_before", request_id="before", **identity
                    ),
                    range(2),
                )
            )
        assert outs[0] == outs[1] and count == 1
        out = verify(card, identity, picture(True))
        session = card.engine.sessions[identity["session_id"]]
        after_args = next(
            k
            for k in card.engine.requests
            if k[0] == identity["session_id"]
            and card.engine.requests[k][1].get("code") == "ROI_CHANGE_MET"
        )
        retry = dict(
            request_id=after_args[2],
            marker_id=session["marker"]["marker_id"],
            **identity,
        )
        assert invoke(card, "verify_after", **retry) == out
        assert (
            invoke(card, "verify_after", **{**retry, "marker_id": "wrong"})["code"]
            == "REQUEST_ID_CONFLICT"
        )
        assert invoke(card, "get_result", **identity) == out
        reset = invoke(card, "reset_session", request_id="reset", **identity)
        assert invoke(card, "reset_session", request_id="reset", **identity) == reset
        identity = {k: reset[k] for k in ("session_id", "session_token")}
        close = invoke(card, "close_session", request_id="close", **identity)
        assert invoke(card, "close_session", request_id="close", **identity) == close
        assert not card.engine.sessions and not card.engine.roi_images
    finally:
        card.stop()


def test_source_error_recovers_on_next_valid_frame_and_stop_retires_sessions():
    card = TabletopROIGoalVerifierPlugin(configuration())
    identity = task(card)
    card.sources.reject("camera", ValueError("broken jpeg"))
    assert (
        invoke(card, "capture_before", request_id="bad", **identity)["code"]
        == "SOURCE_INPUT_ERROR"
    )
    feed(card, picture())
    assert (
        invoke(card, "capture_before", request_id="good", **identity)["state"]
        == "before_captured"
    )
    invoke(card, "stop")
    assert (
        card.sources.closed and not card.engine.sessions and not card.engine.roi_images
    )
    assert invoke(card, "start")["state"] == "running"
    assert invoke(card, "get_result", **identity)["code"] == "UNKNOWN_SESSION"
    card.stop()


@pytest.mark.parametrize(
    "mutation,code",
    [
        ({"provenance": "real"}, "OFFLINE_TEST_DATA_REQUIRED"),
        ({"allow_test_data": False}, "TEST_DATA_REFUSED"),
        ({"kind": "ros_normalized"}, "ROI_IMAGE_SOURCE_REQUIRED"),
    ],
)
def test_offline_data_cannot_claim_real_camera(mutation, code):
    from plugins.tabletop_roi_goal_verifier.contracts import Refusal

    cfg = configuration()
    cfg["sources"]["camera"].update(mutation)
    with pytest.raises(Refusal, match=".") as error:
        TabletopROIGoalVerifierPlugin(cfg)
    assert error.value.code == code


def test_clock_rollback_expiry_and_bounded_sessions():
    from plugins.tabletop_roi_goal_verifier.contracts import Refusal
    from plugins.tabletop_roi_goal_verifier.engine import Engine

    now = [100.0]
    engine = Engine(configuration(), wall=lambda: now[0])
    args = {
        "request_id": "s",
        "task_id": "t",
        "source_id": "camera",
        "target_roi": REGION,
    }
    s = engine.call("start_session", args)
    identity = {k: s[k] for k in ("session_id", "session_token")}
    now[0] = 99.0
    with pytest.raises(Refusal) as e:
        engine.call("get_result", identity)
    assert e.value.code == "CLOCK_ROLLBACK"
    now[0] = 701.0
    with pytest.raises(Refusal) as e:
        engine.call("get_result", identity)
    assert e.value.code == "SESSION_EXPIRED"
    for i in range(32):
        engine.call("start_session", {**args, "request_id": str(i)})
    with pytest.raises(Refusal) as e:
        engine.call("start_session", {**args, "request_id": "overflow"})
    assert e.value.code == "SESSION_LIMIT"


@pytest.mark.parametrize(
    "change,code",
    [
        ({"captured_at": -1}, "STALE_OR_FUTURE"),
        ({"captured_at": time.time() + 1000}, "STALE_OR_FUTURE"),
        ({"source_epoch": "changed"}, "SOURCE_GEOMETRY_CHANGED"),
        ({"camera_id": "changed"}, "SOURCE_BINDING_MISMATCH"),
        ({"image_size": [123, 456]}, "INVALID_IMAGE"),
    ],
)
def test_after_metadata_guards(change, code, monkeypatch):
    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        identity = task(card)
        feed(card, picture())
        invoke(card, "capture_before", request_id="b", **identity)
        marker = invoke(card, "mark_operation_complete", request_id="m", **identity)
        feed(card, picture(True))
        packet = card.sources.read_image("camera")
        packet["frame"].update(change)
        monkeypatch.setattr(
            card.sources, "read_image", lambda *a, **kw: deepcopy(packet)
        )
        out = invoke(
            card,
            "verify_after",
            request_id="v",
            marker_id=marker["marker_id"],
            **identity,
        )
        assert out["result"] == "INCONCLUSIVE" and out["code"] == code
    finally:
        card.stop()


def test_local_brightness_compensation_and_insufficient_stability():
    from plugins.tabletop_roi_goal_verifier.contracts import Refusal

    region, options = settings(REGION)
    before = picture()
    after = np.clip(picture(True).astype(int) + 15, 0, 255).astype(np.uint8)
    assert compare(before, [after, after], region, options)["result"] == "PASS"
    with pytest.raises(Refusal):
        compare(before, [], region, options)
    with pytest.raises(Refusal):
        settings(REGION, {"stable_frames": 1})


def test_bundle_registration_and_unique_tool():
    import main

    card = TabletopROIGoalVerifierPlugin(configuration())
    bundle = main.PerceptionBundle.__new__(main.PerceptionBundle)
    bundle._plugins = [card]
    assert bundle.owns(NAME)
    assert bundle.dispatch(NAME, {"action": "info"})["motion_capability"] is False
    assert [t["name"] for t in bundle.get_all_tools()] == [NAME]
    assert not bundle.owns("piper_object_goal_verifier")
    assert not hasattr(card.sources, "read")
    card.stop()
