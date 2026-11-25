"""Conservative ROI review regressions; synthetic pixels, no ROS subscriptions."""

import base64
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest
from plugins.tabletop_roi_goal_verifier.contracts import Refusal
from plugins.tabletop_roi_goal_verifier.evidence import save_session
from plugins.tabletop_roi_goal_verifier.plugin import TabletopROIGoalVerifierPlugin
from plugins.tabletop_roi_goal_verifier.roi import compare, settings
from test_tabletop_roi_goal_verifier import (
    REGION,
    configuration,
    feed,
    invoke,
    picture,
    task,
    verify,
)


def baseline():
    card = TabletopROIGoalVerifierPlugin(configuration())
    identity = task(card)
    feed(card, picture())
    assert (
        invoke(card, "capture_before", request_id=uuid4().hex, **identity)["state"]
        == "before_captured"
    )
    return card, identity


def test_stop_interrupts_a_waiting_verification_without_waiting_for_frames(monkeypatch):
    card, identity = baseline()
    entered = threading.Event()
    original = card.sources.read_image

    def waiting(*args, **kw):
        entered.set()
        return original(*args, **kw)

    monkeypatch.setattr(card.sources, "read_image", waiting)
    marker = invoke(card, "mark_operation_complete", request_id="marker", **identity)
    result = {}
    caller = threading.Thread(
        target=lambda: result.update(
            invoke(
                card,
                "verify_after",
                request_id="v",
                marker_id=marker["marker_id"],
                **identity,
            )
        )
    )
    stopper = threading.Thread(target=card.stop)
    try:
        caller.start()
        assert entered.wait(1)
        stopper.start()
        stopper.join(0.5)
        assert not stopper.is_alive(), (
            "stop must cancel acquisition, not wait for its timeout"
        )
        caller.join(1)
        assert result["result"] == "INCONCLUSIVE"
        assert not card.engine.sessions
    finally:
        card.sources.close()
        caller.join(4)
        if stopper.ident is not None:
            stopper.join(4)
        card.stop()


def test_camera_error_then_reconnection_cannot_reuse_the_old_baseline():
    card, identity = baseline()
    try:
        epoch = card.engine.sessions[identity["session_id"]]["before"]["source_epoch"]
        card.sources.reject("camera", ValueError("camera disconnected / invalid frame"))
        feed(card, picture(True))
        assert card.sources.read_image("camera")["frame"]["source_epoch"] != epoch
        result = verify(card, identity, picture(True))
        assert (
            result["result"] == "INCONCLUSIVE"
            and result["code"] == "SOURCE_GEOMETRY_CHANGED"
        )
    finally:
        card.stop()


def test_capture_encoding_failure_leaves_no_partial_baseline(monkeypatch):
    from plugins.tabletop_roi_goal_verifier import roi

    card = TabletopROIGoalVerifierPlugin(configuration())
    identity = task(card)
    feed(card, picture())

    def broken(*args):
        raise OSError("image encode unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(roi, "jpeg", broken)
            assert (
                invoke(card, "capture_before", request_id="bad", **identity)["result"]
                == "INCONCLUSIVE"
            )
        assert card.engine.sessions[identity["session_id"]]["before"] is None
        assert not card.engine.roi_images
        assert (
            invoke(card, "capture_before", request_id="retry", **identity)["state"]
            == "before_captured"
        )
    finally:
        card.stop()


def test_failed_new_after_capture_does_not_export_an_old_after_image(monkeypatch):
    card, identity = baseline()
    try:
        assert verify(card, identity, picture(True))["result"] == "PASS"
        marker = invoke(
            card, "mark_operation_complete", request_id="new-marker", **identity
        )
        monkeypatch.setattr(
            card.sources,
            "read_image",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("camera lost")),
        )
        result = invoke(
            card,
            "verify_after",
            request_id="new-after",
            marker_id=marker["marker_id"],
            **identity,
        )
        assert result["result"] == "INCONCLUSIVE"
        assert "after_target_jpeg_base64" not in result.get("roi_evidence", {})
    finally:
        card.stop()


@pytest.mark.parametrize("mode", ["shadow", "reflection", "camera_jitter"])
def test_known_photometric_or_pose_nuisance_cannot_pass(mode):
    before = picture()
    after = before.copy()
    if mode == "shadow":
        after[100:150, 200:240] = 80
    elif mode == "reflection":
        after[100:150, 200:240] = 230
    else:
        rng = np.random.default_rng(3)
        before = rng.integers(30, 220, size=before.shape, dtype=np.uint8)
        after = np.roll(before, 1, axis=1)
    region, options = settings(REGION)
    try:
        result = compare(before, [after, after], region, options)
    except Refusal:
        return
    assert result["result"] != "PASS", result


def exported_fixture():
    image = base64.b64encode(cv2.imencode(".jpg", picture())[1]).decode()
    return {
        "before": {"provenance": "synthetic"},
        "roi_evidence": {
            "before_target_jpeg_base64": image,
            "after_target_jpeg_base64": image,
        },
        "result": {"result": "PASS", "data_valid": True, "code": "ROI_CHANGE_MET"},
    }


def test_incomplete_success_evidence_is_refused(tmp_path):
    exported = exported_fixture()
    exported["roi_evidence"].pop("after_target_jpeg_base64")
    with pytest.raises(Refusal):
        save_session(exported, tmp_path, "PASS")
    assert not list(tmp_path.iterdir())


def test_evidence_write_failure_does_not_publish_a_partial_run(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.write_bytes

    def broken(path, data):
        if path.name == "after-roi.jpg":
            raise OSError("simulated disk full")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", broken)
    with pytest.raises(OSError):
        save_session(exported_fixture(), tmp_path, "PASS")
    assert not list(tmp_path.iterdir()), (
        "failed writes must not leave a seemingly complete evidence run"
    )


@pytest.mark.parametrize("value", [[], 42, {"private": "x"}])
def test_core_trace_metadata_has_a_strict_type(value):
    card = TabletopROIGoalVerifierPlugin(configuration())
    assert (
        card.dispatch(
            "tabletop_roi_goal_verifier", {"action": "info", "_trace_id": value}
        )["result"]
        == "INCONCLUSIVE"
    )
    card.stop()


def test_actual_core_trace_metadata_is_accepted_without_changing_request_identity():
    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        args = {
            "action": "start_session",
            "request_id": "one",
            "task_id": "task",
            "source_id": "camera",
            "target_roi": REGION,
        }
        one = card.dispatch(
            "tabletop_roi_goal_verifier", {**args, "_trace_id": "trace-1"}
        )
        assert "session_id" in one, one
        two = card.dispatch(
            "tabletop_roi_goal_verifier", {**args, "_trace_id": "trace-2"}
        )
        assert one == two
    finally:
        card.stop()


@pytest.mark.parametrize(
    "variant", ["empty", "corrupt", "bad_stamp", "wrong_frame", "format_changed"]
)
def test_invalid_ros_packets_poison_the_cached_frame_and_do_not_pass(variant):
    cfg = configuration()
    cfg["sources"]["camera"].update(
        kind="ros_image",
        domain_id=42,
        topic="/SYNTHETIC_ONLY",
        message_type="compressed",
        stamp_semantics="capture",
        capture_time_record="SYNTHETIC TEST",
    )
    card = TabletopROIGoalVerifierPlugin(cfg)
    try:
        identity = task(card)
        feed(card, picture())
        invoke(card, "capture_before", request_id="before", **identity)
        ok, jpeg = cv2.imencode(".jpg", picture(True))
        assert ok
        now = time.time_ns()
        msg = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="optical",
                stamp=SimpleNamespace(sec=now // 10**9, nanosec=now % 10**9),
            ),
            format="jpeg",
            data=jpeg.tobytes(),
        )
        if variant == "empty":
            msg.data = b""
        if variant == "corrupt":
            msg.data = b"not an image"
        if variant == "bad_stamp":
            msg.header.stamp.nanosec = -1
        if variant == "wrong_frame":
            msg.header.frame_id = "another-camera"
        if variant == "format_changed":
            msg.format = "png"
        card.sources.image("camera", msg)
        assert "camera" in card.sources.errors
        result = verify(card, identity, picture(True))
        assert result["result"] == "INCONCLUSIVE", result
    finally:
        card.stop()


@pytest.mark.parametrize(
    "region",
    [[0, 0, 0, 0], [0.1, 0.1, 0.1001, 0.1001], [-0.1, 0, 0.5, 0.5], [0, 0, 1.1, 0.9]],
)
def test_empty_tiny_or_outside_roi_never_gets_a_valid_baseline(region):
    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        started = invoke(
            card,
            "start_session",
            request_id="s",
            task_id="bad roi",
            source_id="camera",
            target_roi=region,
        )
        if "session_id" not in started:
            assert started["result"] == "INCONCLUSIVE"
            return
        feed(card, picture())
        identity = {k: started[k] for k in ("session_id", "session_token")}
        assert (
            invoke(card, "capture_before", request_id="b", **identity)["result"]
            == "INCONCLUSIVE"
        )
    finally:
        card.stop()


def test_two_sessions_keep_independent_baselines_and_cancellation(monkeypatch):
    # Session isolation is independent of host realtime corrections. Use an
    # explicitly synthetic, monotonic-derived wall clock in this concurrency test;
    # separate regressions exercise real timestamp rejection and clock rollback.
    import plugins.tabletop_roi_goal_verifier.sources as sources_module

    anchor_wall, anchor_mono = time.time(), time.monotonic()
    stable_wall = lambda: anchor_wall + time.monotonic() - anchor_mono
    monkeypatch.setattr(sources_module.time, "time", stable_wall)
    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        first = task(card)
        second = task(card)
        feed(card, picture())
        invoke(card, "capture_before", request_id="first-before", **first)
        feed(card, picture(True))
        invoke(card, "capture_before", request_id="second-before", **second)
        assert verify(card, first, picture(True))["result"] == "PASS"
        result = verify(card, second, picture(True))
        assert result["result"] == "FAIL", result
        invoke(card, "close_session", request_id="close-first", **first)
        assert invoke(card, "get_result", **second)["result"] == "FAIL"
        wrong = {**second, "session_token": first["session_token"]}
        assert invoke(card, "get_result", **wrong)["code"] == "RETIRED_SESSION"
    finally:
        card.stop()


def test_resolution_change_wall_rollback_and_saturated_frame_fail_closed():
    card, identity = baseline()
    try:
        after = cv2.resize(picture(True), (640, 480))
        assert verify(card, identity, after)["result"] == "INCONCLUSIVE"
        original = card.engine.wall
        card.engine.wall = lambda: card.engine.last_wall - 1
        assert invoke(card, "get_result", **identity)["result"] == "INCONCLUSIVE"
        card.engine.wall = original
    finally:
        card.stop()
    card = TabletopROIGoalVerifierPlugin(configuration())
    try:
        identity = task(card)
        feed(card, np.zeros((240, 320, 3), np.uint8))
        assert (
            invoke(card, "capture_before", request_id="zero", **identity)["result"]
            == "INCONCLUSIVE"
        )
    finally:
        card.stop()


def test_wall_clock_rollback_retires_old_baselines_even_after_clock_recovers():
    card, identity = baseline()
    try:
        real_wall = card.engine.wall
        card.engine.wall = lambda: card.engine.last_wall - 1
        assert invoke(card, "get_result", **identity)["code"] == "CLOCK_ROLLBACK"
        card.engine.wall = real_wall
        assert invoke(card, "get_result", **identity)["code"] == "SESSION_CLOCK_INVALID"
        new_identity = task(card)
        feed(card, picture())
        assert (
            invoke(
                card, "capture_before", request_id="new-clock-before", **new_identity
            )["state"]
            == "before_captured"
        )
    finally:
        card.stop()


def test_bundle_logging_does_not_disclose_session_capabilities():
    import main

    payload = {
        "session_id": "public",
        "session_token": "PRIVATE-SESSION-TOKEN",
        "nested": [{"session_token": "SECOND-SECRET"}],
    }
    assert "PRIVATE-SESSION-TOKEN" not in main._brief(payload)
    assert "SECOND-SECRET" not in repr(main._log_payload(payload))
    assert main._log_payload(payload)["session_id"] == "public"


@pytest.mark.parametrize("invalid", [None, "future", "replay", "expired"])
def test_validation_clock_is_sampled_after_delayed_new_frames(monkeypatch, invalid):
    from copy import deepcopy

    # Deterministic synthetic wall clock, independently advanced while acquiring.
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    card, identity = baseline()
    try:
        clock[0] += 0.1
        marker = invoke(card, "mark_operation_complete", request_id="mark", **identity)
        request_time = clock[0]
        original = card.sources.read_image
        first = []

        def delayed(*args, **kwargs):
            clock[0] += 601 if invalid == "expired" else 0.2
            feed(card, picture(True))
            packet = original(*args, **kwargs)
            if invalid == "future" and first:
                packet["frame"]["captured_at"] += 10
            if invalid == "replay" and first:
                return deepcopy(first[0])
            first.append(deepcopy(packet))
            return packet

        monkeypatch.setattr(card.sources, "read_image", delayed)
        result = invoke(
            card,
            "verify_after",
            request_id="after",
            marker_id=marker["marker_id"],
            **identity,
        )
        if invalid is None:
            assert result["result"] == "PASS", result
            stamps = [p["captured_at"] for p in result["evidence"]["after"]]
            assert request_time < stamps[0] < stamps[1] <= result["verified_at"]
        else:
            assert result["result"] == "INCONCLUSIVE", result
            assert (
                result["code"]
                == {
                    "future": "STALE_OR_FUTURE",
                    "replay": "REPLAYED_AFTER_OBSERVATION",
                    "expired": "SESSION_EXPIRED",
                }[invalid]
            )
    finally:
        card.stop()


def test_roi_does_not_require_python310_pairwise(monkeypatch):
    import itertools

    monkeypatch.delattr(itertools, "pairwise", raising=False)
    region, options = settings(REGION)
    assert (
        compare(picture(), [picture(True), picture(True)], region, options)["result"]
        == "PASS"
    )
    assert (
        compare(picture(), [picture(), picture()], region, options)["result"] == "FAIL"
    )
