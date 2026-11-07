"""PiPER visual outcome contracts, no ROS/SDK/CAN/device dependency."""

from copy import deepcopy
import math
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from plugins.goal_verifier.contracts import NAME, OBSERVATION, Refusal
from plugins.goal_verifier.engine import Engine
from plugins.goal_verifier.plugin import GoalVerifierPlugin
from plugins.goal_verifier.sources import from_vop, color_objects


def config():
    return dict(
        piper_id="SYNTHETIC-PIPER",
        sources={
            "cam": dict(
                source_id="cam",
                camera_id="TEST-CAMERA",
                frame_id="optical",
                distortion="rectified",
                kind="normalized",
                allow_test_data=True,
                accept_external_observations=True,
                external_observation_record="SYNTHETIC-UNIT-PRODUCER",
            )
        },
    )


class Clock:
    value = 1000.0

    def __call__(self):
        return self.value


def observed(seq, x=0.5, y=0.5, **extra):
    size = extra.pop("image_size", [1000, 800])
    w, h = size
    value = dict(
        schema=OBSERVATION,
        observation_id="frame-" + str(seq),
        source_id="cam",
        source_epoch="source-epoch",
        sequence=seq,
        camera_id="TEST-CAMERA",
        frame_id="optical",
        view_id="view-1",
        image_size=size,
        distortion="rectified",
        captured_at=999.9 + seq * 0.2,
        timestamp_basis="capture",
        provenance="synthetic",
        objects=[
            dict(
                detection_id="0",
                class_name="bottle",
                color="red",
                confidence=0.95,
                bbox=[x * w - 10, y * h - 10, x * w + 10, y * h + 10],
            )
        ],
    )
    value.update(extra)
    return value


def create(engine, **extra):
    args = dict(
        request_id=uuid4().hex,
        task_id="push-or-place-or-manual",
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
    args.update(extra)
    s = engine.call("start_session", args)
    return {k: s[k] for k in ("session_id", "session_token")}


def baseline(e, s, raw=None):
    e.call(
        "capture_before",
        dict(**s, request_id=uuid4().hex, observation=raw or observed(0)),
    )


def verify(e, c, s, raw):
    c.value = max(c.value, 1000.05)
    m = e.call("mark_operation_complete", dict(**s, request_id=uuid4().hex))
    c.value = 1000.4
    return e.call(
        "verify_after",
        dict(**s, request_id=uuid4().hex, marker_id=m["marker_id"], observation=raw),
    )


@pytest.fixture
def rig():
    c = Clock()
    return Engine(config(), wall=c), c


@pytest.mark.parametrize(
    "x,result",
    [(0.75, "PASS"), (0.65, "PASS"), (0.649, "FAIL"), (0.5, "FAIL"), (0.4, "FAIL")],
)
def test_circle_boundary_motion_direction_and_no_motion(rig, x, result):
    e, c = rig
    s = create(e)
    baseline(e, s)
    r = verify(e, c, s, observed(1, x))
    assert r["result"] == result and r["units"] == "normalized" and r["data_valid"]
    assert (
        r["displacement"] == pytest.approx([x - 0.5, 0])
        and r["before_observation_id"] == "frame-0"
    )
    assert r["after_observation_id"] == "frame-1" and not r["physical_motion_confirmed"]


@pytest.mark.parametrize("x,result", [(0.64, "PASS"), (0.639, "FAIL")])
def test_explicit_circle_tolerance(rig, x, result):
    e, c = rig
    s = create(
        e,
        goal=dict(
            units="normalized",
            frame_id="optical",
            shape="circle",
            center=[0.75, 0.5],
            radius=0.1,
            tolerance=0.01,
        ),
    )
    baseline(e, s)
    assert verify(e, c, s, observed(1, x))["result"] == result


@pytest.mark.parametrize(
    "point,result", [((650, 400), "PASS"), ((649, 400), "FAIL"), ((800, 500), "PASS")]
)
def test_rectangle_pixels_and_boundary(rig, point, result):
    e, c = rig
    s = create(
        e,
        goal=dict(
            units="pixels",
            frame_id="optical",
            shape="rectangle",
            bounds=[650, 300, 800, 500],
            tolerance=0,
        ),
    )
    baseline(e, s)
    r = verify(e, c, s, observed(1, point[0] / 1000, point[1] / 800))
    assert r["result"] == result


@pytest.mark.parametrize(
    "fault,code",
    [
        ("empty", "TARGET_NOT_OBSERVED"),
        ("confidence", "TARGET_NOT_OBSERVED"),
        ("multiple", "IDENTITY_AMBIGUOUS"),
        ("view", "VIEW_OR_RESOLUTION_CHANGED"),
        ("size", "VIEW_OR_RESOLUTION_CHANGED"),
        ("epoch", "SOURCE_GEOMETRY_CHANGED"),
        ("replay", "REPLAYED_OBSERVATION"),
        ("before_marker", "PRE_OPERATION_OBSERVATION"),
        ("large_move", "DISPLACEMENT_AMBIGUOUS"),
    ],
)
def test_uncertain_evidence_never_passes(rig, fault, code):
    e, c = rig
    s = create(e)
    baseline(e, s)
    o = observed(1, 0.75)
    if fault == "empty":
        o["objects"] = []
    if fault == "confidence":
        o["objects"][0]["confidence"] = 0.1
    if fault == "multiple":
        second = deepcopy(o["objects"][0])
        second["detection_id"] = "1"
        o["objects"].append(second)
    if fault == "view":
        o["view_id"] = "other-view"
    if fault == "size":
        o["image_size"] = [1200, 800]
    if fault == "epoch":
        o["source_epoch"] = "new-epoch"
    if fault == "replay":
        o = observed(0)
    if fault == "before_marker":
        o["captured_at"] = 1000.01
    if fault == "large_move":
        o = observed(1, 0.95)
    r = verify(e, c, s, o)
    assert r["result"] == "INCONCLUSIVE" and r["code"] == code
    assert not r["data_valid"] and r["target_error"] is None


def test_multiple_objects_can_select_colour_but_not_force_same_colour_identity(rig):
    e, c = rig
    s = create(
        e,
        selector=dict(class_name="bottle", detection_id="0", observation_id="frame-0"),
    )
    b = observed(0)
    other = deepcopy(b["objects"][0])
    other.update(detection_id="1")
    b["objects"].append(other)
    baseline(e, s, b)
    assert verify(e, c, s, observed(1, 0.75))["code"] == "IDENTITY_AMBIGUOUS"


def test_occlusion_and_reappearance_are_not_silently_same_object(rig):
    e, c = rig
    s = create(e)
    baseline(e, s)
    c.value = 1000.2
    e.ingest(observed(1, objects=[]))
    assert verify(e, c, s, observed(2, 0.75))["code"] == "IDENTITY_LOST"


def test_verified_source_identity_handles_multiple_objects_and_large_move(rig):
    e, c = rig
    e.sources["cam"]["verified_persistent_tracks"] = True
    s = create(e)
    b = observed(0, track_epoch="tracking-1")
    b["objects"][0]["track_id"] = "physical-track-7"
    baseline(e, s, b)
    o = observed(1, 0.95, track_epoch="tracking-1")
    o["objects"][0]["track_id"] = "physical-track-7"
    second = deepcopy(o["objects"][0])
    second.update(detection_id="1", track_id="different-track")
    o["objects"].append(second)
    r = verify(e, c, s, o)
    assert (
        r["result"] == "FAIL"
        and r["association_method"] == "source_verified_persistent_track"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "nan",
        "infinity",
        "invalid_type",
        "size",
        "confidence",
        "camera",
        "frame",
        "future",
        "stale",
        "timestamp",
        "missing",
        "unknown",
    ],
)
def test_strict_observation_refusals(rig, fault):
    e, c = rig
    o = observed(0)
    if fault == "nan":
        o["objects"][0]["bbox"][0] = math.nan
    if fault == "infinity":
        o["captured_at"] = math.inf
    if fault == "invalid_type":
        o["sequence"] = True
    if fault == "size":
        o["image_size"] = [0, 800]
    if fault == "confidence":
        o["objects"][0]["confidence"] = 1.01
    if fault == "camera":
        o["camera_id"] = "wrong-camera"
    if fault == "frame":
        o["frame_id"] = "wrong-frame"
    if fault == "future":
        o["captured_at"] = 1001
    if fault == "stale":
        o["captured_at"] = 990
    if fault == "timestamp":
        o["timestamp_basis"] = "publication"
    if fault == "missing":
        del o["camera_id"]
    if fault == "unknown":
        o["unrecognised"] = 1
    with pytest.raises(Refusal):
        e.ingest(o)
    assert not e.latest


def test_time_rollback_reorder_duplicate_and_conflict(rig):
    e, c = rig
    o = observed(0)
    assert e.ingest(o) == e.ingest(o)
    conflict = deepcopy(o)
    conflict["objects"] = []
    with pytest.raises(Refusal, match="observation id conflict"):
        e.ingest(conflict)
    c.value = 1000.4
    e.ingest(observed(2))
    with pytest.raises(Refusal, match="out of order"):
        e.ingest(observed(1))
    c.value = 999
    with pytest.raises(Refusal, match="clock rollback"):
        e.ingest(observed(3))


def test_idempotency_reset_and_session_isolation(rig):
    e, c = rig
    s = create(e)
    other = create(e)
    baseline(e, s)
    c.value = 1000.05
    mark = e.call("mark_operation_complete", dict(**s, request_id="mark"))
    c.value = 1000.3
    args = dict(
        **s,
        request_id="verify",
        marker_id=mark["marker_id"],
        observation=observed(1, 0.75),
    )
    first = e.call("verify_after", args)
    assert first == e.call("verify_after", args)
    assert len(e.sessions[s["session_id"]]["history"]) == 1
    with pytest.raises(Refusal, match="request id conflict"):
        e.call("verify_after", {**args, "observation": observed(2)})
    bad = {**s, "session_token": other["session_token"]}
    with pytest.raises(Refusal, match="retired"):
        e.call("get_result", bad)
    reset = e.call("reset_session", dict(**s, request_id="reset"))
    with pytest.raises(Refusal, match="retired"):
        e.call("verify_after", args)
    assert reset["session_token"] != s["session_token"]
    assert "session_token" not in e.inspect()["sessions"][0]


def test_subsequent_verification_can_report_leaving_goal(rig):
    e, c = rig
    s = create(e)
    baseline(e, s)
    assert verify(e, c, s, observed(1, 0.75))["result"] == "PASS"
    c.value = 1000.6
    marker = e.sessions[s["session_id"]]["marker"]["marker_id"]
    r = e.call(
        "verify_after",
        dict(**s, request_id="left", marker_id=marker, observation=observed(3, 0.5)),
    )
    assert r["result"] == "FAIL" and e.call("get_result", s)["result"] == "FAIL"


def test_metric_requires_measured_binding_and_can_handle_registered_moving_view(rig):
    e, c = rig
    s = create(
        e,
        goal=dict(
            units="plane_mm",
            frame_id="piper/right/plane",
            shape="circle",
            center=[750, 400],
            radius=100,
            tolerance=0,
        ),
    )
    baseline(e, s)
    missing = verify(e, c, s, observed(1, 0.75))
    assert (
        missing["code"] == "CALIBRATION_REQUIRED" and missing["after_position"] is None
    )
    assert missing["image_positions"]["units"] == "pixels"
    # New baseline; measured-transform mechanics tested with explicitly synthetic records.
    e, c = Engine(config(), wall=c), Clock()
    e.wall = c
    e.sources["cam"]["verified_persistent_tracks"] = True
    cal = dict(
        provenance="operator_measured",
        piper_id="SYNTHETIC-PIPER",
        camera_id="TEST-CAMERA",
        frame_id="piper/right/plane",
        view_id="view-1",
        image_size=[1000, 800],
        distortion="rectified",
        created_at=900,
        expires_at=1100,
        measurement_record="SYNTHETIC-CALIBRATION-TEST",
        object_plane_record="SYNTHETIC-HEIGHT-TEST",
        class_name="bottle",
        pixel_to_plane_mm=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        valid_pixel_bounds=[0, 0, 1000, 800],
        max_error_mm=1,
    )
    e.config["calibrations"] = {"a": cal, "b": {**cal, "view_id": "view-2"}}
    s = create(
        e,
        goal=dict(
            units="plane_mm",
            frame_id="piper/right/plane",
            shape="circle",
            center=[750, 400],
            radius=100,
            tolerance=0,
        ),
    )
    b = observed(0, calibration_id="a", track_epoch="id-source")
    b["objects"][0]["track_id"] = "one"
    baseline(e, s, b)
    o = observed(1, 0.75, calibration_id="b", view_id="view-2", track_epoch="id-source")
    o["objects"][0]["track_id"] = "one"
    r = verify(e, c, s, o)
    assert r["result"] == "PASS" and r["units"] == "plane_mm"


def test_vop_preserves_capture_metadata_and_uses_bbox_not_position():
    b = config()["sources"]["cam"]
    f = observed(0)
    f.pop("objects")
    raw = dict(
        count=1,
        image_size=[1000, 800],
        objects=[
            dict(
                name="bottle",
                position=[-0.8, 0.9],
                bbox=[100, 200, 200, 300],
                confidence=0.8,
                color=dict(color_name="red"),
            )
        ],
    )
    o = from_vop(raw, f, b)
    assert o["captured_at"] == f["captured_at"] and o["objects"][0]["bbox"] == [
        100,
        200,
        200,
        300,
    ]
    assert o["objects"][0]["color"] == "red"
    with pytest.raises(Refusal, match="capture time unproven"):
        from_vop(raw, {"timestamp_basis": "publication"}, b)


def test_cpu_image_path_has_actual_pixels_and_multiple_detectable_objects():
    image = np.zeros((120, 200, 3), np.uint8)
    image[20:50, 20:50] = (0, 0, 255)
    image[60:100, 130:170] = (0, 0, 255)
    classes = [
        dict(
            class_name="red_piece",
            color="red",
            hsv_min=[0, 150, 150],
            hsv_max=[10, 255, 255],
            min_area_px=50,
        )
    ]
    detections = color_objects(image, classes)
    assert len(detections) == 2
    assert all(x["confidence_kind"] == "segmentation_fill_ratio" for x in detections)
    assert not any("track_id" in x for x in detections)


def test_stop_interrupts_wait_without_robot_and_retires_session():
    card = GoalVerifierPlugin(config())
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            card.dispatch(NAME, dict(action="observe", source_id="cam"))
        )
    )
    thread.start()
    time.sleep(0.03)
    card.stop()
    thread.join(1)
    assert (
        not thread.is_alive()
        and results[0]["result"] == "INCONCLUSIVE"
        and not card.engine.sessions
    )


def test_plugin_schema_diagnostic_and_no_motion_api():
    card = GoalVerifierPlugin(config())
    tool = card.get_tools()[0]
    assert tool["type"] == "processor" and tool["annotations"]["readOnlyHint"]
    assert tool["inputSchema"]["x-workbench"]["protocol"] == "motus.workbench/1"
    for action in ("enable", "move", "connect_can", "dispatch_motion"):
        assert card.dispatch(NAME, dict(action=action))["result"] == "INCONCLUSIVE"
    assert not card.dispatch(NAME, dict(action="info"))["motion_capability"]


def test_default_configuration_rejects_synthetic_data():
    cfg = config()
    cfg["sources"]["cam"]["allow_test_data"] = False
    with pytest.raises(Refusal, match="test data refused"):
        Engine(cfg, wall=Clock()).ingest(observed(0))


def test_detection_selection_is_bound_to_exact_observation(rig):
    e, c = rig
    with pytest.raises(Refusal, match="selection frame required"):
        create(e, selector=dict(class_name="bottle", detection_id="0"))
    s = create(
        e,
        selector=dict(
            class_name="bottle", detection_id="0", observation_id="other-frame"
        ),
    )
    with pytest.raises(Refusal, match="selection frame changed"):
        baseline(e, s)


def test_truncated_target_is_inconclusive(rig):
    e, c = rig
    s = create(e)
    baseline(e, s)
    o = observed(1, 0.75)
    o["objects"][0]["bbox"][0] = 0
    assert verify(e, c, s, o)["result"] == "INCONCLUSIVE"


def test_expired_sessions_do_not_permanently_exhaust_capacity(rig):
    engine, clock = rig
    previous = [create(engine) for _ in range(32)]
    clock.value += engine.ttl + 1
    fresh = create(engine)
    assert len(engine.sessions) == 1 and fresh["session_id"] in engine.sessions
    with pytest.raises(Refusal, match="unknown session"):
        engine.call(
            "get_result", {k: previous[0][k] for k in ("session_id", "session_token")}
        )


def test_new_requests_cannot_replay_an_earlier_after_observation(rig):
    engine, clock = rig
    s = create(engine)
    baseline(engine, s)
    first = observed(1, 0.75)
    assert verify(engine, clock, s, first)["result"] == "PASS"
    later = observed(2, 0.5)
    clock.value = 1000.4
    marker = engine.call("mark_operation_complete", dict(**s, request_id="next-marker"))
    later["captured_at"] = 1000.41
    clock.value = 1000.42

    def run(rid, o):
        return engine.call(
            "verify_after",
            dict(**s, request_id=rid, marker_id=marker["marker_id"], observation=o),
        )

    assert run("new", later)["result"] == "FAIL"
    assert run("replay", later)["code"] == "REPLAYED_AFTER_OBSERVATION"
    assert run("replay-again", later)["result"] == "INCONCLUSIVE"
    assert run("new", later)["result"] == "FAIL"


@pytest.mark.parametrize(
    "key", ["calibration_id", "track_epoch", "image_sha256", "confidence_kind"]
)
def test_ros_ingest_rejects_optional_field_type_mismatch(rig, key):
    engine, _ = rig
    value = observed(0)
    if key == "confidence_kind":
        value["objects"][0][key] = {"not": "text"}
    else:
        value[key] = {"not": "text"}
    with pytest.raises(Refusal):
        engine.ingest(value)
