"""PiPER visual outcome contracts, no ROS/SDK/CAN/device dependency."""

from copy import deepcopy
import math
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from plugins.goal_verifier.contracts import NAME, OBSERVATION, Refusal
from plugins.goal_verifier.configuration import validate
from plugins.goal_verifier.engine import Engine
from plugins.goal_verifier.plugin import GoalVerifierPlugin
from plugins.goal_verifier.sources import Sources, from_vop, color_objects


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


INVALID_ROS_DOMAINS = [0, 41, 43, 232, True, False, "42", 42.0, None]


def ros_config(kind="ros_normalized", domain_id=42):
    cfg = config()
    cfg["sources"]["cam"].update(
        kind=kind,
        domain_id=domain_id,
        topic="/synthetic/goal_verifier",
        stamp_semantics="capture",
        capture_time_record="SYNTHETIC-UNIT-CAPTURE",
        message_type="raw",
        view_id="synthetic-view",
        view_mode="fixed",
        fixed_view_record="SYNTHETIC-UNIT-FIXED-VIEW",
        classes=[dict(class_name="bottle", color="red")],
    )
    return cfg


@pytest.mark.parametrize("kind", ["ros_normalized", "ros_image"])
def test_ros_configuration_accepts_integer_domain_42(kind):
    cfg = ros_config(kind)
    accepted = validate(cfg)
    assert accepted == cfg and accepted is not cfg
    assert type(accepted["sources"]["cam"]["domain_id"]) is int


@pytest.mark.parametrize("kind", ["ros_normalized", "ros_image"])
@pytest.mark.parametrize("domain_id", INVALID_ROS_DOMAINS)
def test_ros_configuration_rejects_other_domains_without_correction(kind, domain_id):
    cfg = ros_config(kind, domain_id)
    with pytest.raises(Refusal) as exc:
        validate(cfg)
    assert exc.value.code == "INVALID_ROS_DOMAIN"
    assert cfg["sources"]["cam"]["domain_id"] is domain_id


@pytest.fixture
def ros_startup(monkeypatch):
    """Local ROS doubles; no DDS participant or hardware is started."""
    import rclpy
    import rclpy.executors
    import rclpy.node

    context = SimpleNamespace(
        get_domain_id=Mock(return_value=42), try_shutdown=Mock(), shutdown=Mock()
    )
    executor = SimpleNamespace(
        context=context,
        add_node=Mock(),
        remove_node=Mock(),
        spin=Mock(),
        shutdown=Mock(),
    )
    init = Mock()
    node = Mock()
    context_module = ModuleType("rclpy.context")
    context_module.Context = Mock(return_value=context)
    monkeypatch.setitem(sys.modules, "rclpy.context", context_module)
    monkeypatch.setattr(rclpy, "init", init)
    monkeypatch.setattr(rclpy.node, "Node", node)
    executor_factory = Mock(return_value=executor)
    monkeypatch.setattr(rclpy.executors, "MultiThreadedExecutor", executor_factory)
    monkeypatch.setattr(sys.modules["sensor_msgs.msg"], "Image", Mock(), raising=False)
    return SimpleNamespace(
        init=init,
        context=context,
        executor=executor,
        executor_factory=executor_factory,
        node=node,
    )


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("kind", ["ros_normalized", "ros_image"])
def test_ros_startup_uses_domain_42_for_all_sources(
    ros_startup, monkeypatch, external, kind
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "41")
    cfg = ros_config(kind)
    cfg["sources"]["second"] = dict(cfg["sources"]["cam"], source_id="second")
    sources = Sources(cfg, None)
    try:
        sources.start_ros(ros_startup.executor if external else None)
        if external:
            ros_startup.init.assert_not_called()
        else:
            ros_startup.init.assert_called_once_with(
                context=ros_startup.context, domain_id=42
            )
        assert ros_startup.node.call_count == 2
        assert ros_startup.executor.add_node.call_count == 2
        for call in ros_startup.node.call_args_list:
            assert call.kwargs["context"] is ros_startup.context
    finally:
        sources.close()


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("domain_id", INVALID_ROS_DOMAINS)
def test_ros_startup_rechecks_domain_before_any_ros_side_effect(
    ros_startup, external, domain_id
):
    # Bypass configuration validation to exercise the independent startup guard.
    cfg = ros_config()
    cfg["sources"]["second"] = dict(
        cfg["sources"]["cam"], source_id="second", domain_id=domain_id
    )
    sources = Sources(cfg, None)
    with pytest.raises(Refusal) as exc:
        sources.start_ros(ros_startup.executor if external else None)
    assert exc.value.code == "INVALID_ROS_DOMAIN"
    ros_startup.init.assert_not_called()
    ros_startup.node.assert_not_called()
    ros_startup.executor.add_node.assert_not_called()
    assert not sources.nodes and sources.ros is None


@pytest.mark.parametrize("domain_id", INVALID_ROS_DOMAINS)
def test_ros_startup_rejects_external_context_outside_integer_domain_42(
    ros_startup, domain_id
):
    ros_startup.context.get_domain_id.return_value = domain_id
    sources = Sources(ros_config(), None)
    with pytest.raises(Refusal) as exc:
        sources.start_ros(ros_startup.executor)
    assert exc.value.code == "ROS_CONTEXT_DOMAIN_MISMATCH"
    ros_startup.init.assert_not_called()
    ros_startup.node.assert_not_called()


def test_ros_startup_rejects_external_context_without_domain_api(ros_startup):
    del ros_startup.context.get_domain_id
    with pytest.raises(Refusal) as exc:
        Sources(ros_config(), None).start_ros(ros_startup.executor)
    assert exc.value.code == "ROS_CONTEXT_DOMAIN_MISMATCH"
    ros_startup.init.assert_not_called()
    ros_startup.node.assert_not_called()


@pytest.mark.parametrize("stage", ["init", "executor", "context"])
def test_ros_startup_failure_releases_owned_context(ros_startup, stage):
    failure = RuntimeError("synthetic startup failure")
    if stage == "init":
        ros_startup.init.side_effect = failure
    elif stage == "executor":
        ros_startup.executor_factory.side_effect = failure
    else:
        ros_startup.context.get_domain_id.side_effect = failure
    sources = Sources(ros_config(), None)
    with pytest.raises(RuntimeError, match="synthetic startup failure"):
        sources.start_ros()
    assert sources.closed and not sources.nodes
    assert sources.ros is None and sources.thread is None
    ros_startup.context.try_shutdown.assert_called_once_with()
    ros_startup.node.assert_not_called()
    if stage == "context":
        ros_startup.executor.shutdown.assert_called_once_with(timeout_sec=2)
    else:
        ros_startup.executor.shutdown.assert_not_called()
    sources.close()
    ros_startup.context.try_shutdown.assert_called_once_with()


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("kind", ["ros_normalized", "ros_image"])
@pytest.mark.parametrize("stage", ["node", "subscription", "add_node"])
def test_ros_startup_failure_releases_partial_subscriptions(
    ros_startup, external, kind, stage
):
    failure = RuntimeError("synthetic startup failure")
    first, second = Mock(), Mock()
    ros_startup.node.side_effect = [first, failure if stage == "node" else second]
    if stage == "subscription":
        second.create_subscription.side_effect = failure
    elif stage == "add_node":
        ros_startup.executor.add_node.side_effect = [True, failure]
    cfg = ros_config(kind)
    cfg["sources"]["second"] = dict(cfg["sources"]["cam"], source_id="second")
    sources = Sources(cfg, None)
    with pytest.raises(RuntimeError, match="synthetic startup failure"):
        sources.start_ros(ros_startup.executor if external else None)
    assert sources.closed and not sources.nodes
    assert sources.ros is None and sources.thread is None
    created = [first] if stage == "node" else [first, second]
    for node in created:
        node.destroy_node.assert_called_once_with()
        ros_startup.executor.remove_node.assert_any_call(node)
    if external:
        ros_startup.executor.shutdown.assert_not_called()
        ros_startup.context.try_shutdown.assert_not_called()
        ros_startup.context.shutdown.assert_not_called()
    else:
        ros_startup.executor.shutdown.assert_called_once_with(timeout_sec=2)
        ros_startup.context.try_shutdown.assert_called_once_with()
    sources.close()
    for node in created:
        node.destroy_node.assert_called_once_with()


@pytest.mark.parametrize("stage", ["construct", "start", "started"])
def test_ros_startup_failure_releases_owned_thread(ros_startup, monkeypatch, stage):
    import plugins.goal_verifier.sources as source_module

    stopped = threading.Event()
    worker = threading.Thread(target=stopped.wait, daemon=True)
    real_start = worker.start
    failure = RuntimeError("synthetic startup failure")
    factory = Mock(return_value=worker)
    if stage == "construct":
        factory.side_effect = failure
    else:

        def fail_start():
            if stage == "started":
                real_start()
            raise failure

        monkeypatch.setattr(worker, "start", fail_start)
    monkeypatch.setattr(source_module.threading, "Thread", factory)
    ros_startup.executor.shutdown.side_effect = lambda **_: stopped.set()
    sources = Sources(ros_config(), None)
    try:
        with pytest.raises(RuntimeError, match="synthetic startup failure"):
            sources.start_ros()
        assert sources.closed and not sources.nodes
        assert sources.ros is None and sources.thread is None
        assert not worker.is_alive()
        ros_startup.context.try_shutdown.assert_called_once_with()
        ros_startup.node.return_value.destroy_node.assert_called_once_with()
        sources.close()
    finally:
        stopped.set()
        if worker.ident is not None:
            worker.join(2)


@pytest.mark.parametrize("stage", ["remove_node", "destroy_node", "shutdown"])
def test_ros_cleanup_attempts_all_resources_even_if_one_cleanup_fails(
    ros_startup, stage
):
    sources = Sources(ros_config(), None)
    sources.start_ros()
    node = ros_startup.node.return_value
    target = node if stage == "destroy_node" else ros_startup.executor
    getattr(target, stage).side_effect = RuntimeError("synthetic cleanup failure")
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        sources.close()
    node.destroy_node.assert_called_once_with()
    ros_startup.executor.remove_node.assert_called_once_with(node)
    ros_startup.executor.shutdown.assert_called_once_with(timeout_sec=2)
    ros_startup.context.try_shutdown.assert_called_once_with()
    assert not sources.nodes and sources.ros is None and sources.thread is None
    sources.close()


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
