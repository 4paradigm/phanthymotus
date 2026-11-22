"""Mount amendments revise retained camera observations through the public wire."""
import json
from threading import Thread
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins import rgbd_external_input
from plugins import external_odometry_fusion
from plugins.external_odometry_fusion import ExternalOdometry
from plugins.pointcloud_slam_odom import decode_slam_odom
from plugins.rgbd_fusion import quaternion
from test_external_odometry_fusion import packet
from test_rgbd_odometry_status import sessions  # noqa: F401


START = 1_000_000_000
CAPTURE = 1_100_000_000


def rotation(degrees):
    angle = np.deg2rad(degrees)
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]
    return matrix


def native_pose(degrees, covariance=None):
    pose = dict(position=[0., 0., 0.], quaternion=quaternion(rotation(degrees)[:3, :3]))
    if covariance is not None:
        pose['covariance'] = np.asarray(covariance).ravel().tolist()
    return pose


def mount(stream, stamp, degrees, **metadata):
    value = dict(stamp_ns=stamp, matrix=None if degrees is None else rotation(degrees).tolist(), **metadata)
    stream._receive_mount(NS(data=json.dumps(value)))


def motion(stream, stamp, source='/external'):
    stream._receive(source, NS(data=json.dumps(packet((stamp-START)/1e9, 100.))))


def flush(stream):
    stream._apply_ready(stream._take_ready())


def samples(stream):
    return [decode_slam_odom(json.loads(message)) for message in stream.publisher.messages]


@pytest.fixture
def streams(sessions, monkeypatch):
    _, now, _ = sessions
    tick = [0.]
    monkeypatch.setattr(rgbd_external_input.time, 'monotonic', lambda: tick[0])
    opened = []

    def create(*, learning=False):
        stream = rgbd_external_input.ExternalOdometryInput('/fused/state/odom', dict(
            frame_id='camera', body_mount=dict(topic='/mount'), external_topic='/external',
            external_candidates=['/external', '/other']), learning=learning)
        opened.append(stream)
        return stream

    yield create, now, tick
    for stream in opened:
        stream.close()


@pytest.mark.parametrize('amendment', ['exact', 'unavailable', 'malformed'])
@pytest.mark.parametrize('queue_delay', [0., .35])
def test_mount_amendment_after_publication_matches_before_acceptance(streams, amendment, queue_delay):
    """Without retained camera evidence, an accepted 30 degree body error survives."""
    create, now, tick = streams

    def amend(stream):
        if amendment == 'malformed':
            stream._receive_mount(NS(data=json.dumps(dict(stamp_ns=CAPTURE, matrix=[[1.]]))))
        else:
            mount(stream, CAPTURE, 30. if amendment == 'exact' else None)

    def run(late):
        stream = create()
        now[0] = START+10_000_000
        mount(stream, START, 0.)
        motion(stream, START)
        stream.visual(START, 'native', native_pose(0.))
        flush(stream)
        now[0] = CAPTURE+110_000_000
        mount(stream, CAPTURE+100_000_000, 0.)
        if not late:
            amend(stream)
        motion(stream, CAPTURE)
        stream.visual(CAPTURE, 'native', native_pose(30.))
        tick[0] += queue_delay
        now[0] += round(queue_delay*1e9)
        flush(stream)
        emitted = list(stream.publisher.messages)
        if late:
            assert samples(stream)[-1].stamp_ns == CAPTURE
            amend(stream)
            flush(stream)
            assert stream.publisher.messages == emitted  # Never republish the amended capture.
        return stream

    ordered, late = run(False), run(True)
    if amendment == 'exact':
        np.testing.assert_allclose(late.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)
    else:
        assert CAPTURE not in late.fuser.raw_vision
    np.testing.assert_allclose(late.fuser.current.transform, ordered.fuser.current.transform, atol=1e-14)
    np.testing.assert_allclose(late.fuser.current.covariance, ordered.fuser.current.covariance, atol=1e-14)
    if amendment == 'malformed':
        assert late.last_error is not None
        assert late.fuser.current.visual is None
    # The next fresh measurement publishes the accepted replayed estimate.
    now[0] += 50_000_000
    for stream in (ordered, late):
        motion(stream, now[0]-10_000_000)
        flush(stream)
    np.testing.assert_allclose(samples(late)[-1].quaternion, samples(ordered)[-1].quaternion, atol=1e-12)
    assert len({sample.segment_id for sample in samples(late)}) == 1
    assert all(a.stamp_ns < b.stamp_ns for a, b in zip(samples(late), samples(late)[1:]))


@pytest.mark.parametrize('only', [False, True])
def test_retracting_visual_only_checkpoint_removes_publishable_state(only):
    engine = ExternalOdometry(learning=False)
    if not only:
        engine.vision(START, 'native', np.eye(4))
    engine.vision(CAPTURE, 'native', rotation(30.))
    engine.revise_vision(CAPTURE, 'native', None)
    assert CAPTURE not in engine.checkpoints
    assert all(state.stamp != CAPTURE for state in engine.pending_output_states(0))
    assert engine.current.stamp == (0 if only else START)
    if only:
        assert engine.output() is None


def test_authoritative_revision_updates_variance_and_preserves_regular_duplicate_order():
    engine = ExternalOdometry(learning=False)
    first, second = np.eye(4), rotation(30.)
    lower, higher = sorted((first, second), key=lambda matrix: matrix.tobytes())
    engine.vision(START, 'native', higher, np.ones(6))
    assert not engine.vision(START, 'native', lower, np.full(6, 2.))
    assert engine.revise_vision(START, 'native', lower, np.full(6, 2.))
    np.testing.assert_array_equal(engine.current.transform, lower)
    np.testing.assert_array_equal(np.diag(engine.current.covariance), np.full(6, 2.))
    assert engine.revise_vision(START, 'native', lower, np.full(6, 3.))
    np.testing.assert_array_equal(np.diag(engine.current.covariance), np.full(6, 3.))


@pytest.mark.parametrize('projection', ['exact', 'interpolated'])
def test_mount_cache_aging_preserves_dependencies_for_late_replacement(streams, projection):
    """Evicting a live-cache endpoint must neither retract nor freeze its visual."""
    create, now, _ = streams
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, START, 0., epoch='original')
    mount(stream, CAPTURE+100_000_000, 0., epoch='original')
    if projection == 'exact':
        mount(stream, CAPTURE, 0., epoch='original')
    motion(stream, START)
    stream.visual(START, 'native', native_pose(0.))
    motion(stream, CAPTURE)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    pose = stream.fuser.raw_vision[CAPTURE][1].copy()
    now[0] = 2_010_000_000
    for index in range(160):
        mount(stream, 1_300_000_000+index*1_000_000, 0., epoch='original')
        if index % 16 == 15:
            flush(stream)
    assert START not in stream.mounts.packets
    assert CAPTURE in stream.fuser.raw_vision
    np.testing.assert_array_equal(stream.fuser.raw_vision[CAPTURE][1], pose)
    np.testing.assert_array_equal(stream.fuser.raw_vision[START][1], np.eye(4))
    # Even after the arrival cache evicts the exact packet, a delayed future-TF
    # diagnostic remains a non-authoritative miss for that same source.
    mount(stream, START, None, epoch='original', status='unavailable',
          reason='Lookup would require extrapolation into the future.')
    flush(stream)
    np.testing.assert_array_equal(stream.fuser.raw_vision[START][1], np.eye(4))
    changed_stamp = CAPTURE if projection == 'exact' else CAPTURE+100_000_000
    mount(stream, changed_stamp, 30. if projection == 'exact' else 60., epoch='original')
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)
    mount(stream, changed_stamp, None, epoch='original')
    flush(stream)
    assert CAPTURE not in stream.fuser.raw_vision
    mount(stream, changed_stamp, 30. if projection == 'exact' else 60., epoch='original')
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)


def test_interpolation_source_boundary_stays_effective_after_cache_aging(streams):
    create, now, _ = streams
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, START, 0., epoch='old')
    mount(stream, CAPTURE+100_000_000, 60., epoch='old')
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    np.testing.assert_allclose(stream.fuser.current.transform, np.eye(4), atol=1e-14)
    mount(stream, CAPTURE-10_000_000, None, epoch='new', status='unavailable',
          reason='Lookup would require extrapolation into the future.')
    flush(stream)
    assert CAPTURE not in stream.fuser.raw_vision
    now[0] = 2_010_000_000
    for index in range(160):
        mount(stream, 1_500_000_000+index*1_000_000, 0.)
        if index % 16 == 15:
            flush(stream)
    assert CAPTURE not in stream.fuser.raw_vision
    mount(stream, CAPTURE-10_000_000, None, epoch='old', status='unavailable',
          reason='Lookup would require extrapolation into the future.')
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)


@pytest.mark.parametrize('restore', ['exact', 'new_bracket'])
def test_mount_dependency_overflow_requires_complete_numeric_evidence(streams, restore):
    create, now, _ = streams
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, START, 0., epoch='original')
    mount(stream, CAPTURE+100_000_000, 60., epoch='original')
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    for index in range(140):
        mount(stream, START+(index+1)*100_000, None, epoch='original',
              status='unavailable', reason='Lookup would require extrapolation into the future.')
        if index % 10 == 9:
            flush(stream)
    # A saturated source-evidence bracket cannot become silently trustworthy by
    # forgetting its oldest diagnostic. New numeric evidence makes it usable.
    assert CAPTURE not in stream.fuser.raw_vision
    if restore == 'exact':
        mount(stream, CAPTURE, 30., epoch='original')
    else:
        mount(stream, CAPTURE-10_000_000, 30., epoch='original')
        mount(stream, CAPTURE+10_000_000, 30., epoch='original')
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)


def test_native_observation_cache_is_bounded_and_releases_frozen_dependencies(streams, monkeypatch):
    create, now, _ = streams
    monkeypatch.setattr(external_odometry_fusion, '_MAX_EVENTS', 8)
    monkeypatch.setattr(rgbd_external_input, 'REPLAY_MAX_EVENTS', 8, raising=False)
    stream = create()
    now[0] = 2_010_000_000
    for index in range(24):
        stamp = START+index*1_000_000
        stream.visual(stamp, 'native', native_pose(0.))
        flush(stream)
    assert len(stream.native_visual) <= 8
    assert len(stream.pending_visual) <= 8
    retained_stamp = START+23_000_000
    mount(stream, retained_stamp, 0.)
    flush(stream)
    assert retained_stamp in stream.fuser.raw_vision
    for index in range(24, 48):
        stamp = START+index*1_000_000
        mount(stream, stamp, 0.)
        stream.visual(stamp, 'native', native_pose(0.))
        flush(stream)
    frozen = stream.fuser.base
    assert frozen.stamp > retained_stamp
    assert all(stamp > frozen.stamp for stamp in stream.native_visual)
    assert len(stream.native_visual) <= 8
    assert len(stream.mounts._retained) <= 8
    mount(stream, frozen.stamp, 90.)
    stream.visual(frozen.stamp, 'replacement', native_pose(180.))
    flush(stream)
    assert stream.fuser.base is frozen
    np.testing.assert_allclose(stream.fuser.current.transform, np.eye(4), atol=1e-14)


def test_pending_flood_cannot_evict_an_accepted_visual_owner(streams, monkeypatch):
    create, now, _ = streams
    monkeypatch.setattr(external_odometry_fusion, '_MAX_EVENTS', 8)
    monkeypatch.setattr(rgbd_external_input, 'REPLAY_MAX_EVENTS', 8)
    stream = create()
    now[0] = 2_050_000_000
    mount(stream, CAPTURE, 30.)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    for index in range(24):
        stream.visual(2_000_000_000+index*1_000_000, 'native', native_pose(0.))
        flush(stream)
    assert CAPTURE in stream.native_visual
    assert len(stream.native_visual) <= 8
    mount(stream, CAPTURE, 0.)
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], rotation(30.), atol=1e-14)


@pytest.mark.parametrize('order', [(0., 30.), (30., 0.)])
@pytest.mark.parametrize('late_mount', [False, True])
def test_native_duplicate_selection_precedes_mount_projection(streams, order, late_mount):
    """Native 30 wins the pose tie; body-byte ordering under this mount reverses it."""
    create, now, _ = streams
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, CAPTURE, 0. if late_mount else 30.)
    motion(stream, CAPTURE)
    for degrees in order:
        stream.visual(CAPTURE, 'native', native_pose(degrees))
        flush(stream)
    if late_mount:
        mount(stream, CAPTURE, 30.)
        flush(stream)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)
    assert samples(stream)[0].stamp_ns == CAPTURE
    assert len(samples(stream)) == 1


@pytest.mark.parametrize('amendment', ['exact', 'unavailable'])
def test_mount_replay_rebuilds_covariance_and_rejects_an_old_async_learning_result(
        streams, monkeypatch, amendment):
    create, now, _ = streams
    ordered, late = create(learning=True), create(learning=True)
    # ROS workers remain stubbed; the existing learning pool executes real fits.
    monkeypatch.setattr(rgbd_external_input.threading, 'Thread', Thread)
    covariance = np.diag([.01, .04, .09, .16, .25, .36])
    changed = START+500_000_000
    for stream in (ordered, late):
        for step in range(4):
            elapsed = step/4
            stamp = START+round(elapsed*1e9)
            now[0] = stamp+10_000_000
            degrees = 30. if stamp == changed else 0.
            if stream is ordered and stamp == changed:
                mount(stream, stamp, degrees if amendment == 'exact' else None)
            else:
                mount(stream, stamp, 0.)
            stream._receive('/external', NS(data=json.dumps(packet(elapsed, 100.+elapsed))))
            pose = native_pose(degrees, covariance)
            pose['position'][0] = elapsed
            stream.visual(stamp, 'native', pose)
            flush(stream)
    old_key, _, old_future = late.learning_work
    if amendment == 'exact':
        np.testing.assert_allclose(late.fuser.raw_vision[changed][2],
                                   [.0175, .0325, .09, .1825, .2275, .36], atol=1e-14)
    mount(late, changed, 30. if amendment == 'exact' else None)
    flush(late)
    if amendment == 'exact':
        np.testing.assert_allclose(late.fuser.raw_vision[changed][2], np.diag(covariance), atol=1e-14)
    else:
        assert changed not in late.fuser.raw_vision
    old_future.result(timeout=5.)
    late._complete_learning()
    assert late.fuser.current.learner.effective_after_ns is None
    if amendment == 'exact':
        assert late.learning_work[0] != old_key
    else:
        assert late.learning_work is None
    for stream in (ordered, late):
        for _ in range(3):
            if stream.learning_work is None:
                break
            stream.learning_work[2].result(timeout=5.)
            stream._complete_learning()
        assert stream.learning_work is None
    np.testing.assert_allclose(late.fuser.current.transform, ordered.fuser.current.transform, atol=1e-12)
    np.testing.assert_allclose(late.fuser.current.covariance, ordered.fuser.current.covariance, atol=1e-12)
    np.testing.assert_allclose(late.fuser.current.learner.parameters, ordered.fuser.current.learner.parameters,
                               atol=1e-12)
    np.testing.assert_allclose(late.fuser.current.learner.covariance, ordered.fuser.current.learner.covariance,
                               atol=1e-12)
    assert len(late.fuser.current.learner.window) == len(ordered.fuser.current.learner.window)
    for left, right in zip(late.fuser.current.learner.window, ordered.fuser.current.learner.window):
        assert left[:2] == right[:2]
        np.testing.assert_allclose(left[4], right[4], atol=1e-12)
        np.testing.assert_allclose(left[5], right[5], atol=1e-12)
    # A selected native restart retains the body output segment, and unrelated
    # external sources cannot replace the selected motion during this replay.
    now[0] += 250_000_000
    final_stamp = now[0]-10_000_000
    for stream in (ordered, late):
        mount(stream, final_stamp, 0.)
        stream.visual(final_stamp, 'replacement_native', native_pose(90.))
        motion(stream, final_stamp, source='/other')
        flush(stream)
        assert stream.selected == '/external'
        assert final_stamp not in stream.fuser.raw_motion
        assert len({sample.segment_id for sample in samples(stream)}) == 1


@pytest.mark.parametrize('amendment', ['exact', 'unavailable', 'malformed'])
def test_evicted_mount_amendment_still_revises_accepted_camera_evidence(streams, monkeypatch, amendment):
    create, now, _ = streams
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 4)
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, CAPTURE, 0.)
    motion(stream, CAPTURE)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    if amendment == 'malformed':
        stream._receive_mount(NS(data=json.dumps(dict(stamp_ns=CAPTURE, matrix=[[1.]]))))
    else:
        mount(stream, CAPTURE, 30. if amendment == 'exact' else None)
    for index in range(4):
        motion(stream, CAPTURE+index*1_000_000, source='/other')
    assert stream.dropped == 1
    flush(stream)
    if amendment == 'exact':
        np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)
    else:
        assert CAPTURE not in stream.fuser.raw_vision
        assert stream.fuser.current.visual is None
        assert not stream.fuser.pending_output_states(0)
    assert len(samples(stream)) == 1


@pytest.mark.parametrize('surviving_bracket', [False, True])
def test_mount_eviction_ledger_saturation_invalidates_unknown_dependencies(streams, monkeypatch, surviving_bracket):
    create, now, _ = streams
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 4)
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, CAPTURE, 0.)
    motion(stream, CAPTURE)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    mount(stream, CAPTURE, None)
    for index in range(12):
        mount(stream, 2_000_000_000+index*1_000_000, 0.)
    if surviving_bracket:
        mount(stream, START, 0.)
        mount(stream, CAPTURE+100_000_000, 60.)
    assert stream.events.qsize() <= 4
    flush(stream)
    assert CAPTURE not in stream.fuser.raw_vision
    assert len(stream.evicted_mounts) <= 4
    mount(stream, CAPTURE, 30.)
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)
    assert len(samples(stream)) == 1


def test_unrelated_mounts_do_not_repeat_old_camera_projection(streams, monkeypatch):
    create, now, _ = streams
    stream = create()
    now[0] = CAPTURE+110_000_000
    for stamp in (START, CAPTURE):
        mount(stream, stamp, 0.)
        stream.visual(stamp, 'native', native_pose(0.))
    flush(stream)
    projections = []
    project = rgbd_external_input.body_pose_observation

    def observe_projection(pose, transform):
        projections.append(transform)
        return project(pose, transform)

    monkeypatch.setattr(rgbd_external_input, 'body_pose_observation', observe_projection)
    mount(stream, 2_000_000_000, 90.)
    flush(stream)
    assert not projections
    np.testing.assert_allclose(stream.fuser.current.transform, np.eye(4), atol=1e-14)
