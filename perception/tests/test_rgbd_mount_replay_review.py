"""Regression coverage for mount ingress, ordering, and retained dependencies."""
import json
from threading import Event, Thread
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins import rgbd_external_input
from plugins.rgbd_body_mount import CaptureMounts
from test_rgbd_mount_replay import (
    CAPTURE, START, flush, motion, mount, native_pose, rotation, samples,
    sessions, streams,  # noqa: F401
)


def bad_matrix(kind):
    matrix = np.eye(4)
    if kind == 'singular':
        return np.zeros((4, 4))
    if kind == 'reflection':
        matrix[2, 2] = -1.
    elif kind == 'scale':
        matrix[0, 0] = 2.
    elif kind == 'projective':
        matrix[3, 0] = .01
    return matrix


@pytest.mark.parametrize('kind', ['singular', 'reflection', 'scale', 'projective'])
def test_invalid_rigid_mount_retracts_evidence_and_allows_following_visual(streams, kind):
    create, now, _ = streams
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, CAPTURE, 0.)
    motion(stream, CAPTURE)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    newer = CAPTURE+100_000_000
    stream._receive_mount(NS(data=json.dumps(dict(stamp_ns=CAPTURE, matrix=bad_matrix(kind).tolist()))))
    mount(stream, newer, 0.)
    stream.visual(newer, 'native', native_pose(0.))
    flush(stream)
    assert CAPTURE not in stream.fuser.raw_vision
    np.testing.assert_allclose(stream.fuser.raw_vision[newer][1], np.eye(4), atol=1e-14)
    assert stream.last_error is not None
    assert samples(stream)[-1].stamp_ns == newer


def test_float32_composed_rotation_remains_a_usable_mount():
    observer = CaptureMounts(dict(topic='/mount'), 'camera', None)
    # Composition produces a small ordinary float32 orthogonality residual.
    matrix = np.eye(4, dtype=np.float32)
    increment = rotation(.7).astype(np.float32)
    for _ in range(200):
        matrix = matrix @ increment
    matrix[:3, 3] = [.1, -.2, .3]
    observer.observe(dict(stamp_ns=CAPTURE, matrix=matrix.tolist()), CAPTURE)
    np.testing.assert_allclose(observer.matrix_at(CAPTURE), matrix, atol=1e-7)


@pytest.mark.parametrize('evicted', [False, True])
def test_malformed_mount_json_is_an_ordered_fault_even_if_evicted(streams, monkeypatch, evicted):
    create, now, _ = streams
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 4)
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, CAPTURE, 0.)
    motion(stream, CAPTURE)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    stream._receive_mount(NS(data='{'))
    if evicted:
        for index in range(4):
            motion(stream, CAPTURE+index*1_000_000, source='/other')
    flush(stream)
    assert CAPTURE not in stream.fuser.raw_vision
    assert stream.fuser.current.visual is None
    assert stream.last_error is not None
    assert len(samples(stream)) == 1
    mount(stream, CAPTURE, 30.)
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)


@pytest.mark.parametrize('boundary', ['unavailable', 'malformed'])
@pytest.mark.parametrize('boundary_last', [True, False])
def test_ingress_order_survives_eviction_after_consumer_takes_head(streams, monkeypatch, boundary, boundary_last):
    create, now, _ = streams
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 2)
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, CAPTURE, 0.)
    motion(stream, CAPTURE)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    def boundary_event():
        if boundary == 'unavailable':
            mount(stream, CAPTURE, None)
        else:
            stream._receive_mount(NS(data='{'))

    # Consumer removes A; producer evicts B while A is outside the queue.
    # Both orderings matter: a newer numeric correction must also restore data.
    if boundary_last:
        mount(stream, CAPTURE, 30.)
        boundary_event()
    else:
        boundary_event()
        mount(stream, CAPTURE, 30.)
    took_head, release = Event(), Event()
    original_get = stream.events.get

    def pause_after_head(*args, **kwargs):
        item = original_get(*args, **kwargs)
        if kwargs.get('timeout') is not None:
            took_head.set()
            assert release.wait(2.)
        return item

    monkeypatch.setattr(stream.events, 'get', pause_after_head)
    ready, errors = [], []

    def consume():
        try:
            ready.extend(stream._take_ready())
        except BaseException as error:
            errors.append(error)

    consumer = Thread(target=consume)
    consumer.start()
    try:
        assert took_head.wait(2.)
        for index in range(2):
            motion(stream, CAPTURE+index*1_000_000, source='/other')
    finally:
        release.set()
        consumer.join(2.)
    assert not consumer.is_alive()
    assert not errors
    stream._apply_ready(ready)
    if boundary_last:
        assert CAPTURE not in stream.fuser.raw_vision
    else:
        np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)
    assert len(samples(stream)) == 1


def awaiting(stream, stamp, epoch=None):
    metadata = {} if epoch is None else dict(epoch=epoch)
    mount(stream, stamp, None, status='unavailable',
          reason='Lookup would require extrapolation into the future.', **metadata)


def age_live_mount_cache(stream, now):
    now[0] = 2_210_000_000
    for index in range(140):
        mount(stream, 2_000_000_000+index*1_000_000, 0.)
        if index % 10 == 9:
            flush(stream)


def test_widened_bracket_keeps_source_boundary_outside_previous_nearest_endpoints(streams):
    create, now, _ = streams
    stream = create()
    now[0] = START+60_000_000
    mount(stream, START-50_000_000, 0., epoch='B')
    awaiting(stream, START-30_000_000, epoch='C')
    mount(stream, START-20_000_000, 0., epoch='A')
    mount(stream, START+50_000_000, 0.)
    stream.visual(START, 'native', native_pose(0.))
    flush(stream)
    assert START in stream.fuser.raw_vision
    age_live_mount_cache(stream, now)
    awaiting(stream, START-20_000_000, epoch='B')
    mount(stream, START-50_000_000, 0., epoch='B')
    flush(stream)
    assert START not in stream.fuser.raw_vision
    assert stream.mounts.matrix_at(START) is None


@pytest.mark.parametrize('temporary_resolution', ['exact', 'narrow_bracket'])
def test_missing_evidence_survives_temporary_exact_or_narrow_bracket(streams, temporary_resolution):
    create, now, _ = streams
    stream = create()
    now[0] = START+110_000_000
    mount(stream, START-100_000_000, 0., epoch='A')
    mount(stream, START+100_000_000, 0., epoch='A')
    stream.visual(START, 'native', native_pose(0.))
    flush(stream)
    for index in range(140):
        awaiting(stream, START-90_000_000+index*100_000, epoch='C' if index == 0 else None)
        if index % 10 == 9:
            flush(stream)
    assert START not in stream.fuser.raw_vision
    if temporary_resolution == 'exact':
        mount(stream, START, 0., epoch='A')
    else:
        mount(stream, START-10_000_000, 0., epoch='A')
        mount(stream, START+10_000_000, 0., epoch='A')
    flush(stream)
    assert START in stream.fuser.raw_vision
    age_live_mount_cache(stream, now)
    if temporary_resolution == 'exact':
        awaiting(stream, START, epoch='B')
    else:
        awaiting(stream, START-10_000_000, epoch='B')
        awaiting(stream, START+10_000_000, epoch='B')
    mount(stream, START-100_000_000, 0., epoch='B')
    mount(stream, START+100_000_000, 0., epoch='B')
    flush(stream)
    assert START not in stream.fuser.raw_vision
    assert len(stream.mounts._retained[START]) <= 128


def test_idle_retraction_keeps_previously_accepted_owner_until_replay_base_freezes(streams):
    create, now, _ = streams
    stream = create()
    now[0] = CAPTURE+10_000_000
    mount(stream, CAPTURE, 0.)
    motion(stream, CAPTURE)
    stream.visual(CAPTURE, 'native', native_pose(30.))
    flush(stream)
    frozen = stream.fuser.base
    now[0] = CAPTURE+rgbd_external_input.REPLAY_HISTORY_NS+2_000_000_000
    mount(stream, CAPTURE, None)
    flush(stream)
    assert CAPTURE not in stream.fuser.raw_vision
    assert CAPTURE in stream.native_visual
    mount(stream, CAPTURE, 30.)
    flush(stream)
    np.testing.assert_allclose(stream.fuser.raw_vision[CAPTURE][1], np.eye(4), atol=1e-14)
    assert stream.fuser.base is frozen
    assert len(samples(stream)) == 1
    assert stream.accepted_visual == {CAPTURE}
    # Once fresh measurements advance the immutable base, the extra lifetime
    # marker and its native owner are released together.
    for offset in (0, 100_000_000):
        now[0] += offset
        motion(stream, now[0]-10_000_000)
        flush(stream)
    assert stream.fuser.base.stamp >= CAPTURE
    assert CAPTURE not in stream.native_visual
    assert CAPTURE not in stream.accepted_visual
    assert samples(stream)[-1].stamp_ns > CAPTURE
