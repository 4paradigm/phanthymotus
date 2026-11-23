"""Final integration regressions for async replay and complete mount batches."""
import json
from threading import Event, Thread
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins import rgbd_external_input
from test_external_odometry_fusion import packet
from test_rgbd_mount_replay import (
    CAPTURE, START, flush, motion, mount, native_pose, samples,
    sessions, streams,  # noqa: F401
)


def finish_learning(stream):
    for _ in range(12):
        if stream.learning_work is None:
            return
        stream.learning_work[2].result(timeout=5.)
        stream._complete_learning()
    pytest.fail('bounded training sequence did not finish its asynchronous solves')


@pytest.mark.parametrize('amendment', ['retract_old_endpoint', 'restore_earlier_endpoint'])
def test_async_learning_revisits_eligible_solves_shifted_by_mount_replay(streams, monkeypatch, amendment):
    create, now, _ = streams
    ordered, late = create(learning=True), create(learning=True)
    monkeypatch.setattr(rgbd_external_input.threading, 'Thread', Thread)
    retract = amendment == 'retract_old_endpoint'
    changed = START+(750_000_000 if retract else 500_000_000)
    count = 7 if retract else 5
    for stream in (ordered, late):
        for step in range(count):
            elapsed = step*.25
            stamp = START+round(elapsed*1e9)
            now[0] = stamp+10_000_000
            unavailable = stamp == changed and ((stream is ordered) if retract else (stream is late))
            mount(stream, stamp, None if unavailable else 0.)
            stream._receive('/external', NS(data=json.dumps(packet(elapsed, 100.+elapsed))))
            pose = native_pose(0.)
            pose['position'][0] = 1.4*elapsed
            stream.visual(stamp, 'native', pose)
            flush(stream)
    original_solve = late.learning_work[0][0]
    assert original_solve == START+(750_000_000 if retract else 1_000_000_000)
    mount(late, changed, None if retract else 0.)
    flush(late)
    if retract:
        assert original_solve not in late.fuser.raw_vision
    else:
        assert original_solve in late.fuser.raw_vision
    replay_starts = []
    rebuild = late.fuser._rebuild

    def record_rebuild(stamp):
        replay_starts.append(stamp)
        return rebuild(stamp)

    monkeypatch.setattr(late.fuser, '_rebuild', record_rebuild)
    for stream in (ordered, late):
        finish_learning(stream)
    assert min(replay_starts) == START+(1_000_000_000 if retract else 750_000_000)
    left, right = ordered.fuser.current.learner, late.fuser.current.learner
    assert len(left.window) == len(right.window) == (5 if retract else 4)
    for a, b in zip(left.window, right.window):
        assert a[:2] == b[:2]
        np.testing.assert_allclose(a[4], b[4], atol=1e-12)
        np.testing.assert_allclose(a[5], b[5], atol=1e-12)
    assert np.exp(left.parameters[0]) == pytest.approx(1.4, abs=1e-6)
    assert left.effective_after_ns == START+(1_500_000_000 if retract else 750_000_000)
    assert left.rank == 6
    assert right.effective_after_ns == left.effective_after_ns
    assert right.rank == left.rank
    np.testing.assert_allclose(right.parameters, left.parameters, atol=1e-10)
    np.testing.assert_allclose(right.covariance, left.covariance, atol=1e-10)
    np.testing.assert_allclose(late.fuser.current.transform, ordered.fuser.current.transform, atol=1e-10)
    # A fresh external-only prediction must use the same recovered calibration.
    elapsed = (count-1)*.25+.75
    now[0] = START+round(elapsed*1e9)+10_000_000
    for stream in (ordered, late):
        stream._receive('/external', NS(data=json.dumps(packet(elapsed, 100.+elapsed))))
        flush(stream)
    np.testing.assert_allclose(late.fuser.current.transform, ordered.fuser.current.transform, atol=1e-9)
    assert samples(late)[-1].stamp_ns == START+round(elapsed*1e9)
    # Replayed calibration agrees immediately at the capture-time estimate.
    # A public pose that already departed from this history can converge over
    # fresh measurements without exposing a single discontinuity to SLAM.
    for step in range(1, 9):
        stamp = elapsed + step*.25
        now[0] = START+round(stamp*1e9)+10_000_000
        for stream in (ordered, late):
            stream._receive('/external', NS(data=json.dumps(packet(stamp, 100.+stamp))))
            flush(stream)
    np.testing.assert_allclose(samples(late)[-1].position, samples(ordered)[-1].position, atol=1e-9)


@pytest.mark.parametrize('existing_owner', [True, False])
def test_large_legal_mount_batch_preserves_boundaries_before_live_cache_eviction(
        streams, monkeypatch, existing_owner):
    create, now, _ = streams
    stream = create()
    now[0] = CAPTURE+110_000_000
    mount(stream, CAPTURE-100_000_000, 0.)
    mount(stream, CAPTURE+100_000_000, 0.)
    motion(stream, CAPTURE)
    if existing_owner:
        stream.visual(CAPTURE, 'native', native_pose(30.))
    else:
        mount(stream, CAPTURE-10_000_000, None)
        # Keep known boundary evidence in the current live cache, without a
        # visual owner yet. Two captures first arrive in the large batch below.
        for index in range(10):
            mount(stream, 3_000_000_000+index*1_000_000, 0.)
    flush(stream)
    before = list(stream.publisher.messages)
    if existing_owner:
        mount(stream, CAPTURE-10_000_000, None)
    else:
        mount(stream, 2_000_000_000, 0.)
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
        for index in range(128 if existing_owner else 124):
            mount(stream, 2_001_000_000+index*1_000_000, 0.)
        if not existing_owner:
            mount(stream, CAPTURE-100_000_000, 0.)
            mount(stream, CAPTURE+100_000_000, 0.)
            for stamp in (CAPTURE, CAPTURE+1_000_000):
                stream.visual(stamp, 'native', native_pose(30.))
    finally:
        release.set()
        consumer.join(2.)
    assert not consumer.is_alive()
    assert not errors
    assert len(ready) == 129
    assert not any(event[0] == 'mount_fault' for event in ready)
    stream._apply_ready(ready)
    assert CAPTURE not in stream.fuser.raw_vision
    assert CAPTURE+1_000_000 not in stream.fuser.raw_vision
    assert stream.publisher.messages == before
    assert all(len(evidence) <= 128 for evidence in stream.mounts._retained.values())
    # Both accepted and first-seen captures remain restorable without a resend.
    mount(stream, CAPTURE-10_000_000, 0.)
    flush(stream)
    expected = [CAPTURE] if existing_owner else [CAPTURE, CAPTURE+1_000_000]
    assert sorted(stream.fuser.raw_vision) == expected
