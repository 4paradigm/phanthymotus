"""Depth assistance runs off the session lock, bounded, and never publishes late."""
import threading
import time

import numpy as np
import pytest

from plugins.rgbd_geometry import GeometryWorker
from test_rgbd_fusion import images, pose, source
from test_rgbd_geometry import render_room
from plugins.rgbd_fusion import ShortTermFusion
from test_rgbd_degradation import fused  # noqa: F401
from test_rgbd_odometry_status import sessions  # noqa: F401

K = np.array([[460., 0, 319.5], [0, 460., 239.5], [0, 0, 1.]])


class BlockingRegistration:
    def __init__(self, result=None, error=None):
        self.release, self.started = threading.Event(), threading.Event()
        self.result, self.error, self.calls = result, error, 0

    def register(self, current, reference, initial):
        self.calls += 1
        self.started.set()
        assert self.release.wait(5)
        if self.error:
            raise self.error
        return self.result


def coasting(registration):
    """Warm, track, then lose vision; geometry is attempted asynchronously."""
    c = ShortTermFusion('run', 'optical', 'capture-clock', K, depth_geometry=True)
    worker = GeometryWorker(registration)
    c.geometry_runner = worker
    color, _ = images()
    reference = render_room(np.eye(4), (480, 640), K)
    for i in range(141):
        t = 1_000_000_000+i*5_000_000
        c.observe_imu(source(t), t)
        if i % 20 == 0:
            c.observe_frame(t, color, reference, pose(), t)
    assert c.state == 'tracking'
    for i in range(1, 41):
        stamp = t+i*5_000_000
        c.observe_imu(source(stamp), stamp)
    return c, worker, stamp, color


def real_result():
    transform = np.eye(4); transform[0, 3] = .025
    current = render_room(transform, (480, 640), K)
    reference = render_room(np.eye(4), (480, 640), K)
    from plugins.rgbd_geometry import DepthRegistration
    return DepthRegistration(K).register(current, reference, np.eye(4)), current


def test_registration_runs_without_blocking_frame_processing_then_applies():
    result, current = real_result()
    registration = BlockingRegistration(result)
    c, worker, stamp, color = coasting(registration)
    try:
        begin = time.monotonic()
        out = c.observe_frame(stamp, color, current, None, stamp, reason='visual missing')
        assert time.monotonic()-begin < 1. and registration.started.wait(2)
        assert out['vendor']['fusion']['aid'] == 'inertial'
        assert out['vendor']['fusion']['quality']['geometry']['reason'] == 'pending'
        registration.release.set()
        deadline = time.monotonic()+2
        while c.aid != 'depth_geometry' and time.monotonic() < deadline:
            c.poll_geometry(stamp+10_000_000)
            time.sleep(.01)
        assert c.aid == 'depth_geometry' and c.state == 'coasting'
        assert c.quality['geometry']['accepted'] is True
    finally:
        registration.release.set(); assert worker.close(2)


def test_result_for_superseded_filter_state_is_discarded():
    result, current = real_result()
    registration = BlockingRegistration(result)
    c, worker, stamp, color = coasting(registration)
    try:
        c.observe_frame(stamp, color, current, None, stamp, reason='visual missing')
        assert registration.started.wait(2)
        for i in range(1, 11):
            c.observe_imu(source(stamp+i*5_000_000), stamp+i*5_000_000)
        c.observe_frame(stamp+50_000_000, color, current, pose(), stamp+50_000_000)
        registration.release.set()
        time.sleep(.1)
        c.poll_geometry(stamp+60_000_000)
        assert c.aid == 'visual' and c.geometry_job is None
        assert c.frames[-2]['geometry'] == 'stale_result_discarded'
        assert c.quality.get('geometry', {}).get('accepted') is not True
    finally:
        registration.release.set(); assert worker.close(2)


@pytest.mark.parametrize('failure', [ValueError('bad normals'), np.linalg.LinAlgError('singular'),
                                     RuntimeError('unexpected')])
def test_registration_failure_skips_assistance_without_fusion_fault(failure):
    _, current = real_result()
    registration = BlockingRegistration(error=failure)
    c, worker, stamp, color = coasting(registration)
    try:
        c.observe_frame(stamp, color, current, None, stamp, reason='visual missing')
        registration.release.set()
        deadline = time.monotonic()+2
        while c.geometry_job is not None and time.monotonic() < deadline:
            c.poll_geometry(stamp+10_000_000)
            time.sleep(.01)
        assert c.state == 'coasting', c.reason
        assert 'registration_error' in c.quality['geometry']['reason']
    finally:
        registration.release.set(); assert worker.close(2)


def test_pending_registration_cannot_defer_inertial_bounds_forever():
    _, current = real_result()
    registration = BlockingRegistration()
    c, worker, stamp, color = coasting(registration)
    try:
        c.observe_frame(stamp, color, current, None, stamp, reason='visual missing')
        c.poll_geometry(stamp+300_000_000)  # wall deadline passed, still computing
        assert c.geometry_job is None
        assert c.frames[-1]['geometry'] == 'timeout_discarded'
    finally:
        registration.release.set(); assert worker.close(2)


@pytest.mark.parametrize('sigma,status', [(.15, 'coasting'), (.201, 'lost')])
def test_geometry_timeout_applies_twenty_cm_position_limit(sigma, status):
    registration = BlockingRegistration()
    c, worker, stamp, color = coasting(registration)
    try:
        c._propagate(stamp)
        c.filter.P = np.zeros((15, 15))
        c.filter.P[:3, :3] = np.eye(3)*sigma**2
        c.observe_frame(stamp, color, c.reference[1], None, stamp, reason='visual missing')
        assert c.geometry_job is not None
        c.poll_geometry(stamp+300_000_000)
        assert c.state == status
        if status == 'lost':
            assert 'exceeds 0.20 m' in c.reason
    finally:
        registration.release.set(); assert worker.close(2)


@pytest.mark.parametrize('force', [False, True])
def test_finished_registration_is_discarded_after_its_deadline(force):
    result, current = real_result()
    registration = BlockingRegistration(result)
    c, worker, stamp, color = coasting(registration)
    try:
        c.observe_frame(stamp, color, current, None, stamp, reason='visual missing')
        registration.release.set()
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            with worker.condition:
                if worker.finished is not None:
                    break
            time.sleep(.001)
        with worker.condition:
            assert worker.finished is not None
        covariance = c.filter.P.copy()
        c.poll_geometry(stamp+300_000_000, force=force)
        assert c.aid == 'inertial'
        assert c.quality['geometry']['reason'] == 'timeout_discarded'
        assert c.frames[-1]['geometry'] == 'timeout_discarded'
        np.testing.assert_array_equal(c.filter.P, covariance)
    finally:
        registration.release.set(); assert worker.close(2)


def test_worker_is_bounded_and_close_prevents_late_delivery():
    registration = BlockingRegistration(result={'late': True})
    worker = GeometryWorker(registration)
    assert worker.submit('a', None, None, None)
    assert registration.started.wait(2)
    assert worker.submit('b', None, None, None)
    assert worker.submit('c', None, None, None)  # replaces pending 'b'
    assert worker.dropped == 1
    assert worker.take('b') == ('missing', None)
    registration.release.set()
    assert worker.close(2)
    assert worker.take('a')[0] in ('missing', 'done')
    assert not worker.submit('d', None, None, None)
    assert worker.take('d') == ('missing', None)
    assert not worker.thread.is_alive()


def test_session_owns_polls_and_joins_its_geometry_worker(fused):
    """Session wiring only; this fixture stubs threads, so off-lock execution
    with a real thread is covered by the fusion-level tests above."""
    make, now, _ = fused
    s = make(depth_geometry=True)
    worker = s.geometry_worker
    assert s.fusion.geometry_runner is worker
    assert make().geometry_worker is None  # no ICP thread without depth_geometry
    polls = []
    s.fusion.poll_geometry = lambda now_ns, force=False: polls.append((now_ns, force))
    s._watchdog()
    assert polls and polls[-1][1] is False
    s.process.returncode = 0
    s.close()
    assert worker.closed and not worker.submit('late', None, None, None)


def test_stuck_geometry_worker_keeps_session_owned_for_explicit_stop(fused):
    make, now, _ = fused
    s = make(depth_geometry=True)
    worker = s.geometry_worker
    close = worker.close
    worker.close = lambda timeout=5.: False
    s.process.returncode = 0
    with pytest.raises(RuntimeError, match='geometry worker'):
        s.close()
    assert not s.node.destroyed  # still owned; explicit stop retries cleanup
    worker.close = close
    s.close()
    assert s.node.destroyed and worker.closed
