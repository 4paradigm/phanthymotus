"""Regression cases from the deployed main-process IMU timing trace."""
import threading

import numpy as np
import pytest

from test_rgbd_imu import aid, packet, session
from test_rgbd_fusion import warmed, source


@pytest.mark.parametrize('gap_ns', [100_096_000, 110_105_856, 150_000_000])
def test_bounded_imu_gap_keeps_capture_time_and_integrates_attitude(gap_ns):
    imu = aid()
    imu.observe(packet(), 1_001_000_000)
    stamp = 1_000_000_000 + gap_ns
    result = imu.observe(packet(stamp, angular_velocity=[0, 0, 1]), stamp+200_000_000)
    assert result['stamp_ns'] == stamp
    np.testing.assert_allclose(result['quaternion'],
                              [0, 0, np.sin(gap_ns/2e9), np.cos(gap_ns/2e9)], atol=1e-9)


def test_larger_imu_gap_still_retires_the_segment():
    imu = aid()
    imu.observe(packet(), 1_001_000_000)
    with pytest.raises(ValueError):
        imu.observe(packet(1_150_000_001), 1_350_000_001)
    assert imu.state == 'lost'


def test_fusion_propagates_bounded_gap_with_original_stamp_and_growing_uncertainty():
    fusion, stamp = warmed()
    before = fusion.filter.P.trace()
    end = stamp + 150_000_000
    fusion.observe_imu(source(end), end+200_000_000)
    fusion._propagate(end)
    assert fusion.filter.stamp == end
    assert fusion.filter.P.trace() > before
    np.testing.assert_allclose(fusion.filter.camera_pose()['position'], [0, 0, 0], atol=1e-8)
    assert np.linalg.eigvalsh(fusion.filter.P).min() >= -1e-12


@pytest.mark.parametrize('arrival', ['late', 'missing', 'stop'])
def test_image_waits_boundedly_for_actual_imu_coverage(monkeypatch, arrival):
    obj, _, runtime, NS, _ = session(monkeypatch)
    elapsed = [0.]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(runtime.time, 'time_ns', lambda: 1_020_000_000+round(elapsed[0]*1e9))
    obj.imu.observe(packet(), 1_001_000_000)
    stopped = [False]
    def wait(seconds):
        # Waiting must release the estimator lock so ingress/status can progress.
        result = []
        def check_lock():
            acquired = obj.lock.acquire(blocking=False)
            result.append(acquired)
            if acquired: obj.lock.release()
        t = threading.Thread(target=check_lock); t.start(); t.join()
        assert result == [True]
        elapsed[0] += seconds
        if arrival == 'stop': stopped[0] = True
        return stopped[0]
    obj.stop_event = NS(is_set=lambda: stopped[0], wait=wait)
    def drain():
        if arrival == 'late' and elapsed[0] >= .04 and obj.imu.last_stamp < 1_020_000_000:
            obj.imu.observe(packet(1_020_000_000), runtime.time.time_ns())
    obj._drain_imu = drain
    ready = obj._wait_for_imu(1_010_000_000)
    assert ready is (arrival == 'late')
    assert elapsed[0] <= .151
    if arrival == 'missing':
        assert elapsed[0] >= .15
        assert obj.imu.last_stamp == 1_000_000_000
