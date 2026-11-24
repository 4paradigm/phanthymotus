"""Exercise learned rotation through real asynchronous fusion and publication."""
import json
import threading
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.pointcloud_slam_odom import decode_slam_odom
from plugins.rgbd_external_input import ExternalOdometryInput
from test_external_odometry_fusion import packet
from test_rgbd_odometry_status import sessions  # noqa: F401


def _finish_learning(stream):
    # The real executor runs each fit; replay may request the next checkpoint's
    # fit after applying it. Bound the drain so a replay loop fails the test.
    for _ in range(64):
        if stream.learning_work is None:
            return
        stream.learning_work[2].result(timeout=5.)
        stream._complete_learning()
    pytest.fail('asynchronous learning did not reach the newest checkpoint')


@pytest.mark.parametrize('lateral,learning,source,delayed', [
    (-.3, True, '/external', False),
    (.3, True, '/external', False),
    (.3, True, '/renamed_motion', False),
    (-.3, True, '/external', True),
    (.3, True, '/external', True),
    (.3, False, '/external', False),
    (0., True, '/external', False),
])
def test_translation_mismatch_cannot_turn_stationary_published_pose(
        sessions, monkeypatch, lateral, learning, source, delayed):
    _, now, _ = sessions
    stream = ExternalOdometryInput('/fused/state/odom', dict(
        frame_id='camera', body_from_camera=np.eye(4), external_topic=source,
        external_candidates=[source]), learning=learning)
    # The fixture suppresses ROS spin threads. Restore threads for the real
    # asynchronous parameter executor after constructing the input owner.
    monkeypatch.setattr(threading, 'Thread', Thread)
    visuals = []

    def motion(index, position):
        now[0] = 1_000_000_000 + index * 250_000_000
        stream._receive(source, SimpleNamespace(data=json.dumps(packet(index * .25, position))))

    def visual(index, position):
        return (1_000_000_000 + index * 250_000_000, 'native',
                dict(position=[position, lateral * position, 0.], quaternion=[0., 0., 0., 1.]))

    try:
        position = 0.
        for index in range(31):
            if index:
                position += [1.5, .2][(index - 1) % 2] * .25
            motion(index, position)
            observation = visual(index, position)
            if delayed:
                visuals.append(observation)
            else:
                stream.visual(*observation)
            # Motion and native visual observations share the real input batch;
            # no extra visual-only publication is inserted between samples.
            stream._apply_ready(stream._take_ready())
            _finish_learning(stream)
        if delayed:
            for observation in visuals:
                stream.visual(*observation)
            stream._apply_ready(stream._take_ready())
            _finish_learning(stream)

        learning_info = stream.fuser.current.learner.info()
        if learning:
            assert learning_info['rank'] > 0
            assert learning_info['effective_after_ns'] is not None
        for index in range(31, 51):
            motion(index, position)
            stream._apply_ready(stream._take_ready())
            _finish_learning(stream)

        samples = [decode_slam_odom(json.loads(message)) for message in stream.publisher.messages]
        stamps = [sample.stamp_ns for sample in samples]
        assert all(left < right for left, right in zip(stamps, stamps[1:]))
        assert stamps[-1] == 13_500_000_000
        np.testing.assert_allclose(samples[-1].quaternion, [0., 0., 0., 1.], atol=1e-9)
        np.testing.assert_allclose(learning_info['bias'][3:], 0., atol=1e-10)
        assert stream.last_error is None
    finally:
        stream.close()
