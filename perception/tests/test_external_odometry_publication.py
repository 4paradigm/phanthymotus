"""Capture-time fusion estimates survive real input batching and JSON publication."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins.rgbd_external_input import ExternalOdometryInput
from plugins.pointcloud_slam_odom import decode_slam_odom
from plugins.rgbd_imu import rotation_from_quaternion
from test_rgbd_odometry_status import sessions  # noqa: F401


START_NS = 10_000_000_000
STEP_NS = 100_000_000


def measured_pose(index, spatial):
    # Rz(yaw) Ry(pitch) Rx(roll): changing axes exercises noncommuting rotations.
    roll, pitch, yaw = (.01*index, .02*index, .04*index) if spatial else (0., 0., .04*index)
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw])/2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw])/2)
    quaternion = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                  cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
    position = [.03*index, .015*np.sin(.12*index), -.01*index] if spatial else [0., 0., 0.]
    return dict(position=position, quaternion=quaternion)


def receive_pose(stream, stamp, pose):
    packet = dict(schema='motus.odom/1', stamp_ns=stamp,
                  pose=dict(pose, frame_id='external', child_frame_id='body'), twist=[0.]*6)
    stream._receive('/external', NS(data=json.dumps(packet)))


def drain(stream):
    stream._apply_ready(stream._take_ready())


@pytest.mark.parametrize('spatial', [False, True], ids=['yaw', 'noncommuting_3d'])
def test_equal_information_fresh_publication_is_independent_of_input_batching(sessions, spatial):
    _, now, _ = sessions
    runs = []
    for together in (True, False):
        stream = ExternalOdometryInput('/fused/state/odom', dict(
            frame_id='body', external_topic='/external', external_candidates=['/external']), learning=False)
        try:
            for index in range(51):
                now[0] = START_NS + index*STEP_NS
                receive_pose(stream, now[0], measured_pose(index, spatial))
                if not together:
                    drain(stream)
                published = len(stream.publisher.messages)
                stream.visual(now[0], 'native', measured_pose(0, spatial))
                drain(stream)
                if not together and index:
                    # Corrections at already published captures cannot emit an old stamp.
                    assert len(stream.publisher.messages) == published

            corrected = stream.fuser.current.transform.copy()
            corrected_covariance = stream.fuser.current.covariance.copy()
            # Both runs now have identical information. A held pose supplies a
            # fresh measurement stamp without adding another relative motion.
            now[0] = START_NS + 51*STEP_NS
            receive_pose(stream, now[0], measured_pose(50, spatial))
            drain(stream)
            packets = [json.loads(message) for message in stream.publisher.messages]
            samples = [decode_slam_odom(packet) for packet in packets]
            assert [sample.stamp_ns for sample in samples] == [START_NS+i*STEP_NS for i in range(52)]
            assert len({sample.segment_id for sample in samples}) == 1
            assert samples[-1].pose_reference == 'body'
            assert samples[-1].child_frame_id == 'body'
            assert packets[-1]['vendor']['source'] == 'rgbd_external_fusion'
            assert packets[-1]['vendor']['external_fusion']['update'] == 'external'
            assert packets[-1]['vendor']['external_fusion']['observed'] == [True]*6
            np.testing.assert_allclose(packets[-1]['twist'], np.zeros(6), atol=1e-12)
            assert stream.dropped == 0
            assert stream.last_error is None
            runs.append((corrected, corrected_covariance, stream.fuser.current.transform.copy(), samples[-1]))
        finally:
            stream.close()

    joined, split = runs
    np.testing.assert_allclose(split[0], joined[0], atol=1e-12)
    np.testing.assert_allclose(split[1], joined[1], atol=1e-12)
    np.testing.assert_allclose(split[2], joined[2], atol=1e-12)
    np.testing.assert_allclose(split[3].position, joined[3].position, atol=1e-12)
    np.testing.assert_allclose(rotation_from_quaternion(split[3].quaternion),
                               rotation_from_quaternion(joined[3].quaternion), atol=1e-12)
    np.testing.assert_allclose(split[3].covariance, joined[3].covariance, atol=1e-12)
    for _, _, state, sample in runs:
        np.testing.assert_allclose(sample.position, state[:3, 3], atol=1e-12)
        np.testing.assert_allclose(rotation_from_quaternion(sample.quaternion), state[:3, :3], atol=1e-12)
