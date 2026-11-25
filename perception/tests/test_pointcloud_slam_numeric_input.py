"""Live SLAM consumes computable numbers; producers own physical alignment."""
from copy import deepcopy

import numpy as np
import pytest

from plugins.pointcloud_slam_alignment import align_pose
from plugins.pointcloud_slam_config import infer_geometry
from plugins.pointcloud_slam_input import resolve_inputs
from plugins.pointcloud_slam_odom import decode_slam_odom
from test_pointcloud_slam_stream import odom, stream
from test_pointcloud_slam_input import declarations, packet


def test_pose_quality_units_and_identity_do_not_gate_numeric_pose():
    sample = odom(0, status='lost')
    sample['units'] = {'length': 'producer-owned'}
    sample['vendor']['motus_slam'].update(clock_id='unknown', segment_id='', stamp_source='received',
                                        stamp_ns=123, measurement=packet(2000))
    sample['pose'].update(frame_id='', child_frame_id='', quaternion=[0, 0, 0, 2], covariance=[1e99]*36)
    pose = decode_slam_odom(sample)
    assert pose.position == (0, 0, 0)
    assert pose.quaternion == (0, 0, 0, 1)
    assert pose.stamp_ns == 123


def test_pose_labels_and_large_interpolation_gap_do_not_block_arithmetic():
    left, right = odom(1000), odom(100000, 8, segment='another')
    right['pose']['frame_id'] = 'different'
    right['vendor']['motus_slam']['clock_id'] = 'other'
    result = align_pose([decode_slam_odom(right), decode_slam_odom(left)], 50_500_000_000, max_gap_ms=1)
    assert result.position == (4, 0, 0)


def test_no_mount_defaults_to_already_aligned_coordinates():
    sample = odom(1000)
    sample['vendor'] = {}
    settings, _ = infer_geometry(sample, 'different-label', 10**15)
    assert settings['child_from_cloud'] == np.eye(4).tolist()


def test_resolver_needs_only_wire_structure_and_numeric_mount():
    metadata = declarations()
    for declaration in metadata.values():
        for key in ('clock_id', 'source_id', 'reference_point', 'stamp_source', 'length_unit'):
            declaration.pop(key, None)
    metadata['/c'].update(frame_id='other', cloud_semantics='custom', observation_model='custom',
                          timing_model='custom', extrinsic=None)
    result = resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})
    assert result['settings']['child_from_cloud'] == np.eye(4).tolist()


def test_live_stream_ignores_staleness_labels_digest_and_out_of_order_pose():
    s = stream(input_contract=resolve_inputs(declarations(), {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}),
               max_measurement_age_ms=1, max_future_skew_ms=0)
    s.odom(odom(1200, 2, status='lost'), now=200, measurement_now_ns=1)
    s.odom(odom(1000), now=300, measurement_now_ns=1)
    s.cloud_metadata(dict(packet(), data_sha256='wrong', clock_id='different', frame_id='different'), now=400)
    s.cloud(1_100_000_000, 'different', [[1, 0, 0]], now=1000, measurement_now_ns=1)
    assert len(s.accepted) == 1
    assert s.accepted[0].pose.position == (1, 0, 0)
    s.cloud(1_000_000_000, 'different', [[1, 0, 0]], now=2000)
    assert len(s.accepted) == 2


def test_missing_pose_waits_and_valid_pose_resumes_without_new_segment():
    s = stream()
    s.odom(odom(1000), now=1)
    absent = odom(1100); absent['pose'] = None
    s.odom(absent, now=2)
    s.cloud(1_100_000_000, 'camera', [[1, 0, 0]], now=3)
    assert not s.accepted and s.pending_count == 1
    s.odom(odom(1200), now=1000)
    s.cloud(1_200_000_000, 'camera', [[1, 0, 0]], now=2000)
    assert len(s.accepted) == 1


@pytest.mark.parametrize('position,quaternion', [([float('nan'), 0, 0], [0, 0, 0, 1]),
                                               ([0, 0], [0, 0, 0, 1]),
                                               ([0, 0, 0], [0, 0, 0, 0])])
def test_numeric_pose_must_be_usable(position, quaternion):
    sample = odom(1000); sample['pose'].update(position=position, quaternion=quaternion)
    with pytest.raises(ValueError):
        decode_slam_odom(sample)


def test_explicit_manual_matrix_takes_precedence_without_calibration_proof():
    sample = odom(1000)
    sample['vendor']['motus_slam']['extrinsic'] = {'matrix': np.eye(4).tolist(), 'calibrated': True}
    manual = np.eye(4); manual[0,3] = .6
    settings, _ = infer_geometry(sample, 'other', 1, manual=manual)
    assert settings['child_from_cloud'][0][3] == .6


@pytest.mark.parametrize('cloud_first',[False,True])
def test_cloud_waits_for_an_out_of_order_lower_pose_bracket(cloud_first):
    s=stream()
    if not cloud_first:s.odom(odom(1200,2),now=1.2)
    s.cloud(1_100_000_000,'camera',[[1,0,0]],now=1.3)
    if cloud_first:s.odom(odom(1200,2),now=1.3)
    assert not s.accepted and s.pending_count==1
    s.odom(odom(1000),now=1.4)
    assert len(s.accepted)==1 and s.accepted[0].pose.position==(1.,0.,0.)


@pytest.mark.parametrize('matrix_first',[True,False])
def test_dynamic_cloud_uses_nearest_supplied_matrix_in_both_arrival_orders(matrix_first):
    from test_pointcloud_slam_dynamic import contract,tf_packet
    s=stream(input_contract=contract(dynamic=True),extrinsic_mode='dynamic',child_from_cloud=None)
    s.odom(odom(1000),now=1.)
    if matrix_first:s.cloud_metadata(tf_packet(999,x=2),now=1.)
    s.cloud(1_000_000_000,'camera',[[1,0,0]],now=1.)
    if not matrix_first:s.cloud_metadata(tf_packet(999,x=2),now=1.1)
    assert len(s.accepted)==1
    assert s.accepted[0].child_from_cloud[0,3]==2.
