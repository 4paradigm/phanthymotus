"""Dynamic runtime mounts require actual source clocks and exact capture evidence."""
from copy import deepcopy
import numpy as np
import pytest
from plugins.pointcloud_slam_capture_mount import resolve_capture_mount, CaptureClouds
from test_rgbd_body_mount import declaration, packet
from test_pointcloud_slam_stream import odom, stream

REFERENCE = dict(clock_id='clock', domain_id='host-boot', timestamp_basis='host_realtime')

def camera():
    return dict(schema='motus.camera/1', topic='/cloud', format='sensor/pointcloud2',
        vendor=dict(frame_id='camera_optical', clock_id='camera-clock',
                    clock_reference=deepcopy(REFERENCE), body_mount=declaration()))

def pose():
    p = odom(1000)
    p['pose']['child_frame_id'] = 'base_link'
    p['vendor']['motus_slam']['clock_reference'] = deepcopy(REFERENCE)
    return p


def test_dynamic_geometry_preserves_declaration_and_requires_same_reference():
    c, p = camera(), pose()
    cfg, provenance = resolve_capture_mount(p, c, 'camera_optical')
    assert cfg['child_from_cloud'] is None and cfg['extrinsic_mode'] == 'dynamic'
    assert cfg['capture_mount']['declaration'] == c['vendor']['body_mount']
    assert provenance['clock_accuracy_validated'] is False
    for change in ('missing', 'domain', 'clock', 'frame'):
        changed = deepcopy(c)
        if change == 'missing': del changed['vendor']['clock_reference']
        if change == 'domain': changed['vendor']['clock_reference']['domain_id'] = 'other-host'
        if change == 'clock': changed['vendor']['clock_reference']['clock_id'] = 'other-clock'
        if change == 'frame': changed['vendor']['body_mount']['parent_frame'] = 'other-body'
        assert resolve_capture_mount(p, changed, 'other-name')[0]['extrinsic_mode'] == 'dynamic'


def pairing():
    cfg, _ = resolve_capture_mount(pose(), camera(), 'camera_optical')
    s = stream(cloud_frame='camera_optical', child_frame='base_link',
               child_from_cloud=None, extrinsic_mode='dynamic')
    s.odom(pose(), now=1.)
    later = pose(); later['stamp_ms'] = 1200
    s.odom(later, now=1.2)
    return s, CaptureClouds(cfg['capture_mount'], max_pending=2)


@pytest.mark.parametrize('mount_first', [True, False])
def test_exact_dynamic_pair_preserves_cloud_stamp_and_each_matrix(mount_first):
    s, pair = pairing()
    for i, stamp in enumerate((1_050_000_123, 1_100_000_123)):
        transform = np.eye(4); transform[0, 3] = i + .2
        p = packet(stamp, transform)
        p['provenance'] = {'calibration_kind': 'estimate', 'user_field_checked': True}
        cloud = {'raw': b'unchanged'}
        calls = [lambda: pair.mount(p, s, now=1.2, measurement_now_ns=1_200_000_000),
                 lambda: pair.cloud(stamp, 'camera_optical', cloud, 1, s,
                                    now=1.2, measurement_now_ns=1_200_000_000)]
        for call in calls if mount_first else reversed(calls): call()
        frame = s.accepted[-1]
        assert frame.cloud_stamp_ns == stamp and frame.cloud == cloud
        expected = i + .2 if mount_first else .2
        assert frame.child_from_cloud[0, 3] == expected
        assert frame.capture_context['extrinsic']['provenance']['calibration_kind'] == 'estimate'
    assert len(s.accepted) == 2


def test_missing_numeric_mount_waits_then_nearest_old_matrix_is_used():
    s, pair = pairing()
    pair.cloud(1_050_000_123, 'other-name', [[1,2,3]], 1, s, now=10, measurement_now_ns=1)
    assert not s.accepted
    transform = np.eye(4); transform[0,3] = .4
    pair.mount(packet(1_050_000_124, transform), s, now=100, measurement_now_ns=1)
    assert s.accepted[0].child_from_cloud[0,3] == .4


@pytest.mark.parametrize('raw_depth', [False, True])
@pytest.mark.parametrize('fused', [False, True])
def test_cloud_uses_nearest_numeric_mount_for_either_odom_source(raw_depth, fused):
    from test_pointcloud_slam_stream import external_fusion
    declaration = camera()
    if raw_depth:
        declaration['vendor']['body_mount']['child_frame'] = 'color'
        baseline = np.eye(4); baseline[0, 3] = .03
        declaration['vendor']['color_from_depth'] = baseline.tolist()
    cfg, _ = resolve_capture_mount(external_fusion(1000) if fused else pose(), declaration, 'camera_optical')
    s = stream(cloud_frame='camera_optical', child_frame='body',
               child_from_cloud=None, extrinsic_mode='dynamic')
    pair = CaptureClouds(cfg['capture_mount'], max_pending=2)
    source = external_fusion if fused else odom
    s.odom(source(1000), now=1.)
    s.odom(source(1200), now=1.2)
    first = np.eye(4); first[0, 3] = .1
    pair.mount(packet(1_000_000_000, first), s, now=1.2, measurement_now_ns=1)
    pair.cloud(1_100_000_000, 'camera_optical', [[1, 2, 3]], 1, s, now=1.2, measurement_now_ns=1)
    assert len(s.accepted) == 1
    assert s.accepted[0].capture_context['extrinsic']['stamp_ns'] == 1_000_000_000
    last = np.eye(4); last[0, 3] = .3
    pair.mount(packet(1_200_000_000, last), s, now=1.2, measurement_now_ns=1)
    assert s.accepted[0].child_from_cloud[0, 3] == pytest.approx(.13 if raw_depth else .1)


def test_changed_clock_and_mount_labels_do_not_retire_numeric_data():
    s,pair = pairing()
    changed = pose(); changed['vendor']['motus_slam']['clock_reference']['domain_id'] = 'other'
    pair.validate_pose(changed,s)
    value = packet(1_100_000_123, epoch='other'); value.update(clock_id='other',version='other',child_frame='other')
    pair.mount(value,s,now=100,measurement_now_ns=1)
    pair.cloud(1_100_000_123,'other',[[1,2,3]],1,s,now=100,measurement_now_ns=1)
    assert len(s.accepted) == 1


@pytest.mark.parametrize('matrix', [[[1]], [[float('nan')]*4]*4])
def test_malformed_numeric_mount_does_not_admit_cloud_and_later_data_recovers(matrix):
    s,pair = pairing()
    bad = packet(1_100_000_123,matrix)
    with pytest.raises(ValueError): pair.mount(bad,s,now=1,measurement_now_ns=1)
    pair.mount(packet(1_100_000_123),s,now=100,measurement_now_ns=1)
    pair.cloud(1_100_000_123,'camera_optical',[[1,2,3]],1,s,now=100,measurement_now_ns=1)
    assert len(s.accepted) == 1


def test_mount_wait_preserves_original_receipt_without_expiry():
    s,pair = pairing()
    pair.cloud(1_100_000_123,'camera_optical',[[1,2,3]],1,s,now=1,measurement_now_ns=1)
    pair.mount(packet(1_100_000_123),s,now=10000,measurement_now_ns=1)
    assert len(s.accepted) == 1


@pytest.mark.parametrize('change',['missing_reference','other_clock','receipt','stamp'])
def test_observation_labels_do_not_gate_numeric_cloud_stamp(change):
    from types import SimpleNamespace
    from plugins.pointcloud_slam_capture_mount import validate_cloud_clock
    context = SimpleNamespace(normalized_clock_id=change,observation_stamp_ns=1,time_basis=change)
    validate_cloud_clock(context,'different',100,None)
