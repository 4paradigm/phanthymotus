"""Dynamic runtime mounts require actual source clocks and exact capture evidence."""
from copy import deepcopy
import numpy as np
import pytest
from plugins.pointcloud_slam_capture_mount import resolve_capture_mount, CaptureClouds
from spatial_test_helpers import declaration, packet
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
        bad = deepcopy(c)
        if change == 'missing': del bad['vendor']['clock_reference']
        if change == 'domain': bad['vendor']['clock_reference']['domain_id'] = 'other-host'
        if change == 'clock': bad['vendor']['clock_reference']['clock_id'] = 'other-clock'
        if change == 'frame': bad['vendor']['body_mount']['parent_frame'] = 'other-body'
        with pytest.raises(ValueError): resolve_capture_mount(p, bad, 'camera_optical')


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
        assert frame.child_from_cloud[0, 3] == i + .2
        assert frame.capture_context['extrinsic']['provenance']['calibration_kind'] == 'estimate'
    assert len(s.accepted) == 2


def test_missing_nearest_and_expired_mount_never_admit_cloud():
    s, pair = pairing()
    pair.mount(packet(1_050_000_124), s, now=1.2, measurement_now_ns=1_200_000_000)
    pair.cloud(1_050_000_123, 'camera_optical', [[1, 2, 3]], 1, s,
               now=1.2, measurement_now_ns=1_200_000_000)
    assert not s.accepted
    pair.mount(packet(1_100_000_123), s, now=1.7, measurement_now_ns=1_700_000_000)
    pair.cloud(1_100_000_123, 'camera_optical', [[1, 2, 3]], 1, s,
               now=1.7, measurement_now_ns=1_700_000_000)
    assert not s.accepted


def test_changed_identity_clears_waiting_clouds_and_fails_closed():
    s, pair = pairing()
    pair.cloud(1_050_000_123, 'camera_optical', [[1, 2, 3]], 1, s,
               now=1.2, measurement_now_ns=1_200_000_000)
    with pytest.raises(ValueError):
        pair.mount(packet(1_050_000_123, epoch='other'), s,
                   now=1.2, measurement_now_ns=1_200_000_000)
    pair.mount(packet(1_050_000_123), s, now=1.2, measurement_now_ns=1_200_000_000)
    assert not s.accepted


def test_clock_reference_change_retires_mount_evidence():
    s, pair = pairing()
    changed = pose()
    changed['vendor']['motus_slam']['clock_reference']['domain_id'] = 'other-host'
    with pytest.raises(ValueError, match='clock reference'):
        pair.validate_pose(changed, s)
    assert pair.mounts.state == 'lost'
    pair.mount(packet(1_100_000_123), s, now=1.2, measurement_now_ns=1_200_000_000)
    assert not s.accepted


def test_mount_wait_does_not_restart_cloud_receipt_budget():
    cfg, _ = resolve_capture_mount(pose(), camera(), 'camera_optical')
    s = stream(cloud_frame='camera_optical', child_frame='base_link',
               child_from_cloud=None, extrinsic_mode='dynamic', max_age_s=.2)
    pair = CaptureClouds(cfg['capture_mount'], max_pending=2)
    s.odom(pose(), now=1.)
    pair.cloud(1_100_000_123, 'camera_optical', [[1, 2, 3]], 1, s,
               now=1., measurement_now_ns=1_100_000_123)
    p = pose(); p['stamp_ms'] = 1050
    s.odom(p, now=1.15)
    pair.mount(packet(1_100_000_123), s, now=1.19, measurement_now_ns=1_290_000_123)
    p['stamp_ms'] = 1200
    s.odom(p, now=1.21)
    assert not s.accepted


@pytest.mark.parametrize('changes', [dict(clock_id='other'), dict(version='changed'),
    dict(child_frame='other'), dict(stamp_ns=1_300_000_000)])
def test_invalid_mount_does_not_reach_native_admission(changes):
    s, pair = pairing()
    p = packet(1_100_000_123); p.update(changes)
    with pytest.raises(ValueError):
        pair.mount(p, s, now=1.2, measurement_now_ns=1_200_000_000)
    assert not s.accepted and pair.mounts.state == 'lost'


@pytest.mark.parametrize('change', ['missing_reference', 'other_clock', 'receipt', 'stamp'])
def test_observation_clock_alias_requires_resolved_capture_evidence(change):
    from dataclasses import replace
    from plugins.spatial_time import TimePolicy
    from plugins.pointcloud_slam_capture_mount import validate_cloud_clock
    cfg, _ = resolve_capture_mount(pose(), camera(), 'camera_optical')
    context = TimePolicy('ingress').select(dict(stamp_ns=1_000_000_000,
        clock_id='camera-clock', timestamp_semantics='source_capture_mapped'),
        1_000_000_000, 1_000_000_000)
    mount = cfg['capture_mount']
    if change == 'missing_reference': mount = None
    if change == 'other_clock': context = replace(context, normalized_clock_id='other')
    if change == 'receipt': context = TimePolicy('ingress').select({}, 1_000_000_000, 1_000_000_000)
    if change == 'stamp': context = replace(context, observation_stamp_ns=1_000_000_001)
    with pytest.raises(ValueError):
        validate_cloud_clock(context, cfg['clock_id'], 1_000_000_000, mount)
