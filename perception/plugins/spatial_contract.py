"""Perception-only v2 adapter; the mirrored strict v1 contract stays unchanged."""
from copy import deepcopy

from .pointcloud_slam_contract import validate_tracking_pose as validate_v1_pose
from .spatial_time import FrameContext


def validate_tracking_pose(sample):
    quality = (sample.get('vendor') or {}).get('motus_slam') if isinstance(sample, dict) else None
    if not isinstance(quality, dict) or quality.get('profile', 'motus_slam/1') == 'motus_slam/1':
        return validate_v1_pose(sample)
    if quality.get('profile') != 'motus_slam/2':
        raise ValueError('unsupported SLAM pose profile')
    context = FrameContext.from_dict(quality.get('observation'))
    expected = 'measurement' if context.time_basis == 'source_capture' else 'receipt_estimated'
    if (quality.get('stamp_source') != expected or quality.get('clock_id') != context.normalized_clock_id or
            quality.get('stamp_ns') != context.observation_stamp_ns or
            sample.get('stamp_ms') != context.observation_stamp_ns // 1_000_000):
        raise ValueError('pose and observation time declarations conflict')
    # Reuse v1 geometry/unit validation on a private copy, after validating v2
    # time above. The wire object remains v2; strict old readers still reject it.
    geometry = deepcopy(sample)
    geometry['vendor']['motus_slam']['stamp_source'] = 'measurement'
    validate_v1_pose(geometry)
    return sample
