"""Build validated external-odometry samples; no pose estimation."""
from copy import deepcopy
from .spatial_contract import validate_tracking_pose


def make_sample(status, stamp_ns, segment, clock, pose=None, *, reason=None, context=None):
    valid = status == 'tracking'
    sample = {'schema': 'motus.odom/1', 'stamp_ms': stamp_ns // 1_000_000,
              'units': {'length': 'm', 'linear': 'm/s', 'angular': 'rad/s'},
              'frame': 'body', 'twist': [None] * 6,
              'pose': deepcopy(pose) if valid else None,
              'vendor': {'accuracy_validated': False, 'reason': reason,
                         'motus_slam': {'status': status, 'stamp_source': 'measurement' if valid else 'unknown',
                                        'segment_id': segment, 'clock_id': clock, 'stamp_ns': stamp_ns}}}
    if context is not None:
        quality = sample['vendor']['motus_slam']
        quality.update(profile='motus_slam/2', observation=context.to_dict())
        if valid and context.time_basis == 'receipt_estimated':
            quality['stamp_source'] = 'receipt_estimated'
    if valid:
        validate_tracking_pose(sample)
    return sample
