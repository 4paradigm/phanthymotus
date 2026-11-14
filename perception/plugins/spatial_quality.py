"""Optional observation evidence and explicit admission decisions."""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass(frozen=True)
class QualityReport:
    contrast: Optional[float] = None
    sharpness: Optional[float] = None
    depth_fraction: Optional[float] = None
    inliers: Optional[int] = None
    residual: Optional[float] = None
    covariance_available: Optional[bool] = None
    native_lost: Optional[bool] = None
    source: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QualityDecision:
    action: str
    reason: str
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.action not in ('accept', 'drop', 'pause', 'new_segment'):
            raise ValueError('unknown quality decision')


class MotionGate:
    """Use only supplied limits in the actual pose reference, never body guesses."""
    def __init__(self, limits=None):
        self.limits = dict(limits or {})
        import math
        for key in ('max_speed_mps','max_angular_rate_radps'):
            value=self.limits.get(key)
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError('invalid declared motion limit: '+key)

    def evaluate(self, previous, current, context, quality):
        from .spatial_contract import validate_tracking_pose
        from .pointcloud_slam_transforms import pose_matrix
        from .pointcloud_slam_odom import decode_slam_odom
        import numpy as np
        try:validate_tracking_pose(current)
        except (ValueError,TypeError) as error:return QualityDecision('pause',str(error))
        if quality.native_lost is True:
            return QualityDecision('pause','native tracking lost')
        diagnostics=tuple(name+' unknown' for name in ('sharpness','depth_fraction','covariance_available')
                          if getattr(quality,name) is None)
        if previous is None:return QualityDecision('accept','valid pose',diagnostics)
        pose,old=decode_slam_odom(current),decode_slam_odom(previous)
        if (pose.segment_id,pose.frame_id,pose.child_frame_id,pose.clock_id) != (old.segment_id,old.frame_id,old.child_frame_id,old.clock_id):
            return QualityDecision('new_segment','pose reference changed',diagnostics)
        dt=(current['vendor']['motus_slam']['stamp_ns']-previous['vendor']['motus_slam']['stamp_ns'])/1e9
        if dt <= 0:return QualityDecision('drop','non-increasing pose time',diagnostics)
        if context is not None and context.time_basis=='receipt_estimated':
            return QualityDecision('accept','receipt motion rate uncertain',diagnostics)
        reference=current['vendor'].get('pose_reference')
        if self.limits.get('reference') != reference:
            return QualityDecision('accept','no motion limit for pose reference',diagnostics)
        a,b=pose_matrix(old),pose_matrix(pose)
        speed=float(np.linalg.norm(b[:3,3]-a[:3,3]))/dt
        angle=float(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))/dt
        if (speed>self.limits.get('max_speed_mps',float('inf')) or
                angle>self.limits.get('max_angular_rate_radps',float('inf'))):
            return QualityDecision('pause','declared motion limit exceeded',diagnostics)
        return QualityDecision('accept','valid pose',diagnostics)
