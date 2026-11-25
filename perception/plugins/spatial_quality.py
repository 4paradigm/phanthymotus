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
    """Check numeric pose format; producer motion limits remain descriptive."""
    def __init__(self, limits=None):
        self.limits = dict(limits or {})
    def evaluate(self, previous, current, context, quality):
        from .spatial_contract import validate_tracking_pose
        try:validate_tracking_pose(current)
        except (ValueError,TypeError) as error:return QualityDecision('pause',str(error))
        diagnostics=tuple(name+' unknown' for name in ('sharpness','depth_fraction','covariance_available')
                          if getattr(quality,name,None) is None)
        return QualityDecision('accept','numeric pose format',diagnostics)
