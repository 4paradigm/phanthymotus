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
