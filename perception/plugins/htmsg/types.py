"""
htmsg/types.py — Shared data structures for HTMSG system.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Pose6D:
    """6DoF pose: translation + quaternion rotation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4), "y": round(self.y, 4), "z": round(self.z, 4),
            "qw": round(self.qw, 4), "qx": round(self.qx, 4),
            "qy": round(self.qy, 4), "qz": round(self.qz, 4),
        }

    def translation_distance(self, other: "Pose6D") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def rotation_distance(self, other: "Pose6D") -> float:
        """Approximate rotation angle difference in radians."""
        dot = abs(self.qw * other.qw + self.qx * other.qx +
                  self.qy * other.qy + self.qz * other.qz)
        dot = min(dot, 1.0)
        return 2.0 * np.arccos(dot)


@dataclass
class Keyframe:
    """A keyframe in the pose graph."""
    id: int
    pose: Pose6D
    timestamp: float
    scan_context: Optional[np.ndarray] = None  # (RING_NUM x SECTOR_NUM)
    cloud_xyz: Optional[np.ndarray] = None     # Nx3 points (kept temporarily)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pose": self.pose.to_dict(),
            "timestamp": self.timestamp,
            "has_scan_context": self.scan_context is not None,
        }


@dataclass
class SceneNode:
    """A semantic object node in the scene graph."""
    id: str
    label: str
    confidence: float
    centroid: tuple  # (x, y, z) world frame
    bbox_size: tuple  # (w, h, d) meters
    embedding: Optional[np.ndarray] = None  # 512-d CLIP vector
    first_seen: float = 0.0
    last_seen: float = 0.0
    observation_count: int = 0
    keyframe_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "centroid": {"x": self.centroid[0], "y": self.centroid[1], "z": self.centroid[2]},
            "bbox_size": list(self.bbox_size),
            "observation_count": self.observation_count,
        }


@dataclass
class SceneEdge:
    """A spatial relation edge between two scene nodes."""
    source_id: str
    target_id: str
    relation: str  # "on", "next_to", "above", "inside", "near"
    confidence: float = 1.0
