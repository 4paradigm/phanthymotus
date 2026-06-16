"""
plugins/htmsg/pose_graph.py — Pose graph manager with keyframe selection and Scan Context.

Manages:
- Keyframe insertion based on distance/angle thresholds
- Scan Context fingerprint generation for each keyframe (loop closure prep)
- Keyframe storage (in-memory, with optional SQLite persistence)
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

from .types import Pose6D, Keyframe


class ScanContext:
    """Scan Context descriptor for place recognition.

    Based on the algorithm from scan_context.py in phanthymotus-driver,
    reimplemented here for the perception stack.
    """

    RING_NUM = 20
    SECTOR_NUM = 60
    MAX_RADIUS = 20.0

    @staticmethod
    def make(points_xyz: np.ndarray) -> np.ndarray:
        """Generate Scan Context descriptor (RING_NUM x SECTOR_NUM) from Nx3 points.

        Each bin stores the maximum height (z) of points in that ring-sector cell.
        """
        if points_xyz.shape[0] == 0:
            return np.zeros((ScanContext.RING_NUM, ScanContext.SECTOR_NUM), dtype=np.float32)

        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]

        # Polar coordinates
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x) + np.pi  # [0, 2pi]

        # Filter out-of-range
        valid = r < ScanContext.MAX_RADIUS
        r, theta, z = r[valid], theta[valid], z[valid]

        if len(r) == 0:
            return np.zeros((ScanContext.RING_NUM, ScanContext.SECTOR_NUM), dtype=np.float32)

        # Bin assignment
        ring_idx = np.clip(
            (r / ScanContext.MAX_RADIUS * ScanContext.RING_NUM).astype(np.int32),
            0, ScanContext.RING_NUM - 1
        )
        sector_idx = np.clip(
            (theta / (2 * np.pi) * ScanContext.SECTOR_NUM).astype(np.int32),
            0, ScanContext.SECTOR_NUM - 1
        )

        # Build descriptor: max z per bin
        sc = np.full((ScanContext.RING_NUM, ScanContext.SECTOR_NUM), -np.inf, dtype=np.float32)
        linear_idx = ring_idx * ScanContext.SECTOR_NUM + sector_idx
        np.maximum.at(sc.ravel(), linear_idx, z.astype(np.float32))
        sc[sc == -np.inf] = 0.0

        return sc

    @staticmethod
    def ring_key(sc: np.ndarray) -> np.ndarray:
        """Ring Key = mean per ring (rotation invariant)."""
        return sc.mean(axis=1).astype(np.float32)

    @staticmethod
    def distance(sc_a: np.ndarray, sc_b: np.ndarray) -> float:
        """Compute distance between two Scan Contexts (with rotation alignment)."""
        min_dist = float("inf")
        for shift in range(ScanContext.SECTOR_NUM):
            sc_b_shifted = np.roll(sc_b, shift, axis=1)
            dist = ScanContext._column_cosine_distance(sc_a, sc_b_shifted)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    @staticmethod
    def _column_cosine_distance(sc_a: np.ndarray, sc_b: np.ndarray) -> float:
        """Mean column-wise cosine distance."""
        num_sectors = sc_a.shape[1]
        total_dist = 0.0
        valid_cols = 0
        for j in range(num_sectors):
            col_a = sc_a[:, j]
            col_b = sc_b[:, j]
            norm_a = np.linalg.norm(col_a)
            norm_b = np.linalg.norm(col_b)
            if norm_a < 1e-9 or norm_b < 1e-9:
                continue
            cos_sim = np.dot(col_a, col_b) / (norm_a * norm_b)
            total_dist += 1.0 - cos_sim
            valid_cols += 1
        return total_dist / max(valid_cols, 1)


class PoseGraphManager:
    """Manages keyframes and pose graph for HTMSG.

    Thread-safe: called from odometry callback thread.
    """

    def __init__(self, dist_thresh: float = 1.0, angle_thresh: float = 0.35):
        self._dist_thresh = dist_thresh
        self._angle_thresh = angle_thresh
        self._keyframes: list[Keyframe] = []
        self._lock = threading.Lock()
        self._next_id = 0
        self._last_kf_pose: Optional[Pose6D] = None

    @property
    def keyframe_count(self) -> int:
        with self._lock:
            return len(self._keyframes)

    def maybe_add_keyframe(self, pose: Pose6D, timestamp: float,
                           cloud_xyz: Optional[np.ndarray] = None) -> Optional[Keyframe]:
        """Check if a new keyframe should be inserted based on motion thresholds.

        Returns the new Keyframe if inserted, None otherwise.
        """
        with self._lock:
            if self._last_kf_pose is not None:
                dist = pose.translation_distance(self._last_kf_pose)
                angle = pose.rotation_distance(self._last_kf_pose)
                if dist < self._dist_thresh and angle < self._angle_thresh:
                    return None

            # Create keyframe
            kf_id = self._next_id
            self._next_id += 1

            # Generate Scan Context if we have a point cloud
            sc = None
            if cloud_xyz is not None and len(cloud_xyz) >= 100:
                sc = ScanContext.make(cloud_xyz)

            kf = Keyframe(
                id=kf_id,
                pose=pose,
                timestamp=timestamp,
                scan_context=sc,
                cloud_xyz=None,  # Don't keep full cloud in memory long-term
            )

            self._keyframes.append(kf)
            self._last_kf_pose = pose

            # Memory management: only keep last 1000 keyframes in memory
            if len(self._keyframes) > 1000:
                # Drop cloud references from old keyframes
                self._keyframes = self._keyframes[-1000:]

            return kf

    def get_keyframes(self, last_n: int = 0) -> list[Keyframe]:
        """Get keyframes. If last_n > 0, return only the most recent N."""
        with self._lock:
            if last_n > 0:
                return list(self._keyframes[-last_n:])
            return list(self._keyframes)

    def get_latest_pose(self) -> Optional[Pose6D]:
        """Get the latest keyframe pose."""
        with self._lock:
            if self._keyframes:
                return self._keyframes[-1].pose
            return None
