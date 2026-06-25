"""
plugins/htmsg/loop_closure.py — Loop closure detection (Phase 2).

Uses Scan Context for geometric loop closure detection.
Will add visual features (ORB/SuperPoint) in future iterations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .pose_graph import ScanContext
from .types import Keyframe


# Matching thresholds
SC_DIST_THRESHOLD = 0.15  # Cosine distance threshold (lower = stricter)
TOP_K_RING = 10           # Ring Key candidates for refinement


def detect_loop_closure(current_kf: Keyframe, history: list[Keyframe],
                        min_id_gap: int = 10) -> Optional[tuple[Keyframe, float]]:
    """Check if current keyframe forms a loop closure with any historical keyframe.

    Args:
        current_kf: The current keyframe with scan_context
        history: List of historical keyframes to search
        min_id_gap: Minimum keyframe ID gap to consider (avoid matching neighbors)

    Returns:
        (matched_keyframe, score) if loop detected, None otherwise.
    """
    if current_kf.scan_context is None:
        return None

    # Filter candidates by ID gap
    candidates = [kf for kf in history
                  if kf.scan_context is not None
                  and abs(kf.id - current_kf.id) >= min_id_gap]

    if not candidates:
        return None

    current_sc = current_kf.scan_context
    current_ring_key = ScanContext.ring_key(current_sc)

    # Step 1: Ring Key coarse filtering
    ring_keys = np.array([ScanContext.ring_key(kf.scan_context) for kf in candidates],
                         dtype=np.float32)
    ring_dists = _cosine_distance_batch(current_ring_key, ring_keys)
    top_k_indices = np.argsort(ring_dists)[:TOP_K_RING]

    # Step 2: Fine matching with rotation alignment
    best_score = float("inf")
    best_kf = None

    for idx in top_k_indices:
        kf = candidates[idx]
        dist = ScanContext.distance(current_sc, kf.scan_context)
        if dist < best_score:
            best_score = dist
            best_kf = kf

    # Step 3: Threshold check
    if best_score < SC_DIST_THRESHOLD and best_kf is not None:
        return (best_kf, best_score)

    return None


def _cosine_distance_batch(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Compute cosine distance between query and each row in keys."""
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-9:
        return np.ones(len(keys), dtype=np.float32)
    k_norms = np.linalg.norm(keys, axis=1)
    k_norms[k_norms < 1e-9] = 1e-9
    cos_sim = (keys @ query) / (k_norms * q_norm)
    return 1.0 - cos_sim
