"""ByteTrack association adapted to VOP's TensorRT detections and real frame time.

Algorithm reference: https://github.com/FoundationVision/ByteTrack
"""
from __future__ import annotations

from itertools import count

import lap
import numpy as np


LOW_CONFIDENCE = 0.1
_IDENTITY = np.eye(8)
_OBSERVE = _IDENTITY[:4]


class _Track:
    def __init__(self, box: np.ndarray, class_id: int, now: float):
        self.class_id = class_id
        self.last_seen = self.predicted_at = now
        self.active = True
        self.mean = np.r_[self._measurement(box), np.zeros(4)]
        scale = max(self.mean[3], 1.0)
        self.covariance = np.diag([(.1 * scale) ** 2] * 4 + [scale ** 2] * 4)

    @staticmethod
    def _measurement(box: np.ndarray) -> np.ndarray:
        return np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2]]

    @property
    def box(self) -> np.ndarray:
        center = self.mean[:2]
        radius = np.maximum(self.mean[2:4], 1e-3) / 2
        return np.r_[center - radius, center + radius]

    def predict(self, now: float) -> None:
        dt = now - self.predicted_at
        transition = _IDENTITY.copy()
        transition[np.arange(4), np.arange(4, 8)] = dt
        acceleration = np.vstack((np.eye(4) * (dt ** 2 / 2), np.eye(4) * dt))
        scale = max(self.mean[3], 1.0)
        self.mean = transition @ self.mean
        self.covariance = (transition @ self.covariance @ transition.T
                           + scale ** 2 * (acceleration @ acceleration.T))
        self.predicted_at = now

    def correct(self, box: np.ndarray, now: float) -> None:
        measurement = self._measurement(box)
        noise = np.eye(4) * max(0.05 * measurement[3], 0.1) ** 2
        cross_covariance = self.covariance @ _OBSERVE.T
        innovation_covariance = _OBSERVE @ cross_covariance + noise
        gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
        self.mean += gain @ (measurement - _OBSERVE @ self.mean)
        residual = _IDENTITY - gain @ _OBSERVE
        self.covariance = residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        self.last_seen = now
        self.active = True


def _associate(tracks, boxes, classes, detection_indices, min_iou):
    """Globally assign eligible detections to predictions within the IoU gate."""
    if not tracks or not len(detection_indices):
        return {}
    track_ids = list(tracks)
    predictions = np.stack([track.box for track in tracks.values()])[:, None, :]
    detections = boxes[detection_indices][None, :, :]
    overlap = np.maximum(0, np.minimum(predictions[..., 2:], detections[..., 2:])
                         - np.maximum(predictions[..., :2], detections[..., :2])).prod(axis=2)
    union = ((predictions[..., 2:] - predictions[..., :2]).prod(axis=2)
             + (detections[..., 2:] - detections[..., :2]).prod(axis=2) - overlap)
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    track_classes = np.asarray([track.class_id for track in tracks.values()])
    same_class = track_classes[:, None] == classes[detection_indices]
    cost = np.where(same_class, 1.0 - iou, 2.0)
    _, assignment, _ = lap.lapjv(cost, extend_cost=True, cost_limit=1.0 - min_iou)
    return {int(detection_indices[column]): track_ids[row]
            for row, column in enumerate(assignment) if column >= 0}


class VOPTracker:
    """Two-stage ByteTrack matching, with IDs scoped to one VOP stream."""

    def __init__(self, max_age_seconds: float = 1.0):
        self._max_age_seconds = max_age_seconds
        self._next_id = count(1)
        self._tracks: dict[int, _Track] = {}

    def update(self, boxes, scores, classes, now: float, confidence: float) -> dict[int, int]:
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if now - track.last_seen <= self._max_age_seconds
        }
        for track in self._tracks.values():
            track.predict(now)

        high = np.flatnonzero(scores >= confidence)
        low = np.flatnonzero((scores < confidence) & (scores >= LOW_CONFIDENCE))
        strong = _associate(self._tracks, boxes, classes, high, min_iou=0.2)
        matched = set(strong.values())
        unmatched_active = {
            track_id: track for track_id, track in self._tracks.items()
            if track.active and track_id not in matched
        }
        weak = _associate(unmatched_active, boxes, classes, low, min_iou=0.5)
        matched.update(weak.values())
        for track_id, track in self._tracks.items():
            track.active = track_id in matched
        for index, track_id in {**strong, **weak}.items():
            self._tracks[track_id].correct(boxes[index], now)
        for index in high:
            if index not in strong:
                track_id = next(self._next_id)
                self._tracks[track_id] = _Track(boxes[index], int(classes[index]), now)
                strong[int(index)] = track_id
        # Weak detections maintain IDs internally without changing VOP's output threshold.
        return strong
