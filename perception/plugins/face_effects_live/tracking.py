"""Bounded geometry tracker. Ambiguity hides effects; IDs are not identity."""
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import cv2
from .contracts import Face
from .filters import OneEuro, SimilarityFilter


@dataclass
class Track:
    id: int
    face: Face
    timestamp: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    visible: bool = False
    filter: OneEuro = field(default_factory=OneEuro)
    points: np.ndarray | None = None
    stabilizer: SimilarityFilter = field(default_factory=SimilarityFilter)
    appearance: np.ndarray | None = None


class Tracker:
    def __init__(self, max_faces=8, ttl=.35, smooth=True, stabilization="similarity"):
        self.max_faces, self.ttl, self.smooth = max_faces, ttl, smooth
        self.tracks = []
        self.next_id = 1
        self.ambiguities = 0
        self.stabilization = stabilization

    def reset(self):
        self.tracks.clear()  # keep monotonically increasing IDs across scene resets

    def update(self, faces, timestamp, size):
        faces = [f for f in faces if f.valid()][:self.max_faces]
        self.tracks = [t for t in self.tracks if 0 <= timestamp-t.timestamp <= self.ttl]
        tracks = self.tracks
        cost = np.full((len(tracks), len(faces)), 10.)
        for i, t in enumerate(tracks):
            center = t.face.box[:2] + t.face.box[2:]/2
            predicted = center + t.velocity * min(timestamp-t.timestamp, .15)
            for j, f in enumerate(faces):
                scale = max(16, np.linalg.norm(t.face.box[2:]))
                ratio = f.box[2:] / t.face.box[2:]
                distance = np.linalg.norm(f.box[:2]+f.box[2:]/2-predicted)/scale
                if .5 <= ratio.min() and ratio.max() <= 2 and distance < .8:
                    appearance_cost = 0.
                    if t.appearance is not None and f.appearance is not None:
                        appearance_cost = float(np.linalg.norm(t.appearance-f.appearance))
                        if appearance_cost > .85 and distance > .2:
                            continue
                    cost[i, j] = distance + .1 * np.abs(np.log(ratio)).sum() + .4*appearance_cost

        # Global bounded assignment; miss cost competes with bad associations.
        @lru_cache(None)
        def solve(i, used):
            if i == len(tracks):
                return 0., ()
            tail, assignment = solve(i+1, used)
            best = (.85+tail, (-1,)+assignment)
            for j in range(len(faces)):
                if used & (1 << j) or cost[i, j] >= .8:
                    continue
                tail, assignment = solve(i+1, used | (1 << j))
                candidate = (cost[i, j]+tail, (j,)+assignment)
                if candidate[0] < best[0]:
                    best = candidate
            return best

        _, assignment = solve(0, 0)
        used, ambiguous_detections = set(), set()
        for i, t in enumerate(tracks):
            was_visible = t.visible
            t.visible = False
            j = assignment[i]
            if j < 0:
                continue
            row, col = np.sort(cost[i]), np.sort(cost[:, j])
            ambiguous = ((len(row) > 1 and row[1]-row[0] < .12)
                         or (len(col) > 1 and col[1]-col[0] < .12))
            if ambiguous:
                self.ambiguities += 1
                ambiguous_detections.update(np.where(cost[i] < .8)[0].tolist())
                continue
            f = faces[j]
            dt = timestamp-t.timestamp
            old_center = t.face.box[:2]+t.face.box[2:]/2
            if dt > 0:
                velocity = (f.box[:2]+f.box[2:]/2-old_center)/dt
                t.velocity = .5*t.velocity + .5*velocity
            # After any miss, reset the filter to avoid drawing along a stale path.
            if (dt > .12 or not was_visible
                    or t.face.landmark_source != f.landmark_source):
                t.filter = OneEuro()
                t.stabilizer = SimilarityFilter()
            if f.appearance is not None:
                if t.appearance is None:
                    t.appearance = f.appearance.copy()
                else:
                    vector = .9*t.appearance + .1*f.appearance
                    t.appearance = vector/max(np.linalg.norm(vector), 1e-9)
            t.face, t.timestamp = f, timestamp
            t.hits += 1
            t.visible = t.hits >= 2
            used.add(j)
        for j, face in enumerate(faces):
            if j in used or j in ambiguous_detections or len(tracks) >= self.max_faces:
                continue
            # A gated but unassigned detection belongs to an uncertain existing
            # track. Do not create a duplicate sticker beside it.
            if len(cost) and np.any(cost[:, j] < .8):
                continue
            t = Track(self.next_id, face, timestamp)
            t.appearance = face.appearance.copy() if face.appearance is not None else None
            self.next_id += 1
            tracks.append(t)
        for t in tracks:
            if t.timestamp == timestamp:
                if not self.smooth:
                    t.points = t.face.points.copy()
                elif self.stabilization == "similarity":
                    t.points = t.stabilizer(t.face.points, timestamp, size, t.face.box)
                else:
                    normalized = t.face.points / np.asarray(size)
                    t.points = t.filter(normalized, timestamp) * size
        return [t for t in tracks if t.visible]


class OpticalFlow:
    """Forward/backward checked LK only between detector frames; bounded age."""
    def __init__(self):
        self.gray = None
        self.faces = []

    def reset(self):
        self.gray, self.faces = None, []

    def seed(self, gray, faces):
        self.gray, self.faces = gray, faces

    def advance(self, gray):
        result = []
        if self.gray is None or self.gray.shape != gray.shape:
            self.reset()
            return result
        for f in self.faces:
            src = f.landmarks if f.landmarks is not None else f.points
            old = src.astype(np.float32).reshape(-1, 1, 2)
            new, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, old, None,
                                                winSize=(21, 21), maxLevel=2)
            if new is None:
                continue
            back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, new, None,
                                                  winSize=(21, 21), maxLevel=2)
            if back is None:
                continue
            good = (ok[:, 0] > 0) & (ok2[:, 0] > 0) & (np.linalg.norm(back-old, axis=2)[:, 0] < 1.5)
            if good.mean() < .85:
                continue
            matrix, inliers = cv2.estimateAffinePartial2D(src[good], new[:, 0][good],
                                                        method=cv2.RANSAC, ransacReprojThreshold=2.)
            if matrix is None or inliers.mean() < .8 or not np.isfinite(matrix).all():
                continue
            transform = lambda p: p @ matrix[:, :2].T + matrix[:, 2]
            x, y, w, h = f.box
            box_corners = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
            moved = transform(box_corners)
            box = np.r_[moved.min(0), moved.max(0)-moved.min(0)]
            result.append(Face(box, transform(f.points), f.confidence,
                               transform(f.landmarks) if f.landmarks is not None else None,
                               f.quality*.98, topology=f.topology, appearance=f.appearance,
                               landmark_source=f.landmark_source))
        self.seed(gray, result)
        return result
