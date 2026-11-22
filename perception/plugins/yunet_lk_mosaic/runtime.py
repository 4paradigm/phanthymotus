"""YuNet detections, per-face LK tracks and local JPEG mosaic rendering."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .config import MosaicError
from .model import YuNetDetector, resolve_model

MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_SOURCE_SIDE = 2048
COAST_LIMIT_S = .55
HOLD_LIMIT_S = .12
MIN_POINTS = 7
MIN_POINT_RATIO = .5
MAX_LK_ERROR = 20.
MAX_LK_RESIDUAL = 3.
MAX_STEP_PX = 40.
MAX_ADAPTIVE_INTERVAL = 5
EMPTY_ADAPTIVE_INTERVAL = 2
MATCH_IOU = .45
AMBIGUOUS_IOU = .20


@dataclass
class Track:
    box: tuple
    anchor_stamp: int
    anchor_at: float
    points: object
    reliable_at: float
    missed: bool = False
    association_clear: bool = True


@dataclass(frozen=True)
class Mask:
    box: tuple
    expanded: bool = False
    correctable: bool = False


@dataclass(frozen=True)
class Correction:
    track_id: int
    anchor_stamp: int
    end_stamp: int
    delta: tuple


@dataclass
class Analysis:
    image: np.ndarray
    masks: dict
    corrections: list


def _clip(box, width, height):
    x1, y1, x2, y2 = box
    x1, y1 = max(0., x1), max(0., y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    return (x1, y1, x2, y2) if x2 - x1 >= 1 and y2 - y1 >= 1 else None


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _iou(a, b):
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    overlap = width * height
    return overlap / (_area(a) + _area(b) - overlap)


def _pixels(box, width, height, expansion=0.):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * expansion, (y2 - y1) * expansion
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(width, int(x2 + dx)), min(height, int(y2 + dy)))


class MosaicEngine:
    """One camera's mutable state. Owned by its frame worker."""

    def __init__(self, model_path, config, *, detector=None, cv=None, timing=None, progress=None):
        if cv is None:
            import cv2 as cv
        self.cv = cv
        self.cfg = config
        self.timing = timing
        self.detector = detector if detector is not None else YuNetDetector(
            cv, resolve_model(model_path, progress), config,
        )
        self.detection_calls = 0
        self.lk_calls = 0
        self.bad_tracks = 0
        self.over_capacity = 0
        self.next_id = 0
        self.reset()

    def reset(self):
        self.tracks = {}
        self.prev_gray = None
        self.prev_capture = None
        self.frames_since_detection = 0

    def close(self):
        self.reset()
        self.detector.close()

    def decode(self, jpeg):
        if (not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= MAX_JPEG_BYTES
                or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
            raise MosaicError("invalid_jpeg")
        cv = self.cv
        image = cv.imdecode(np.frombuffer(jpeg, np.uint8),
                            cv.IMREAD_COLOR | cv.IMREAD_IGNORE_ORIENTATION)
        if image is None or max(image.shape[:2]) > MAX_SOURCE_SIDE:
            raise MosaicError("invalid_image_size")
        height, width = image.shape[:2]
        if width == self.cfg.output_width:
            return image
        h = max(1, round(height * self.cfg.output_width / width))
        if h > MAX_SOURCE_SIDE:
            raise MosaicError("invalid_image_aspect_ratio")
        return cv.resize(image, (self.cfg.output_width, h), interpolation=cv.INTER_AREA)

    def _boxes(self, image):
        started = time.perf_counter()
        self.detection_calls += 1
        try:
            faces = self.detector.detect(image)
        finally:
            if self.timing is not None:
                self.timing("detect", time.perf_counter() - started)
        if faces is None:
            return []
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] != 15 or not np.isfinite(faces).all():
            raise MosaicError("invalid_yunet_output")
        height, width = image.shape[:2]
        candidates = []
        for face in faces:
            score = float(face[14])
            if score < self.cfg.confidence_threshold:
                continue
            x, y, w, h = map(float, face[:4])
            if w <= 0 or h <= 0:
                raise MosaicError("invalid_yunet_box")
            box = _clip((x, y, x + w, y + h), width, height)
            if box is not None:
                candidates.append((score, box))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > self.cfg.max_faces:
            self.over_capacity += 1
        return [box for _, box in candidates[:self.cfg.max_faces]]

    def _seed(self, gray, box, limit):
        height, width = gray.shape
        x1, y1, x2, y2 = _pixels(box, width, height)
        points = self.cv.goodFeaturesToTrack(
            gray[y1:y2, x1:x2], maxCorners=limit, qualityLevel=.025, minDistance=6,
        )
        if points is None or len(points) < MIN_POINTS:
            return None
        points[:, 0] += np.array([x1, y1], dtype=np.float32)
        return points

    def _advance(self, gray, at):
        grouped = [(track, track.points) for track in self.tracks.values()
                   if track.points is not None]
        if not grouped:
            return False, bool(self.tracks)
        points = np.concatenate([pts for _, pts in grouped])
        cv = self.cv
        self.lk_calls += 1
        started = time.perf_counter()
        try:
            moved, status, error = cv.calcOpticalFlowPyrLK(
                self.prev_gray, gray, points, None, winSize=(15, 15), maxLevel=2,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, .03),
            )
        except cv.error:
            moved = status = error = None
        finally:
            if self.timing is not None:
                self.timing("lk", time.perf_counter() - started)

        started = time.perf_counter()
        fast = False
        offset = 0
        height, width = gray.shape
        for track, old in grouped:
            end = offset + len(old)
            track.points = None
            if moved is not None and status is not None and error is not None:
                new = moved[offset:end, 0]
                err = error[offset:end].reshape(-1)
                candidate = ((status[offset:end].reshape(-1) == 1)
                             & np.isfinite(new).all(axis=1) & np.isfinite(err)
                             & (err <= MAX_LK_ERROR)
                             & (new >= 0).all(axis=1)
                             & (new < [width, height]).all(axis=1))
                required = max(MIN_POINTS, math.ceil(len(old) * MIN_POINT_RATIO))
                if np.count_nonzero(candidate) >= required:
                    indices = np.flatnonzero(candidate)
                    shifts = new[indices] - old[indices, 0]
                    middle = np.median(shifts, axis=0)
                    coherent = np.max(np.abs(shifts - middle), axis=1) <= MAX_LK_RESIDUAL
                    if np.count_nonzero(coherent) >= required:
                        dx, dy = map(float, np.median(shifts[coherent], axis=0))
                        x1, y1, x2, y2 = track.box
                        box = _clip((x1 + dx, y1 + dy, x2 + dx, y2 + dy), width, height)
                        speed = max(abs(dx), abs(dy))
                        if box is not None and speed < MAX_STEP_PX:
                            fast |= speed >= max(4., .12 * min(x2 - x1, y2 - y1))
                            track.box, track.reliable_at = box, at
                            track.points = new[indices[coherent]].reshape(-1, 1, 2)
                        elif box is None and speed < MAX_STEP_PX:
                            # Coherent flow moved the face out of the image.
                            track.reliable_at = -math.inf
            offset = end
            if track.points is None:
                self.bad_tracks += 1
        if self.timing is not None:
            self.timing("lk_quality", time.perf_counter() - started)
        return fast, any(track.points is None for track in self.tracks.values())

    def _mark_ambiguous_tracks(self):
        tracks = list(self.tracks.values())
        for i, first in enumerate(tracks):
            for second in tracks[i + 1:]:
                if _iou(first.box, second.box) >= AMBIGUOUS_IOU:
                    first.association_clear = second.association_clear = False

    def _update_detections(self, gray, boxes, stamp, at):
        # The detection must agree with one prediction, and neither side can
        # overlap a competing candidate. Held boxes may veto an ambiguous match.
        tracks = list(self.tracks.items())
        overlaps = [[_iou(track.box, box) for box in boxes] for _, track in tracks]
        by_detection = list(zip(*overlaps)) if tracks else []
        updated, corrections, matched = {}, [], set()
        limit = max(MIN_POINTS, 75 // max(1, len(boxes)))
        for index, box in enumerate(boxes):
            candidates = [i for i, score in enumerate(by_detection[index])
                          if score >= AMBIGUOUS_IOU] if tracks else []
            match = None
            if len(candidates) == 1:
                i = candidates[0]
                track_id, previous = tracks[i]
                if (previous.points is not None and overlaps[i][index] >= MATCH_IOU
                        and sum(s >= AMBIGUOUS_IOU for s in overlaps[i]) == 1):
                    match = track_id
                    matched.add(track_id)
                    if (self.cfg.publish_delay_ms and self.cfg.miss_compensation
                            and previous.association_clear):
                        corrections.append(Correction(
                            track_id, previous.anchor_stamp, stamp,
                            tuple(a - b for a, b in zip(box, previous.box)),
                        ))
            if match is None:
                self.next_id += 1
                match = self.next_id
            updated[match] = Track(box, stamp, at, self._seed(gray, box, limit), at)
        if self.cfg.miss_compensation:
            for i, (track_id, track) in enumerate(tracks):
                if len(updated) >= self.cfg.max_faces:
                    break
                if track_id not in matched and all(v < AMBIGUOUS_IOU for v in overlaps[i]):
                    track.missed = True
                    updated[track_id] = track
        self.tracks = updated
        return corrections

    def analyze(self, jpeg, stamp, captured_at):
        image = self.decode(jpeg)
        gray = self.cv.cvtColor(image, self.cv.COLOR_BGR2GRAY)
        if (self.prev_gray is None or self.prev_gray.shape != gray.shape
                or captured_at - self.prev_capture >= COAST_LIMIT_S):
            self.reset()
        first = self.prev_gray is None
        self.tracks = {
            key: track for key, track in self.tracks.items()
            if captured_at - track.anchor_at < COAST_LIMIT_S
        }
        self.frames_since_detection += 1
        fixed_due = self.frames_since_detection >= self.cfg.detect_every_frames
        fast = poor = False
        if self.tracks and (self.cfg.adaptive_detection or not fixed_due
                            or self.cfg.miss_compensation):
            fast, poor = self._advance(gray, captured_at)
            if self.cfg.publish_delay_ms and self.cfg.miss_compensation:
                self._mark_ambiguous_tracks()
        if self.cfg.adaptive_detection:
            started = time.perf_counter()
            interval = MAX_ADAPTIVE_INTERVAL if self.tracks else EMPTY_ADAPTIVE_INTERVAL
            detect = first or fast or poor or self.frames_since_detection >= interval
            if self.timing is not None:
                self.timing("adaptive_decision", time.perf_counter() - started)
        else:
            detect = first or fixed_due

        corrections = []
        if detect:
            boxes = self._boxes(image)
            corrections = self._update_detections(gray, boxes, stamp, captured_at)
            if self.cfg.publish_delay_ms and self.cfg.miss_compensation:
                self._mark_ambiguous_tracks()
            self.frames_since_detection = 0

        masks = {}
        for key, track in self.tracks.items():
            if track.anchor_stamp == stamp:
                masks[key] = Mask(track.box)
            elif track.points is not None and (not detect or self.cfg.miss_compensation):
                masks[key] = Mask(track.box, expanded=track.missed,
                                  correctable=self.cfg.miss_compensation)
            elif (self.cfg.miss_compensation and captured_at - track.reliable_at <= HOLD_LIMIT_S
                  and math.isfinite(track.reliable_at)):
                masks[key] = Mask(track.box, expanded=True)
                # One held output. The track cannot be matched for correction.
                track.reliable_at = -math.inf
        self.tracks = {key: track for key, track in self.tracks.items() if key in masks}
        self.prev_gray, self.prev_capture = gray, captured_at
        return Analysis(image, masks, corrections)

    def render(self, analysis):
        cv = self.cv
        image = analysis.image
        h, w = image.shape[:2]
        rectangles = []
        for mask in analysis.masks.values():
            expansion = self.cfg.unstable_expansion if mask.expanded else self.cfg.stable_expansion
            a, b, c, d = _pixels(mask.box, w, h, expansion)
            if c <= a or d <= b:
                continue
            roi = image[b:d, a:c]
            sh, sw = roi.shape[:2]
            block = self.cfg.pixelate_block
            small = cv.resize(roi, (max(1, sw // block), max(1, sh // block)),
                              interpolation=cv.INTER_AREA)
            image[b:d, a:c] = cv.resize(small, (sw, sh), interpolation=cv.INTER_NEAREST)
            rectangles.append((a, b, c, d))
        ok, encoded = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        if not ok or encoded is None:
            raise MosaicError("jpeg_encode_failed")
        return encoded.tobytes(), rectangles
