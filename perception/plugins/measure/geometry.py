"""Metric measurements from a detection box and an aligned depth image.

This module is deliberately ROS-free. The card in `plugin.py` owns the topics
and the timer; everything that turns a box plus a depth map into metres is a
pure function here, so the geometry can be tested without a robot.

What is measured, and what is not
---------------------------------

A single RGB-D view gives three useful numbers for a detection:

* distance to the visible surface, from the depth map;
* the visible width, from the box's horizontal span and the camera's `fx`;
* the visible height, from the box's vertical span and the camera's `fy`.

It does **not** give the object's full 3D extent. The back face is not visible,
and a box seen at an angle projects a different width than its physical side.
`depth_span_m` is the range of depths inside the box, which is a useful sanity
signal, but it is not "length". Consumers that need a true oriented 3D box need
segmentation and a point cloud or multiple views; this card does not invent
that claim.

The depth format is `image/depth-zlib`: 640x480 little-endian uint16
millimetres, with 0 meaning "no reading". Zero is never turned into zero metres.
"""

from __future__ import annotations

import math
import zlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .camera import CameraModel

WIDTH = 640
HEIGHT = 480
DEPTH_PERCENTILE = 25
DEFAULT_MATCH_DISTANCE = 0.15
_MEDIAN_FIELDS = ("distance_m", "visible_width_m", "visible_height_m",
                  "depth_span_m")


class DepthError(ValueError):
    """A depth payload that cannot be decoded into a map."""


@dataclass(frozen=True)
class Box:
    """A normalised box plus why it may not describe the whole object."""

    norm: tuple[float, float, float, float] | None
    clipped: bool = False
    approximate: bool = False
    reason: str = ""


@dataclass(frozen=True)
class DepthSample:
    distance_m: float | None
    min_m: float | None
    max_m: float | None
    valid_ratio: float
    valid_pixels: int
    total_pixels: int

    @property
    def span_m(self) -> float | None:
        if self.min_m is None or self.max_m is None:
            return None
        return round(self.max_m - self.min_m, 3)


def decode(payload: bytes, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Decode `image/depth-zlib` into a float array of metres.

    Invalid samples become `np.nan`, never 0.0. A consumer that later takes a
    minimum must not read "no reading" as "touching the robot".
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise DepthError(f"invalid depth shape {width!r}x{height!r}")
    try:
        raw = zlib.decompress(payload)
    except zlib.error as exc:
        raise DepthError(f"not a zlib buffer: {exc}") from exc

    expected = width * height * 2
    if len(raw) != expected:
        # Do not reshape a short buffer. A wrong shape produces plausible
        # numbers for the wrong pixels, which is worse than refusing the frame.
        raise DepthError(f"expected {expected} bytes ({width}x{height} uint16), got {len(raw)}")

    millimetres = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    metres = millimetres.astype(np.float32) / 1000.0
    metres[millimetres == 0] = np.nan
    return metres


def normalise_bbox(obj: dict, frame_width: int, frame_height: int) -> Box:
    """Get a 0..1 box for a VOP object, preferring a publisher's bbox_norm.

    VOP publishes `bbox` in its camera's pixels. A pixel box is meaningless
    without that camera's resolution, so a missing or invalid frame is reported
    as no box rather than guessed from the depth map's dimensions.
    """
    if not isinstance(obj, dict):
        return Box(None, reason="object is not a mapping")

    raw = obj.get("bbox_norm")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = _finite_values(raw)
        if values is not None:
            return _clamp_box(values, clipped=_outside(values))

    raw = obj.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return Box(None, reason="no bbox")
    values = _finite_values(raw)
    if values is None or frame_width <= 0 or frame_height <= 0:
        return Box(None, reason="no usable bbox or frame size")

    x1, y1, x2, y2 = values
    clipped = _outside(values, frame_width, frame_height)
    norm = (x1 / frame_width, y1 / frame_height,
            x2 / frame_width, y2 / frame_height)
    return _clamp_box(norm, clipped=clipped)


def _finite_values(values) -> tuple[float, float, float, float] | None:
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) for v in out):
        return None
    return out


def _outside(values, width: int | None = None, height: int | None = None) -> bool:
    x1, y1, x2, y2 = values
    x2edge = width if width is not None else 1.0
    y2edge = height if height is not None else 1.0
    return x1 <= 0.0 or y1 <= 0.0 or x2 >= x2edge or y2 >= y2edge


def _clamp_box(values, *, clipped: bool) -> Box:
    x1, y1, x2, y2 = values
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return Box(None, clipped=clipped, reason="degenerate bbox")
    return Box((x1, y1, x2, y2), clipped=clipped)


def sample_box(depth: np.ndarray, bbox_norm, percentile: float = DEPTH_PERCENTILE) -> DepthSample:
    """Distance to the visible surface inside a normalised box.

    The distance is a low percentile rather than the minimum: a detector box
    contains some background at its edges, and a minimum would report the
    nearest stray pixel. The percentile is configurable because a box around a
    thin object is a different problem from a box around a wall.
    """
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        return DepthSample(None, None, None, 0.0, 0, 0)
    if not isinstance(bbox_norm, (list, tuple)) or len(bbox_norm) != 4:
        return DepthSample(None, None, None, 0.0, 0, 0)

    height, width = depth.shape
    x1, y1, x2, y2 = (float(v) for v in bbox_norm)
    left = max(0, min(width - 1, int(math.floor(x1 * width))))
    right = max(left + 1, min(width, int(math.ceil(x2 * width))))
    top = max(0, min(height - 1, int(math.floor(y1 * height))))
    bottom = max(top + 1, min(height, int(math.ceil(y2 * height))))

    patch = depth[top:bottom, left:right]
    total = int(patch.size)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    ratio = (float(valid.size) / total) if total else 0.0
    if valid.size == 0:
        return DepthSample(None, None, None, ratio, 0, total)

    return DepthSample(
        distance_m=round(float(np.percentile(valid, percentile)), 3),
        min_m=round(float(np.min(valid)), 3),
        max_m=round(float(np.max(valid)), 3),
        valid_ratio=round(ratio, 3),
        valid_pixels=int(valid.size),
        total_pixels=total,
    )


def _visible_extent(distance_m: float, normalised_span: float,
                    intrinsic: float | None, pixels: int | None) -> float | None:
    if distance_m is None or normalised_span <= 0:
        return None
    if intrinsic and intrinsic > 0 and pixels and pixels > 0:
        return round(distance_m * normalised_span * pixels / intrinsic, 3)
    return None


def _angle_extent(distance_m: float, normalised_span: float,
                  half_fov_rad: float | None) -> float | None:
    if distance_m is None or normalised_span <= 0 or not half_fov_rad:
        return None
    return round(2.0 * distance_m * math.tan(half_fov_rad) * normalised_span, 3)


def measure_object(obj: dict, depth: np.ndarray, camera: CameraModel,
                   object_frame: tuple[int, int], *,
                   depth_percentile: float = DEPTH_PERCENTILE,
                   max_depth_m: float | None = None) -> dict | None:
    """One detection plus depth, as a measurement record.

    The returned record always says which dimensions were actually measured.
    A missing camera or an unreadable depth patch yields `None` dimensions, not
    zero and not a plausible-looking fallback.
    """
    frame_width, frame_height = object_frame
    box = normalise_bbox(obj, frame_width, frame_height)
    record = {
        "name": str(obj.get("name") or ""),
        "color": str(obj.get("color") or obj.get("color_name") or ""),
        "confidence": _round_optional(obj.get("confidence")),
        "bbox_norm": list(box.norm) if box.norm else None,
        "clipped": bool(box.clipped),
        "approximate": bool(box.approximate),
        "distance_m": None,
        "visible_width_m": None,
        "visible_height_m": None,
        "depth_span_m": None,
        "extent_3d_m": None,
        "method": "bbox+depth",
        "quality": {
            "valid_ratio": 0.0,
            "valid_pixels": 0,
            "total_pixels": 0,
            "camera": camera.source,
            "note": box.reason,
        },
    }
    if box.norm is None:
        return record

    sample = sample_box(depth, box.norm, depth_percentile)
    record["quality"].update({
        "valid_ratio": sample.valid_ratio,
        "valid_pixels": sample.valid_pixels,
        "total_pixels": sample.total_pixels,
    })
    if sample.distance_m is None:
        record["quality"]["note"] = "no valid depth inside bbox"
        return record
    if max_depth_m is not None and sample.distance_m > max_depth_m:
        record["quality"]["note"] = f"distance beyond max_depth_m={max_depth_m:g}"
        return record

    record["distance_m"] = sample.distance_m
    record["depth_span_m"] = sample.span_m

    span_x = box.norm[2] - box.norm[0]
    span_y = box.norm[3] - box.norm[1]
    width = _visible_extent(sample.distance_m, span_x, camera.fx, camera.width)
    height = _visible_extent(sample.distance_m, span_y, camera.fy, camera.height)
    if width is None:
        width = _angle_extent(sample.distance_m, span_x, camera.half_fov_h)
    if height is None:
        height = _angle_extent(sample.distance_m, span_y, camera.half_fov_v)
    record["visible_width_m"] = width
    record["visible_height_m"] = height

    if box.clipped:
        record["quality"]["note"] = "bbox touches the frame edge; extents are lower bounds"
    elif width is None or height is None:
        record["quality"]["note"] = "camera geometry missing; only distance measured"
    return record


def measure_objects(objects, depth: np.ndarray, camera: CameraModel,
                    object_frame: tuple[int, int], *,
                    min_confidence: float = 0.3,
                    max_objects: int = 8,
                    depth_percentile: float = DEPTH_PERCENTILE,
                    max_depth_m: float | None = None) -> list[dict]:
    """Measure the qualifying detections in one VOP payload."""
    if not isinstance(objects, list):
        return []
    out = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        try:
            confidence = float(obj.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue
        record = measure_object(obj, depth, camera, object_frame,
                                depth_percentile=depth_percentile,
                                max_depth_m=max_depth_m)
        if record is not None:
            out.append(record)
        if len(out) >= max_objects:
            break
    return out


def _round_optional(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 3) if math.isfinite(number) else None


# ── short-window stabilisation ───────────────────────────────────────────────


@dataclass
class _Track:
    key: tuple[str, str]
    center: tuple[float, float]
    last_seen: float
    samples: deque = field(default_factory=deque)

    def median(self, field: str):
        values = [sample[field] for sample in self.samples
                  if sample.get(field) is not None]
        if not values:
            return None
        return round(float(np.median(values)), 3)


class Stabilizer:
    """A median over the last few frames, matched by name/colour and position.

    VOP boxes jitter frame to frame, and a single box edge moving by two pixels
    changes the reported width. A raw per-frame number is therefore noisier than
    the detector. This is deliberately not a tracker: it has no velocity model
    and no identity beyond a nearby box with the same name/colour. It exists to
    make a measurement stable enough to quote, not to follow an object through
    an occlusion.
    """

    def __init__(self, *, window: int = 5, min_samples: int = 3,
                 timeout_s: float = 2.0,
                 match_distance_norm: float = DEFAULT_MATCH_DISTANCE):
        self.window = max(1, int(window))
        self.min_samples = max(1, min(int(min_samples), self.window))
        self.timeout_s = max(0.0, float(timeout_s))
        self.match_distance_norm = max(0.0, float(match_distance_norm))
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._tracks.clear()

    @staticmethod
    def _center(record: dict) -> tuple[float, float] | None:
        bbox = record.get("bbox_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        return ((float(bbox[0]) + float(bbox[2])) / 2.0,
                (float(bbox[1]) + float(bbox[3])) / 2.0)

    @staticmethod
    def _key(record: dict) -> tuple[str, str]:
        return (str(record.get("name") or ""),
                str(record.get("color") or record.get("color_name") or ""))

    def update(self, measurements: list[dict], now_s: float) -> list[dict]:
        seen = set()
        for record in measurements:
            center = self._center(record)
            if center is None:
                continue
            key = self._key(record)
            track = self._match(key, center)
            if track is None:
                track = _Track(key=key, center=center, last_seen=now_s)
                self._tracks.append(track)
            track.center = center
            track.last_seen = now_s
            track.samples.append(record)
            seen.add(id(track))
            while len(track.samples) > self.window:
                track.samples.popleft()

        if self.timeout_s:
            cutoff = now_s - self.timeout_s
            self._tracks = [track for track in self._tracks
                            if track.last_seen >= cutoff]

        out = []
        for track in sorted(self._tracks, key=lambda item: item.last_seen,
                            reverse=True):
            latest = track.samples[-1]
            record = {
                "key": f"{track.key[0]}#{track.key[1]}" if track.key[1] else track.key[0],
                "name": latest.get("name"),
                "confidence": latest.get("confidence"),
                "bbox_norm": latest.get("bbox_norm"),
                "clipped": bool(latest.get("clipped")),
                "approximate": bool(latest.get("approximate")),
                "distance_m": track.median("distance_m"),
                "visible_width_m": track.median("visible_width_m"),
                "visible_height_m": track.median("visible_height_m"),
                "depth_span_m": track.median("depth_span_m"),
                "extent_3d_m": None,
                "method": latest.get("method", "bbox+depth"),
                "samples": len(track.samples),
                "stable": len(track.samples) >= self.min_samples,
                # A retained track is not a new observation. age_ms alone cannot
                # express that at a 5 Hz tick: two calls can share a monotonic
                # timestamp closely enough to round to 0 ms.
                "observed": id(track) in seen,
                "age_ms": int(max(0.0, now_s - track.last_seen) * 1000),
                "quality": dict(latest.get("quality") or {}),
            }
            out.append(record)
        return out

    def _match(self, key: tuple[str, str],
               center: tuple[float, float]) -> _Track | None:
        best = None
        best_distance = None
        for track in self._tracks:
            if track.key != key:
                continue
            distance = math.dist(track.center, center)
            if distance > self.match_distance_norm:
                continue
            if best_distance is None or distance < best_distance:
                best = track
                best_distance = distance
        return best


# ── session summary and bounded history ─────────────────────────────────────


@dataclass
class _SessionObject:
    key: str
    name: str
    color: str
    heights: deque = field(default_factory=deque)
    widths: deque = field(default_factory=deque)
    distances: deque = field(default_factory=deque)
    samples: int = 0
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    clipped: bool = False


class Session:
    """A compact answer to "what did I measure while I was away?".

    The card publishes at a few Hz; keeping every frame in memory or in a log is
    both noisy and hard to read. This keeps:

    * per-key min/max/median/latest over a bounded recent window;
    * a bounded, rate-limited history for the values that actually changed.

    It is not an identity tracker. The key is name/colour (and the stabilizer's
    nearby-box match), so two people with the same label can merge. For a single
    person walking in front of the robot — the immediate use case — that is
    enough; multi-person identity is deliberately out of scope.
    """

    def __init__(self, *, history_size: int = 200,
                 record_interval_ms: int = 1000,
                 record_min_delta_m: float = 0.01,
                 only_stable: bool = True,
                 max_objects: int = 20,
                 max_samples_per_object: int = 500):
        self.history_size = max(1, int(history_size))
        self.record_interval_ms = max(0, int(record_interval_ms))
        self.record_min_delta_m = max(0.0, float(record_min_delta_m))
        self.only_stable = bool(only_stable)
        self.max_objects = max(1, int(max_objects))
        self.max_samples_per_object = max(1, int(max_samples_per_object))
        self._history: deque = deque(maxlen=self.history_size)
        self._last_history: dict[str, dict] = {}
        self._objects: dict[str, _SessionObject] = {}
        self._started_ms = 0

    def reset(self, now_ms: int) -> None:
        self._history.clear()
        self._last_history.clear()
        self._objects.clear()
        self._started_ms = int(now_ms)

    def record(self, measurements: list[dict], now_ms: int) -> list[dict]:
        """Add one tick's stable measurements; return any new history rows."""
        if not self._started_ms:
            self._started_ms = int(now_ms)
        new_rows = []
        for record in measurements:
            if not isinstance(record, dict):
                continue
            # Stabilizer.update() keeps a track for track_timeout_ms after the
            # object disappears. Those retained records have age_ms > 0 and are
            # predictions, not observations; recording them would advance
            # last_seen_ms and weight the summary with phantom samples.
            if record.get("observed") is False:
                continue
            if record.get("age_ms", 0) > 0:
                continue
            if self.only_stable and not record.get("stable"):
                continue
            height = _number(record.get("visible_height_m"))
            width = _number(record.get("visible_width_m"))
            distance = _number(record.get("distance_m"))
            if height is None and width is None:
                # Distance alone is not a size measurement; recording it would
                # make the summary look like it measured something it did not.
                continue
            key = str(record.get("key") or
                      f"{record.get('name') or ''}#{record.get('color') or ''}")
            item = self._objects.get(key)
            if item is None:
                item = _SessionObject(
                    key=key, name=str(record.get("name") or ""),
                    color=str(record.get("color") or ""),
                    first_seen_ms=int(now_ms))
                self._objects[key] = item
            item.samples += 1
            item.last_seen_ms = int(now_ms)
            item.clipped = bool(item.clipped or record.get("clipped"))
            _append(item.heights, height, self.max_samples_per_object)
            _append(item.widths, width, self.max_samples_per_object)
            _append(item.distances, distance, self.max_samples_per_object)

            last = self._last_history.get(key)
            changed = (
                last is None
                or self.record_interval_ms == 0
                or int(now_ms) - int(last["timestamp_ms"]) >= self.record_interval_ms
                or _changed(height, last.get("visible_height_m"),
                            self.record_min_delta_m)
                or _changed(width, last.get("visible_width_m"),
                            self.record_min_delta_m)
            )
            if changed:
                row = {
                    "timestamp_ms": int(now_ms),
                    "key": key,
                    "name": item.name,
                    "visible_height_m": height,
                    "visible_width_m": width,
                    "distance_m": distance,
                    "clipped": bool(record.get("clipped")),
                }
                self._history.append(row)
                self._last_history[key] = row
                new_rows.append(row)
        self._evict()
        return new_rows

    def _evict(self) -> None:
        """Keep the retained object set bounded, not only the rendered list.

        `summary()` truncates its output, but `_objects` and `_last_history`
        would otherwise retain every distinct name/colour ever seen. On a
        long-running card that is unbounded memory, and `summary_max_objects`
        would be a display limit pretending to be a resource limit.
        """
        while len(self._objects) > self.max_objects:
            oldest = min(self._objects.values(), key=lambda item: item.last_seen_ms)
            del self._objects[oldest.key]
            self._last_history.pop(oldest.key, None)

    def summary(self, now_ms: int) -> dict:
        objects = sorted(self._objects.values(),
                         key=lambda item: item.last_seen_ms,
                         reverse=True)
        truncated = max(0, len(objects) - self.max_objects)
        items = [_summary_item(item) for item in objects[:self.max_objects]]
        return {
            "started_ms": self._started_ms or None,
            "duration_s": (round(max(0, int(now_ms) - self._started_ms) / 1000.0, 1)
                           if self._started_ms else 0),
            "objects": items,
            "objects_truncated": truncated,
        }

    def history(self) -> list[dict]:
        return [{
            "ts_ms": row["timestamp_ms"],
            "key": row["key"],
            "name": row["name"],
            "w_cm": _cm(row.get("visible_width_m")),
            "h_cm": _cm(row.get("visible_height_m")),
            "d_cm": _cm(row.get("distance_m")),
        } for row in self._history]


def _append(target: deque, value, limit: int) -> None:
    if value is None:
        return
    target.append(float(value))
    while len(target) > limit:
        target.popleft()


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _changed(current, previous, minimum: float) -> bool:
    current = _number(current)
    previous = _number(previous)
    if current is None or previous is None:
        return current is not None or previous is not None
    return abs(current - previous) >= minimum


def _cm(value) -> float | None:
    number = _number(value)
    return round(number * 100.0, 1) if number is not None else None


def _stats_cm(values: deque) -> dict:
    if not values:
        return {"min": None, "max": None, "median": None, "latest": None}
    data = [value * 100.0 for value in values]
    return {
        "min": round(min(data), 1),
        "max": round(max(data), 1),
        "median": round(float(np.median(data)), 1),
        "latest": round(data[-1], 1),
    }


def _summary_item(item: _SessionObject) -> dict:
    out = {
        "key": item.key,
        "name": item.name,
        "n": item.samples,
        "last_ms": item.last_seen_ms,
        "visible_w_cm": _stats_cm(item.widths),
        "visible_h_cm": _stats_cm(item.heights),
        "distance_cm": _stats_cm(item.distances),
    }
    if item.clipped:
        out["clipped"] = True
    return out
