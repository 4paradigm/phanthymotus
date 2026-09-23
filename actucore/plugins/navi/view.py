"""What the card is looking at and what it decided, drawn on one frame.

A second output port, `image/jpeg`, next to the command stream. Nothing reads
it but a human — and a human is exactly what this card has been missing. Every
bug found in it so far took the same shape: **the numbers all looked reasonable
and the robot did something else.** A field of view that made the corridor
1.86 m wide, angular bands that were reading the lens barrel rather than the
room, a sidestep amplified ninefold by a deadband — each of those was invisible
in `info()` and obvious the moment somebody watched the robot.

So this draws the four things a decision is actually made of, over the depth map
it was made from:

* **the depth map**, with "no reading" in flat grey rather than a colour. That
  distinction is load-bearing — a confident wrong reading and a missing one look
  identical in any colour ramp, and telling them apart is what the lens-barrel
  mask exists for. Grey is what that mask looks like when it is working.
* **the corridor**, at the distance it currently clears to. This is the thing
  that decides whether the robot moves, and until now its width was something
  you could only infer from behaviour.
* **the target**, its box and its range.
* **the command**, as arrows whose length is the speed on that axis.

Deliberately ASCII-only text: `cv2.putText` has Hershey fonts and no CJK, so a
Chinese reason string would render as boxes. Status codes and numbers carry, the
prose stays in `info().last.reason` where it renders properly.

Pure rendering — no ROS, no plugin state — so it can be asserted frame by frame
in a test rather than by looking at a robot.
"""
from __future__ import annotations

import math

import numpy as np

# **The dashboard's own depth ramp, not a second one.**
#
# The first version used viridis, and the result was two panels side by side
# showing the same depth map in different colours — which makes an operator
# translate between them before they can compare, at exactly the moment they are
# trying to work out why a robot did something. Whichever ramp is better, two is
# worse than one.
#
# Ported from `agent-core/web/js/renderers/camera.js`, which is the source of
# truth and carries the reasoning: near is loud and far washes out (red belongs
# to what you are about to hit); the stops sit at the distances the robot acts
# on rather than spread evenly, so most of the contrast lands where a decision
# changes; lightness rises monotonically with distance, so the reading survives
# greyscale and all three common colour-vision deficiencies.
#
# Duplicated rather than shared for the same reason `motus.control/1` is a
# document and not a package — this is Python in a container, that is JavaScript
# in a browser. Change one, change both; a test pins the two tables together.
_DEPTH_STOPS = [
    (0.00, 0x5E, 0x14, 0x10), (0.30, 0xA8, 0x21, 0x0F),
    (0.60, 0xDC, 0x5A, 0x14), (0.80, 0xE8, 0x74, 0x1E),
    (1.10, 0xF0, 0x9C, 0x2E), (1.45, 0xF3, 0xC4, 0x52),
    (1.80, 0xC9, 0xCC, 0x8E), (2.10, 0xB4, 0xD2, 0xCC),
    (3.20, 0xC6, 0xD6, 0xD6), (5.00, 0xE6, 0xE4, 0xDD),
]
_DEPTH_MAX_M = 5.0
_DEPTH_NEAR_M = 2.0
_DEPTH_NEAR_SHARE = 0.72

_NO_READING = (70, 70, 70)
_CORRIDOR = (255, 255, 255)
_TARGET = (80, 220, 255)
_ARROW = (120, 255, 120)
_WARN = (80, 120, 255)


def depth_lut() -> np.ndarray:
    """256x3 BGR lookup, the same one the dashboard builds.

    A table rather than per-pixel interpolation for the reason it is one there:
    307k pixels a frame for a result with only 256 distinct values.
    """
    table = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        f = i / 255.0
        if f <= _DEPTH_NEAR_SHARE:
            metres = (f / _DEPTH_NEAR_SHARE) * _DEPTH_NEAR_M
        else:
            metres = _DEPTH_NEAR_M + ((f - _DEPTH_NEAR_SHARE)
                                      / (1 - _DEPTH_NEAR_SHARE)) * (
                _DEPTH_MAX_M - _DEPTH_NEAR_M)
        k = 0
        while k < len(_DEPTH_STOPS) - 2 and metres > _DEPTH_STOPS[k + 1][0]:
            k += 1
        d0, r0, g0, b0 = _DEPTH_STOPS[k]
        d1, r1, g1, b1 = _DEPTH_STOPS[k + 1]
        t = min(1.0, max(0.0, (metres - d0) / (d1 - d0)))
        # BGR: what cv2 writes out.
        table[i] = (b0 + (b1 - b0) * t, g0 + (g1 - g0) * t, r0 + (r1 - r0) * t)
    return table


_LUT = depth_lut()


def _index_of(depth_m: np.ndarray) -> np.ndarray:
    """Metres to LUT index, near band stretched — the same warp as the panel."""
    metres = np.clip(np.nan_to_num(depth_m, nan=_DEPTH_MAX_M), 0.0, _DEPTH_MAX_M)
    near = metres <= _DEPTH_NEAR_M
    out = np.empty(metres.shape, dtype=np.float64)
    out[near] = metres[near] / _DEPTH_NEAR_M * _DEPTH_NEAR_SHARE
    out[~near] = _DEPTH_NEAR_SHARE + (
        (metres[~near] - _DEPTH_NEAR_M) / (_DEPTH_MAX_M - _DEPTH_NEAR_M)
    ) * (1 - _DEPTH_NEAR_SHARE)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _colormap(depth_m: np.ndarray, far_m: float):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    frame = _LUT[_index_of(depth_m)]
    frame[~valid] = _NO_READING
    return frame


def _composite(rgb, depth_m: np.ndarray, config, width: int, height: int,
               blend: float):
    """The camera's structure, the depth's colour.

    **Not an average of the two.** `addWeighted` on two bright images is a
    brighter image with less contrast in both — on r1_sz it read as "the depth
    is barely there", because the office and the far half of the depth ramp are
    both pale and averaging them destroyed the little that distinguished them.

    Multiplying instead keeps the two channels of information separate and
    intact: **every pixel's brightness comes from the camera, every pixel's hue
    from the depth.** You can still see what a thing is, and its colour says how
    far away it is. `blend` then pulls back towards the plain camera image for
    anyone who wants less of it.

    Two fused OpenCV ops on 307k pixels, which is what this can afford next to a
    10 Hz command stream. `INTER_LINEAR` for the same reason — `INTER_AREA` is
    visibly better for big downscales and measurably slower, and nobody reads
    this image for its resampling.
    """
    import cv2

    base = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    if depth_m is None:
        return base
    colour = _colormap(depth_m, 0.0)
    if colour.shape[:2] != (height, width):
        colour = cv2.resize(colour, (width, height),
                            interpolation=cv2.INTER_NEAREST)
    luma = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    # 1/160 rather than 1/255: multiplying two images darkens, and the depth
    # ramp's own mid-tones are what carry the reading. The gain puts a mid-grey
    # scene back at roughly the ramp's own brightness instead of half of it.
    tinted = cv2.multiply(colour, cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR),
                          scale=1.0 / 160.0)
    if blend >= 1.0:
        return tinted
    return cv2.addWeighted(tinted, blend, base, 1.0 - blend, 0.0)


def _column_of(bearing_rad: float, half_fov_rad: float, width: int) -> int:
    """Image column for an angle off the nose. Right of centre is positive."""
    u = bearing_rad / max(half_fov_rad, 1e-6)
    return int(round((u + 1.0) / 2.0 * width))


def corridor_edges(clearance: float, config, width: int) -> tuple:
    """The two image columns the metric corridor occupies at `clearance`.

    Pulled out of the drawing so it can be checked without an image library —
    and because this mapping, metres to pixels via the field of view, is the
    exact computation that has produced every serious bug in this card. A
    picture drawn from a different mapping than the decision used would be worse
    than no picture: it would corroborate whatever the robot did.

    Returns `(left, right)`, or `None` when there is no reading to draw.
    """
    if not clearance or not math.isfinite(clearance) or clearance <= 0:
        return None
    keep = getattr(config, "half_width_m", 0.35) + config.clearance_margin_m
    edge = math.atan2(keep, max(clearance, 0.05))
    return (_column_of(-edge, config.half_fov_rad, width),
            _column_of(edge, config.half_fov_rad, width))


# Everything drawn on top gets a dark casing first.
#
# **The ramp this overlay now shares washes the far field out to near-white**
# (0xE6E4DD), which is the right thing for depth — the eye should pass over
# distances nobody acts on — and fatal for white text and white lines drawn on
# top of it. The first version was legible only because viridis happens to be
# dark; against the real ramp the distances disappeared into the background.
#
# Casing rather than a translucent box behind each label: it costs one extra
# draw, works on any background including the flat grey of "no reading", and
# does not hide the depth it is sitting on.
_CASING = (20, 20, 20)


def _text(frame, text: str, org, colour, scale: float = 0.5):
    import cv2

    for thickness, shade in ((3, _CASING), (1, colour)):
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, shade,
                    thickness, cv2.LINE_AA)


def _line(frame, start, end, colour, thickness: int = 2):
    import cv2

    cv2.line(frame, start, end, _CASING, thickness + 3, cv2.LINE_AA)
    cv2.line(frame, start, end, colour, thickness, cv2.LINE_AA)


def _arrow(frame, origin, dx, dy, colour, label: str):
    import cv2

    end = (int(origin[0] + dx), int(origin[1] + dy))
    cv2.arrowedLine(frame, origin, end, _CASING, 6, tipLength=0.3)
    cv2.arrowedLine(frame, origin, end, colour, 3, tipLength=0.3)
    if label:
        _text(frame, label, (end[0] + 6, end[1] + 4), colour, 0.45)


def render(*, depth_m, decision, track, box, clearance, coverage, config,
           measured=None, rgb_jpeg=None, blend: float = 0.5,
           width: int = 640, height: int = 480):
    """One frame. Every input may be absent — the overlay still draws.

    `box` is the target's `bbox_norm` (x1, y1, x2, y2 in 0..1) or None;
    `clearance`/`coverage` are what the corridor read this tick; `rgb_jpeg` is
    the raw camera frame, which turns the depth into a translucent mask over the
    real picture instead of a picture of its own.
    """
    import cv2

    far = max(config.slow_distance_m * 1.5, 2.0)
    rgb = None
    if rgb_jpeg:
        try:
            rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, np.uint8),
                               cv2.IMREAD_COLOR)
        except Exception:                                     # noqa: BLE001
            rgb = None

    if rgb is not None:
        # The camera and the depth map cover the same field of view — perception
        # resizes rather than crops — so a plain resize puts them in register.
        frame = _composite(rgb, depth_m, config, width, height, blend)
    elif depth_m is None:
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
    else:
        frame = _colormap(depth_m, far)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_NEAREST)

    keep = getattr(config, "half_width_m", 0.35) + config.clearance_margin_m

    # ── the corridor, at the distance it currently clears to ─────────────────
    #
    # Drawn at that distance rather than at a fixed one because the corridor is
    # metric: the same half-width covers a different slice of the picture at
    # every range, which is the single fact that took longest to become visible.
    edges = corridor_edges(clearance, config, width)
    if edges:
        left, right = edges
        blocked = clearance <= config.obstacle_stop_m
        colour = _WARN if blocked else _CORRIDOR

        # **Drawn as a dimension, not as two lines.**
        #
        # Two bare verticals were asked about twice: they read as unrelated
        # marks, and nothing connected them to the `+-0.33m` in the corner. A
        # cross-bar with arrow ends is the drawing convention for "this is a
        # width", and putting the number on the bar rather than in a corner is
        # what makes it answer its own question.
        bar = int(height * 0.30)
        for x in (left, right):
            _line(frame, (x, bar), (x, height), colour)
        cv2.arrowedLine(frame, (left, bar), (right, bar), _CASING, 6,
                        tipLength=0.04)
        cv2.arrowedLine(frame, (right, bar), (left, bar), _CASING, 6,
                        tipLength=0.04)
        cv2.arrowedLine(frame, (left, bar), (right, bar), colour, 2,
                        tipLength=0.04)
        cv2.arrowedLine(frame, (right, bar), (left, bar), colour, 2,
                        tipLength=0.04)
        span = f"path {keep * 2:.2f}m wide"
        (tw, _), _ = cv2.getTextSize(span, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        _text(frame, span, ((left + right - tw) // 2, bar - 8), colour, 0.5)
        _text(frame,
              f"clear to {clearance:.2f}m   cov {coverage * 100:.0f}%",
              (8, 24), colour, 0.55)

    # ── the target ───────────────────────────────────────────────────────────
    if box:
        x1, y1, x2, y2 = (int(box[0] * width), int(box[1] * height),
                          int(box[2] * width), int(box[3] * height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), _CASING, 5)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _TARGET, 2)
        state = (track or {}).get("state", "")
        rng = (track or {}).get("range_m")
        label = f"{state}" + (f"  {rng:.2f}m" if rng is not None else "  ?m")
        colour = _TARGET
        if measured is not None:
            # **Both numbers, because the gap between them is a real failure
            # mode.** The tracked range is a filtered estimate that has been
            # ego-motion compensated every tick since the last observation, and
            # without odometry that compensation runs on the *commanded* twist
            # — so a robot told to walk faster than it does drags its target
            # closer with nothing but a 4 Hz detector pulling back. The symptom
            # is a tracked range below the nearest thing in the whole picture,
            # which nothing was reporting.
            label += f"  (meas {measured:.2f}m)"
            if rng is not None and abs(rng - measured) > max(0.3, measured * 0.3):
                colour = _WARN
        _text(frame, label, (x1, max(y1 - 8, 18)), colour, 0.6)
    elif track and track.get("state") in ("coasting", "reacquiring"):
        # No box means the detector is not seeing it; the track is a prediction.
        # Saying so is the point — a robot walking towards a guess looks exactly
        # like one walking towards something it can see.
        _text(frame, f"{track['state']} (predicted)", (8, height - 62), _WARN,
              0.55)

    # ── the command ──────────────────────────────────────────────────────────
    values = (decision.values if decision is not None else None) or [0.0] * 6
    vx, vy, wz = values[0], values[1], values[5]
    base = (width // 2, height - 30)
    cv2.circle(frame, base, 6, _CASING, -1)
    cv2.circle(frame, base, 4, _ARROW, -1)
    if abs(vx) > 1e-6:
        _arrow(frame, base, 0, -vx * 90, _ARROW, f"vx {vx:+.2f}")
    if abs(vy) > 1e-6:
        # y is left-positive, so a positive command draws to the left.
        _arrow(frame, base, -vy * 90, 0, _ARROW, f"vy {vy:+.2f}")
    if abs(wz) > 1e-6:
        # Yaw has no direction in the image plane; drawn as a bar above the
        # origin, leaning the way the robot is turning.
        span = int(np.clip(wz / max(config.wz_max, 1e-6), -1, 1) * 70)
        cv2.arrowedLine(frame, (base[0], base[1] - 40),
                        (base[0] - span, base[1] - 40), _CASING, 6, tipLength=0.3)
        cv2.arrowedLine(frame, (base[0], base[1] - 40),
                        (base[0] - span, base[1] - 40), _ARROW, 3, tipLength=0.3)
        _text(frame, f"wz {wz:+.2f}", (base[0] - span - 20, base[1] - 48),
              _ARROW, 0.45)

    status = (decision.status if decision is not None else "idle") or "idle"
    distance = decision.distance_m if decision is not None else None
    line = status + (f"  target {distance:.2f}m" if distance is not None else "")
    _text(frame, line, (8, height - 10), (255, 255, 255), 0.65)
    return frame


def encode(frame, quality: int = 70) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:                                    # pragma: no cover - encoder
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)
