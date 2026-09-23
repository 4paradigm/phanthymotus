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

# Near is bright, far is dark. Viridis rather than the jet/rainbow ramp: it is
# perceptually uniform and survives colour blindness, and this image exists to
# be read quickly by someone standing next to a moving robot.
_NEAR_M = 0.3
_NO_READING = (70, 70, 70)
_CORRIDOR = (255, 255, 255)
_TARGET = (80, 220, 255)
_ARROW = (120, 255, 120)
_WARN = (80, 120, 255)


def _colormap(depth_m: np.ndarray, far_m: float):
    import cv2

    valid = np.isfinite(depth_m) & (depth_m > 0)
    span = max(far_m - _NEAR_M, 0.1)
    t = np.clip((np.nan_to_num(depth_m, nan=far_m) - _NEAR_M) / span, 0.0, 1.0)
    # Inverted so near reads bright: the near field is what the robot is about
    # to walk into, and it should be the loudest thing in the picture.
    frame = cv2.applyColorMap(((1.0 - t) * 255).astype(np.uint8),
                              cv2.COLORMAP_VIRIDIS)
    frame[~valid] = _NO_READING
    return frame


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


def _arrow(frame, origin, dx, dy, colour, label: str):
    import cv2

    end = (int(origin[0] + dx), int(origin[1] + dy))
    cv2.arrowedLine(frame, origin, end, colour, 3, tipLength=0.3)
    if label:
        cv2.putText(frame, label, (end[0] + 6, end[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def render(*, depth_m, decision, track, box, clearance, coverage, config,
           width: int = 640, height: int = 480):
    """One frame. `depth_m` may be None — the overlay still draws.

    `box` is the target's `bbox_norm` (x1, y1, x2, y2 in 0..1) or None;
    `clearance`/`coverage` are what the corridor read this tick.
    """
    import cv2

    far = max(config.slow_distance_m * 1.5, 2.0)
    if depth_m is None:
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
        for x in (left, right):
            cv2.line(frame, (x, int(height * 0.25)), (x, height), colour, 2)
        cv2.putText(frame,
                    f"corridor {clearance:.2f}m  cov {coverage * 100:.0f}%  "
                    f"+-{keep:.2f}m",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    # ── the target ───────────────────────────────────────────────────────────
    if box:
        x1, y1, x2, y2 = (int(box[0] * width), int(box[1] * height),
                          int(box[2] * width), int(box[3] * height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), _TARGET, 2)
        state = (track or {}).get("state", "")
        rng = (track or {}).get("range_m")
        label = f"{state}" + (f"  {rng:.2f}m" if rng is not None else "  ?m")
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TARGET, 1, cv2.LINE_AA)
    elif track and track.get("state") in ("coasting", "reacquiring"):
        # No box means the detector is not seeing it; the track is a prediction.
        # Saying so is the point — a robot walking towards a guess looks exactly
        # like one walking towards something it can see.
        cv2.putText(frame, f"{track['state']} (predicted)", (8, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WARN, 1, cv2.LINE_AA)

    # ── the command ──────────────────────────────────────────────────────────
    values = (decision.values if decision is not None else None) or [0.0] * 6
    vx, vy, wz = values[0], values[1], values[5]
    base = (width // 2, height - 24)
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
                        (base[0] - span, base[1] - 40), _ARROW, 3, tipLength=0.3)
        cv2.putText(frame, f"wz {wz:+.2f}", (base[0] - span - 20, base[1] - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ARROW, 1, cv2.LINE_AA)

    status = (decision.status if decision is not None else "idle") or "idle"
    distance = decision.distance_m if decision is not None else None
    line = status + (f"  target {distance:.2f}m" if distance is not None else "")
    cv2.putText(frame, line, (8, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def encode(frame, quality: int = 70) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:                                    # pragma: no cover - encoder
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)
