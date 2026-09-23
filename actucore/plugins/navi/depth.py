"""Reading distances out of whatever depth source is wired to the card.

Three shapes reach this card and only the first is precise:

  `image/depth-zlib`   zlib of 640x480 little-endian uint16 millimetres, 0
                       meaning "no reading". Published by perception's
                       `visual_depth` (monocular) and by the RealSense driver
                       (a real depth camera), so one decoder serves both.
  `data/json`          `visual_depth`'s summary — three numbers, one per
                       vertical third. Enough to avoid a wall, not enough to
                       say how far away the chair is.
  nothing              the card refuses to start.

Deliberately free of the plugin and of ROS: a zlib buffer in, numbers out, so
the geometry is testable without a camera. The encoder's contract is mirrored
rather than imported — `perception/plugins/visual_depth.py::encode_depth` is in
a different bundle and a different image, and a test that needs both installed
is a test nobody runs.

── the rule that runs through all of it ─────────────────────────────────────

**Zero means "no reading", never "zero metres".** The encoder writes 0 for
anything out of range, and an obstacle at 0.0 m is the most alarming thing a
navigation policy can be told. Every function here filters invalid samples
before doing arithmetic, and returns `None` rather than a number when nothing
valid is left — the same null-versus-zero distinction `motus.odom/1` is built
around, for the same reason.
"""

from __future__ import annotations

import zlib

import numpy as np

# Fixed by the renderer and by the encoder — see perception's visual_depth.
WIDTH = 640
HEIGHT = 480

# Percentile used for "how far is the thing in this box". Not the minimum: a
# monocular depth map has a scatter of near-zero outliers on every object edge,
# and one of them would report the target a metre closer than it is. Not the
# mean either: a box around a chair contains a great deal of floor and wall.
TARGET_PERCENTILE = 25

# Percentile for "how close is the nearest obstacle in this band". Lower,
# because here the near tail is the signal rather than the noise — the same
# choice, and the same number, perception's own `summarize_depth` makes.
OBSTACLE_PERCENTILE = 5

# Bands, left to right. Named rather than indexed so a caller cannot get the
# order wrong, and matching `visual_depth`'s summary keys so the degraded path
# and the precise path are interchangeable.
BANDS = ("left", "center", "right")

# Rows to consider when looking for obstacles: the middle band of the image.
# The top of a frame is ceiling and sky, the bottom is the floor directly under
# the robot — both are always "close" and neither is something to steer around.
OBSTACLE_ROW_SPAN = (0.35, 0.85)


class DepthError(ValueError):
    """A depth payload that cannot be decoded into a map."""


def decode(payload: bytes) -> np.ndarray:
    """A `image/depth-zlib` buffer as a 640x480 float array of metres.

    Invalid samples come back as `np.nan`, not 0.0, so any arithmetic on them
    propagates instead of quietly reading as "touching the robot".
    """
    try:
        raw = zlib.decompress(payload)
    except zlib.error as exc:
        raise DepthError(f"not a zlib buffer: {exc}") from exc

    expected = WIDTH * HEIGHT * 2
    if len(raw) != expected:
        # Refused rather than reshaped. The renderer silently drops a short
        # buffer and shows a blank panel with nothing in any log; a consumer
        # that steers a robot must not inherit that behaviour.
        raise DepthError(
            f"expected {expected} bytes ({WIDTH}x{HEIGHT} uint16), got {len(raw)}")

    millimetres = np.frombuffer(raw, dtype="<u2").reshape(HEIGHT, WIDTH)
    metres = millimetres.astype(np.float32) / 1000.0
    metres[millimetres == 0] = np.nan
    return metres


def _valid(patch: np.ndarray) -> np.ndarray:
    return patch[np.isfinite(patch) & (patch > 0)]


def sample_box(depth_m: np.ndarray, bbox_norm) -> float | None:
    """Distance to whatever fills a normalised box, or None if unreadable.

    `bbox_norm` is `(x1, y1, x2, y2)` in 0..1 of the frame. Normalised rather
    than pixels because vop's box is in *its* camera's resolution and this map
    has been resampled to 640x480 — passing pixels across that boundary is an
    error that produces plausible numbers for the wrong part of the image.
    """
    x1, y1, x2, y2 = bbox_norm
    left = max(0, min(WIDTH - 1, int(x1 * WIDTH)))
    right = max(left + 1, min(WIDTH, int(x2 * WIDTH)))
    top = max(0, min(HEIGHT - 1, int(y1 * HEIGHT)))
    bottom = max(top + 1, min(HEIGHT, int(y2 * HEIGHT)))

    valid = _valid(depth_m[top:bottom, left:right])
    if valid.size == 0:
        return None
    return round(float(np.percentile(valid, TARGET_PERCENTILE)), 3)


def sample_point(depth_m: np.ndarray, position, window: float = 0.05) -> float | None:
    """Distance at a centre point, sampling a small window around it.

    The degraded path, used when vop is publishing without `bbox`. `position`
    is vop's centre-relative pair in -1..1. A window rather than one pixel
    because a single sample of a monocular depth map is noise.
    """
    x, y = position
    half_w, half_h = window, window * WIDTH / HEIGHT
    return sample_box(depth_m, (
        (x + 1) / 2 - half_w, (y + 1) / 2 - half_h,
        (x + 1) / 2 + half_w, (y + 1) / 2 + half_h,
    ))


def nearest_by_band(depth_m: np.ndarray) -> dict:
    """Closest obstacle in each vertical third, or None where nothing is valid.

    Restricted to `OBSTACLE_ROW_SPAN`: the floor under the robot's feet is
    always the closest thing in the frame and is never what it should turn to
    avoid.
    """
    top = int(HEIGHT * OBSTACLE_ROW_SPAN[0])
    bottom = int(HEIGHT * OBSTACLE_ROW_SPAN[1])
    rows = depth_m[top:bottom, :]

    out = {}
    edges = [round(i * WIDTH / len(BANDS)) for i in range(len(BANDS) + 1)]
    for i, name in enumerate(BANDS):
        valid = _valid(rows[:, edges[i]:edges[i + 1]])
        out[name] = (round(float(np.percentile(valid, OBSTACLE_PERCENTILE)), 3)
                     if valid.size else None)
    return out


def bands_from_summary(summary: dict) -> dict:
    """The same three numbers out of `visual_depth`'s json summary.

    The degraded path. `visual_depth` already computed these with the same
    percentile over the whole frame height, so they are slightly more
    pessimistic than `nearest_by_band` — it has no way to exclude the floor.
    Kept anyway: a card with only the summary wired can still avoid a wall.
    """
    regions = (summary or {}).get("nearest_by_region") or {}
    return {name: regions.get(name) for name in BANDS}


# ── the corridor: what the robot will actually drive through ─────────────────


def corridor(depth_m: np.ndarray, *, half_width_m: float, half_fov_rad: float,
             reference_m: float, lateral_offset_m: float = 0.0):
    """Nearest obstacle inside a metric corridor, and how much of it was measured.

    Returns `(clearance_m | None, coverage)`. `clearance_m` is None when nothing
    valid falls inside the corridor; `coverage` is the fraction of the frame
    region the corridor occupies at `reference_m` that carries a real reading.

    ── why this exists, and why `nearest_by_band` is not enough ─────────────

    A band is a fixed slice of the *image*, so it covers a different width of
    the *world* at every distance. With a 63° lens the centre third spans
    `0.204 × distance` either side of the axis:

        at 0.8 m   ±0.16 m      at 1.1 m   ±0.22 m      at 2.0 m   ±0.41 m

    A humanoid is about ±0.18 m wide at the shoulders before any margin. So the
    band is **narrower than the robot below about a metre** — which is to say
    below the distance at which anything is decided — and wider than it beyond,
    where it only causes needless slowing. The near case is how a shoulder
    clips a doorframe while the depth map reports the way ahead as clear: the
    obstacle was in the frame, in the left third, and the left third has never
    stopped forward motion.

    So the test is metric: convert each pixel's column to a lateral offset at
    its own measured depth, and keep what lies inside the robot's width.

    ── unknown is not free ──────────────────────────────────────────────────

    `coverage` is the other half, and the more important one. A depth map has
    holes — dark, shiny, thin, or too close — and the objects that produce them
    are exactly the ones that catch a shoulder: chair legs, table edges, glass.
    Returning only a clearance lets a corridor full of holes read as an empty
    one, because every invalid pixel silently drops out of the minimum.

    The denominator is the image region the corridor would occupy at
    `reference_m` (pass the distance you care about — the stop distance).
    Reporting it separately rather than folding it into the clearance keeps the
    two questions apart: "how far is the nearest thing I can see" and "how much
    of what I need to see can I see at all".

    `lateral_offset_m` shifts the corridor sideways, **right-positive** to match
    `position[0]` from vop. That is how a sidestep is checked: the question is
    not whether the way ahead is clear but whether the way the robot is about to
    move into is.
    """
    top = int(HEIGHT * OBSTACLE_ROW_SPAN[0])
    bottom = int(HEIGHT * OBSTACLE_ROW_SPAN[1])
    rows = depth_m[top:bottom, :]

    # Tangent of the horizontal angle for each column, right-positive. Derived
    # from the half-FOV rather than from intrinsics because **nothing in this
    # project publishes intrinsics** — see the card's degradations. It is the
    # one number here that is a configuration rather than a measurement.
    tangent = ((np.arange(WIDTH) - (WIDTH - 1) / 2.0)
               * (2.0 * np.tan(half_fov_rad) / WIDTH))

    lateral = tangent[None, :] * rows
    inside = np.abs(lateral - lateral_offset_m) <= half_width_m
    valid = inside & np.isfinite(rows)
    samples = rows[valid]
    clearance = (round(float(np.percentile(samples, OBSTACLE_PERCENTILE)), 3)
                 if samples.size else None)

    # Coverage denominator: the columns the corridor projects onto at
    # `reference_m` — **including the part that falls outside the lens**.
    #
    # That last clause matters more than it looks. Close in, the robot is wider
    # than the field of view: at half a metre a ±0.33 m corridor spans about 67°
    # and a 63° lens does not reach the edges of it. Counting only the visible
    # columns would report a fully-measured corridor while a third of the
    # robot's width was never in frame — the same "unknown reads as clear"
    # failure this function exists to close, one level up.
    reference = max(1e-6, reference_m)
    low = (lateral_offset_m - half_width_m) / reference
    high = (lateral_offset_m + half_width_m) / reference
    step = 2.0 * np.tan(half_fov_rad) / WIDTH
    wanted = max(1e-9, (high - low) / step)

    window = (tangent >= low) & (tangent <= high)
    visible = int(window.sum())
    if not visible:
        # The corridor is entirely outside the lens. Zero, not a division by
        # zero — and zero is the truth: the camera cannot see there at all.
        return clearance, 0.0
    region = rows[:, window]
    measured = float(np.isfinite(region).mean()) if region.size else 0.0
    return clearance, round(measured * min(1.0, visible / wanted), 3)
