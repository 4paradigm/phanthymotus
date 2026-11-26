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


def observation_ms(depth, fallback=None):
    """Return the observation time used for motion admission and registration."""
    if depth is None:
        return fallback
    if depth.get("receipt_timed"):
        return depth.get("frame_id", fallback)
    return depth.get("capture_ms") or (depth.get("frame_id") if fallback is None else fallback)


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
    return np.frombuffer(metres.tobytes(), dtype=metres.dtype).reshape(metres.shape)



def has_valid_returns(frame: np.ndarray | None) -> bool:
    return frame is not None and bool((np.isfinite(frame) & (frame > 0.)).any())


def _valid(patch: np.ndarray) -> np.ndarray:
    return patch[np.isfinite(patch) & (patch > 0)]


def _box_slices(shape, bbox_norm):
    height, width = shape
    x1, y1, x2, y2 = bbox_norm
    left = max(0, min(width - 1, int(x1 * width)))
    right = max(left + 1, min(width, int(x2 * width)))
    top = max(0, min(height - 1, int(y1 * height)))
    bottom = max(top + 1, min(height, int(y2 * height)))
    return slice(top, bottom), slice(left, right)


def sample_box(depth_m: np.ndarray, bbox_norm) -> float | None:
    """Representative optical depth in a normalized detection box."""
    valid = _valid(depth_m[_box_slices(depth_m.shape, bbox_norm)])
    if valid.size == 0:
        return None
    return round(float(np.percentile(valid, TARGET_PERCENTILE)), 3)


def front_surface(depth_m, bbox_norm, intrinsics, ground=None):
    """Project the nearest coherent patch in a detection box to (forward, left)."""
    return _person_surface(depth_m, bbox_norm, intrinsics, ground)[0]


def person_box_measurement(depth_m, box, intrinsics, ground=None):
    """Exclude a foreground layer only when the coherent surface identifies an occluder."""
    measured = sample_box(depth_m, box)
    if measured is None:
        return None, None
    point, split = _person_surface(depth_m, box, intrinsics, ground)
    if split is not None and measured < split:
        measured = float(point[0])
    return measured, point


def _person_surface(depth_m, bbox_norm, intrinsics, ground):
    rows, cols = _box_slices(depth_m.shape, bbox_norm)
    patch = depth_m[rows, cols]
    if min(patch.shape) < 3:
        return None, None
    valid = np.isfinite(patch) & (patch > 0.)
    if ground is not None:
        valid &= ~ground[rows, cols]
    windows = np.lib.stride_tricks.sliding_window_view(np.where(valid, patch, np.nan), (3, 3))
    low = windows.min(axis=(-2, -1))
    high = windows.max(axis=(-2, -1))
    supported = high - low <= .08
    if not supported.any():
        return None, None
    centre = windows[:, :, 1, 1]
    fx, cx = intrinsics.for_width(depth_m.shape[1])
    ray = (np.arange(cols.start + 1, cols.stop - 1) - cx) / fx
    horizontal_range = np.where(supported, centre * np.hypot(1., ray), np.inf)
    row, col = np.unravel_index(np.argmin(horizontal_range), horizontal_range.shape)
    middle = centre.shape[0] // 2
    bottom = 3 * centre.shape[0] // 4
    occluded_below = None
    if middle > 0 and row >= middle:
        # A low occluder can enter a person's box without belonging to the
        # person. Require a separate measured upper surface, and for a near
        # layer extending above the bottom quarter, evidence at a lower
        # corner where the occluder continues out of the detection box.
        edge = bottom if row >= bottom else middle
        upper = centre[:edge][supported[:edge]]
        if upper.size >= 9 and float(np.percentile(upper, TARGET_PERCENTILE) - centre[row, col]) > .60:
            ordered = np.sort(centre[supported])
            gaps = np.flatnonzero(np.diff(ordered) > .40)
            for gap in gaps:
                split = ordered[gap + 1]
                if centre[row, col] < split <= upper.min():
                    if row < bottom and not np.any(supported[-1, [0, -1]] & (centre[-1, [0, -1]] < split)):
                        continue
                    occluded_below = split
                    horizontal_range[centre < split] = np.inf
                    row, col = np.unravel_index(np.argmin(horizontal_range), horizontal_range.shape)
                    break
            if occluded_below is None and row >= bottom:
                lower = _wide_lower_occluder(depth_m, ground, rows, cols, centre, supported, row, col)
                if lower is not None:
                    horizontal_range[lower] = np.inf
                    row, col = np.unravel_index(np.argmin(horizontal_range), horizontal_range.shape)
                    occluded_below = float(centre[row, col])
    forward = float(centre[row, col])
    return np.array([forward, -forward * ray[col]]), occluded_below


def _wide_lower_occluder(depth_m, ground, rows, cols, centre, supported, row, col):
    """Recognize a lower occluder whose sloped top bridges the global depth gap."""
    near = centre[row, col]
    middle = len(centre) // 2
    upper = centre[:middle][supported[:middle]]
    if upper.size < 9 or upper.min() - near <= .40:
        return None
    y, x = rows.start + row + 1, cols.start + col + 1
    width = cols.stop - cols.start
    for end in (cols.start - width // 2, cols.stop - 1 + width // 2):
        if not 0 <= end < depth_m.shape[1]:
            continue
        strip = np.s_[y - 1:y + 2, min(x, end):max(x, end) + 1]
        if ground is not None and ground[strip].any():
            continue
        band = depth_m[strip]
        if not np.all((band > 0.) & (np.abs(band - near) <= .08)):
            continue
        # A separate hand or foot in the box keeps its own standoff constraint.
        layer = supported & (centre < near + .40)
        layer[:middle] = False
        return _connected_surface(layer, row, col)
    return None


def _connected_surface(mask, row, col):
    """Four-connected component of coherent samples, filled one scanline at a time."""
    found = np.zeros_like(mask)
    pending = [(row, col)]
    while pending:
        y, x = pending.pop()
        if found[y, x]:
            continue
        line = mask[y]
        left = np.flatnonzero(~line[:x])
        right = np.flatnonzero(~line[x:])
        lo = left[-1] + 1 if left.size else 0
        hi = x + right[0] if right.size else len(line)
        found[y, lo:hi] = True
        for other in (y - 1, y + 1):
            if 0 <= other < len(mask):
                run = mask[other, lo:hi] & ~found[other, lo:hi]
                starts = np.flatnonzero(run & ~np.r_[False, run[:-1]])
                pending.extend((other, lo + i) for i in starts)
    return found


def sample_point(depth_m: np.ndarray, position, window: float = 0.05) -> float | None:
    """Distance at a centre point, sampling a small window around it.

    The degraded path, used when vop is publishing without `bbox`. `position`
    is vop's centre-relative pair in -1..1. A window rather than one pixel
    because a single sample of a monocular depth map is noise.
    """
    height, width = depth_m.shape

    x, y = position
    half_w, half_h = window, window * width / height
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
    height, width = depth_m.shape

    top = int(height * OBSTACLE_ROW_SPAN[0])
    bottom = int(height * OBSTACLE_ROW_SPAN[1])
    rows = depth_m[top:bottom, :]

    out = {}
    edges = [round(i * width / len(BANDS)) for i in range(len(BANDS) + 1)]
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


_GROUND_TOLERANCE_M = 0.025


def ground_mask(depth_m: np.ndarray, horizontal, vertical) -> np.ndarray | None:
    """Identify a broadly supported floor in this rectified optical depth frame.

    Fit Y = a*X + b*Z + c from below the obstacle crop. The camera coordinates
    have positive Y down. A plausible floor is below the camera, roughly
    horizontal, and has support both across the image and along depth.
    """
    height, width = depth_m.shape
    fx, cx = horizontal.for_width(width)
    source_height, fy, cy = vertical
    scale = height / source_height
    fy, cy = fy * scale, (cy + 0.5) * scale - 0.5
    u = (np.arange(width) - cx) / fx
    v = (np.arange(height) - cy) / fy

    bottom = slice(int(height * OBSTACLE_ROW_SPAN[1]), height, 4)
    columns = slice(width // 16, width - width // 16, 4)
    seed = depth_m[bottom, columns]
    count = seed.shape[1]
    narrow = max(2, count // 8)
    quarter = max(narrow, count // 4)
    valid = np.isfinite(seed) & (seed > 0.10) & (seed < 4.0)
    if valid.mean() < 0.6:
        return None
    x = (u[columns][None, :] * seed)[valid]
    y = (v[bottom][:, None] * seed)[valid]
    z = seed[valid]
    points = np.column_stack((x, z, np.ones_like(z)))

    if not (np.isfinite(points).all() and np.isfinite(y).all()):
        return None

    positions = np.nonzero(valid)[1]
    all_points = np.ones(len(z), dtype=bool)

    def on_ground(distance):
        # Be stricter above the plane to retain low obstacles.
        return (distance >= -0.020) & (distance <= _GROUND_TOLERANCE_M)

    def fit(inliers, *, near_thirds=(0, 1, 2), visible_span=None,
            window_halves=False, near_wall=False):
        start, end = visible_span or (0, seed.shape[1])
        visible = (positions >= start) & (positions < end)
        min_inliers = 0.6 * seed.shape[0] * (end - start)
        inliers = inliers & visible
        if np.count_nonzero(inliers) < 3:
            return None
        first_tolerance = _GROUND_TOLERANCE_M if near_thirds is None else 0.05
        for tolerance in (first_tolerance, _GROUND_TOLERANCE_M, _GROUND_TOLERANCE_M):
            try:
                a, b, c = np.linalg.lstsq(points[inliers], y[inliers], rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            inliers = visible & (np.abs(a * x + b * z + c - y) <= tolerance)
            if inliers.sum() < min_inliers:
                return None
        if not (0.15 <= c <= 2.0 and np.hypot(a, b) <= 0.5):
            return None
        near, far = np.percentile(z[inliers], [10, 90])
        if far - near < 0.1:
            return None
        support = np.zeros_like(valid)
        support[valid] = inliers
        thirds = np.array_split(support, 3, axis=1)
        if window_halves:
            if any(part.mean() < 0.65 for part in np.array_split(support[:, start:end], 2, axis=1)):
                return None
        elif near_thirds is None:
            if sum(third.mean() >= 0.50 for third in thirds) < 2:
                return None
        elif any(thirds[third].mean() < 0.65 for third in near_thirds):
            return None

        def residual(rows, cols):
            return (v[rows, None] - a * u[None, cols] - b) * depth_m[rows, cols] - c

        # Verify the fit farther into the scene to reject a raised platform
        # occupying most of the bottom band. Only project the sampled columns
        # until a candidate passes the checks.
        far_end = 0.78 if visible_span is not None else 0.7375
        far_start = int(height * 0.69)
        sampled = residual(slice(far_start, None), columns)
        far_count = int(height * far_end) - far_start
        if near_wall:
            # A close back wall can hide the original verification band.
            # Require measured foreground or floor throughout that occluded
            # region, then verify a disjoint strip of the original height
            # immediately above the near fit.
            end_row = bottom.start - far_start
            if not 0 < far_count < end_row:
                return None
            upper = sampled[:end_row - far_count]
            explained = np.isfinite(upper) & ((upper < -2 * _GROUND_TOLERANCE_M) | on_ground(upper))
            if any(part.mean() < .80 for part in np.array_split(explained, 3, axis=1)):
                return None
            farther = on_ground(sampled[end_row - far_count:end_row])
        else:
            farther = on_ground(sampled[:far_count])
        far_thirds = np.array_split(farther, 3, axis=1)
        verified = near_thirds if visible_span is not None else (0, 1, 2)
        if window_halves:
            far_support = [part.mean() for part in np.array_split(farther[:, start:end], 2, axis=1)]
            if min(far_support) < .06 or max(far_support) < .35:
                return None
        else:
            far_support = [far_thirds[third].mean() for third in verified]
        one_far_side = sum(support >= 0.06 for support in far_support) < 2
        if one_far_side and max(far_support) < 0.35:
            return None
        # Veto a substantial measured surface below the proposed floor.
        # This rejects a raised platform even if its top fits one patch.
        below = sampled > 2 * _GROUND_TOLERANCE_M
        if end - start <= quarter and below[:, start:end].mean() >= .12:
            return None
        if any(third.mean() >= 0.12 for third in np.array_split(below, 3, axis=1)):
            return None
        keep_columns = None
        if visible_span is not None:
            # Each excluded side must show a foreground occlusion. Returns
            # below the candidate can reveal a raised platform.
            occlusion_margin = 2 * _GROUND_TOLERANCE_M
            for excluded in (slice(0, start), slice(end, seed.shape[1])):
                side = sampled[bottom.start - far_start::bottom.step, excluded]
                if not side.size:
                    continue
                observed = valid[:, excluded]
                foreground = observed & (side < -occlusion_margin)
                below = observed & (side > occlusion_margin)
                if foreground.mean() < 0.2 or below.mean() >= 0.2:
                    return None
            # The occluded sides did not verify this plane. Keep their returns
            # available to obstacle detection even if they touch the plane.
            keep_columns = np.ones(width, dtype=bool)
            if start:
                keep_columns[:columns.start + start * columns.step] = False
            if end < seed.shape[1]:
                keep_columns[columns.start + end * columns.step:] = False
            if end - start <= quarter:
                # The edge of a narrow window can still include a jamb.
                # A return where the jamb crosses the plane is not floor.
                keep_columns &= on_ground(residual(bottom, slice(None))).mean(axis=0) >= .65
                if not keep_columns.any():
                    return None
        ground = on_ground(residual(slice(None), slice(None)))
        if keep_columns is not None:
            ground[:, ~keep_columns] = False
        return ground

    ground = fit(all_points)
    if ground is not None:
        return ground

    # A foreground box can bias the seed fit even when most samples are floor.
    # Seed from the outside of the image and verify against the full bottom band.
    third = seed.shape[1] // 3
    left, right = positions < third, positions >= 2 * third
    for outside in (left | right, left, right):
        ground = fit(outside, near_thirds=(0, 2))
        if ground is not None:
            return ground

    # A box against one side can hide most of that third. Require the same
    # plane in the centre and the other side, both in the seed and farther away.
    visible_spans = (((0, 1), (0, seed.shape[1] - third)),
                     ((1, 2), ((seed.shape[1] + 2) // 3, seed.shape[1])))
    for near_thirds, span in visible_spans:
        ground = fit(all_points, near_thirds=near_thirds, visible_span=span)
        if ground is not None:
            return ground

    # A near box can bias the full-width fit. Seed hypotheses on narrow
    # strips farther out, then verify them against the independent near band.
    farther_rows = slice(int(height * 0.72), bottom.start, 4)
    farther = depth_m[farther_rows, columns]
    farther_x = u[columns] * farther
    farther_y = v[farther_rows, None] * farther
    for patch in np.array_split(np.arange(farther.shape[1]), 6):
        patch_z = farther[:, patch]
        usable = np.isfinite(patch_z) & (patch_z > .10) & (patch_z < 4.)
        patch_z = patch_z[usable]
        if patch_z.size < 12 or np.ptp(patch_z) < .15:
            continue
        patch_points = np.column_stack((farther_x[:, patch][usable], patch_z,
                                       np.ones_like(patch_z)))
        patch_y = farther_y[:, patch][usable]
        a, b, c = np.linalg.lstsq(patch_points, patch_y, rcond=None)[0]
        if np.mean(np.abs(patch_points @ (a, b, c) - patch_y) <= _GROUND_TOLERANCE_M) < .80:
            continue
        ground = fit(np.abs(a * x + b * z + c - y) <= _GROUND_TOLERANCE_M,
                     near_thirds=None)
        if ground is not None:
            return ground

    # Mixed centre returns can bias a fit even when the outer third is clear.
    # Verify the centre and the visible side, including a check for lower surfaces.
    for (near_thirds, span), edge in zip(visible_spans, (left, right)):
        ground = fit(edge, near_thirds=near_thirds, visible_span=span)
        if ground is not None:
            return ground

    # A wide obstacle can cover the centre and leave a continuous strip on
    # only one side. Require support in both halves of that strip and leave
    # all returns on the occluded side available to obstacle detection.
    visible = 2 * count // 5
    for edge, span in ((left, (0, visible)), (right, (count - visible, count))):
        ground = fit(edge, visible_span=span, window_halves=True)
        if ground is not None:
            return ground

    # Two foreground surfaces can leave a continuous strip of floor in the
    # middle. Apply the same support checks and require occlusion on both sides.
    start = (count - visible) // 2
    ground = fit(all_points, visible_span=(start, start + visible),
                 window_halves=True)
    if ground is not None:
        return ground

    # An oblique view can leave a narrower floor strip away from the centre.
    # Keep the same near/far checks and retain all depth outside the strip.
    if count <= narrow:
        return None
    last = count - narrow
    for start in (0, *range(1, last, max(1, narrow // 8)), last):
        ground = fit(all_points, visible_span=(start, start + narrow),
                     window_halves=True)
        if ground is not None:
            return ground

    # Far floor can be visible on the inner part of an edge strip while its
    # outer rays are occluded. Validate both halves across the whole strip.
    for start in (0, count - quarter):
        ground = fit(all_points, visible_span=(start, start + quarter),
                     window_halves=True)
        if ground is not None:
            return ground
    return fit(all_points, near_wall=True)


def corridor(depth_m: np.ndarray, *, half_width_m: float, half_fov_rad: float,
             reference_m: float, lateral_offset_m: float = 0.0,
             intrinsics=None, ground=None, ignored_low_pixels=None):
    """Nearest obstacle inside a metric corridor, and how much of it was measured.

    Returns `(clearance_m | None, coverage)`. With floor returns only, clearance
    is bounded by their observed depth across the corridor. `coverage` is the fraction of the frame
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

    `coverage` reports how much of that corridor has a valid depth return.

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
    height, width = depth_m.shape

    top = int(height * OBSTACLE_ROW_SPAN[0])
    bottom = int(height * OBSTACLE_ROW_SPAN[1])
    rows = depth_m[top:bottom, :]

    # Right-positive ray tangents. Optical Z converts them to lateral metres.
    if intrinsics is None:
        cx = (width - 1) / 2.0
        step = 2.0 * np.tan(half_fov_rad) / width
    else:
        fx, cx = intrinsics.for_width(width)
        step = 1.0 / fx
    tangent = (np.arange(width) - cx) * step

    lateral = tangent[None, :] * rows
    inside = np.abs(lateral - lateral_offset_m) <= half_width_m
    returns = np.isfinite(rows) & (rows > 0.)
    valid = inside & returns
    # Ground returns still contribute to depth coverage. Holes remain unknown.
    if ground is not None:
        valid &= ~ground[top:bottom]
    if ignored_low_pixels is not None:
        valid &= ~ignored_low_pixels[top:bottom]
    samples = rows[valid]
    clearance = (round(float(np.percentile(samples, OBSTACLE_PERCENTILE)), 3)
                 if samples.size else None)
    if clearance is None and ground is not None:
        # If everything we saw was floor, report a conservative observed
        # depth bound. Require floor returns across the width of the corridor.
        floor = inside & returns & ground[top:bottom]
        strip_width = 2 * half_width_m / 3
        left = lateral_offset_m - half_width_m
        extents = []
        for strip in range(3):
            low = left + strip * strip_width
            depths = rows[floor & (lateral >= low) & (lateral <= low + strip_width)]
            if depths.size < 16:
                break
            extents.append(float(np.percentile(depths, 80)))
        if len(extents) == 3:
            clearance = round(min(extents), 3)

    # Count the entire projected corridor at `reference_m`, including columns
    # outside the lens, so the reported coverage reflects the field of view.
    reference = max(1e-6, reference_m)
    low = (lateral_offset_m - half_width_m) / reference
    high = (lateral_offset_m + half_width_m) / reference
    wanted = max(1e-9, (high - low) / step)

    window = (tangent >= low) & (tangent <= high)
    visible = int(window.sum())
    if not visible:
        # The corridor is entirely outside the lens. Zero, not a division by
        # zero — and zero is the truth: the camera cannot see there at all.
        return clearance, 0.0
    region = returns[:, window]
    measured = float(region.mean()) if region.size else 0.0
    return clearance, round(measured * min(1.0, visible / wanted), 3)
