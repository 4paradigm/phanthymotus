"""motus.camera/1 for a processor in the middle of the chain.

Spec: `phanthymotus-driver/README_dev.md` § Camera Parameters. A **separate
implementation** of it, not a shared import — same rule as `motus.control/1`,
where the document is what is shared and the `schema` field is what keeps the
repositories' release cadence independent.

A camera card only ever *emits* a declaration. A navigation card only ever
*reads* one. A perception processor does both, and the interesting part is the
middle: **it must rewrite the parts its own processing changed, and only those.**

`visual_depth` is the case that makes this concrete. It resizes 1280x720 into
640x480 before publishing, so:

* `half_fov_rad` is **unchanged** — `cv2.resize` stretches the whole frame rather
  than cropping it, so the same angular extent is still in the picture. If that
  ever becomes a crop, this stops being true and the number has to be recomputed.
* `width`/`height` **are** changed: the declaration describes the image *this*
  port publishes, not the one it was handed.
* `K` **must be rescaled**, and forgetting to is a trap. `fx` is in pixels, so
  halving the width without halving `fx` leaves a consumer deriving
  `atan((640/2)/fx_original)` — an angle roughly half the truth, arrived at from
  two individually plausible fields. See `_rescale_K`.
* `D` survives unchanged: the radial terms are dimensionless in normalised image
  coordinates.
* `id` survives, which is what makes "both of my inputs are the same lens" a
  checkable statement three stages downstream.
* `pipeline` gains this stage, so a wrong number can be traced to whoever last
  touched it.

Why any of it exists: the field of view used to be typed into the *navigation*
card's config by hand. On r1_sz it read 0.55 rad against a lens measuring 0.888,
which made the metric avoidance corridor 1.86 m wide — wider than any door — so
every doorframe read as dead ahead and the robot turned away from openings it
fitted through, while the depth map reported clear.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "motus.camera/1"


def for_topic(declarations, topic: str) -> dict:
    """The upstream declaration covering `topic`, or `{}`.

    `declarations` is what agent-core passes on `start`: `{topic: declaration}`.
    A list is accepted too, so this also works against a card's raw `info()`.
    Joined on topic because that is the one thing both sides already agree on —
    not on list position (either side may have several ports) and not on the
    upstream card's name.
    """
    if isinstance(declarations, dict):
        entry = declarations.get(topic)
        return dict(entry) if isinstance(entry, dict) else {}
    for entry in declarations or ():
        if isinstance(entry, dict) and entry.get("topic") == topic:
            return dict(entry)
    return {}


def _rescale_K(K, from_size, to_size):
    """The intrinsic matrix after a resize, or None if it cannot be carried.

    **The trap this function exists for.** `K` holds focal lengths and a
    principal point in *pixels*, and a consumer derives the field of view as
    `atan((width/2) / fx)`. Publish a resized image while passing `K` through
    untouched and both fields are individually plausible while their ratio is
    wrong by exactly the resize factor — the same class of silent, confidently
    wrong geometry that this whole format was written against.

    The two axes scale independently on purpose: 1280x720 into 640x480 is not a
    uniform scale, and `fx`/`fy` are separate entries precisely so that case is
    expressible.
    """
    if not (isinstance(K, (list, tuple)) and len(K) == 9):
        return None
    fw, fh = from_size
    tw, th = to_size
    if not (fw and fh and tw and th):
        return None
    try:
        sx, sy = float(tw) / float(fw), float(th) / float(fh)
        k = [float(v) for v in K]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    # Row-major [fx, s, cx, 0, fy, cy, 0, 0, 1]: the x row scales with sx, the y
    # row with sy, and the bottom row is not in pixels at all.
    return [k[0] * sx, k[1] * sx, k[2] * sx,
            k[3] * sy, k[4] * sy, k[5] * sy,
            k[6], k[7], k[8]]


def inherit(upstream: dict, *, topic: str, fmt: str, stage: str,
            width: int = None, height: int = None) -> list:
    """This port's declaration, derived from what fed it. `[]` if nothing did.

    **An empty list when the upstream declared nothing, rather than an entry
    full of nulls.** The format requires a non-empty `id`, and this stage has no
    way to invent one — the camera's identity is not derivable from a topic
    name. A consumer seeing no entry falls back conservatively and says so,
    which is exactly right; a consumer seeing an entry with a made-up `id` would
    compare it against another made-up one and conclude two different lenses
    are the same.

    `width`/`height` default to the upstream's when this stage does not resize.
    """
    if not isinstance(upstream, dict) or not upstream:
        return []
    if upstream.get("schema") != SCHEMA:
        return []
    ident = str(upstream.get("id") or "")
    if not ident:
        return []

    src_w, src_h = upstream.get("width"), upstream.get("height")
    out_w = width if width is not None else src_w
    out_h = height if height is not None else src_h

    K = upstream.get("K")
    if K is not None and (out_w != src_w or out_h != src_h):
        K = _rescale_K(K, (src_w, src_h), (out_w, out_h))

    pipeline = list(upstream.get("pipeline") or [])
    if stage and (not pipeline or pipeline[-1] != stage):
        pipeline.append(stage)

    # `source` becomes `inherited` unless nothing was known to inherit: an
    # `unknown` upstream stays `unknown`, because relabelling "nobody knows" as
    # "inherited" would make it look like a number had been passed down.
    source = upstream.get("source") or "unknown"
    if upstream.get("half_fov_rad") is not None or K is not None:
        source = "inherited"

    return [{
        "schema": SCHEMA,
        "topic": topic,
        "format": fmt,
        "id": ident,
        "width": out_w,
        "height": out_h,
        "distortion_model": upstream.get("distortion_model") or "unknown",
        "D": upstream.get("D"),
        "K": K,
        # Unchanged by a resize that stretches rather than crops — see the module
        # docstring. A crop would have to recompute this.
        "half_fov_rad": upstream.get("half_fov_rad"),
        "half_fov_v_rad": upstream.get("half_fov_v_rad"),
        "source": source,
        "measured_on": upstream.get("measured_on") or "",
        "pipeline": pipeline,
        "vendor": dict(upstream.get("vendor") or {}),
    }]


def camera_id(declaration: dict) -> str:
    if not isinstance(declaration, dict):
        return ""
    return str(declaration.get("id") or "")


def _k_from_half_fov(camera, width, height):
    """K from a declared half field of view, or None if none was declared.

    **A declaration with no `K` is not an incomplete declaration.** The spec
    (`phanthymotus-driver/README_dev.md` § Camera Parameters) carries
    `half_fov_rad` precisely because "a tape measure produces one directly
    while producing no K at all", and its reference declaration — R1's
    `unitree/r1/camera_main` — is exactly that: `K: null`,
    `half_fov_rad: 0.888`, `source: measured`. Reading only `K` rejected the
    canonical camera of the robot this project is built around.

    Only consulted when `K` is absent, which keeps the spec's own precedence:
    when both are present K wins, because a calibration matrix is solved from
    many observations and the angle beside it is usually a tape measure.

    `fy` needs the vertical angle, which **cannot** be derived from the aspect
    ratio — the spec is explicit, and a stage that resizes 1280x720 into
    640x480 is why. So `half_fov_v_rad` is used when declared, and square
    pixels assumed when it is not. That assumption holds at a camera, which is
    where these declarations are born; it is the thing to suspect first if a
    chain ever puts a non-uniform resize upstream of this card.

    What it cannot fix: a pinhole K is an approximation on a wide lens. R1's is
    ~102 degrees with `distortion_model: unknown` and `D: null`, and `tan`
    overstates the lateral offset towards the edges. A metric result will need site
    calibration on that lens.
    """
    half = camera.get("half_fov_rad")
    if half in (None, ""):
        return None
    # Present but unusable is a different fault from absent, and saying
    # "declares neither" about a declaration that declares one sends the reader
    # to the wrong file.
    try:
        half = float(half)
    except (TypeError, ValueError):
        raise ValueError(
            f"camera_info half_fov_rad is not a number ({camera.get('half_fov_rad')!r}); "
            f"it is the horizontal half field of view in radians") from None
    if not np.isfinite(half) or not 0.0 < half < np.pi / 2:
        raise ValueError(
            f"camera_info half_fov_rad={half!r} is out of range; it is the "
            f"horizontal **half** field of view in radians, so a 102 degree "
            f"lens declares about 0.89, not 102")
    fx = (width / 2.0) / float(np.tan(half))
    half_v = camera.get("half_fov_v_rad")
    fy = fx
    if half_v not in (None, ""):
        try:
            half_v = float(half_v)
            if 0.0 < half_v < np.pi / 2 and np.isfinite(half_v):
                fy = (height / 2.0) / float(np.tan(half_v))
        except (TypeError, ValueError):
            pass
    return np.asarray([fx, 0.0, width / 2.0 - 0.5,
                       0.0, fy, height / 2.0 - 0.5,
                       0.0, 0.0, 1.0], dtype=np.float32)


def camera_matrix(camera):
    """Read a motus.camera/1 declaration; K is expressed at width × height.

    `K` when it is there, otherwise derived from `half_fov_rad` — see
    `_k_from_half_fov` for why the second case is a valid declaration and not a
    missing one.
    """
    if not isinstance(camera, dict):
        raise ValueError("camera_info requires K, width and height")
    try:
        width, height = int(camera["width"]), int(camera["height"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("camera_info requires width and height") from exc
    if width <= 0 or height <= 0:
        raise ValueError("camera_info must contain positive width/height")
    declared = camera.get("K")
    if declared in (None, "") or (isinstance(declared, (list, tuple)) and not declared):
        matrix = _k_from_half_fov(camera, width, height)
        if matrix is None:
            raise ValueError(
                "camera_info declares neither K nor half_fov_rad, so there "
                "is no focal length to work from. A camera that was measured "
                "with a tape measure should declare half_fov_rad.")
    else:
        try:
            matrix = np.asarray(declared, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("camera_info requires a nine-value K, width and height") from exc
    if matrix.shape != (9,):
        raise ValueError("camera_info must contain a nine-value K and positive width/height")
    matrix = matrix.reshape(3, 3).copy()
    if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-6)
            or abs(float(matrix[1, 0])) > 1e-6):
        raise ValueError("camera_info K must be finite pinhole intrinsics with positive focal lengths")
    return matrix, width, height
