"""Reading `motus.camera/1` — the consumer half of the camera declaration.

The spec is `phanthymotus-driver/README_dev.md` § Camera Parameters. This is a
**separate implementation of it**, not a shared import: the same rule as
`motus.control/1`, where a shared schema package would couple the two
repositories' release cadence and the `schema` field exists precisely to keep
them independent. What is shared is the document.

So this module is deliberately small and forgiving where the driver's is strict.
A driver must not be able to *emit* a declaration that is out of spec — its
`build()` validates and raises, in its own unit test. A consumer facing a
declaration it does not like has a different job: carry on, conservatively, and
say what it could not read. Refusing to navigate because a camera card shipped a
malformed field would turn one card's bug into a robot that will not move.

Why any of this exists: navi's avoidance corridor is metric, so every tick it
converts its half-width back into a range of image columns using the camera's
horizontal half field of view. On r1_sz that number was typed into navi's own
config as 0.55 rad against a lens measuring 0.888, and the corridor came out
1.86 m wide — wider than any door. Every doorframe read as dead ahead and the
robot turned away from openings it fitted through, while the depth map reported
clear. The camera knew, and had no way to say so.
"""
from __future__ import annotations

import math
from typing import NamedTuple

SCHEMA = "motus.camera/1"


def for_topic(declarations, topic: str) -> dict:
    """The declaration covering `topic`, or `{}`.

    How a consumer joins: it already knows which topic it bound as its depth
    input, so it looks that topic up. Not by position in the list — a card may
    publish several ports and a consumer may have several inputs — and not by
    the upstream card's name, which would undo the reason this card dispatches
    inputs by what they carry.

    `declarations` is what agent-core passes on `start`: a mapping from topic to
    declaration. A list is accepted too, so this also works when handed a
    card's raw `info()` output.
    """
    if isinstance(declarations, dict):
        entry = declarations.get(topic)
        return entry if isinstance(entry, dict) else {}
    for entry in declarations or ():
        if isinstance(entry, dict) and entry.get("topic") == topic:
            return entry
    return {}


class HorizontalIntrinsics(NamedTuple):
    width: float
    fx: float
    cx: float

    def for_width(self, width: int) -> tuple[float, float]:
        """Scale the focal length and principal point using pixel centres."""
        scale = width / self.width
        return self.fx * scale, (self.cx + 0.5) * scale - 0.5


def horizontal_intrinsics(declaration: dict) -> HorizontalIntrinsics | None:
    """Read the horizontal intrinsics of a rectified depth image."""
    if declaration.get("schema") != SCHEMA or declaration.get("distortion_model") != "pinhole":
        return None
    k = declaration.get("K")
    width = declaration.get("width")
    if not isinstance(k, (list, tuple)) or len(k) != 9:
        return None
    if type(width) is not int or width <= 0 or k[1] != 0 or tuple(k[6:]) != (0, 0, 1):
        return None
    values = (k[0], k[2])
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return None
    try:
        width, fx, cx = map(float, (width, *values))
    except OverflowError:
        return None
    if not (0 < fx < math.inf and 0 <= cx < width and math.isfinite(width / fx)):
        return None
    return HorizontalIntrinsics(width, fx, cx)


def vertical_intrinsics(declaration: dict) -> tuple[int, float, float] | None:
    """Read the vertical axis of the rectified depth camera calibration."""
    if horizontal_intrinsics(declaration) is None:
        return None
    height = declaration.get("height")
    k = declaration["K"]
    if type(height) is not int or height <= 0 or k[3] != 0:
        return None
    fy, cy = k[4], k[5]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (fy, cy)):
        return None
    try:
        fy, cy = float(fy), float(cy)
    except OverflowError:
        return None
    if not (0 < fy < math.inf and 0 <= cy < height and math.isfinite(height / fy)):
        return None
    return height, fy, cy


def half_fov(declaration: dict) -> tuple:
    """`(half_fov_rad, source)` for the horizontal axis, or `(None, reason)`.

    **`K` wins over the declared angle when both are present.** A calibration
    matrix is solved from many observations; the angle beside it is usually a
    tape measure and some trigonometry. Preferring the coarser number because it
    sits in a handier field would be backwards.

    Anything unusable returns `None` with a reason fit to put in front of an
    operator, rather than a default — telling the two apart is the entire point
    of the format, and a consumer that substitutes a plausible number here
    recreates the bug it was written to prevent.
    """
    if not isinstance(declaration, dict) or not declaration:
        return None, "没有声明"

    schema = declaration.get("schema")
    if schema != SCHEMA:
        return None, f"schema 是 {schema!r}，不是 {SCHEMA!r}"

    K = declaration.get("K")
    width = declaration.get("width")
    if isinstance(K, (list, tuple)) and len(K) == 9 and width:
        try:
            fx, width = float(K[0]), float(width)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if 0 < fx < math.inf and 0 < width < math.inf:
                angle = math.atan((width / 2.0) / fx)
                if 0 < angle < math.pi / 2:
                    return angle, "derived-from-K"

    angle = declaration.get("half_fov_rad")
    if angle is None:
        return None, "声明里没有视场角"
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        return None, f"half_fov_rad 不是数字：{angle!r}"
    angle = float(angle)
    if not 0.0 < angle < math.pi / 2:
        # The driver's `build()` refuses these, so reaching here means a card
        # that did not go through it. Named rather than clamped: the likeliest
        # cause is the full angle in a half-angle field, and silently halving it
        # would be a guess about somebody else's bug.
        return None, (f"half_fov_rad={angle:.3f} 不在 (0, π/2) 内 —— 这个字段是"
                      f"**半**视场角，填进来的像是全视场角")

    source = declaration.get("source") or "unspecified"
    return angle, str(source)


def camera_id(declaration: dict) -> str:
    """The camera identity, which survives every stage of the chain.

    What makes "these two inputs are looking through the same lens" a checkable
    statement rather than an assumption.
    """
    if not isinstance(declaration, dict):
        return ""
    return str(declaration.get("id") or "")


def mount_topic(vendor):
    """Resolve the dynamic transform only from the bound depth camera's declaration."""
    mount = vendor.get("body_mount")
    if (not isinstance(mount, dict) or mount.get("schema") != "motus.camera_mount/1"
            or mount.get("mode") != "dynamic" or mount.get("length_unit") != "m"
            or mount.get("parent_frame") not in ("base_link", "base_footprint", "body")
            or mount.get("transform_direction") != "body_from_camera"
            or mount.get("child_frame") != vendor.get("frame_id")
            or not all(isinstance(value, str) and value for value in
                       (mount.get("topic"), mount.get("epoch"), mount.get("version"),
                        vendor.get("frame_id"), vendor.get("clock_id")))):
        return None
    return mount["topic"]


def capture_mount(vendor, sample):
    """Read a rigid body-from-camera transform for one identified measurement."""
    import numpy as np

    if not mount_topic(vendor):
        return None
    declared = vendor["body_mount"]
    if (sample.get("status") != "available" or sample.get("stamp_source") != "measurement"
            or sample.get("clock_id") != vendor["clock_id"]
            or any(sample.get(key) != declared[key] for key in
                   ("schema", "mode", "length_unit", "transform_direction", "parent_frame",
                    "child_frame", "epoch", "version"))):
        return None
    try:
        mat = np.asarray(sample.get("matrix"), dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if (mat.shape != (4, 4) or not np.isfinite(mat).all()
            or not np.allclose(mat[3], (0, 0, 0, 1), atol=1e-6)
            or not np.allclose(mat[:3, :3].T @ mat[:3, :3], np.eye(3), atol=1e-5)
            or not math.isclose(np.linalg.det(mat[:3, :3]), 1., abs_tol=1e-5)):
        return None
    return mat, declared.get("accuracy_validated") is True and sample.get("accuracy_validated") is True
