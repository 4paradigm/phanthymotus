"""`motus.camera/1` for the browser camera behind /remote_control/camera.

Separate from `start.py` because `start.py` mounts ./web at import time and so
cannot be imported by a test. Everything here is pure except one module-level
fact — the frame shape the browser settled on — which is kept beside the code
that reads it rather than in the websocket handler.

Spec: `phanthymotus-driver/README_dev.md` § Camera Parameters.
"""
from __future__ import annotations

import math

# None until the first frame: before that there is nothing truthful to declare.
_frame_size: "tuple[int, int] | None" = None

TOPIC = "/remote_control/camera"
CAMERA_ID = "agentcore/remote_camera"


def jpeg_size(data: bytes) -> "tuple[int, int] | None":
    """(width, height) from a JPEG's SOF marker, without decoding it.

    The browser is the only one who knows what resolution its camera settled
    on: the card's `width` is a *request* for the long edge and the aspect
    ratio comes from the device, so neither field here is the frame's real
    shape. The bytes are, and the SOF header is a few hundred bytes in — far
    cheaper than decoding, and agent-core has no image library to decode with.
    """
    if not data or data[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # SOF0..SOF15, excluding DHT(c4) DAC(c8) and the RSTn/SOI/EOI range.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return (width, height) if width > 0 and height > 0 else None
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = int.from_bytes(data[i + 2:i + 4], "big")
        if seg < 2:
            return None
        i += 2 + seg
    return None


def declaration(cfg: dict) -> list:
    """`motus.camera/1` for /remote_control/camera, or `[]`.

    **Only if somebody typed a field of view in.** A browser exposes no focal
    length — `MediaTrackSettings` has zoom and focus distance, neither of which
    is one — so the only honest K here is derived from a number a human
    supplied. Rule 1 of the spec is that an unknown quantity is `null` and
    never a guess, and the reason is sharpest in this exact case: DepthART
    computes its metres *from* K, so a default "typical webcam 60°" would not
    make the card fail, it would make it publish confidently wrong distances on
    a topic whose whole contract is that the number is in metres.

    So with the field empty this returns nothing and a DepthART consumer
    refuses to start, saying why. That is the intended outcome, not a gap.

    What a typed FOV buys is a *nominal* K: a phone's effective field of view
    moves with digital zoom, EIS crop and whatever resolution the browser
    picked, so the scale will be off by the focal-length error. `visual_depth`'s
    `calibrate` action exists to absorb exactly that — measure one known
    distance and the log-affine fit takes out the residual. Hence
    `source: "manual"`: the consumer can see this was asserted, not measured.
    """
    fov_deg = (cfg or {}).get("camera_fov_deg")
    if fov_deg in (None, ""):
        return []
    try:
        fov_deg = float(fov_deg)
    except (TypeError, ValueError):
        return []
    if not 1.0 <= fov_deg < 180.0:
        return []
    size = _frame_size
    if not size:
        return []
    width, height = size
    half = math.radians(fov_deg) / 2.0
    # Square pixels: the browser scales a frame, it does not stretch it, so fy
    # equals fx. Said out loud because the spec warns that the vertical angle
    # is NOT derivable from the aspect ratio once something stretches a frame —
    # if that ever stops holding here, this is the line that is wrong.
    fx = (width / 2.0) / math.tan(half)
    cx, cy = width / 2.0 - 0.5, height / 2.0 - 0.5
    return [{
        "schema": "motus.camera/1",
        "topic": TOPIC,
        "format": "image/jpeg",
        # Stable across reconnects and distinct from any on-robot lens, so a
        # downstream calibration table keyed by `id` does not confuse the two.
        "id": CAMERA_ID,
        "width": width, "height": height,
        "distortion_model": "unknown",
        "D": None,
        "K": [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0],
        "half_fov_rad": round(half, 6),
        "half_fov_v_rad": None,
        "source": "manual",
        "pipeline": [CAMERA_ID],
        "vendor": {"note": f"K derived from a hand-entered {fov_deg:g} deg "
                           f"horizontal FOV at {width}x{height}; nominal, not "
                           f"calibrated — run visual_depth's calibrate to fix "
                           f"the scale."},
    }]




def frame_size() -> "tuple[int, int] | None":
    """The last frame shape read off the wire, or None."""
    return _frame_size


def note_frame(data: bytes) -> None:
    """Record this frame's shape, keeping the previous one if it is unreadable."""
    global _frame_size
    _frame_size = jpeg_size(data) or _frame_size
