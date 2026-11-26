"""Capture provenance preserved by image processors."""

import re

CAPTURE_PREFIX = "motus.capture/1|"
CAPTURE_FRAME = re.compile(
    r"motus\.capture/1\|([A-Za-z][A-Za-z0-9_]{0,47})\|"
    r"([A-Za-z][A-Za-z0-9_]{0,79})\|([0-9a-f]{32})\|([1-9][0-9]{0,15})\|"
    r"(host_read|hardware)\|([0-9a-f]{32})"
)


def capture_identity(frame_id):
    if not isinstance(frame_id, str) or not frame_id.startswith(CAPTURE_PREFIX):
        return None
    match = CAPTURE_FRAME.fullmatch(frame_id)
    if not match:
        return {"frame_valid": False}
    camera, optical, session, sequence, clock, device = match.groups()
    return dict(camera_id=camera, optical_frame_id=optical, capture_session_id=session,
                frame_sequence=int(sequence), timestamp_source=clock, device_id=device,
                frame_valid=True)


def is_invalid_capture(header):
    identity = capture_identity(getattr(header, "frame_id", None))
    return identity is not None and not identity["frame_valid"]


def source_header(header):
    stamp = getattr(header, "stamp", None)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    frame_id = getattr(header, "frame_id", None)
    if (type(sec) is not int or sec < 0 or type(nanosec) is not int
            or not 0 <= nanosec < 1_000_000_000 or not isinstance(frame_id, str)):
        return None
    return {"frame_id": frame_id, "stamp_ns": sec * 1_000_000_000 + nanosec,
            **(capture_identity(frame_id) or {})}
