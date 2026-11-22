"""Capture provenance preserved by image processors."""


def source_header(header):
    stamp = getattr(header, "stamp", None)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    frame_id = getattr(header, "frame_id", None)
    if (type(sec) is not int or sec < 0 or type(nanosec) is not int
            or not 0 <= nanosec < 1_000_000_000 or not isinstance(frame_id, str)):
        return None
    return {"frame_id": frame_id, "stamp_ns": sec * 1_000_000_000 + nanosec}
