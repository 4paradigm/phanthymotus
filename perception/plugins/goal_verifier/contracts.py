"""Versioned PiPER visual evidence contract. All coordinates are explicit."""

from copy import deepcopy
import hashlib
import json
import math
import re

NAME = "piper_object_goal_verifier"
OBSERVATION = "motus.piper-object-observation/1"
RESULT = "motus.piper-object-goal-result/1"


class Refusal(ValueError):
    def __init__(self, code, message=None):
        self.code = code
        super().__init__(message or code.replace("_", " ").lower())


def need(condition, code, message=None):
    if not condition:
        raise Refusal(code, message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise Refusal("INVALID_JSON", "输入必须是有限数值的 JSON 数据") from exc


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value):
    need(type(value) in (int, float) and math.isfinite(value), "INVALID_NUMBER")
    return float(value)


def text(value):
    need(
        isinstance(value, str) and 0 < len(value) <= 256 and value.strip() == value,
        "INVALID_IDENTIFIER",
    )
    return value


def vector(value, size):
    need(isinstance(value, list) and len(value) == size, "INVALID_VECTOR")
    return [number(v) for v in value]


def fields(value, required, optional=()):
    need(isinstance(value, dict), "INVALID_OBJECT")
    need(
        not set(required) - value.keys(),
        "MISSING_FIELD",
        "缺少字段: " + str(sorted(set(required) - value.keys())),
    )
    need(not value.keys() - set(required) - set(optional), "UNKNOWN_FIELD")


def observation(raw, binding, now, max_age):
    fields(
        raw,
        (
            "schema",
            "observation_id",
            "source_id",
            "source_epoch",
            "sequence",
            "camera_id",
            "frame_id",
            "view_id",
            "image_size",
            "distortion",
            "captured_at",
            "timestamp_basis",
            "provenance",
            "objects",
        ),
        ("calibration_id", "track_epoch", "image_sha256"),
    )
    o = deepcopy(raw)
    canonical(o)
    need(o["schema"] == OBSERVATION, "OBSERVATION_SCHEMA")
    for key in (
        "observation_id",
        "source_id",
        "source_epoch",
        "camera_id",
        "frame_id",
        "view_id",
    ):
        text(o[key])
    need(o["timestamp_basis"] == "capture", "CAPTURE_TIME_UNPROVEN")
    need(o["provenance"] in ("real", "synthetic", "replay"), "PROVENANCE_REQUIRED")
    for key in ("calibration_id", "track_epoch", "image_sha256"):
        if key in o:
            text(o[key])
    if "image_sha256" in o:
        need(
            re.fullmatch("[0-9a-f]{64}", o["image_sha256"]) is not None,
            "INVALID_IMAGE_HASH",
        )
    need(
        o["provenance"] == "real" or binding.get("allow_test_data") is True,
        "TEST_DATA_REFUSED",
    )
    for key in ("source_id", "camera_id", "frame_id", "distortion"):
        need(
            o[key] == binding[key],
            "SOURCE_BINDING_MISMATCH",
            "观测与配置不匹配: " + key,
        )
    need(o["distortion"] in ("raw", "rectified"), "DISTORTION_REQUIRED")
    need(type(o["sequence"]) is int and o["sequence"] >= 0, "INVALID_SEQUENCE")
    size = o["image_size"]
    need(
        isinstance(size, list)
        and len(size) == 2
        and all(type(v) is int and 1 <= v <= 8192 for v in size),
        "INVALID_IMAGE_SIZE",
    )
    need(
        0 <= now - number(o["captured_at"]) <= max_age,
        "STALE_OR_FUTURE",
        "采集时间过期、未来或时钟不一致",
    )
    need(
        isinstance(o["objects"], list) and len(o["objects"]) <= 100,
        "INVALID_DETECTIONS",
    )
    ids = set()
    tracks = set()
    for obj in o["objects"]:
        fields(
            obj,
            ("detection_id", "class_name", "confidence", "bbox"),
            ("color", "track_id", "confidence_kind"),
        )
        need(text(obj["detection_id"]) not in ids, "DUPLICATE_DETECTION")
        ids.add(obj["detection_id"])
        text(obj["class_name"])
        need(0 <= number(obj["confidence"]) <= 1, "INVALID_CONFIDENCE")
        x1, y1, x2, y2 = vector(obj["bbox"], 4)
        need(0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1], "INVALID_BBOX")
        if "color" in obj:
            text(obj["color"])
        if "confidence_kind" in obj:
            text(obj["confidence_kind"])
        if "track_id" in obj:
            need(text(obj["track_id"]) not in tracks, "DUPLICATE_TRACK")
            tracks.add(obj["track_id"])
    return o


def goal(raw):
    fields(
        raw, ("units", "frame_id", "shape", "tolerance"), ("center", "radius", "bounds")
    )
    g = deepcopy(raw)
    canonical(g)
    text(g["frame_id"])
    need(g["units"] in ("pixels", "normalized", "plane_mm"), "GOAL_UNITS")
    need(number(g["tolerance"]) >= 0, "INVALID_TOLERANCE")
    if g["shape"] == "circle":
        fields(g, ("units", "frame_id", "shape", "tolerance", "center", "radius"))
        vector(g["center"], 2)
        need(number(g["radius"]) > 0, "INVALID_REGION")
    elif g["shape"] == "rectangle":
        fields(g, ("units", "frame_id", "shape", "tolerance", "bounds"))
        a, b, c, d = vector(g["bounds"], 4)
        need(a < c and b < d, "INVALID_REGION")
    else:
        raise Refusal("INVALID_REGION")
    coordinates = g["center"] if g["shape"] == "circle" else g["bounds"]
    if g["units"] == "normalized":
        need(all(0 <= v <= 1 for v in coordinates), "GOAL_OUTSIDE_IMAGE")
    if g["units"] == "pixels":
        need(all(v >= 0 for v in coordinates), "GOAL_OUTSIDE_IMAGE")
    return g


def center(obj):
    # Reuse official measure's bbox normalisation; strict bounds were checked
    # above, so its clamping/degradation path never repairs invalid input.
    return [
        (obj["bbox"][0] + obj["bbox"][2]) / 2,
        (obj["bbox"][1] + obj["bbox"][3]) / 2,
    ]


def error_result(exc):
    return dict(
        schema=RESULT,
        state="error",
        result="INCONCLUSIVE",
        data_valid=False,
        code=getattr(exc, "code", "INPUT_ERROR"),
        message=str(exc),
    )
