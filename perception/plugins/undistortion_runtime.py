"""Explicit, per-camera RGB rectification for a shared perception input.

The calibration must describe the actual input stream, including any vendor
crop. Never infer a distortion model or scale a calibration across crops.

Where a calibration comes from, in order (see `resolve`):

1. typed into the card (`calibration` is a full object);
2. declared by the upstream camera (`motus.camera/1` with a supported
   distortion model and K/D) — the camera owns its optics, so a driver that
   learns to declare them takes over from the table below;
3. `CALIBRATIONS`, keyed on the camera id the upstream declares.

Nothing found is an error naming the camera, never a guess.
"""
from __future__ import annotations

import copy
import math

import numpy as np

MODELS = ("plumb_bob", "rational_polynomial", "equidistant")

# Chessboard calibrations measured on real lenses, keyed on `motus.camera/1` id.
#
# **Keyed on the id, and the id is per model, not per unit.** The driver spec
# requires `<vendor>/<model>/<port>` to be identical on every unit of a model,
# so a row measured on one robot is applied to every robot of that model. Lens
# to lens variation within a model is usually small; `measured_on` says which
# unit and when, and a robot that rectifies badly gets its own measurement
# typed into the card, which wins over this table.
#
# Each row is exactly what `perception/tools/calibrate_chessboard.py` writes to
# `table_entry.json`: K/D at the resolution the camera publishes (here the S10
# driver's 640x360), plus where the numbers came from. Every row is validated
# by the tests, so a typo fails there rather than on a robot.
CALIBRATIONS = {
    "deeprobotics/s10_pro/camera_front": {
        "id": "deeprobotics/s10_pro/camera_front",
        "width": 640,
        "height": 360,
        "distortion_model": "plumb_bob",
        "K": [314.963248, 0, 336.650604, 0, 314.934089, 182.285271, 0, 0, 1],
        "D": [-0.264916006540722, 0.08280010720226033, -0.00018047856465669117,
              3.14558084286742e-05, -0.013244463692919645],
        "camera": "DeepRobotics S10 Pro front wide (Sunplus SPCA2650 UVC, RTSP /video4)",
        "measured_on": "2026-11-07, S10 Pro CD1 (host CA9B_NO.0), standing",
        "method": "chessboard 10x13 inner corners, 18 mm squares; perception/tools/calibrate_chessboard.py",
        "source_resolution": {"width": 1024, "height": 576},
        "frames_used": 103,
        "rms_px": 0.213,
        "note": ("Calibrated on the 1024x576 RTSP stream and scaled to the driver's "
                 "640x360 (K x 0.625, D unchanged). Horizontal FOV 90.8 deg after "
                 "rectification; no invalid border at this K."),
    },
    "deeprobotics/s10_pro/camera_rear": {
        "id": "deeprobotics/s10_pro/camera_rear",
        "width": 640,
        "height": 360,
        "distortion_model": "plumb_bob",
        "K": [312.750288, 0, 308.919092, 0, 312.683048, 183.364774, 0, 0, 1],
        "D": [-0.2622894409938081, 0.08153733337528488, 0.00012089478431258563,
              0.00040150868161954143, -0.013044993729573065],
        "camera": "DeepRobotics S10 Pro rear wide (Sunplus SPCA2650 UVC, RTSP /video0)",
        "measured_on": "2026-11-08, S10 Pro CD1 (host CA9B_NO.0), standing",
        "method": "chessboard 10x13 inner corners, 18 mm squares; perception/tools/calibrate_chessboard.py",
        "source_resolution": {"width": 1024, "height": 576},
        "frames_used": 103,
        "rms_px": 0.208,
        "note": ("Calibrated on the 1024x576 RTSP stream and scaled to the driver's "
                 "640x360 (K x 0.625, D unchanged). Horizontal FOV 91.3 deg after "
                 "rectification; no invalid border at this K."),
    },
    "deeprobotics/s10_pro/camera_left": {
        "id": "deeprobotics/s10_pro/camera_left",
        "width": 640,
        "height": 360,
        "distortion_model": "plumb_bob",
        "K": [314.208908, 0, 311.686392, 0, 314.175661, 192.549502, 0, 0, 1],
        "D": [-0.25929876926166817, 0.07786005715673677, 0.00035419731549147994,
              0.0005598844289782525, -0.011875881953120242],
        "camera": "DeepRobotics S10 Pro left wide (Sunplus SPCA2650 UVC, RTSP /video2)",
        "measured_on": "2026-11-08, S10 Pro CD1 (host CA9B_NO.0), standing",
        "method": "chessboard 10x13 inner corners, 18 mm squares; perception/tools/calibrate_chessboard.py",
        "source_resolution": {"width": 1024, "height": 576},
        "frames_used": 106,
        "rms_px": 0.222,
        "note": ("Calibrated on the 1024x576 RTSP stream and scaled to the driver's "
                 "640x360 (K x 0.625, D unchanged). Horizontal FOV 91.0 deg after "
                 "rectification; no invalid border at this K."),
    },
    "deeprobotics/s10_pro/camera_right": {
        "id": "deeprobotics/s10_pro/camera_right",
        "width": 640,
        "height": 360,
        "distortion_model": "plumb_bob",
        "K": [314.351144, 0, 327.541354, 0, 314.28902, 177.870404, 0, 0, 1],
        "D": [-0.2617530191348165, 0.0801398926074851, 9.564678897937886e-05,
              -0.0001381378356953391, -0.012449424682861728],
        "camera": "DeepRobotics S10 Pro right wide (Sunplus SPCA2650 UVC, RTSP /video6)",
        "measured_on": "2026-11-08, S10 Pro CD1 (host CA9B_NO.0), standing",
        "method": "chessboard 10x13 inner corners, 18 mm squares; perception/tools/calibrate_chessboard.py",
        "source_resolution": {"width": 1024, "height": 576},
        "frames_used": 98,
        "rms_px": 0.219,
        "note": ("Calibrated on the 1024x576 RTSP stream and scaled to the driver's "
                 "640x360 (K x 0.625, D unchanged). Horizontal FOV 91.0 deg after "
                 "rectification; no invalid border at this K."),
    },
}


def lookup(camera_id):
    """The table row for this camera id (a deep copy), or None."""
    row = CALIBRATIONS.get(camera_id or "")
    return copy.deepcopy(row) if row is not None else None


def _upstream_calibration(upstream):
    """A calibration the upstream camera declares itself, or None.

    Only a declaration naming a model this card rectifies counts. `unknown`,
    `pinhole` or a missing model say nothing about distortion, so the table is
    consulted instead. A declaration that *does* claim a supported model owns
    its numbers: they are validated and an error is raised rather than quietly
    replaced by the table.
    """
    if (upstream or {}).get("distortion_model") not in MODELS:
        return None
    return validate({key: upstream.get(key) for key in
                     ("id", "width", "height", "K", "D", "distortion_model")})


def resolve(explicit, upstream):
    """`(calibration, source)` for a card: typed in, upstream, table, or an error.

    `source` is `manual`, `upstream` or `table:<id>` and is reported by the
    card, so a reader can tell which kind of number is rectifying the picture.
    """
    cal = validate(explicit)
    if cal is not None:
        return cal, "manual"
    upstream = upstream or {}
    cal = _upstream_calibration(upstream)
    if cal is not None:
        return cal, "upstream"
    camera_id = upstream.get("id") or ""
    row = lookup(camera_id)
    if row is not None:
        return validate(row), f"table:{camera_id}"
    who = f"相机 {camera_id}" if camera_id else "上游相机（没有声明 id）"
    raise ValueError(
        f"{who} 没有可用的去畸变参数：上游声明里没有 K/D，内置标定表里也没有这台相机。"
        "用 perception/tools/calibrate_chessboard.py 标定，把输出的 table_entry.json "
        "加入 undistortion_runtime.CALIBRATIONS；或在卡片的 Camera calibration 里手填。")


def validate(config):
    if config in (None, {}):
        return None
    if not isinstance(config, dict):
        raise ValueError("undistortion must be a calibration object or {} to disable")
    result = copy.deepcopy(config)
    if not isinstance(result.get("id"), str) or not result["id"].strip():
        raise ValueError("undistortion requires the camera id")
    for key in ("width", "height"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"undistortion {key} must be a positive integer")
    model = result.get("distortion_model")
    if model not in MODELS:
        raise ValueError("undistortion model must be plumb_bob, rational_polynomial or equidistant")
    for key in ("K", "D"):
        values = result.get(key)
        if not isinstance(values, list) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in values):
            raise ValueError(f"undistortion {key} must contain finite numbers")
    if len(result["K"]) != 9:
        raise ValueError("undistortion K requires nine row-major values")
    k = result["K"]
    if (k[0] <= 0 or k[4] <= 0 or k[1] != 0 or k[3] != 0
            or k[6:] != [0, 0, 1]
            or not 0 <= k[2] < result["width"] or not 0 <= k[5] < result["height"]):
        raise ValueError("undistortion K requires positive focal lengths, an in-frame principal point and zero skew")
    lengths = {"plumb_bob": (4, 5), "rational_polynomial": (8,), "equidistant": (4,)}
    if len(result["D"]) not in lengths[model]:
        raise ValueError("undistortion D length does not match distortion_model")
    return result


def declaration(upstream, config):
    cal = validate(config)
    if cal is None:
        return copy.deepcopy(upstream or {})
    upstream = upstream or {}
    if upstream.get("id") and upstream["id"] != cal["id"]:
        raise ValueError("undistortion calibration belongs to a different camera")
    for axis in ("width", "height"):
        if upstream.get(axis) is not None and upstream[axis] != cal[axis]:
            raise ValueError("undistortion calibration resolution differs from the input stream")
    out = copy.deepcopy(upstream)
    out.update({key: cal[key] for key in ("id", "width", "height", "K")})
    out.update(schema="motus.camera/1", distortion_model="plumb_bob",
               D=[0.0] * 5, source="manual")
    # Keep the original K and canvas size. No implicit optimal-matrix crop.
    k = cal["K"]
    out["half_fov_rad"] = (math.atan(k[2] / k[0])
                           + math.atan((cal["width"] - k[2]) / k[0])) / 2
    out["half_fov_v_rad"] = (math.atan(k[5] / k[4])
                             + math.atan((cal["height"] - k[5]) / k[4])) / 2
    out["pipeline"] = list(out.get("pipeline") or []) + ["perception/undistort"]
    vendor = out.setdefault("vendor", {})
    vendor["undistortion"] = {"input_calibration": cal, "output_matrix": "same_K",
                              "invalid_border": "cropped"}
    return out


def _valid_rectangle(mask):
    """Largest all-valid rectangle; calculated once with the remap tables."""
    heights = [0] * mask.shape[1]
    best = (0, 0, 0, 0)
    area = 0
    for y, row in enumerate(mask):
        heights = [h + 1 if valid else 0 for h, valid in zip(heights, row)]
        stack = []
        for x, h in enumerate(heights + [0]):
            start = x
            while stack and stack[-1][1] > h:
                left, height = stack.pop()
                candidate = (x - left) * height
                if candidate > area:
                    area = candidate
                    best = (left, y - height + 1, x - left, height)
                start = left
            if not stack or stack[-1][1] < h:
                stack.append((start, h))
    if area == 0:
        raise ValueError("Calibration produces no valid rectified image")
    return best


class Rectifier:
    """Cached image remap shared by every downstream perception branch."""

    def __init__(self, config):
        self.cal = validate(config)
        if self.cal is None:
            raise ValueError("An exact camera calibration is required")
        self._maps = None
        self._prepare()

    def _prepare(self):
        import cv2
        cal = self.cal
        size = (cal["width"], cal["height"])
        if self._maps is None:
            k = np.asarray(cal["K"], np.float64).reshape(3, 3)
            d = np.asarray(cal["D"], np.float64)
            init = (cv2.fisheye.initUndistortRectifyMap if cal["distortion_model"] == "equidistant"
                    else cv2.initUndistortRectifyMap)
            mx, my = init(k, d, np.eye(3), k, size, cv2.CV_32FC1)
            # Linear interpolation needs all four source neighbours.
            valid = (np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (my >= 0)
                     & (mx < size[0] - 1) & (my < size[1] - 1))
            self._maps = mx, my, valid
            self.roi = _valid_rectangle(valid)

    def camera_info(self, upstream):
        out = declaration(upstream, self.cal)
        x, y, width, height = self.roi
        k = list(out["K"])
        k[2] -= x
        k[5] -= y
        out.update(width=width, height=height, K=k)
        out["half_fov_rad"] = (math.atan(k[2] / k[0]) + math.atan((width - k[2]) / k[0])) / 2
        out["half_fov_v_rad"] = (math.atan(k[5] / k[4]) + math.atan((height - k[5]) / k[4])) / 2
        out["vendor"]["undistortion"].update(output_matrix="same_K_then_crop",
                                               crop_xywh=list(self.roi))
        return out

    def apply(self, frame):
        import cv2
        if (frame.shape[1], frame.shape[0]) != (self.cal["width"], self.cal["height"]):
            raise ValueError("undistortion input size differs from the calibrated stream")
        mx, my, valid = self._maps
        rectified = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        x, y, width, height = self.roi
        return rectified[y:y + height, x:x + width].copy()
