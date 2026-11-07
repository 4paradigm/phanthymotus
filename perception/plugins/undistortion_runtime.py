"""Explicit, per-camera RGB rectification for a shared perception input.

The calibration must describe the actual input stream, including any vendor
crop. Never infer a distortion model or scale a calibration across crops.
"""
from __future__ import annotations

import copy
import math

import numpy as np


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
    if model not in ("plumb_bob", "rational_polynomial", "equidistant"):
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
