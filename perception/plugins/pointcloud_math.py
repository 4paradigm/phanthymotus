#!/usr/bin/env python3
"""
plugins/pointcloud_math.py — pointcloud 插件的纯数学部分。

只依赖 numpy：不 import cv2 / rclpy，因此在本机（无 cv2）也能跑单元测试。
坐标约定：相机光学系 x 右 / y 下（图像行向下）/ z 前。
"""

from __future__ import annotations

import json
import logging
import os
import struct
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# pointcloud.js 的 MAX_POINTS / MAX_POINTS_PER_FRAME 上限
MAX_POINTS = 40000

# 标定文件落盘位置（/models 是容器唯一可持久化挂载）
CALIB_DIR = "/models/stereo_calib"

# 已校正双目的判定阈值：对应角点 Δy 中位数小于此值视为已 rectify
RECTIFIED_DY_PX = 1.5


def decimation_stride(w: int, h: int, max_points: int) -> int:
    """q5 参照：stride = ceil(sqrt(w*h/max_points))，至少 1。"""
    return max(1, int(((w * h) / max(max_points, 1)) ** 0.5 + 0.999))


def backproject_depth(depth_m: np.ndarray, fx: float, fy: float,
                      cx: float, cy: float, stride: int,
                      min_depth_m: float, max_depth_m: float) -> np.ndarray:
    """深度图 → 相机光学系点云 (N, 3)，列序 (x右, y下, z前)。

    同 q5 CameraPointCloudPlugin 的反投影：跳步抽样 + 有效值掩码。
    """
    z = depth_m[::stride, ::stride]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)
    rows, cols = np.indices(z.shape, dtype=np.float32)
    rows *= stride
    cols *= stride
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    xyz = np.stack((x, y, z), axis=-1)
    return np.ascontiguousarray(xyz[valid], dtype=np.float32)


def make_Q(fx: float, cx: float, cy: float, Tx: float) -> np.ndarray:
    """OpenCV stereo-rectify 标准形式，同 Go1 pointcloud_stream.cc 的 make_Q。

    Q = [[1,0,0,-cx],
         [0,1,0,-cy],
         [0,0,0, fx],
         [0,0,-1/Tx,0]]

    Tx 是 stereoCalibrate 的符号化 T[0]（先左后右的标准双目为负）。
    """
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    Q[0, 3] = -cx
    Q[1, 3] = -cy
    Q[2, 3] = fx
    Q[3, 2] = -1.0 / Tx
    return Q


def Q_from_baseline(fx: float, cx: float, cy: float, baseline_m: float) -> np.ndarray:
    """make_Q 的取绝对值版：Tx 只表达基线长度，不问符号。

    本插件自己用 cv2.SGBM 算视差（d ≥ 0），因此 W = Q[3,2]·d 必须为正、
    即 Q[3,2] = -1/Tx > 0、Tx < 0。无论标定 blob 里 Tx 是 OpenCV 的负值
    还是 Unitree 出厂标定的正值，统一归一到负号，深度恒为正。
    """
    return make_Q(fx, cx, cy, -abs(float(baseline_m)))


def filter_cloud(xyz: np.ndarray, min_depth_m: float,
                 max_depth_m: float) -> np.ndarray:
    """Go1 参照：剔除非有限值，再按 z（光学系前向）截断。"""
    if xyz.size == 0:
        return np.ascontiguousarray(xyz, dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=-1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return np.ascontiguousarray(xyz, dtype=np.float32)
    ok = (xyz[:, 2] >= min_depth_m) & (xyz[:, 2] <= max_depth_m)
    return np.ascontiguousarray(xyz[ok], dtype=np.float32)


def decimate_cloud(xyz: np.ndarray, stride: int) -> np.ndarray:
    """对 (N,3) 均匀抽稀（视差反投影后点数仍可能超限）。"""
    if stride <= 1 or xyz.shape[0] == 0:
        return np.ascontiguousarray(xyz, dtype=np.float32)
    return np.ascontiguousarray(xyz[::stride], dtype=np.float32)


def encode_packet(xyz: np.ndarray) -> bytes:
    """点云 → pointcloud.js / ros2_bridge 的二进制协议。

    struct.pack("<II", 12, N) + N × float32 小端。

    轴序在此统一：入参是光学系 (x右, y下, z前)，出包 (p_x,p_y,p_z) =
    (−z, x, y_down)。渲染器默认映射 display = (p_y, −p_z, −p_x)，代入后
    display = (x_cam, −y_down, z_cam) —— 即 x 右 / y 上 / z 前，不镜像。
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected (N, 3) cloud, got {xyz.shape}")
    x, y_down, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    packet_xyz = np.stack((-z, x, y_down), axis=-1).astype("<f4")
    return struct.pack("<II", 12, len(packet_xyz)) + packet_xyz.tobytes()


def summarize_cloud(xyz: np.ndarray, min_depth_m: float,
                    max_depth_m: float) -> dict:
    """摘要：点数、范围、三区最近（左/中/右按 x 三等分）。"""
    if xyz.shape[0] == 0:
        return {"points": 0, "nearest": None, "farthest": None, "average": None,
                "nearest_by_region": {"left": None, "center": None, "right": None},
                "z_range_m": [min_depth_m, max_depth_m]}
    z = xyz[:, 2]
    span = max(float(np.ptp(xyz[:, 0])), 1e-6)
    third = np.clip(np.floor((xyz[:, 0] - xyz[:, 0].min()) / span * 3).astype(int), 0, 2)
    regions = {"left": None, "center": None, "right": None}
    for i, name in enumerate(("left", "center", "right")):
        sel = z[third == i]
        if sel.size:
            regions[name] = round(float(np.percentile(sel, 5)), 3)
    return {
        "points": int(xyz.shape[0]),
        "nearest": round(float(z.min()), 3),
        "farthest": round(float(z.max()), 3),
        "average": round(float(z.mean()), 3),
        "nearest_by_region": regions,
        "z_range_m": [min_depth_m, max_depth_m],
    }


def intrinsics_from_hfov(w: int, h: int, hfov_deg: float) -> tuple:
    """hfov（度）→ (fx=fy, cx, cy)，画幅 w×h 中心。零配置兜底。"""
    fx = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return float(fx), float(fx), float(w) / 2.0, float(h) / 2.0


def load_calibration_blob(value) -> Optional[dict]:
    """实例配置里的 calibration：dict、JSON 字符串，或 /models 下文件路径。"""
    if not value:
        return None
    if isinstance(value, dict):
        return value
    text = str(value)
    try:
        return json.loads(text)
    except Exception:
        pass
    if os.path.isfile(text):
        try:
            with open(text, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as error:
            log.error(f"[pointcloud] cannot read calibration file {text}: {error}")
    return None


def calibration_tier(blob: dict) -> str:
    """标定 blob 分档：
    raw       —— 带 K1/K2/D1/D2 的完整 stereoCalibrate 结果，运行时先 rectify；
    rectified —— 只有 {fx, cx, cy, Tx}（左右已校正），直接 Q 反投影；
    pinhole   —— {fx, fy, cx, cy} / {hfov}，A/B 模式的内参。
    """
    if any(k in blob for k in ("K1", "K2", "D1", "D2")):
        return "raw"
    if "Tx" in blob:
        return "rectified"
    if "hfov" in blob or any(k in blob for k in ("fx", "fy", "cx", "cy")):
        return "pinhole"
    return "unknown"


def pinhole_of(blob: dict, w: int, h: int) -> tuple:
    """从 blob 取 (fx, fy, cx, cy)；缺省按 hfov，再缺省 90°。"""
    fx = float(blob.get("fx") or 0.0)
    fy = float(blob.get("fy") or 0.0)
    if "hfov" in blob and fx == 0.0 and fy == 0.0:
        return intrinsics_from_hfov(w, h, float(blob["hfov"]))
    if fx == 0.0:
        fx, _, _, _ = intrinsics_from_hfov(w, h, 90.0)
    return fx, fy or fx, float(blob.get("cx", w / 2.0)), float(blob.get("cy", h / 2.0))
