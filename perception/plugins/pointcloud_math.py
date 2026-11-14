#!/usr/bin/env python3
"""
plugins/pointcloud_math.py — pointcloud 插件的纯数学部分。

只依赖 numpy：不 import cv2 / rclpy，因此在本机（无 cv2）也能跑单元测试。
坐标约定：相机光学系 x 右 / y 下（图像行向下）/ z 前。
"""

from __future__ import annotations

import struct

import numpy as np


# pointcloud.js 的 MAX_POINTS / MAX_POINTS_PER_FRAME 上限
MAX_POINTS = 40000

def decimation_stride(w: int, h: int, max_points: int) -> int:
    """Bound depth-grid sampling; a single-column input uses a linear stride."""
    n = w * h
    if w == 1:  # 1-D：按点数抽稀，不是 sqrt；ceil(N/max(max_points,1))
        return max(1, -((-n) // max(max_points, 1)))
    return max(1, int((n / max(max_points, 1)) ** 0.5 + 0.999))


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


def pinhole_of(blob: dict, w: int, h: int) -> tuple:
    """从 blob 取 (fx, fy, cx, cy)；缺省按 hfov，再缺省 90°。"""
    fx = float(blob.get("fx") or 0.0)
    fy = float(blob.get("fy") or 0.0)
    if "hfov" in blob and fx == 0.0 and fy == 0.0:
        return intrinsics_from_hfov(w, h, float(blob["hfov"]))
    if fx == 0.0:
        fx, _, _, _ = intrinsics_from_hfov(w, h, 90.0)
    return fx, fy or fx, float(blob.get("cx", w / 2.0)), float(blob.get("cy", h / 2.0))
