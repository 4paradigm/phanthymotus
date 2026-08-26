"""
utils/qos.py — Shared ROS2 QoS profiles for perception plugins.

CAMERA_QOS is the platform convention for camera CompressedImage
subscriptions. Robot camera publishers offer BEST_EFFORT; a RELIABLE reader
does not match a BEST_EFFORT writer, so a plugin using RELIABLE would "start"
successfully and never receive a frame (vop, OCR and obstacle all hit this).
BEST_EFFORT readers match both offer kinds. depth=1 because vision plugins
process the latest frame only (see utils.latest_frame) — the DDS history is
not a work queue.
"""

from __future__ import annotations

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

__all__ = ["CAMERA_QOS"]
