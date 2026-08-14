"""ROS-independent goal validation against a Nav2 occupancy grid."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable


INSCRIBED_COST = 99


class CostmapError(ValueError):
    """Raised when a costmap cannot provide a trustworthy goal cell."""


class GoalCellRejected(CostmapError):
    """Carry the stable public error code for a rejected navigation goal."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CostmapSnapshot:
    frame_id: str
    stamp_ns: int
    received_monotonic: float
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: tuple[int, ...]

    @classmethod
    def from_values(
        cls,
        *,
        frame_id: str,
        stamp_ns: int,
        resolution: float,
        width: int,
        height: int,
        origin_x: float,
        origin_y: float,
        origin_yaw: float,
        data: Iterable[int],
        received_monotonic: float | None = None,
    ) -> "CostmapSnapshot":
        values = tuple(int(value) for value in data)
        if not frame_id:
            raise CostmapError("costmap frame_id is empty")
        if not math.isfinite(resolution) or resolution <= 0:
            raise CostmapError("costmap resolution must be positive and finite")
        if width <= 0 or height <= 0 or len(values) != width * height:
            raise CostmapError("costmap dimensions do not match data")
        if not all(
            math.isfinite(value)
            for value in (origin_x, origin_y, origin_yaw)
        ):
            raise CostmapError("costmap origin must be finite")
        return cls(
            frame_id=frame_id.strip("/"),
            stamp_ns=int(stamp_ns),
            received_monotonic=(
                time.monotonic()
                if received_monotonic is None
                else float(received_monotonic)
            ),
            resolution=float(resolution),
            width=int(width),
            height=int(height),
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            origin_yaw=float(origin_yaw),
            data=values,
        )

    def cost_at(self, x: float, y: float) -> tuple[int, int, int]:
        if not math.isfinite(x) or not math.isfinite(y):
            raise CostmapError("goal coordinates must be finite")
        dx = x - self.origin_x
        dy = y - self.origin_y
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        cell_x = math.floor(local_x / self.resolution)
        cell_y = math.floor(local_y / self.resolution)
        if not 0 <= cell_x < self.width or not 0 <= cell_y < self.height:
            raise CostmapError("goal lies outside the current global costmap")
        return self.data[cell_y * self.width + cell_x], cell_x, cell_y

    def diagnostics(self, *, now_monotonic: float | None = None) -> dict:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        unknown = sum(value < 0 for value in self.data)
        free = sum(value == 0 for value in self.data)
        inflated = sum(0 < value < INSCRIBED_COST for value in self.data)
        inscribed = sum(value == INSCRIBED_COST for value in self.data)
        lethal = sum(value > INSCRIBED_COST for value in self.data)
        known = len(self.data) - unknown
        return {
            "state": "ready",
            "frame_id": self.frame_id,
            "stamp_ns": self.stamp_ns,
            "receive_age_sec": round(max(0.0, now - self.received_monotonic), 3),
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "unknown_cells": unknown,
            "free_cells": free,
            "inflated_cells": inflated,
            "inscribed_cells": inscribed,
            "lethal_cells": lethal,
            "collision_cell_ratio": (
                0.0 if known == 0 else round((inscribed + lethal) / known, 6)
            ),
            "nonfree_cell_ratio": (
                0.0
                if known == 0
                else round((inflated + inscribed + lethal) / known, 6)
            ),
        }


def goal_cell_receipt(
    snapshot: CostmapSnapshot | None,
    *,
    x: float,
    y: float,
    expected_frame: str,
    max_receive_age_sec: float,
    now_monotonic: float | None = None,
) -> dict:
    if snapshot is None:
        raise CostmapError("global costmap has not been received")
    now = time.monotonic() if now_monotonic is None else float(now_monotonic)
    receive_age = now - snapshot.received_monotonic
    if receive_age < 0 or receive_age > max_receive_age_sec:
        raise CostmapError(
            f"global costmap receive age {receive_age:.3f}s is invalid"
        )
    if snapshot.frame_id != expected_frame.strip("/"):
        raise CostmapError(
            f"global costmap frame is {snapshot.frame_id}, expected {expected_frame}"
        )
    cost, cell_x, cell_y = snapshot.cost_at(x, y)
    return {
        "cost": cost,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "collision": cost >= INSCRIBED_COST,
        "unknown": cost < 0,
        "costmap_stamp_ns": snapshot.stamp_ns,
        "costmap_receive_age_sec": round(receive_age, 3),
    }


def validated_goal_cell_receipt(
    snapshot: CostmapSnapshot | None,
    *,
    x: float,
    y: float,
    expected_frame: str,
    max_receive_age_sec: float,
    now_monotonic: float | None = None,
) -> dict:
    try:
        receipt = goal_cell_receipt(
            snapshot,
            x=x,
            y=y,
            expected_frame=expected_frame,
            max_receive_age_sec=max_receive_age_sec,
            now_monotonic=now_monotonic,
        )
    except CostmapError as exc:
        code = (
            "goal_outside_costmap"
            if str(exc) == "goal lies outside the current global costmap"
            else "goal_costmap_unavailable"
        )
        raise GoalCellRejected(code, str(exc)) from exc
    if receipt["unknown"]:
        raise GoalCellRejected(
            "goal_cost_unknown",
            "goal cell is unknown in the current global costmap",
        )
    if receipt["collision"]:
        raise GoalCellRejected(
            "goal_in_collision",
            f"goal cell cost {receipt['cost']} is occupied or inscribed",
        )
    return receipt


__all__ = [
    "CostmapError",
    "CostmapSnapshot",
    "GoalCellRejected",
    "INSCRIBED_COST",
    "goal_cell_receipt",
    "validated_goal_cell_receipt",
]
