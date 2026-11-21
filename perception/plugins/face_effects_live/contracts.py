from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
import math
import time
import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp: float  # source seconds; never subtract from a monotonic clock
    frame_id: int
    received: float = field(default_factory=time.monotonic)
    color: str = "bgr8"
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.color != "bgr8":
            raise ValueError("Frame requires bgr8; convert in the input adapter")
        if (self.image.dtype != np.uint8 or self.image.ndim != 3
                or self.image.shape[2] != 3 or min(self.image.shape[:2]) < 2
                or self.image.shape[0] * self.image.shape[1] > 16777216):
            raise ValueError("expected uint8 HWC BGR, 2..16M pixels")
        if not math.isfinite(self.timestamp) or not math.isfinite(self.received):
            raise ValueError("non-finite timestamp")

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]


@dataclass
class Face:
    box: np.ndarray  # x,y,width,height in input pixels
    points: np.ndarray  # screen-left eye, screen-right eye, nose, mouth corners
    confidence: float
    landmarks: np.ndarray | None = None
    quality: float = 1.0
    topology: str = "osf66"
    appearance: np.ndarray | None = None
    landmark_source: str = ""

    def valid(self):
        return (self.box.shape == (4,) and self.points.shape == (5, 2)
                and np.isfinite(self.box).all() and np.isfinite(self.points).all()
                and min(self.box[2:]) >= 4 and math.isfinite(self.confidence)
                and 0 <= self.confidence <= 1)


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Face]: ...
    def close(self): ...


class Landmarker(Protocol):
    def refine(self, image: np.ndarray, faces: list[Face]) -> list[Face]: ...
    def close(self): ...


class Source(Protocol):
    def read(self) -> Frame | None: ...
    def close(self): ...


class Renderer(Protocol):
    def render(self, image, commands): ...
    def close(self): ...
