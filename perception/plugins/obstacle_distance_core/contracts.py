from __future__ import annotations

from enum import Enum
from typing import Protocol


class ErrorCode(str, Enum):
    INVALID_IMAGE = "invalid_image"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    INVALID_DEPTH = "invalid_depth"
    NO_VALID_DEPTH = "no_valid_depth"


class ObstacleDistanceError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class IndoorDistanceBackend(Protocol):
    def predict_indoor_distance(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> float:
        ...
