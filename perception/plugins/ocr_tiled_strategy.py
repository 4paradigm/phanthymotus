from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class LargeImageStrategyConfig:
    enabled: bool = False
    trigger_side: int = 2400
    decode_side: int = 3200
    decode_hard_limit: int = 4096
    tile_size: int = 1280
    overlap: int = 192
    max_tiles: int = 6
    global_pass: bool = True
    dedup_iou: float = 0.5
    dedup_text_similarity: float = 0.8

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object] | None
    ) -> LargeImageStrategyConfig:
        values = values or {}
        known_values = {
            name: values[name]
            for name in cls.__dataclass_fields__
            if name in values
        }
        config = cls(**known_values)
        config.validate()
        return config

    def validate(self) -> None:
        for name in (
            "trigger_side",
            "decode_side",
            "decode_hard_limit",
            "tile_size",
            "max_tiles",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must not be negative")
        if self.overlap >= self.tile_size:
            raise ValueError("overlap must be smaller than tile_size")
        if self.decode_side > self.decode_hard_limit:
            raise ValueError("decode_side must not exceed decode_hard_limit")
        if self.tile_size > self.decode_hard_limit:
            raise ValueError("tile_size must not exceed decode_hard_limit")
        for name in ("dedup_iou", "dedup_text_similarity"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class DecodePlan:
    flag: int
    factor: int
    reduced_size: tuple[int, int]
    target_size: tuple[int, int]


class AdaptiveTiledOCRStrategy:
    def __init__(
        self,
        config: LargeImageStrategyConfig | Mapping[str, object] | None,
        global_max_side: int,
    ):
        if isinstance(config, LargeImageStrategyConfig):
            self.config = config
            self.config.validate()
        else:
            self.config = LargeImageStrategyConfig.from_mapping(config)
        if global_max_side <= 0:
            raise ValueError("global_max_side must be positive")
        if global_max_side > self.config.decode_hard_limit:
            raise ValueError(
                "global_max_side must not exceed decode_hard_limit"
            )
        self.global_max_side = global_max_side

    def _plan_jpeg_decode(
        self, cv2, source_size: tuple[int, int]
    ) -> DecodePlan:
        width, height = source_size
        choices = (
            (1, cv2.IMREAD_COLOR),
            (2, cv2.IMREAD_REDUCED_COLOR_2),
            (4, cv2.IMREAD_REDUCED_COLOR_4),
            (8, cv2.IMREAD_REDUCED_COLOR_8),
        )
        for factor, flag in choices:
            reduced_size = (
                math.ceil(width / factor),
                math.ceil(height / factor),
            )
            if max(reduced_size) <= self.config.decode_hard_limit:
                break
        else:
            raise ValueError("JPEG dimensions exceed reduced decode hard limit")

        target_size = reduced_size
        longest_side = max(reduced_size)
        if longest_side > self.config.decode_side:
            scale = self.config.decode_side / longest_side
            target_size = (
                max(1, round(reduced_size[0] * scale)),
                max(1, round(reduced_size[1] * scale)),
            )
        return DecodePlan(flag, factor, reduced_size, target_size)
