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


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    edge = length - tile_size
    if starts[-1] != edge:
        starts.append(edge)
    return starts


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

    def _select_tiles(
        self, image_size: tuple[int, int]
    ) -> list[tuple[int, int, int, int]]:
        width, height = image_size
        stride = self.config.tile_size - self.config.overlap
        xs = _axis_starts(width, self.config.tile_size, stride)
        ys = _axis_starts(height, self.config.tile_size, stride)
        grid = [
            (
                x,
                y,
                min(x + self.config.tile_size, width),
                min(y + self.config.tile_size, height),
            )
            for y in ys
            for x in xs
        ]
        if len(grid) <= self.config.max_tiles:
            return grid

        selected = []

        def add(row: int, column: int) -> None:
            tile = grid[row * len(xs) + column]
            if tile not in selected and len(selected) < self.config.max_tiles:
                selected.append(tile)

        last_row = len(ys) - 1
        last_column = len(xs) - 1
        middle_row = last_row // 2
        middle_column = last_column // 2
        for row, column in (
            (0, 0),
            (0, last_column),
            (last_row, 0),
            (last_row, last_column),
            (0, middle_column),
            (last_row, middle_column),
            (middle_row, 0),
            (middle_row, last_column),
        ):
            add(row, column)

        if len(selected) == self.config.max_tiles:
            return selected

        target_indexes = [
            round(
                index
                * (len(grid) - 1)
                / max(1, self.config.max_tiles - 1)
            )
            for index in range(self.config.max_tiles)
        ]
        for target in target_indexes:
            for candidate in sorted(
                range(len(grid)),
                key=lambda index: (abs(index - target), index),
            ):
                tile = grid[candidate]
                if tile not in selected:
                    selected.append(tile)
                    break
            if len(selected) == self.config.max_tiles:
                break
        return selected

    @staticmethod
    def _offset_items(
        items: list[dict], offset_x: int, offset_y: int
    ) -> list[dict]:
        offset_items = []
        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            offset_items.append(
                {
                    **item,
                    "bbox": [
                        x1 + offset_x,
                        y1 + offset_y,
                        x2 + offset_x,
                        y2 + offset_y,
                    ],
                }
            )
        return offset_items

    @staticmethod
    def _scale_items(
        items: list[dict],
        scale_x: float,
        scale_y: float,
        bounds: tuple[int, int],
    ) -> list[dict]:
        width, height = bounds
        scaled_items = []
        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            scaled_items.append(
                {
                    **item,
                    "bbox": [
                        max(0, min(width, math.floor(x1 * scale_x))),
                        max(0, min(height, math.floor(y1 * scale_y))),
                        max(0, min(width, math.ceil(x2 * scale_x))),
                        max(0, min(height, math.ceil(y2 * scale_y))),
                    ],
                }
            )
        return scaled_items
