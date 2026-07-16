from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Mapping


log = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class _DecodedImage:
    image: object
    source_size: tuple[int, int]
    factor: int


@dataclass(frozen=True)
class _Candidate:
    item: dict
    from_tile: bool


def _normalize_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def jpeg_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    if len(image_bytes) < 4 or image_bytes[:2] != b"\xff\xd8":
        return None

    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    offset = 2
    while offset + 3 < len(image_bytes):
        if image_bytes[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(image_bytes) and image_bytes[offset] == 0xFF:
            offset += 1
        if offset >= len(image_bytes):
            break

        marker = image_bytes[offset]
        offset += 1
        if marker in (0x01, 0xD8, 0xD9):
            continue
        if offset + 2 > len(image_bytes):
            break

        segment_len = int.from_bytes(image_bytes[offset : offset + 2], "big")
        if segment_len < 2 or offset + segment_len > len(image_bytes):
            break
        if marker in sof_markers and segment_len >= 7:
            height = int.from_bytes(
                image_bytes[offset + 3 : offset + 5], "big"
            )
            width = int.from_bytes(
                image_bytes[offset + 5 : offset + 7], "big"
            )
            if width > 0 and height > 0:
                return width, height
        offset += segment_len
    return None


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

    def should_handle(self, source_size: tuple[int, int] | None) -> bool:
        return self.config.enabled and (
            source_size is None
            or max(source_size) > self.config.trigger_side
        )

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

    @staticmethod
    def _resize_to_size(image, target_size: tuple[int, int]):
        import cv2

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def _resize_longest(self, image, max_side: int):
        height, width = image.shape[:2]
        longest_side = max(width, height)
        if longest_side <= max_side:
            return image
        scale = max_side / longest_side
        target_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        return self._resize_to_size(image, target_size)

    def _decode_image(self, image_bytes: bytes) -> _DecodedImage:
        import cv2
        import numpy as np

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        source_size = jpeg_dimensions(image_bytes)
        if source_size is not None:
            plan = self._plan_jpeg_decode(cv2, source_size)
            image = cv2.imdecode(encoded, plan.flag)
            factor = plan.factor
            target_size = plan.target_size
        else:
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            factor = 1
            target_size = None
        if image is None:
            raise ValueError("invalid compressed image")

        decoded_height, decoded_width = image.shape[:2]
        if source_size is None:
            source_size = (decoded_width, decoded_height)
            longest_side = max(decoded_width, decoded_height)
            if longest_side > self.config.decode_side:
                scale = self.config.decode_side / longest_side
                target_size = (
                    max(1, round(decoded_width * scale)),
                    max(1, round(decoded_height * scale)),
                )

        if target_size is not None and target_size != (
            decoded_width,
            decoded_height,
        ):
            image = self._resize_to_size(image, target_size)
        if max(image.shape[:2]) > self.config.decode_hard_limit:
            raise ValueError("decoded image exceeds decode_hard_limit")
        return _DecodedImage(image=image, source_size=source_size, factor=factor)

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

    @staticmethod
    def _candidate(item: dict, from_tile: bool) -> _Candidate:
        return _Candidate(item=item, from_tile=from_tile)

    @staticmethod
    def _bbox_iou(first: list, second: list) -> float:
        first_x1, first_y1, first_x2, first_y2 = first
        second_x1, second_y1, second_x2, second_y2 = second
        intersection_width = max(
            0.0, min(first_x2, second_x2) - max(first_x1, second_x1)
        )
        intersection_height = max(
            0.0, min(first_y2, second_y2) - max(first_y1, second_y1)
        )
        intersection = intersection_width * intersection_height
        first_area = max(0.0, first_x2 - first_x1) * max(
            0.0, first_y2 - first_y1
        )
        second_area = max(0.0, second_x2 - second_x1) * max(
            0.0, second_y2 - second_y1
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def _is_duplicate(self, first: _Candidate, second: _Candidate) -> bool:
        if (
            self._bbox_iou(first.item["bbox"], second.item["bbox"])
            < self.config.dedup_iou
        ):
            return False
        similarity = SequenceMatcher(
            None,
            _normalize_text(first.item.get("text", "")),
            _normalize_text(second.item.get("text", "")),
        ).ratio()
        return similarity >= self.config.dedup_text_similarity

    def _deduplicate(self, candidates: list[_Candidate]) -> list[dict]:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                float(candidate.item.get("score", 0.0)),
                candidate.from_tile,
            ),
            reverse=True,
        )
        kept = []
        for candidate in ordered:
            if any(self._is_duplicate(candidate, other) for other in kept):
                continue
            kept.append(candidate)
        return sorted(
            (candidate.item for candidate in kept),
            key=lambda item: (item["bbox"][1], item["bbox"][0]),
        )

    def _log_summary(
        self,
        decoded: _DecodedImage,
        global_count: int,
        selected_tiles: int,
        successful_tiles: int,
        before_dedup: int,
        after_dedup: int,
        decode_elapsed: float,
        global_elapsed: float,
        tile_elapsed: float,
        started: float,
    ) -> None:
        decoded_height, decoded_width = decoded.image.shape[:2]
        log.info(
            "OCR tiled summary source=%sx%s decoded=%sx%s factor=%s "
            "global_items=%s tiles=%s/%s items=%s/%s "
            "elapsed_decode=%.3fs elapsed_global=%.3fs "
            "elapsed_tiles=%.3fs elapsed_total=%.3fs",
            decoded.source_size[0],
            decoded.source_size[1],
            decoded_width,
            decoded_height,
            decoded.factor,
            global_count,
            successful_tiles,
            selected_tiles,
            before_dedup,
            after_dedup,
            decode_elapsed,
            global_elapsed,
            tile_elapsed,
            time.perf_counter() - started,
        )

    def recognize(
        self,
        image_bytes: bytes,
        infer_image: Callable[[object], list[dict]],
    ) -> list[dict]:
        started = time.perf_counter()
        decoded = self._decode_image(image_bytes)
        decode_elapsed = time.perf_counter() - started
        image = decoded.image
        decoded_height, decoded_width = image.shape[:2]
        source_width, source_height = decoded.source_size

        if max(source_width, source_height) <= self.config.trigger_side:
            single_image = self._resize_longest(image, self.global_max_side)
            single_height, single_width = single_image.shape[:2]
            items = infer_image(single_image)
            return self._scale_items(
                items,
                scale_x=source_width / single_width,
                scale_y=source_height / single_height,
                bounds=decoded.source_size,
            )

        candidates = []
        successful_passes = 0
        first_error = None
        global_count = 0
        successful_tiles = 0
        global_elapsed = 0.0

        if self.config.global_pass:
            global_image = self._resize_longest(image, self.global_max_side)
            global_height, global_width = global_image.shape[:2]
            global_started = time.perf_counter()
            try:
                global_items = infer_image(global_image)
                successful_passes += 1
                global_count = len(global_items)
                global_items = self._scale_items(
                    global_items,
                    scale_x=decoded_width / global_width,
                    scale_y=decoded_height / global_height,
                    bounds=(decoded_width, decoded_height),
                )
                candidates.extend(
                    self._candidate(item, from_tile=False)
                    for item in global_items
                )
            except Exception as exc:
                first_error = exc
                log.warning("OCR global pass failed: %s", exc)
            finally:
                global_elapsed = time.perf_counter() - global_started

        tiles = self._select_tiles((decoded_width, decoded_height))
        tile_started = time.perf_counter()
        for x1, y1, x2, y2 in tiles:
            tile = image[y1:y2, x1:x2]
            try:
                tile_items = infer_image(tile)
                successful_passes += 1
                successful_tiles += 1
                tile_items = self._offset_items(tile_items, x1, y1)
                candidates.extend(
                    self._candidate(item, from_tile=True)
                    for item in tile_items
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                log.warning(
                    "OCR tile pass failed at (%d,%d,%d,%d): %s",
                    x1,
                    y1,
                    x2,
                    y2,
                    exc,
                )
        tile_elapsed = time.perf_counter() - tile_started

        if successful_passes == 0:
            if first_error is not None:
                raise first_error
            raise RuntimeError("OCR strategy scheduled no inference passes")

        merged = self._deduplicate(candidates)
        result = self._scale_items(
            merged,
            scale_x=source_width / decoded_width,
            scale_y=source_height / decoded_height,
            bounds=decoded.source_size,
        )
        self._log_summary(
            decoded=decoded,
            global_count=global_count,
            selected_tiles=len(tiles),
            successful_tiles=successful_tiles,
            before_dedup=len(candidates),
            after_dedup=len(merged),
            decode_elapsed=decode_elapsed,
            global_elapsed=global_elapsed,
            tile_elapsed=tile_elapsed,
            started=started,
        )
        return result
