# OCR Adaptive Tiled Strategy Implementation Plan

> **For the AI implementation agent:** Required sub-skill: use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task by task. Track each
> step with the checkboxes below.

**Goal:** Add an optional, memory-bounded OCR strategy that combines one global
pass with at most six sequential high-detail tile passes for large images while
preserving the current result contract and small-image behavior.

**Architecture:** A new `AdaptiveTiledOCRStrategy` module owns large-image
decode planning, tile selection, coordinate restoration, pass orchestration,
and duplicate suppression. `RapidOCRAdapter` retains model construction and the
single inference lock, exposes one decoded-image inference callback, and routes
only configured large-image work through the strategy. The strategy returns the
same item dictionaries already consumed by `build_ocr_payload`.

**Technical stack:** Python 3.10+, OpenCV, NumPy, RapidOCR 3.9.1, standard
library `dataclasses`, `difflib`, `logging`, `time`, `unicodedata`, and
`unittest`.

---

## File Structure

- Create `perception/plugins/ocr_tiled_strategy.py`: validated strategy config,
  bounded decode planning, tile generation/selection, coordinate transforms,
  duplicate suppression, sequential inference orchestration, and summary logs.
- Create `perception/tests/test_ocr_tiled_strategy.py`: isolated unit tests for
  the strategy with fake inference callbacks; no RapidOCR model is loaded.
- Modify `perception/plugins/ocr_runtime.py`: keep engine ownership and locking,
  extract decoded-image inference, retain the current single-pass fallback, and
  delegate configured large images to the strategy.
- Modify `perception/plugins/ocr.py`: pass strategy configuration into the
  adapter and include it in the shared-adapter cache signature.
- Modify `perception/config.yaml`: enable the approved leaderboard defaults.
- Modify `perception/tests/test_ocr_contract.py`: verify adapter construction,
  route selection, one-engine behavior, and unchanged output/error contracts.
- Modify `perception/tests/test_ocr_packaging.py`: pin the expected strategy
  defaults in the submitted configuration.

## Task 1: Configuration and JPEG Decode Planning

**Files:**
- Create: `perception/plugins/ocr_tiled_strategy.py`
- Create: `perception/tests/test_ocr_tiled_strategy.py`

- [ ] **Step 1: Write failing config-validation and decode-planning tests**

Add tests with a tiny fake OpenCV constant namespace. The tests must assert the
exact reduction decisions from the approved design:

```python
import types
import unittest
from unittest import mock

import numpy as np

from plugins.ocr_tiled_strategy import (
    AdaptiveTiledOCRStrategy,
    LargeImageStrategyConfig,
)


class OCRTiledStrategyTest(unittest.TestCase):
    def setUp(self):
        self.cv2 = types.SimpleNamespace(
            IMREAD_COLOR=1,
            IMREAD_REDUCED_COLOR_2=2,
            IMREAD_REDUCED_COLOR_4=4,
            IMREAD_REDUCED_COLOR_8=8,
        )

    def test_rejects_overlap_not_smaller_than_tile(self):
        with self.assertRaisesRegex(ValueError, "overlap must be smaller"):
            LargeImageStrategyConfig.from_mapping(
                {"tile_size": 1280, "overlap": 1280}
            )

    def test_strategy_is_opt_in_when_config_section_is_absent(self):
        config = LargeImageStrategyConfig.from_mapping(None)
        self.assertFalse(config.enabled)

    def test_decode_plan_preserves_detail_without_crossing_hard_limit(self):
        strategy = AdaptiveTiledOCRStrategy({}, global_max_side=1600)

        cases = {
            4000: (1, 3200),
            6000: (2, 3000),
            7000: (2, 3200),
            9000: (4, 2250),
        }
        for source_side, expected in cases.items():
            plan = strategy._plan_jpeg_decode(
                self.cv2, (source_side, source_side)
            )
            self.assertEqual((plan.factor, max(plan.target_size)), expected)

    def test_decode_plan_never_exceeds_hard_limit(self):
        strategy = AdaptiveTiledOCRStrategy(
            {"decode_side": 4096, "decode_hard_limit": 4096},
            global_max_side=1600,
        )
        plan = strategy._plan_jpeg_decode(self.cv2, (9000, 4500))
        self.assertEqual(plan.factor, 4)
        self.assertLessEqual(max(plan.target_size), 4096)

    def test_rejects_tile_larger_than_decode_hard_limit(self):
        with self.assertRaisesRegex(ValueError, "tile_size must not exceed"):
            LargeImageStrategyConfig.from_mapping(
                {"tile_size": 5000, "decode_hard_limit": 4096}
            )

    def test_decode_plan_rejects_image_too_large_for_reduced_decode(self):
        strategy = AdaptiveTiledOCRStrategy({}, global_max_side=1600)
        with self.assertRaisesRegex(ValueError, "JPEG dimensions exceed"):
            strategy._plan_jpeg_decode(self.cv2, (40000, 30000))
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=perception python3 -m unittest \
  perception.tests.test_ocr_tiled_strategy -v
```

Expected: `ERROR` because `plugins.ocr_tiled_strategy` does not exist.

- [ ] **Step 3: Implement validated config and decode planning**

Start the module with immutable configuration and a plan carrying the OpenCV
flag, reduction factor, reduced size, and final bounded target size:

```python
from __future__ import annotations

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
    ) -> "LargeImageStrategyConfig":
        values = values or {}
        config = cls(**{name: values[name] for name in cls.__dataclass_fields__
                        if name in values})
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("trigger_side", "decode_side", "decode_hard_limit",
                     "tile_size", "max_tiles"):
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
```

Implement `AdaptiveTiledOCRStrategy.__init__(config, global_max_side)` and
`_plan_jpeg_decode(cv2, source_size)`. Evaluate factors `1, 2, 4, 8` in order,
choose the smallest factor whose reduced longest side does not exceed
`decode_hard_limit`, and resize that reduced image down to `decode_side` only
when still larger. If even factor 8 exceeds `decode_hard_limit`, raise
`ValueError("JPEG dimensions exceed reduced decode hard limit")` before calling
OpenCV. Round each planned dimension with `ceil(source / factor)`; when
resizing, round dimensions and clamp them to at least one pixel.

The strategy constructor also rejects a non-positive `global_max_side` or a
value above `decode_hard_limit`, because the global callback must obey the same
inference-image hard limit as tile callbacks.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the command from Step 2. Expected: all configuration and decode-planning
tests report `ok`.

- [ ] **Step 5: Commit the configuration and planner**

```bash
git add perception/plugins/ocr_tiled_strategy.py \
  perception/tests/test_ocr_tiled_strategy.py
git commit -m "feat(ocr): add bounded large-image decode planner"
```

## Task 2: Tile Coverage and Coordinate Transforms

**Files:**
- Modify: `perception/plugins/ocr_tiled_strategy.py`
- Modify: `perception/tests/test_ocr_tiled_strategy.py`

- [ ] **Step 1: Write failing tile and coordinate tests**

Add tests for edge anchoring, deterministic selection, local offsets, source
scaling, and clipping:

```python
def test_tile_grid_anchors_right_and_bottom_edges(self):
    strategy = AdaptiveTiledOCRStrategy(
        {"tile_size": 1280, "overlap": 192, "max_tiles": 20}, 1600
    )
    tiles = strategy._select_tiles((2500, 2100))
    self.assertIn((1220, 820, 2500, 2100), tiles)
    self.assertEqual(len(tiles), len(set(tiles)))

def test_tile_limit_is_deterministic_and_covers_corners(self):
    strategy = AdaptiveTiledOCRStrategy(
        {"tile_size": 1280, "overlap": 192, "max_tiles": 6}, 1600
    )
    first = strategy._select_tiles((4000, 3000))
    second = strategy._select_tiles((4000, 3000))
    self.assertEqual(first, second)
    self.assertEqual(len(first), 6)
    self.assertIn((0, 0, 1280, 1280), first)
    self.assertIn((2720, 1720, 4000, 3000), first)

def test_offsets_then_scales_tile_box_to_source_pixels(self):
    strategy = AdaptiveTiledOCRStrategy({}, 1600)
    decoded_items = strategy._offset_items(
        [{"text": "small", "bbox": [10, 20, 110, 60], "score": 0.8}],
        offset_x=1000,
        offset_y=500,
    )
    source_items = strategy._scale_items(
        decoded_items,
        scale_x=2.0,
        scale_y=2.0,
        bounds=(4000, 3000),
    )
    self.assertEqual(source_items[0]["bbox"], [2020, 1040, 2220, 1120])
```

- [ ] **Step 2: Run the focused tests and verify missing helpers fail**

Run the Task 1 test command. Expected: `AttributeError` for `_select_tiles`,
`_offset_items`, or `_scale_items`.

- [ ] **Step 3: Implement tile generation and deterministic selection**

Implement these private helpers:

```python
def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    edge = length - tile_size
    if starts[-1] != edge:
        starts.append(edge)
    return starts


def _select_tiles(self, image_size: tuple[int, int]) \
        -> list[tuple[int, int, int, int]]:
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

    target_indexes = [
        round(index * (len(grid) - 1) / max(1, self.config.max_tiles - 1))
        for index in range(self.config.max_tiles)
    ]
    for target in target_indexes:
        for candidate in sorted(
            range(len(grid)), key=lambda index: (abs(index - target), index)
        ):
            tile = grid[candidate]
            if tile not in selected:
                selected.append(tile)
                break
        if len(selected) == self.config.max_tiles:
            break
    return selected
```

The selection order must be stable and all chosen tiles must be unique. Use a
small `add(tile)` closure to enforce uniqueness and `max_tiles`. If an image is
smaller than `tile_size` on one axis, the tile ends at the actual image edge.

Implement `_offset_items` by copying item dictionaries and adding tile origins
to all four bbox coordinates. Implement `_scale_items` with `floor` for top-left,
`ceil` for bottom-right, then clip to `[0, source_width]` and
`[0, source_height]`. Do not mutate callback-owned items.

- [ ] **Step 4: Run focused tests and verify they pass**

Expected: all tests in `test_ocr_tiled_strategy.py` report `ok`.

- [ ] **Step 5: Commit tile geometry**

```bash
git add perception/plugins/ocr_tiled_strategy.py \
  perception/tests/test_ocr_tiled_strategy.py
git commit -m "feat(ocr): add deterministic tile coverage"
```

## Task 3: Duplicate Suppression and Reading Order

**Files:**
- Modify: `perception/plugins/ocr_tiled_strategy.py`
- Modify: `perception/tests/test_ocr_tiled_strategy.py`

- [ ] **Step 1: Write failing duplicate-suppression tests**

Use `_Candidate` inputs so tie handling can distinguish global and tile output:

```python
def test_dedup_keeps_higher_confidence_duplicate(self):
    strategy = AdaptiveTiledOCRStrategy({}, 1600)
    candidates = [
        strategy._candidate(
            {"text": "Old Navy", "bbox": [10, 10, 110, 40], "score": 0.7},
            from_tile=False,
        ),
        strategy._candidate(
            {"text": "old   navy", "bbox": [12, 11, 112, 41], "score": 0.9},
            from_tile=True,
        ),
    ]
    self.assertEqual(strategy._deduplicate(candidates), [candidates[1].item])

def test_dedup_prefers_tile_when_scores_tie(self):
    strategy = AdaptiveTiledOCRStrategy({}, 1600)
    global_item = strategy._candidate(
        {"text": "TEST", "bbox": [0, 0, 100, 30], "score": 0.8}, False
    )
    tile_item = strategy._candidate(
        {"text": "test", "bbox": [1, 1, 101, 31], "score": 0.8}, True
    )
    self.assertEqual(
        strategy._deduplicate([global_item, tile_item]), [tile_item.item]
    )

def test_dedup_retains_overlapping_different_text(self):
    strategy = AdaptiveTiledOCRStrategy({}, 1600)
    items = [
        strategy._candidate(
            {"text": "price", "bbox": [0, 0, 100, 30], "score": 0.9}, False
        ),
        strategy._candidate(
            {"text": "total", "bbox": [2, 1, 102, 31], "score": 0.8}, True
        ),
    ]
    self.assertEqual(len(strategy._deduplicate(items)), 2)

def test_results_are_sorted_top_to_bottom_then_left_to_right(self):
    strategy = AdaptiveTiledOCRStrategy({}, 1600)
    items = [
        strategy._candidate(
            {"text": "right", "bbox": [100, 10, 150, 30], "score": 0.9}, True
        ),
        strategy._candidate(
            {"text": "bottom", "bbox": [0, 50, 50, 70], "score": 0.9}, True
        ),
        strategy._candidate(
            {"text": "left", "bbox": [10, 10, 60, 30], "score": 0.9}, False
        ),
    ]
    self.assertEqual(
        [item["text"] for item in strategy._deduplicate(items)],
        ["left", "right", "bottom"],
    )
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Expected: missing `_candidate` or `_deduplicate` methods.

- [ ] **Step 3: Implement candidate metadata and deduplication**

Add a private immutable candidate type and comparison helpers:

```python
from difflib import SequenceMatcher
import re
import unicodedata


@dataclass(frozen=True)
class _Candidate:
    item: dict
    from_tile: bool


def _normalize_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()
```

Compute axis-aligned bbox IoU with zero-area boxes returning `0.0`. Two
candidates are duplicates only when IoU and `SequenceMatcher(...).ratio()` meet
both configured thresholds. Process candidates by descending `(score,
from_tile)` so a higher confidence wins and tile wins exact ties. Keep the first
non-duplicate candidate, remove `_Candidate` metadata, and sort final item
dictionaries by `(bbox[1], bbox[0])`. Missing scores compare as `0.0`.

- [ ] **Step 4: Run focused tests and verify they pass**

Expected: all strategy tests report `ok`.

- [ ] **Step 5: Commit duplicate suppression**

```bash
git add perception/plugins/ocr_tiled_strategy.py \
  perception/tests/test_ocr_tiled_strategy.py
git commit -m "feat(ocr): merge overlapping OCR passes"
```

## Task 4: Sequential Recognition Orchestration and Failure Handling

**Files:**
- Modify: `perception/plugins/ocr_tiled_strategy.py`
- Modify: `perception/tests/test_ocr_tiled_strategy.py`

- [ ] **Step 1: Write failing orchestration tests**

Use real NumPy arrays while replacing the decode and global-resize helpers. The
callback records image shapes and returns deterministic global/tile items:

```python
def test_large_image_runs_global_then_bounded_sequential_tiles(self):
    decoded = np.zeros((3000, 4000, 3), dtype=np.uint8)
    global_image = np.zeros((1200, 1600, 3), dtype=np.uint8)
    strategy = AdaptiveTiledOCRStrategy(
        {"tile_size": 1280, "overlap": 192, "max_tiles": 6}, 1600
    )
    strategy._decode_image = mock.Mock(
        return_value=types.SimpleNamespace(
            image=decoded,
            source_size=(4000, 3000),
            factor=1,
        )
    )
    strategy._resize_longest = mock.Mock(return_value=global_image)
    calls = []

    def infer(image):
        calls.append(image.shape[:2])
        return []

    strategy.recognize(b"jpeg", infer)

    self.assertEqual(calls[0], (1200, 1600))
    self.assertEqual(len(calls), 7)
    self.assertTrue(all(max(shape) <= 1280 for shape in calls[1:]))

def test_partial_tile_failure_returns_successful_passes(self):
    strategy = AdaptiveTiledOCRStrategy(
        {"tile_size": 1280, "overlap": 192, "max_tiles": 2}, 1600
    )
    strategy._decode_image = mock.Mock(
        return_value=types.SimpleNamespace(
            image=np.zeros((2000, 2600, 3), dtype=np.uint8),
            source_size=(2600, 2000),
            factor=1,
        )
    )
    calls = 0

    def infer(_image):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("tile failed")
        if calls == 3:
            return [{"text": "kept", "bbox": [10, 20, 110, 60], "score": 0.8}]
        return []

    result = strategy.recognize(b"jpeg", infer)

    self.assertEqual([item["text"] for item in result], ["kept"])
    self.assertEqual(calls, 3)

def test_global_failure_still_attempts_tiles(self):
    strategy = AdaptiveTiledOCRStrategy(
        {"tile_size": 1280, "overlap": 192, "max_tiles": 1}, 1600
    )
    strategy._decode_image = mock.Mock(
        return_value=types.SimpleNamespace(
            image=np.zeros((2000, 2600, 3), dtype=np.uint8),
            source_size=(2600, 2000),
            factor=1,
        )
    )
    infer = mock.Mock(
        side_effect=[
            RuntimeError("global failed"),
            [{"text": "tile", "bbox": [5, 5, 50, 30], "score": 0.9}],
        ]
    )

    result = strategy.recognize(b"jpeg", infer)

    self.assertEqual(result[0]["text"], "tile")
    self.assertEqual(infer.call_count, 2)

def test_total_inference_failure_raises_first_error(self):
    strategy = AdaptiveTiledOCRStrategy(
        {"tile_size": 1280, "overlap": 192, "max_tiles": 1}, 1600
    )
    strategy._decode_image = mock.Mock(
        return_value=types.SimpleNamespace(
            image=np.zeros((2000, 2600, 3), dtype=np.uint8),
            source_size=(2600, 2000),
            factor=1,
        )
    )

    with self.assertRaisesRegex(ValueError, "global failed"):
        strategy.recognize(
            b"jpeg", mock.Mock(side_effect=ValueError("global failed"))
        )

def test_empty_success_is_not_reported_as_failure(self):
    strategy = AdaptiveTiledOCRStrategy(
        {"tile_size": 1280, "overlap": 192, "max_tiles": 1}, 1600
    )
    strategy._decode_image = mock.Mock(
        return_value=types.SimpleNamespace(
            image=np.zeros((2000, 2600, 3), dtype=np.uint8),
            source_size=(2600, 2000),
            factor=1,
        )
    )

    self.assertEqual(strategy.recognize(b"jpeg", lambda _image: []), [])

def test_small_non_jpeg_uses_one_single_pass(self):
    strategy = AdaptiveTiledOCRStrategy({"enabled": True}, 1600)
    strategy._decode_image = mock.Mock(
        return_value=types.SimpleNamespace(
            image=np.zeros((600, 800, 3), dtype=np.uint8),
            source_size=(800, 600),
            factor=1,
        )
    )
    infer = mock.Mock(
        return_value=[
            {"text": "small", "bbox": [10, 20, 100, 50], "score": 0.9}
        ]
    )

    result = strategy.recognize(b"png", infer)

    self.assertEqual(result[0]["bbox"], [10, 20, 100, 50])
    infer.assert_called_once()
```

For `_decode_image` tests, add `_jpeg(width, height)` as a test helper that
constructs the same minimal SOF-header bytes already used in
`test_ocr_contract.py`.

- [ ] **Step 2: Run focused tests and verify orchestration is missing**

Expected: `AttributeError` because `recognize` and/or decode helpers are not yet
implemented.

- [ ] **Step 3: Implement decode, pass orchestration, and one summary log**

Add a decoded-image carrier and implement the following control flow:

```python
@dataclass(frozen=True)
class _DecodedImage:
    image: object
    source_size: tuple[int, int]
    factor: int


def recognize(self, image_bytes: bytes, infer_image: Callable[[object], list]) \
        -> list[dict]:
    started = time.perf_counter()
    decoded = self._decode_image(image_bytes)
    decode_elapsed = time.perf_counter() - started
    image = decoded.image
    decoded_height, decoded_width = image.shape[:2]
    source_width, source_height = decoded.source_size

    if max(source_width, source_height) <= self.config.trigger_side:
        items = infer_image(image)
        return self._scale_items(
            items,
            scale_x=source_width / decoded_width,
            scale_y=source_height / decoded_height,
            bounds=decoded.source_size,
        )

    candidates = []
    successful_passes = 0
    first_error = None
    global_count = 0
    successful_tiles = 0
    global_elapsed = 0.0
    tile_elapsed = 0.0

    if self.config.global_pass:
        global_image = self._resize_longest(image, self.global_max_side)
        global_height, global_width = global_image.shape[:2]
        global_started = time.perf_counter()
        try:
            global_items = infer_image(global_image)
            successful_passes += 1
            global_count = len(global_items)
            decoded_items = self._scale_items(
                global_items,
                scale_x=decoded_width / global_width,
                scale_y=decoded_height / global_height,
                bounds=(decoded_width, decoded_height),
            )
            candidates.extend(self._candidate(item, False)
                              for item in decoded_items)
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
            decoded_items = self._offset_items(tile_items, x1, y1)
            candidates.extend(self._candidate(item, True)
                              for item in decoded_items)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            log.warning("OCR tile pass failed at (%d,%d,%d,%d): %s",
                        x1, y1, x2, y2, exc)
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
```

Implement `_decode_image`, `_resize_longest`, and `_log_summary` directly above
`recognize`. `_decode_image` reads JPEG header dimensions, applies
`_plan_jpeg_decode`, calls `cv2.imdecode`, raises
`ValueError("invalid compressed image")` for a `None` result, and applies the
plan's final resize. For non-JPEG input it decodes with `IMREAD_COLOR`, records
the decoded shape as source size, and bounds the retained image to
`decode_side`. `_resize_longest` returns the original object when already under
the limit and otherwise uses `cv2.INTER_AREA` with aspect-ratio-preserving
rounded dimensions.

Use `time.perf_counter()` for decode, global, tile, and total durations. Store
the first inference exception. A callback return value of `[]` increments the
successful-pass count. Log global and tile failures without recognized text or
image bytes. If no pass succeeds, raise the first exception; otherwise return
all successful results, including an empty list.

For non-JPEG input, decode in full with `IMREAD_COLOR`, treat the decoded shape
as source size, then resize the strategy image so its longest side is at most
`decode_side`. This does not claim to bound the transient OpenCV decode
allocation. Before every callback, assert the image's longest side is no larger
than `decode_hard_limit`.

Emit one `INFO` summary containing source/decoded dimensions, reduction factor,
global item count, selected/successful tile counts, before/after dedup counts,
and elapsed timings. Emit individual pass failures at `WARNING`; other per-tile
messages stay at `DEBUG`.

- [ ] **Step 4: Run focused tests and verify they pass**

Expected: all strategy tests report `ok` and callback count never exceeds
`1 + max_tiles` when `global_pass` is enabled.

- [ ] **Step 5: Commit orchestration**

```bash
git add perception/plugins/ocr_tiled_strategy.py \
  perception/tests/test_ocr_tiled_strategy.py
git commit -m "feat(ocr): orchestrate adaptive tiled inference"
```

## Task 5: Integrate the Strategy with RapidOCRAdapter

**Files:**
- Modify: `perception/plugins/ocr_runtime.py:60-230`
- Modify: `perception/plugins/ocr.py:412-443`
- Modify: `perception/tests/test_ocr_contract.py:102-391`

- [ ] **Step 1: Write failing adapter wiring and route-selection tests**

Update the expected constructor call in
`test_default_provider_builds_local_adapter`:

```python
config = {
    "provider": "rapidocr",
    "model_dir": "/models/ocr/ppocrv6-tiny",
    "use_angle_cls": True,
    "num_threads": 2,
    "max_side_len": 1600,
    "large_image_strategy": {"enabled": True, "trigger_side": 2400},
}
result = self.ocr._build_ocr_adapter(config)

adapter.assert_called_once_with(
    "/models/ocr/ppocrv6-tiny",
    use_angle_cls=True,
    num_threads=2,
    max_side_len=1600,
    large_image_strategy={
        "enabled": True,
        "trigger_side": 2400,
    },
)
```

Add tests:

```python
def test_adapter_signature_changes_with_large_image_strategy(self):
    first = self.ocr._adapter_signature({
        "provider": "rapidocr",
        "large_image_strategy": {"enabled": True, "max_tiles": 6},
    })
    second = self.ocr._adapter_signature({
        "provider": "rapidocr",
        "large_image_strategy": {"enabled": True, "max_tiles": 4},
    })
    self.assertNotEqual(first, second)

def test_large_jpeg_delegates_to_strategy(self):
    adapter = object.__new__(self.ocr.RapidOCRAdapter)
    adapter._large_image_strategy = mock.Mock()
    adapter._large_image_strategy.should_handle.return_value = True
    adapter._large_image_strategy.recognize.return_value = [{"text": "tile"}]
    adapter._infer_image = mock.Mock()

    image_bytes = self._jpeg(4000, 3000)
    result = adapter.recognize(image_bytes)

    self.assertEqual(result, [{"text": "tile"}])
    adapter._large_image_strategy.recognize.assert_called_once_with(
        image_bytes, adapter._infer_image
    )

def test_small_image_keeps_existing_single_pass_path(self):
    adapter = object.__new__(self.ocr.RapidOCRAdapter)
    adapter._large_image_strategy = mock.Mock()
    adapter._large_image_strategy.should_handle.return_value = False
    adapter._recognize_single_pass = mock.Mock(
        return_value=[{"text": "small", "bbox": [10, 20, 100, 50]}]
    )
    image_bytes = self._jpeg(800, 600)

    result = adapter.recognize(image_bytes)

    self.assertEqual(result[0]["bbox"], [10, 20, 100, 50])
    adapter._recognize_single_pass.assert_called_once_with(image_bytes)
    adapter._large_image_strategy.recognize.assert_not_called()

def test_strategy_uses_same_locked_engine_callback(self):
    adapter = object.__new__(self.ocr.RapidOCRAdapter)
    adapter._use_angle_cls = True
    adapter._inference_lock = mock.MagicMock()
    adapter._engine = mock.Mock(
        return_value=types.SimpleNamespace(boxes=[], txts=(), scores=())
    )
    image = object()

    result = adapter._infer_image(image)

    self.assertEqual(result, [])
    adapter._inference_lock.__enter__.assert_called_once_with()
    adapter._inference_lock.__exit__.assert_called_once()
    adapter._engine.assert_called_once_with(
        image, use_det=True, use_cls=True, use_rec=True
    )
```

Use a shared `_jpeg(width, height)` test helper rather than duplicating the SOF
byte construction in each contract test.

- [ ] **Step 2: Run contract tests and verify new expectations fail**

Run:

```bash
PYTHONPATH=perception python3 -m unittest \
  perception.tests.test_ocr_contract -v
```

Expected: constructor/signature/delegation assertions fail because the strategy
is not wired into the adapter.

- [ ] **Step 3: Extract one locked decoded-image inference callback**

In `RapidOCRAdapter`, add:

```python
def _infer_image(self, image) -> list[dict]:
    with self._inference_lock:
        output = self._engine(
            image,
            use_det=True,
            use_cls=self._use_angle_cls,
            use_rec=True,
        )
    return normalize_rapidocr_output(output)
```

Change the current single-pass implementation to call `_infer_image(image)` and
then scale those item bboxes to source pixels. Keep its existing decode flag,
resize behavior, errors, and output ordering unchanged. Name the extracted path
`_recognize_single_pass(image_bytes)` so route selection is explicit.

- [ ] **Step 4: Construct and route the optional strategy**

Extend the constructor:

```python
def __init__(
    self,
    model_dir: str,
    use_angle_cls: bool = True,
    num_threads: int = 2,
    max_side_len: int = 1600,
    large_image_strategy: dict | None = None,
):
    ...
    strategy_config = LargeImageStrategyConfig.from_mapping(
        large_image_strategy
    )
    self._large_image_strategy = (
        AdaptiveTiledOCRStrategy(strategy_config, max_side_len)
        if strategy_config.enabled
        else None
    )
```

Add the route predicate to `AdaptiveTiledOCRStrategy`:

```python
def should_handle(self, source_size: tuple[int, int] | None) -> bool:
    return self.config.enabled and (
        source_size is None or max(source_size) > self.config.trigger_side
    )
```

`recognize` reads source JPEG dimensions once. Delegate only when the strategy
exists and `strategy.should_handle(source_size)` is true. For non-JPEG input,
the strategy's `should_handle(None)` must return true so it can decode and
inspect dimensions; if the decoded non-JPEG image is at or below
`trigger_side`, its strategy path performs exactly one unscaled inference and
returns source-pixel coordinates. Disabled strategy and known small JPEG input
must call `_recognize_single_pass` directly.

This non-JPEG rule is the only route where dimensions are unavailable before
decode; cover it in `test_ocr_tiled_strategy.py` so small PNG input performs one
callback call and large PNG input uses tiles.

- [ ] **Step 5: Pass config and freeze it in adapter identity**

In `_build_ocr_adapter`, add:

```python
large_image_strategy=dict(cfg.get("large_image_strategy") or {}),
```

Add a recursive helper in `ocr.py` so nested configuration is hashable and
stable:

```python
def _freeze_config(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_config(item))
                            for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_config(item) for item in value)
    return value
```

Append `_freeze_config(cfg.get("large_image_strategy", {}))` to the RapidOCR
adapter signature. This prevents a running plugin from reusing an adapter built
with stale tile settings.

- [ ] **Step 6: Run strategy and contract tests**

Run:

```bash
PYTHONPATH=perception python3 -m unittest \
  perception.tests.test_ocr_tiled_strategy \
  perception.tests.test_ocr_contract -v
```

Expected: all tests pass; existing small JPEG and non-strategy bbox assertions
remain unchanged.

- [ ] **Step 7: Commit adapter integration**

```bash
git add perception/plugins/ocr_runtime.py perception/plugins/ocr.py \
  perception/tests/test_ocr_contract.py \
  perception/tests/test_ocr_tiled_strategy.py
git commit -m "feat(ocr): integrate adaptive tiled strategy"
```

## Task 6: Enable Leaderboard Defaults and Add Packaging Guards

**Files:**
- Modify: `perception/config.yaml:54-64`
- Modify: `perception/tests/test_ocr_packaging.py:19-31`

- [ ] **Step 1: Write failing default-configuration assertions**

Replace loose string-only checks for the strategy with parsed YAML assertions:

```python
def test_default_config_is_bounded_for_ocr_leaderboard(self):
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "perception" / "config.yaml").read_text(encoding="utf-8")
    )
    ocr = config["plugins"]["ocr"]
    self.assertTrue(ocr["enabled"])
    self.assertEqual(ocr["max_side_len"], 1600)
    self.assertEqual(
        ocr["large_image_strategy"],
        {
            "enabled": True,
            "trigger_side": 2400,
            "decode_side": 3200,
            "decode_hard_limit": 4096,
            "tile_size": 1280,
            "overlap": 192,
            "max_tiles": 6,
            "global_pass": True,
            "dedup_iou": 0.5,
            "dedup_text_similarity": 0.8,
        },
    )
```

Keep the existing ASR-disabled, empty remote credential, model path, and thread
count assertions; do not weaken packaging coverage.

- [ ] **Step 2: Run packaging tests and verify missing config fails**

Run:

```bash
PYTHONPATH=perception python3 -m unittest \
  perception.tests.test_ocr_packaging -v
```

Expected: failure because `large_image_strategy` is absent.

- [ ] **Step 3: Add the approved defaults to `config.yaml`**

Under `plugins.ocr`, add exactly:

```yaml
    large_image_strategy:
      enabled: true
      trigger_side: 2400
      decode_side: 3200
      decode_hard_limit: 4096
      tile_size: 1280
      overlap: 192
      max_tiles: 6
      global_pass: true
      dedup_iou: 0.5
      dedup_text_similarity: 0.8
```

- [ ] **Step 4: Run packaging plus OCR tests**

Run:

```bash
PYTHONPATH=perception python3 -m unittest \
  perception.tests.test_ocr_tiled_strategy \
  perception.tests.test_ocr_contract \
  perception.tests.test_ocr_packaging -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the default configuration**

```bash
git add perception/config.yaml perception/tests/test_ocr_packaging.py
git commit -m "feat(ocr): enable adaptive tiling defaults"
```

## Task 7: Regression, Synthetic Integration, and Submission Checks

**Files:**
- Modify: `perception/tests/test_ocr_tiled_strategy.py`

- [ ] **Step 1: Add a deterministic synthetic recall integration test**

Add a test with a 4000x3000 NumPy image containing a marked small-text region.
The fake inference callback returns a large-text item for the 1600-pixel global
image and returns a small-text item only when its tile contains the marker at
tile resolution. Assert:

```python
self.assertEqual(
    [item["text"] for item in result],
    ["large text", "small text"],
)
self.assertEqual(sum(item["text"] == "large text" for item in result), 1)
self.assertTrue(all(0 <= item["bbox"][0] < item["bbox"][2] <= 4000
                    for item in result))
self.assertTrue(all(0 <= item["bbox"][1] < item["bbox"][3] <= 3000
                    for item in result))
```

This validates the pipeline and coordinate contract without requiring local
model files. It does not claim model-accuracy validation; that remains a Jetson
smoke/evaluation step.

- [ ] **Step 2: Run all perception unit tests**

Run:

```bash
PYTHONPATH=perception python3 -m unittest discover \
  -s perception/tests -p 'test_*.py' -v
```

Expected: `OK`, with no failures or errors.

- [ ] **Step 3: Run static repository checks**

Run:

```bash
python3 -m py_compile \
  perception/plugins/ocr_tiled_strategy.py \
  perception/plugins/ocr_runtime.py \
  perception/plugins/ocr.py
git diff --check
git status --short
```

Expected: compilation and `git diff --check` produce no output. `git status`
shows only intentional changes if the synthetic test required a final edit.

- [ ] **Step 4: Commit the synthetic integration test**

```bash
git add perception/tests/test_ocr_tiled_strategy.py
git commit -m "test(ocr): cover tiled small-text recovery"
```

- [ ] **Step 5: Build and smoke-test on Jetson**

On the Jetson checkout, build the exact final commit without modifying files:

```bash
docker build --network=host --no-cache \
  -f perception/Dockerfile.jetson \
  -t phanthymotus-ocr:$(git rev-parse --short HEAD) .
```

Run the existing deployment/evaluation entrypoint with the model downloaded by
the Dockerfile. Verify from logs that one RapidOCR engine initializes, a large
image reports no more than six selected tiles, returned boxes stay inside source
dimensions, and the process does not OOM.

- [ ] **Step 6: Run the 250-case leaderboard evaluation before promotion**

Submit the final commit ID through `PHANTHYMOTUS_COMMIT_ID`. Acceptance requires:

- 250/250 requests complete without OOM or request failures;
- no image schedules more than six tile passes;
- output payload remains `text + items + timestamp + language`;
- F1 improves relative to the `0.2241` baseline before this strategy replaces
  the current leaderboard commit.

Do not push model files or any generated image larger than 1 MB.
