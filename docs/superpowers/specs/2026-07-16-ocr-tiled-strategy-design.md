# OCR Adaptive Tiled Strategy Design

## Goal

Improve small-text recall on high-resolution OCR leaderboard images without
changing the PP-OCRv6 tiny model, the MCP/ROS contracts, or the existing
single-pass behavior for small images. The large-image path must remain
memory-bounded and independently configurable so it can be enabled, disabled,
or replaced without changing the RapidOCR adapter.

## Problem

The current JPEG reduced-decode selection chooses the first reduction factor
whose output is no larger than `max_side_len`. A 4000-pixel side with a
1600-pixel target is therefore decoded at one quarter size, producing only
1000 pixels. This avoids full-resolution memory use but removes substantially
more small-text detail than requested.

Even when decoding reaches the configured side length, a single global pass
must trade image context against text resolution. Large signs remain visible,
but small receipt, map, packaging, and scene text can disappear. Raising the
global side alone increases detector memory and latency for every large image
and still cannot preserve enough detail on the largest inputs.

## Scope

This change adds an optional adaptive tiled inference strategy for the existing
RapidOCR engine. It includes bounded JPEG decoding, a global pass, overlapping
tile passes, coordinate restoration, and duplicate suppression.

This change does not:

- replace or quantize the OCR model;
- change detector, recognizer, or classifier thresholds;
- change the ROS topics, MCP tool schema, or result payload;
- add image enhancement, rotation ensembles, or language correction rules;
- create additional RapidOCR or ONNX Runtime sessions.

Keeping these variables unchanged makes the next leaderboard run an isolated
test of the resolution and tiling strategy.

## Module Boundary

Add a separate module:

```text
perception/plugins/ocr_tiled_strategy.py
```

The module owns only large-image image processing and result composition. It
does not know how models are constructed and does not own ROS nodes.

The production interface is a strategy object with a callback for one-image
inference:

```python
strategy = AdaptiveTiledOCRStrategy(config)
items = strategy.recognize(
    image_bytes=image_bytes,
    infer_image=infer_image,
)
```

`infer_image` accepts a decoded BGR image and returns OCR items in that image's
pixel coordinate system. The existing `RapidOCRAdapter` supplies this callback
and continues to serialize access to its single RapidOCR engine.

The first implementation is an importable strategy module rather than a
standalone process. The production plugin must not launch a subprocess or
duplicate model state.

## Configuration

The OCR plugin configuration gains one nested section:

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

Configuration is validated at strategy construction. Invalid dimensions,
negative overlap, overlap greater than or equal to tile size, or a non-positive
tile limit fail plugin initialization with a descriptive error.

When `enabled` is false or the source longest side is not greater than
`trigger_side`, `RapidOCRAdapter` uses the current single-pass path. This path
must remain behaviorally identical to the pre-strategy implementation.

## Decode Planning

The strategy reads JPEG dimensions from the compressed header before decoding.
It selects the OpenCV JPEG reduction level that preserves the most detail while
keeping the decoded longest side close to `decode_side` and never above
`decode_hard_limit`.

Representative plans are:

| Source longest side | Reduced decode | Strategy image |
| ---: | ---: | ---: |
| 4000 | full, 4000 | resize to 3200 |
| 6000 | 1/2, 3000 | keep 3000 |
| 7000 | 1/2, 3500 | resize to 3200 |
| 9000 | 1/4, 2250 | keep 2250 |

The planner prioritizes the hard memory limit over the preferred decode side.
Non-JPEG formats retain the existing full decode followed by a bounded resize,
because OpenCV does not provide equivalent reduced-decode flags for them. The
configured hard limit therefore bounds JPEG decode and all images passed into
inference, but it cannot bound the transient full-decode allocation of an
oversized PNG, BMP, or WebP without adding a different image codec.

## Recognition Flow

For an enabled large image:

1. Parse source dimensions and create a bounded decoded image.
2. Run one global pass resized to the adapter's existing `max_side_len` when
   `global_pass` is enabled.
3. Generate overlapping square tiles over the bounded decoded image.
4. Select at most `max_tiles` tiles deterministically, preserving coverage of
   image edges and distributing selected tiles over the full image.
5. Run tiles sequentially through the shared inference callback.
6. Offset tile-local boxes into bounded decoded-image coordinates.
7. Merge global and tile items with duplicate suppression.
8. Scale the merged boxes once from decoded-image coordinates to source-image
   coordinates.

Tiles never run concurrently. The existing inference lock remains the final
guard around the RapidOCR engine.

## Tile Generation

Tiles use `tile_size` on both axes and overlap adjacent tiles by `overlap`.
The last tile on each axis is anchored to the image edge so rightmost and
bottommost content is covered.

If the complete grid exceeds `max_tiles`, selection is deterministic:

- include corner coverage first;
- include edge coverage next;
- fill remaining capacity with tiles nearest evenly spaced grid targets;
- never emit the same tile twice.

This gives predictable latency and avoids selecting only the top-left region of
a dense image.

## Duplicate Suppression

Global and overlapping tile passes can return the same text multiple times.
Items are compared in decoded-image coordinates after normalizing text with
Unicode NFKC, case folding, and whitespace collapse for comparison only. The
published text remains exactly as recognized.

Two items are duplicates when:

- their axis-aligned bbox IoU is at least `dedup_iou`; and
- normalized text similarity is at least `dedup_text_similarity`.

Text similarity uses `difflib.SequenceMatcher` to avoid adding a dependency.
The higher-confidence item is retained. Ties prefer the tile result because it
was inferred from a higher-detail view. Spatially overlapping items with
different text are retained to avoid deleting nearby labels or receipt fields.

Results are sorted in reading order after deduplication using the existing
top-to-bottom, then left-to-right coordinate convention.

## Error Handling

- A decode failure produces the existing publishable error payload through
  `RapidOCRAdapter`.
- A global-pass failure is logged and tile processing continues.
- Failure of one tile is logged with its tile coordinates and does not discard
  successful global or tile results.
- If all tile passes fail but the global pass succeeds, return the global
  results.
- If no pass succeeds, propagate the first inference error so the adapter can
  publish an error payload rather than a false successful empty result.
- Invalid source bytes continue to fail as `invalid compressed image`.

## Compatibility

The output remains:

```json
{
  "text": "...",
  "items": [
    {"text": "...", "bbox": [x1, y1, x2, y2], "score": 0.9}
  ],
  "timestamp": 0.0,
  "language": "zh"
}
```

All published bboxes remain integer axis-aligned source-image pixel
coordinates. The strategy does not change the existing polygon-to-rectangle
normalization, even though that remains a possible future mAP optimization.

## Observability

One summary log entry is emitted per large image with:

- source and decoded dimensions;
- selected JPEG reduction factor;
- global-pass item count;
- selected and successful tile counts;
- item count before and after deduplication;
- decode, global, tile, and total elapsed time.

Per-tile logs are debug-level except for failures. No recognized full text or
image data is added to logs.

## Verification

Add:

```text
perception/tests/test_ocr_tiled_strategy.py
```

Unit tests cover:

- small images bypassing the strategy;
- a 4000-pixel JPEG preserving more than 1000 pixels of detail;
- decode hard-limit selection;
- edge-anchored tile generation;
- deterministic `max_tiles` enforcement;
- tile-local to decoded-image coordinate offsets;
- decoded-image to source-image coordinate scaling;
- duplicate removal by IoU, text similarity, and confidence;
- retention of overlapping items with different text;
- partial tile failure and total failure behavior;
- disabled configuration preserving the current single-pass call pattern.

The existing OCR contract, packaging, and repository tests must remain green.
A local integration check uses a synthetic high-resolution image containing
both large and small text and verifies that the strategy returns the large text
without duplication and recovers at least one small text item missed by the
single global pass.

## Acceptance Criteria

- No model files or model download URLs change.
- Images at or below `trigger_side` follow the existing path.
- Enabled large images never schedule more than `max_tiles` tile passes.
- Only one RapidOCR engine exists per adapter.
- JPEG decoded longest side never exceeds `decode_hard_limit`, and no image
  passed to RapidOCR exceeds it.
- Returned bboxes are in source-image pixels and remain within source bounds.
- Disabling the strategy is a configuration-only rollback.
- The Jetson container completes the existing 250-case evaluation without OOM
  or request failures before the strategy is considered leaderboard-ready.
