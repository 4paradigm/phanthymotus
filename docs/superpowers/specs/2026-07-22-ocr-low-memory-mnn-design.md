# OCR Low-Memory MNN Architecture Design

## Goal

Reduce OCR inference memory on the 8 GB Jetson leaderboard environment without
rejecting valid images, changing the OCR plugin contract, or storing model
files in Git. The default OCR backend will move from RapidOCR with ONNX Runtime
to MNN, while a bounded libvips decode pipeline prevents full-resolution image
materialization.

The design must preserve or improve the current PP-OCRv6 tiny accuracy and must
complete all 250 leaderboard cases without OOM, container restart, dynamic
memory rejection, or evaluator waits caused by incomplete responses.

## Problem

The current pipeline performs several memory-expensive steps:

1. OpenCV decodes compressed input into a complete `uint8` BGR image.
2. The complete image is resized into another array.
3. RapidOCR preprocessing normalizes and transposes image data into float
   tensors. Float32 uses four bytes per element and can exist alongside both
   decoded images.
4. ONNX Runtime sessions and their CPU memory arenas retain model and temporary
   allocations between requests.
5. Detector, recognizer, and potentially classifier configuration are created
   together even when angle classification is disabled.

Float32 conversion is therefore an important contributor, but not the only
cause of the working set. Full-image decode, duplicated arrays, runtime
allocators, and session lifetime must be addressed together.

The current dynamic memory guard is not an acceptable solution. In the latest
leaderboard run it rejected valid cases when system headroom became low. Empty
or incomplete responses then caused the evaluator to wait for its 120-second
timeout. Memory pressure must change how images are decoded and scheduled, not
whether a valid image is processed.

## Scope

This change includes:

- a libvips-based bounded image decoder;
- an MNN detector and recognizer backend;
- sequential overview and tile inference for large images;
- a startup-only ONNX Runtime fallback when its model files fit the model-size
  budget;
- removal of dynamic memory-based case rejection;
- explicit completion responses for empty OCR results and per-case failures;
- memory, latency, and backend observability;
- unit, compatibility, Jetson stress, and leaderboard regression tests.

This change does not:

- alter ROS topics, MCP tools, request schemas, or response schemas;
- add a new OCR model family or language model;
- enable angle classification;
- add GPU or TensorRT inference;
- upload model files larger than 1 MB to Git;
- run detector or recognizer inference concurrently inside one plugin instance;
- redesign the leaderboard evaluator.

## Architecture

The external OCR plugin remains unchanged. Only the adapter's internal image
and inference path is replaced:

```text
compressed image bytes
  -> header probe
  -> libvips bounded decode or region decode
  -> overview and sequential tile planner
  -> MNN PP-OCRv6 tiny detector
  -> sequential text-region crops
  -> MNN PP-OCRv6 tiny recognizer
  -> duplicate suppression and reading-order sort
  -> source-coordinate restoration
  -> existing OCR result payload
```

MNN is the default backend. ONNX Runtime is a startup fallback, not a second
resident engine. If MNN initialization fails, all partially created MNN state
is released before ORT is initialized.

## Implementation Boundary

Keep the implementation in the existing OCR files instead of introducing a
new abstraction layer.

### `perception/plugins/ocr_runtime.py`

The existing adapter owns MNN detector and recognizer sessions, direct MNN
image preprocessing, RapidOCR-compatible postprocessing, startup fallback, and
the ordinary-image path. It reuses RapidOCR's DB postprocessor, CTC decoder,
and perspective-crop helper instead of reimplementing OCR algorithms. The
adapter loads only detector and recognizer models and processes recognition
crops one at a time.

Low precision means FP16 arithmetic and buffers where the Jetson MNN backend
supports it. It does not require converting quantized model weights back to
FP16. INT8 activation quantization is deferred until accuracy can be measured,
because detector quantization can reduce small-text recall.

The existing ORT construction remains in this file as a startup fallback. It
disables the CPU memory arena and memory pattern, uses one thread, and never
creates a classifier session. MNN and ORT are not resident at the same time.

### `perception/plugins/ocr_tiled_strategy.py`

The existing large-image strategy replaces OpenCV full-image decode with
libvips overview and region decode. It continues to own tile planning,
coordinate offsets, duplicate suppression, and reading-order sorting. Tiles
run sequentially through the callback supplied by `ocr_runtime.py`.

### `perception/utils/ocr_model_downloader.py`

The existing downloader accepts an explicit file list, downloads MNN and ORT
bundles, validates non-empty files, and enforces the aggregate 15 MiB model
asset limit.

No additional production Python modules are introduced. The current dynamic
memory rejection module is removed after all call sites and tests are migrated.

## Model Distribution

Model artifacts remain outside Git and are downloaded during the Jetson image
build from the internal HTTP service.

The MNN directory is:

```text
http://172.28.4.81:34567/zengzhitao/embodied-ai/ppocrv6-tiny-mnn/
  det.mnn
  rec.mnn
  keys.txt
```

The existing ONNX directory remains the fallback source:

```text
http://172.28.4.81:34567/zengzhitao/embodied-ai/ppocrv6-tiny/
  det.onnx
  rec.onnx
  keys.txt
```

The downloader preserves these filenames. Installed runtime paths are:

```text
/models/ocr/ppocrv6-tiny-mnn/det.mnn
/models/ocr/ppocrv6-tiny-mnn/rec.mnn
/models/ocr/ppocrv6-tiny-mnn/keys.txt
/models/ocr/ppocrv6-tiny-ort/det.onnx
/models/ocr/ppocrv6-tiny-ort/rec.onnx
/models/ocr/ppocrv6-tiny-ort/keys.txt
```

Downloads are validated by existence, non-zero size, and successful backend
model loading. SHA256 values are not fixed because the internal artifacts can
be updated in place.

The build calculates the aggregate size of all OCR model assets included in the
leaderboard image, including detector, recognizer, and character dictionary
files. When MNN plus ORT exceeds 15 MiB, the leaderboard image contains only
MNN assets and disables ORT fallback. A separate debug build may include ORT.
Classifier models are never included.

## Configuration

The OCR section gains explicit backend and decode settings:

```yaml
ocr:
  enabled: true
  backend: mnn
  fallback_backend: onnxruntime
  model_dir: /models/ocr/ppocrv6-tiny-mnn
  fallback_model_dir: /models/ocr/ppocrv6-tiny-ort
  language: zh
  num_threads: 1
  use_angle_cls: false
  min_interval_ms: 0

  image_pipeline:
    overview_side: 960
    tile_trigger_side: 2400
    tile_size: 1280
    tile_overlap: 128
    max_tiles: 12
    dedup_iou: 0.5
    dedup_text_similarity: 0.8

  mnn:
    precision: low
    memory: low
    power: normal

  onnxruntime:
    cpu_mem_arena: false
    memory_pattern: false
    intra_op_threads: 1
    inter_op_threads: 1
```

`fallback_backend` is set to an empty value in the leaderboard build when ORT
models are excluded by the 15 MiB size gate. Configuration validation fails at
startup if a selected backend does not have its required model files.

The old `max_input_mb`, `max_decode_mb`, and `memory_guard` settings no longer
reject valid images. A compressed-byte limit may still exist at the transport
layer to prevent malformed requests from exhausting request buffers, but it
must be derived from the protocol limit and must return a completed request
error rather than silently dropping the case.

## Image Data Flow

### Ordinary Images

When the source longest side is at most 2400 pixels, libvips decodes directly
to an overview whose longest side is at most 960 pixels. The detector runs once
on that overview. Detected regions are cropped from the bounded decoded image
and recognized sequentially.

### High-Resolution Images

When the source longest side exceeds 2400 pixels:

1. Decode a 960-pixel-longest-side overview and run one global detection pass.
2. Plan a grid of 1280 by 1280 source-coordinate tiles with 128-pixel overlap.
3. Select at most 12 tiles deterministically, preserving corners and edges and
   then distributing remaining tiles over the image.
4. Decode, detect, recognize, and release one tile before decoding the next.
5. Offset tile polygons into source coordinates.
6. Merge overview and tile results using polygon overlap and normalized text
   similarity.
7. Sort merged items using the existing top-to-bottom, left-to-right reading
   order.

The overview protects large text and global layout. Tiles preserve small text
that disappears at overview resolution. Sequential region decode bounds memory
independently of source resolution. The deterministic 12-tile cap bounds
latency; images larger than the covered grid are still processed and return a
valid result, but unsampled interior regions may have lower recall.

## Tensor and Memory Strategy

At most the following request-specific data may be live at once:

- compressed request bytes;
- one 960-side overview or one 1280 by 1280 tile as `uint8`;
- one detector input tensor;
- detector outputs for the current image;
- one recognition crop and one recognizer input tensor;
- accumulated compact OCR item metadata.

The implementation must not retain:

- the full decoded source image for high-resolution inputs;
- all decoded tiles;
- all recognition crops;
- Python Float32 preprocessing copies;
- both MNN and ORT sessions;
- a classifier session or classifier model.

Memory readings are logged but do not gate requests. Normal cleanup relies on
bounded reusable buffers and object lifetime rather than calling garbage
collection after every crop.

## Result Compatibility

The published payload remains compatible with the existing plugin:

```json
{
  "text": "recognized text",
  "items": [
    {"text": "text", "bbox": [x1, y1, x2, y2], "score": 0.9}
  ],
  "timestamp": 0.0,
  "language": "zh"
}
```

Coordinates remain integer, axis-aligned, source-image pixel coordinates. MNN
does not change the ROS topic, MCP tool name, plugin registration, or evaluator
submission format.

## Error Handling and Fallback

### Startup

1. Validate config and model paths.
2. Initialize MNN detector and recognizer.
3. Run a small synthetic warm-up through both models.
4. If initialization or warm-up fails, close MNN completely.
5. Initialize and warm up ORT only when fallback is enabled and installed.
6. If no configured backend succeeds, fail container startup.

The plugin must not log `OCRPlugin loaded` after adapter creation failed.

### Request Processing

- A backend inference failure triggers one rebuild and retry of the same
  backend.
- ORT is not loaded as a per-request fallback because that would create an
  unpredictable memory spike.
- Failure of one tile is logged and other tiles continue.
- If the overview succeeds, successful overview results remain valid even when
  one or more tiles fail.
- If every inference pass fails, return a completed single-case error.
- Corrupt or unsupported images return a completed single-case decode error.
- A valid image with no detected text returns a completed successful response
  with an empty `items` list and explicit protocol completion.

Errors distinguish input decode, model initialization, detection, recognition,
coordinate restoration, and response publication. No request may be left
waiting only because its result is empty or an exception occurred.

## Observability

Startup logs include:

- selected backend and fallback status;
- MNN or ORT version;
- model paths and aggregate model bytes;
- thread count and precision mode;
- warm-up time;
- RSS before and after backend initialization.

Each request emits one summary entry containing:

- source format and dimensions;
- overview and tile dimensions;
- selected and successful tile counts;
- detected, recognized, and deduplicated item counts;
- backend name;
- decode, detection, recognition, merge, and total elapsed time;
- RSS and process peak RSS after completion.

Logs do not include image bytes or the full recognized text.

## Testing

### Unit Tests

Tests cover:

- header probing for JPEG, PNG, BMP, and WebP;
- direct bounded overview decode;
- tile planning, edge coverage, overlap, and deterministic 12-tile selection;
- proof that high-resolution decode does not return a full-source array;
- tile-to-source and overview-to-source coordinate restoration;
- duplicate removal and reading order;
- sequential crop processing;
- empty, corrupt, and unsupported inputs;
- MNN initialization failure and startup-only ORT fallback;
- no classifier model or session creation;
- complete success and error response publication.

### Backend Compatibility Tests

A fixed image corpus runs through MNN and ORT. Tests compare:

- result schema equality;
- detector polygon overlap;
- normalized recognized text;
- item count and empty-result behavior.

The comparison allows backend numeric variation but fails on systematic missing
regions, unreadable output, or coordinate-system mismatch.

### Jetson Stress Tests

The Jetson test set includes ordinary images, 4000 by 3000 scene images, long
images, dense receipts, empty images, corrupt data, and a continuous 250-case
run. It also runs the number of concurrent containers used by the leaderboard.

The stress report records initialization RSS, per-case RSS, process peak RSS,
total container RSS, latency percentiles, completion count, and backend.

### Acceptance Criteria

- All 250 cases produce a completed success or completed single-case error.
- No valid image is rejected because of image dimensions or memory headroom.
- No OOM, exit 137, container restart, or 120-second evaluator wait occurs.
- Aggregate RSS for leaderboard concurrency stays below 80 percent of the 8 GB
  environment during the representative stress run.
- All OCR model assets in the leaderboard image total no more than 15 MiB.
- The external OCR payload and coordinates remain compatible with the current
  plugin.
- Sample-set F1 is not below the current PP-OCRv6 tiny baseline.
- The high-resolution sample set shows no reduction in small-text recall; the
  overview-plus-tile path is expected to improve it.

## Rollout

1. Convert and upload PP-OCRv6 tiny detector and recognizer MNN artifacts.
2. Update the existing model downloader, Jetson Dockerfile, and default config.
3. Replace the existing adapter's inference and image-decode internals in
   `ocr_runtime.py` and `ocr_tiled_strategy.py`.
4. Run focused unit and contract tests.
5. Build once on Jetson and compare MNN with the current ORT baseline for
   memory, latency, and sample accuracy.
6. Submit the MNN commit after the Jetson comparison passes.

The current ORT implementation remains available behind explicit configuration
until the MNN leaderboard run demonstrates equivalent accuracy and stable
memory behavior.
