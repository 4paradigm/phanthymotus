# OCR-Only Integration

## Goal

Extend `feat/zengzhitao` with an enabled, on-device OCR capability for the
Jetson leaderboard while preserving the existing ASR benchmark behavior.

The OCR implementation must detect text boxes and recognize Chinese, English,
and numeric text. Its complete model bundle must remain below 15 MiB, GPU usage
must remain below 10%, and no model or other file larger than 1 MiB may be
committed to Git.

## Integration Boundary

Use the existing feature head `02d1ddc` as the code baseline. Do not merge
`origin/main` in this change, and do not modify ASR selection, model download,
VAD/KWS behavior, or restart scripts.

Port only the OCR capability from `origin/feat/wanglimin-test`:

- Keep the `ocr` MCP tool and its `start`, `stop`, `info`, and `config` actions.
- Keep ROS2 `sensor_msgs/CompressedImage` input and JSON `std_msgs/String`
  output.
- Do not port `asr_local.py`, the OCR branch's unrelated plugin enable/disable
  changes, or its ASR experiments.
- Keep cloud OCR adapters optional, but use a local RapidOCR adapter by default.

## OCR Contract

OCR is enabled by default. `ocr start` requires an `input_topic`, subscribes to
JPEG-compatible compressed image messages, and publishes to
`<input_topic>/ocr`.

Each output message has this shape:

```json
{
  "text": "full text in reading order",
  "items": [
    {
      "text": "recognized text line",
      "bbox": [10, 20, 110, 50],
      "score": 0.98
    }
  ],
  "timestamp": 0.0,
  "language": "zh"
}
```

Bounding boxes use original-image pixel coordinates in `[x1, y1, x2, y2]`
order. Invalid images and inference failures publish the same structure with an
empty result plus an `error` field instead of terminating the worker.

## Local Model

Use a PP-OCRv6 tiny ONNX pipeline through RapidOCR on CPU. The official
RapidOCR model repository provides the required ONNX artifacts directly, so no
Paddle-to-ONNX conversion is required:

- PP-OCRv6 tiny text detector (`1,829,618` bytes).
- PP-OCRv6 tiny Chinese-English text recognizer (`4,489,813` bytes).
- Mobile text-line orientation classifier.
- Chinese-English recognition dictionary.

The complete bundle is approximately 6.61 MiB, including the 585,532-byte
classifier and 27,156-byte dictionary.

The internal bundle uses stable normalized names:

```text
det.onnx
rec.onnx
cls.onnx
keys.txt
```

The files are hosted under:

```text
http://172.28.4.81:34567/zengzhitao/embodied-ai/ocr/ppocrv6-tiny/
```

The Jetson image downloads them at build time into:

```text
/models/ocr/ppocrv6-tiny/
```

The downloader uses temporary files and atomic replacement, rejects empty
files, and fails the build if the total model bundle exceeds 15 MiB. It does not
pin SHA256 values because the internal model bundle may be updated in place.

RapidOCR is version-pinned for Python 3.8 compatibility. Any default ONNX model
assets bundled by its wheel are removed in the same Docker layer that installs
the package, so only the leaderboard model bundle remains in the final image.

## Image Transport

Keep the existing ROS2 middleware configuration unchanged. Standard Fast DDS
fragmentation is used for compressed image messages. Do not activate the OCR
branch's process-wide custom UDP profile because it would also alter ASR, TTS,
and every other ROS2 topic in the perception process. Add a transport override
only after Jetson testing demonstrates a reproducible image transport failure.

## Configuration

Add this default configuration without changing existing ASR defaults:

```yaml
ocr:
  enabled: true
  provider: rapidocr
  model_dir: /models/ocr/ppocrv6-tiny
  language: zh
  use_angle_cls: true
  num_threads: 2
```

Cloud providers remain available only through explicit configuration and are
not used by the leaderboard path.

## Verification

Automated tests cover:

- OCR plugin registration and default enablement.
- MCP action schema and ROS2 topic naming.
- RapidOCR result normalization into text, boxes, and scores.
- Invalid-image and inference-error output behavior.
- Model downloader success, partial-download cleanup, non-empty files, and the
  15 MiB aggregate limit.
- Docker model download, dependency pinning, bundled-model removal, and absence
  of a process-wide Fast DDS override.
- Existing ASR runtime and benchmark contract regression tests remain green
  without changing ASR production code.

Local verification includes unit tests, Python compilation, diff checks, and a
Git large-file audit. Final acceptance requires a Jetson no-cache Docker build,
container startup, one JPEG OCR round trip, model-size measurement, latency
measurement, and confirmation that GPU utilization remains below 10%.

## External Prerequisite

Before the Jetson Docker build can pass, the four normalized OCR model files
must be prepared from the official RapidOCR PP-OCRv6 tiny artifacts and uploaded
to the internal HTTP directory above. They are deployment artifacts and remain
outside Git history.
