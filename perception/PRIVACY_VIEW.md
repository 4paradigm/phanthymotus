# Privacy View

`privacy_view` is a generic, multi-instance Perception **processor**. Connect a
camera JPEG output to it, then connect its JPEG output to a display or downstream
Card. It uses the existing SCRFD model registry and shared ORT worker. It adds no
driver, model family, Python dependency, container or production listening port.

`face_recognition` answers who a person is and maintains a roster. Privacy View
redacts detected face regions without matching identities, aligning faces,
running embeddings, or writing face images/features. Image quality monitoring
answers whether an image is usable; it does not provide this redacted image.

## Privacy boundary

**Opaque black rectangles are the default.** Pixelation and Gaussian blur are
weaker transformations and must not be treated as having the same privacy
strength. All modes depend on detection: missed faces remain visible. A
successful zero-face result means the detector found no qualifying face, not
that no face exists. The status always declares `privacy_guaranteed: false`.

This protects the Card's output, not access to the original camera topic or the
whole robot. Route downstream consumers to the privacy output explicitly. Do
not use detection-based redaction where an absolute anonymity guarantee is
required. Validate recall in the actual camera position, lighting and scene.

Every output JPEG is decoded, detected and redacted from the **same** source
JPEG. A failed decode, malformed/incomplete result, invalid box, mismatch,
expired frame, inference error or encode failure produces **no image**. There
is no exception handler that returns the input JPEG. An empty successful result
is re-encoded through the same validated output path.

SCRFD returns every post-threshold, post-NMS detection, with `max_faces=0`.
Privacy View does not select a subject, apply the recognition quality gate, or
truncate a crowd. Broken/nonfinite detector heads fail the frame. The finite
anchor grid, NMS, resolution and threshold still limit detection recall.

## Canvas and MCP

The tool declares `type: processor`, `multiInstance: true`, one `image/jpeg`
input, then two outputs in this stable order:

1. `image/jpeg`: `sensor_msgs/msg/CompressedImage` with `format="jpeg"`.
2. `data/json`: `std_msgs/msg/String`, schema `motus.privacy_view/1`.

Use the existing Canvas drag/drop, connection, instance configuration and
project start/stop flow. Core supplies `instance_id`; like sibling Cards it is
not a user-entered `inputSchema` field. No Core or web source changes are needed.

```json
{"action":"config","instance_id":"front_privacy","mode":"opaque","bbox_padding":0.25}
{"action":"start","instance_id":"front_privacy","input_topic":"/robot/camera/rgb"}
{"action":"info","instance_id":"front_privacy"}
{"action":"stop","instance_id":"front_privacy"}
```

`input_topics` is also accepted when it contains exactly one topic. Conflicting
or multiple inputs are rejected. Stop before changing an instance's camera.
Without an `instance_id`, start/info address `_default`, config updates shared
defaults and instances without overrides, and stop cancels all instances.
Per-instance configuration is removed by stop; the saved Canvas configuration
is replayed by Core before the next start.

For input `T` and instance key `K`, the output prefix is
`T/privacy_view/instance_<first 16 hex characters of SHA256(K)>`. The suffixes
are `/image` and `/status`. Always use the concrete `info.topic_out` values;
two instances on the same camera have separate outputs. `info` with an input
topic resolves ports even before model loading. Input/output image and status
QoS is BEST_EFFORT, KEEP_LAST depth 1, VOLATILE (`utils.qos.CAMERA_QOS`).

## Configuration

Unknown fields, booleans masquerading as numbers, nonfinite numbers and invalid
ranges are rejected atomically. Configuration is frozen per processing task.
An actual change invalidates queued/in-flight tasks; identical replay is a no-op.
Subsequent source frames use the new settings without rebuilding the model.

| Field | Default | Valid values |
|---|---:|---|
| `mode` | `opaque` | `opaque`, `pixelate`, `gaussian` |
| `detect_fps` | 10 | 0.1–60, upper bound rather than guaranteed throughput |
| `confidence_threshold` | 0.5 | 0.1–0.9; raising it increases missed-face risk |
| `bbox_padding` | 0.25 | 0–1, added to **each side** as a fraction of box size |
| `pixelate_strength` | 16 | integer 4–128, downsampling factor |
| `gaussian_strength` | 51 | odd integer 3–301, OpenCV kernel side |
| `inference_timeout_s` | 2 | 0.05–10, ready-worker inference and shared-lock wait |
| `max_frame_age_s` | 2 | 0.1–10, source and receive age gates |

Deployment-only model settings are in `plugins.privacy_view` in `config.yaml`:
`model` (existing default `scrfd25g_edgeface_s`), `model_dir` (existing registry
resolution), `device` (default `cpu`) and `num_threads` (default 2, integer 1–16). They are not
per-instance model knobs. Existing pinned model bundles are verified through
the official downloader; fetching reports progress through `info.desc`.
The existing pair can be downloaded/cached, but the identity session is loaded
lazily only when a recognition consumer uses it.

## Frames, status and recovery

Sources must publish positive, strictly increasing epoch-based `header.stamp`
and a string `frame_id`. An empty frame ID is preserved as unknown, as used
by existing camera publishers; no coordinate frame is invented. Zero/simulation-relative, stale, backwards or more
than 100 ms future stamps are rejected. The Card does not substitute its own
processing time for camera exposure time. Camera/host clocks must agree;
normalize device clocks in the existing source driver before connecting it.
Restart the instance after a source timestamp sequence reset.

The original stamp and frame ID are copied to the output. EXIF auto-rotation
is disabled in both decoders so the same coordinates refer to the same pixels.
JPEGs are capped at 8 MiB, 8192 pixels on either side and 16,777,216 pixels
total, checked before decoded pixel allocation. No input object is modified.
Boxes are validated, expanded, rounded outwards and clipped. Every overlapping
region is transformed; tiny ROIs fall back to opaque masking.

Each instance has a one-slot latest-frame buffer, one processing worker and
independent state, metrics, cancellation and output topics. High input rates
overwrite pending frames rather than accumulate history. The cap is 32 live
instances and 32 saved instance overrides. Raw frames are never written to disk
or included in status/log payloads.

Status is published after results and every 0.5 seconds, including during model
loads and inference. It includes:

- Lifecycle `state` (`idle`, `loading`, `running`, `error`, `stopping`) and error.
- Instance `generation`, `processing_generation`, effective configuration.
- `model_ready`, loading `desc`, `image_published`, `fresh`, source/receive/output
  ages. `image_published` refers to that status event, not a cached image.
- Per-frame source stamp, frame ID, local sequence/token, detection count and
  completeness on successful processing. Fault statuses never imply zero faces.
- Received, processed, published, rejected, error, discarded-generation and
  queue-overwrite counters; latest measured decode/detect/redact/encode and
  receive-to-publish latency. Counters are not a DDS delivery guarantee.

Camera silence eventually reports `camera_timeout`; no cached image is replayed.
Consumers should observe status/freshness alongside JPEG. The generic web image
viewer has its own stale display interval; it may retain previously delivered
pixels while this Card publishes no new image during a fault.

Stop closes the publication gate first, then cancels work. Publication and
configuration commit use the same gate, so old tasks cannot publish after stop
acknowledgement or across a configuration generation. DDS messages already
published before that gate closed can still arrive in a subscriber's transport.
An uncancellable existing model download/worker rebuild may finish later:
`stopping` remains observable and restart is refused until its tracked thread
returns, releases its lease and removes/destroys its ROS node. No replacement
thread is started to hide the pending cleanup. Shutdown uses the same path.

Face service keys are derived from normalized engine configuration. Equal
configuration acquires a reference-counted lease, avoiding replacement and
duplicate sessions. Different engines have different keys. Only the last lease
closes a service. Recognition retains its existing idle engine cache for photo
and roster actions; stopping a privacy stream releases only that stream's lease.
Worker startup/recovery uses the existing worker's 30-second handshake and
90-second per-service build bounds; it can exceed the ready-worker inference
timeout. The publication gate, source age checks and independent status timer
remain active throughout recovery. Worker death invalidates readiness and
rebuilds desired services. Termination closes both IPC pipe endpoints and reaps
the JPEG queue feeder, including a transfer blocked on a large frame. Failed rebuild
specifications are retained for retry. A timed-out running inference can restart
the shared worker, temporarily interrupting its other tenants; they recover via
the existing worker protocol. No raw-frame fallback is used during recovery.

## Reproducible checks

Normal offline tests use real pixel algorithms and synthetic images, with ROS
transport and detector doubles clearly confined to `perception/tests`:

```bash
python -m pytest perception/tests/test_privacy_runtime.py \
  perception/tests/test_privacy_view.py perception/tests/test_privacy_face_service.py -q
python -m pytest perception/tests -q
```

The opt-in suite runs separately because normal tests install ROS stubs. It needs
real rclpy, OpenCV, Pillow, ORT, pytest and, for the Core/browser scenario, the
normal Core dependencies plus Playwright/Chromium and psutil. These are test-host
requirements; production gains no dependency.

```bash
export PYTHONPATH=perception PRIVACY_VIEW_LIVE=1
export PRIVACY_TEST_MODELS="$PWD/.local-test/models"
export PRIVACY_TEST_EVIDENCE="$PWD/.local-test/evidence"
mkdir -p "$PRIVACY_TEST_EVIDENCE"
# Set a dedicated local test DDS domain, ROS_LOG_DIR, and an official-style
# loopback-only FASTRTPS_DEFAULT_PROFILES_FILE before running.
python -m pytest perception/integration_tests/test_privacy_view_live.py -v -s
```

The harness relocates only the `/models` path guard to its isolated cache in
parent/child; official hashes, SCRFD, recognition sessions, worker IPC, MCP and
DDS are real. Core runs its unchanged app/routes and DDS bridge in a separate
process with normal token auth. Host Docker/network/peer/agent bootstrap is
disabled. This is local Core integration, not a deployed production or robot
acceptance result. The synthetic drawing is not a detector recall benchmark.

The existing `perception/Dockerfile.jetson` copies `plugins/` and `utils/` and
already includes all runtime dependencies. Keep the official build route:
`deploy/build_perception.sh --jp-version 5.11` or `--jp-version 6.1`. The script
can push/register if registry credentials exist; use an isolated build checkout
with no registry credentials for a local-only build. Target ARM64 image builds,
Jetson latency/memory, real camera clock compatibility, side faces, occlusion,
lighting and actual missed faces still require target-environment acceptance.
