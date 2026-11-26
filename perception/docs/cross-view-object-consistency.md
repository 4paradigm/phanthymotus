# Cross-view object consistency

`cross_view_object_consistency` is a read-only, single-instance Perception Processor.
It compares contemporaneous visible-surface observations in a common frame.
The output is `std_msgs/String` (`data/json`) on
`/perception/cross_view_object_consistency/state`, published at 5 Hz.
MCP provides `config`, `start`, `stop` and `info`.

## Input contracts and upstream investigation

- PiPER driver [#382](https://github.com/4paradigm/phanthymotus-driver/pull/382)
  introduced an optional On camera sensor for one selected V4L2 color node.
  The dual-camera Driver branch adds On camera B to the same service with
  two serial-bound color devices and independently opened capture sessions.
  `motus.camera/1` can describe optics. Valid installed-device intrinsics,
  registered depth, and workbench extrinsics still need field evidence.
  [Driver #530](https://github.com/4paradigm/phanthymotus-driver/pull/530)
  calibrates gripper clearance using arm feedback.
- VOP streams `objects[]` with `name`, `confidence` and an optional pixel
  `bbox=[x1,y1,x2,y2]`. Keep `publish_bbox` on. This change adds
  `source={frame_id,stamp_ns}` and `source_size=[width,height]` to streaming
  results. Existing `timestamp` remains publication time. The visual_depth
  stream now carries its input JPEG header. It publishes a zlib stream of
  640×480 little-endian uint16 millimetres, zero for invalid samples, labeled
  `16UC1; compressedDepth zlib` in `CompressedImage.format`. Model depth
  needs independent validation for metric use. One-shot inference has no
  capture provenance and cannot enter a synchronized pair.
- [#407](https://github.com/4paradigm/phanthymotus/pull/407) publishes
  source-resolution PNG semantic class masks, pixel-count JSON and optional
  overlays. Use VOP with this Card. Class masks and pixel counts lack the
  per-instance boxes required by its association algorithm. [#425](https://github.com/4paradigm/phanthymotus/pull/425)
  combines a single view’s semantic masks and registered depth into visible
  surfaces, with unknown instance attribution for semantic-only input.
- [#374](https://github.com/4paradigm/phanthymotus/pull/374) verifies an object's
  before/after progress toward a goal region. [#393](https://github.com/4paradigm/phanthymotus/pull/393)
  detects before/after changes in an image ROI. This Card compares synchronized
  observations from two optical frames and reports metric residuals and visibility.

## Dashboard configuration

Add two existing RGB camera cards and one VOP instance for each camera. Connect each
camera to its VOP as well as to the corresponding A/B RGB port. Connect each
VOP JSON output to the corresponding A/B Objects port. To check RGB provenance
and timing alone, omit the Objects ports and their topic bindings. If verified depth is
available, wire each corresponding Depth port. Route the JSON output to a
Dashboard inspection panel. The canvas passes `input_topics` as a list. The
Card checks the connected set against the exact configured topic names, so equal
formats and UI connection order do not determine camera identity.

Set the Card's **rig_json** string to a JSON object with these fields.

```json
{
  "camera_ids": ["bench_a", "bench_b"],
  "device_ids": ["REPLACE_WITH_CAMERA_A_DEVICE_ID", "REPLACE_WITH_CAMERA_B_DEVICE_ID"],
  "topics": {
    "a": {"rgb": "/bench/a/image", "objects": "/bench/a/image/objects"},
    "b": {"rgb": "/bench/b/image", "objects": "/bench/b/image/objects"}
  },
  "max_time_offset_ms": 40,
  "max_age_ms": 1000,
  "buffer_size": 8,
  "tolerance_m": 0.06,
  "match_gate_m": 0.25,
  "ambiguity_m": 0.03,
  "min_confidence": 0.35
}
```

For geometry, add `depth` to both topic maps and add a `calibration` map keyed by
camera ID. Each record includes `camera_id`, the same `world_frame`, `size=[width,height]`,
`K` (row-major rectified 3×3), `rectified:true`, `depth_registered:true`,
`direction:"camera_to_world"`, `T_world_camera` (row-major 4×4, translation
in metres), and `error_m` (measured position error allowance). The transform
maps optical coordinates (right, down, forward) to the chosen common frame.
Depth values must be optical-axis Z, aligned and full-frame resampled with the
same pixel mapping as the RGB. Verify these properties before declaring them.

Read the fingerprints from Driver MCP `info().device_id`. The capture tag
`motus.capture/1|camera_id|optical_frame_id|capture_session_id|frame_sequence|host_read|device_id`
travels atomically with JPEG, and its full value and nanosecond stamp must
survive VOP and depth. The capture sessions are independent. A new session
observed on either RGB input clears both views' buffered evidence. Delayed
retired frames, wrong devices, wrong camera IDs, duplicates and stale data
cannot populate a match. An older deployment with explicit
`<camera_id>@<session_id>` tags can keep its existing `session_id` config and
omit `device_ids`.

For the provenance-enabled Driver, include the corresponding `device_id` and
the same nonempty `calibration_version` in both calibration records. Stop the
Card before changing the rig or calibration. Set `time_error_ms` only after
measuring a conservative bound on cross-camera clock offset and differential
host-read timing uncertainty at the intended frame rates. The Card requires
observed skew plus this bound to fit inside `max_time_offset_ms` before any
calibrated geometry is evaluated. Host-read timing alone cannot establish
hardware synchronization.

The two PiPER camera cards send color only. The deployed depth source must
explicitly register the maps to RGB, use the declared wire format, and
propagate the originating RGB frame tag and stamp. Validate any monocular
model's metric and spatial accuracy before supplying it as qualified depth.

## Reading diagnostics

- RGB and detections join on exact source header per camera. The two resulting
  views must be within the configured skew. Invalid, duplicate, reordered,
  pre-start, expired and other-session samples cannot update the match buffers.
  `timestamp` is the later source time in Unix seconds for a joined pair.
  Before a pair is available it is the report time.
- Level 0 reports source identity, session, observed timing and freshness.
  Level 1 reports a 2D epipolar candidate count with valid rectified calibration
  and a qualified timing bound, when depth is unavailable. Level 2 adds the
  existing 3D comparison on two valid, registered depth maps. Geometry reports
  the calibration version when using provenance-enabled inputs. The epipolar
  count describes compatible pairs and cannot establish identity. The Card keeps a bounded number of
  depth maps and detection lists and retains only the headers of RGB observations.
- Candidate edges require both compatible class and spatial proximity.
  Mutually preferred edges with a clear margin are reported. Close alternatives
  and assignment conflicts stay ambiguous. Target IDs are frame-local indexes.
  Geometry alone cannot recover identity after physically indistinguishable
  objects exchange places. `confidence` is a heuristic in [0,1].
- The residual is the largest reported pair distance in metres, when available.
  Sampling uses a small robust depth patch near each box center. A mixed patch
  can indicate partial occlusion. An observed nearer surface at the other
  camera's reprojection is possible occlusion evidence. A prediction outside
  the image is a field-of-view diagnostic. A large unexplained residual may come
  from bias or a physically wrong but mathematically valid transform.
- `matched_targets` carries frame-local indexes, image points and residuals.
  `unmatched_targets` carries per-view causes and projected pixels when available.

## Workstation acceptance

1. Bind the two selected color nodes by serial, give them distinct ROS topics,
   verify real capture frame stamps and establish a session-aware capture source.
   Stop and restart either source and verify that old-session messages are rejected.
2. Calibrate both RGB optical frames at the configured resolution. Rectify both
   streams and estimate camera-to-world poses using a common calibration target.
   Verify direction and scale on held-out 3D points throughout the shared field
   of view. Register and validate both depth maps at surveyed distances and edges.
   Record timing skew and spatial error to configure skew, allowance and tolerance.
3. Start the Card and test the Dashboard connections in both RGB-only and
   calibrated configurations. Inspect output during and after a disconnect,
   dropped detection, intentional skew, view exit and partial/full occlusion.
4. Record both RGB and VOP topics and any registered depth topics while
   the clocks and session tags are intact. Use
   `ros2 bag record --storage sqlite3 -o piper-cross-view <topic-a-rgb> <topic-a-objects> <topic-b-rgb> <topic-b-objects>`
   and append the verified depth topics if present. Save the original rig JSON
   and survey records with the bag. Include both views of a common flashing
   timing reference, stationary surveyed targets, occlusions, and disconnects.
5. Point a Card running in a separate Perception service and isolated ROS domain at
   `/cross_view_replay/{a,b}/{rgb,objects,depth}`. With `PYTHONPATH=perception`
   and the ROS 2 Python bag plugin, run
   `python perception/tools/replay_cross_view.py <bag-directory> <original-rig.json>`.
   Start the Card shortly before replay. The tool rebases stamps together and
   preserves the original relative timing, capture provenance and dropouts.
   It creates new replay session IDs and leaves the source bag untouched.
   Report measured synchronization error, false positives and negatives,
   ambiguity coverage and metric residuals against the held-out survey.

From the repository root, set `PYTHONPATH=perception` and run
`python perception/tests/test_cross_view_object_consistency.py`. Add
`--benchmark` for generated matching accuracy, residual, compute latency and
traced peak allocations. On a ROS 2 host with the existing Perception dependencies,
run `python perception/tests/cross_view_ros_integration.py` for virtual
publishers, MCP over loopback HTTP, DDS output and repeated startup/teardown. The existing
`perception/Dockerfile.jetson` copies the plugin and utility directories and
needs no Card-specific dependency or image layer.


## Local image and integration reproduction

```sh
docker buildx build --platform linux/arm64 --load --build-arg JP_VERSION=511 \
  -f perception/Dockerfile.jetson -t motus-cross-view:jp511-dockerfile .
docker run --rm --network none --platform linux/arm64 --entrypoint bash \
  -v "$PWD/perception/tests:/tests:ro" motus-cross-view:jp511-dockerfile \
  -lc 'PYTHONPATH=/work python3 /tests/cross_view_ros_integration.py'
```

For browser acceptance, run a local Core and
`cross_view_dashboard_fixture.py` on an isolated Docker network and ROS
domain. Run `cross_view_dashboard_browser.mjs` with the browser-reachable Core
URL, the Core-reachable fixture MCP URL, a local Chromium executable and a PNG
output path. The test stores a fixture-only canvas in that Core. The virtual
VOP outputs are labelled JSON fixtures and the test uses real MCP and DDS.
