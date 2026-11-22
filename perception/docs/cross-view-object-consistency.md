# Cross-view object consistency

`cross_view_object_consistency` is a read-only, single-instance Perception Processor.
It compares contemporaneous visible-surface observations in a common frame.
The output is `std_msgs/String` (`data/json`) on
`/perception/cross_view_object_consistency/state`, published at 5 Hz.
MCP provides `config`, `start`, `stop` and `info`.

## Input contracts and upstream investigation

- PiPER driver [#382](https://github.com/4paradigm/phanthymotus-driver/pull/382) exposes one
  optional On camera sensor for the selected V4L2 color node per service. The two
  `/dev/piper-camera-{1,2}` names are device choices. The reviewed PiPER camera
  code sends JPEG in `sensor_msgs/CompressedImage`. Its ROS stamp is sourced
  from the host-side capture pipeline and the optical frame name is constant.
  Establish two independent color sources with unique topics on the workstation.
  Verify the physical device/serial and the acquisition clock of each. An offline
  placeholder with a fresh publish timestamp must be excluded upstream.
- `motus.camera/1` can carry lens identity and intrinsics through the canvas.
  A valid `K` and image size for each physical PiPER camera, registered sensor
  depth, inter-camera extrinsics, and a common capture session were unavailable
  in the inspected PiPER wiring. [Driver #530](https://github.com/4paradigm/phanthymotus-driver/pull/530)
  calibrates gripper clearance against a table using arm feedback.
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
VOP JSON output to the corresponding A/B Objects port. If verified depth is
available, wire each corresponding Depth port. Route the JSON output to a
Dashboard inspection panel. The canvas passes `input_topics` as a list. The
Card checks the connected set against the exact configured topic names, so equal
formats and UI connection order do not determine camera identity.

Set the Card's **rig_json** string to a JSON object with these fields.

```json
{
  "camera_ids": ["bench_a", "bench_b"],
  "session_id": "REPLACE_WITH_CAPTURE_SESSION",
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

The workstation acquisition publisher or existing capture middleware must set
`header.frame_id` to `<camera_id>@<session_id>` on the input RGB and preserve
the original shared-clock Unix acquisition stamp. Use a fresh shared session ID
when either camera restarts, changes source, or begins replay, and restart the
Card with the new ID. Do not stamp at processing or reception time.
The stock PiPER RGB output inspected here cannot supply this tag or identify
offline placeholders in the image payload. A session-aware workstation source
is an integration prerequisite for positive results. An unmodified RGB stream
can exercise the port wiring and unverified-provenance diagnosis.

## Reading diagnostics

- RGB and detections join on exact source header per camera. The two resulting
  views must be within the configured skew. Invalid, duplicate, reordered,
  pre-start, expired and other-session samples cannot update the match buffers.
  `timestamp` is the later source time in Unix seconds for a joined pair.
  Before a pair is available it is the report time.
- Stream freshness and timing diagnostics can run without metric calibration.
  Metric agreement, disagreement, field-of-view and occlusion evidence need
  validated calibration and aligned depth. The Card keeps a bounded number of
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
4. Record a shared-clock, labelled two-camera session with RGB, detections,
   depth and reference geometry. To replay with the live-clock interface,
   preserve inter-frame timing while rebasing all source stamps together to the
   current epoch and issuing a fresh session ID. Report recall, wrong matches
   among predicted pairs, ambiguity coverage, residual quantiles and CPU/RSS
   separately for the deployment data. Synthetic benchmarks measure only their
   generated fixtures.

From the repository root, set `PYTHONPATH=perception` and run
`python perception/tests/test_cross_view_object_consistency.py`. Add
`--benchmark` for generated matching accuracy, residual, compute latency and
traced peak allocations. On a ROS 2 host with the existing Perception dependencies,
run `python perception/tests/cross_view_ros_integration.py` for virtual
publishers, MCP over loopback HTTP, DDS output and repeated startup/teardown. The existing
`perception/Dockerfile.jetson` copies the plugin and utility directories and
needs no Card-specific dependency or image layer.
