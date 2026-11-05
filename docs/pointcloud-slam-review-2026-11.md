# Pointcloud and SLAM review follow-up

This follow-up checks the externally supplied 2026-11-04 review against the
independent development branch. It does not claim a target image build or a
robot acceptance. The card remains experimental and does not publish a SLAM
map, accepted loop, or map-to-odom correction.

## Verified and addressed in code

- A fixed child frame blocked Bumi's segment-specific `segment_id + '_base'`
  frame after a reset. An explicitly configured `{segment_id}` placeholder now
  permits the verified naming convention while retaining separate subgraphs.
  The operator must still verify the same physical child reference and
  calibrated extrinsic for each segment; the code does not infer calibration.
- Matching `state/odom` format alone could bind a velocity-only source. Project
  startup now carries the runtime source's odometry declaration to SLAM, which
  requires `motus_slam/1` pose capability. Per-sample quality remains checked.
- Malformed tracking samples were treated like explicit lost events and could
  lock the current segment. A malformed tracking sample now clears pairing
  buffers; a late same-segment sample is dropped without refreshing freshness.
  Explicit lost, unknown source state and untrusted clock/frame still fail
  closed. A valid later tracking sample is needed before fusion can resume.
- `info()` previously expired and cleared pending data as a side effect. Status
  queries now report stale state without mutating the stream; data callbacks
  enforce stale/lost transitions before further processing.
- Display backprojection ignored a declared field of view and could use 90°
  despite a better approximate declaration. It now prefers valid K, then a
  declared FOV, then the clearly marked 90° fallback. FOV remains an
  approximation and is not accepted as measured intrinsics for metric output.
  Invalid K cannot override a usable FOV. A configured stereo calibration is
  rejected for depth/mono, and null pinhole principal points are rejected.
- An explicitly unreadable calibration is now an error rather than silent
  capture-only mode; truly absent stereo calibration still permits calibration
  capture and is reported as `capture_only`.
- Metric pointcloud diagnostics now expose publish and categorized skip counts;
  a declared incompatible depth format is rejected at start. Existing output
  port advertisement remains necessary for runtime discovery, even before the
  first valid message; it is not a promise that data is already flowing.
- A stopped pointcloud worker no longer intentionally starts mono inference
  after a blocked model load returns, and publication checks the stop event.
  Pending mono starts and stop are serialized so a drained request does not
  resurrect a stopped node. This does not cancel an in-progress downloader or
  inference call; a timeout is logged and that limitation remains.
- PointCloudRenderer is registered in the canvas detail and dashboard renderer
  lists as well as monitor mode. This concerns the legacy visualization packet;
  `sensor/pointcloud2` remains a metadata preview in the bridge. No browser or
  robot rendering acceptance is implied.
- Experimental SLAM is now explicitly disabled in `perception/config.yaml`, and
  a missing optional SLAM module does not prevent other Perception plugins from
  loading. Supplying the required clock, calibration and resource settings is
  still an operator/deployment responsibility.

## Confirmed gaps and claims requiring more evidence

- There is no validated runtime geometric loop acceptance, SLAM output port,
  map rendering, automatic graph worker, or published map-to-odom correction.
  GTSAM and Open3D are not installed in the target image by this branch. Host
  tests can run them from an explicit temporary environment, but target ABI,
  image build and CI dependency coverage need separate verification. The
  offline gated graph is an experiment and never accepts real loop edges.
- The capture-time depth and metric pointcloud path remains optional; Bumi's
  recorded metric depth source was limited to about 2 Hz and pointcloud
  decimation remains shared with display. Increasing frequency/point count is
  not an accuracy improvement proven by the existing data and needs target CPU,
  latency, memory and quality measurements before choosing limits.
- Bumi's shipped `body_axes` extrinsic is assumed, not a measured body origin.
  `clock_relation_verified` and `extrinsic_calibrated` are operator assertions,
  not evidence. Sensor epoch/calibration provenance does not yet travel through
  every PointCloud2 frame. A calibrated input and verified clock are required
  before real fusion; do not set the flags merely to make startup succeed.
- Resource limits bound stored frame and point counts, not peak RSS in bytes;
  the current full-store policy refuses new frames rather than silently evicting
  history. Persistence/export, configurable byte budgets and a validated
  retention policy remain open. `optimize_odometry()` is a manual Python
  experiment and must not be returned wholesale through MCP or the browser.
- Actual cold-start Bumi scale, leg motion, exposure timing and end-to-end
  latency were not measured in this follow-up. The claimed ~1.9× display scale
  is a hypothesis about the old 90° fallback, not an observed robot error.
  The recorded 200 ms interpolation bound and prospective motion gates are
  sensitivity settings, not validated universal thresholds. Point-level deskew
  cannot be justified from a snapshot cloud without per-point timing.
- The monitor/UI registration was exercised in an isolated JavaScript module
  selection test, not a full browser/WebGL test. The existing display packet's
  axis mapping is retained; no claim is made that PointCloud2 geometry is
  rendered by the canvas.
- A non-cancellable model download may continue after its last user stops. Stop
  guards and pending-start synchronization limit subsequent work; true
  cancellation requires a cooperative downloader or an isolated worker and is
  not claimed here. DDS-level loss counters, differentiated pending-expiry
  counts, estimator recovery policy and instance configuration remain open.

The independent-label requirement still applies. The existing Bumi recording
does not supply independent revisit and similar-place negative labels. Do not
turn experimental gate thresholds or reciprocal ICP scores into accepted loop
edges until that evaluation and target-device validation are complete.
