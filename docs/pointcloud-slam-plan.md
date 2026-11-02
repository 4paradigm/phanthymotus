# Point-cloud SLAM: implementation plan and interface audit

Status: interface proposal, not an integrated or validated SLAM implementation.
Snapshot checked 2026-11-02: main `fc64fbc5541ef3966d81c2f28f9637475a673282`,
point-cloud PR #269 head `9e3e69e8bf0ecdce4108e56cc5d6784da67011db`,
Bumi VO reference `04c4d7a3` on `feat/bumi-visual-odometry-port`.
PR #269 was open and reported non-mergeable at inspection. Recheck all refs and
the robot image before implementation and deployment.

## Interface decisions and outstanding agreements

### Implementation under review

The feature branch merges PR #269 at the snapshot above. For Bumi the proposed
path is the existing VO camera branch's `32FC1` depth image, whose depth source
timestamp and aligned optical frame are already attached before publication.
The companion Driver branch adds its runtime declaration. This branch is enabled
only while the VO card's raw-input control is enabled; it is not an independent
camera and is not a new SLAM input. The older public compressed depth still has
publication timestamps and must not be used for the computational output.

The point-cloud card's opt-in `metric_cloud: true` expects `mode: depth`,
`sensor/depth-f32`, and explicitly configured measured `fx`, `fy`, `cx`, `cy`,
`width`, and `height` for the actual aligned depth image. Enabling it asserts
that the upstream timestamp is capture time in a clock domain compatible with
odometry; a ROS header alone cannot prove that assertion. It publishes
`<input>/pointcloud_metric` as `sensor/pointcloud2` (`PointCloud2`, float32
little-endian XYZ in metres in the input's physical optical frame), while
retaining both existing visualization and summary outputs. It rejects missing
frame/stamp and uncalibrated or different-sized input. Only `32FC1` produces
the computational output. The input header travels through the bounded worker
queue to the cloud unchanged; no callback-arrival timestamp is substituted.

The Core bridge maps `sensor/depth-f32` to Image and `sensor/pointcloud2` to
PointCloud2. Canvas receives a JSON metadata preview (not a geometry stream)
for these new formats; full messages including headers are consumed over ROS.
The existing `sensor/pointcloud` visualization wire format remains unchanged.
The new declarations and this bridge must be deployed together. This is a
proposed integration for review by the point-cloud owner, not an agreed upstream
contract. Real camera timestamp propagation, optics, TF, image build, and actual
Canvas behavior still require target verification. In particular a hardware
clock is mapped with an arrival-derived offset, an approximation rather than
exposure synchronization; never describe this as exact absolute time.

* Exactly two required connections: a pose-bearing odometry stream and a
  timestamped metric point cloud. Discover topics from runtime `info`; no
  robot-specific topic name or connection-order inference.
* Bumi's `motus.odom/1` is a String containing `stamp_ms`, `pose` with its own
  `frame_id`, `child_frame_id`, position, xyzw quaternion and covariance, and
  `vendor.status` and `vendor.segment_id`. Its top-level `frame` describes
  twist, not pose. Reject absent poses, bad coordinates, unknown frames,
  non-tracking, stale samples, and unsupported time domains. Do not interpret
  the Bumi IMU-only twist stream as a pose. Other producers need an explicit
  quality/segment adapter when their schema differs; that adapter is not yet
  specified.
* Prefer a separate, versioned computational `PointCloud2` output with source
  acquisition stamp, physical `frame_id`, metric XYZ and declared calibration
  provenance. This is a proposal for agreement with the point-cloud owner,
  not an existing wire contract. Keep the `sensor/pointcloud` visualization
  output and its `UInt8MultiArray` type intact. PR #269's packet encodes optical
  `(x,y,z)` as `(-z,x,y)` and has no header; it cannot prove temporal or frame
  alignment. Its depth input retains the message in the queue, but `_emit_depth`
  calls `_publish(xyz)` without propagating its header. The 90-degree HFOV
  fallback is not calibrated geometry.
* Core maps `sensor/pointcloud` and `sensor/mapping` to `UInt8MultiArray` and
  its bridge forwards `msg.data`, losing headers. A new format must specify the
  ROS type, discovery, bridge serialization and Canvas behavior together.
  Direct processor-to-processor transport and Canvas display must both be
  tested; merely adding a mapping does not preserve a PointCloud2 header.
* Bumi's public depth header is filled at publication, after compression; it is
  not a capture timestamp. The existing VO input branch maps source camera
  timestamps into its clock domain. Reuse a single shared capture-time mapping
  in the existing camera pipeline, preserve it through compression and point
  cloud processing, and test timestamp provenance and resets. Never synthesize
  a capture time from arrival. Verify alignment against real VO messages.
* A deployment supplies a measured `T_pose_reference_cloud` when frames differ;
  define it to transform cloud coordinates into the pose child frame. Identity
  is valid only when those physical frames are actually identical. Bumi's
  `body_axes` is body-oriented at the camera origin, not a measured body origin.
  Depth intrinsics must match the aligned image and the depth scale must be in
  metres. Calibration and transform provenance travel in configuration or the
  message, never through a third mandatory connection.

## Commit sequence and acceptance gates

1. **Contract and reproducible interface tests.** Define the agreed format and
   quality mapping; add tests for absent pose, invalid quaternion, stale or
   out-of-order data, frame/scale mismatch, timestamp provenance and segment
   transitions. Verify discovery from runtime info. No live integration claim
   until upstream and platform formats agree.
2. **Upstream capture-time and computational cloud changes.** Separate Driver,
   point-cloud and platform commits as ownership requires. Test source stamp
   survives queueing and compression, physical axes and calibration are correct,
   and existing visualization remains compatible. Avoid copying the other
   developer's point-cloud algorithms. Document PR dependencies.
3. **Recording and offline feasibility.** Record only the two agreed streams;
   retain version, calibration, clock and ground-truth markings. Compare
   single-frame and short-submap retrieval, including ESF and Scan Context as
   candidates, followed by coarse registration and ICP/GICP geometric checks.
   Target at least 10 independently labelled revisits and 20 non-revisits;
   top-5 recall at least 8/10 and no accepted false loop in this initial set,
   followed by independent threshold validation. Report insufficient data,
   degeneracy, latency and memory. Freeze the algorithm only after the trial.
4. **Registration, graph and map.** Use mature registration and Pose3 graph
   libraries chosen in step 3; preserve local keyframe clouds and rebuild map
   snapshots from optimized poses. Tests must show a verified loop changes
   historical poses and map, rejects wrong/degenerate matches, and keeps new
   segments separate until independently verified relocalization. Lost or stale
   observations must stop fusion. Publish map pose and map-to-odom separately.
5. **Perception card and display.** Implement bounded queues and map resources,
   isolated worker lifetime, two inputs, separate outputs and concise state.
   Exercise start/stop/restart, missing inputs, worker failure, dropped frames,
   and ensure stopping never stops upstream components. Replay recorded data.
6. **Image and field validation.** Build the full target image; record target
   versions, image digest and rollback. SSH deploy and observe five minutes at
   rest. The operator alone drives at least three marked loops plus non-revisit
   cases. Record accepted edges, before/after trajectories and rebuilt maps,
   physical endpoint and heading errors, resource usage and input-failure
   recovery. Initial targets are 0.15 m and 5 degrees and, where meaningful,
   50% reduction in endpoint error. Report actual evidence and uncertainty.

If geometric feasibility or input provenance fails, preserve the reproducible
failure and stop the dependent integration; never present accumulation as SLAM
or add RGB as a hidden dependency. Recordings, robot credentials and large
artifacts stay outside the repository.
