# Optional RGB loops and recovery

This extends the existing SLAM card and SensorData path. It does not alter the
RGB-D odometry card or Driver. It uses numerical camera, cloud, pose and mount
data; there are no device names, fixed producer topics or device-specific
coordinate corrections.

## Computation path

1. The gateway asynchronously decodes JPEG in a bounded worker. Numerical image
   association, poses and mounts project cloud points to optical-Z depth; nearest
   measured depth wins and holes remain zero/unknown. A pending matching decode
   can finish within the existing total 20ms optional wait. One projection task,
   bounded image/cloud reservations and a bounded native-input queue prevent
   accumulation. A late/hung job cannot stop mandatory scans.
2. SensorData carries image, projected depth, camera model, numerical camera
   local transform and XYZ scan. No IMU is supplied: its optional local transform
   is explicitly null (ROS geometry defaults otherwise imply identity).
3. Native RTAB-Map extracts features/words and retrieves appearance hypotheses.
   With no pose guess, pinned Memory registration computes a visual transform
   before ICP. Image observations disable `RGBD/ProximityOdomGuess` so ICP cannot
   bypass visual initialization. Matching, covariance and graph acceptance
   thresholds are unchanged. Neighbor scan refinement is disabled on image frames
   to preserve motion covariance: on the deterministic surface, scan refinement
   replaced it with overconfident constraints and rejected correct visual loops.
   Image frames also enable native `RGBD/LoopCovLimited`: non-neighbor information
   diagonals cannot exceed the native historical odometry information maximum.
   This changes graph weighting (the upstream option defaults off), not matching
   thresholds. It prevents early very precise scan constraints from dominating
   later visual motion; it does not repair existing archive edges or guarantee
   all geometrically valid candidates pass global optimization. Scan fallback
   retains the previous scan registration settings.
4. Accepted constraints update the optimized graph and native cloud map. Live
   preview joins native mapGraph/cloud_map at their numerical stamp/frame.
   On close, native global graph optimization persists all connected optimized
   poses, including long-term-memory nodes. Python rebuilds geometry from
   `Admin.opt_ids/opt_poses`, the actual scan local transform (after seven float
   metadata fields), and the final optimized marker. It never combines raw
   `Node.pose` with an incomplete optimized cache. Corruption retains the last
   valid preview. Archive reopening preserves optimized preview bytes.

`Info.loop_closure_id`, `Link.type=1` or `accepted_loops>0` is insufficient alone:
the pinned core can promote a proximity constraint to a global edge. Evidence
must also contain visual inliers, a historical revisit, a persisted edge and
accurate optimized geometry. Type 0 is a neighbor edge; type 2 is local-space
proximity. Some ignored query/rehearsal nodes remain in the raw database without
being members of the connected optimized map.

## External pose loss

`VisualRecovery.h`, installed by `visual_recovery_ros.patch`, runs serially inside
the existing CoreWrapper. It reuses native RTAB-Map localization, RegistrationVis,
ICP, graph validation and OdometryF2M. The fixed core rejects null odometry in
incremental mapping but supports it in localization mode; simply publishing a
null Odometry message was insufficient.

- A new map needs an initial numerical pose. An existing or reopened map accepts
  pose-less observations. Explicit nulls are forwarded immediately; numerical
  pose brackets arriving late retain bounded waiting. If odometry silently stops,
  pending overflow forwards the oldest observation for native estimation.
- Loss switches the existing map to localization, disables proximity recovery
  and retains its keyframes/features/database. A native global visual match,
  adequate native inliers, valid covariance and a native localization pose are
  required. Appearance alone cannot authorize a recovery.
- The same verified observation is attached to the original graph using a
  checked native edge. The mapping-mode transition's graph clear is repaired by
  explicitly restoring original optimized poses/constraints. Failure restores
  that graph and does not add an unlocated global scan.
- OdometryF2M then estimates local motion, anchored to the verified map pose.
  Every later pose needs its own valid motion/inlier/covariance result. Lost
  motion or images return to searching; the last pose is never frozen as tracking.
- Returning external odometry is aligned using a simultaneous valid visual pose.
  A continuous numerical origin jump is checked against independent VO and native
  uncertainty/error bounds; the last reliable paired VO/native anchor survives
  an external-only image gap. Labels never force origin resets.
- `start_mapping` with `map_id` and optional `revision` privately copies a verified
  native database under the existing disk budget. No new external origin is
  assumed equal to its saved map. The resumed session saves the next revision of
  that map ID, including if its display name was renamed while mapping.

Native runtime capability marker `motus_visual_recovery/1` is checked before a
segment is registered. Unpatched installations fail explicitly. `native_recovery`
reports external/searching/localized/visual_tracking and visual inliers. Input
health remains independent: external input may still say waiting while native
VO tracks. These are the last native result, not a new motion estimate when the
depth/cloud source stops: absent Info cannot authorize a current pose or a new
global node. Legacy `segments` counts input segment labels; `segment_lifecycle`
and native artifacts identify actual native databases/processes.

Different RGB/cloud numerical times without relative poses cannot currently be
projected during loss. Coincident captures work; this is an explicit arithmetic
limitation, not a freshness, clock, source or skew policy. Sparse depth, texture,
overlap, ambiguous geometry and drift still limit recovery. Unknown external
origin changes with no independent observation cannot be inferred from labels.

## Reproduction

Use the core/ROS pins and patches in `Dockerfile.jetson`:
core `df6300e0ba3e90058f90b09c4d646279366d3516`, ROS
`2a31e433d2b072795ae90a62abcdd0c00691e04d`. The native core regression requires
RTAB-Map, Eigen, OpenCV and PCL development headers; compile against that exact
core (do not mix a system 0.23 header/ABI with pinned 0.22 libraries).

```bash
python perception/tools/slam_visual_scene.py /tmp/slam-scene
g++ -O2 -std=c++17 perception/tests/native/test_visual_recovery.cpp \
  -Iperception/native/rtabmap -I/usr/include/eigen3 \
  -I/usr/local/include/rtabmap-0.22 $(pkg-config --cflags opencv4 pcl_common-1.12) \
  -L/usr/local/lib -lrtabmap_core -lrtabmap_utilite \
  $(pkg-config --libs opencv4) -o /tmp/test_visual_recovery
LD_LIBRARY_PATH=/usr/local/lib /tmp/test_visual_recovery /tmp/slam-scene /tmp/rgb rgb
LD_LIBRARY_PATH=/usr/local/lib /tmp/test_visual_recovery /tmp/slam-scene /tmp/no-rgb no_rgb
LD_LIBRARY_PATH=/usr/local/lib /tmp/test_visual_recovery /tmp/slam-scene /tmp/recovery recovery
python perception/tools/evaluate_slam_visual_replay.py --rgb /tmp/rgb \
  --no-rgb /tmp/no-rgb --recovery /tmp/recovery --output /tmp/core-evaluation.json

# Source the pinned patched ROS overlay first; use new output paths on each run.
python perception/tools/slam_visual_scene.py /tmp/origin-scene --origin 7 -3 .5
ROS_DOMAIN_ID=193 ROS_LOCALHOST_ONLY=1 python3 perception/tools/rtabmap_visual_replay.py \
  /tmp/origin-scene /tmp/ros-replay
python perception/tools/evaluate_slam_visual_replay.py --ros /tmp/ros-replay \
  --output /tmp/ros-evaluation.json
python perception/tools/slam_visual_scene.py /tmp/resume-scene --origin 7 -3 .5 --resume
ROS_DOMAIN_ID=194 ROS_LOCALHOST_ONLY=1 python3 perception/tools/rtabmap_visual_replay.py \
  /tmp/resume-scene /tmp/ros-resume --resume-run /tmp/ros-replay
python perception/tools/evaluate_slam_visual_replay.py --ros /tmp/ros-resume \
  --previous /tmp/ros-replay --output /tmp/resume-evaluation.json
AGENT_CORE_URL=https://127.0.0.1:1 python -m pytest -q perception/tests
```

The scene fixes texture/geometry generation and contains departure/revisit,
appearance-identical incompatible-depth and novel-place negatives, absent external
poses, continued visual motion, changed external origins, corrupt RGB, repeated
loss and recovery. Native feature extraction/optimization and OS scheduling can
vary the exact accepted frame; the evaluator checks geometry and graph relations,
not an exact count. Core evaluation compares the same supplied drifted odometry
with/without RGB. Surface residuals measure map misalignment against the known
corrugated surface, rather than declaring a cloud visually correct by inspection.

The ROS tool saves Info, mapGraph, SensorData image presence, process identity,
database and archive. `--resume-run PREVIOUS_OUTPUT` loads its saved database;
the supplied scene must describe observations in that original world. Evaluate
with `--previous PREVIOUS_OUTPUT` to require a historical match, preserved origin
and new tracked graph nodes. Fresh outputs must be evaluated; replay exit status
alone is not acceptance. See [2026-11-17 evidence](slam-visual-recovery-validation-20261117.md)
for actual results and remaining limits.
