# SLAM visual recovery implementation plan

**Goal:** Extend PR371's existing optional RGB SLAM path to verify visual loops, preserve optimized saved geometry, and recover mapping without external poses.

**Authority:** User's 2026-11-17 task in this session. Work is explicitly authorized; continue inline in the supplied linked worktree. Do not change PR395, deploy, restart robot services or command motion.

**Architecture:** Reuse pinned RTAB-Map's localization, visual registration plus ICP, and native OdometryF2M. Keep the existing gateway, map process, database, input contract and preview. Extend the native SensorData path rather than create another visual SLAM implementation. External numeric poses remain accepted regardless of labels. Unlocalized frames cannot grow the global map.

**Tech stack:** Python gateway, C++ RTAB-Map core df6300e / ROS 2a31e43, SQLite, pytest, isolated local Linux native replay.

## Evidence and branch scope

- PR371 open/draft at 9a268b84; supplied worktree clean at that commit (local branch alias feat/generic-rgbd-odometry-20261111).
- PR395 open at f914ab6c; extracted earlier implementation, excludes rtabmap_odom build and differs in numeric admission rules. No simultaneous edits.
- Baseline RGB/gateway/process suite: 62 passed.
- Existing smoke explicitly reports loop_closure_detection_verified=false; historical timings do not establish current acceptance.
- Pinned Rtabmap.cpp rejects null odom in incremental memory, supports it in localization mode. Memory.cpp computes a visual initial transform before ICP when no guess exists.
- Pinned DBDriverSqlite3.cpp persists optimized poses in Admin.opt_ids/opt_poses. Existing snapshot uses Node.pose and is incorrect after optimization.

## Constraints and review focus

- Keep integer capture/read/receipt times in computation; no metadata/status/freshness/identity gates.
- Numerical matrices, intrinsics and relative motion needed for projection must exist; never freeze a pose or fabricate motion.
- Distinguish format admission, visual geometry validation, localization and subsequent local tracking.
- Unknown or similar places must not grow an unlocalized map; loss/recovery must remain bounded.
- Native algorithm defaults and verification thresholds must not be relaxed to force success.
- Only supported physical projection conventions; providers own calibration and synchronization.
- Saved-map selection currently reopens a preview, not an active localization session. Do not claim restart/resume without implementing it.

## Tasks (each uses failing regression -> minimal implementation -> verification)

1. [ ] Fix saved graph reconstruction in pointcloud_slam_snapshot.py; tests with different raw/optimized poses, corrupt/incomplete blobs, scan extrinsics, archive reopening. Verify snapshot + library suites.
2. [ ] Add native recovery at the existing CoreWrapper process boundary, with OdometryF2M and map localization. Feed pose-less SensorData through SlamStream/gateway without inventing a pose. Preserve the original map origin across external origin recovery. Native tracking failure pauses mapping. Add state/evidence diagnostics and cleanup/resource tests.
3. [ ] Measure RGB preparation and fix actual computational waste without increasing 20ms budget. Verify projection geometry and bounded task behavior.
4. [ ] Add deterministic offline scene/replay with revisit, RGB-disabled comparison, unrelated and similar-texture geometry negatives, external pose loss, external reset and continued visual-only motion. Preserve Info, Link types, raw/optimized graph, error/continuity and latency evidence. Native tests are distinct from Python doubles.
5. [ ] Run relevant regressions and full Perception suite; review changes; document implemented/verified/unverified and commits/commands/artifacts. No robot acceptance claim.

## Execution ledger

- 2026-11-17: Read user-specified paths and pinned upstream sources. Local machine has no ROS/native/container runtime. Installed isolated test venv and Lima; native VM setup in progress. No robot connection or service action.
- Ruling: use native process-boundary recovery rather than asynchronous set-mode services plus a second estimator process. It keeps localization and graph mutation serialized in RTAB-Map, avoiding ROS service/result ordering races; requires a small pinned-source patch and native compilation/replay verification.
