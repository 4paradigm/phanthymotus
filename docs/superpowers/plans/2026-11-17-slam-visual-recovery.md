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

1. [x] Fix saved graph reconstruction in pointcloud_slam_snapshot.py; tests with different raw/optimized poses, corrupt/incomplete blobs, scan extrinsics, archive reopening. Verify snapshot + library suites.
2. [x] Add native recovery at the existing CoreWrapper process boundary, with OdometryF2M and map localization. Feed pose-less SensorData through SlamStream/gateway without inventing a pose. Preserve the original map origin across external origin recovery. Native tracking failure pauses mapping. Add state/evidence diagnostics and cleanup/resource tests.
3. [x] Measure RGB preparation and fix actual computational waste without increasing 20ms budget. Verify projection geometry and bounded task behavior.
4. [x] Add deterministic offline scene/replay with revisit, RGB-disabled comparison, unrelated and similar-texture geometry negatives, external pose loss, external reset and continued visual-only motion. Preserve Info, Link types, raw/optimized graph, error/continuity and latency evidence. Native tests are distinct from Python doubles.
5. [x] Run relevant regressions and full Perception suite; review changes; document implemented/verified/unverified and commits/commands/artifacts. No robot acceptance claim.

## Execution ledger

- 2026-11-17: Read user-specified paths and pinned upstream sources. Local machine has no ROS/native/container runtime. Installed isolated test venv and Lima; native VM setup in progress. No robot connection or service action.
- Ruling: use native process-boundary recovery rather than asynchronous set-mode services plus a second estimator process. It keeps localization and graph mutation serialized in RTAB-Map, avoiding ROS service/result ordering races; requires a small pinned-source patch and native compilation/replay verification.
- Snapshot task committed as 10c2174e; Python snapshot/library checks passed. Independent review requested via requesting-code-review; identified native finalization of the full connected graph, continuous numerical origin reset, and replay RGB-gap coverage. Addressed all three in implementation, awaiting final validation.
- Native environment: local Lima ubuntu-22.04 arm64 (2 CPU, 3 GiB, 2 GiB swap); source core built successfully at the exact pin plus existing rotation patch. ROS binary dependencies are 0.23.13; source overlay MUST explicitly set RTABMap_DIR=/usr/local/lib/rtabmap-0.22 to avoid ABI/header mixing. First ROS attempt exhausted memory with parallel jobs; subsequent source build uses one job.
- Actual deterministic core reproduction: original ICP neighbor refinement can replace external covariance with overconfident scan constraints and reject valid loop candidates in graph optimization. Visual+ICP strategy alone improved trajectory but did not remove graph inconsistency. Chosen repair retains supplied motion covariance on image frames; missing-image frames retain original scan behavior. No matching/optimizer threshold was reduced.
- Further pinned-source audit: a proximity edge can be promoted to kGlobalClosure when its id equals the appearance candidate. With ProximityOdomGuess=true, ICP can bypass visual geometry, so Info.loop_id/type=1 alone are insufficient. Image frames now use ProximityOdomGuess=false; relocalization disables proximity entirely. Re-running final native evidence with inlier/graph/trajectory checks; earlier experiments are diagnostic, not final acceptance.
- Extended the existing start_mapping API with optional map_id/revision, private verified database copying, original map name/revision binding and null-pose startup. Restored native core requires new geometric alignment, including when the first returning external pose is numerically valid but has an unknown origin.
- Full Perception first run: 3490 passed, 60 skipped, 13 failures due missing PyYAML in new venv; installed it and all 13 failed tests passed on rerun. A final full suite remains required.

- Final review reproduced/fixed cloud/image lifetime mismatch on odom silence (reserve decoded image with its bounded pending cloud, including late decode), loss of the paired native/VO anchor across an external-only RGB gap, bound-map rename forking save identity, and preflight capability rejection after segment registration. Native reset-gap regression went from ~100m error to ~0.018m. ROS run3 completed 135 frames, two recoveries, sustained mapping and archive roundtrip with explicitly absent IMU; next final run includes all review fixes.
- Further review reproduced explicit pose=null inflight-cloud/late JPEG loss within the 20ms budget. Added failing regression, bounded matching-decode wait outside owner lock, decoder wakeups, then 58 RGB/gateway checks passed. Production replay now deliberately does not wait for decode before publishing clouds. Independent read-only review found no remaining lock/bounds defect.
- Full Perception after this repair: 3530 passed,46 skipped,71 subtests. Ordinary full run also exposed three existing TTS tests making real localhost HTTPS completion calls; isolated AGENT_CORE_URL=https://127.0.0.1:1 avoids those callbacks. No TTS code changes.
- Real ROS zero-origin and nonzero-origin (7,-3,.5) replays passed graph/negative/VO/snapshot checks. Cold reopening the nonzero-origin run grew 114 to129 optimized nodes, 8 native VO-only nodes, XYZ max.030m. A second cold replay of the async run failed checked graph attachment; old saved edge50->87 has near-zero saved residual, but reoptimization with new constraints produced .084m/27sigma error. Keep this rejected run as evidence, never claim universal saved-map recovery.
- Disabling image proximity alone did not repair repeated core recovery (discarded). Native RGBD/LoopCovLimited=true bounds new non-neighbor information by native odometry information. It changes weighting, not registration/graph thresholds, and cannot retroactively fix old edges. Current new core136-frame recovery and16-frame cold-resume pass; rebuilding final ROS and revalidating from newly generated archives. Final global graph may still have >3sigma residuals on some existing constraints; record absolute errors and this native limitation, do not weaken validation.

- Final result: pinned ROS Release/O3 build passed (6m10s); full suite3530 passed/46 skipped/71 subtests; focused final75 passed. New ROS136-frame nonzero-origin run: two geometric recoveries (436/636 inliers),37+8 VO-only nodes,8 rejected negatives,XYZ max.0713m,117 optimized nodes,graph linear ratio1.646. Two independent cold restarts with depth interruption both117→130 nodes and6 VO-only new nodes,XYZmax.0261/.0401m; graph ratios2.330/1.826. All archive previews roundtrip identically; actual saved marker error0 and known-surface medians.0155/.0080/.0135m. Native children exited; SQLite/hash manifest recorded. Current implementation/read-only review found no remaining blocking defect.
- Disclosure: 20ms budget unchanged; main1/132 and each cold1/14 image preparations timed out; cold backend preparation maxima47/70ms remain. Different capture-time/no-relative-motion projection, old archive graph conflicts, real datasets/robot motion, canvas UI, Jetson image build and rotation-reset replay remain unverified/limited as documented. Validation report: perception/docs/slam-visual-recovery-validation-20261117.md. No PR395 edits or remote push.
