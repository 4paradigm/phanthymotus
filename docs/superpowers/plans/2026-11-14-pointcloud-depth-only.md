# Depth-only pointcloud implementation plan

**Goal:** Keep one depth input on the pointcloud card and remove internal mono/stereo implementations.
**Spec:** User-confirmed scope on 2026-11-14: delete mono/stereo internals, retain one depth input, test and push PR371.
**Architecture:** Keep the existing bounded depth worker, projection and output contracts. Remove image inference, stereo matching, their queues/model lifecycle and exclusive math helpers. No changes to RGB-D odometry or the separate visual_depth card.
**Tech stack:** Python 3.8+, NumPy, existing ROS Image/CompressedImage transports.

## Constraints and review focus

- Accept one Z16/zlib depth topic; reject multiple/contradictory topics and RGB declarations.
- Saved single-input auto/depth calls remain usable; mono/stereo requests fail explicitly.
- Keep late publisher discovery, bounded decoding, frame/header provenance and stop cleanup.
- Preserve calibration admission for metric output and existing display-only fallback semantics.
- No deployment, robot actions, dependency additions or unrelated refactors.

## Execution

1. Add failing card/schema, input-count/type and depth-callback regressions in test_pointcloud_depth_only.py.
2. Remove mono/stereo code in pointcloud.py and exclusively used helpers in pointcloud_math.py; update main.py description and README.
3. Remove obsolete mono/stereo tests; retain depth, metadata, geometry, configuration and lifecycle coverage. Verify explicit compressed encoding and late-discovery reporting.
4. Run focused tests, full Perception and Python 3.8 coverage; independent review, correct findings, then commit and fast-forward push PR371.
