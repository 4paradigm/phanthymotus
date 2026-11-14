# RGB-D Automatic Recovery Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user waived incremental confirmations.

**Goal:** Automatically recover a lost generic camera odometry instance into a fresh segment, while retaining strict quality gates.
**Architecture:** An external plugin timer supervises real RgbdSession instances; the session continues recording fresh image ingress after lost without admitting it to the retired estimator.
**Tech Stack:** Existing Python, ROS 2 and RTAB-Map; no new dependencies.
**Spec:** docs/superpowers/specs/2026-11-13-rgbd-auto-recovery-design.md

## Global Constraints
- PR371 only; preserve both output topics, all instance settings and measurement gates.
- 0.8 s recovery grace; at least 1.5 s between attempts; fresh color and depth required.
- New segment/origin/filter state per successful replacement; no cross-segment pose continuity claim.
- No deployment, restart of business services, container replacement or motion.

## Review Focus
- Stop/shutdown and independent instances cannot be restarted by an obsolete timer.
- Candidate construction failure leaves the original live lost session available for bounded retry.
- Cleanup failure disposes the candidate and never activates two tracked sessions.
- IMU/fusion restarts retain their settings and reset all histories.
- Nontracking summaries remain null; recovery does not emit a misleading final idle.

### Task 1: Recovery lifecycle
- [x] Add real-session tests using existing ROS/process boundary stubs: grace/freshness, new segment/origin, startup retry/backoff, stop/shutdown, cleanup failure, instance isolation and IMU/fusion settings.
- [x] Run the new tests and observe failure on missing recovery behavior.
- [x] Modify perception/plugins/rgbd_odometry.py and rgbd_odometry_runtime.py minimally; update old fake executors where they now own a real supervisor node.
- [x] Run recovery/RGB-D regressions and the complete Perception suite.

### Task 2: Tianyi source evidence and verification
- [x] Finish comparing raw gyro/accel and forwarded capture intervals; preserve diagnostic evidence outside source trees.
- [x] If source queue loss is confirmed, reproduce bounded backlog loss offline, fix only PR519 and run Tianyi/lifecycle regression.
- [x] Review final diff and record ready-for-push verification; use only the explicit PR371 and PR519 remote refs, rechecking their heads. Report live evidence separately from offline fix verification.
