# RGB-D Offline Stability Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user waived incremental confirmations.

**Goal:** Make start failures locatable and keep healthy RGB-D output through ESKF/aid/visual transients, without relaxing safety bounds.
**Spec:** docs/superpowers/specs/2026-11-13-rgbd-offline-stability-design.md
**Tech Stack:** Existing Python, NumPy, ROS 2 and RTAB-Map; no new dependencies.

## Global Constraints
- PR371 only; PR519 unchanged unless a source defect is proven.
- First `state/odom` port and the seven status fields unchanged; camera pose is primary.
- Non-tracking never carries a held or predicted pose; stale IMU never fused.
- Q/P, 0.10 m, 1 s and 500 ms IMU limits unchanged.
- No deployment, robot access, restart or motion.

### Task A: Start inputs
- [x] Core-to-contract integration test with Tianyi-shaped cards (RED: 7 of 13).
- [x] Per-input diagnostics; singular `input_topic`; plugin tests.

### Task B: ESKF decisions and fallback
- [x] Fusion RED tests: predicted sigma vs legal correction, bad/no visual, corrected sigma, info latency, frame silence, bounded diagnostics.
- [x] Decision order, wall/capture domain split, diagnostics.
- [x] Session visual-only fallback with continuity proof; IMU faults stop only the ESKF.

### Task C: Pure RGB-D transients
- [x] Session RED tests: single rejection, persistent loss to new segment, discontinuity, hard frame error, out-of-order, dropped captures.
- [x] `recovering` gate with continuity and 1.5 s bound; update tests encoding latch semantics.

### Task D: Off-lock depth assistance
- [x] RED: worker missing; apply/stale/error/timeout/bounded/close tests.
- [x] GeometryWorker, revision-matched application, session ownership and close.

### Task E: Auto frontend
- [x] RED: schema, switch, non-visual kinds, cooldown/limit, manual, restart rules.
- [x] Supervisor-only implementation reusing new-segment recovery.

### Verification
- [x] Focused RGB-D, Core startup, full Perception (current vs baseline), unchanged PR519.
- [ ] Hardware: see handoff report checklist.
