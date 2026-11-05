# Persistent SLAM History Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation.

**Goal:** Continue experimental mapping beyond 300 keyframes without unbounded resident points or a growing optimization graph, preserving historical display and explicit restore.

**Architecture:** Page immutable local keyframe points to atomic NPY files with a committed journal. Optimize fixed batches of at most 300 frames with one overlapping anchor, retaining stable capture-stamp identities and original odometry segments. Freeze completed batch poses; align the next batch through its overlap anchor. Sample historical files before reading/transforming their points. This is bounded local graph optimization with persistent history, not verified global loop correction.

**Tech Stack:** Existing Python/NumPy/Open3D/GTSAM/ROS; no new dependencies.

**Design decisions:** User explicitly requested direct implementation and deployment, with no odometry changes or extra canvas configuration. Keep bounded in-memory legacy mode as default for offline callers, enable paging and motion-based keyframes in the managed Bumi profile. Disk quota and a free-space reserve remain explicit resource limits. Storage failure stops new fusion while preserving committed history; it never invents a lost event or segment. Save/restore fixes both keyframe membership and optimized-history state. Real source timestamps and false verification flags remain unchanged.

- [x] Add paged keyframe storage and movement selection. Verify >600 frames, stationary health, rotations, new-segment first frame, failed write/manifest commit, bounded file lifetimes and disk-full state.
- [x] Add fixed-batch geometry history with an overlap anchor and global preview sampling. Verify >600 frames/two transitions with rotation, continuous map coordinates, retained old points, and true-segment isolation. Bound active frames and points.
- [x] Integrate plugin/profile, input status, save/restore and preview lifecycle. Reject incompatible relocalization mode for paged history rather than silently dropping constraints. Verify restore of frozen poses and strict verification-policy checks.
- [x] Run local/target tests and resource-limited >600-keyframe replay. Measure computation, memory and map changes. Review the implementation before deployment.
- [x] Preserve current trial history, deploy Perception only and verify live progress and new retention behavior.
- [ ] Push final deployment evidence and request updated PR build/review. No robot motion commands.


Ruling: use independent history output topics per real source segment on restore/reset; never overlay unrelated frames. A saved geometry snapshot displays its exact optimized prefix, while an archive without any computed snapshot is explicitly an odometry preview. An end-to-end restore test verifies actual history packets, not only loaded dictionaries.

Evidence: 192 SLAM tests on both host and target, real-rclpy DDS paging/start/stop/save/restore/history-output test pass. Target 620-keyframe captured-cloud/synthetic-trajectory replay crossed two batch boundaries, kept all 620 frames/poses, RSS peak 331720 KiB, median compute 1.129 s, p95 1.853 s, max 2.224 s; 378637-byte checkpoint saved in 0.068 s, restored in 4.408 s with identical history. Atomic directory fsync was additionally added after journal commits; final target regression checks this version before deployment.
