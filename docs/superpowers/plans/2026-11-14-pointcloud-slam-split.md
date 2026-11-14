# Standalone pointcloud and SLAM extraction plan

**Goal:** Extract the depth-only pointcloud and external-odometry SLAM cards from PR371 into a main-based branch and separate PR.

**Architecture:** Preserve tested implementations from e472c49. Include Core metadata routing, native PointCloud2 registration, RTAB-Map packaging and DDS input buffering. Exclude RGB-D odometry, IMU/ESKF, body-mount code and their dedicated tests. Keep PR371 unchanged.

**Constraints:** One depth input; no inferred physical calibration or capture clocks; no deployment, robot action, forced push or merge.

- [x] Check clean source and remote heads; fetch main 7b16c43.
- [x] Copy scoped tests first and establish missing-plugin failures on main.
- [x] Extract scoped runtime/build dependencies; remove RGB-D registration/config/docs; correct obsolete mono/stereo comments.
- [x] Verify focused tests, full Perception/Core regressions and Python 3.8 coverage.
- [x] Independent review of extraction boundaries and runtime dependencies.
- [ ] Commit, push a new branch, create a separate PR and verify both PR heads.

Review focus: missing shared imports; accidental RGB-D card exposure; native build dependencies; connected metadata route preservation; depth-only advertised schema.

Validation: Perception 2787 passed / 41 skipped / 71 subtests; Core 1729 passed / 1 skipped / 19 subtests; Python 3.8 scoped suite 532 passed / 5 skipped. Independent review: no P1/P2. Native image build and field validation pending.
