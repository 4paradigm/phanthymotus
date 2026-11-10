# Perception deployment configuration directory

This directory is intentionally present for
`perception/Dockerfile.jetson` (`COPY perception/configs/ /work/configs/`).
It must remain tracked even when no optional YAML profile is installed.

The generic `pointcloud_slam` card has no robot-model presets. Its
physical frame, clock and extrinsic are
resolved from connected sensor/odometry inputs or an explicit
`child_from_cloud` matrix, as described in
[`SLAM_INPUT_CONTRACT.md`](../SLAM_INPUT_CONTRACT.md).

This file is documentation/build-context only; it does not enable a plugin
or override any sensor values.
