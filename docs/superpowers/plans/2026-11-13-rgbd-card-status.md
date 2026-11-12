# RGB-D compact card and status implementation plan

> Execute inline using superpowers:executing-plans and TDD; final fresh read-only review. User opted out of incremental confirmations.

Goal: hide RGB-D manual form, preserve instance config and automatic inputs, append Bumi-shaped status without changing existing odometry connections.
Architecture: existing RgbdOdometryPlugin schema/config; existing RgbdSession owns both publishers and status throttle/origin. No dependencies or driver changes.
Spec: docs/superpowers/specs/2026-11-13-rgbd-card-status-design.md

## Task 1: Schema and status
Files: perception/plugins/rgbd_odometry.py, perception/plugins/rgbd_odometry_runtime.py; tests/test_rgbd_odometry.py plus new perception/tests/test_rgbd_odometry_status.py; perception/README.md.
- RED: tool only exposes system actions, all algorithm config remains instance-scoped and config replay/start still works; original state/odom first topic preserved, extra status isolated per instance.
- RED: real RgbdSession with stubbed ROS/native process only creates declared status publisher; real _publish emits Bumi fields, correct displacement/age and null angle; no prediction leakage on lost/coasting; identical-state throttle and immediate transitions; actual timer stale gate emits lost; close emits idle and disposes node; fresh session resets origin.
- GREEN: minimal schema and status publication implementation; amend existing manually constructed test sessions for new owned state, no production test-only fallbacks.
- Run relevant RGB-D suite, then complete Perception suite. Verify automatic injection/config behavior in agent-core focused regression if environment supports it. Document output and field semantics.
- Fresh review; fix concrete significant findings with red/green; commit and push only PR371 branch, verify remote SHA. No hardware/deployment.

## Review focus
1. Existing saved SLAM links retain first state/odom port and exact topic.
2. Hidden instance/calibration properties must still arrive through auto start and config replay.
3. IMU failure after sample creation must yield lost summary without stale pose or inertial prediction.
4. Startup/no input/stale native output and stopped instances must not retain tracking display.
5. Restart/multiple instances must not share summary origin, throttle or status topic.
