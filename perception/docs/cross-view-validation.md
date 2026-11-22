# Dual-camera validation record

Driver worktree is D:/phanthymotus-driver/.worktrees/piper-dual-camera.
Driver branch is codex/piper-dual-camera at 8242cffab75ec78a1b62ba4569e3ff4f71888d65.
Perception worktree is D:/phanthymotus-driver/.worktrees/cross-view-object-consistency.
Perception branch is codex/cross-view-object-consistency. Original e0f0404 remains
an ancestor. Source integration is d0874b8 and acceptance tooling is 7930576.

## Executed validation

- Final host Perception run passed 172 tests across test_cross_view_object_consistency.py,
  test_vop_plugin.py and test_visual_depth_plugin.py, in 19.12 seconds.
- Linux Python 3.10 ARM64 Driver run passed 15 camera tests and 67 unified-interface
  tests in a hardware-free container based on bumi-driver-arm64:local-20261108.
  These are deterministic mocked-camera and control-interface tests.
- The official perception/Dockerfile.jetson completed all 36 steps using Buildx
  0.30.1 and JP_VERSION=511. The configured official base resolved to
  sha256:92c4c12a1dc5d4a4e8cb479a69164260578b4c3b022ef3b94c6f0fc20f2462d6.
  Existing dependency layers were cached. The build exported local tag
  motus-cross-view:jp511-dockerfile with manifest-list digest
  sha256:c8105578da37094bf5cd9b2eec77b379fd8530a9cb5cd7a128b37c5a6c9189ff,
  image manifest sha256:138ff2ba8a0c92083a8b518c47566ac3925dda97541cc4608463220bdf88b7a6,
  and config sha256:eb1d058ba65926581dfa3eb4345fa0fd3e5dcd6487c32df9afc222850dda7b6f.
  Subsequent source edits added RGB-only reporting and additional recovery checks.
  Rebuilding and running the final source remains pending.
- Prior e0f0404 MCP HTTP and three ROS DDS lifecycle cycles passed in ARM64.
  The expanded independent-session/recovery test has been implemented. Its final
  execution is pending environment recovery.
- An isolated Core started successfully and served /api/mcp inside WSL on
  127.0.0.1:25678. Chromium acceptance was attempted and failed at Dashboard load.
  A six-source MCP/DDS fixture and browser runner are committed for reproduction.
  No browser end-to-end success or screenshot was obtained.

## Environment blocker and cleanup

WSL Ubuntu-22.04 subsequently returned Wsl/Service/0x8007274c on repeated commands,
including /bin/true. Windows could not reach the mapped Core port. Rebuilding,
final DDS verification, and Docker cleanup could not execute through WSL.
A whole-WSL shutdown was avoided because other local tasks share that environment.

After WSL is operational, inspect and remove the two task-owned test containers
cross-view-core-20261122 and cross-view-fixture-20261122, then remove the dedicated
Docker network cross-view-acceptance-20261122. They have no hardware device
mappings or production data mounts. Container state after the WSL failure is
unverified.

The reproduction commands for the full image, ROS integration, browser acceptance
and recording replay are in cross-view-object-consistency.md. Full Dockerfile
build completion above applies to the recorded intermediate source. A final
source build and final-image runtime acceptance are required.

## Field evidence

No PiPER dual-camera rosbag or relevant measured stereo calibration was located
in the inspected repositories. WSL had no /dev/v4l/by-id devices. Real sensor
capability, serial binding, capture timing uncertainty, registered depth,
intrinsics, extrinsics and calibration version remain field acceptance inputs.
The device fingerprint hashes a stable configured by-id path. It relies on the
operator verifying that path's physical color device. Host-read timestamps have
no hardware exposure-time guarantee.

The capture contract encodes provenance in CompressedImage.header.frame_id.
TF consumers must extract optical_frame_id before using it as a TF frame name.
The Driver deployment guide documents this decision.
