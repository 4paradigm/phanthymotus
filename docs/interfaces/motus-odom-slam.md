# SLAM pose profile for motus.odom/1

This profile is an additive convention, not a new `motus.odom/1` core schema.
The transport remains JSON in `std_msgs/msg/String`, on a `state/odom` output
resolved from the running card's `info()`; never assume a fixed topic name.
Existing velocity-only samples and vendor-specific fields remain valid odometry,
but they do not meet this profile. A consumer must not integrate twist to invent
an absolute pose.

A source advertises `odom_interface.slam_pose = {"available": true,
"profile": "motus_slam/1"}` only when it can supply the following information.
Absence of this declaration means unknown, not available. `provides` still
lists velocity axes; `pose_drift` is not a pose capability flag.

For an accepted sample:

* `stamp_ms` is the pose estimate's measurement time in milliseconds, not its
  publication or reception time. `vendor.motus_slam.stamp_source` is
  `measurement`; when the time is unavailable or publication time was used,
  set `unknown` and do not fuse. `clock_id` names the producer's time domain.
  A matching name alone does not synchronize independent clocks: the consumer
  must verify that point cloud and pose actually share a clock or have a
  measured mapping, and enforce a bounded alignment window. Conversions from a
  hardware clock and their uncertainty must remain visible in source metadata.
* `units.length` is `m`. `pose.position` has three finite coordinates;
  `pose.quaternion` is a finite normalized `[x,y,z,w]` quaternion.
  `pose.frame_id` is the pose's parent odometry frame and `pose.child_frame_id`
  is its rigid-body reference frame. The pose is the transform from child into
  parent (`T_odom_child`). `frame` outside `pose` describes twist, not pose.
* `pose.covariance` is either 36 finite row-major entries in the source's stated
  convention or `null` if unknown. Never substitute zero for missing values.
* `vendor.motus_slam` contains `status`, `segment_id`, `stamp_source`, and
  `clock_id`. Status values are `tracking`, `degraded`, `lost`, `initializing`,
  or `unknown`. The present parser accepts only `tracking`. A segment ID is a
  non-empty string stable during one odometry coordinate system and changes
  after a reset, loss recovery or other discontinuity. A lost sample has no
  usable pose. Producers preserve their original quality fields separately.

The cloud-to-pose-reference transform `T_child_cloud` and calibration must be
provided with the cloud or startup configuration, never inferred from a robot
name or coordinate-axis convention. The cloud pose is
`T_odom_child * T_child_cloud`. No extra canvas input is required for metadata.

Parsing one message does not establish freshness, order, time synchronization,
calibration validity, safe interpolation, segment transition handling, loop
closure, or SLAM. Consumers must implement those checks before fusion. A 2D-only
estimate must not silently fill unmeasured axes and claim a 3D pose.

Bumi's visual odometry uses a mapped ROS clock and retains its existing mapping
and diagnostic metadata. Its `body_axes` is an axis convention; without a
measured mount it is not a measured body-origin extrinsic. The old status topic
and full odometry topic remain available for existing canvases.
