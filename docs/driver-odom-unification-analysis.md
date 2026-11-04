# Driver odometry unification: source audit and proposed sequence

Scope: static inspection of the local `unified-odom-driver` checkout at
`09c1389c`, not a claim about every deployed image or every robot. Existing
outputs must remain available. The target is the additive `motus.odom/1`
contract in the driver's `README_dev.md`, with the separate `motus_slam/1`
pose capability; standard velocity alone is not a SLAM pose.

## Inventory and unknowns

| Source | Evidence in Driver | Current assessment and work before SLAM use |
| --- | --- | --- |
| Bumi visual odometry | `noetix/bumi/visual_odometry.py` advertises `motus_slam/1`; `odom_spec.py` is a separate velocity source | Retain VO's existing quality, freshness and segment behavior. Verify its measurement clock against the particular point-cloud stream and calibrate `T_child_cloud`; don't substitute the velocity-only stream. |
| Unitree AS2W | `unitree/as2w/device.py` builds a standard twist sample and explicitly advertises `slam_pose.available: false` | Velocity can be standardized without claiming a pose. Find and verify an independent pose source before enabling the pose profile. |
| Unitree R1 | `unitree/r1/device.py` builds a standard twist sample without a pose; its interface does not advertise the SLAM pose profile | Treat missing capability as unavailable. Vendor position requires separate validity and frame investigation; do not integrate velocity into a fabricated pose. |
| Robotera Q5 | `robotera/q5_bundle/odom.py` subscribes to ROS `Odometry` and exports position, quaternion and frames as legacy JSON, but `timestamp_ms` is publication time | Candidate adapter at the original ROS callback: preserve header measurement stamp and covariance if valid. Verify clock, frame meanings, quality, reset detection and source frequency; 2 Hz legacy publishing is not proof of adequate SLAM timing. |
| Booster K1 | `booster/k1/device.py` publishes SDK position, orientation, velocities and timestamp as legacy JSON, and exposes `reset_odom` | Candidate only: establish timestamp unit/clock, quaternion order, coordinate frames, origin, quality and resets, including resets initiated outside this driver. The existing topic is `data/json` despite its name. |
| Agibot AimDK X2 | `agibot/AimDK_X2/device.py` mirrors `/slam/lidar_odom` ROS `Odometry` as JSON | Candidate for a separate adapter after checking header time, frame, health and reset. This source is already a SLAM output; its upstream corrections and discontinuities need explicit semantics to avoid treating map corrections as uninterrupted raw odometry. |
| EngineAI T800 | `engineai/t800/device.py` subscribes to configurable ROS odometry and separately has a point-cloud accumulation path | Candidate at the original odometry subscription; verify time, frames, quality and resets. Existing point accumulation is not evidence of a validated loop-closure backend. |
| Unitree G1 / Go2 | `unitree/g1/device.py` and `unitree/go2/device.py` publish legacy locomotion state with vendor velocity and position fields | Assess velocity axes, units and timestamp from actual messages before adding a standard twist output. Presence of a position field alone does not establish a usable SLAM pose. |

This is a prioritized inventory of inspected sources, not a completeness claim
about every driver. Unknown frame, measurement timestamp, clock or reset
semantics must stay unknown, not be guessed from a topic name.

## Proposed implementation and checks

1. **Contract and regression fixtures.** Keep each existing topic intact; add a
   discoverable `state/odom` output and `odom_interface` metadata. Add captured
   or faithfully constructed message fixtures covering units, quaternion order,
   missing axes, invalid values, time origin and old-topic compatibility.
   Capability remains unavailable until the pose profile's requirements are met.
2. **ROS Odometry candidates, beginning with Q5.** Map original header stamp,
   pose, frame IDs and covariance; retain the separate source's twist-frame
   semantics. Check source clock against cloud time and determine tracking and
   reset signals from the source. Do not label receipt time as measurement time
   or assume an absent health signal means tracking. Exercise stale data,
   restart, timestamp regression, frame change and reset in tests and real logs.
3. **Vendor SDK candidates.** For K1, establish the SDK's timestamp, frames,
   quaternion and reset/health semantics through documentation and captured
   samples, then add an adapter with the same tests. Do not infer that this
   driver's reset API observes resets triggered by other clients.
4. **Velocity-only robots.** Extend the established standard twist output
   where source semantics are known; preserve unknown axes as null and mark
   SLAM pose unavailable. Promote a robot only after an independent valid pose
   source meets the same contract; don't integrate twist to manufacture one.
5. **End-to-end acceptance per robot.** Verify runtime `info()` discovery,
   source timestamps and shared or calibrated clock relation, extrinsic and
   actual pose reference point, loss/recovery and segment transitions using
   read-only captures. Check legacy consumers and ensure shutdown of the SLAM
   consumer does not stop producers. Only then attempt labelled loop tests.

The proposed order favors sources already carrying ROS pose fields, but an
adapter must remain unavailable for SLAM if its required semantics cannot be
proven. Perception still has exactly two external connections: odometry and
cloud; calibration is message metadata or startup configuration.
