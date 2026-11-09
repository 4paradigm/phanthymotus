# Consuming the cross-robot Spatial SLAM Input Contract

**Canonical driver specification:** `phanthymotus-driver/docs/spatial_slam_input.md`
([public source](https://github.com/Nidhogg2002/phanthymotus-driver/blob/feat/unified-odom-driver/docs/spatial_slam_input.md)).
The normative schema identifiers are `motus.odom/1` with
`vendor.motus_slam`, `motus.slam_input/1` for the two input declarations,
and `motus.slam_frame/1` for exact per-frame provenance.

This Perception image is deployed separately from the robot driver. It embeds
a **source-identical** pure Python validator in
`plugins/pointcloud_slam_contract.py` (canonical copy
`phanthymotus-driver/common/spatial_slam.py`) and tests compatible valid and
invalid examples in `tests/test_pointcloud_slam_contract.py`.
The live consumer is `plugins/pointcloud_slam_input.py`, which additionally
enforces cross-topic frame, calibration, synchronization, epoch and payload
consistency.

## Acceptance modes

- **Explicit declaration:** a caller supplies topic-keyed `input_metadata`
  and optionally `input_bindings` and `odom_interfaces`. The current
  `resolve_inputs()` path requires verified clock mapping for **both**
  sources, calibrated static/dynamic extrinsic, and a
  `frame_metadata_topic` linked by exact timestamp, frame and payload SHA.
  Independently sourced clocks require matching verified relation IDs.
  `experimental=True` exists for ROS-free compatibility tests and is **not**
  an exposed user-policy switch.
- **Core main / topic-only:** Core forwards the two connected
  `input_topics`. The SLAM card discovers the actual ROS publisher message
  types and, on `start_mapping`, briefly subscribes to live tracking
  `motus.odom/1` and native `PointCloud2` to resolve frame names,
  measurement times and mount evidence. Default probe timeout is five seconds;
  `max_gap_ms` defaults to 200. This is an **unverified timestamp-overlap
  fallback**, not independent-clock synchronization proof.
- **Manual mount:** the card exposes only `child_from_cloud`, a rigid 4x4
  `T_odom_child_cloud` with translation in meters. If an upstream mount
  exists, a conflicting manually entered matrix is rejected. A producer
  assumption (`calibrated:false`) cannot be adopted automatically.
  If the cloud and pose child frames are genuinely identical and no
  different mount is declared, identity is valid.

## Required ROS topics

| Format | Message | Requirements |
| --- | --- | --- |
| `state/odom` | `std_msgs/msg/String` | JSON `motus.odom/1`, tracking SLAM pose, stamp, frame and segment |
| `sensor/pointcloud2` | `sensor_msgs/msg/PointCloud2` | Finite float32 XYZ fields, capture header stamp, frame, metric coordinates |

`sensor/pointcloud` is a separate `UInt8MultiArray` visualization wire
format and is not a substitute for native PointCloud2. A robot laser that
only publishes `LaserScan` requires conversion upstream. The current
RTAB-Map process uses `subscribe_scan_cloud=true` and external odometry,
so adding another odometry estimator inside this card is unnecessary.

The card's single fixed `sensor/mapping` output is **preview only**.
Mapping and localization must use the saved native RTAB-Map database, not
the preview packet. Maps and revisions retain `input_provenance` and
the resolved frame/clock/extrinsic settings.

## Diagnostics and tests

Before SLAM deployment, verify that the driver publishes exactly one native
PointCloud2 input and one tracking pose, both in the same ROS 2 DDS domain.
Inspect `pointcloud_slam.info()` for `input_status` and
`input_provenance`, and never treat
`timestamp_overlap_only_not_independent_clock_proof` as calibrated
time synchronization.

For portable checks run:

```bash
python -m pytest -q perception/tests/test_pointcloud_slam_contract.py \
  perception/tests/test_pointcloud_slam_input.py \
  perception/tests/test_pointcloud_slam_config.py
```

The source-identical Driver-side validator is
`common/spatial_slam.py`; update it and the Perception copy in the
same cross-repository change. The public contract name stays `/1`
unless a breaking change warrants a new version.
