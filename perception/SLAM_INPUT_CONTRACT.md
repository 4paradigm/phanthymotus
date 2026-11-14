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

- **Current Core / declarations:** Core forwards topic-keyed `input_metadata`,
  `input_bindings` and `odom_interfaces` from the connected ports. Complete
  declarations are checked for matching frames, clocks and mount geometry,
  with `frame_metadata_topic` linked by exact timestamp, frame and payload SHA.
  Independently sourced clocks require matching verified relation IDs.
  Same-source clock accuracy and mount calibration remain recorded provenance;
  their presence alone is not a physical accuracy guarantee.
- **Startup and topic-only discovery:** missing/incomplete startup declarations
  are reported in diagnostics. When geometry remains unresolved, `start_mapping`
  probes live tracking `motus.odom/1` and native `PointCloud2` to resolve frame
  names, measurement times and mount evidence. This also supports older callers
  that supply only `input_topics`. Explicit contradictory declarations are
  rejected. Default probe timeout is five seconds; `max_gap_ms` defaults to 200.
  Timestamp overlap is recorded as **unverified**, not independent-clock
  synchronization proof; passing an empty metadata map does not certify inputs.
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


## Jetson native RTAB-Map image dependencies

The native `pointcloud_slam` backend requires a pinned RTAB-Map core,
`rtabmap_ros` Humble wrapper and compatible PCL, geometry2, ROS messages and
Python bindings. `perception/Dockerfile.jetson` builds against the selected
JetPack base (JP5.11/Python 3.8 or JP6.1/Python 3.10), rather than assuming an
unversioned host installation. The repository does not currently ship an
audited prebuilt RTAB-Map runtime bundle matching both base-image ABIs;
adopting one would require additional native compatibility validation.

**Image-size growth is expected.** The current single-stage source build
installs `build-essential`, `libpcl-dev`, Boost, Eigen, SQLite and OpenSSL
development packages, which add substantial weight to the image. These
packages are required by the present native compilation approach, so this is
not a packaging-neutral change. To avoid retaining downloaded archives in
the new installation layer, the Dockerfile runs `apt-get clean` and removes
`/var/cache/apt/archives/*` and `/var/lib/apt/lists/*` in that same layer.
Removing toolchains and headers from the delivered image safely would require
a separate multi-stage build with a verified ABI-matched runtime dependency
closure; that larger packaging change is out of scope for this fix.
