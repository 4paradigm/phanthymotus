# Consuming spatial SLAM inputs

The live consumer follows the [spatial input data standard](docs/spatial-inputs.md):
only wire structure and the numeric values needed for computation determine
input acceptance. The Driver's
[producer specification](https://github.com/Nidhogg2002/phanthymotus-driver/blob/feat/unified-odom-driver/docs/spatial_slam_input.md)
documents physical conventions and provenance. Its stricter producer validators
are not the live consumer's admission policy. Perception's validator is therefore
no longer a source-identical copy of `common/spatial_slam.py`.

## Input paths

- Explicit declarations use topic-keyed `input_metadata`, with optional
  `input_bindings` and `odom_interfaces`. Declarations identify supported formats;
  connected topics determine routing. Source/frame/clock names, quality, units,
  calibration, hashes and timing evidence are diagnostic, not admission gates.
- Topic-only starts discover the connected ROS message types and probe live
  numeric pose/cloud inputs. The default probe timeout is five seconds. This
  discovers usable data; it does not prove physical synchronization.
- A supplied finite numeric manual `child_from_cloud` 4×4 matrix takes precedence
  over producer matrices, including later sidecars. Without a matrix, incoming
  cloud coordinates are assumed already aligned with pose child coordinates.
  Explicit dynamic mounting waits for its numeric matrix. Exact timestamps are
  preferred; otherwise the nearest supplied matrix is used without a skew limit.

## Required ROS topics

| Format | Message | Numeric requirements |
| --- | --- | --- |
| `state/odom` | `std_msgs/msg/String` | `motus.odom/1` JSON, integer timestamp, finite position[3] and nonzero quaternion[4], optional finite covariance[36] |
| `sensor/pointcloud2` | `sensor_msgs/msg/PointCloud2` | Float32 XYZ fields, valid offsets/strides and payload length, integer header timestamp |

Precise `vendor.motus_slam.stamp_ns` or observation time takes precedence when
supplied; otherwise the explicit pose time, then legacy top-level time is used.
Read/receipt times participate in full computation. Neither time-label equality
nor source freshness is checked. Poses are sorted, duplicate times use the latest
sample, and clouds wait in bounded buffers for numeric interpolation brackets.
A status such as `lost` does not veto a usable pose; `pose: null` supplies none.

`sensor/pointcloud` is a separate visualization format, not native PointCloud2.
LaserScan requires upstream conversion. Native RTAB-Map uses external odometry
and `subscribe_scan_cloud=true`. The single `sensor/mapping` output is preview
only; mapping/localization uses the saved native database. Archive path,
checksum and file-integrity checks remain storage requirements.

`pointcloud_slam.info()` exposes input status and provenance. These diagnostics
are not calibration or synchronization certificates. Portable checks:

```bash
python -m pytest -q perception/tests/test_pointcloud_slam_contract.py \
  perception/tests/test_pointcloud_slam_input.py \
  perception/tests/test_pointcloud_slam_numeric_input.py
```

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


## Scan-only proximity correction

The pointcloud SLAM card uses ICP with spatial proximity detection. Its
one-to-one path explicitly enables `RGBD/ProximityOdomGuess`: the pinned
RTAB-Map implementation otherwise attempts visual matching without a pose
guess, although this card supplies no images. The scan-neighbor path remains
limited to one node (`RGBD/ProximityPathMaxNeighbors=1`, matching the pinned
ROS wrapper's scan-cloud default). ICP acceptance and graph validation
thresholds are unchanged. This is local geometric correction near the estimated
trajectory, not appearance-based global place recognition; zero accepted loops
can still mean insufficient overlap, drift beyond the search radius, or a
rejected match.

The native adapter publishes segment-local dynamic odom-to-base TF from the
same admitted pose and exact capture stamp as its Odometry/PointCloud2 pair.
Static camera extrinsics remain on tf_static; dynamic camera extrinsics remain
on tf. This satisfies the native odometry lookup without inventing poses or
using latest-time TF. No robot-facing odometry interface or clock budget changes.


## Optional RGB appearance input

The optional `image/jpeg` port uses the same format-only policy. JPEG, dimensions,
intrinsics and the numeric poses/transforms needed for projection must be usable.
Source/time/frame labels, capture proof and RGB/scan skew do not gate use. The
provider owns rectification and physical alignment; see the
[spatial input standard](docs/spatial-inputs.md).

Images are optional per native frame. Missing imagery, unusable projection data
or preparation timeout leaves the same native map process using scan and external
odometry. RGB recovery does not reset the map origin. Feature depth is projected
from numeric cloud points; holes remain unknown. No missing pose or transform
is invented. Bounded queues and preparation budgets still limit resource usage.
