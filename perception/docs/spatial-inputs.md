# Spatial input data standard

Live RGB-D, pointcloud, odometry, transform and SLAM inputs use the same
format-only admission policy as [IMU inputs](rgbd-imu.md). Consumers parse the
numeric structure needed for computation. Source conventions and physical
quality are responsibilities of the data provider, documented here rather than
enforced through provenance, identity or freshness gates.

## Formats used by computation

| Input | Transport / payload | Numeric structure |
| --- | --- | --- |
| RGB | `image/jpeg`, ROS `CompressedImage` | Decodable JPEG; positive declared width/height; finite 3×3 `K` with nonzero focal entries |
| Depth | `image/depth-zlib`, ROS `CompressedImage` | Zlib-compressed uint16 grid matching declared depth dimensions; legacy dimensionless depth defaults to 640×480 |
| Metric depth | ROS `Image` | Supported depth encoding, matching buffer length, row layout, dimensions and intrinsics |
| Pointcloud | `sensor/pointcloud2`, ROS `PointCloud2` | Float32 x/y/z fields with valid offsets, point/row stride and payload length; organized/padded layouts are supported |
| Odometry | `state/odom`, `motus.odom/1` JSON | Integer millisecond time, finite position[3], finite nonzero xyzw quaternion[4]; optional finite covariance[36] |
| Precise pose time | `vendor.motus_slam.stamp_ns` / observation context | Integer nanoseconds when provided |
| Static transform | JSON array or declaration matrix | Finite numeric 4×4 matrix |
| Dynamic transform | Per-frame JSON sidecar | Integer nanosecond time and finite numeric 4×4 matrix |

The accepted camera dimensions are bounded to 4096 per axis and buffers/queues
have capacity limits. These are allocation/layout limits, not source-quality
certificates. Data that cannot be decoded, reshaped or numerically used is a
format error. An absent numeric pose or explicitly dynamic transform cannot be
invented; computation waits for those values.

## Provider conventions

- Supply registered, rectified RGB and depth on the same optical rays. Supply
  the actual pinhole intrinsics, dimensions and appropriate zero-distortion
  images. Names, registration labels, `D` values and IDs are not verification
  of this physical relationship.
- RGB-D depth is optical axial Z, not Euclidean range. Its compressed uint16
  values represent millimetres. Metric projections and pointcloud XYZ use
  metres. Unit/depth-source labels do not convert values or restrict use.
- Optical axes are +X right, +Y down, +Z forward. Pose position and quaternion
  must describe the same reference point and coordinate convention as their
  incoming cloud. Keep a continuous coordinate origin within a mapping session.
- Timestamps are integer nanoseconds on ROS/observation inputs and integer
  milliseconds on the base pose field. If a precise pose/observation timestamp
  is supplied, it is used by numerical association. Provide one aligned
  numerical timeline and the samples needed for interpolation. Labels do not
  establish synchronization and the consumer does not fit an undeclared clock
  conversion.
- For meaningful temporal integration, provide ordered samples and consistent
  capture values. The consumer accepts repeated, out-of-order, delayed, future
  or zero/negative integer times without treating them as source-policy failures.
- A transform uses column vectors: `p_parent = R @ p_child + t`. Supply proper
  homogeneous rigid matrices and the actual installation origins. The live
  parser checks shape/finite values, not SO(3), last-row properties or calibration
  quality. Pose quaternions are normalized numerically when decoded.
- If no transform is supplied, the arithmetic convention is that incoming cloud
  coordinates are already in the pose child coordinates (identity). This is a
  caller-provided alignment assumption, not measured calibration evidence.
  An explicit numeric manual transform takes precedence over a producer matrix.
- Pair RGB and depth correctly upstream. The consumer pairs the next retained
  RGB and depth inputs in FIFO order; it does not reject a pair based on source
  identity, timestamp skew or capture labels. Rate imbalance can exhaust a
  bounded queue, which retains newer pending input.

## Metadata and association

Source, frame, clock, epoch, unit, registration, calibration, verification,
status, confidence and timing labels are descriptive. Their names, equality,
presence or changes do not provide admission authority or force a restart.
Different source/frame/clock names can use the same numeric computation path.
`motus.slam_input/1` and `/2` declarations identify supported transport formats;
the actual connected topics determine routing, not descriptive `kind`/topic
names in the declaration.

Live sidecars are optional for ordinary static mapping. Source checksums,
capture identity, verified clock relations and exact-match evidence are not
prerequisites for using a numeric cloud/pose. Computed hashes and any original
metadata remain diagnostics where available. Stored-map file integrity is a
separate storage concern.

Odometry samples are associated numerically: poses are sorted by timestamp,
the newest sample for a duplicate timestamp wins, and position/quaternion are
interpolated when a bracket exists. There is no configured maximum time-gap,
wall-age, source-name, covariance-magnitude or confidence admission threshold.
Clouds arriving before their numeric pose brackets wait in bounded buffers,
including when the lower bracket arrives late. Missing numeric brackets remain
missing computation data, not a clock proof failure. Source status labels alone do not veto an otherwise numeric pose;
`pose: null` still provides no pose to integrate.

For optional camera/body transforms, an exact numeric timestamp is preferred;
otherwise the nearest supplied numeric matrix is used without an age/skew
threshold. A matrix's status/source/clock/frame labels do not veto its numeric
use. If there is no numeric matrix, the optional body pose is unavailable and
the camera pose remains available. Dynamic cloud mounting waits for the numeric
matrix it requires; it does not manufacture one.

RGB used for loop closure follows the same format-only input policy. JPEG
decode, dimensions and numeric camera/pose/projection data must be usable.
Missing optional imagery or missing numerical projection data leaves the
pointcloud path available; it does not become a provenance failure.

## Runtime behavior

Decoded native RGB-D copies use one internal optical-frame name so native TF
lookup does not add a name-matching gate. Original input headers, numerical
timestamps and observation metadata are retained. A display/reference label
does not apply a numerical rotation.

Input data acceptance is independent of backend availability, resource limits,
session start/stop and the estimation/optimization algorithm. Native execution
still requires its installed runtime; absent numerical results are not generated
by format acceptance. No input convention is promoted into a physical accuracy,
mount calibration or clock-synchronization claim.

For optional SLAM RGB, supplied RGB/cloud poses permit numerical projection at
their distinct times. During external-pose loss the current adapter can project
coincident numerical captures using the supplied mount; it cannot infer missing
inter-capture motion for distinct times. Providers must supply physically aligned,
rectified images, optical-Z metric geometry, camera intrinsics and mount direction.
Neither a timestamp name nor a robot name substitutes for these quantities.
See [SLAM visual recovery](slam-visual-recovery.md): format acceptance is separate
from appearance retrieval, geometric validation, visual tracking and optimization.
