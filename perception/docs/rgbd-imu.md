# RGB-D odometry IMU data standard

The optional `sensor/imu` input carries a `motus.imu/1` JSON object over ROS
`std_msgs/String`. The same input supports native attitude assistance, ESKF,
depth-assisted ESKF and auxiliary recovery. It can originate from a camera,
body controller, standalone IMU, replay or another producer.

## Runtime format

| Field | Required format | Meaning used by computation |
| --- | --- | --- |
| `schema` | String `motus.imu/1` | Packet layout |
| `stamp_ns` | Integer, not a boolean | Gyroscope sample time in nanoseconds |
| `accel_stamp_ns` | Integer, not a boolean | Accelerometer sample time in nanoseconds |
| `angular_velocity` | Three finite numbers | Raw angular velocity, rad/s |
| `linear_acceleration` | Three finite numbers | Raw specific force, m/s², including gravity |
| `camera_from_gyro` | Finite 3×3 matrix | Maps gyroscope axes into the image optical axes |
| `camera_from_accel` | Finite 3×3 matrix | Maps accelerometer axes into the image optical axes |
| `camera_from_gyro_translation` | Three finite numbers; required for ESKF | Gyroscope origin expressed in optical coordinates, metres |
| `camera_from_accel_translation` | Three finite numbers; required for ESKF | Accelerometer origin expressed in optical coordinates, metres |

Numeric vectors and matrices must have the listed dimensions. Timestamp values
are consumed as supplied, including zero, repeated, decreasing, delayed or
widely spaced values; they are not rewritten to the consumer's receipt time.
The consumer does not reject IMU packets on wall age, sample gap or gyro/accel
skew. Missing or malformed numeric fields are format errors.

## Provider responsibilities

These are data conventions for providers, not runtime admission gates:

- Convert angular velocity to rad/s, acceleration to m/s², translations to
  metres and timestamps to integer nanoseconds before publication. Unit labels
  do not perform conversion.
- Express both timestamps on the same numerical timeline as the RGB-D images.
  Use a monotonically increasing sample sequence for physically meaningful
  integration and provide the samples needed to interpolate image times.
- Supply proper rotation matrices. For a raw vector `v`, the consumer computes
  `v_optical = camera_from_sensor @ v` once. Do not pre-rotate a vector and also
  declare the same rotation.
- Optical axes are +X right, +Y down, +Z forward. Translation is the sensor
  origin expressed in those axes, not the reverse transform.
- Supply the real accelerometer origin for ESKF. Being rigidly attached to a
  camera does not make its translation zero. A rotation-only packet has the
  attitude format but lacks the full ESKF numeric format.
- Keep an installation fixed while an estimator uses its accumulated state.
  Source names and calibration labels cannot establish physical consistency.

The runtime checks matrix dimensions and finite values, not orthogonality,
determinant, unit labels, clock names or physical installation accuracy. The
provider is responsible for those properties.

## Diagnostic metadata

`source_id`, `epoch`, `gyro_frame_id`, `accel_frame_id`, `camera_frame_id`,
`clock_id`, `rotation_source`, `translation_source`, `calibration_id`,
`stamp_source`, `clock_mapping` and `units` are optional descriptive metadata.
They are preserved in `imu_assistance` diagnostics. Their names, values,
presence or changes do not authorize, reject or reset computation.

There is no source-name whitelist and no required mapping method, calibration
ID or verification flag. `measurement`, `read_completion`, `received`,
`receipt_estimated`, custom timestamp labels and an omitted timestamp label
all enter the same selected computation path. Receipt-based image timing also
preserves a connected IMU and the requested fusion options.

Source labels are not converted into measurement claims: a read or receipt
timestamp remains a read or receipt timestamp in diagnostics. The integration
uses its numerical differences; accepting the format does not establish
capture synchronization or physical accuracy.

## Example

The following values illustrate the wire format, not any robot's measured
installation. Replace the matrices and translations with the producer's data.

```json
{
  "schema": "motus.imu/1",
  "stamp_ns": 1794800000000000000,
  "accel_stamp_ns": 1794800000000000000,
  "angular_velocity": [0.01, -0.02, 0.03],
  "linear_acceleration": [0.0, 0.0, 9.80665],
  "camera_from_gyro": [[0,-1,0],[0,0,-1],[1,0,0]],
  "camera_from_accel": [[0,-1,0],[0,0,-1],[1,0,0]],
  "camera_from_gyro_translation": [0.02,-0.03,0.04],
  "camera_from_accel_translation": [0.02,-0.03,0.04],
  "source_id": "controller/body",
  "stamp_source": "read_completion",
  "rotation_source": "installation",
  "clock_mapping": {"method": "producer_mapping", "verified": false},
  "units": {"angular_velocity": "rad/s", "linear_acceleration": "m/s^2"}
}
```

The input queue remains bounded to 100 packets. Overflow retains the newest
queued packets and does not itself mark a numeric IMU source lost. Recovery
uses the same numeric format checks as active computation, without a separate
source-health dwell or metadata policy. Session scheduling, estimator
initialization, interpolation, visual correction and mapping behavior are
computational/lifecycle behavior, not extra IMU provenance requirements.
