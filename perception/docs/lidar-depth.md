## Hardware depth adapter (`lidar_depth`)

`lidar_depth` is a model-free processor for an **existing metric hardware depth
stream**. On Tianyi, its source is the Orbbec `camera_depth` driver card. The
historical/requested card name does not mean that the Orbbec camera is a LiDAR;
this adapter does not accept LaserScan, JSON laser ranges, point clouds, or RGB.
It does not open the camera again, download a model, or use the monocular
`cal_a` / `cal_b` fit.

Enable `plugins.lidar_depth.enabled` in Perception (enabled in the shipped
configuration). No new apt/pip packages, models, Docker build steps, or driver
changes are required. Deploy the rebuilt **Perception** image, not just a new
Tianyi driver image, then refresh Perception's card list.

### Canvas wiring

```text
camera_depth -> lidar_depth -> navi depth input
head RGB -> vop ------------> navi objects input
loco_state ----------------> navi odometry input
navi ----------------------> loco_servo
```

Disconnect the monocular `visual_depth` outputs from this `navi` instance. Wire
the two `lidar_depth` output ports as depth map and, optionally, summary. A
summary alone only gives regional distances; it does not provide a per-target
pixel measurement.

For input `/nvidia_desktop/camera/head/depth`, outputs are:

| Topic | Format |
| --- | --- |
| `/nvidia_desktop/camera/head/depth/lidar_depth/visual_depth` | `image/depth-zlib` |
| `/nvidia_desktop/camera/head/depth/lidar_depth/visual_depth_summary` | `data/json` |

The extra `visual_depth` suffix is intentional: older deployed `navi` versions
recognize those suffixes before they recognize hardware `/camera/head/depth`.
The `lidar_depth` namespace avoids collisions with either the raw driver topic
or the original RGB-fed `visual_depth`. This is a wire compatibility adapter,
not a claim that every old navigation version is compatible in every respect.

### Data and lifecycle contract

Input must be `sensor_msgs/CompressedImage` with format
`16UC1; compressedDepth zlib`, containing one zlib stream of exactly 640x480
little-endian uint16 **millimetres**. The validated compressed payload and source
header are forwarded unchanged; zeros remain invalid. JPEG, PNG, malformed,
truncated, oversized and wrong-sized frames are rejected. No resizing, cropping,
registration, interpolation, hole filling, or scale correction occurs here.

The JSON summary uses the same `scale`, `unit`, `nearest_by_region`, `range`,
and `valid_fraction` contract as `visual_depth`: metres and null for unknown
regions. `source` identifies hardware depth, and timing/alignment provenance is
included. `read` samples `center` (default), `left`, `right`, or `full`, only
while a fresh valid observation exists.

`start`, `stop`, `info`, `read`, and `config` are exposed on tool `lidar_depth`
at Perception's MCP endpoint (default `http://127.0.0.1:15720/mcp`). As on other
Perception cards, `instance_id` is supplied by the canvas; preserve it for
manual lifecycle calls. `config` merges per-instance changes, retires a running
instance, and reports `restart_required`; start that same instance again to
apply the settings. Runtime `config` calls do not persist to disk. Save the
canvas configuration or the actual Perception startup YAML to retain them.

`fps` defaults to an 8 Hz ceiling. Only the newest pending frame is retained,
and nothing is republished when the input stops. `max_frame_age_ms` defaults to
500 ms; queued work and usable Unix capture stamps count toward this limit.
Zero/uptime source stamps are preserved, with receipt-based timing explicitly
reported. `info` distinguishes `waiting`, `ready`, `invalid`, and `stale`, and
reports counts, errors, configuration, and camera metadata. Stop/shutdown
signals workers before joining and removes/destroys their ROS nodes.

### Hardware verification before navigation

Hardware depth is metric but **is not automatically registered to the RGB
camera used by VOP**. The existing Tianyi driver reads Orbbec
`depth/image_raw` and may already centre-crop/resize it. This card cannot undo
that transformation or invent RGB intrinsics/extrinsics. It inherits only a
valid upstream `motus.camera/1` declaration describing the actual input image;
missing metadata stays missing, and the depth camera is never relabelled as
the RGB camera.

`rgb_alignment_verified` defaults to false and is reported in `info` and every
summary. Setting it to true is an operator attestation, **not an alignment
algorithm**. Keep `loco_servo` paused while confirming lens-to-target distance
against a tape measure at 1, 1.5 and 2 metres, checking both the image centre
and off-centre target boxes. Verify that VOP's box corresponds to the same
surface in the depth image. If it does not, configure Orbbec depth-to-color
registration and matching camera metadata at the source before using visual
target navigation. Format recognition alone does not establish that geometry.

The host tests exercise the byte contract, summary, navigation decoder,
freshness and lifecycle using ROS stubs. No Orbbec/ROS/DDS hardware performance,
RGB registration, absolute distance accuracy, or stopping distance is claimed
from those tests.
