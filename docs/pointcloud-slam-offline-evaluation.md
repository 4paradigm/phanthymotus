# Offline point-cloud SLAM feasibility experiments

These tools evaluate retrieval and demonstrate graph optimization. They are not
a Perception SLAM card, do not subscribe to a robot, and never accept or publish
loop edges. A geometric match or a caller-supplied `verified` boolean is not
independent proof of a loop. Do not use synthetic graph weights on robot data.

The separate `plugins/pointcloud_slam_keyframes.py` implements bounded local
keyframe storage for future integration. It assumes its caller supplies a
validated, time-aligned tracking pose and cloud already expressed in the pose
child frame. It retains the original cloud timestamp in nanoseconds and checks
that it agrees with the aligned pose's millisecond timestamp. A lost signal
blocks the current segment until a new segment is
provided. Segment IDs cannot be reused, and a full store rejects new frames
rather than evicting historical points. It does not subscribe to inputs,
establish clock synchronization, validate loops, or merge submaps; the runtime
must handle a full store and signal tracking loss. It is not a SLAM card.

## Experimental live input and graph wiring

An opt-in `pointcloud_slam` card now subscribes to `state/odom` and
`sensor/pointcloud2`. Agent-core passes the formats from live producer `info()`
keyed by resolved topic; unknown or ambiguous bindings are rejected. Canvas
connection order is not an input role. The selected odometry producer must
advertise the `motus_slam/1` pose capability in its runtime `info()`; a
velocity-only `state/odom` port is rejected. The ROS adapter checks PointCloud2 XYZ
layout and capture header; a bounded buffer waits for a compatible odometry
bracket and transforms points using `T_child_cloud`. The monotonic receipt
clock detects stream interruptions and is not substituted for capture time.
The card separately checks measurement age against its ROS clock for incoming
and queued measurements. This requires the measurement stamps to have been
verified comparable to that ROS clock; configuring a clock name or assertion
does not establish the relationship. ROS-clock behavior on the target, including
simulated time and clock resets, remains to be tested.

The card is disabled by default. To start it, `plugins.pointcloud_slam` needs
`enabled`, `cloud_frame`, `child_frame`, `clock_id`, `child_from_cloud`,
`max_gap_ms`, `max_age_s`, `max_measurement_age_ms`, `max_future_skew_ms`,
`max_pending`, `max_cloud_points`, `max_frames`, and `max_points`. It also requires the operator assertions
`clock_relation_verified: true` and `extrinsic_calibrated: true`; these flags
do not constitute evidence, so set them only after independent verification.
There is deliberately no default extrinsic or claimed safe timing limit.
The canvas exposes these fields in the card's configuration; enter
`child_from_cloud` as a JSON 4x4 matrix. Configuration is rejected while the
card is running and changes take effect on the next start. The two verification
checkboxes default to false and are operator assertions, not calibration.
When a producer changes only the *name* of a physically identical child frame
per segment, `child_frame` may contain one explicit `{segment_id}` placeholder
(for example `{segment_id}_base`). The deployment must independently establish
that the same calibrated `T_child_cloud` applies to all those frames; the
placeholder does not measure or infer the physical transform.

Keyframes create one odometry-only graph per segment. The GTSAM adapter accepts
only a consecutive odometry chain and explicit positive noise sigmas; a manual
Python call can optimize a graph captured from the input node, then rebuild each
segment's local keyframe points using the resulting pose snapshot. Returned
snapshots carry the segment frame and last cloud capture stamp; independently
anchored segments are not combined. A synthetic loop test also exercises
history rebuilding, but uses the separate experimental graph tool and does not
validate a real loop. The live backend accepts no loop edges and does not
publish corrected poses, SLAM maps, or map-to-odom corrections. The current
Perception image does not include GTSAM, and its Jetson compatibility has not
been built or tested.

For diagnostics only, set `preview_enabled: true` to advertise the separate
`sensor/slam-preview` output and a `data/json` status output under the resolved
cloud topic. The preview is a snapshot of the active segment's accumulated local
keyframes, never a merged multi-segment map. Its renderer always labels it as
unvalidated and without loop closure; its pose indicator is the last keyframe,
not a separately updated live robot pose. Each binary packet begins with a 4-byte
little-endian metadata length, followed by bounded UTF-8 JSON
`motus.slam-preview/1` (segment, source frame, cloud measurement stamp as a
decimal nanosecond string, backend and zero accepted loops), then the existing
17-byte mapping header and XYZ data. Metadata and points therefore belong to
the same snapshot; the separate status stream reports current input state.
When tracking is lost or either input becomes stale, no map snapshot is
published and status reports the input state. Repeated or backwards cloud
measurement timestamps do not refresh freshness;
clients must heed status or stream timeouts rather than treating an old display
as current.

`preview_backend: raw` uses the odometry chain without graph optimization;
`preview_backend: gtsam` invokes the odometry-only optimizer and requires a
working target GTSAM installation plus explicit `preview_odom_sigmas` (six
positive values in rotation-then-translation tangent order). Neither mode can
detect or accept loops. `preview_interval_s` defaults to 1 second (allowed
range 0.1–60 seconds), and
`preview_max_points` to 40000 (hard limit 80000); the preview is only rebuilt
when a ROS subscriber is present. The agent-core bridge subscribes to this
large stream on demand and bounds slow viewers' queues to two snapshots; the
dashboard retains only the latest preview snapshot. The
worker stops independently of upstream producers. The preview format wraps
legacy mapping packet bytes in metadata and is distinct from `sensor/mapping`
to avoid implying SLAM validity. Its full browser/WebGL and target ROS paths still need
validation. A card `running` state means subscriptions started, not SLAM ready.
The card `info()` includes input freshness, queue drops, capacity and segment
state; `tracking` there describes only the input odometry, not SLAM validity.
The ROS tests use stubs; the input adapter has not been validated on a robot.

## Inputs and reproducibility

The experimental reader currently supports the recorded
`bumi-pointcloud-slam-record/1` format: `manifest.json`, `cloud.jsonl`,
`cloud.xyz`, and (for submaps) `odom.jsonl`. Recordings stay outside Git. The
original experiment used Python 3.9, NumPy 1.26.4, Open3D 0.18.0, SciPy 1.13.1,
and GTSAM 4.2, with `OMP_NUM_THREADS=1`. Exact package availability on a target
Jetson image is not established.

Run a timestamp audit with an explicitly chosen experimental limit:

```sh
python3 perception/tools/pointcloud_slam_timing_audit.py RECORDING --max-gap-ms 200
```

This only audits pose timestamp bracketing; it does not establish clock
synchronization or a safe interpolation limit. Lost and malformed samples are
barriers, timestamps must not regress, and pose interpolation cannot cross a
segment or frame. The input decoder in this audit is specific to the recorded
Bumi protocol, not an adapter for every robot.

For a single-frame exploratory comparison:

```sh
OMP_NUM_THREADS=1 python3 perception/tools/pointcloud_slam_retrieval.py RECORDING > retrieval.json
```

The two baselines are a Scan Context-*style ring key* and mean FPFH. They are
not the full Scan Context algorithm or PCL ESF, and neither has been selected
for production. Similarity only proposes candidates; no geometry validation is
performed by this tool. The default excludes pairs within 10 seconds and uses
five results per query; both are experimental parameters, not calibrated gates.

For five-frame submaps, the user must provide an extrinsic with explicit source,
cloud and pose-child frames. The matrix is `T_child_cloud`, transforming points
from the optical cloud frame into the odometry child frame. Frames and segments
must match. All five clouds must have valid, bounded interpolation; otherwise
the center is skipped. The clock flag records an **assumption**, not evidence
of synchronization:

```sh
OMP_NUM_THREADS=1 python3 perception/tools/pointcloud_slam_retrieval.py RECORDING \
  --stride 4 --submap-frames 5 --extrinsic EXTRINSIC.json \
  --max-gap-ms 200 --assume-shared-clock > submaps.json
```

The JSON extrinsic contains `cloud_frame_id`, `child_frame_id`, `provenance` and
`child_from_cloud` (a 4x4 matrix). `provenance` must state whether the transform
was measured or assumed. An assumed transform is useful only for exploratory
experiments. Different sensors or robots require their own calibrated transform.

An optional independent labels JSON is a list of entries such as
`[{"query": 10, "positive": [305]}]`, using original recording indices. This
format is illustrative: it is **not a claim** that those two frames have been
independently labelled. A run rejects labels for frames it did not process and
reports recall only for supplied labels. Self-matches are forbidden. Top-k
retrieval recall is not a geometric-verification false-positive rate; a labelled
negative-pair evaluation and independent threshold set are still required.
`--candidate-separation-s` optionally suppresses candidates close in time to
already selected candidates, after ranking by descriptor distance. It defaults
to zero (disabled); an experimental value does not establish loop validity or
select a calibrated production threshold.

The reciprocal ICP diagnostic accepts an externally supplied initial transform
and reports directional fitness, inlier RMSE, correspondence counts and cycle
error. It never accepts an edge. For example, using the earlier exploratory
pairwise registration result as the initial guess:

```sh
OMP_NUM_THREADS=1 python3 perception/tools/pointcloud_slam_registration_experiment.py \
  RECORDING 10 305 --initial PREVIOUS_RESULT.json \
  --voxel-m 0.08 --max-distance-m 0.08
```

Reusing an earlier estimate is not independent validation; the initial result
must name the same source and target indices. A no-correspondence result reports
unknown RMSE rather than a misleading zero.

For a later independently labelled set, `pointcloud_slam_pair_benchmark.py`
reads a JSON spec with explicit `voxel_m`, `max_distance_m`, `thresholds`, and
`pairs`. Every pair supplies original recording indices `source` and `target`,
an independently established boolean `match`, and a 4x4 `initial` transform
mapping source points to target points. Thresholds require `min_fitness`,
`max_rmse_m`, `min_correspondences`, `max_cycle_translation_m`, and
`max_cycle_angle_deg`. Run:

```sh
OMP_NUM_THREADS=1 python3 perception/tools/pointcloud_slam_pair_benchmark.py \
  RECORDING LABELLED_SPEC.json > pair-results.json
```

The report hashes the recording, spec, tools, and records the Open3D version.
Its recall and false-positive rate describe only the provided pairs; they are
unknown when the corresponding positive or negative class is absent. Threshold
passing is an experiment, not an accepted loop. Reciprocal ICP can still match
unrelated but similar geometry; labels must not be derived from odometry or the
same registration result, and thresholds need separate evaluation data.

`pointcloud_slam_candidate_graph.py` provides a separate synthetic/offline
experiment connecting the explicit geometry gates to a robust GTSAM factor and
rebuilding history. It inverts the source-to-target ICP transform for the
corresponding GTSAM between factor. It does not feed the live card or accept
real loop edges; its core is packaged in `plugins/pointcloud_slam_experimental_backend.py`,
but noise, gates and initial guesses remain uncalibrated. Its
computed map-to-odom correction is an experimental output, never published or
fed back to raw odometry.

## Bumi exploratory run, 2026-11-04

The 2026-11-02 recording contains 317 cloud frames, one reported odometry
segment, and no independently labelled loop/non-loop pairs. The operator
estimated the end was within a few centimetres of the start but the recorder
ended before the operator's completion message, so the end frame is not
verified ground truth. The original archive SHA-256 is
`053c2767e75fd52ffd8e257b5b8495788df9b548014694378206f18dfa9a928e`.

- With a **hypothetical**, unvalidated 200 ms maximum pose bracket, 199/317
  cloud times align and 118 are rejected. The common-clock assumption itself
  still needs independent verification.
- For the 3412 recorded tracking samples containing both `stamp_ms` and
  `vendor.published_ns`, the recorded publication-minus-measurement difference
  had median 278.62 ms, p95 390.37 ms, p99 432.26 ms and max 560.55 ms.
  This is a historical publisher-side distribution, not end-to-end latency,
  synchronization proof or a production age threshold; it assumes the two
  recorded fields are comparable and the millisecond stamp is rounded down.
- A further bracket-width sensitivity sweep on the same recording produced
  0, 69, 180, 199, 301, 316, 317 and 317 aligned frames at 50, 100, 150,
  200, 250, 300, 350 and 500 ms respectively (317 frames total). A wider
  bracket improves coverage but does not prove interpolation accuracy. These
  counts must not be used to choose a production threshold without measured
  timing and motion-error bounds.
- Both single-frame descriptors processed all 317 frames. For the exploratory
  pair 10 and 305, neither method ranked the other frame in its top five in
  either direction. There are no labelled queries, so recall is unknown.
- Initialized from the prior exploratory 10→305 ICP result, reciprocal ICP
  produced fitness 0.618 / 0.648 and cycle error about 0.011 m / 0.125°. Because
  the two directions share that initial estimate and the pair is unlabelled,
  this is a diagnostic only and accepted-loop count remains zero.
- With stride 4, five-frame submaps and the **assumed** Bumi level-forward
  transform from the recorded calibration ID, the 200 ms experiment retained
  10 of 80 requested centers. Repeating at 350 ms retained 78 of 80. The larger
  gap is a sensitivity test, not a demonstrated safe choice; widening it can
  increase motion distortion and synchronization error.
- A synthetic GTSAM test demonstrates that adding a caller-supplied loop edge
  moves historical poses and changes the map rebuilt from local point clouds.
  It does not prove actual loop detection, robustness, cross-segment merging or
  the quality of a real map. An unverified edge is rejected by the experiment.
- On the development Mac, the 317-frame single-frame run took 80.47 s and
  peaked at 184.9 MB process RSS; the two submap runs took 11.56 s / 224.2 MB
  and 65.50 s / 222.1 MB respectively. Runs overlapped with each other, and
  these totals include descriptor generation for both methods. They are not
  per-frame latency guarantees or Jetson performance measurements.

The two sets of descriptor scores and rankings are saved with the local Bumi
recording as `retrieval-exploratory.json`,
`retrieval-submaps-exploratory.json`, and
`retrieval-submaps-gap350-exploratory.json`. The nominal extrinsic is saved
beside them as `extrinsic-assumed-for-experiment.json`; none of these establish
measured calibration or acceptance. Collect independently labelled revisits and
similar-looking non-revisits, assess geometric validation and compare against
an RTAB-Map pointcloud-only baseline before freezing the retrieval algorithm.
The reciprocal ICP result is `reciprocal-10-305-exploratory.json` in the same
local recording directory.
