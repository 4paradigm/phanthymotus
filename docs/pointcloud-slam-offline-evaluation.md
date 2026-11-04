# Offline point-cloud SLAM feasibility experiments

These tools evaluate retrieval and demonstrate graph optimization. They are not
a Perception SLAM card, do not subscribe to a robot, and never accept or publish
loop edges. A geometric match or a caller-supplied `verified` boolean is not
independent proof of a loop. Do not use synthetic graph weights on robot data.

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
