# Bumi SLAM compatibility and build validation — 2026-11-05

Status: experimental; no real SLAM acceptance or validated loop closure.

## Baselines and integration

Read-only GitHub metadata and clean worktrees were checked before editing:

| Source | Observed head | Treatment |
| --- | --- | --- |
| SLAM PR #371 | `98fb43249e9bd64c6cb1348329940125792d3574` | Work only on `feat/pointcloud-slam-offline-eval` |
| Pointcloud/Core PR #367 | `7b1a3ec077cb009308dfe96d50d392b92be30eff` | Selectively port depth-zlib and odometry-summary changes |
| Driver PR #471 | `fc910c713c0afcd798188db3b96f4f30ca5a783a` | Read only; no deployment or branch changes |

PR #269 and #459 were not modified. No force push or upstream branch replacement.
A direct patch of #367 does not apply: the SLAM branch already has subscription
selection, metric skip diagnostics, stop guards and different odometry rendering.
Keep those protections. Port only qualified compressed input and vendor summary
fields; keep the existing renderer layout and tracking-only pose/summary display.

## Input compatibility gates

| Input | Required checks | Result scope |
| --- | --- | --- |
| Deployed f32 | `32FC1`, metres, measured intrinsics, valid capture stamp and physical frame | Existing baseline retained |
| Driver #471 zlib | `motus.camera/1`, `image/depth-zlib`, measured 640x480 mm, pinhole/derived-from-K, mapped capture timestamp | Host transport/geometry tests; not deployed |
| Old display depth or unqualified zlib | Missing capture/measurement declaration | Refused for metric computation |
| Malformed frame | Missing frame, zero stamp, short/extra decompressed payload | No metric publication; subsequent valid frame recovers |

Qualified zlib subscribes only to CompressedImage; declared f32 subscribes only
to Image. The input capture header reaches PointCloud2 unchanged. No arrival
time substitution, VO reimplementation, camera IMU or RSUSB changes.

## Bumi observations

SSH read-only preflight found the expected running images:

- Perception: `local/phanthymotus-perception:pr367-31a0b82-jp61`.
- Driver: `release.261105.ef1cd0f`.
- Core: `release.261105.2a74292`.

Approximately 42 GiB disk and 11 GiB available RAM; no leftover build/compiler
process before the new isolated build. No service container was switched and no
motion command was sent. The operator has sole control of robot motion.

A 20-second subscriber-only observation of the running input chain received 38
depth frames, 32 clouds, and 330 odometry samples (all tracking). All 32 cloud
headers matched a depth stamp and `ubuntu_rgb_optical` frame; the last cloud had
18,963 points. All 32 had same-segment bracketing odometry. Bracket span median
233 ms, maximum 267 ms; 18 exceeded 200 ms and 3 exceeded 250 ms. This is an
observation under compilation load, not a calibrated clock relationship or a
justification to relax the interpolation threshold. No independent motion or
endpoint ground truth was collected.

## Build provenance

At initial recheck, fork branch and PR metadata reported `98fb432`, but the main
repository `refs/pull/371/head` still reported `62da1a9`. The review worker's
`create_worktree` merges the latter ref while using the metadata SHA in its
job name. The previous build log actually ran the old GTSAM git clone and
failed TLS. Its Core-only result cannot validate this branch's Perception code.
The latest existing request was still Building; it was not duplicated.

An isolated native test overlay uses the existing Perception image, offline
ARM64 CPython 3.10 wheels and GTSAM source checksum
`48c760e17a8b6c0e59fcc1d6a71602907fedbb9c39ba400d13f998dab1fd5080`.
It runs with no build network, CPU quota 200000 and memory 6 GiB. It is named
`local/phanthymotus-perception:slam98-offline-test`, based on the pre-integration
98fb432 source context, and is not the formal PR image. Compilation progress
alone does not establish GTSAM availability. Final import, ICP and graph checks
and target lifecycle/archive tests remain required.

## Validation and remaining gates

- Pointcloud plugin: 143 passed before four additional malformed-zlib cases;
  the focused zlib suite including those cases: 8 passed.
- Core full suite: 1710 passed, 19 subtests passed (one dependency deprecation warning).
- Host SLAM algorithm suite with NumPy 1.26.4, Open3D 0.18.0 and GTSAM: 173 passed.
- Initial Perception full suite: 2525 passed, 66 skipped, 3 failed, 69 subtests
  passed. The failures were `test_speak_without_a_language_reads_by_context`,
  `test_auto_is_told_the_previous_utterances_language`, and
  `test_unknown_reading_language_is_refused_before_queueing` in
  `test_vits2_tts_plugin.py`; an isolated rerun reproduced all three with HTTPS
  ACP completion callback timeouts. Rerunning the full suite with
  `AGENT_CORE_URL=https://127.0.0.1:1 NO_PROXY=127.0.0.1,localhost` passed:
  **2532 passed, 66 skipped, 69 subtests passed**, six existing warnings.
  This includes the four added malformed-zlib cases. No TTS source was changed.
  Python 3.12 lacks Open3D/GTSAM; their relevant SLAM tests ran separately in
  the Python 3.9 dependency environment above, not on JetPack.

Do not set `extrinsic_calibrated` or `clock_relation_verified` for live Bumi:
measured extrinsics, independent clock evidence, labelled real similar-place
negatives, endpoint truth and real loop/history correction acceptance are absent.
Target lifecycle/archive tests must be labelled synthetic when using synthetic
calibration/timing. Lost stops fusion; segments remain separate unless geometric
verification proposes an experimental merge. No correction replaces raw odometry.

Only after dependency/build checks and real input gates are satisfied should the
operator drive labelled loops and negatives. Record proposed/correct loop edges,
pre/post trajectories and historical maps, and measured endpoint errors separately.


## Target isolation checks on the deployed base image

The operator confirmed that calibration/clock/negative-label evidence is not yet
available and requested continued isolation testing. No live verification flags
were changed.

A `git archive` snapshot of `1742cc4` was mounted read-only into disposable
containers based on the deployed Perception image. Containers used `--network
none --cpus 1 --memory 1g`; ROS checks used a separate `ROS_DOMAIN_ID=173`.

- Target Python 3.10 focused tests: **205 passed, 15 skipped**. Use
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; automatic image pytest plugins initially
  pulled in GTSAM-dependent tests during collection, blocking even dependency-free
  tests. The remaining 15 skips explicitly require missing GTSAM/Open3D.
- `pointcloud_slam_target_lifecycle.py`: real rclpy node creation/destruction,
  synthetic message callbacks, missing-calibration gate, capture-stamp preservation,
  wrong-frame rejection, lost handling, independent segments, save/stop/restore,
  and restored history requiring a new segment all passed. The source sentinel
  node survived card stops. This check does not test DDS transport.
- `pointcloud_slam_target_depth_transport.py`: real DDS publisher/subscriber
  transport passed separately for both qualified zlib and f32. Each synthetic
  640x480 constant-2m frame yielded 19,200 metric points; the first point was
  (-1.6, -1.2, 2.0) m. Source capture headers matched exactly. Input format
  reporting and stopping only the card's nodes also passed.

The scripts are under `perception/tools/`. Run only in the isolated container,
with the perception source root in `PYTHONPATH`, after sourcing the ROS setup.
Their synthetic calibration and clock settings describe the generated fixtures;
they provide no evidence about Bumi's physical extrinsic or sensor clocks. They
create no motion publisher. Geometry backend dependency/preview validation is
still separate from these base-image checks.


## Open3D target dependency failure and correction

The first offline build reached GTSAM Python binding compilation. An independent
check in the same base image then reproduced `import open3d` failing with
`ModuleNotFoundError: plotly`. The ARM64 wheel declares Dash/Werkzeug/nbformat
and imports Plotly/Dash unconditionally through visualization, even for ICP.
The previous `--no-deps` wheel install plus three small packages was insufficient.
That build was explicitly cancelled (exit 130), not recorded as a successful
GTSAM image or compiler failure.

The Dockerfile now installs pinned Dash 2.18.2 and nbformat 5.10.4 with their
resolver-managed dependencies, retains the NumPy ABI pin, and imports Open3D
**before** compiling GTSAM. On this Ubuntu base, installing Flask also encountered
a non-uninstallable distutils blinker 1.4; install only blinker 1.9.0 with
`--ignore-installed`, shadowing it under /usr/local without removing distro files.
A fresh network-isolated target container with the corrected dependency recipe
passed Open3D 0.18.0 import and ICP and retained NumPy 1.26.4.

The second local offline build separates dependency preflight, cached GTSAM
compilation, and final import/ICP/graph checks. It retains the 2-core/6-GiB limits
and no build network. GTSAM availability remains unverified until its final
checks succeed. The operator will report the GitHub build result; this chat no
longer polls that build.


## Formal build 1742cc4 idle-timeout diagnosis

The operator reported the result; inspection of PR comment 6656005306 showed:
`[agent] Build killed: no output for 600s (908s into the run)`.
This is the build agent's idle watchdog, not a reported OOM or compiler failure.
The actual log now contains codeload/checksum code, confirming the stale-ref
problem was bypassed for this build. Last output was Open3D installation at
290.3s; there is no archive checksum or CMake output. The next command is silent
curl, so a stalled archive transfer is the supported hypothesis, not a proven
compiler hang. Connect timeout alone did not bound the response-body transfer.

The download now has a 120s per-attempt maximum, a 240s retry budget, a 30s
low-speed timeout below 1024 bytes/s, and an explicit stage log. The retry budget
is checked between attempts, so an in-progress final attempt can run past 240s;
the per-attempt cap still prevents reaching the 600s idle watchdog. SHA-256
validation remains mandatory. No watchdog setting was loosened or fake heartbeat
added. A local stalled-body HTTP reproduction showed the old connect-only command
outliving a 3s harness, while scaled transfer/retry limits exited with curl28
within the 5s test bound.

The full production timeout/retry flags also successfully downloaded the pinned
archive on the host; its SHA-256 matched
`48c760e17a8b6c0e59fcc1d6a71602907fedbb9c39ba400d13f998dab1fd5080`.
This verifies the normal path on the host, not connectivity from the build runner.


## Formal build 0d2cb78: slow transfer, not compiler failure

The next operator-reported result (PR comment 6656702999) materially refines the
earlier hypothesis. Open3D import completed and the download stage was reached.
Curl reported exit28 after each 120s attempt, receiving 3,987,475 and 3,737,945
bytes respectively out of 29,687,819. The 120s cap was too short for this runner's
observed roughly 31–33 KB/s average transfer rate. No GTSAM compile had started.

The revised command exposes curl's real byte-count/speed progress (removes silent
mode), retains the 30s/1024B/s low-speed bound and 20s connect bound, and permits
1800s per attempt with a 2400s retry budget. A final attempt can extend beyond the
retry budget; the absolute build cap remains in force. The review worker reads
output chunks, so carriage-return progress updates refresh its idle timer. This
is transfer telemetry, not an unrelated heartbeat; the 600s watchdog is unchanged.
At the observed rate the 29.7MB archive needs roughly 15–16 minutes, within the new
per-attempt budget. Future network behavior remains unverified.

A controlled slow-body test reproduced curl28 under a short budget; the revised
progress command completed and verified 100,000 bytes in about 3 seconds
with multiple genuine progress updates. The pinned archive/checksum is unchanged.


## Completed Bumi offline target validation

The resource-limited, network-isolated native build completed with exit 0.
The GTSAM build/install step took 2381 seconds, including serial LTO over 128
partitions; quiet LTO was making progress and was not a failed build. Final
image checks passed imports, identity ICP, and a one-node Pose3 optimization.
This is a local test overlay, not an official PR image or SLAM acceptance.

A separate runtime overlay removes inherited application modules and copies
snapshot `187c612beaac5ac6a2c5bdc36ba29f53c1310886`; all runtime source directories
remain identical through `864ec5e`. Byte/hash comparison matched all 129 files
in plugins, utils, deploy, main.py and config.yaml against the committed snapshot.

- Image: `local/phanthymotus-perception:slam187-offline-test`, ARM64.
- Docker image ID: `sha256:3e07006d63f2816e2058402f1a011f7ab6779d4b29689c9a880fef5563a8dace`.
- NumPy 1.26.4, Open3D 0.18.0, GTSAM distribution 4.2a9, built from the pinned
  source commit/archive described above.
- Target Python 3.10 regression: **319 passed, 1 skipped** in 40.04 seconds,
  covering all `test_pointcloud_slam_*.py` and `test_pointcloud_plugin.py`.
  The skip requires fake-cv2-only SGBM stubs; actual OpenCV is present here.
  Pytest plugin autoload was disabled; no unavailable-geometry skips remained.
- Both committed target scripts passed again against `/work` in this image:
  real-rclpy lifecycle/fault/archive checks via direct callbacks, and real DDS
  synthetic f32/zlib contracts (19,200 points each, exact capture headers).
- Runtime tests used no network, one CPU and 1–2 GiB memory. Existing Perception,
  Driver and Core service containers were not replaced. No motion was commanded.

Robot evidence logs: `/tmp/bumi-slam-build-98-offline-v2.log`,
`/tmp/slam187-target-tests.log`, `/tmp/slam187-target-runtime.log`.
Physical extrinsics, independent clocks, labelled scene negatives, actual loop
closures, corrected historical maps and endpoint ground truth remain unverified.

The formal build provenance problem remains open: after pushing `864ec5e`, PR
metadata/fork advertised that commit but the upstream pull head still returned
`0d2cb78`. Request comment 6656953463 was followed by explicit provenance warning
6657004807. The labelled SHA alone must not establish which Dockerfile was built.
No repeated full build was requested while that job was outstanding.


## Authorized deployment and read-only calibration audit

At the operator's explicit request, Perception was switched to the local
`slam187-offline-test` image and `pointcloud_slam.enabled` was persisted as true
in `/opt/phanthy-motus/perception-slam-config.yaml`, mounted read-only at
`/work/config.yaml`. The tool is discoverable and reports idle. This enables
card discovery, not fusion or a claim of calibrated inputs.

The canvas startup failure was traced to the old Core lacking `input_bindings`
and `odom_interfaces`, despite receiving both topic names. After inspecting the
actual candidate image source, Core was upgraded through `/api/system/update`
and its restart helper to `release.261105.8c424ea` (registry digest
`sha256:5a3a6f229b2a0d8d36ecc2982bc7e4e6e2e7227cf03ed63f2d5b27cd5a14d378`).
Helper exited 0; the running Core contains both bindings and declaration paths.
Ten camera-info/SLAM startup contract tests passed in a separate network-isolated
container of this image with mocked tool dispatch. Pytest autoload was disabled
because the system pytest and installed anyio plugin were incompatible. No live
whole-canvas start or motion command was issued. Driver remains unchanged.

Backups: `/opt/phanthy-motus/docker-compose.before-slam187.yml`,
`docker-compose.before-slam-enable.yml`, and `/tmp/slam-compose-before-core.yml`.
The latter retains the enabled SLAM deployment with the previous Core image.

Two 20-second subscriber-only samples checked camera parameters, static TF and
capture provenance. The detailed second sample observed 150 camera metadata
messages, 34 metric depth frames, 36 point clouds and 358 odometry messages:

- RGB optical frame: `ubuntu_rgb_optical`, 640x480. K=(fx 603.85986328125,
  fy 603.5826416015625, cx 324.97772216796875, cy 250.24855041503906), D=zeros.
  SDK depth scale 0.0010000000474974513 m/unit; published depth is 32FC1 metres.
  These are runtime/SDK declarations, not a physical accuracy calibration.
- The operator confirms a shared optical origin. Active-segment TF agrees with
  zero translation but has quaternion xyzw=(-0.5,0.5,-0.5,0.5). Thus the axes
  differ: T_child_cloud rotation rows are (0,0,1),(-1,0,0),(0,-1,0).
  The running VO explicitly reports `body_axes` and assumed-level-forward
  calibration. This verifies the software coordinate convention, not a measured
  body origin or mounting calibration. Identity rotation would be incorrect.
- Static TF retains 32 segment-specific parents for the optical child. Only
  the current odometry segment's transform was used in this audit; a generic
  global TF lookup must not be assumed unambiguous.
- All 150 metadata samples report `global_time`; each mapped stamp exactly
  equals int(SDK source_ms*1e6), with no receive-time substitution. RGB/depth
  skew was 0.017664–0.018176 ms. Receive-minus-RGB-capture age was 53.09–69.22 ms.
- All odometry stamp_ms values preserve visual_stamp_ns truncated to ms.
  106 of 116 distinct visual stamps matched received RGB frames; 33 of 36
  clouds matched received metric-depth stamps. Unmatched observations were
  not declared validated; bounded sampling/DDS coverage is incomplete.
- Odometry was tracking; capture-to-publication age was 224.73–505.86 ms
  (median 319.38 ms). Thresholds were not relaxed. Controller-IMU rotation
  assistance still reports `read_completion`, so independent timing of the
  full estimator input chain remains unverified.

Raw local evidence: `/private/tmp/slam-calibration-audit-result.json` and
`/private/tmp/slam-calibration-audit-detailed.json` (also under `/tmp` on Bumi).
No calibration verification flags were set. Physical depth/registration accuracy
needs a measured target; a known-size board or flat target with reliable distance
measurements is the next operator preparation. Motion sampling must be announced
and remains under operator control.


## Formal build 30736ef: HTTP/2 stream failure

PR result comment 6657255497 confirms the corrected curl progress/1800s/2400s
command was actually used. The transfer then failed with curl92, HTTP/2 stream
CANCEL, at build time 864.8s (download began at 180.8s). This is neither the idle
watchdog nor GTSAM compilation. Core succeeded as release.261105.c66214b;
Perception still has no successful formal artifact.

The source download now requests HTTP/1.1 and enables --retry-all-errors so
transport failures outside curl's default retry set are retried, retaining all
existing time/low-speed bounds and the mandatory SHA-256. The operation is a
read-only GET to a regular output file. A controlled truncated-body test showed
the previous flags failing with curl18 after one request; the revised flags
retried and produced the exact complete payload in two requests. This exercises
partial-transfer recovery, not an exact reproduction of the remote HTTP/2 fault.
Target curl 7.81.0 supports both flags. Runner connectivity remains unverified
until the formal build succeeds.

The same truncated-transfer test also passed using curl 7.81.0 in a separate
network-isolated Bumi container. A real HTTP/1.1 archive download on the host
completed and matched the pinned SHA-256
`48c760e17a8b6c0e59fcc1d6a71602907fedbb9c39ba400d13f998dab1fd5080`.

## Simple-start trial deployment change

The operator authorized a direct trial without new calibration work and asked
that technical parameters not appear in the ordinary configuration panel.
An explicit device-side `input_policy: experimental` now exposes only preview,
archive name and restore controls. Hidden legacy canvas settings cannot override
that profile. Default strict behavior is unchanged. Experimental policy and the
actual (false) verification flags are included in info/preview metadata and
archive configuration; strict restore rejects an experimental archive.

The Bumi profile is `perception/configs/pointcloud_slam_bumi_trial.yaml`. It uses
only the audited same-origin axis convention, with a 400ms interpolation bound,
1500ms measurement-age bound and bounded storage/candidate processing. These are
trial settings, not validated motion thresholds. Other robots require their own
deployment profile; generic automatic TF discovery is not implemented here.

Startup does not require a first pose: subscriptions enter running/waiting, late
tracking inputs can create keyframes, and lost still stops integration. The
existing format/capability, capture-time, frame, segment and capacity checks stay
in effect. No Driver or visual-odometry lifecycle code is modified.

Validation of this change: all 175 SLAM tests passed on the host and in the
network-isolated Bumi target environment. The deployment profile also started
with real rclpy nodes and the experimental geometry backend before any input
frame existed, reported waiting, and stopped cleanly. An initial profile check
accidentally imported the old image source via /work; it correctly rejected the
unverified profile. Repeating with the source snapshot as working directory
verified the intended changed implementation. Narrow independent review reported
no actionable correctness findings.

First live trial exposed an existing ROS-clock name collision: SlamInputNode's
`self._clock = time.monotonic` overwrote rclpy Node's own `_clock`, so actual
measurement callbacks raised AttributeError. Earlier synthetic lifecycle tests
replaced the measurement clock and missed it. The live SLAM instance was stopped;
receipt timing is now named `_receipt_clock`. The real-rclpy lifecycle check now
asserts the untouched ROS clock and default measurement clock before injecting
synthetic timestamps. No producer or odometry code was changed.

### Preview starvation repair (after f00acc4)

The live input store continued growing while a 20-second subscriber observation
received neither preview nor status. Geometry processed every queued frame,
repeated graph optimization and rebuilt both complete maps per candidate; status
shared this worker. A completed snapshot older than the input measurement-age
limit was silently discarded whenever new input had arrived. Later inspection
also found the independent 300-keyframe storage limit had been reached.

Online retrieval now computes descriptors for the first anchor and newest frame
of a batch, retaining all frames/poses but explicitly reporting unindexed frames.
There is no deferred descriptor queue. Intermediate candidate steps omit map
materialization. The final preview samples the global point index before
transforming points (40,000 points on Bumi); retained history and offline full-map
outputs remain intact. This trades retrieval coverage for bounded online work;
it does not validate the candidates. A separate worker reports input health and
elapsed computation while geometry is busy. Completed historical snapshots keep
their capture stamp and clear current pose/correction output when too old. Lost,
segment changes and invalid inputs still suppress map publication.

Resource-limited Bumi comparison: 2 CPUs, 2 GiB, no network, same eight captured
clouds (~18.9k points each) replayed with a **synthetic stationary trajectory**,
four added frames per update. These are performance fixtures, not loop truth.
Old code: 20 updates through 80 frames, median 2.156 s, p95 2.975 s, max 3.015 s.
New code: 75 updates through 300 frames, median 0.514 s, p95 0.675 s, max 2.116 s
(first initialization). At 300 frames: descriptors 236 ms, registration/graph
212 ms (optimization 204 ms), bounded map 147 ms, total 596 ms. Raw logs retained
on Bumi at `/tmp/slam-perf-old-batch.jsonl` and `/tmp/slam-perf-fixed.jsonl`.
Single-thread OpenMP did not improve the observed descriptor latency and was not
added to the deployment configuration.

Validation: 179 SLAM tests pass on both local real-geometry environment and Bumi
Python 3.10/Open3D/GTSAM, including slow compute status, aged snapshot starvation,
map-budget historical coverage, lost/segment protection and bounded retrieval.
The isolated real-rclpy lifecycle/archive restoration check also passes. Initial
target test invocation accidentally mixed mounted and installed namespace-package
modules; corrected working directory/PYTHONPATH selects the mounted source first.
Code review found a status-thread timestamp race, fixed by capturing one local
compute-start value. No remaining blocking findings.

Limits: the 300-frame/6-million-point retention cap still intentionally stops
fusion without discarding history; this is not unlimited mapping. Connected
multi-segment relocalization still has a full-map rebuilding path and is **off**
in the Bumi profile, so the bounded-map performance result applies to that
single-segment configuration. No Driver/odometry code or source timing was changed.

Deployment verification for runtime 67464c6: local image
`local/phanthymotus-perception:slam-67464c6-trial`, 18 source files byte-matched;
37 focused tests also passed against the built image. Only Perception was
recreated. Driver and Core container IDs/start times remained unchanged. The
previous 300-frame session was explicitly saved and backed up on Bumi as
`/tmp/slam-before-67464c6.json` (323 MiB) before starting a new trial session.

A 120.049-second read-only live DDS observation received 193 clouds, 39 preview
packets with 39 strictly increasing capture timestamps and distinct packet
hashes, and 76 status messages. Odometry stayed tracking. Preview interval:
median 2.941 s, maximum 3.787 s. Compute time: median 829 ms, maximum 1688 ms.
Each preview contained at most 40,000 points. Historical-snapshot output was
observed when compute exceeded freshness, rather than silently dropping all map
updates; these outputs did not claim a current pose. Raw report is
`/tmp/slam-live-progress-67464c6.json` on Bumi. No movement commands were sent.
This verifies continued publication during the measured window, **not** unlimited
map growth: the operator correctly noted that the remaining 300-frame global
retention limit is inadequate for full-map sessions. Storage/submap lifecycle
and actual keyframe selection require further work; increasing a constant is
not sufficient. Do not describe complete-map acceptance as finished.
