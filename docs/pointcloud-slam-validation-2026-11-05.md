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
