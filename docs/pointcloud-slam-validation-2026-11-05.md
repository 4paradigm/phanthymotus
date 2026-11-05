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
