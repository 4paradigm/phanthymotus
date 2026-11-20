# PR #393 verification — 2026-11-14

Evidence root: `/home/utopus/phanthy-work/artifacts/roi-pr393-fix-20261114/`. Initial local/remote HEAD: `50da4cdf94d038c72b7af000c59cdaa41864be19`. Latest checked upstream: `7b16c433463dcb7017eb95b5bb40ef99297d3d34`. Older 2026-11-13 counts are historical, not current verification.

| Check | Actual result |
|---|---|
| Unmodified candidate, Python 3.8 | 10 failed / 56 passed, exactly the official ten ROI failure names |
| Fixed focused suite, Python 3.8 | 74 passed |
| Fixed focused suite, system Python 3.10.12 venv | 74 passed |
| Real Humble 3.3.22, x86_64 native libraries | 10 synthetic ROS/DDS integration checks passed |
| Chromium real loopback MCP and unchanged upstream Workbench host | Six drag/four-step/three-state/evidence scenarios passed |
| Complete branch Perception, Python 3.10 | 2327 passed, 2 failed, 40 skipped; 69 subtests passed |
| First latest-main merged Perception, Python 3.8 | 2327 passed, 3 failed, 39 skipped |
| Final latest-main merged Perception, Python 3.8 | 2329 passed, 1 failed (music), 39 skipped |
| Untouched latest upstream, full Perception Python 3.8 | 2255 passed, 1 failed (same music), 39 skipped |
| Core registry/MCP/skill regression | 37 passed |
| Object #374 disposable trial merge | Clean merge; 147 Object tests passed |
| Installed kit, cross-process native DDS in WSL | NOT PASSED: graph/type matched, no samples; independent minimal subscriber also received zero |
| Real D435, lab camera, ARM64 runtime | Not tested |

The first branch full run failed only existing music provider and VITS2 tests. The first merged run failed music and two visual-depth timing tests. These are disclosed, not counted as successes. Latest baseline and confirmation results, Core/Object compatibility and kit installation are recorded in DELIVERY.md and immutable `verification/*.log`, `*.result.json`, and XML evidence.

## Official 11 failures

The official [test report](https://github.com/4paradigm/phanthymotus/pull/393#issuecomment-6875179725) ran Python 3.8 in the JetPack 5 image. The first ten below all reproduce before the fix and pass unchanged after the fix in both 3.8 and 3.10:

| Official test (`perception/tests/`) | Treatment |
|---|---|
| test_tabletop_roi_field::test_evidence_saves_images_coordinates_parameters_times_and_hashes | Fixed; complete after evidence and checksum export pass |
| test_tabletop_roi_goal_verifier::test_roi_rgb_end_to_end_without_classes_or_detector[True-PASS] | Fixed; stable change PASS |
| test_tabletop_roi_goal_verifier::test_roi_rgb_end_to_end_without_classes_or_detector[False-FAIL] | Fixed; unchanged FAIL |
| test_tabletop_roi_goal_verifier::test_vop_failure_does_not_affect_roi_preview_or_verification | Fixed; no detector/VOP call |
| test_tabletop_roi_goal_verifier::test_compressed_ros_image_callback_to_roi_without_detector | Fixed; JPEG callback and capture ordering |
| test_tabletop_roi_goal_verifier::test_uncomparable_roi_images_do_not_pass[unstable] | Fixed; instability refused |
| test_tabletop_roi_goal_verifier::test_idempotence_and_conflicts_including_concurrent_capture_and_retired_tokens | Fixed; receipts and concurrent retry assertions retained |
| test_tabletop_roi_goal_verifier::test_local_brightness_compensation_and_insufficient_stability | Fixed; compensation and insufficient-frame checks retained |
| test_tabletop_roi_review::test_failed_new_after_capture_does_not_export_an_old_after_image | Fixed; incomplete latest evidence never borrows old after image |
| test_tabletop_roi_review::test_two_sessions_keep_independent_baselines_and_cancellation | Fixed; isolation retained |
| test_vits2_tts_plugin::test_speak_without_a_language_reads_by_context | Naturally reproduced on untouched latest upstream; no speech edits |

The directly observed common ROI cause is `itertools.pairwise` missing on Python 3.8. Several assertion/export errors are downstream of that exception. No expected result, assertion or skip was weakened. The CompressedImage synthetic fixtures now use required Domain 42; they still never initialize DDS.

## ROS diagnosis and native integration

Old Doctor reproduced the exact field `AttributeError: __enter__` and secondary `_sigint_gc` destructor error using actual Humble native libraries in a Python 3.10 venv. Full traceback: `verification/humble-before-traceback.log`. Root: private context initialization followed by default-context global `spin_once`; the global context is uninitialized. Explicit `SingleThreadedExecutor(context=context)` fixes it. The venv itself works with correctly sourced ROS package/library paths; no ROS/Python replacement is required.

Humble debs were downloaded/extracted only into a personal task root, not installed. Open Robotics signature and all 77 ROS deb hashes were checked against the signed index (`humble-signature.log`, `humble-packages.json`). Failed official repository TLS and initial extraction-path attempts remain in logs. Runtime NumPy takes precedence over extracted distro dependencies.

The standalone tool `perception/tools/tabletop_roi_ros_smoke.py` runs outside pytest because shared conftest installs fake ROS modules. It uses real rclpy/RMW/FastDDS, official loopback XML, Domain 42 and unique synthetic image topics. Checks: Doctor JPEG/private Context; Sources subscription; stop/restart; silent publisher; corrupt JPEG; invalid timestamp; missing publisher; wrong publisher type; partial initialization cleanup; untouched global Context. All images are synthetic. It does not access a camera or actuator.

```bash
ROI_TEST_PYTHON=/path/to/python3.8 bash perception/plugins/tabletop_roi_goal_verifier/deploy/run-tests.sh --tb=short
ROI_TEST_PYTHON=/path/to/python3.10 bash perception/plugins/tabletop_roi_goal_verifier/deploy/run-tests.sh --tb=short
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/agent-core/deploy/dds-local.xml"
python3 perception/tools/tabletop_roi_ros_smoke.py --output /path/to/new/ros-smoke.json
python3 perception/tools/tabletop_roi_browser.py --chromium /path/to/chrome --output /path/to/new/browser
```

New deterministic timing tests advance an explicit synthetic clock while acquisition waits, then check legitimate new frames, future frames, repeated frames and expiry during acquisition. Existing marker, stale/rollback/source geometry, session isolation, corrupt-image and evidence-write tests remain active. The native smoke confirms private ROS lifecycles; D435 identity, actual timestamp semantics and fixed view still require field evidence.

The VITS2 test waits for `prepared` length before asserting downstream `readings`, leaving a worker scheduling race. An ordinary upstream run reproduced exactly `['', ''] != ['', '', '']`; a separately labelled controlled-scheduling probe also reproduces it without editing upstream source/tests. This is not presented as a deterministic new ROI failure. Music expects not_playable/not_found while its external provider returns unavailable; baseline repeats it. No unrelated modules were modified.

Final full confirmation (`merged-final38.log/xml`) passes both previously failing visual-depth tests and VITS2. Untouched baseline full run (`baseline-full38.log/xml`) also passes those tests. The initial visual-depth timeout interleavings were not reconstructed, so their precise cause remains unproven; they are disclosed as transient failures, not claimed to reproduce on upstream. The two visual-depth files are byte-identical to upstream and no related code was changed. Baseline's entire visual-depth module separately passed (92 passed with one unrelated music failure in that invocation).

Final focused counts are 74 per interpreter, covering the final publisher-type/QoS changes. Native Humble final output is `10 real-rclpy synthetic checks passed`. `lint.log`, shell syntax check and real Python 3.8 compilation cover the added standalone code/tool. Installer/package tests are recorded in `verification/INSTALL.json` and `verification/package-tests.log`; the source archive includes these tests, the native smoke tool and the unchanged official profile.

## Additional installed-kit DDS limit (not passed)

After successful offline installation/self-tests, an additional real-Humble cross-process publisher → installed HTTP server test returned SOURCE_TIMEOUT before baseline capture. Explicitly spinning the publisher and waiting for a matched subscription did not resolve delivery. A separate minimal plain-rclpy subscriber, importing no Card code, also discovered the publisher but received zero messages (`plain-dds-probe.log`). Installed Doctor correctly reported Domain 42, publisher type, QoS compatibility WARNING detail, zero received messages and TOPIC_NO_NEW_FRAMES (`installed-doctor-probe.json`), never camera-ready or PASS. The owned server was stopped each time.

This distinguishes discovery from delivery but does not identify the lower-level cause. It is not proven to be WSL networking, a specific DDS defect or a Card defect. The native ten-case smoke succeeds with distinct Contexts in one process; that must not be advertised as successful cross-process/D435 acceptance. Cross-process transport and real camera ingress remain unverified; no profile/domain changes, network modifications or threshold workarounds were made. All failed initial/spun/probe logs are retained. This extra test uses synthetic images and touches no laboratory equipment.
