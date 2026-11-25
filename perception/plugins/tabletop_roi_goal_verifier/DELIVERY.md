# ROI PR #393 repair delivery — 2026-11-14

- Workspace: `/home/utopus/phanthy-work/phanthymotus-roi-goal-verifier`
- Branch: `feat/tabletop-roi-goal-verifier`; existing PR: https://github.com/4paradigm/phanthymotus/pull/393
- Protected initial HEAD: `50da4cdf94d038c72b7af000c59cdaa41864be19`, initially clean. Initial status/patch/refs are retained in the task artifact.
- Runtime repair commit: `deb1b1c42483978576046e14f164fc192e3154ad`; follow-up commit completes diagnostics, native smoke compatibility and documentation. Final exact HEAD, changed-file lists, checksums and Git audit: artifact `RELEASE.json` / `DELIVERY.md`.
- Artifact root: `/home/utopus/phanthy-work/artifacts/roi-pr393-fix-20261114/`.
- New field version: `20261114-r1`; default new deployment `/home/agilex/goal-verifier-roi-20261114-r1`, localhost port 18084 with collision fallback. Existing lab directories/packages are untouched.

## Results

Code fixes ready for local review; existing PR can be updated manually after review. No automatic remote update was made. 74 focused tests pass on Python 3.8 and system Python 3.10.12. Ten native Humble same-process synthetic transport/lifecycle/diagnostic checks, six Chromium scenarios, 37 Core checks and 147 Object compatibility tests pass.

Final latest-main merged regression: **2329 passed, 1 failed, 39 skipped**. Untouched latest-upstream baseline: **2255 passed, 1 failed, 39 skipped**, same external music provider error-code failure. The earlier branch full run's VITS2 failure naturally reproduces upstream; an initial merged run's two visual-depth timeouts passed baseline and final confirmation, with the precise earlier timing cause unproven. Full records remain disclosed in TESTING.md and artifact logs. ROI expectations/skips/thresholds were not weakened.

## Field blocker and remaining validation

Official `agents/pr_review/rules/common.md` requires Domain 42 everywhere. The old camera was observed in Domain 222; this is preserved as a true incompatibility. No direct 222 runtime or bridge is supplied. A maintainer-approved Domain 42/loopback camera data ingress is required before real D435 testing. The personal kit ships the unchanged official DDS XML, rejects 222 and never changes shared nodes.

The real Humble tests use native x86_64 packages privately extracted under WSL; no system installation or lab connection occurred. All generated frames/evidence are synthetic. Actual camera publication rate/JPEG/header timing, camera identity, fixed view, restart/reconnect and FAIL → PASS → INCONCLUSIVE plus saved evidence must still be verified on real D435. ARM64 wheels are included, not ARM64 runtime acceptance.

## Materials and safety

PR.md contains the English PR text/review response. README.md / API.md describe use; TESTING.md records all eleven official CI cases and current failures; REVIEW.md explains root causes/boundaries; deploy/field/ONSITE.md provides fresh-directory install, launch/stop, Doctor and Windows SSH steps. No Perception Dockerfile, common ROS config, VOP, driver, Object #374 or Bumi code was modified in this repair.

The new task artifact contains `field-kit/`, `tabletop-roi-d435-field.tar.gz`, `FIELD-SHA256SUMS`, source archive/hash, install validation, exact commit and diff manifests. Previous artifacts remain immutable. No push, PR write, review request, lab/shared service operation, device control or robot motion was performed.

Additional installed-kit cross-process DDS integration did not pass in WSL: graph matched but no samples arrived, reproduced by a standalone plain-rclpy subscriber. Root cause is unproven; no shared/network/profile changes were attempted. Native same-process checks and offline package self-tests remain passing; field capture is explicitly not accepted. Refer to TESTING.md and task `verification/installed-native-*.log`, `plain-dds-probe.log`, `installed-doctor-probe.json`.
