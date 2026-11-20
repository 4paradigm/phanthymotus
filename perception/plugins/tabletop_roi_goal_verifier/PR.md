# PR title

feat(perception): add independent tabletop ROI goal verifier

# PR body

## What / Why

Add `tabletop_roi_goal_verifier` to verify a selected tabletop region before and after an operation, without VOP, YOLOE, object classes or robot control. Stable local visual change returns PASS, comparable unchanged imagery returns FAIL, and unreliable evidence returns INCONCLUSIVE. PASS does not prove object identity, physical robot motion or metric position.

## Implementation

An independent default-off plugin owns its MCP schema, configuration, bounded sessions, ROS image adapter, CPU comparison, Workbench and loopback deployment. It accepts actual ROS header capture times by default; synthetic/replay buffers require explicit test provenance. Source failure/reconnect, clock rollback and cancellation invalidate affected evidence. Neutral photometric ambiguity, saturated images, occlusion, instability and geometry changes fail conservatively.

The browser supports preview, ROI dragging, presets, the four-step flow, statistics, before/after crops and atomic evidence saving. Installation uses a new personal directory with offline venv/target fallback. Stop verifies and signals only its pinned process handle. The only shared-file edits are default-off registration and session-capability log redaction; shared config and Core/Web are unchanged.

## Safety boundaries / Independence

No motion, driver or CAN imports exist. The Card does not depend on #374; it trial-merges cleanly with current main and #374, and 147 original Object tests pass in the combined tree. The separate Object Card and original worktree were not changed. The standalone service binds loopback and is intended for trusted local/SSH use.

## Test evidence

After PR #393 review fixes: all ten official ROI failures reproduced under Python 3.8, then passed with the original assertions. Focused suites: **74 passed on Python 3.8 and 74 on Python 3.10**. Actual Humble libraries: ten native loopback DDS checks with synthetic images. Chromium: six standalone/Workbench drag, four-step, three-state and evidence scenarios. Related Core: 37 passed. Updated full-regression, baseline attribution, Object compatibility and offline kit records are in TESTING.md / DELIVERY.md.

PR #393's VITS2 failure naturally reproduces on untouched latest upstream; music provider also fails independently. Initial visual-depth timing failures are retained with the baseline/confirmation results. No skips/assertions/audio modules were changed.

## Follow-up fixes for #393

Doctor now uses an explicit Executor on its initialized private Context (reproduced native Humble `__enter__` root cause), retains structured diagnostics/tracebacks, and cleans partial startup. Frame validation refreshes host wall time after acquisition and rechecks session lifetime. Pairwise no longer requires Python 3.10. Real DDS configurations require Domain 42 and the field kit uses the unchanged official loopback profile; it cannot directly consume the lab camera's historical Domain 222 stream. No shared-service reconfiguration or bridge is included. Standalone wheel requirements affect the personal field kit only, not the production image layers.

## Known limitations / Remaining hardware validation

The local visual criterion can confuse a stable obstruction or colored illumination with a state change. Grayscale-only changes may be inconclusive. Camera identity, fixed pose and true capture-time semantics require operator evidence; the algorithm/Doctor cannot manufacture that proof.

Additional installed-kit cross-process DDS in WSL did not deliver frames, including a plain-rclpy control subscriber; cause remains unproven and is disclosed in TESTING.md. Real D435 acceptance is outstanding; native Humble software integration used synthetic frames only. All automated frames are marked synthetic. The existing regular PR #393 remains unchanged remotely; hardware acceptance and a compliant Domain 42 camera ingress remain outstanding. See README.md, TESTING.md, REVIEW.md and deploy/field/ONSITE.md for commands, retained failures and risks.
