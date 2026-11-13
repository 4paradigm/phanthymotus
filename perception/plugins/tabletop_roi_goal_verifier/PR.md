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

66 focused tests passed, 37 related Core tests passed, and six Chromium standalone/upstream-Workbench scenarios passed. Full Perception: **2320 passed, 1 failed, 40 skipped** (69 subtests passed). The remaining music provider error-code test reproduces on untouched upstream; the historically failing music HTTP test passes now. No music code or skips were changed. Rebuilt field-kit install/self-test and immutable archive hashes are recorded in delivery evidence.

## Known limitations / Remaining hardware validation

The local visual criterion can confuse a stable obstruction or colored illumination with a state change. Grayscale-only changes may be inconclusive. Camera identity, fixed pose and true capture-time semantics require operator evidence; the algorithm/Doctor cannot manufacture that proof.

Real D435/ROS2 Humble acceptance is outstanding. All automated frames are marked synthetic. Keep this as a Draft until the staged field steps and required review are complete. See README.md, TESTING.md, REVIEW.md and deploy/field/ONSITE.md for commands, retained failures and risks.
