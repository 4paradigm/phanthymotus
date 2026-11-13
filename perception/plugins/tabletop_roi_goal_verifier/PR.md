# PR title

feat(perception): add independent tabletop ROI goal verifier

# PR body

Adds `tabletop_roi_goal_verifier` to compare a selected tabletop region before and after an operation. A stable local visual change returns PASS, valid comparable images without sufficient change return FAIL, and unreliable evidence returns INCONCLUSIVE. This Card requires no object detector, VOP service, depth input, or robot driver, and sends no motion commands.

The implementation has its own plugin, MCP schema, configuration, bounded sessions, ROS image adapter, Workbench and loopback HTTP deployment entry. It reuses the ROI algorithm developed alongside the Object Goal Verifier, with at least two fresh after frames required for stability and serialized idempotent requests. The only platform edit is a default-off registration block in `perception/main.py`; shared configuration and Core/Web are unchanged.

This branch starts from upstream `main` at `541d53461818f220a56f902e5c7093be61e155d8`. It does not include or import #374's `plugins/goal_verifier`, and can merge first. ROI state change is different from selecting an object's target position or testing single-frame presence. PASS establishes neither persistent identity nor physical robot motion/position.

Validation: 39 focused/field tests, 39 independent-install tests, six Chromium scenarios and 37 related Core tests passed. Offline venv/target installation and missing-venv fallback passed on Ubuntu Python 3.10. Full Perception: 2293 passed, 1 failed, 40 skipped (69 subtests passed). The failing unchanged music HTTP test also fails on the unmodified upstream snapshot; that snapshot has a second music failure. Details and known failures are recorded in `TESTING.md` and the delivery's `verification/` directory. Real D435 testing remains outstanding; synthetic tests must not be used as hardware acceptance evidence. Keep this PR Draft until the isolated field deployment has been checked with the real camera.


Field delivery includes loopback browser video/ROI controls, a read-only ROS/image/MCP Doctor, isolated offline installation without sudo/system pip/ensurepip, PID-checked lifecycle scripts and evidence saving under the new personal directory. Wheel binaries remain outside git in the delivery package; no laboratory device or shared service was accessed. See `deploy/field/ONSITE.md` for operator steps and the outstanding camera identity/fixed-view checks.
