# ROI verification — 2026-11-13 closure

Current source is independent of Object Card #374 and starts from upstream `541d53461818f220a56f902e5c7093be61e155d8`. Counts below are newly executed, not the earlier 39/2293 report.

| Check | Actual result |
|---|---|
| Python 3.10 focused unit, HTTP, lifecycle and field tests | 66 passed |
| Complete Perception regression | 2320 passed, 1 failed, 40 skipped; 69 subtests passed |
| Related Core registry/MCP/skill tests | 37 passed |
| Chromium standalone and unchanged upstream Workbench | Six three-state/drag/four-step/evidence scenarios passed |
| Object Card tests in a disposable real #374 + ROI merge | 147 passed |
| Latest main and #374 trial merges | Both clean, no conflicts |
| Real D435 / ROS2 Humble / physical motion | Not executed |

The final field archive is rebuilt from this source. Its installation/self-test, size and SHA256 are recorded in the task delivery's RELEASE/DELIVERY records, not copied from an older archive. Offline venv/target/missing-venv fallback and existing-directory/checksum refusal are checked against the rebuilt kit.

## Commands

```bash
ROI_TEST_PYTHON=python3.10 bash perception/plugins/tabletop_roi_goal_verifier/deploy/run-tests.sh --tb=short
PYTHONPATH=perception PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.10 -m pytest -q perception/tests --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q agent-core/tests/test_mcp_registry_merge.py agent-core/tests/test_mcp_call_errors.py agent-core/tests/test_skill_execution.py
python3 perception/tools/tabletop_roi_browser.py --chromium /path/to/chrome --output /path/to/new-browser-results
```

Focused tests need pytest plus runtime requirements; HTTP fixtures bind loopback only. Browser tests additionally need Playwright/Chromium. All test pixels and clocks are explicitly synthetic. No test opens a real camera or publishes robot control.

## Failure attribution

Current full-suite failure: `test_music.py::PluginTests::test_playing_a_specific_track_still_works_for_stored_callers` expects `not_playable|not_found`, but the existing provider returns `unavailable`. The untouched current upstream reproduces it. The specifically named historical `test_phanthy_music.py::MCPTests::test_real_http_tools_list_dispatch_and_log_redaction` passes this task's baseline and final run; the old HTTP timeout/502 is not presented as a current failure. No music code, expectations or skip rules were changed.

New review regressions first reproduced stop waiting for acquisition, stale baseline reuse after source failure, partial baseline mutation on JPEG failure, reuse of an earlier after crop, shadow/reflection PASS, incomplete evidence and partial saved directories, Core `_trace_id` rejection and capability logging. Each was fixed and rerun. Failed red logs are retained.

One initial full run failed the new two-session isolation test with an INCONCLUSIVE result. Its original assertion did not retain a reason code, so the exact incident cannot be reconstructed. A separate read-only probe recorded wall/monotonic disagreements of 1.27–2.34 s. The isolation test now uses a declared synthetic monotonic-derived clock to test isolation independently; dedicated rollback/stale/future tests continue to verify product refusal and session retirement. This is not proof of the first incident's precise cause. Confirmation and final full runs retain only the baseline music failure. Product clocks and refusal thresholds were not relaxed.

An early pidfd test exposed that the portable Python 3.10 build lacks some standard bindings. The Linux x86_64/aarch64 syscall fallback is tested with real owned child processes; the syscall numbers were verified against local architecture/generic kernel headers. Numeric-PID signalling is not used. Earlier sandbox socket errors and interrupted runs remain in the evidence directory and are not counted as passing.

## Reviewed coverage

PASS/FAIL/INCONCLUSIVE; brightening, neutral local shadow/reflection, one-pixel jitter and severe occlusion; invalid/tiny/outside ROI; empty/corrupt/format-changed ROS packets; stale, future, replayed and out-of-order timestamps; resolution/source epoch changes; disconnect/reconnect; rollback retirement; separate sessions/token rejection/close; stop during acquisition; retry conflicts; encoding and storage errors; missing evidence; atomic evidence publication; Core trace metadata and log redaction; port collision and pinned-process stopping.

The tests establish software behavior, not camera identity, fixed pose, physical target identity or stopping time. A small stable local obstruction or colored lighting can still be indistinguishable from a visual state change; PASS retains only its documented pixel-state meaning.

Evidence root: `/home/utopus/phanthy-work/artifacts/dual-card-pr-ready-20261113/roi/`. Primary records: `verification/targeted-delivery.*`, `verification/regression-delivery.*`, `verification/music-upstream.log`, `verification/core-regression.log`, `verification/object-compatibility.log`, `browser-verified/`. Host clock probe: task root `CLOCK.json`.
