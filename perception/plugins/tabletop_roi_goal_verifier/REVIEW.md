# ROI review — 2026-11-13

Code Ready: YES. Draft PR Ready: YES. Regular PR Ready: NO in this delivery: real D435/source-time/fixed-view acceptance remains outstanding, and the known baseline music failure is disclosed rather than hidden. No known unresolved introduced P0/P1 software finding remains in the reviewed paths.

## Fixed findings

- Acquisition previously held the dispatch lock while stop waited behind it. Stop now signals a source interruption before lifecycle locking, wakes waiting readers and uses a generation check to prevent a late start/capture from committing after cancellation.
- A camera error or acquisition gap could preserve the old source epoch. Source failure removes the cached frame; reconnection/gaps rotate that source's epoch. Existing baselines then fail geometry comparison and require a new session.
- A failed JPEG could leave half a baseline; baseline metadata/image/evidence now commit together. A later failed after-capture removes its predecessor's after crop from the latest export.
- Wall rollback could leave a reusable old baseline after clock recovery. Affected sessions and receipts are retired; recovery requires a new session.
- Neutral local shadow/reflection could satisfy the difference threshold. Achromatic-only change now returns `ROI_PHOTOMETRIC_AMBIGUITY`; saturated empty images are refused. Existing change/stability thresholds were not loosened. Grayscale-only objects may conservatively be inconclusive.
- Incomplete PASS/FAIL evidence and a failed disk write could leave a seemingly complete run. Success evidence requires both decodable ROI images, all files/checksums are staged privately and only a complete run is atomically published.
- Real Core trace metadata was rejected; bounded string `_trace_id` is now accepted as diagnostic metadata and excluded from request identity. It grants no session authority.
- Bundle logs could disclose `session_token`. Request/response diagnostic serialization now redacts that field recursively; schema marks it sensitive. This small common logging change is the only public change beyond default-off registration.
- Stop verified a numeric PID before signalling, leaving a reuse window. It now signals a pinned Linux process handle, with a Python 3.10 binding fallback and no numeric-PID signal.

All fixes have corresponding regressions. Source preparation, mock ROS packets, direct transport and real browser interaction were reviewed. See TESTING.md for counts, earlier failures and limits of attribution.

## Isolation and compatibility

The tool, plugin/config/schema, sessions, source adapter, algorithm, Workbench and deployment are independent. There is no import of `plugins/goal_verifier`, VOP, YOLOE, a robot driver or CAN. Defaults do not register the Card; no shared config was edited. ROI and Object #374 trial-merge without conflict, and 147 Object tests pass in their combined disposable tree. The original Object worktree/branch and its prior dirty files remain unchanged. No object mode is added to or removed from this separate ROI-only Card.

Session tokens and per-session processing buffers isolate concurrent sessions; cancelling one does not mutate another. Source errors invalidate only that source. Card-level stop intentionally retires all its sessions. The standalone loopback service is for trusted local/SSH users, not a multi-tenant web security boundary. Callers must use unique request IDs; no public service/account system was added.

## Limits and remaining acceptance

Image differencing does not prove physical identity, robot motion or metric position. Low texture can hide camera motion; local colored lighting or a small stable obstruction can look like a state change. The added nuisance guard is conservative, not an exhaustive visual authenticity detector. Driver restamping/replayed pixels cannot be independently authenticated by image content. Fixed-view/capture-time records are operator evidence, not assertions manufactured by Doctor.

Actual D435 identity, current topic/frame/domain, capture-time semantics, fixed camera pose and three real-camera outcomes remain to be collected. The local environment has no ROS2 Humble, and no laboratory device/service was accessed. ARM64 wheel preparation is not ARM64 runtime acceptance.

Evidence-save or source failure returns an error/INCONCLUSIVE and must be investigated, not relabelled PASS. Retained historical tests and baseline failures are documented. No push, PR creation, remote merge, shared deployment or physical motion was performed.
