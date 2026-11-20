# ROI review — updated 2026-11-14 for PR #393

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

Actual D435 identity, current topic/frame/domain, capture-time semantics, fixed camera pose and three real-camera outcomes remain to be collected. Actual Humble native libraries were privately extracted and exercised with synthetic loopback topics in WSL; no laboratory device/service was accessed. ARM64 wheel preparation is not ARM64 runtime acceptance.

Evidence-save or source failure returns an error/INCONCLUSIVE and must be investigated, not relabelled PASS. Retained historical tests and baseline failures are documented. No push, PR creation, remote merge, shared deployment or physical motion was performed.

## PR #393 fixes and review response

- Reproduced Doctor's exact `__enter__` and `_sigint_gc` failure with actual Humble 3.3.22 in a Python 3.10 venv. A private Context had been initialized, but `rclpy.spin_once(node)` chose the global Executor whose default Context was uninitialized. Doctor now owns an explicit same-context Executor and cleans partial initialization. Sources records Context ownership before Executor construction; shutdown attempts all owned cleanup and always retires Engine sessions. No rclpy internal API, Context handle manipulation or system ROS change is used. See the official [Humble implementation](https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/__init__.py) and retained native traceback.
- Doctor retains stage, error type and full initialization traceback, runtime package/interpreter paths, publisher types and QoS compatibility. Its own subscription cannot be mistaken for proof of a publisher's message type. Missing publishers, inactive streams, decoder failure, invalid capture stamps and frame mismatch have distinct codes. It cannot infer whether an absent remote publisher is caused by domain, DDS profile or a stopped camera; that limit is explicit.
- All ten official ROI CI failures reproduced on Python 3.8 and pass after replacing pairwise with adjacent zip iteration. A local Ruff target prevents fixes from reintroducing 3.10/3.11-only helpers. No thresholds or assertions were loosened.
- Timing review: in the checked HEAD, plugin dispatch already completes Sources reads before Engine.call. The review's claimed pre-wait Engine.call ordering was not reproduced in that source. The Engine now explicitly re-samples wall time at the image-validation boundary and rechecks session lifetime, making the required post-acquisition ordering local and reviewable. Delayed-arrival, real future, replay and acquisition-time expiry regressions preserve all guards. The ten observed Python 3.8 failures are attributed to pairwise, not speculatively to this timing finding.
- Official common DDS policy says Domain 42 everywhere. Config, Sources, Runner and Doctor reject 222. The kit ships the unchanged official loopback profile and applies it only to its own processes. Historical camera Domain 222 remains a real field ingress blocker; no exception, camera reconfiguration or bridge was implemented. Hardware testing needs an owner-approved compliant data ingress first.
- Standalone requirements/wheels remain necessary for the user-requested private field kit. No Perception Dockerfile or production dependency layer changed, and the isolated runtime dependencies are not injected into that image.

The unrelated VITS2 race was naturally reproduced on a clean latest-upstream clone. Music failure also reproduces there. Initial full-run visual-depth timeouts are retained and investigated with baseline and confirmation runs; see TESTING.md / DELIVERY.md. No unrelated code changes are included.

Software fixes are locally reviewable for the existing PR #393. Local tests cannot establish D435 identity/capture semantics, real frame publication, fixed view or hardware tri-state acceptance. Domain 222-to-42 access is an external field blocker, not a completed deployment. Existing PR is regular/open; this work does not change its remote status.

Additional installed-kit cross-process DDS check did not pass: the Card and an independent minimal subscriber both saw a synthetic publisher but received zero messages in WSL. Doctor accurately reports TOPIC_NO_NEW_FRAMES, with no false PASS. Underlying transport cause is not proven; this is an explicit integration limitation, separate from the successful native same-process Context tests. See TESTING.md. Do not present field capture as ready or accepted until this and compliant camera ingress are verified in the target environment.
