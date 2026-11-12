# RGB-D offline stability (2026-11-13)

The user requests offline-verifiable fixes only: code, offline tests, docs and
local commits. No SSH, robot access, deployment, restarts or motion. Generic
changes belong to PR371; PR519 changes only if a Tianyi source defect is proven.

## Evidence

- Start failure "connect one calibrated color and one measured depth input":
  `validate_start_inputs` received two topics but fewer than two image
  declarations keyed by them. The field cause is unknown. Offline, the real
  start-project chain reproduces it when a source answers without
  `camera_info` (Tianyi camera card `unavailable` before first calibrated
  capture; Core treats that as started) or when a source `info()` fails and
  the link falls back to a snapshot topic. Duplicate links collapse to Core's
  singular `input_topic`, which produced an unrelated type message.
- ESKF private-source probe: sessions reached `lost` with sigma 0.102-0.127 m
  and `last_measurement - last_visual` of 400-534 ms while image quality was
  usable and native latency was 135-240 ms. Static review: `observe_frame`
  judged predicted sigma before the frame's visual correction; wall-clock
  `info`/watchdog polls demoted tracking to coasting and reset the evidence
  counter whenever latency plus frame gap exceeded 250 ms. Not established:
  that all failures share one cause, or that ICP was involved (no ICP attempt
  appears in the probe's snapshots).
- Pure RGB-D business session: one covariance rejection latched `lost` and
  stopped feeding the native estimator.
- `DepthRegistration.register`: ~47 ms median on an arm64 development host,
  executed under the session lock; IMU/fusion sessions run every callback on a
  single-threaded executor.

## Design

A. Keep validation strict; report per-input declaration/format evidence with
missing roles first and bounded topic text; accept Core's singular input.

B. Per frame: hard errors -> IMU prediction -> legal visual correction ->
uncertainty judgement (predicted fatal only without correction; corrected
pauses output, fatal after 1 s). Capture-time bounds stay in frame
processing; wall-clock polls report frame silence after 500 ms only. When the
ESKF stops, the independent native visual estimate continues as labelled
`visual_only` if continuity is proven (fused/visual offset <= 2 cm / 2 deg at
the last published fused capture, or nothing fused was published); otherwise
retire the segment. Q/P, the 0.10 m limit and the 1 s inertial bound are
unchanged. Bounded per-frame diagnostics (16 frames).

C. Rejected estimates pause output (`recovering`) while the same estimator
keeps receiving frames; resume after three consecutive valid estimates
continuous with the last published pose; retire after 1.5 s, discontinuity,
frame change or future capture time. Late/repeated captures are dropped.

D. One worker thread per fused session runs ICP off the lock; one running, one
pending, one result. Results apply only to the exact filter revision before
the next frame; otherwise discarded (deadline 250 ms, failure, stop, restart).

E. `visual_frontend=auto` (opt-in): start F2M, one frontend at a time; switch
only on visual-kind loss with fresh inputs, via existing supervised recovery
into a new segment; 20 s cooldown; suspend after two untracked switches.

## Not in scope

Depth-only odometry, ESKF re-entry within a fallback segment, Core readiness
semantics for `unavailable` source cards, noise calibration, physical
accuracy, body TF compensation, map save/reopen.
