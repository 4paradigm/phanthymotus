# RGB-D automatic segment recovery

The user requests automatic recovery after lost, permitting a new segment, and
has waived incremental approval. Robot access currently authorizes diagnosis,
not deployment, service restarts or motion.

Live diagnosis: the deployed files match dc11e84. The active F2M session produced
171 estimates before rejecting visual covariance and permanently stopping image
admission. IMU forwarding independently contains >100 ms capture gaps, although
the raw gyro/accel streams have approximately 10 ms capture intervals. Preserve
all visual/IMU freshness and quality gates; do not hide these failures.

Approaches: accepting poses again in the retired segment risks coordinate resets;
reinitializing the whole runtime in its own callback risks self-join and lifecycle
races. Use an external plugin-owned timer, like Bumi's recovery supervision.

After 0.8 s of lost and with both image inputs fresh, attempt a replacement
session, at most once per 1.5 s. Retain the original topics, calibration and
configuration. Each replacement uses a new native process, segment and origin;
tracking still requires the existing admission/warmup checks. Start the candidate
before closing the failed session so a constructor failure can be retried without
retaining a disposed instance. Suppress the old session's final idle during an
automatic replacement. Destroy the candidate if old-session cleanup fails.
Stop and shutdown serialize against supervision and cancel future recovery.
Expose attempt/restart counts, previous segment and last recovery error via info.
Candidates cannot publish public status or odometry before replacement commits.
Constructor failure suppresses idle. A failed candidate cleanup remains owned
until explicit stop/shutdown succeeds; no further replacement starts meanwhile.
Only successfully started worker threads are joined during constructor cleanup.
No old position or prediction is promoted to valid tracking or joined across a
segment boundary. Missing dynamic body TF remains a separate unavailable state.

Tianyi IMU work stays in PR519: a 200 ms queued-callback reproduction shows that
the five-sample subscription creates a 160 ms gap. Buffer 100 raw samples per
sensor, preserving original times and the existing pairing. Do not relax the
100 ms interval or 500 ms freshness limits. G1/Bumi
remain unchanged. Offline checks and PR pushes are authorized; live fixes await
explicit deployment authorization.
