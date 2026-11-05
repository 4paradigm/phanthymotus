# SLAM simple-start deployment plan

Goal: Bumi's operator connects two inputs and starts the card without filling
calibration/clock internals. A deployed robot profile supplies the audited
same-origin axis transform and bounded experimental timing settings. It does
not claim measured calibration or independently verified clock accuracy.

- Keep strict legacy behavior when no experimental deployment policy exists.
- For an explicit deployment policy, expose only preview and archive controls;
  stale canvas technical fields cannot override the robot profile.
- Start subscribes immediately and reports waiting until tracking data arrive.
  Missing/wrong input declarations still fail. Lost/frame/time/segment guards
  remain; no receipt-time substitution, no odometry overwrite or motion control.
- Preserve policy in info, preview status and archives; do not silently load an
  experimental archive under the strict policy.
- Test late first inputs, lost protection, UI fields, archive policy mismatch;
  run target SLAM regression and isolated ROS lifecycle checks.
- Build a local runtime overlay over the validated dependency image, deploy only
  Perception, then verify real subscriptions/keyframes/preview. Commit and push.

This is deployment-profile configuration, not completed generic TF discovery.
No additional operator approval is needed: the user explicitly requested direct
trial with relaxed timing, simplified controls and deployment.
