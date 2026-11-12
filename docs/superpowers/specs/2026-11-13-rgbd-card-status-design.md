# RGB-D odometry compact card and status output

User request: remove visible instance/calibration/algorithm form fields, keep settings in configuration, add data/json status following Bumi visual odometry. Prior user authorization opts out of incremental approvals and permits PR371 push; hardware remains paused.

Use existing configSchema as the sole algorithm-setting UI. inputSchema advertises only system start/stop/info actions, like Bumi. Dispatch still accepts config replay and automatically injected instance_id/input_topics/camera_info. Keep existing state/odom first port/topic unchanged for saved connections; append instance-local /status data/json.

Status fields follow Bumi: status, turn_angle_deg, displacement_m, position_m, pose_reference, age_ms, reason. Position is the main camera pose, displacement is straight-line distance from the first valid pose in this estimator segment (not path length), never the ESKF prediction. turn_angle_deg remains null because this generic camera contract does not define body yaw. age_ms is null without a measurement timestamp. Same-state/reason summaries are capped at5Hz; transitions bypass the cap. Stop publishes idle with null pose before disposing the existing node; lost/initializing/coasting/recovering likewise have no held valid position.

Status transport is best effort: log publication failures without retiring valid odometry or retaining a disposed instance during stop/restart.

Approaches: keep duplicate form fields (contradicts request); change global canvas rendering (unnecessary impact); selected implementation changes only generic plugin schema and session status publication using existing ROS String/JSON dependencies. No driver edits, robot branches, native algorithm changes or new dependencies. Offline schema/config replay and real session publication tests; no live UI/hardware acceptance claim.
