"""ROS-independent readiness evaluation for the G1 Nav2 companion."""

from __future__ import annotations


def _age(now: float, received_at: float | None) -> float | None:
    if received_at is None:
        return None
    return max(0.0, now - received_at)


def evaluate_readiness(
    *,
    now_monotonic: float,
    max_age_sec: float,
    source_age_tolerance_sec: float = 0.0,
    odom_received_at: float | None,
    odom_source_age_sec: float | None,
    odom_frame_ready: bool,
    obstacle_received_at: float | None,
    obstacle_source_age_sec: float | None,
    obstacle_frame_ready: bool,
    source_stamp_skew_sec: float | None,
    lifecycle_states: dict[str, int],
    action_server_ready: bool,
    global_to_base_ready: bool,
) -> dict:
    """Evaluate the planner/controller boundary against FAST-LIVO2 outputs."""

    odom_receive_age = _age(now_monotonic, odom_received_at)
    obstacle_receive_age = _age(now_monotonic, obstacle_received_at)
    runtime_blockers: list[str] = []

    if odom_receive_age is None or odom_receive_age > max_age_sec:
        runtime_blockers.append("fast_livo2_odom_stale")
    if (
        odom_source_age_sec is None
        or not -0.1
        <= odom_source_age_sec
        <= max_age_sec + source_age_tolerance_sec
    ):
        runtime_blockers.append("odom_source_stamp_stale")
    if not odom_frame_ready:
        runtime_blockers.append("fast_livo2_odom_frame_invalid")
    if obstacle_receive_age is None or obstacle_receive_age > max_age_sec:
        runtime_blockers.append("registered_cloud_stale")
    if (
        obstacle_source_age_sec is None
        or not -0.1
        <= obstacle_source_age_sec
        <= max_age_sec + source_age_tolerance_sec
    ):
        runtime_blockers.append("registered_cloud_source_stamp_stale")
    if not obstacle_frame_ready:
        runtime_blockers.append("registered_cloud_frame_invalid")
    if not isinstance(source_stamp_skew_sec, (int, float)):
        runtime_blockers.append("fast_livo2_stamp_pair_unavailable")
    elif not 0.0 <= source_stamp_skew_sec <= max_age_sec:
        runtime_blockers.append("fast_livo2_source_stamp_skew")
    inactive = sorted(
        name for name, state_id in lifecycle_states.items() if state_id != 3
    )
    if inactive:
        runtime_blockers.append("lifecycle_not_active:" + ",".join(inactive))
    if not action_server_ready:
        runtime_blockers.append("navigate_to_pose_unavailable")
    if not global_to_base_ready:
        runtime_blockers.append("map_to_base_unavailable")

    return {
        "n3_ready": not runtime_blockers,
        "navigation_ready": not runtime_blockers,
        "readiness_blockers": runtime_blockers,
        "navigation_blockers": list(runtime_blockers),
        "odom_status_age_sec": odom_receive_age,
        "odom_source_age_sec": odom_source_age_sec,
        "odom_frame_ready": odom_frame_ready,
        "registered_cloud_receive_age_sec": obstacle_receive_age,
        "registered_cloud_source_age_sec": obstacle_source_age_sec,
        "registered_cloud_frame_ready": obstacle_frame_ready,
        "fast_livo2_source_stamp_skew_sec": source_stamp_skew_sec,
        "lifecycle_states": dict(lifecycle_states),
        "action_server_ready": action_server_ready,
        "global_to_base_ready": global_to_base_ready,
    }


def navigation_motion_blocker(readiness: dict) -> str | None:
    """Return the fail-closed reason that must suppress a non-zero proposal."""

    if readiness.get("navigation_ready") is True:
        return None
    blockers = readiness.get("navigation_blockers")
    if not isinstance(blockers, list) or not blockers:
        blockers = ["navigation_not_ready"]
    return "navigation_not_ready:" + ",".join(str(item) for item in blockers)


__all__ = ["evaluate_readiness", "navigation_motion_blocker"]
