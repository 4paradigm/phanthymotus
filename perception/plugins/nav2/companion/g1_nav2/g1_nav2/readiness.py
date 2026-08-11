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
    odom_status: dict,
    odom_status_received_at: float | None,
    scan_received_at: float | None,
    scan_source_age_sec: float | None,
    sensor_stamp_skew_sec: float | None,
    max_sensor_stamp_skew_sec: float,
    lifecycle_states: dict[str, int],
    action_server_ready: bool,
    map_ready: bool,
    map_to_base_ready: bool,
) -> dict:
    """Return fail-closed runtime and navigation readiness receipts."""

    odom_receive_age = _age(now_monotonic, odom_status_received_at)
    scan_receive_age = _age(now_monotonic, scan_received_at)
    runtime_blockers: list[str] = []

    if odom_status.get("state") != "ready":
        runtime_blockers.append("odom_not_ready")
    if odom_receive_age is None or odom_receive_age > max_age_sec:
        runtime_blockers.append("odom_status_stale")
    source_age = odom_status.get("source_age_sec")
    timestamp_source = odom_status.get("timestamp_source")
    source_stamp_fresh = timestamp_source == "adapter_receive" or (
        timestamp_source == "driver"
        and isinstance(source_age, (int, float))
        and -0.1 <= source_age <= max_age_sec
    )
    if not source_stamp_fresh:
        runtime_blockers.append("odom_source_stamp_stale")
    if scan_receive_age is None or scan_receive_age > max_age_sec:
        runtime_blockers.append("scan_stale")
    if (
        scan_source_age_sec is None
        or not -0.1 <= scan_source_age_sec <= max_age_sec
    ):
        runtime_blockers.append("scan_source_stamp_stale")
    # Legacy adapters timestamp the streams independently on receipt, so their
    # relative phase is not a Driver synchronization guarantee. Enforce the
    # cross-stream source-time contract only after odometry has upgraded to the
    # normalized Driver clock; individual freshness checks still protect legacy.
    if timestamp_source == "driver" and (
        not isinstance(sensor_stamp_skew_sec, (int, float))
        or sensor_stamp_skew_sec < 0.0
        or sensor_stamp_skew_sec > max_sensor_stamp_skew_sec
    ):
        runtime_blockers.append("sensor_stamp_skew")
    inactive = sorted(
        name for name, state_id in lifecycle_states.items() if state_id != 3
    )
    if inactive:
        runtime_blockers.append("lifecycle_not_active:" + ",".join(inactive))
    if not action_server_ready:
        runtime_blockers.append("navigate_to_pose_unavailable")

    navigation_blockers = list(runtime_blockers)
    if not map_ready:
        navigation_blockers.append("map_not_ready")
    if not map_to_base_ready:
        navigation_blockers.append("map_to_base_unavailable")

    return {
        "n3_ready": not runtime_blockers,
        "navigation_ready": not navigation_blockers,
        "readiness_blockers": runtime_blockers,
        "navigation_blockers": navigation_blockers,
        "odom_status_age_sec": odom_receive_age,
        "scan_receive_age_sec": scan_receive_age,
        "scan_source_age_sec": scan_source_age_sec,
        "sensor_stamp_skew_sec": sensor_stamp_skew_sec,
        "max_sensor_stamp_skew_sec": max_sensor_stamp_skew_sec,
        "lifecycle_states": dict(lifecycle_states),
        "action_server_ready": action_server_ready,
        "map_ready": map_ready,
        "map_to_base_ready": map_to_base_ready,
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
