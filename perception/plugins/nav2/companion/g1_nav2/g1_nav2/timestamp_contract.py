"""Shared source-timestamp validation for Nav2 sensor adapters."""

from __future__ import annotations


DEFAULT_MAX_SOURCE_AGE_NS = 500_000_000
DEFAULT_MAX_FUTURE_SKEW_NS = 100_000_000


class InvalidSourceTimestamp(ValueError):
    """Raised when a Driver timestamp is outside the ROS system clock domain."""


def validate_source_timestamp_ns(
    source_stamp_ns: int,
    receive_stamp_ns: int,
    *,
    max_source_age_ns: int = DEFAULT_MAX_SOURCE_AGE_NS,
    max_future_skew_ns: int = DEFAULT_MAX_FUTURE_SKEW_NS,
) -> int:
    """Validate a Unix-epoch source stamp against adapter receive time.

    Both values must use the ROS system/Unix clock domain. The returned value is
    ``receive_stamp_ns - source_stamp_ns``; negative values mean the source is
    slightly ahead of the receiver clock.
    """

    for name, value in (
        ("source_stamp_ns", source_stamp_ns),
        ("receive_stamp_ns", receive_stamp_ns),
        ("max_source_age_ns", max_source_age_ns),
        ("max_future_skew_ns", max_future_skew_ns),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidSourceTimestamp(f"{name} must be an integer")
    if source_stamp_ns <= 0:
        raise InvalidSourceTimestamp("source_stamp_ns must be positive")
    if receive_stamp_ns <= 0:
        raise InvalidSourceTimestamp("receive_stamp_ns must be positive")
    if max_source_age_ns <= 0:
        raise InvalidSourceTimestamp("max_source_age_ns must be positive")
    if max_future_skew_ns < 0:
        raise InvalidSourceTimestamp("max_future_skew_ns must be non-negative")

    age_ns = receive_stamp_ns - source_stamp_ns
    if age_ns > max_source_age_ns:
        raise InvalidSourceTimestamp(
            f"source timestamp is stale by {age_ns / 1_000_000:.3f} ms"
        )
    if age_ns < -max_future_skew_ns:
        raise InvalidSourceTimestamp(
            "source timestamp is ahead of the receiver by "
            f"{-age_ns / 1_000_000:.3f} ms"
        )
    return age_ns


__all__ = [
    "DEFAULT_MAX_FUTURE_SKEW_NS",
    "DEFAULT_MAX_SOURCE_AGE_NS",
    "InvalidSourceTimestamp",
    "validate_source_timestamp_ns",
]
