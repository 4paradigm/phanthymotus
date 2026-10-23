"""ROS-independent FAST-LIVO2 process lifecycle helpers."""

from __future__ import annotations

import signal


_CONTROLLED_STOP_RETURN_CODES = frozenset(
    {
        0,
        -int(signal.SIGINT),
        -int(signal.SIGABRT),
    }
)


def controlled_stop_succeeded(return_code: int) -> bool:
    """Accept known FAST-LIVO2 teardown codes only after an owned stop signal."""

    return int(return_code) in _CONTROLLED_STOP_RETURN_CODES


__all__ = ["controlled_stop_succeeded"]
