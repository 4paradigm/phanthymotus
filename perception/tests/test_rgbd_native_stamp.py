"""Mirror pinned RTAB-Map's pair stamp choice without rounding capture stamps."""
from types import SimpleNamespace as NS

import pytest

from plugins import rgbd_odometry_runtime as runtime


@pytest.mark.parametrize('color,depth,expected', [
    (1_000_800_123, 1_000_000_123, 1_000_800_123),
    (999_200_123, 1_000_000_123, 1_000_000_123),
    (1_000_000_123, 1_000_000_123, 1_000_000_123),
    # rclcpp::Time::seconds() comparison ties here; native chooses depth,
    # retaining its exact integer header even though color is 1 ns later.
    (1_794_654_700_000_000_002, 1_794_654_700_000_000_001, 1_794_654_700_000_000_001),
])
def test_native_pair_stamp_preserves_the_selected_source_header(color, depth, expected):
    def msg(stamp):
        sec, nanosec = divmod(stamp, 10**9)
        return NS(header=NS(stamp=NS(sec=sec, nanosec=nanosec)))
    c, d = msg(color), msg(depth)
    assert runtime.native_capture_stamp(c, d) == expected
    assert runtime.stamp_ns(c) == color and runtime.stamp_ns(d) == depth
