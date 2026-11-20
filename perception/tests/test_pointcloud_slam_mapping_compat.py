"""The mapping body is consumable without a SLAM-specific metadata prefix."""
import struct

import numpy as np

from plugins import pointcloud_slam_preview as preview


def test_packet_has_only_native_mapping_header_and_xyz():
    packet = preview.encode_mapping([[-1.25, 2.5, -3.75]], np.eye(4), max_points=1)
    assert struct.unpack('<fffBI', packet[:17]) == (0., 0., 0., 39, 1)
    assert len(packet) == 29
    assert np.frombuffer(packet[17:], dtype='<f4').tolist() == [-1.25, 2.5, -3.75]
