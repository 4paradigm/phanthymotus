from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "navigation"
    / "runtime"
    / "g1_fast_livo2"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from g1_fast_livo2.frame_adapter_core import (  # noqa: E402
    InvalidFastLivo2Frame,
    Pose3,
    Quaternion,
    encode_map_view_points,
    quaternion_from_rpy,
    transform_points,
)
from g1_fast_livo2.vectorized_cloud import (  # noqa: E402
    decode_xyz_array,
    map_view_with_pose,
    transform_xyz_array,
    xyz_array_bytes,
)


def _field(name: str, offset: int, datatype: int = 7):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


class FastLivo2VectorizedCloudTest(unittest.TestCase):
    def test_decode_supports_padded_rows_and_drops_nonfinite_points(self) -> None:
        row_one = struct.pack("<fff", 1.0, 2.0, 3.0) + b"PAD!"
        row_two = struct.pack("<fff", math.nan, 5.0, 6.0) + b"PAD!"
        decoded = decode_xyz_array(
            fields=[_field("x", 0), _field("y", 4), _field("z", 8)],
            data=row_one + row_two,
            point_step=12,
            row_step=16,
            width=1,
            height=2,
            is_bigendian=False,
            max_points=10,
            max_data_bytes=1024,
        )

        self.assertEqual(decoded.dtype, np.float32)
        self.assertTrue(decoded.flags.c_contiguous)
        np.testing.assert_array_equal(
            decoded,
            np.asarray(((1.0, 2.0, 3.0),), dtype=np.float32),
        )

    def test_decode_supports_float64_and_rejects_float32_overflow(self) -> None:
        decoded = decode_xyz_array(
            fields=[
                _field("x", 0, 8),
                _field("y", 8, 8),
                _field("z", 16, 8),
            ],
            data=struct.pack("<ddd", 1.25, -2.5, 3.75),
            point_step=24,
            row_step=24,
            width=1,
            height=1,
            is_bigendian=False,
        )
        np.testing.assert_allclose(decoded, ((1.25, -2.5, 3.75),))

        with self.assertRaisesRegex(
            InvalidFastLivo2Frame,
            "float32 coordinate range",
        ):
            decode_xyz_array(
                fields=[
                    _field("x", 0, 8),
                    _field("y", 8, 8),
                    _field("z", 16, 8),
                ],
                data=struct.pack("<ddd", 1e308, 0.0, 0.0),
                point_step=24,
                row_step=24,
                width=1,
                height=1,
                is_bigendian=False,
            )

    def test_vectorized_transform_matches_validated_scalar_transform(self) -> None:
        pose = Pose3(
            0.75,
            -1.25,
            0.50,
            quaternion_from_rpy(0.10, -0.20, 0.35),
        )
        points = np.asarray(
            ((1.0, 2.0, 3.0), (-4.0, 5.5, -6.0)),
            dtype=np.float32,
        )

        expected = np.asarray(tuple(transform_points(pose, points)))
        actual = transform_xyz_array(pose, points)

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(xyz_array_bytes(actual), actual.astype("<f4").tobytes())
        self.assertEqual(xyz_array_bytes(()), b"")

    def test_map_view_pose_patch_preserves_cached_point_body_and_metadata(self) -> None:
        original_pose = Pose3(
            0.0,
            0.0,
            0.0,
            Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        cached = encode_map_view_points(
            ((1.0, 2.0, 0.1), (3.0, 4.0, 0.2)),
            original_pose,
            obstacle_min_height_m=-0.30,
            obstacle_max_height_m=0.30,
            max_points=80_000,
        )
        updated_pose = Pose3(
            5.0,
            -6.0,
            0.0,
            quaternion_from_rpy(0.0, 0.0, 1.25),
        )

        updated = map_view_with_pose(cached, updated_pose)

        self.assertEqual(updated[12:], cached[12:])
        x, y, yaw = struct.unpack_from("<fff", updated, 0)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, -6.0)
        self.assertAlmostEqual(yaw, 1.25)

    def test_invalid_layouts_remain_fail_closed(self) -> None:
        with self.assertRaises(InvalidFastLivo2Frame):
            decode_xyz_array(
                fields=[_field("x", 0), _field("y", 4), _field("z", 8)],
                data=struct.pack("<fff", 1.0, 2.0, 3.0),
                point_step=12,
                row_step=11,
                width=1,
                height=1,
                is_bigendian=False,
            )
        with self.assertRaises(InvalidFastLivo2Frame):
            map_view_with_pose(
                b"short",
                Pose3(
                    0.0,
                    0.0,
                    0.0,
                    Quaternion(0.0, 0.0, 0.0, 1.0),
                ),
            )


if __name__ == "__main__":
    unittest.main()
