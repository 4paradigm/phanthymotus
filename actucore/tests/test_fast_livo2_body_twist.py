"""Offline tests for the canonical odom body-twist reliability contract."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "controlled_semantic_spatial"
    / "runtime"
    / "fast_livo2"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from fast_livo2.frame_adapter_core import (  # noqa: E402
    TWIST_UNAVAILABLE_VARIANCE,
    BodyTwistEstimator,
    Pose3,
    Quaternion,
    compose_pose,
    quaternion_from_rpy,
)


IDENTITY = Quaternion(0.0, 0.0, 0.0, 1.0)
SEC = 1_000_000_000
T0 = 1_700_000_000 * SEC


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _se3_exp(linear, angular, dt) -> Pose3:
    """Independent closed-form exp of a constant body twist over dt."""

    rho = tuple(value * dt for value in linear)
    phi = tuple(value * dt for value in angular)
    theta = math.sqrt(sum(value * value for value in phi))
    if theta < 1e-12:
        return Pose3(*rho, IDENTITY)
    half = 0.5 * theta
    axis = tuple(value / theta for value in phi)
    q = Quaternion(*(math.sin(half) * value for value in axis), math.cos(half))
    a = (1.0 - math.cos(theta)) / (theta * theta)
    b = (theta - math.sin(theta)) / (theta ** 3)
    c1 = _cross(phi, rho)
    c2 = _cross(phi, c1)
    t = tuple(rho[i] + a * c1[i] + b * c2[i] for i in range(3))
    return Pose3(*t, q)


def _trajectory(start: Pose3, linear, angular, dt, count):
    pose = start
    step = _se3_exp(linear, angular, dt)
    poses = [pose]
    for _ in range(count - 1):
        pose = compose_pose(pose, step)
        poses.append(pose)
    return poses


def _stamp(index: int, dt: float) -> int:
    return T0 + int(round(index * dt * SEC))


class BodyTwistEstimatorTest(unittest.TestCase):
    def assert_unavailable(self, estimator, twist, reason) -> None:
        self.assertIsNone(twist)
        self.assertEqual(estimator.reason, reason)
        values, covariance = estimator.odometry_twist_fields(twist)
        self.assertEqual(len(values), 6)
        self.assertTrue(all(math.isnan(value) for value in values))
        self.assertEqual(len(covariance), 36)
        for index in range(6):
            self.assertEqual(covariance[index * 7], TWIST_UNAVAILABLE_VARIANCE)

    def assert_twist(self, twist, linear, angular, places=6) -> None:
        self.assertIsNotNone(twist)
        for got, want in zip(twist.linear + twist.angular, tuple(linear) + tuple(angular)):
            self.assertAlmostEqual(got, want, places=places)

    def test_recovers_constant_full_3d_body_twist(self) -> None:
        linear = (0.6, 0.15, 0.03)
        angular = (0.05, -0.04, 0.9)
        start = Pose3(3.0, -2.0, 0.4, quaternion_from_rpy(0.05, -0.02, 1.2))
        poses = _trajectory(start, linear, angular, 0.1, 6)
        estimator = BodyTwistEstimator()

        self.assert_unavailable(
            estimator, estimator.observe(_stamp(0, 0.1), poses[0]), "first_sample"
        )
        for index in range(1, len(poses)):
            twist = estimator.observe(_stamp(index, 0.1), poses[index])
            self.assert_twist(twist, linear, angular)
            self.assertAlmostEqual(twist.interval_sec, 0.1, places=9)
            self.assertIsNone(estimator.reason)

    def test_body_frame_forward_motion_with_world_heading(self) -> None:
        heading = quaternion_from_rpy(0.0, 0.0, math.pi / 2.0)
        estimator = BodyTwistEstimator()
        estimator.observe(T0, Pose3(1.0, 1.0, 0.0, heading))
        # Facing +y in the parent frame: moving +y is body forward.
        twist = estimator.observe(T0 + SEC // 10, Pose3(1.0, 1.05, 0.0, heading))
        self.assert_twist(twist, (0.5, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_rotation_in_place_reports_yaw_rate_without_linear(self) -> None:
        estimator = BodyTwistEstimator()
        estimator.observe(T0, Pose3(0.5, 0.5, 0.0, quaternion_from_rpy(0, 0, 3.0)))
        twist = estimator.observe(
            T0 + SEC // 10,
            Pose3(0.5, 0.5, 0.0, quaternion_from_rpy(0, 0, 3.0 + 0.08)),
        )
        # Crosses +/-pi; the quaternion log must not wrap to a huge negative rate.
        self.assert_twist(twist, (0.0, 0.0, 0.0), (0.0, 0.0, 0.8))
        twist = estimator.observe(
            T0 + 2 * SEC // 10,
            Pose3(0.5, 0.5, 0.0, quaternion_from_rpy(0, 0, 3.0 + 0.16)),
        )
        self.assert_twist(twist, (0.0, 0.0, 0.0), (0.0, 0.0, 0.8))

    def test_twist_is_invariant_to_fixed_left_transform(self) -> None:
        linear, angular = (0.4, -0.1, 0.0), (0.0, 0.0, -0.6)
        poses = _trajectory(Pose3(0.0, 0.0, 0.0, IDENTITY), linear, angular, 0.1, 3)
        map_from_session = Pose3(10.0, -4.0, 0.2, quaternion_from_rpy(0, 0, -2.0))
        session = BodyTwistEstimator()
        mapped = BodyTwistEstimator()
        for index, pose in enumerate(poses):
            left = session.observe(_stamp(index, 0.1), pose)
            right = mapped.observe(
                _stamp(index, 0.1), compose_pose(map_from_session, pose)
            )
        for got, want in zip(left.linear + left.angular, right.linear + right.angular):
            self.assertAlmostEqual(got, want, places=6)

    def test_stationary_robot_is_a_reliable_finite_zero(self) -> None:
        estimator = BodyTwistEstimator()
        pose = Pose3(1.0, 2.0, 0.0, quaternion_from_rpy(0, 0, 0.3))
        estimator.observe(T0, pose)
        twist = estimator.observe(T0 + SEC // 10, pose)
        self.assert_twist(twist, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        values, covariance = estimator.odometry_twist_fields(twist)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(all(covariance[index * 7] > 0 for index in range(6)))
        self.assertTrue(all(math.isfinite(value) for value in covariance))

    def test_duplicate_and_out_of_order_stamps_fail_closed_without_corrupting_history(self) -> None:
        estimator = BodyTwistEstimator()
        estimator.observe(T0, Pose3(0.0, 0.0, 0.0, IDENTITY))
        self.assertIsNotNone(estimator.observe(T0 + SEC // 10, Pose3(0.05, 0, 0, IDENTITY)))

        self.assert_unavailable(
            estimator,
            estimator.observe(T0 + SEC // 10, Pose3(9.0, 0, 0, IDENTITY)),
            "duplicate_stamp",
        )
        self.assert_unavailable(
            estimator,
            estimator.observe(T0 + SEC // 20, Pose3(-9.0, 0, 0, IDENTITY)),
            "out_of_order",
        )
        twist = estimator.observe(T0 + 2 * SEC // 10, Pose3(0.10, 0, 0, IDENTITY))
        self.assert_twist(twist, (0.5, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_source_gap_is_unavailable_then_restarts_from_new_reference(self) -> None:
        estimator = BodyTwistEstimator(max_interval_sec=0.35)
        estimator.observe(T0, Pose3(0.0, 0.0, 0.0, IDENTITY))
        # A paused canonical odom followed by a new sample: no differencing
        # across the gap, even though the pose did not move.
        self.assert_unavailable(
            estimator,
            estimator.observe(T0 + SEC, Pose3(0.0, 0.0, 0.0, IDENTITY)),
            "interval_too_large",
        )
        twist = estimator.observe(T0 + SEC + SEC // 10, Pose3(0.02, 0, 0, IDENTITY))
        self.assert_twist(twist, (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_short_interval_uses_older_reference_or_fails_closed(self) -> None:
        estimator = BodyTwistEstimator(min_interval_sec=0.02)
        estimator.observe(T0, Pose3(0.0, 0.0, 0.0, IDENTITY))
        self.assert_unavailable(
            estimator,
            estimator.observe(T0 + 5_000_000, Pose3(0.001, 0, 0, IDENTITY)),
            "interval_too_small",
        )
        twist = estimator.observe(T0 + 30_000_000, Pose3(0.006, 0, 0, IDENTITY))
        # The newest sample at least min_interval old (the 5 ms one) is chosen.
        self.assertAlmostEqual(twist.interval_sec, 0.025, places=9)
        self.assert_twist(twist, (0.2, 0.0, 0.0), (0.0, 0.0, 0.0))
        twist = estimator.observe(T0 + 35_000_000, Pose3(0.007, 0, 0, IDENTITY))
        self.assertAlmostEqual(twist.interval_sec, 0.030, places=9)

    def test_reset_and_frame_change_require_a_new_pair(self) -> None:
        estimator = BodyTwistEstimator()
        sensor_a = Pose3(0.1, 0.0, 0.5, IDENTITY)
        sensor_b = Pose3(0.2, 0.0, 0.5, IDENTITY)
        estimator.observe(T0, Pose3(0, 0, 0, IDENTITY), frame_key=sensor_a)
        self.assertIsNotNone(
            estimator.observe(T0 + SEC // 10, Pose3(0.05, 0, 0, IDENTITY), frame_key=sensor_a)
        )
        self.assert_unavailable(
            estimator,
            estimator.observe(T0 + 2 * SEC // 10, Pose3(0.1, 0, 0, IDENTITY), frame_key=sensor_b),
            "frame_changed",
        )
        self.assertIsNotNone(
            estimator.observe(T0 + 3 * SEC // 10, Pose3(0.15, 0, 0, IDENTITY), frame_key=sensor_b)
        )

        estimator.reset("localization_reset")
        self.assertEqual(estimator.reason, "localization_reset")
        self.assertFalse(estimator.diagnostics()["reliable"])
        self.assert_unavailable(
            estimator,
            estimator.observe(T0 + 4 * SEC // 10, Pose3(5.0, 0, 0, IDENTITY), frame_key=sensor_b),
            "first_sample",
        )

    def test_non_finite_or_invalid_samples_fail_closed_and_reset(self) -> None:
        cases = (
            (T0 + SEC // 10, Pose3(math.nan, 0.0, 0.0, IDENTITY)),
            (T0 + SEC // 10, Pose3(0.0, math.inf, 0.0, IDENTITY)),
            (T0 + SEC // 10, Pose3(0.0, 0.0, 0.0, Quaternion(0, 0, math.nan, 1))),
            (T0 + SEC // 10, Pose3(0.0, 0.0, 0.0, Quaternion(0, 0, 0, 0))),
            (0, Pose3(0.0, 0.0, 0.0, IDENTITY)),
        )
        for stamp, bad in cases:
            with self.subTest(bad=bad, stamp=stamp):
                estimator = BodyTwistEstimator()
                estimator.observe(T0, Pose3(0, 0, 0, IDENTITY))
                self.assert_unavailable(estimator, estimator.observe(stamp, bad), "invalid_sample")
                self.assert_unavailable(
                    estimator,
                    estimator.observe(T0 + 2 * SEC // 10, Pose3(0, 0, 0, IDENTITY)),
                    "first_sample",
                )

    def test_diagnostics_count_reasons_and_are_json_safe(self) -> None:
        import json

        estimator = BodyTwistEstimator()
        estimator.observe(T0, Pose3(0, 0, 0, IDENTITY))
        estimator.observe(T0, Pose3(0, 0, 0, IDENTITY))
        estimator.observe(T0 + SEC // 10, Pose3(0, 0, 0, IDENTITY))
        diagnostics = estimator.diagnostics()
        self.assertTrue(diagnostics["reliable"])
        self.assertEqual(diagnostics["reliable_count"], 1)
        self.assertEqual(
            diagnostics["unavailable_counts"],
            {"first_sample": 1, "duplicate_stamp": 1},
        )
        json.dumps(diagnostics, allow_nan=False)

    def test_reset_epoch_and_stale_reasons_let_callers_drop_samples(self) -> None:
        estimator = BodyTwistEstimator()
        epoch = estimator.epoch
        estimator.observe(T0, Pose3(0, 0, 0, IDENTITY))
        self.assertEqual(estimator.epoch, epoch)
        estimator.reset("localization_reset")
        self.assertEqual(estimator.epoch, epoch + 1)
        estimator.observe(T0 + SEC, Pose3(0, 0, 0, IDENTITY))
        estimator.observe(T0 + SEC // 2, Pose3(0, 0, 0, IDENTITY))
        self.assertIn(estimator.reason, BodyTwistEstimator.STALE_SAMPLE_REASONS)
        estimator.observe(T0 + SEC, Pose3(0, 0, 0, IDENTITY))
        self.assertIn(estimator.reason, BodyTwistEstimator.STALE_SAMPLE_REASONS)
        # Gaps and first samples are unavailable twists, not stale poses.
        self.assertNotIn("interval_too_large", BodyTwistEstimator.STALE_SAMPLE_REASONS)
        self.assertNotIn("first_sample", BodyTwistEstimator.STALE_SAMPLE_REASONS)

    def test_high_rate_samples_can_reach_minimum_interval(self) -> None:
        estimator = BodyTwistEstimator()
        estimates = [
            estimator.observe(T0 + i * 1_000_000, Pose3(i * 0.001, 0, 0, IDENTITY))
            for i in range(100)
        ]
        reliable = [twist for twist in estimates if twist is not None]
        self.assertGreater(len(reliable), 0)
        self.assertAlmostEqual(reliable[-1].linear[0], 1.0)

    def test_frame_change_cannot_hide_older_stamp(self) -> None:
        estimator = BodyTwistEstimator()
        pose = Pose3(0, 0, 0, IDENTITY)
        estimator.observe(T0 + SEC, pose, frame_key="new")
        estimator.observe(T0, pose, frame_key="old")
        self.assertEqual(estimator.reason, "out_of_order")
        estimator.observe(T0 + SEC, pose, frame_key="old")
        self.assertEqual(estimator.reason, "duplicate_stamp")
        estimator.observe(T0 + SEC + 30_000_000, pose, frame_key="new")
        self.assertIsNone(estimator.reason)

    def test_rejects_invalid_configuration(self) -> None:
        for kwargs in (
            {"min_interval_sec": 0.0},
            {"min_interval_sec": 0.5, "max_interval_sec": 0.4},
            {"max_interval_sec": math.nan},
            {"position_noise_m": -1.0},
            {"history_size": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BodyTwistEstimator(**kwargs)


class AdapterTwistWiringTest(unittest.TestCase):
    def test_adapter_publishes_estimated_twist_not_raw_passthrough(self) -> None:
        adapter = (PACKAGE_ROOT / "fast_livo2" / "adapter_node.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("output.twist = message.twist", adapter)
        self.assertIn("self._odom_twist.observe(", adapter)
        self.assertIn("frame_key=base_to_sensor", adapter)
        self.assertIn("self._odom_twist.odometry_twist_fields(", adapter)
        self.assertIn('self._odom_twist.reset("raw_odom_rejected")', adapter)
        # Every localization/health reset also restarts the twist estimator.
        self.assertEqual(
            adapter.count("self._odom_health.reset("),
            adapter.count('self._odom_twist.reset("localization_reset")'),
        )
        self.assertIn('"odom_twist": self._odom_twist.diagnostics()', adapter)
        self.assertIn("self._odom_twist.epoch != twist_epoch", adapter)
        self.assertIn("BodyTwistEstimator.STALE_SAMPLE_REASONS", adapter)


if __name__ == "__main__":
    unittest.main()
