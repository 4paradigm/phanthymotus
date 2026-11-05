"""Offline regressions for concurrent raw odom callbacks and reset."""
from collections import deque
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace as NS
from unittest import mock

# Reuse the existing ROS import shim; no ROS graph or robot is contacted.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fast_livo2_runtime_supervisor import ADAPTER_MODULE as module


TIMEOUT = 3.0


def vector():
    return NS(x=0.0, y=0.0, z=0.0)


def odometry():
    return NS(header=NS(stamp=None, frame_id=""), child_frame_id="",
              pose=NS(pose=NS(position=vector(), orientation=NS(x=0., y=0., z=0., w=1.)), covariance=[0.] * 36),
              twist=NS(twist=NS(linear=vector(), angular=vector()), covariance=[0.] * 36))


def transform():
    return NS(header=None, child_frame_id="", transform=NS(translation=vector(), rotation=None))


def raw(sec, x):
    result = odometry()
    result.header.frame_id = "camera_init"
    result.header.stamp = NS(sec=sec, nanosec=0)
    result.child_frame_id = "aft_mapped"
    result.pose.pose.position.x = x
    return result


class OdomRaceTest(unittest.TestCase):
    def setUp(self):
        self.adapter = a = object.__new__(module.FastLivo2Adapter)
        a._lock = threading.Lock()
        a._source_age = lambda _: 0.0
        a._source_max_age = 0.5
        a._source_age_tolerance = 0.05
        a._sensor_frame = "sensor"
        a._base_to_sensor_tf_ready = True
        identity = module.Pose3(0., 0., 0., module.Quaternion(0., 0., 0., 1.))
        a._base_to_sensor = identity
        a._map_from_session = identity
        a._odom_health = module.OdomHealthMonitor()
        a._odom_epoch = 0
        a._last_published_odom = None
        a._latest_session_pose = a._latest_pose = None
        a._last_odom_monotonic = a._last_odom_source_age = None
        a._pose_history = deque(maxlen=128)
        a._invalid_odom = 0
        a._observe_pipeline_event = lambda _: None
        a._mark_valid = lambda _: None
        a._warn_rejected = lambda *args, **kwargs: None
        a._drain_pending_cloud = lambda: None
        self.published = []
        self.transforms = []
        a._odom_pub = NS(publish=self.published.append)
        a._tf = NS(sendTransform=self.transforms.append)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.errors = []
        real_convert = module.canonical_base_pose

        def convert(*args, **kwargs):
            if threading.current_thread().name == "old-odom":
                self.entered.set()
                if not self.release.wait(TIMEOUT):
                    raise TimeoutError("old callback was not released")
            return real_convert(*args, **kwargs)

        self.patches = [mock.patch.object(module, "canonical_base_pose", convert),
                        mock.patch.object(module, "Odometry", odometry),
                        mock.patch.object(module, "TransformStamped", transform)]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.release.set)

    def start(self, name, fn):
        def run():
            try:
                fn()
            except BaseException as exc:
                self.errors.append((name, exc))
        thread = threading.Thread(name=name, target=run, daemon=True)
        thread.start()
        return thread

    def finish(self, *threads):
        self.release.set()
        for thread in threads:
            thread.join(TIMEOUT)
        self.assertFalse([t.name for t in threads if t.is_alive()], "worker join timed out")
        self.assertEqual(self.errors, [], "worker exceptions")

    def test_reset_after_health_does_not_resurrect_old_sample(self):
        a = self.adapter
        old = self.start("old-odom", lambda: a._on_odom(raw(1, 0.0)))
        reset_done = threading.Event()
        def reset():
            # Same critical section and epoch update as _on_reset; the
            # identity alignment deliberately remains equal across sessions.
            with a._lock:
                a._odom_health.reset()
                a._odom_epoch += 1
                a._pose_history.clear()
                a._latest_pose = a._latest_session_pose = None
            reset_done.set()
        reset_thread = None
        try:
            self.assertTrue(self.entered.wait(TIMEOUT), "old callback did not pass health")
            reset_thread = self.start("reset", reset)
            self.assertTrue(reset_done.wait(TIMEOUT), "reset blocked while callback paused")
        finally:
            self.finish(*([old, reset_thread] if reset_thread else [old]))
        with self.subTest("published after reset"):
            self.assertEqual(self.published, [], "old odom published after reset")
        with self.subTest("new history"):
            self.assertEqual(list(a._pose_history), [], "old sample entered reset history")
        with self.subTest("transform after reset"):
            self.assertEqual(self.transforms, [], "old TF published after reset")

    def test_delayed_rejection_does_not_reset_new_session(self):
        a = self.adapter
        real_convert = module.canonical_base_pose

        def reject_old(*args, **kwargs):
            if threading.current_thread().name == "old-odom":
                self.entered.set()
                if not self.release.wait(TIMEOUT):
                    raise TimeoutError("old callback was not released")
                raise module.InvalidFastLivo2Frame("old conversion failed")
            return real_convert(*args, **kwargs)

        with mock.patch.object(module, "canonical_base_pose", reject_old):
            old = self.start("old-odom", lambda: a._on_odom(raw(1, 0.0)))
            new = None
            try:
                self.assertTrue(self.entered.wait(TIMEOUT))
                with a._lock:
                    a._odom_health.reset()
                    a._odom_epoch += 1
                new = self.start("new-odom", lambda: a._on_odom(raw(2, 0.1)))
                new.join(TIMEOUT)
                self.assertFalse(new.is_alive())
                epoch = a._odom_epoch
            finally:
                self.finish(*([old, new] if new else [old]))
        self.assertEqual(a._odom_epoch, epoch)
        self.assertEqual([stamp for stamp, _ in a._pose_history], [2_000_000_000])
        self.assertEqual([m.header.stamp.sec for m in self.published], [2])

    def test_health_fault_discards_inflight_old_callback(self):
        a = self.adapter
        old = self.start("old-odom", lambda: a._on_odom(raw(1, 0.0)))
        bad = None
        try:
            self.assertTrue(self.entered.wait(TIMEOUT))
            bad = self.start("bad-odom", lambda: a._on_odom(raw(1, 0.0)))
            bad.join(TIMEOUT)
            self.assertFalse(bad.is_alive())
        finally:
            self.finish(*([old, bad] if bad else [old]))
        self.assertEqual(self.published, [])
        self.assertEqual(self.transforms, [])
        self.assertEqual(list(a._pose_history), [])

    def test_sensor_geometry_aba_discards_old_callback(self):
        a = self.adapter
        a._lidar_frame = a._imu_frame = "sensor"

        def lookup(_base, frame, _time):
            return NS(transform=NS(
                translation=NS(x=0.2 if frame == "other" else 0., y=0., z=0.),
                rotation=NS(x=0., y=0., z=0., w=1.)))

        a._tf_buffer = NS(lookup_transform=lookup)
        old = self.start("old-odom", lambda: a._on_odom(raw(1, 0.0)))
        try:
            self.assertTrue(self.entered.wait(TIMEOUT))
            with a._lock:
                a._lidar_frame = a._imu_frame = "other"
            a._refresh_sensor_contract()
            with a._lock:
                a._lidar_frame = a._imu_frame = "sensor"
            a._refresh_sensor_contract()
        finally:
            self.finish(old)
        self.assertEqual(self.published, [])
        self.assertEqual(self.transforms, [])
        self.assertEqual(list(a._pose_history), [])

    def test_changed_sensor_geometry_discards_old_callback(self):
        a = self.adapter
        old = self.start("old-odom", lambda: a._on_odom(raw(1, 0.0)))
        new = None
        try:
            self.assertTrue(self.entered.wait(TIMEOUT))
            with a._lock:
                a._base_to_sensor = module.Pose3(
                    0.2, 0., 0., module.Quaternion(0., 0., 0., 1.))
            new = self.start("new-odom", lambda: a._on_odom(raw(2, 0.1)))
            new.join(TIMEOUT)
            self.assertFalse(new.is_alive())
        finally:
            self.finish(*([old, new] if new else [old]))
        self.assertEqual([m.header.stamp.sec for m in self.published], [2])
        self.assertEqual([m.header.stamp.sec for m in self.transforms], [2])

    def test_later_callback_must_not_be_followed_by_older_publication(self):
        a = self.adapter
        old = self.start("old-odom", lambda: a._on_odom(raw(1, 0.0)))
        new = None
        try:
            self.assertTrue(self.entered.wait(TIMEOUT), "old callback did not pass health")
            new = self.start("new-odom", lambda: a._on_odom(raw(2, 0.1)))
            new.join(TIMEOUT)
            self.assertFalse(new.is_alive(), "new callback blocked while old paused")
        finally:
            self.finish(*([old, new] if new else [old]))
        stamps = [m.header.stamp.sec for m in self.published]
        self.assertIn(2, stamps, "new sample was not published")
        self.assertEqual(stamps, sorted(stamps), "older odom published after newer odom")


if __name__ == "__main__":
    unittest.main()
