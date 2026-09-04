"""
test_peer_ble.py — P5 BLE bootstrap 测试。

验证：
1. is_available() 正确检测 bleak 可用性
2. start() 在 bleak 不可用时静默跳过
3. advertiser is_available() 检测
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

from peer import identity, ble_bootstrap, ble_advertiser  # noqa: E402


class TestBLE(unittest.TestCase):
    def setUp(self):
        identity.reset_cache()
        identity.ensure_identity()

    def test_is_available_returns_bool(self):
        """is_available() 返回布尔值（节点 1）"""
        available = ble_bootstrap.is_available()
        self.assertIsInstance(available, bool)

    def test_start_stop_without_crash(self):
        """start/stop 不会崩溃（节点 2）"""
        # Should not raise regardless of bleak availability
        ble_bootstrap.start()
        ble_bootstrap.stop()

    def test_advertiser_is_available_returns_bool(self):
        """advertiser is_available() 返回布尔值"""
        available = ble_advertiser.is_available()
        self.assertIsInstance(available, bool)

    def test_bleak_not_available_in_test_env(self):
        """当前测试环境没有 bleak（验证降级）"""
        # This test documents the expected state: bleak is not installed
        # in the CI/test environment. If it is installed, this test will
        # fail, which is fine — it means BLE can actually run.
        if ble_bootstrap.BLEAK_AVAILABLE:
            self.skipTest('bleak is installed, BLE actually available')
        else:
            self.assertFalse(ble_bootstrap.is_available())


if __name__ == '__main__':
    unittest.main()
