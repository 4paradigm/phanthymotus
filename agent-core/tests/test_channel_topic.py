"""Regression tests for ROS-safe topics derived from user-facing Channel IDs."""
import os
import pathlib
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from channel.manager import channel_request_topic  # noqa: E402


class ChannelRequestTopicTest(unittest.TestCase):
    def test_valid_ascii_channel_id_keeps_its_existing_topic(self):
        self.assertEqual(channel_request_topic('feishu_r1'),
                         '/channel/request/feishu_r1')
        self.assertEqual(channel_request_topic(''), '/channel/request')

    def test_non_ascii_and_numeric_ids_become_ros_safe_and_deterministic(self):
        ids = ['上海 G1', '北京 G1', '123 channel', 'hr-feishu']
        topics = [channel_request_topic(channel_id) for channel_id in ids]

        self.assertTrue(all(
            re.fullmatch(r'/channel/request/[A-Za-z_][A-Za-z0-9_]*', topic)
            for topic in topics
        ))
        self.assertEqual(channel_request_topic('上海 G1'), topics[0])
        self.assertEqual(len(set(topics)), len(topics))


if __name__ == '__main__':
    unittest.main()
