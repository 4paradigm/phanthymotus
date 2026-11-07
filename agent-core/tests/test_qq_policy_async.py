"""ACL isolation must not block the asynchronous Agent Core event loop."""
import asyncio
import pathlib
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from channel import tool_policy


class QQPolicyAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_does_not_block_event_loop_when_sqlite_is_busy(self):
        entered = asyncio.Event()

        def blocked_get_user(platform, user_id):
            time.sleep(0.12)  # simulate SQLite busy_timeout on worker thread
            return {"role": "viewer", "tool_filter": "*"}

        with mock.patch.object(tool_policy.acl, "get_user", side_effect=blocked_get_user):
            read = asyncio.create_task(tool_policy.snapshot_principals([("qq", "person")]))
            await asyncio.wait_for(asyncio.sleep(0.015), timeout=0.06)
            self.assertFalse(read.done(), "The synthetic SQLite read should still be running")
            snapshot = await read
        self.assertEqual(snapshot[("qq", "person")]["role"], "viewer")
        self.assertTrue(tool_policy.allowed_for_principals(
            [("qq", "person")], "mcp__channel__channel_reply", snapshot=snapshot))

    async def test_dispatch_rechecks_revoked_role(self):
        users = iter([
            {"role": "operator", "tool_filter": "*"},
            {"role": "blocked", "tool_filter": "*"},
        ])
        with mock.patch.object(tool_policy.acl, "get_user", side_effect=lambda *_: next(users)):
            first = await tool_policy.async_allowed_for_principals(
                [("qq", "person")], "mcp__channel__channel_reply")
            second = await tool_policy.async_allowed_for_principals(
                [("qq", "person")], "mcp__channel__channel_reply")
        self.assertTrue(first)
        self.assertFalse(second)


if __name__ == "__main__":
    unittest.main()
