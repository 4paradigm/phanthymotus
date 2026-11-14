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


class QQBarrierAndMemoryAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_revocation_after_blocked_acp_barrier(self):
        from pathlib import Path
        import config
        owner="official:1900000000:OWNEREXAMPLE123456789"
        principals=[("qq",owner)]
        grant={"qq_c2c_owner_grant":{"enabled":True,"principal":owner}}
        state={"role":"owner","tool_filter":"*"}
        ready,release=asyncio.Event(),asyncio.Event()

        async def simulated_acp_wait():
            ready.set()
            await release.wait()

        with (mock.patch("config.main",grant),
              mock.patch.object(tool_policy.acl,"get_user",
                                side_effect=lambda *_:dict(state))):
            self.assertTrue(await tool_policy.async_allowed_for_principals(
                principals,"mcp__robot__loco"))
            wait=asyncio.create_task(simulated_acp_wait())
            await ready.wait()
            self.assertFalse(wait.done())
            state["role"]="blocked"
            release.set()
            await wait
            self.assertFalse(await tool_policy.async_allowed_for_principals(
                principals,"mcp__robot__loco"))
        # Verify the dispatch actually rechecks after waiting, not just before.
        source=(Path(__file__).resolve().parents[1]/"src"/"event"/"llm.py").read_text(encoding="utf-8")
        start=source.index("if _bar_needed:",source.index("# ACP barrier: 只等与本次调用资源冲突"))
        dispatch=source.index("result = await mcp_client.call_tool(name, args)",start)
        self.assertIn("elif channel_principals and not await async_allowed_for_principals(channel_principals, name):",
                      source[start:dispatch])

    async def test_chat_memory_sqlite_busy_does_not_stall_event_loop(self):
        import event.memory
        from channel.manager import manager,qq_memory_scope
        token=qq_memory_scope.set(("qq_official_1900000000","p2p:TESTUSER"))
        try:
            operations=[
                (event.memory.channel_memory_read,"get_chat_memory",["FACT"]),
                (lambda:event.memory.channel_memory_remember("FACT"),"remember_chat_memory",True),
                (lambda:event.memory.channel_memory_forget("FACT"),"forget_chat_memory",1),
            ]
            for call, method, expected in operations:
                def blocked(*args,result=expected):
                    time.sleep(.12)
                    return result
                with mock.patch.object(manager,method,side_effect=blocked):
                    pending=asyncio.create_task(call())
                    await asyncio.wait_for(asyncio.sleep(.015),timeout=.075)
                    self.assertFalse(pending.done())
                    self.assertTrue(await pending)
        finally:
            qq_memory_scope.reset(token)
