"""Regression tests for the QQ inbound SQLite ACL hot path."""

import asyncio
import sqlite3
import threading
import time
from unittest import IsolatedAsyncioTestCase, mock

from channel.adapter import InboundMessage
from channel.manager import ChannelManager


def _message(user_id="qq-user"):
    return InboundMessage(
        platform="qq", channel_id="qq-main", user_id=user_id,
        chat_id="chat-1", display_name="Test user", text="hello",
        message_id="source-1", chat_type="p2p",
    )


class QQInboundAclAsyncTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = ChannelManager()
        self.manager._active_input_channels.add("qq-main")
        self.manager.sync_from_canvas = mock.Mock()
        self.manager._set_last_context = mock.Mock()
        self.manager._record_chat_exchange = mock.Mock()

    async def _run_message(self, msg=None):
        with mock.patch(
            "api.inspection.publish_to_topic", new_callable=mock.AsyncMock
        ) as publish, mock.patch(
            "api.motus_stream.push_event", new_callable=mock.AsyncMock
        ) as _push:
            await self.manager._on_inbound_message(msg or _message())
            return publish

    async def test_slow_sqlite_acl_does_not_block_sibling_task(self):
        started = threading.Event()
        sibling_ran = asyncio.Event()

        def slow_get_user(_platform, _user_id):
            started.set()
            time.sleep(0.20)
            return {"role": "viewer", "tool_filter": "*"}

        async def sibling():
            sibling_ran.set()

        with (
            mock.patch("channel.manager.acl.get_user", side_effect=slow_get_user),
            mock.patch("channel.manager.acl.check_permission", return_value=(True, "ok")),
        ):
            task = asyncio.create_task(self._run_message())
            await asyncio.wait_for(asyncio.to_thread(started.wait, 1), 1)
            await asyncio.wait_for(asyncio.create_task(sibling()), 0.05)
            self.assertTrue(sibling_ran.is_set())
            await task

    async def test_locked_sqlite_denies_without_publishing(self):
        with mock.patch(
            "channel.manager.acl.get_user",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            publish = await self._run_message()

        publish.assert_not_awaited()
        self.manager._set_last_context.assert_not_called()

    async def test_registration_preserves_blocked_race_and_uses_worker(self):
        blocked = {"role": "blocked", "tool_filter": "*"}
        with (
            mock.patch("channel.manager.acl.get_user", side_effect=[None, blocked]),
            mock.patch("channel.manager.acl.upsert_user") as upsert,
            mock.patch("channel.manager.acl.check_permission", return_value=(False, "blocked")),
        ):
            publish = await self._run_message(_message("racing-user"))

        upsert.assert_called_once()
        self.assertTrue(upsert.call_args.kwargs["preserve_existing"])
        publish.assert_not_awaited()
        self.manager._set_last_context.assert_not_called()

    async def test_preserve_existing_does_not_overwrite_owner_or_blocked_row(self):
        from channel import acl

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute(
            "CREATE TABLE channel_users ("
            "platform TEXT, platform_user_id TEXT, display_name TEXT, "
            "role TEXT, tool_filter TEXT, alert_subscriptions TEXT DEFAULT '[]', "
            "created_at REAL, UNIQUE(platform, platform_user_id))"
        )
        try:
            for role in ("owner", "blocked"):
                conn.execute(
                    "DELETE FROM channel_users WHERE platform=? AND platform_user_id=?",
                    ("qq", "race-user"),
                )
                conn.execute(
                    "INSERT INTO channel_users "
                    "(platform, platform_user_id, role, tool_filter) VALUES (?, ?, ?, ?)",
                    ("qq", "race-user", role, "*"),
                )
                conn.commit()
                with mock.patch("channel.acl._conn", return_value=conn):
                    result = await asyncio.to_thread(
                        acl.upsert_user,
                        "qq", "race-user", "new display name",
                        role="viewer", tool_filter="chat-only",
                        preserve_existing=True,
                    )
                self.assertEqual(result["role"], role)
                self.assertEqual(result["tool_filter"], "*")
        finally:
            conn.close()
