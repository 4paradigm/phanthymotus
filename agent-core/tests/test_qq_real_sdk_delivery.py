"""True SDK HTTP retry semantics, not a mock send_text implementation.

This suite is skipped without qqbot-agent-sdk and runs in the company Core
image where the official dependency is installed.
"""
import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAS_SDK = importlib.util.find_spec("qqbot_agent_sdk") is not None

@unittest.skipUnless(HAS_SDK, "Official QQ SDK not installed here")
class QQRealSDKSendSemantics(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_makes_exactly_one_underlying_http_call(self):
        from qqbot_agent_sdk import QQApiClient
        from channel.adapter import OutboundMessage
        from channel.adapters.qq_official import QQOfficialAdapter

        sdk = QQApiClient("1900000000", "synthetic-credential")
        adapter = QQOfficialAdapter(
            "qq", "qq", {"protocol": "official", "app_id": "1900000000"},
            mock.AsyncMock())
        adapter._api = sdk
        adapter._connected = asyncio.Event()
        adapter._connected.set()
        adapter._running = True
        # A passive reply is permitted only after the original QQ message
        # has been accepted and bound to this exact C2C destination.
        from types import SimpleNamespace
        inbound = SimpleNamespace(
            event_type="C2C_MESSAGE_CREATE",
            chat_scope="c2c",
            chat_id="OfficialOpenIDTest001",
            user_id="OfficialOpenIDTest001",
            message_id="sdk-timeout-once",
            content="synthetic question",
            user_name="test",
            attachments=[],
            msg_elements=[],
        )
        self.assertIsNotNone(adapter._parse_message(inbound))
        outbound = OutboundMessage(
            chat_id="OfficialOpenIDTest001",
            text="synthetic reply",
            reply_to_message_id="sdk-timeout-once")

        with mock.patch.object(
            sdk, "post_c2c_message",
            new_callable=mock.AsyncMock, side_effect=TimeoutError("synthetic timeout"),
        ) as post:
            with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                await adapter.send_message(outbound)
            self.assertEqual(post.await_count, 1,
                             "SDK retries=1 must mean one total HTTP attempt")
            with self.assertRaisesRegex(ValueError, "already attempted"):
                await adapter.send_message(outbound)
            self.assertEqual(post.await_count, 1)
