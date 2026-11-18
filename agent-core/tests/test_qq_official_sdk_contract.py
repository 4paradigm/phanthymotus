"""Tencent SDK contract checks without network access.

These cover the real SDK DTOs that mocks cannot validate, on local Python
and the Linux 3.10/ROS test image. Skip if the optional SDK is unavailable.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_HAS_SDK = importlib.util.find_spec("qqbot_agent_sdk") is not None


@unittest.skipUnless(_HAS_SDK, "Official QQ SDK not installed in this environment")
class RealSdkContractTests(unittest.TestCase):
    def test_c2c_event_parser_and_adapter_conversion(self):
        from qqbot_agent_sdk import EventParser
        from channel.adapters.qq_official import QQOfficialAdapter

        user = "OfficialOpenIDTest001"
        raw = {"id": "message-test-100", "content": "你好",
               "author": {"user_openid": user}, "timestamp": "2026-11-07T00:00:00+08:00"}
        event = EventParser.parse("C2C_MESSAGE_CREATE", raw)
        self.assertIsNotNone(event)
        self.assertEqual(event.chat_scope, "c2c")
        self.assertEqual(event.chat_id, user)
        self.assertEqual(event.message_id, raw["id"])
        adapter = QQOfficialAdapter(
            "official", "qq", {"protocol": "official", "app_id": "1900000000"},
            lambda msg: None)
        msg = adapter._parse_message(event)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.chat_id, user)
        self.assertEqual(msg.message_id, raw["id"])
        self.assertEqual(msg.text, raw["content"])
        self.assertEqual(msg.user_id, "official:1900000000:" + user)

    def test_sdk_websocket_and_reply_contracts(self):
        from qqbot_agent_sdk import QQWebSocket, QQApiClient, WSCallbacks
        sig = inspect.signature(QQWebSocket.start)
        self.assertIn("main_loop", sig.parameters)
        self.assertIn("gateway_url", sig.parameters)
        self.assertTrue(inspect.iscoroutinefunction(QQWebSocket.async_stop))
        reply = inspect.signature(QQApiClient.send_text)
        for field in ("chat_type", "chat_id", "content", "reply_to",
                      "markdown", "retries"):
            self.assertIn(field, reply.parameters)
        fields = WSCallbacks.__dataclass_fields__
        for field in ("on_message_event", "on_connected", "on_disconnected",
                      "get_token", "get_gateway_url"):
            self.assertIn(field, fields)
