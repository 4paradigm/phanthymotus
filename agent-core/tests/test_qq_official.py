"""Mocked official QQ adapter tests: no network, credentials or QQ PII."""
import asyncio
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from channel.adapter import OutboundMessage
from channel.adapters.qq_official import QQOfficialAdapter, validate_qq_official_config

class MockApi:
    instances = []

    def __init__(self, app_id, credential):
        self.messages = []
        self.__class__.instances.append(self)

    def setup(self, http):
        self.http = http

    async def ensure_token(self):
        return "test"

    async def get_gateway_url(self):
        return "wss://mock.invalid"

    def ensure_token_sync(self):
        return "test"

    def get_gateway_url_sync(self):
        return "wss://mock.invalid"

    def clear_token(self):
        pass

    async def send_text(self, *args, **kwargs):
        self.messages.append((args, kwargs))


class MockWs:
    instances = []

    def __init__(self, callbacks, log_tag):
        self.callbacks = callbacks
        self.closed = False
        self.__class__.instances.append(self)

    def start(self, _url, _loop):
        self.callbacks.on_connected()

    async def async_stop(self):
        self.closed = True


class MockCallbacks:
    def __init__(self, *args):
        names = (
            "on_message_event", "on_connected", "on_disconnected", "on_fatal_error",
            "get_token", "get_session", "set_session", "set_heartbeat_interval",
            "clear_token", "fail_pending", "get_gateway_url",
        )
        for name, callback in zip(names, args):
            setattr(self, name, callback)


class MockParser:
    def parse(self, event_type, raw):
        return types.SimpleNamespace(event_type=event_type, **raw)


def message(user="OpenID123456789", mid="mid-01", **overrides):
    value = {
        "chat_scope": "c2c", "chat_id": user, "user_id": user,
        "message_id": mid, "content": "hello", "user_name": "someone",
        "attachments": [], "msg_elements": [],
    }
    value.update(overrides)
    return types.SimpleNamespace(event_type="C2C_MESSAGE_CREATE", **value)


class QQOfficialTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        MockApi.instances.clear()
        MockWs.instances.clear()
        self.inbound = []

        async def callback(msg):
            self.inbound.append(msg)

        config = {"protocol": "official", "app_id": "1900000000",
                  "private_only": True, "connect_timeout": 5}
        config["app_secret_env"] = "QQ_OFFICIAL_TEST"
        self.adapter = QQOfficialAdapter("official_test", "qq", config, callback)
        env = mock.patch.dict(os.environ, {"QQ_OFFICIAL_TEST": "test"})
        env.start()
        self.addCleanup(env.stop)

        sdk = mock.patch.dict(sys.modules, {"qqbot_agent_sdk": types.SimpleNamespace(
            QQApiClient=MockApi, QQWebSocket=MockWs,
            WSCallbacks=MockCallbacks, EventParser=MockParser,
        )})
        sdk.start()
        self.addCleanup(sdk.stop)

    async def asyncTearDown(self):
        await self.adapter.stop()

    async def test_lifecycle_and_status(self):
        await self.adapter.start()
        self.assertEqual(self.adapter.status(), "connected")
        self.assertEqual(await self.adapter.health_check(), (True, ""))
        MockWs.instances[-1].callbacks.on_disconnected()
        await asyncio.sleep(0)
        self.assertEqual(self.adapter.status(), "degraded")
        await self.adapter.stop()
        self.assertTrue(MockWs.instances[-1].closed)

    async def test_c2c_parser_and_dedup(self):
        a = self.adapter._parse_message(message())
        b = self.adapter._parse_message(message("OpenIDother12345", "mid-02"))
        self.assertEqual(a.platform, "qq")
        self.assertEqual(a.chat_type, "p2p")
        self.assertNotEqual(a.user_id, b.user_id)
        self.assertTrue(a.user_id.startswith("official:1900000000:"))
        self.assertIsNone(self.adapter._parse_message(message()))
        self.assertIsNone(self.adapter._parse_message(message(mid="a", chat_scope="group")))
        self.assertIsNone(self.adapter._parse_message(message(mid="b", attachments=[object()])))
        self.assertIsNone(self.adapter._parse_message(message(mid="c", msg_elements=[object()])))
        self.assertIsNotNone(self.adapter._parse_message(message(
            mid="text-element", msg_elements=[types.SimpleNamespace(
                content="hello", attachments=[])])))
        self.assertIsNone(self.adapter._parse_message(message(
            mid="media-element", msg_elements=[types.SimpleNamespace(
                content="hello", attachments=[object()])])))
        self.assertIsNone(self.adapter._parse_message(message(mid="d", chat_id="other")))
        self.assertIsNone(self.adapter._parse_message(message(mid="e", content="")))

    async def test_callback_only_once(self):
        await self.adapter.start()
        raw = vars(message()).copy()
        raw.pop("event_type")
        cb = MockWs.instances[-1].callbacks
        await cb.on_message_event("C2C_MESSAGE_CREATE", raw)
        await cb.on_message_event("C2C_MESSAGE_CREATE", raw)
        self.assertEqual(len(self.inbound), 1)

    async def test_reply_has_source_and_never_retries(self):
        await self.adapter.start()
        target = "OpenID123456789"
        with self.assertRaises(ValueError):
            await self.adapter.send_message(OutboundMessage(chat_id=target, text="reply"))
        await self.adapter.send_message(OutboundMessage(
            chat_id=target, text="reply", reply_to_message_id="mid-01"))
        args, kwargs = MockApi.instances[-1].messages[0]
        self.assertEqual(args, ("c2c", target, "reply"))
        self.assertEqual(kwargs["reply_to"], "mid-01")
        self.assertEqual(kwargs["retries"], 1)
        self.assertFalse(kwargs["markdown"])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            await self.adapter.send_message(OutboundMessage(
                chat_id=target, text="duplicate", reply_to_message_id="mid-01"))
        self.assertEqual(len(MockApi.instances[-1].messages), 1)

    async def test_unconfirmed_timeout_is_not_retried(self):
        await self.adapter.start()
        api = MockApi.instances[-1]
        api.send_text = mock.AsyncMock(side_effect=TimeoutError())
        original = OutboundMessage(
            chat_id="OpenID123456789", text="answer",
            reply_to_message_id="retry-unsafe-message")
        with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
            await self.adapter.send_message(original)
        with self.assertRaisesRegex(ValueError, "already attempted"):
            await self.adapter.send_message(original)
        api.send_text.assert_awaited_once()

    async def test_group_and_proactive_are_not_supported(self):
        await self.adapter.start()
        with self.assertRaisesRegex(ValueError, "source message"):
            await self.adapter.send_message(OutboundMessage(
                chat_id="OpenID123456789", text="unsolicited"))
        with self.assertRaisesRegex(ValueError, "text only"):
            await self.adapter.send_message(OutboundMessage(
                chat_id="OpenID123456789", text="image",
                reply_to_message_id="m2", image_bytes=b"x"))

    async def test_manager_routes_official_reply_to_original_openid(self):
        import json
        from channel.manager import ChannelManager
        manager = ChannelManager()
        store = {
            "canvas_layout": {"cards": [
                {"id": "in", "mcpId": "channel", "toolName": "channel_request"},
                {"id": "out", "mcpId": "channel", "toolName": "channel_reply"},
            ]},
            "tool_config:channel:channel_request:in": {"channel_id": "official_test"},
            "tool_config:channel:channel_reply:out": {"channel_id": "official_test"},
            "channel_configs": [{
                "id": "official_test", "platform": "qq", "enabled": True,
                "config": {"protocol": "official"},
            }],
        }
        self.adapter._on_message = manager._on_inbound_message
        manager._adapters["official_test"] = self.adapter
        replies = []

        async def publish(_topic, raw):
            value = json.loads(raw)
            outcome = await manager.send_to_channel(
                "official_test", text="answer:" + value["text"],
                source_message_id=value["message_id"],
            )
            replies.append(outcome)

        with (mock.patch("channel.manager.config.main", store),
              mock.patch("channel.acl.get_user", return_value={
                  "role": "viewer", "tool_filter": "*"}),
              mock.patch("channel.acl.check_permission", return_value=(True, "")),
              mock.patch("api.inspection.publish_to_topic",
                         new_callable=mock.AsyncMock, side_effect=publish),
              mock.patch("api.motus_stream.push_event", new_callable=mock.AsyncMock)):
            await self.adapter.start()
            callback = MockWs.instances[-1].callbacks.on_message_event
            for user, mid, word in (
                    ("OpenID123456789", "mid-A", "first"),
                    ("OpenIDother12345", "mid-B", "second")):
                raw = vars(message(user=user, mid=mid, content=word)).copy()
                raw.pop("event_type")
                await callback("C2C_MESSAGE_CREATE", raw)
            sent = [x[0] for x in MockApi.instances[-1].messages]
            meta = [x[1] for x in MockApi.instances[-1].messages]
            self.assertEqual([args[1:] for args in sent], [
                ("OpenID123456789", "answer:first"),
                ("OpenIDother12345", "answer:second"),
            ])
            self.assertEqual([m["reply_to"] for m in meta], ["mid-A", "mid-B"])
            self.assertEqual(len(replies), 2)

    def test_config_fail_closed(self):
        cfg = {"protocol": "official", "app_id": "1900000000"}
        cfg["app_secret_env"] = "QQ_OFFICIAL_TEST"
        validate_qq_official_config(cfg)
        for changes in (
            {"protocol": "onebot"},
            {"app_id": "not-numeric"},
            {"private_only": False},
            {"app_secret_env": "INVALID"},
        ):
            with self.assertRaises(ValueError):
                validate_qq_official_config({**cfg, **changes})
