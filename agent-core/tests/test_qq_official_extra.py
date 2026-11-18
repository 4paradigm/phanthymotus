"""Checks for optional QQ audio, Gateway intents, and admin CLI."""
from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))
from channel.adapters.qq_official_voice import read_voice_text
from qq_official_onboard import save_new_credential


class QQExtraTests(unittest.IsolatedAsyncioTestCase):
    async def test_asr_fallback_without_network(self):
        event = types.SimpleNamespace(
            attachments=[types.SimpleNamespace(asr_refer_text="spoken words")],
            msg_elements=[])
        self.assertEqual(await read_voice_text(None, event, []), "spoken words")

    async def test_gateway_intents_follow_enabled_scopes(self):
        """No private/interactive/guild bits when C2C-only is configured."""
        class FakeGateway:
            def __init__(self, callbacks, log_tag=""):
                self._cb = callbacks
                self.payload = None
            async def _send_ws_json(self, packet, reason):
                self.payload = packet
        fake = types.SimpleNamespace(
            QQWebSocket=FakeGateway,
            OPCode=types.SimpleNamespace(IDENTIFY=2),
            Intent=types.SimpleNamespace(
                GROUP_MESSAGES=1 << 25,
                GUILD_AT_MESSAGE=1 << 30,
                DIRECT_MESSAGES=1 << 12,
                GUILD_MESSAGES=1 << 9,
                INTERACTION=1 << 26,
            ),
        )
        with mock.patch.dict(sys.modules, {"qqbot_agent_sdk": fake}):
            sys.modules.pop("channel.adapters.qq_official_gateway", None)
            from channel.adapters.qq_official_gateway import (
                QQScopedWebSocket, qq_gateway_intents,
            )
            cases = (
                ({"enabled_scopes": ["c2c"]}, 1 << 25),
                ({"enabled_scopes": ["c2c", "group"]}, 1 << 25),
                ({"enabled_scopes": ["guild"]}, 1 << 30),
                ({"enabled_scopes": ["guild_dm"]}, 1 << 12),
                ({"enabled_scopes": ["guild"],
                  "enable_guild_plain_messages": True},
                 (1 << 30) | (1 << 9)),
                ({"enabled_scopes": ["c2c"],
                  "enable_interactions": True},
                 (1 << 25) | (1 << 26)),
            )
            for conf, expected in cases:
                self.assertEqual(qq_gateway_intents(conf), expected)
                ws = QQScopedWebSocket(
                    types.SimpleNamespace(get_token=lambda: "test"),
                    intents=qq_gateway_intents(conf),
                )
                await ws._send_identify()
                self.assertEqual(ws.payload["d"]["intents"], expected)
                self.assertEqual(ws.payload["op"], 2)
            self.assertFalse(qq_gateway_intents({"enabled_scopes": ["c2c"]})
                             & (1 << 9))
        sys.modules.pop("channel.adapters.qq_official_gateway", None)

    async def test_installed_sdk_accepts_scoped_identify_payload(self):
        try:
            from qqbot_agent_sdk import Intent
        except ImportError:
            self.skipTest("official SDK not installed in this environment")
        from channel.adapters.qq_official_gateway import (
            QQScopedWebSocket, qq_gateway_intents,
        )
        scope = qq_gateway_intents({"enabled_scopes": ["c2c"]})
        self.assertEqual(scope, int(Intent.GROUP_MESSAGES))
        ws = QQScopedWebSocket(
            types.SimpleNamespace(get_token=lambda: "test-access-token"),
            intents=scope,
            log_tag="local-test",
        )
        ws._send_ws_json = mock.AsyncMock()
        await ws._send_identify()
        packet = ws._send_ws_json.await_args.args[0]
        self.assertEqual(packet["d"]["intents"], 1 << 25)
        self.assertEqual(packet["op"], 2)

    async def test_group_full_event_routes_through_scoped_gateway(self):
        from qqbot_agent_sdk import Intent
        from channel.adapters.qq_official_gateway import QQScopedWebSocket
        events = []
        callback = types.SimpleNamespace(
            on_message_event=lambda name, raw: (name, raw),
        )
        gateway = QQScopedWebSocket(
            callback, intents=int(Intent.GROUP_MESSAGES), log_tag="test")
        gateway._is_duplicate = mock.Mock(return_value=False)
        gateway._dispatch_to_main = lambda payload: events.append(payload)
        event = {"id": "group-full-message-001", "mentions": [{"is_you": True}]}
        gateway._handle_dispatch("GROUP_MESSAGE_CREATE", event)
        self.assertEqual(events, [("GROUP_MESSAGE_CREATE", event)])
        gateway._is_duplicate.assert_called_once_with("group-full-message-001")


    async def test_full_group_websocket_does_not_log_every_message(self):
        """Busy group traffic must not bypass adapter-side sampled logging."""
        # Load the real custom gateway class against a minimal SDK stub so
        # this log-volume regression also runs on SDK-less developer hosts.
        import importlib.util

        class FakeGateway:
            def __init__(self, callbacks, log_tag=""):
                self._cb = callbacks

        fake_sdk = types.SimpleNamespace(
            QQWebSocket=FakeGateway,
            OPCode=types.SimpleNamespace(IDENTIFY=2),
            Intent=types.SimpleNamespace(GROUP_MESSAGES=1))
        spec = importlib.util.spec_from_file_location(
            "_qq_gateway_log_test",
            root / "src" / "channel" / "adapters" / "qq_official_gateway.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"qqbot_agent_sdk": fake_sdk}):
            spec.loader.exec_module(module)

        received = []
        callback = types.SimpleNamespace(
            on_message_event=lambda name, raw: (name, raw))
        gateway = module.QQScopedWebSocket(
            callback, intents=1, log_tag="test")
        gateway._is_duplicate = mock.Mock(
            side_effect=lambda message_id: message_id == "duplicate")
        gateway._dispatch_to_main = received.append

        # Includes non-@ messages: only the adapter may decide whether
        # to accept and sample their diagnostics.
        with mock.patch("builtins.print") as log:
            for index in range(201):
                gateway._handle_dispatch(
                    "GROUP_MESSAGE_CREATE", {"id": f"group-{index}",
                    "mentions": []})
            gateway._handle_dispatch(
                "GROUP_MESSAGE_CREATE", {"id": "duplicate", "mentions": []})

        self.assertEqual(len(received), 201)
        self.assertTrue(all(name == "GROUP_MESSAGE_CREATE"
                            for name, _ in received))
        log.assert_not_called()

    async def test_full_group_mention_filter_fails_closed(self):
        from channel.adapters.qq_official import _full_group_mentions_bot
        self.assertFalse(_full_group_mentions_bot({"mentions": []}))
        self.assertFalse(_full_group_mentions_bot({"mentions": [{"is_you": False}]}))
        self.assertFalse(_full_group_mentions_bot({"mentions": [{"openid": "OtherUser123456"}]}))
        self.assertTrue(_full_group_mentions_bot({"mentions": [{"is_you": True}]}))
        self.assertTrue(_full_group_mentions_bot(
            {"mentions": [{"openid": "BotGroupOpenID12345"}]},
            "BotGroupOpenID12345"))
        self.assertFalse(_full_group_mentions_bot(
            {"mentions": [{"openid": "OtherUser123456"}]},
            "BotGroupOpenID12345"))

    async def test_group_full_payload_matches_sdk_group_parser(self):
        from qqbot_agent_sdk import EventParser
        from channel.adapters.qq_official import QQOfficialAdapter
        raw = {
            "id": "group-full-message-002",
            "group_openid": "GroupOpenID12345",
            "author": {"member_openid": "MemberOpenID12345"},
            "content": " hello",
            "mentions": [{"is_you": True}],
            "timestamp": "2026-11-12T12:00:00+00:00",
        }
        event = EventParser.parse("GROUP_AT_MESSAGE_CREATE", raw)
        self.assertIsNotNone(event)
        adapter = QQOfficialAdapter("qq-test", "qq", {
            "protocol": "official", "app_id": "1900000000",
            "enabled_scopes": ["c2c", "group"], "private_only": False,
        }, lambda _: None)
        parsed = adapter._parse_message(event)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.chat_type, "group")
        self.assertEqual(parsed.chat_id, "GroupOpenID12345")

    async def test_admin_setup_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config"
            save_new_credential(path, "example")
            self.assertEqual(path.read_text(), "example\n")
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                save_new_credential(path, "different")
            self.assertEqual(path.read_text(), "example\n")
