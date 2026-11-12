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

    async def test_extended_gateway_intent_is_local(self):
        class FakeGateway:
            def __init__(self, callbacks, log_tag=""):
                self._cb = callbacks
                self.payload = None
            async def _send_ws_json(self, packet, reason):
                self.payload = packet
        fake = types.SimpleNamespace(
            QQWebSocket=FakeGateway,
            OPCode=types.SimpleNamespace(IDENTIFY=2),
            DEFAULT_INTENTS=1,
            Intent=types.SimpleNamespace(GUILD_MESSAGES=512),
        )
        with mock.patch.dict(sys.modules, {"qqbot_agent_sdk": fake}):
            sys.modules.pop("channel.adapters.qq_official_gateway", None)
            from channel.adapters.qq_official_gateway import QQGuildWebSocket
            ws = QQGuildWebSocket(
                types.SimpleNamespace(get_token=lambda: "example"))
            await ws._send_identify()
            self.assertEqual(ws.payload["d"]["intents"], 513)
        sys.modules.pop("channel.adapters.qq_official_gateway", None)

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
