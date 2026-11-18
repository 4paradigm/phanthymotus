"""QQ API config validation and response-redaction tests."""
import pathlib
import sys
import unittest
from unittest import mock

import fastapi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from api import channel as api_channel  # noqa: E402


class QQApiSecurityTests(unittest.IsolatedAsyncioTestCase):
    def cfg(self):
        return {"ws_url": "ws://127.0.0.1:3001/", "private_only": True,
                "access_token": "demo"}

    async def test_add_response_redacts_qq_credential(self):
        entry = {"id": "qq_main", "platform": "qq", "config": self.cfg()}
        with mock.patch.object(api_channel, "add_channel_config", return_value=entry):
            result = await api_channel.add_channel(api_channel.AddChannelReq(
                id="qq_main", platform="qq", config=self.cfg()))
        self.assertEqual(result["channel"]["config"]["access_token"], "***")

    async def test_edit_blank_credential_keeps_existing(self):
        existing = {"id": "qq_main", "platform": "qq", "config": self.cfg()}
        with (
            mock.patch.object(api_channel, "get_channel_config", return_value=existing),
            mock.patch.object(api_channel, "update_channel_config", return_value=existing) as save,
            mock.patch.object(api_channel.manager, "restart_adapter", new_callable=mock.AsyncMock),
        ):
            result = await api_channel.update_channel("qq_main", api_channel.UpdateChannelReq(
                config={"ws_url": "ws://127.0.0.1:3001/", "access_token": ""}))
        self.assertEqual(save.call_args.kwargs["config"]["access_token"], "demo")
        self.assertEqual(result["channel"]["config"]["access_token"], "***")

    def test_qq_invalid_url_does_not_echo_input(self):
        with self.assertRaises(fastapi.HTTPException) as caught:
            api_channel._validate_channel_config("qq", {
                "ws_url": "ws://127.0.0.1:3001/?query=x", "access_token": "demo"})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertNotIn("query=x", str(caught.exception.detail))


if __name__ == "__main__":
    unittest.main()
