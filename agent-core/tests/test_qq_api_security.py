"""QQ API config validation and response-redaction tests."""
import pathlib
import sys
import unittest
from unittest import mock

import fastapi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from api import channel as api_channel  # noqa: E402


class QQApiSecurityTests(unittest.IsolatedAsyncioTestCase):
    def official_cfg(self):
        cfg = {"protocol": "official", "private_only": True,
               "app_id": "1900000000"}
        cfg["app_secret_env"] = "QQ_OFFICIAL_TEST"
        return cfg

    async def test_official_add_accepts_env_reference(self):
        cfg = self.official_cfg()
        entry = {"id": "qq_official", "platform": "qq", "config": cfg}
        with mock.patch.object(api_channel, "add_channel_config", return_value=entry):
            result = await api_channel.add_channel(api_channel.AddChannelReq(
                id="qq_official", platform="qq", config=cfg))
        self.assertEqual(result["channel"]["config"]["app_id"], "1900000000")

    def test_official_invalid_config_fails_closed(self):
        cfg = self.official_cfg()
        cfg["private_only"] = False
        with self.assertRaises(fastapi.HTTPException) as caught:
            api_channel._validate_channel_config("qq", cfg)
        self.assertEqual(caught.exception.status_code, 400)

    async def test_official_update_keeps_same_env_source(self):
        existing = {"id": "qq_official", "platform": "qq",
                    "config": self.official_cfg()}
        with (
            mock.patch.object(api_channel, "get_channel_config", return_value=existing),
            mock.patch.object(api_channel, "update_channel_config", return_value=existing) as save,
            mock.patch.object(api_channel.manager, "restart_adapter", new_callable=mock.AsyncMock),
        ):
            await api_channel.update_channel(
                "qq_official", api_channel.UpdateChannelReq(
                    config={"protocol": "official", "app_id": "1900000001"}))
        saved_cfg = save.call_args.kwargs["config"]
        self.assertEqual(saved_cfg["app_id"], "1900000001")
        self.assertEqual(saved_cfg["app_secret_env"], "QQ_OFFICIAL_TEST")

if __name__ == "__main__":
    unittest.main()
