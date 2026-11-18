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

    def test_malformed_credential_types_are_http_400(self):
        # Client-controlled JSON must never escape validation as TypeError/500.
        for key in ("app_secret", "app_secret_env", "app_secret_file"):
            for invalid in (123, False, None, [], {}):
                with self.subTest(key=key, invalid=invalid):
                    cfg = self.official_cfg()
                    cfg.pop("app_secret_env")
                    cfg[key] = invalid
                    with self.assertRaises(fastapi.HTTPException) as caught:
                        api_channel._validate_channel_config(
                            "qq", cfg, allow_legacy_inline=(key == "app_secret"))
                    self.assertEqual(caught.exception.status_code, 400)
                    self.assertIn(key, str(caught.exception.detail))

    async def test_official_add_bad_env_type_returns_400_without_write(self):
        cfg = self.official_cfg()
        cfg["app_secret_env"] = 123
        with mock.patch.object(api_channel, "add_channel_config") as save:
            with self.assertRaises(fastapi.HTTPException) as caught:
                await api_channel.add_channel(api_channel.AddChannelReq(
                    id="qq_official", platform="qq", config=cfg))
        self.assertEqual(caught.exception.status_code, 400)
        save.assert_not_called()

    async def test_official_update_bad_env_type_returns_400_without_write(self):
        current = {"id": "qq_official", "platform": "qq",
                   "config": self.official_cfg()}
        for invalid in (123, False, None, [], {}):
            with self.subTest(invalid=invalid):
                with (
                    mock.patch.object(api_channel, "get_channel_config",
                                      return_value=current),
                    mock.patch.object(api_channel, "update_channel_config") as save,
                    mock.patch.object(api_channel.manager, "restart_adapter",
                                      new_callable=mock.AsyncMock) as restart,
                ):
                    with self.assertRaises(fastapi.HTTPException) as caught:
                        await api_channel.update_channel(
                            "qq_official", api_channel.UpdateChannelReq(
                                config={"app_secret_env": invalid}))
                self.assertEqual(caught.exception.status_code, 400)
                save.assert_not_called()
                restart.assert_not_awaited()

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
