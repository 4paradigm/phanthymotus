"""QQ Official rejects new inline credentials and migrates legacy stored config."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import fastapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api import channel as api

def cfg(**kwargs):
    return {"protocol": "official", "private_only": True,
            "app_id": "1900000000", **kwargs}

class QQCredentialMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_inline_credential_rejected_without_persistence(self):
        with mock.patch.object(api, "add_channel_config") as save:
            with self.assertRaises(fastapi.HTTPException) as exc:
                await api.add_channel(api.AddChannelReq(
                    id="qq", platform="qq", config=cfg(app_secret="synthetic-only")))
            self.assertEqual(exc.exception.status_code, 400)
            save.assert_not_called()

    async def test_env_and_file_references_are_accepted(self):
        for reference in (
            {"app_secret_env": "QQ_OFFICIAL_TEST"},
            {"app_secret_file": "/run/qq-official-app-secret"},
        ):
            item = {"id": "qq", "platform": "qq", "config": cfg(**reference)}
            with mock.patch.object(api, "add_channel_config", return_value=item) as save:
                await api.add_channel(api.AddChannelReq(
                    id="qq", platform="qq", config=item["config"]))
                self.assertNotIn("app_secret", save.call_args.args[2])

    async def test_update_new_inline_credential_rejected(self):
        item = {"id": "qq", "platform": "qq",
                "config": cfg(app_secret_env="QQ_OFFICIAL_TEST")}
        with (mock.patch.object(api, "get_channel_config", return_value=item),
              mock.patch.object(api, "update_channel_config") as save):
            with self.assertRaises(fastapi.HTTPException) as exc:
                await api.update_channel("qq", api.UpdateChannelReq(
                    config={"app_secret": "synthetic-only"}))
            self.assertEqual(exc.exception.status_code, 400)
            save.assert_not_called()

    async def test_existing_inline_credential_still_works_on_role_update(self):
        old = {"id": "qq", "platform": "qq",
               "config": cfg(app_secret="synthetic-only")}
        with (mock.patch.object(api, "get_channel_config", return_value=old),
              mock.patch.object(api, "update_channel_config", return_value=old) as save,
              mock.patch.object(api.manager, "restart_adapter",
                                new_callable=mock.AsyncMock)):
            result = await api.update_channel("qq", api.UpdateChannelReq(enabled=False))
            self.assertEqual(result["channel"]["config"]["app_secret"], "***")
            self.assertNotIn("config", save.call_args.kwargs)

    async def test_existing_inline_config_requires_migration_on_edit(self):
        old = {"id": "qq", "platform": "qq",
               "config": cfg(app_secret="synthetic-only")}
        with (mock.patch.object(api, "get_channel_config", return_value=old),
              mock.patch.object(api, "update_channel_config") as save):
            with self.assertRaises(fastapi.HTTPException):
                await api.update_channel("qq", api.UpdateChannelReq(
                    config={"app_id": "1900000001"}))
            save.assert_not_called()

    async def test_inline_to_environment_migration_removes_old_field(self):
        old = {"id": "qq", "platform": "qq",
               "config": cfg(app_secret="synthetic-only")}
        with (mock.patch.object(api, "get_channel_config", return_value=old),
              mock.patch.object(api, "update_channel_config",
                                side_effect=lambda channel_id, **updates: {
                                    "id": channel_id, "platform": "qq",
                                    "config": updates["config"]}) as save,
              mock.patch.object(api.manager, "restart_adapter",
                                new_callable=mock.AsyncMock)):
            await api.update_channel("qq", api.UpdateChannelReq(
                config={"app_secret_env": "QQ_OFFICIAL_TEST"}))
            persisted = save.call_args.kwargs["config"]
            self.assertNotIn("app_secret", persisted)
            self.assertEqual(persisted["app_secret_env"], "QQ_OFFICIAL_TEST")
