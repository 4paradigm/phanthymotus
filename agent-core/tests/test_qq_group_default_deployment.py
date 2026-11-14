"""QQ group @ works without per-group onboarding after install, upgrade, or recovery."""
import asyncio
from unittest import mock

from api import qq_settings
from channel import qq_groups


APP = "1900000000"
CHANNEL = "qq_official_" + APP


def qq_channel(scopes=None, **overrides):
    cfg = {"protocol": "official", "app_id": APP,
           "enabled_scopes": scopes or ["c2c"], "private_only": True,
           "enable_inbound_media": True,
           "app_secret_runtime": "A" * 43, **overrides}
    return {"id": CHANNEL, "platform": "qq", "enabled": True,
            "config": cfg}


def test_old_private_only_qq_is_upgraded_to_group_by_default():
    before = qq_channel()
    writes = []

    def persist(_id, **changes):
        assert _id == CHANNEL
        writes.append(changes["config"])
        return {**before, "config": changes["config"]}

    with mock.patch("channel.manager.update_channel_config", side_effect=persist):
        after = qq_groups.ensure_default_group_scope(before)
        assert after["config"]["enabled_scopes"] == ["c2c", "group"]
        assert after["config"]["private_only"] is False
        assert after["config"]["enable_inbound_media"] is True
        assert len(writes) == 1
        # Repeated upgrade is a no-op; no duplicate scopes.
        assert qq_groups.ensure_default_group_scope(after) is after
        assert len(writes) == 1


def test_explicit_global_group_opt_out_is_not_overridden():
    old = qq_channel(qq_group_scope_opt_out=True)
    with mock.patch("channel.manager.update_channel_config") as write:
        assert qq_groups.ensure_default_group_scope(old) is old
    write.assert_not_called()


def test_unrelated_channels_and_existing_group_configs_are_untouched():
    others = [
        {"id": "f", "platform": "feishu", "config": {}},
        qq_channel(scopes=["c2c", "group"], private_only=False),
    ]
    with mock.patch("channel.manager.update_channel_config") as write:
        for channel in others:
            assert qq_groups.ensure_default_group_scope(channel) is channel
    write.assert_not_called()


def test_standard_settings_default_to_group_without_touching_credential():
    original = qq_channel()
    original["config"]["app_secret_runtime"] = "A" * 43
    saved = {"value": original}
    calls = []

    def get_channel(_id):
        return saved["value"]

    def persist(_id, **kwargs):
        calls.append(kwargs["config"])
        saved["value"] = {**saved["value"], "config": kwargs["config"]}
        return saved["value"]

    async def restart(_id, retries=1):
        qq_settings.manager._health[_id] = (True, "")

    with (mock.patch("api.qq_settings.auth.is_enabled", return_value=True),
          mock.patch("api.qq_settings.get_channel_config", side_effect=get_channel),
          mock.patch("api.qq_settings.update_channel_config", side_effect=persist),
          mock.patch.object(qq_settings.manager, "restart_adapter",
                            side_effect=restart),
          mock.patch.object(qq_settings.manager, "_health", {}),
          mock.patch("api.qq_settings._credential_available", return_value=True)):
        result = asyncio.run(qq_settings.save_settings(
            CHANNEL, qq_settings.QQSettingsChange(mode="standard")))

    assert result["connected"] is True
    assert result["group_enabled"] is True
    assert saved["value"]["config"]["enabled_scopes"] == ["c2c", "group"]
    assert saved["value"]["config"]["private_only"] is False
    assert saved["value"]["config"]["app_secret_runtime"] == "A" * 43
    assert len(calls) == 1

