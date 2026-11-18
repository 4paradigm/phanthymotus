"""QQ settings must not perform synchronous config I/O on the event loop."""

import asyncio
import threading
from unittest.mock import AsyncMock

from api import channel as channel_api


def test_group_config_io_runs_in_worker(monkeypatch):
    event_loop_thread = threading.get_ident()
    operations = []

    def read_channel(_):
        assert threading.get_ident() != event_loop_thread
        operations.append("read")
        return {"platform": "qq", "config": {"group_enabled": False}}

    def write_channel(*args, **kwargs):
        assert threading.get_ident() != event_loop_thread
        operations.append("write")

    monkeypatch.setattr(channel_api, "get_channel_config", read_channel)
    monkeypatch.setattr(channel_api, "update_channel_config", write_channel)
    monkeypatch.setattr(channel_api.manager, "restart_adapter", AsyncMock())

    async def run():
        result = await channel_api.qq_group(
            "qq_test", channel_api.QQGroupSetting(group_enabled=True))
        assert result["group_enabled"] is True

    asyncio.run(run())
    assert operations == ["read", "write"]


def test_qq_delete_config_and_credential_io_use_worker(monkeypatch):
    from channel import qq_credentials

    event_loop_thread = threading.get_ident()
    operations = []

    def read_channel(channel_id):
        assert threading.get_ident() != event_loop_thread
        assert channel_id == "qq_test"
        operations.append("read")
        return {"platform": "qq", "config": {"app_id": "12345678"}}

    async def stop_adapter():
        operations.append("stop")

    def delete_config(channel_id):
        assert threading.get_ident() != event_loop_thread
        assert channel_id == "qq_test"
        operations.append("delete")
        return True

    def remove_credential(app_id):
        assert threading.get_ident() != event_loop_thread
        assert app_id == "12345678"
        operations.append("credential")

    adapter = AsyncMock()
    adapter.stop.side_effect = stop_adapter
    monkeypatch.setattr(channel_api.manager, "_adapters", {"qq_test": adapter})
    monkeypatch.setattr(channel_api, "_qq_idle", lambda _: operations.append("idle"))
    monkeypatch.setattr(channel_api, "get_channel_config", read_channel)
    monkeypatch.setattr(channel_api, "delete_channel_config", delete_config)
    monkeypatch.setattr(qq_credentials, "remove_secret", remove_credential)

    async def run():
        result = await channel_api.delete_channel("qq_test")
        assert result == {"deleted": "qq_test"}

    asyncio.run(run())
    assert operations == ["read", "idle", "stop", "delete", "credential"]
    assert "qq_test" not in channel_api.manager._adapters
