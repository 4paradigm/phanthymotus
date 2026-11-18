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
