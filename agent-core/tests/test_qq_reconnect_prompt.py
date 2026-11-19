"""QQ channel recovery and shared robot prompt regression tests."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from channel import chat_only, qq_canvas
from channel import manager as manager_module
from channel.manager import ChannelManager
import prompt


def test_qq_watchdog_keeps_retrying_after_long_outage(monkeypatch):
    module = importlib.import_module("channel.manager")
    clock = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module, "_update_status", lambda *_: None)

    manager = ChannelManager()
    adapter = SimpleNamespace(
        reconnecting=False,
        health_check=AsyncMock(return_value=(False, "network offline")),
    )
    manager._adapters["qq_test"] = adapter
    manager._push_error = AsyncMock()
    network = {"online": False}

    async def reconnect(channel_id, retries=1):
        assert channel_id == "qq_test" and retries == 1
        if network["online"]:
            adapter.health_check.return_value = (True, "")

    manager.restart_adapter = AsyncMock(side_effect=reconnect)

    async def exercise():
        # More than half an hour of simulated outage; no real-time sleep.
        for _ in range(7):
            clock[0] += 301
            await manager._check_one("qq_test")
        assert manager._retry_backoff["qq_test"] == 300.0
        assert manager.restart_adapter.await_count == 7

        # The watchdog retries again after network recovery and actually reconnects.
        network["online"] = True
        clock[0] += 301
        await manager._check_one("qq_test")
        assert manager.restart_adapter.await_count == 8
        await manager._check_one("qq_test")
        assert "qq_test" not in manager._retry_backoff
        assert "qq_test" not in manager._retry_at

    asyncio.run(exercise())


def test_qq_canvas_not_ready_logs_once(monkeypatch, capsys):
    manager = ChannelManager()
    manager._active_input_channels.add("qq_test")
    monkeypatch.setattr(manager, "sync_from_canvas", lambda: None)
    monkeypatch.setattr(qq_canvas, "input_ready", lambda _: False)
    monkeypatch.setattr(manager_module, "get_channel_config",
                        lambda _: {"id": "qq_test", "platform": "qq", "chat_only": True})
    event = SimpleNamespace(channel_id="qq_test", platform="qq")

    async def exercise():
        await manager._on_inbound_message(event)
        await manager._on_inbound_message(event)

    asyncio.run(exercise())
    output = capsys.readouterr().out
    assert output.count("Canvas input is not ready") == 1


def test_shared_prompt_contains_dynamic_context_and_qq_reply_contract(monkeypatch):
    monkeypatch.setattr(prompt, "_env_dynamic", lambda: "status=ready")
    system_message = {"role": "system", "content": "Unified robot system"}
    request = prompt.build(
        system_msg=system_message,
        message_list=[],
        trigger_event={"source": "collector", "text": "hello"},
    )
    assert request[0] is system_message
    assert "status=ready" in request[-1]["content"]
    assert not hasattr(chat_only, "SYSTEM")
    assert "source_message_id" in chat_only.REPLY_SCHEMA["description"]
    assert chat_only.REPLY_SCHEMA["parameters"]["additionalProperties"] is False


def test_chat_only_is_a_channel_policy_not_a_platform_one():
    assert chat_only.enabled({"platform": "feishu", "chat_only": True})
    assert not chat_only.enabled({"platform": "qq"})
    assert not chat_only.enabled(None)
    assert not hasattr(chat_only, "PLATFORMS")
    assert "QQ" not in chat_only.REPLY_SCHEMA["description"]
