"""Chat-only pending messages must be scoped to their channel and chat."""

import asyncio
import json
from unittest.mock import AsyncMock

import collector
import config
import event_bus
from api import inspection, motus_stream
from channel import qq_canvas
from channel.adapter import InboundMessage
from channel.manager import ChannelManager


def test_identical_message_ids_from_different_chats_reach_agent(monkeypatch):
    channels = ("telegram_a", "telegram_b")
    monkeypatch.setattr(config, "main", {
        "core": {"project_running": True},
        "channel_configs": [
            {"id": channel_id, "platform": "telegram", "chat_only": True}
            for channel_id in channels
        ],
    })
    manager = ChannelManager()
    manager._active_input_channels.update(channels)
    monkeypatch.setattr(manager, "sync_from_canvas", lambda: None)
    monkeypatch.setattr("channel.manager.manager", manager)
    monkeypatch.setattr(qq_canvas, "input_ready", lambda channel_id: True)
    monkeypatch.setattr(motus_stream, "push_event", AsyncMock())
    monkeypatch.setattr(event_bus, "_queue", asyncio.Queue())
    monkeypatch.setattr(collector, "_output", asyncio.Queue())
    monkeypatch.setattr(collector, "_busy", False)
    monkeypatch.setattr(collector, "_priority_pending", collector.deque())

    published = []

    async def publish(topic, body):
        published.append((topic, body))
        await event_bus.enqueue(f"dds:{topic}", body)
        await event_bus.enqueue(f"dds:{topic}", body)  # DDS replay

    monkeypatch.setattr(inspection, "publish_to_topic", publish)

    messages = [
        InboundMessage("telegram", "telegram_a", "user1", "group_a",
                       "User 1", "first collision marker", message_id="100"),
        InboundMessage("telegram", "telegram_a", "user2", "group_b",
                       "User 2", "second collision marker", message_id="100"),
        InboundMessage("telegram", "telegram_b", "user3", "group_a",
                       "User 3", "third collision marker", message_id="100"),
    ]

    async def run():
        # Publish all three before Collector consumes any of them: this used
        # to overwrite pending["100"] twice.
        for message in messages:
            await manager._on_inbound_message(message)

        assert len(manager._chat_only_pending) == 3
        assert not manager.consume_chat_input(
            published[0][0], published[0][1], "telegram_a", "group_b", "100"
        )
        assert not manager.consume_chat_input(
            published[0][0], json.dumps({"tampered": True}), "telegram_a", "group_a", "100"
        )

        worker = asyncio.create_task(collector._drain_loop())
        try:
            results = [
                await asyncio.wait_for(collector._output.get(), timeout=2)
                for _ in messages
            ]
            for message, result in zip(messages, results):
                assert result["_chat_only_channel"] == message.channel_id
                assert message.text in result["text"]
                assert result["_channel_message_ids"] == ["100"]
            await asyncio.sleep(0)
            assert collector._output.empty()  # replay cannot trigger again
            assert not manager._chat_only_pending
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(run())
