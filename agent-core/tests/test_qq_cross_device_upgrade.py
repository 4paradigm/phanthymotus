"""Regression tests for QQ canvas integration and nonblocking inbound writes."""
import asyncio
import os
import threading
import time
from unittest import mock
import pytest
from channel.qq_default_flow import ensure_default_flow
from channel.manager import ChannelManager
from channel.adapter import InboundMessage

CHANNEL = "qq_official_1900000000"
CONFIG = {"id": CHANNEL, "platform": "qq", "enabled": True,
          "config": {"app_id": "1900000000", "protocol": "official"}}


def test_existing_robot_canvas_augmented_idempotently():
    canvas = {"cards": [
        {"id": "robot-core", "mcpId": "agentcore", "toolName": "decision_core"},
        {"id": "robot-camera", "mcpId": "camera", "toolName": "camera"},
    ], "connections": [{"id": "edge", "fromCardId": "robot-camera",
                        "toCardId": "robot-core"}], "execConnections": []}
    state = {"canvas_layout": canvas}
    with (mock.patch("config.main", state),
          mock.patch("channel.manager.get_channel_config", return_value=CONFIG)):
        assert ensure_default_flow(CHANNEL)["status"] == "extended"
        updated = state["canvas_layout"]
        assert len(updated["cards"]) == 4
        assert len(updated["connections"]) == 2
        assert len(updated["execConnections"]) == 1
        assert updated["cards"][:2] == canvas["cards"]
        assert updated["connections"][0] == canvas["connections"][0]
        assert ensure_default_flow(CHANNEL)["status"] == "existing"
        assert len(state["canvas_layout"]["cards"]) == 4


def test_ambiguous_core_stays_untouched():
    canvas = {"cards": [
        {"id": "a", "mcpId": "agentcore", "toolName": "decision_core"},
        {"id": "b", "mcpId": "agentcore", "toolName": "decision_core"},
    ]}
    state = {"canvas_layout": canvas}
    with (mock.patch("config.main", state),
          mock.patch("channel.manager.get_channel_config", return_value=CONFIG)):
        assert ensure_default_flow(CHANNEL)["status"] == "manual_required"
    assert state["canvas_layout"] is canvas


def test_slow_persistence_does_not_block_event_loop():
    async def check():
        mgr = ChannelManager()
        mgr._active_input_channels.add("qq-main")
        mgr.sync_from_canvas = mock.Mock()
        entered = threading.Event()
    
        def slow_write(*args, **kwargs):
            entered.set()
            time.sleep(.15)
    
        mgr._set_last_context = mock.Mock(side_effect=slow_write)
        mgr._record_chat_exchange = mock.Mock()
        msg = InboundMessage(platform="qq", channel_id="qq-main",
                             user_id="test-user", chat_id="test-user",
                             display_name="Tester", text="hello",
                             message_id="msg-1", chat_type="p2p")
        user = {"role": "viewer", "tool_filter": "chat-only"}
        with (mock.patch("channel.manager._qq_inbound_acl", new_callable=mock.AsyncMock,
                         return_value=user),
              mock.patch("api.inspection.publish_to_topic", new_callable=mock.AsyncMock),
              mock.patch("api.motus_stream.push_event", new_callable=mock.AsyncMock)):
            task = asyncio.create_task(mgr._on_inbound_message(msg))
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.wait_for(asyncio.sleep(0), .04)
            await task
            mgr._record_chat_exchange.assert_called_once()
    asyncio.run(check())


def test_legacy_mount_is_migrated_idempotently():
    from channel import qq_credential_store as store
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import os

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_file = root / "legacy.txt"
        sample = chr(88) * 24
        old_file.write_text(sample, encoding="utf-8")
        os.chmod(old_file, 0o600)
        unavailable = root / "unavailable.txt"
        key = "app_secret_file"
        existing = dict(CONFIG)
        existing["config"] = {**CONFIG["config"], key: str(unavailable)}
        updated = []

        def update(_channel_id, **changes):
            existing["config"] = changes["config"]
            updated.append(changes["config"])
            return existing

        with (mock.patch.object(store, "_LEGACY_MOUNT", str(unavailable)),
              mock.patch.object(store, "_LEGACY_INSTALLED_FILE", old_file),
              mock.patch.object(store, "directory", return_value=root / "persisted"),
              mock.patch("channel.manager.get_channel_config", return_value=existing),
              mock.patch("channel.manager.update_channel_config", side_effect=update)):
            migrated = store.migrate_legacy(existing)
            target = Path(migrated["config"][key])
            assert target.is_file()
            assert store.read("1900000000", str(target)) == sample
            if os.name != "nt":
                assert os.stat(target).st_mode & 0o077 == 0
            assert store.migrate_legacy(existing)["config"][key] == str(target)
            assert len(updated) == 1
            assert old_file.read_text(encoding="utf-8") == sample


@pytest.mark.skipif(os.name == "nt", reason="NTFS does not enforce POSIX 0600 bits")
def test_untrusted_legacy_file_is_not_imported():
    from channel import qq_credential_store as store
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import os

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        file = root / "legacy.txt"
        file.write_text(chr(88) * 24)
        os.chmod(file, 0o644)
        unavailable = root / "unavailable.txt"
        existing = dict(CONFIG)
        existing["config"] = {**CONFIG["config"], "app_secret_file": str(unavailable)}
        with (mock.patch.object(store, "_LEGACY_MOUNT", str(unavailable)),
              mock.patch.object(store, "_LEGACY_INSTALLED_FILE", file),
              mock.patch("channel.manager.get_channel_config", return_value=existing),
              mock.patch("channel.manager.update_channel_config") as save):
            assert store.migrate_legacy(existing) is existing
            save.assert_not_called()


