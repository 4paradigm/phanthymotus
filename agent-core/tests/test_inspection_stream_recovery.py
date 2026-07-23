from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


CORE_ROOT = Path(__file__).resolve().parents[1]


class _FakeRouter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def _decorator(self, *args, **kwargs):
        return lambda function: function

    get = _decorator
    post = _decorator
    websocket = _decorator


class _FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.closed = False
        self.text_messages: list[str] = []
        self.binary_messages: list[bytes] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, payload: str) -> None:
        self.text_messages.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        self.binary_messages.append(payload)

    async def close(self) -> None:
        self.closed = True


class _FakeWebSocketDisconnect(Exception):
    pass


def _load_inspection_module():
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.APIRouter = _FakeRouter
    fastapi_stub.WebSocket = _FakeWebSocket
    fastapi_stub.WebSocketDisconnect = _FakeWebSocketDisconnect
    fastapi_stub.HTTPException = RuntimeError

    bridge_stub = types.ModuleType("ros2_bridge")
    bridge_stub.get_last_seen = mock.Mock(return_value=0.0)
    bridge_stub.get_dds_topics = mock.Mock(return_value=set())
    bridge_stub.subscribe = mock.Mock()
    bridge_stub.publish = mock.Mock()

    module_path = CORE_ROOT / "src/api/inspection.py"
    spec = importlib.util.spec_from_file_location(
        "inspection_stream_recovery_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"fastapi": fastapi_stub, "ros2_bridge": bridge_stub},
    ):
        spec.loader.exec_module(module)
    return module, bridge_stub


class InspectionStreamRecoveryTest(unittest.TestCase):
    def test_force_refresh_replaces_an_existing_primary_subscription(self) -> None:
        inspection, bridge = _load_inspection_module()
        topic = "/camera/dynamic/rgb"
        inspection._active_primary_subs.add(topic)

        self.assertFalse(
            inspection._ensure_primary_sub(topic, "image/jpeg", object())
        )
        bridge.subscribe.assert_not_called()

        self.assertTrue(
            inspection._ensure_primary_sub(
                topic,
                "image/jpeg",
                object(),
                force=True,
            )
        )
        bridge.subscribe.assert_called_once()
        key, actual_topic, actual_format, _, callback = bridge.subscribe.call_args.args
        self.assertEqual(key, f"__primary__#{topic}")
        self.assertEqual(actual_topic, topic)
        self.assertEqual(actual_format, "image/jpeg")
        self.assertTrue(callable(callback))

    def test_recent_frame_detection_rejects_never_seen_and_stale_topics(self) -> None:
        inspection, bridge = _load_inspection_module()
        topic = "/camera/dynamic/rgb"

        bridge.get_last_seen.return_value = 0.0
        self.assertFalse(inspection._has_recent_frame(topic, now=100.0))

        bridge.get_last_seen.return_value = 95.0
        self.assertTrue(inspection._has_recent_frame(topic, now=100.0))

        bridge.get_last_seen.return_value = 80.0
        self.assertFalse(inspection._has_recent_frame(topic, now=100.0))

    def test_stale_image_stream_refreshes_then_returns_explicit_error(self) -> None:
        inspection, bridge = _load_inspection_module()
        topic = "/camera/dynamic/rgb"
        inspection._topic_registry[topic] = {
            "format": "image/jpeg",
            "mcp_id": "camera",
            "registered_at": 1.0,
        }
        inspection._active_primary_subs.add(topic)
        inspection._last_frame[topic] = b"\xff\xd8stale\xff\xd9"
        bridge.get_last_seen.return_value = 0.0
        websocket = _FakeWebSocket()

        async def timeout_without_consuming(awaitable, timeout):
            del timeout
            awaitable.close()
            raise asyncio.TimeoutError

        fake_time = types.SimpleNamespace(
            time=lambda: 1_784_788_800.0,
            monotonic=mock.Mock(side_effect=[100.0, 105.0, 111.0]),
        )
        with (
            mock.patch.object(
                inspection.asyncio,
                "wait_for",
                side_effect=timeout_without_consuming,
            ),
            mock.patch.object(inspection, "time", fake_time),
        ):
            asyncio.run(inspection.bus_ws(websocket, "camera/dynamic/rgb"))

        self.assertTrue(websocket.accepted)
        self.assertTrue(websocket.closed)
        self.assertEqual(websocket.binary_messages, [])
        messages = [json.loads(payload) for payload in websocket.text_messages]
        self.assertEqual(messages[0]["type"], "meta")
        self.assertEqual(messages[-1]["type"], "error")
        self.assertIn("10 秒内未收到 JPEG", messages[-1]["message"])
        bridge.subscribe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
