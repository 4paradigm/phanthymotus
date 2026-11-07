"""Official QQ Channel can be selected and exercised by existing Canvas Cards.

No account or network access: tests the actual internal Channel MCP dispatch
which the Dashboard's channel_request/channel_reply cards call.
"""
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from api import mcp_manage
from channel.manager import ChannelManager


class QQOfficialCanvasTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "canvas_layout": {
                "cards": [
                    {"mcpId": "channel", "toolName": "channel_request", "id": "qq-in"},
                    {"mcpId": "agentcore", "toolName": "decision_core", "id": "core"},
                    {"mcpId": "channel", "toolName": "channel_reply", "id": "qq-out"},
                ],
            },
            "tool_config:channel:channel_request:qq-in": {"channel_id": "qq_official"},
            "tool_config:channel:channel_reply:qq-out": {"channel_id": "qq_official"},
            "channel_configs": [{
                "id": "qq_official", "platform": "qq", "enabled": True,
                "config": {"protocol": "official", "app_id": "1900000000"},
            }],
        }

    async def test_canvas_cards_select_the_same_official_qq_channel(self):
        mgr = ChannelManager()
        with mock.patch("channel.manager.config.main", self.config):
            mgr.sync_from_canvas()
            self.assertEqual(mgr.active_input_channels, {"qq_official"})
            self.assertEqual(mgr.active_output_channels, {"qq_official"})
            channel_id, reason = mgr.resolve_target_channel("qq-out")
        self.assertEqual((channel_id, reason), ("qq_official", ""))

    async def test_mcp_request_card_starts_and_exposes_correct_topic(self):
        with (
            mock.patch("api.mcp_manage.config.main", self.config),
            mock.patch("channel.manager.manager.ensure_connected",
                       new_callable=mock.AsyncMock, return_value=(True, "")) as check,
        ):
            start = await mcp_manage.mcp_call_tool(
                "channel", mcp_manage.MCPCallRequest(
                    tool="channel_request",
                    arguments={"action": "start", "instance_id": "qq-in"}))
            info = await mcp_manage.mcp_call_tool(
                "channel", mcp_manage.MCPCallRequest(
                    tool="channel_request",
                    arguments={"action": "info", "instance_id": "qq-in"}))
        self.assertEqual(start["data"], {"state": "running", "channel": "qq_official"})
        self.assertEqual(info["data"]["topic_out"][0]["topic"],
                         "/channel/request/qq_official")
        check.assert_awaited_once_with("qq_official")

    async def test_mcp_reply_card_routes_source_message_id(self):
        with (
            mock.patch("api.mcp_manage.config.main", self.config),
            mock.patch("channel.manager.manager.ensure_connected",
                       new_callable=mock.AsyncMock, return_value=(True, "")) as check,
            mock.patch("channel.manager.manager.send_reply",
                       new_callable=mock.AsyncMock, return_value="Reply sent") as send,
        ):
            start = await mcp_manage.mcp_call_tool(
                "channel", mcp_manage.MCPCallRequest(
                    tool="channel_reply",
                    arguments={"action": "start", "instance_id": "qq-out"}))
            reply = await mcp_manage.mcp_call_tool(
                "channel", mcp_manage.MCPCallRequest(
                    tool="channel_reply",
                    arguments={"action": "send", "instance_id": "qq-out",
                               "source_message_id": "mid-001", "text": "你好"}))
        self.assertEqual(start["data"]["channel"], "qq_official")
        self.assertEqual(reply["data"]["result"], "Reply sent")
        check.assert_awaited_once_with("qq_official")
        self.assertEqual(send.await_args.kwargs["source_message_id"], "mid-001")
        self.assertEqual(send.await_args.kwargs["instance_id"], "qq-out")
        self.assertEqual(send.await_args.kwargs["text"], "你好")

    async def test_canvas_card_start_reports_disconnected_gateway(self):
        with (
            mock.patch("api.mcp_manage.config.main", self.config),
            mock.patch("channel.manager.manager.ensure_connected",
                       new_callable=mock.AsyncMock,
                       return_value=(False, "official Gateway not connected")),
        ):
            start = await mcp_manage.mcp_call_tool(
                "channel", mcp_manage.MCPCallRequest(
                    tool="channel_request",
                    arguments={"action": "start", "instance_id": "qq-in"}))
        self.assertEqual(start["data"]["state"], "error")
        self.assertIn("Gateway", start["data"]["message"])
