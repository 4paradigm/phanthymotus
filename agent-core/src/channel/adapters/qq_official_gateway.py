"""Optional QQ Gateway guild-message Intent without global SDK monkeypatching.

The pinned official SDK uses a module-global DEFAULT_INTENTS on Identify.
This isolated subclass only overrides that payload when a deployment
explicitly enables plain Guild channel messages.
"""
from __future__ import annotations

import asyncio
from qqbot_agent_sdk import QQWebSocket, OPCode, DEFAULT_INTENTS, Intent


class QQGuildWebSocket(QQWebSocket):
    async def _send_identify(self) -> None:
        token = await asyncio.to_thread(self._cb.get_token)
        packet = {
            "op": OPCode.IDENTIFY,
            "d": {
                "token": f"QQBot {token}",
                "intents": int(DEFAULT_INTENTS | Intent.GUILD_MESSAGES),
                "shard": [0, 1],
                "properties": {
                    "$os": "linux",
                    "$browser": "qqbot-sdk",
                    "$device": "qqbot-sdk",
                },
            },
        }
        await self._send_ws_json(packet, "Identify")
