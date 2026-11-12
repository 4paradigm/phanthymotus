"""QQ Gateway subscription flags: request only configured, authorized events.

QQ sends CLOSE 4014 for any unauthorized Identify intent. The official
SDK 1.2.2 hardcodes a broader DEFAULT_INTENTS mask. We override the
Identify payload per adapter instance, without global monkeypatching.
"""
from __future__ import annotations

import asyncio
from qqbot_agent_sdk import QQWebSocket, OPCode, Intent


def qq_gateway_intents(config: dict) -> int:
    """Derive a minimal mask; C2C and group @ share GROUP_MESSAGES."""
    scopes = set(config.get("enabled_scopes", ["c2c"]))
    mask = 0
    if "c2c" in scopes or "group" in scopes:
        mask |= int(Intent.GROUP_MESSAGES)
    if "guild" in scopes:
        mask |= int(Intent.GUILD_AT_MESSAGE)
    if "guild_dm" in scopes:
        mask |= int(Intent.DIRECT_MESSAGES)
    if config.get("enable_guild_plain_messages", False):
        mask |= int(Intent.GUILD_MESSAGES)
    if config.get("enable_interactions", False):
        mask |= int(Intent.INTERACTION)
    if not mask:
        raise ValueError("QQ Gateway intents cannot be empty")
    return mask


class QQScopedWebSocket(QQWebSocket):
    """SDK Gateway with exact configured Identify bits."""

    def __init__(self, callbacks, *, intents: int, log_tag: str = ""):
        if not isinstance(intents, int) or isinstance(intents, bool) or intents <= 0:
            raise ValueError("Invalid QQ Gateway intents")
        super().__init__(callbacks=callbacks, log_tag=log_tag)
        self._qq_intents = intents

    def _handle_dispatch(self, event_type: str, data: dict) -> None:
        # SDK 1.2.2 does not recognize GROUP_MESSAGE_CREATE, delivered by
        # QQ's "all group messages" mode (including @bot messages).
        # Preserve normal SDK duplicate suppression and dispatch this event
        # to the adapter, where non-mentions are rejected before reaching LLM.
        if event_type == "GROUP_MESSAGE_CREATE":
            msg_id = str(data.get("id", "")) if isinstance(data, dict) else ""
            if msg_id and self._is_duplicate(msg_id):
                return
            print("[qq-group] gateway_event_received type=full", flush=True)
            self._dispatch_to_main(
                self._cb.on_message_event(event_type, data))
            return
        super()._handle_dispatch(event_type, data)

    async def _send_identify(self) -> None:
        token = await asyncio.to_thread(self._cb.get_token)
        packet = {
            "op": OPCode.IDENTIFY,
            "d": {
                "token": f"QQBot {token}",
                "intents": self._qq_intents,
                "shard": [0, 1],
                "properties": {
                    "$os": "linux",
                    "$browser": "qqbot-sdk",
                    "$device": "qqbot-sdk",
                },
            },
        }
        await self._send_ws_json(packet, "Identify")


# Backwards compatibility for older test and tools invoking Guild subclass.
class QQGuildWebSocket(QQScopedWebSocket):
    def __init__(self, callbacks, log_tag: str = ""):
        from qqbot_agent_sdk import DEFAULT_INTENTS
        super().__init__(
            callbacks=callbacks,
            intents=int(DEFAULT_INTENTS | Intent.GUILD_MESSAGES),
            log_tag=log_tag,
        )
