"""QQ interactive buttons and approvals, bound to exact original actor/chat.

Interaction results only update the local registry; they never execute
robot tools or grant lasting ACL permissions.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import secrets
import time
from typing import Any

_MAX_PENDING = 128
_LIFETIME = 300.0


@dataclass
class Pending:
    kind: str
    scope: str
    chat: str
    actor: str
    deadline: float
    choices: tuple[str, ...]


class QQInteractionRegistry:
    def __init__(self):
        self.pending: OrderedDict[str, Pending] = OrderedDict()
        self.results: OrderedDict[str, str] = OrderedDict()

    def issue(self, kind: str, scope: str, chat: str, actor: str,
              choices: tuple[str, ...]) -> str:
        if scope not in ("c2c", "group") or not chat or not actor:
            raise ValueError("invalid QQ interactive target")
        self.expire()
        nonce = secrets.token_urlsafe(24)
        self.pending[nonce] = Pending(
            kind, scope, chat, actor, time.monotonic() + _LIFETIME, choices)
        while len(self.pending) > _MAX_PENDING:
            self.pending.popitem(last=False)
        return nonce

    def expire(self):
        now = time.monotonic()
        for key, record in list(self.pending.items()):
            if record.deadline <= now:
                self.pending.pop(key, None)

    async def send_approval(self, api: Any, scope: str, chat: str, actor: str,
                            source_id: str, title: str, description: str = "") -> str:
        if not source_id:
            raise ValueError("source message is required")
        from qqbot_agent_sdk import ApprovalSender, ApprovalRequest
        nonce = self.issue(
            "approval", scope, chat, actor,
            ("allow-once", "allow-always", "deny"))
        req = ApprovalRequest(
            session_key=nonce, title=title[:150], description=description[:1000])
        try:
            delivered = await ApprovalSender(api).send(
                scope, chat, req, msg_id=source_id)
        except Exception:
            delivered = False
        if not delivered:
            self.pending.pop(nonce, None)
            raise RuntimeError("QQ approval prompt not delivered")
        return nonce

    async def send_choices(self, api: Any, scope: str, chat: str, actor: str,
                           source_id: str, label: str,
                           choices: dict[str, str]) -> str:
        if not source_id or not choices or len(choices) > 3:
            raise ValueError("QQ button prompt requires source and 1-3 options")
        if any(not name.isascii() or not name.isalnum() or len(name) > 20
               for name in choices):
            raise ValueError("invalid QQ button choice")
        from qqbot_agent_sdk import InlineKeyboard
        from qqbot_agent_sdk.dto import (
            KeyboardButton, KeyboardButtonAction, KeyboardButtonPermission,
            KeyboardButtonRenderData, KeyboardContent, KeyboardRow,
        )
        nonce = self.issue("choice", scope, chat, actor, tuple(choices))
        buttons = [
            KeyboardButton(
                id=f"option{i}",
                render_data=KeyboardButtonRenderData(
                    label=title[:36], visited_label=title[:36], style=1),
                action=KeyboardButtonAction(
                    type=1, data=f"qqchoice:{nonce}:{key}",
                    permission=KeyboardButtonPermission(type=2), click_limit=1),
                group_id="qqchoice",
            ) for i, (key, title) in enumerate(choices.items())
        ]
        keyboard = InlineKeyboard(content=KeyboardContent(
            rows=[KeyboardRow(buttons=buttons)]))
        try:
            body = api.build_text_body(
                label[:1200], reply_to=source_id, markdown=False)
            if scope == "c2c":
                await api.post_c2c_message(chat, body, keyboard=keyboard)
            else:
                await api.post_group_message(chat, body, keyboard=keyboard)
        except Exception:
            self.pending.pop(nonce, None)
            raise RuntimeError("QQ interactive button delivery unconfirmed") from None
        return nonce

    async def handle(self, api: Any, event_type: str, raw: dict) -> bool:
        if event_type != "INTERACTION_CREATE" or not isinstance(raw, dict):
            return False
        from qqbot_agent_sdk import parse_interaction_event, parse_approval_button_data
        event = parse_interaction_event(raw)
        button = event.data.resolved.button_data
        approval = parse_approval_button_data(button)
        if approval is None:
            fields = button.split(":", 2)
            if len(fields) != 3 or fields[0] != "qqchoice":
                return False
            nonce, decision = fields[1:]
            kind = "choice"
        else:
            nonce, decision = approval
            kind = "approval"
        self.expire()
        item = self.pending.get(nonce)
        if not item or item.kind != kind or decision not in item.choices:
            return False
        chat = event.group_openid if item.scope == "group" else event.user_openid
        actor = event.group_member_openid if item.scope == "group" else event.user_openid
        if item.chat != chat or item.actor != actor:
            return False
        await api.acknowledge_interaction(event.id, code=0)
        self.pending.pop(nonce, None)
        # The SDK offers "always"; only one-shot approvals are honored.
        self.results[nonce] = "allow-once" if decision == "allow-always" else decision
        while len(self.results) > _MAX_PENDING:
            self.results.popitem(last=False)
        return True

    def take(self, nonce: str) -> str | None:
        self.expire()
        return self.results.pop(nonce, None)
