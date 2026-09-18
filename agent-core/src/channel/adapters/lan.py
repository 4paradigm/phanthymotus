"""
channel/adapters/lan.py — peer-to-peer messaging over the local network.

The design decision worth understanding: LAN A2A is implemented as a
**ChannelAdapter**, not as a parallel messaging stack. `channel/` already solved
the hard parts — the unified InboundMessage/OutboundMessage shape, ACL roles,
`expect_reply` loop guarding, the proactive-mention rate limit, collector
batching by trust class, and canvas `channel_request` routing. Reusing it means
Feishu and LAN present *identical* semantics to the agent, so:

  * not a line of prompt changes;
  * "internet when there is one, LAN when there isn't" is two configured
    channels, not a fallback mechanism anyone had to write.

Unlike every other adapter this one opens no socket. Inbound messages arrive at
`POST /api/peer/inbox/message` on the Agent Core's existing FastAPI server
(api/peer.py), which verifies the Ed25519 signature and calls `deliver()` here.
That is why the README can promise peer collaboration adds no new port.

Trust does **not** come from the channel config: `USES_TRUSTED_BOTS` is False.
A peer is trusted because it is in the `peers` table with a pinned public key,
put there by a human comparing a short code. There is no per-channel allowlist
to get out of sync with that.
"""

import asyncio
import time

from channel.adapter import (
    ChannelAdapter, InboundMessage, OnMessageCallback, OutboundMessage,
)
from peer import store, transport
# Avoid circular import: registry is imported at call sites rather than module load


INBOX_MESSAGE_PATH = '/api/peer/inbox/message'
INBOX_PING_PATH = '/api/peer/inbox/ping'

# channel_id → adapter, so api/peer.py can route a verified inbound request to
# the right channel. Module-level because the HTTP handler has no other handle
# on the adapter instance.
_active: dict[str, 'LanAdapter'] = {}


def active_adapters() -> list['LanAdapter']:
    return list(_active.values())


async def deliver(peer_id: str, payload: dict) -> tuple[bool, str]:
    """Hand a signature-verified peer message to the channel stack.

    Called from api/peer.py. The signature is already checked there; this only
    decides which channel should own the message.
    """
    adapters = active_adapters()
    if not adapters:
        return False, 'no lan channel is running on this agent'

    target = payload.get('channel_id', '')
    adapter = _active.get(target) if target else None
    if adapter is None:
        # A peer does not necessarily know our channel ids. One running lan
        # channel is the common case, so route there rather than reject; with
        # several, requiring channel_id beats guessing.
        if len(adapters) == 1:
            adapter = adapters[0]
        else:
            return False, (f'channel_id required: this agent runs '
                           f'{len(adapters)} lan channels ({", ".join(_active)})')
    return await adapter.accept(peer_id, payload)


class LanAdapter(ChannelAdapter):
    """Peer messaging over HTTPS + per-request Ed25519 signatures."""

    SUPPORTED_FILE_KINDS = ()  # P1 is text-only; attachments need a transfer path
    SUPPORTS_BOT_TO_BOT = True
    USES_TRUSTED_BOTS = False

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._last_error = ''

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        # Registering into the module map *is* "listening" for this adapter —
        # the HTTP route already exists on the shared server.
        _active[self.channel_id] = self
        self._running = True
        print(f'[channel] lan channel "{self.channel_id}" ready at {INBOX_MESSAGE_PATH}')

    async def stop(self) -> None:
        _active.pop(self.channel_id, None)
        self._running = False

    async def health_check(self) -> tuple[bool, str]:
        """Probe every paired peer, not just our own flag.

        `adapter.py` is explicit that health must be observable rather than
        "start() didn't throw". For a peer channel the meaningful question is
        whether any peer answers: a robot whose peers have all vanished is
        degraded even though nothing here crashed.
        """
        if not self._running:
            return False, 'adapter not running'
        peers = [p for p in store.list_peers() if p['role'] != 'blocked']
        if not peers:
            return True, 'no peers paired'

        from peer.registry import registry
        reachable, failures = 0, []
        for peer in peers:
            ok, reason = await self.ping(peer['peer_id'])
            if ok:
                reachable += 1
            else:
                failures.append(f'{peer["display_name"] or peer["peer_id"][:8]}: {reason}')
        if reachable:
            self._last_error = '; '.join(failures)
            return True, self._last_error
        self._last_error = '; '.join(failures)
        return False, f'no paired peer reachable — {self._last_error}'

    def status(self) -> str:
        if not self._running:
            return 'disconnected'
        # health_check() records partial failures; surface them as degraded
        # rather than a flat "connected" that hides a peer being down.
        return 'degraded' if self._last_error else 'connected'

    async def ping(self, peer_id: str) -> tuple[bool, str]:
        from peer.registry import registry
        endpoints = registry.endpoints_for(peer_id)
        result, err = await transport.post_json(
            endpoints, INBOX_PING_PATH, {'channel_id': self.channel_id}, timeout=5.0
        )
        if result is None:
            return False, err
        store.touch(peer_id)
        return True, ''

    # ── inbound ──────────────────────────────────────────────────────────────

    async def accept(self, peer_id: str, payload: dict) -> tuple[bool, str]:
        """Turn a verified peer payload into an InboundMessage.

        `sender_type='bot'` is what makes the existing manager treat this as A2A:
        it skips human ACL auto-registration and applies the bot trust class that
        collector.py already batches on. Nothing peer-specific is needed there.
        """
        text = (payload.get('text') or '').strip()
        if not text:
            return False, 'empty text'

        peer = store.get(peer_id)
        if peer is None:
            return False, 'unknown peer'
        if peer['role'] == 'blocked':
            return False, 'blocked'

        msg = InboundMessage(
            platform='lan',
            channel_id=self.channel_id,
            user_id=peer_id,
            chat_id=peer_id,  # one conversation per peer
            display_name=peer['display_name'] or peer_id[:12],
            text=text,
            message_id=payload.get('message_id') or f'{peer_id}:{time.time():.6f}',
            sender_type='bot',
            chat_type='p2p',
            expect_reply=bool(payload.get('expect_reply')),
        )
        store.touch(peer_id)
        await self._on_message(msg)
        return True, ''

    # ── outbound ─────────────────────────────────────────────────────────────

    async def send_message(self, msg: OutboundMessage) -> None:
        """Send to the peer identified by `chat_id`.

        Raises on failure, per the adapter contract — the agent must be told the
        message did not land rather than assume it did.
        """
        peer_id = msg.chat_id
        if not peer_id:
            raise ValueError('lan channel requires chat_id = peer_id')
        if msg.files:
            raise ValueError('lan channel does not support attachments yet')

        peer = store.get(peer_id)
        if peer is None:
            raise ValueError(f'unknown peer "{peer_id}" — pair it first (Settings → Peers)')
        if peer['role'] == 'blocked':
            raise ValueError(f'peer "{peer_id}" is blocked')

        from peer.registry import registry
        endpoints = registry.endpoints_for(peer_id)
        if not endpoints:
            raise RuntimeError(
                f'no known endpoint for peer "{peer_id}"; it has not been seen by any '
                f'discovery provider since this agent started'
            )

        payload = {
            'channel_id': self.channel_id,
            'text': msg.text,
            'expect_reply': bool(msg.expect_reply),
            'message_id': f'{self.channel_id}:{time.time():.6f}',
        }
        result, err = await transport.post_json(endpoints, INBOX_MESSAGE_PATH, payload)
        if result is None:
            raise RuntimeError(f'lan send to "{peer_id}" failed: {err}')
        if not result.get('accepted', False):
            raise RuntimeError(
                f'peer "{peer_id}" rejected the message: {result.get("reason", "unknown")}'
            )
        store.touch(peer_id)
