"""Post an ACP completion to Agent Core, off whatever thread noticed.

`tts.py` has its own copy of this (`_complete_action`), posted synchronously with
a 3 s timeout. That is fine there — it runs on a synthesis worker that owes
nobody a deadline. It is not fine for audio: the music card notices a track has
drained on its 100 ms output loop, and a 3 s HTTP call on that thread is 30
missed frames, i.e. an audible 3-second gap in whatever plays next.

So this posts from a throwaway daemon thread and never raises. A completion that
fails to arrive is not silent: agent-core's barrier logs the timeout, and the
pending's own bookkeeping records the outcome.

Not merged with tts.py's copy on purpose — that would mean editing the TTS hot
path in a change about music. Worth doing separately.
"""
from __future__ import annotations

import json
import ipaddress
import logging
import os
import ssl
import threading
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

# Loopback by default because /api/acp/complete is loopback-only (auth.py
# rejects anything else, loudly, and tells the caller to use the peer inbox).
AGENT_CORE_URL = os.environ.get('AGENT_CORE_URL', 'https://localhost:15678')
TIMEOUT_S = 3


def _ssl_context(url: str):
    """Match Core's local self-signed HTTPS without relaxing remote TLS."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https':
        return None
    host = parsed.hostname or ''
    try:
        local = host.lower() == 'localhost' or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    context = ssl.create_default_context()
    if local:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def _post(action_id: str, status: str, result: dict) -> None:
    body = json.dumps({'action_id': action_id, 'status': status,
                       'result': result}, ensure_ascii=False).encode()
    request = urllib.request.Request(
        f'{AGENT_CORE_URL.rstrip("/")}/api/acp/complete', body,
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S,
                                    context=_ssl_context(AGENT_CORE_URL)) as response:
            response.read()
    except Exception as error:   # noqa: BLE001
        # Deliberately not re-raised: the caller is a playback thread, and a
        # robot must not stop playing because a bookkeeping POST failed.
        log.warning('[acp] completion for %s (%s) did not reach agent-core: %s',
                    action_id, status, error)


def complete(action_id: str, status: str = 'completed', result: dict | None = None) -> None:
    """Fire-and-forget. Safe to call from an audio loop or a ROS callback."""
    if not action_id:
        return
    threading.Thread(target=_post, args=(action_id, status, result or {}),
                     name=f'acp-complete-{action_id}', daemon=True).start()
