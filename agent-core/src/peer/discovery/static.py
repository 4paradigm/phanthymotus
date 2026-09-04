"""
peer/discovery/static.py — hand-entered peer addresses.

The escape hatch. mDNS is blocked on many corporate networks and DDS needs a
shared domain, so there has to be a path that depends on nothing but someone
typing an address. Keep it working.

Unlike the automatic providers this one has no key material and often no
peer_id: the operator knows a URL, not a fingerprint. It emits an advert with a
provisional `peer_id` of `static:<url>`; the pairing flow replaces that with the
real fingerprint once it has talked to the endpoint.
"""

import asyncio
import time

import config
from peer.discovery.base import DiscoveryProvider, PeerAdvert


PROVISIONAL_PREFIX = 'static:'

# Re-emit on this interval. The registry drops any advert unseen for
# STALE_AFTER_S (300s), and a configured address is not a sighting that can go
# away — but emitting once at start() meant it aged out five minutes later and
# only came back if discovery was restarted. Observed on a real machine: the
# address was still listed in settings, the `static` provider badge was still
# green, and "发现到的机器人" was empty. MdnsProvider already carries the same
# loop for the same reason (see its _refresh_loop docstring).
REFRESH_INTERVAL_S = 60


def is_provisional(peer_id: str) -> bool:
    """True for an advert that carries a URL but not yet a verified identity."""
    return peer_id.startswith(PROVISIONAL_PREFIX)


class StaticProvider(DiscoveryProvider):
    name = 'static'

    _refresh_task = None

    async def start(self) -> None:
        self._running = True
        # Tolerated the same way the loop tolerates it: a bad first round should not
        # prevent discovery from starting at all, since the next round recovers on
        # its own. registry.start() raising here would take mDNS down with it.
        try:
            self.refresh()
        except Exception as e:
            print(f'[peer] static first refresh failed: {type(e).__name__}: {e}')
        self._refresh_task = asyncio.ensure_future(self._refresh_loop())

    async def stop(self) -> None:
        self._running = False
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None

    async def _refresh_loop(self) -> None:
        """Keep the configured addresses from ageing out of the registry."""
        while self._running:
            try:
                await asyncio.sleep(REFRESH_INTERVAL_S)
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            try:
                self.refresh()
            except Exception as e:      # 一轮失败不该让静态发现就此停摆
                print(f'[peer] static refresh failed: {type(e).__name__}: {e}')

    def refresh(self) -> None:
        """Re-read the configured list and emit an advert per entry."""
        settings = config.main.get('peer_settings', {})
        entries = (settings.get('discovery') or {}).get('static') or []
        now = time.time()
        for entry in entries:
            if isinstance(entry, str):
                url, name, pid = entry, '', ''
            elif isinstance(entry, dict):
                url = entry.get('url', '')
                name = entry.get('display_name', '')
                pid = entry.get('peer_id', '')
            else:
                continue
            if not url:
                continue
            self._on_advert(PeerAdvert(
                peer_id=pid or f'{PROVISIONAL_PREFIX}{url}',
                display_name=name,
                endpoints=[url],
                source=self.name,
                last_seen=now,
            ))
