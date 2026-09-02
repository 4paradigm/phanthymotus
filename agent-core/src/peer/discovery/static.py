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

import time

import config
from peer.discovery.base import DiscoveryProvider, PeerAdvert


PROVISIONAL_PREFIX = 'static:'


def is_provisional(peer_id: str) -> bool:
    """True for an advert that carries a URL but not yet a verified identity."""
    return peer_id.startswith(PROVISIONAL_PREFIX)


class StaticProvider(DiscoveryProvider):
    name = 'static'

    async def start(self) -> None:
        self._running = True
        self.refresh()

    async def stop(self) -> None:
        self._running = False

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
