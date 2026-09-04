"""
peer/registry.py — the merged view of who is out there.

Providers each report sightings; this collapses them by `peer_id`. That merge is
the whole reason discovery is keyed on a public-key fingerprint: the same robot
seen over mDNS on the LAN and over a cloud roster from another site is *one*
peer with *two* links, so when the LAN drops the other link is already known and
no re-pairing is needed.

Kept separate from peer/store.py on purpose. This is "seen recently", which is
volatile and untrusted; the store is "paired with", which is durable and
trusted. Conflating them would make an advert able to change a peer's role.
"""

import asyncio
import time

import config
from peer import identity, store
from peer.discovery.base import DiscoveryProvider, PeerAdvert


# A sighting older than this is dropped from the live view. Long enough to
# survive a provider that only refreshes on a slow poll (cloud roster), short
# enough that a powered-off robot stops looking present.
STALE_AFTER_S = 300


class PeerRegistry:
    def __init__(self):
        self._adverts: dict[str, PeerAdvert] = {}
        self._providers: list[DiscoveryProvider] = []
        self._provider_errors: dict[str, str] = {}
        self._lock = asyncio.Lock()

    # ── merge ────────────────────────────────────────────────────────────────

    def observe(self, advert: PeerAdvert) -> PeerAdvert:
        """Fold one sighting into the live view; returns the merged record.

        Synchronous and lock-free by design: providers call this from callbacks
        (zeroconf's own thread hops here via call_soon_threadsafe) and dict
        mutation under the GIL is enough for this shape of update.
        """
        if advert.peer_id == identity.peer_id():
            return advert  # never list ourselves as a peer

        existing = self._adverts.get(advert.peer_id)
        if existing is None:
            merged = PeerAdvert(**advert.to_dict())
            merged.last_seen = advert.last_seen or time.time()
            self._adverts[advert.peer_id] = merged
            return merged

        # Union the links rather than replacing: two providers reporting two
        # reachable addresses means the peer has two, not that the later one is
        # correct. Order is preserved so the first-discovered link is tried first.
        for ep in advert.endpoints:
            if ep not in existing.endpoints:
                existing.endpoints.append(ep)
        for cap in advert.capabilities:
            if cap not in existing.capabilities:
                existing.capabilities.append(cap)

        # Later, non-empty scalars win — a provider that carries less detail
        # (BLE has no room for a key) must not blank out what another supplied.
        if advert.display_name:
            existing.display_name = advert.display_name
        if advert.public_key:
            existing.public_key = advert.public_key
        if advert.version:
            existing.version = advert.version
        if advert.rssi is not None:
            existing.rssi = advert.rssi
        if advert.source and advert.source not in existing.source.split(','):
            existing.source = f'{existing.source},{advert.source}' if existing.source else advert.source
        existing.last_seen = max(existing.last_seen, advert.last_seen or time.time())
        return existing

    def sources_for(self, peer_id: str) -> list[str]:
        advert = self._adverts.get(peer_id)
        return [s for s in (advert.source.split(',') if advert else []) if s]

    # ── query ────────────────────────────────────────────────────────────────

    def discovered(self, include_paired: bool = True) -> list[dict]:
        """Live view, freshest first, annotated with pairing state."""
        self.prune()
        out = []
        for advert in self._adverts.values():
            paired = store.get(advert.peer_id)
            if paired and not include_paired:
                continue
            item = advert.to_dict()
            item['sources'] = self.sources_for(advert.peer_id)
            item['paired'] = paired is not None
            item['role'] = paired['role'] if paired else None
            out.append(item)
        return sorted(out, key=lambda i: i['last_seen'], reverse=True)

    def get(self, peer_id: str) -> PeerAdvert | None:
        return self._adverts.get(peer_id)

    def endpoints_for(self, peer_id: str) -> list[str]:
        """Every known way to reach a peer: what it advertises now, plus what it
        advertised when we paired. The stored ones come last but are what make a
        paired peer reachable after discovery goes dark."""
        seen, out = set(), []
        advert = self._adverts.get(peer_id)
        for ep in (advert.endpoints if advert else []):
            if ep not in seen:
                seen.add(ep)
                out.append(ep)
        paired = store.get(peer_id)
        for ep in (paired['endpoints'] if paired else []):
            if ep not in seen:
                seen.add(ep)
                out.append(ep)
        return out

    def forget(self, peer_id: str) -> None:
        """Drop one advert.

        Used when a provisional (`static:<url>`) advert is replaced by the real
        fingerprint at pairing time: leaving both would show the same machine
        twice, one of them un-pairable.
        """
        self._adverts.pop(peer_id, None)

    def prune(self) -> None:
        cutoff = time.time() - STALE_AFTER_S
        for pid in [p for p, a in self._adverts.items() if a.last_seen < cutoff]:
            self._adverts.pop(pid, None)

    def provider_status(self) -> list[dict]:
        return [
            {
                'name': p.name,
                'running': p.is_running,
                'error': self._provider_errors.get(p.name, ''),
            }
            for p in self._providers
        ]

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bring up whichever providers are configured and reachable.

        One provider failing is expected (blocked multicast, missing zeroconf)
        and must not stop the others — the error is recorded and surfaced in
        provider_status() so the dashboard can say *why* nothing was found,
        rather than silently showing an empty list.
        """
        settings = config.main.get('peer_settings', {})
        if not settings.get('enabled'):
            return
        identity.ensure_identity()

        disc = settings.get('discovery') or {}
        self._providers = []

        if disc.get('mdns', True):
            from peer.discovery.mdns import MdnsProvider
            self._providers.append(MdnsProvider(self.observe))
        if disc.get('static'):
            from peer.discovery.static import StaticProvider
            self._providers.append(StaticProvider(self.observe))

        for provider in self._providers:
            try:
                await provider.start()
                self._provider_errors.pop(provider.name, None)
                print(f'[peer] discovery provider "{provider.name}" started')
            except Exception as e:
                # Always include the exception type: several failures here
                # (zeroconf's EventLoopBlocked among them) carry an empty
                # str(), which logged as 'unavailable: ' and said nothing.
                detail = f'{type(e).__name__}: {e}' if str(e) else type(e).__name__
                self._provider_errors[provider.name] = detail
                print(f'[peer] discovery provider "{provider.name}" unavailable: {detail}')

    async def stop(self) -> None:
        for provider in self._providers:
            try:
                await provider.stop()
            except Exception as e:
                print(f'[peer] provider "{provider.name}" stop failed: {e}')
        self._providers = []

    def reset(self) -> None:
        """Drop the live view. For tests."""
        self._adverts.clear()
        self._provider_errors.clear()
        self._providers = []


registry = PeerRegistry()
