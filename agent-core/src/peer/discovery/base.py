"""
peer/discovery/base.py — the one shape every discovery provider produces.

Providers differ wildly (multicast, a cloud roster, a BLE advert, a hand-typed
address) but they all answer the same question: "there is a peer, here is its
identity, here is how to reach it". Keeping that answer in one dataclass is what
lets the registry merge a peer seen over three paths into one record.

`peer_id` is an Ed25519 public-key fingerprint. Not an IP, not a platform
account — those move, and a peer that changes identity when it changes network
cannot be paired with once and trusted afterwards.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PeerAdvert:
    """A sighting of a peer. Says nothing about whether we trust it."""

    peer_id: str
    display_name: str = ''
    # Raw 32-byte Ed25519 public key, base64. A provider that can only carry a
    # fingerprint (BLE's advert budget is tiny) leaves this empty; pairing then
    # fetches the full key over the link and checks it hashes to peer_id.
    public_key: str = ''
    endpoints: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    version: str = ''
    source: str = ''       # 'mdns' | 'dds' | 'cloud' | 'ble' | 'static'
    rssi: int | None = None  # BLE only — proximity ordering when pairing
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            'peer_id': self.peer_id,
            'display_name': self.display_name,
            'public_key': self.public_key,
            'endpoints': list(self.endpoints),
            'capabilities': list(self.capabilities),
            'version': self.version,
            'source': self.source,
            'rssi': self.rssi,
            'last_seen': self.last_seen,
        }


OnAdvert = Callable[[PeerAdvert], None]


class DiscoveryProvider(ABC):
    """One way of finding peers.

    Providers are expected to be independently failable: mDNS is blocked on
    plenty of corporate networks, and the static list exists precisely so that
    the platform still works when every automatic path is dead.
    """

    name: str = 'base'

    def __init__(self, on_advert: OnAdvert):
        self._on_advert = on_advert
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    async def start(self) -> None:
        """Begin discovering. Must raise if it cannot — the registry logs the
        reason and carries on with the other providers rather than pretending
        this one is live."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...
