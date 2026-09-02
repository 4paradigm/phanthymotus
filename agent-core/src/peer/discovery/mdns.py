"""
peer/discovery/mdns.py — zero-config discovery on the local network.

Service type `_motus._tcp.local`. The TXT record carries what the registry needs
to merge and rank a sighting without opening a connection: the fingerprint, the
full public key, a capability summary and a version.

Two things worth knowing before relying on this:

  * Plenty of corporate and guest networks drop multicast, and Wi-Fi client
    isolation drops it between clients on the same AP. When that happens this
    provider comes up "running" and simply never sees anything — which is why
    DDS presence and the static list exist, and why the registry never treats
    "mDNS found nothing" as "there are no peers".
  * `zeroconf` is an optional dependency, imported inside start() exactly like
    the channel adapters import their SDKs. A missing package must degrade this
    one provider, not stop the Agent Core from booting.

TXT values are attacker-controlled: anyone on the LAN can advertise anything,
including a key that does not match the fingerprint it claims. Nothing here is
trusted. The advert only ever gets a peer as far as the pairing screen, where a
human compares a short code derived from both real keys.
"""

import asyncio
import socket
import time

import config
from peer.discovery.base import DiscoveryProvider, PeerAdvert
from peer import identity


SERVICE_TYPE = '_motus._tcp.local.'
_MAX_TXT_VALUE = 255  # DNS-SD limit per key/value pair


def _txt_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return str(value or '')


def advert_from_txt(props: dict, addresses: list[str], port: int) -> PeerAdvert | None:
    """Build an advert from a TXT record, or None if it is not one of ours.

    Rejects a record whose public key does not hash to the advertised
    fingerprint. That check costs nothing and removes the entire class of
    "advertise someone else's peer_id with my key" confusion from the registry —
    a peer that reaches the pairing screen at least has an internally
    consistent identity.
    """
    decoded = {_txt_str(k): _txt_str(v) for k, v in (props or {}).items()}
    peer_id = decoded.get('pid', '')
    if not peer_id:
        return None

    public_key = decoded.get('pk', '')
    if public_key:
        import base64
        try:
            raw = base64.b64decode(public_key)
        except (ValueError, TypeError):
            return None
        if len(raw) != 32 or identity.fingerprint(raw) != peer_id:
            return None

    endpoints = [f'https://{a}:{port}' for a in addresses if a]
    caps = [c for c in decoded.get('caps', '').split(',') if c]
    return PeerAdvert(
        peer_id=peer_id,
        display_name=decoded.get('name', ''),
        public_key=public_key,
        endpoints=endpoints,
        capabilities=caps,
        version=decoded.get('ver', ''),
        source='mdns',
        last_seen=time.time(),
    )


class MdnsProvider(DiscoveryProvider):
    name = 'mdns'

    def __init__(self, on_advert, port: int = 15678):
        super().__init__(on_advert)
        self._port = port
        self._zc = None
        self._browser = None
        self._info = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        try:
            from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
        except ImportError as e:
            raise RuntimeError(
                'mDNS discovery unavailable (missing zeroconf). '
                'Install it, or use DDS presence / a static peer list instead.'
            ) from e

        self._loop = asyncio.get_running_loop()
        self._zc = Zeroconf()
        self._register_self(ServiceInfo)
        # zeroconf calls back on its own thread; _emit hops to the loop.
        self._browser = ServiceBrowser(self._zc, SERVICE_TYPE, handlers=[self._on_change])
        self._running = True

    async def stop(self) -> None:
        self._running = False
        if self._zc is not None:
            zc, info = self._zc, self._info
            self._zc, self._browser, self._info = None, None, None

            def _close():
                try:
                    if info is not None:
                        zc.unregister_service(info)
                finally:
                    zc.close()

            # Both calls block on network I/O; keep them off the event loop.
            await asyncio.to_thread(_close)

    # ── advertising ──────────────────────────────────────────────────────────

    def _register_self(self, ServiceInfo) -> None:
        ident = identity.ensure_identity()
        settings = config.main.get('peer_settings', {})
        name = settings.get('display_name') or socket.gethostname()
        caps = ','.join(self._local_capabilities())
        props = {
            b'pid': ident['peer_id'].encode(),
            b'pk': ident['public_key'].encode(),
            b'name': name.encode()[:_MAX_TXT_VALUE],
            b'caps': caps.encode()[:_MAX_TXT_VALUE],
            b'ver': b'1',
        }
        # Instance name must be unique on the link; the fingerprint already is.
        info = ServiceInfo(
            SERVICE_TYPE,
            f'{ident["peer_id"][:12]}.{SERVICE_TYPE}',
            addresses=[socket.inet_aton(self._primary_ip())],
            port=self._port,
            properties=props,
            server=f'motus-{ident["peer_id"][:12]}.local.',
        )
        self._zc.register_service(info)
        self._info = info

    @staticmethod
    def _local_capabilities() -> list[str]:
        """Coarse summary of what this agent can do, for peers to rank us by.

        Derived from registered MCP ids rather than a hand-maintained list, so
        it cannot drift from what is actually plugged in. Best-effort only —
        a peer must still call tools/list to learn anything actionable.
        """
        try:
            import mcp_client
            return sorted({mid for mid, info in mcp_client.registry.items()
                           if info.get('online')})[:12]
        except Exception:
            return []

    @staticmethod
    def _primary_ip() -> str:
        """Address a peer on the LAN could reach us on.

        Uses a UDP connect to pick the interface the default route would use —
        no packet is sent. `gethostname()` resolution is not a substitute: in a
        container it usually yields a loopback or the container-internal
        address, which advertises an endpoint no peer can dial.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
        except OSError:
            return '127.0.0.1'
        finally:
            s.close()

    # ── browsing ─────────────────────────────────────────────────────────────

    def _on_change(self, zeroconf, service_type, name, state_change):
        from zeroconf import ServiceStateChange
        if state_change is ServiceStateChange.Removed:
            return
        info = zeroconf.get_service_info(service_type, name, timeout=1500)
        if info is None:
            return
        addresses = []
        for packed in info.addresses or []:
            try:
                addresses.append(socket.inet_ntoa(packed))
            except OSError:
                continue
        advert = advert_from_txt(info.properties, addresses, info.port or self._port)
        if advert is None or advert.peer_id == identity.peer_id():
            return  # ignore malformed records and our own advert
        self._emit(advert)

    def _emit(self, advert: PeerAdvert) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._on_advert, advert)
