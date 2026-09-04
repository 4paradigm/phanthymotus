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

# Must stay comfortably below registry.STALE_AFTER_S (300s), or a peer that is
# perfectly healthy ages out of the live view between refreshes.
REFRESH_INTERVAL_S = 60


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
        self._aiozc = None
        self._browser = None
        self._info = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resolving: set = set()
        self._seen_names: set = set()
        self._refresh_task: asyncio.Task | None = None

    async def start(self) -> None:
        # The *async* API is mandatory here, not a preference. The synchronous
        # Zeroconf class dispatches its work onto an internal event loop and
        # blocks waiting for it; called from inside a running loop — which is
        # where this provider lives — it raises EventLoopBlocked and the
        # provider never comes up.
        try:
            from zeroconf import ServiceInfo
            from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf
        except ImportError as e:
            raise RuntimeError(
                'mDNS discovery unavailable (missing zeroconf). '
                'Install it, or use DDS presence / a static peer list instead.'
            ) from e

        self._loop = asyncio.get_running_loop()
        self._aiozc = AsyncZeroconf()
        await self._register_self(ServiceInfo)
        self._browser = AsyncServiceBrowser(
            self._aiozc.zeroconf, SERVICE_TYPE, handlers=[self._on_change]
        )
        self._running = True
        self._refresh_task = asyncio.ensure_future(self._refresh_loop())

    async def stop(self) -> None:
        self._running = False
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        self._seen_names.clear()
        aiozc, browser, info = self._aiozc, self._browser, self._info
        self._aiozc, self._browser, self._info = None, None, None
        if browser is not None:
            await browser.async_cancel()
        if aiozc is not None:
            if info is not None:
                await aiozc.async_unregister_service(info)
            await aiozc.async_close()

    # ── advertising ──────────────────────────────────────────────────────────

    async def _register_self(self, ServiceInfo) -> None:
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
        # allow_name_change: the instance name is cosmetic — identity comes from
        # the TXT 'pid' record, which advert_from_txt verifies against the
        # public key. Without this, a stale registration left by a restart (or a
        # second Agent Core on the same host) raises NonUniqueNameException and
        # kills discovery until the old record ages out.
        await self._aiozc.async_register_service(info, allow_name_change=True)
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
        """Browser callback. Runs *on the event loop* with AsyncServiceBrowser.

        Resolving a service is a network round-trip, so it must not happen
        here — the synchronous get_service_info() would stall the loop that
        zeroconf itself needs to receive the reply, and deadlock until timeout.
        Hand it to a task instead.
        """
        from zeroconf import ServiceStateChange
        if state_change is ServiceStateChange.Removed:
            # A peer that said goodbye should disappear now, not linger until the
            # registry's staleness timeout.
            self._seen_names.discard(name)
            return
        self._seen_names.add(name)
        if name in self._resolving:
            return  # a resolve for this service is already in flight
        self._resolving.add(name)
        asyncio.ensure_future(self._resolve(service_type, name))

    async def _refresh_loop(self) -> None:
        """Re-resolve known services periodically to keep their adverts fresh.

        Without this, discovery silently empties a few minutes after startup.
        The registry drops adverts unseen for STALE_AFTER_S (300s), but zeroconf
        only re-announces a stable service on its record TTL, which is far
        longer — so a peer that is up, reachable and unchanged ages out and
        never comes back. Every earlier two-machine test happened to run inside
        that window, which is why it looked fine.

        Re-resolving is a link-local multicast query; at this interval the cost
        is negligible, and a peer that has genuinely gone still ages out because
        the resolve fails and last_seen stops advancing.
        """
        while self._running:
            try:
                await asyncio.sleep(REFRESH_INTERVAL_S)
                if not self._running:
                    return
                for name in list(self._seen_names):
                    if name in self._resolving:
                        continue
                    self._resolving.add(name)
                    await self._resolve(SERVICE_TYPE, name)
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f'[peer] mdns refresh failed: {type(e).__name__}: {e}')

    async def _resolve(self, service_type: str, name: str) -> None:
        from zeroconf.asyncio import AsyncServiceInfo
        try:
            aiozc = self._aiozc
            if aiozc is None:
                return
            info = AsyncServiceInfo(service_type, name)
            if not await info.async_request(aiozc.zeroconf, 3000):
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
            self._on_advert(advert)
        except Exception as e:
            print(f'[peer] mdns resolve failed for {name}: {type(e).__name__}: {e}')
        finally:
            self._resolving.discard(name)
