"""
peer/discovery/ble.py — finding a peer with no network at all.

Every other provider needs IP first: mDNS needs a shared link that forwards
multicast, the static list needs a human who already knows an address, a cloud
roster needs the internet. BLE needs none of them. Two robots in the same room
with nothing configured can still exchange the one thing pairing is about — a
public key — because the radio is its own discovery layer.

## What this is not

**BLE is a discovery path, not a transport.** What it carries is a fingerprint,
the full public key, a display name and whatever IP endpoints the peer believes
it has. The SAS pairing handshake that follows still runs over HTTPS, so two
robots that share no IP network can see each other here and still not be able to
pair. That is the honest boundary: BLE removes the need for *discovery*
infrastructure, not the need for *connectivity*.

## Two halves, independently optional

Scanning (central role) uses `bleak`, a declared dependency that works on Linux,
macOS and Windows. Advertising (peripheral role) uses `bluez_peripheral`, which
is Linux-only, pre-1.0 and installed by hand — see peer/ble_advertiser.py. A
robot that can only scan still discovers peers that advertise; a mesh where
nobody advertises simply finds nothing. Both are reported in `last_error` rather
than being silently absent, because "provider is green and finds nothing" is the
failure mode this codebase keeps having to fix.

## Trust

Nothing read here is trusted. A BLE advert is unauthenticated — anyone with a
radio can claim any service UUID and serve any bytes. The one check worth doing
locally is internal consistency, and it is free: `peer_id` is *derived* from the
public key we read rather than read alongside it, so a device cannot advertise
someone else's identity. Everything beyond that is settled at the pairing
screen, where a human compares a code derived from both real keys.

## Verification status (2026-09-04, Orin5 + Orin6)

Verified on real BlueZ 5.53 hardware:

  * the peripheral half registers and **stays** registered — Orin6 kept seeing
    Orin5's advert (correct service UUID, local name `M-Orin5`) for minutes,
    including past t+181s, which is what confirms `timeout=0` means indefinite
    rather than BlueZ's 180s cap;
  * the scanner half finds it through the service-UUID filter, extracts rssi,
    and reports/backs off a failed read instead of killing its loop.

**Not verified on hardware: the GATT characteristic read.** The two rigs sit at
-93..-99 dBm of each other — adverts are one-way broadcasts and get through,
but every connection attempt timed out, and no third central was in range (this
workstation and Tianyi both see plenty of BLE devices, neither sees Orin5).
D-Bus policy blocks reading the characteristics locally: only bluez, as root,
may call into a GATT application. So the decorators below are covered by unit
tests with stubbed GATT, which exercises the parsing and validation but not
BlueZ actually serving the bytes. Closing that needs two robots in one room.
"""

import asyncio
import base64
import json
import socket
import time

import config
from peer import identity
from peer.discovery.base import DiscoveryProvider, PeerAdvert


# Motus peer discovery service. Kept here rather than in the advertiser because
# the scanner is the half that always exists; the advertiser imports these.
SERVICE_UUID = '12345678-1234-5678-1234-56789abcdef0'
CHAR_PUBLIC_KEY = '12345678-1234-5678-1234-56789abcdef1'
CHAR_ENDPOINTS = '12345678-1234-5678-1234-56789abcdef2'
CHAR_NAME = '12345678-1234-5678-1234-56789abcdef3'

# How often a scan round starts. Deliberately far slower than the 10s the first
# draft used: each round may open a GATT connection per device, and connecting
# is both slow and disruptive — a BLE peripheral can usually serve one central
# at a time, so aggressive polling makes two robots fight over each other.
SCAN_INTERVAL_S = 45
SCAN_TIMEOUT_S = 8.0
CONNECT_TIMEOUT_S = 12.0

# Once a device's identity is known, re-emit from cache instead of reconnecting.
# The advert only has to stay fresher than registry.STALE_AFTER_S (300s); a key
# does not change without the peer restarting, at which point its BLE address
# changes too (BlueZ rotates it) and we reconnect anyway.
IDENTITY_TTL_S = 900

# A device advertising our service UUID that fails to serve our characteristics
# is either not one of ours or is busy. Backing off stops one such device from
# consuming every round.
FAILURE_BACKOFF_S = 300


class BleProvider(DiscoveryProvider):
    name = 'ble'

    def __init__(self, on_advert):
        super().__init__(on_advert)
        self._scan_task: asyncio.Task | None = None
        # address → (peer_id, public_key_b64, display_name, endpoints, read_at)
        self._identities: dict[str, tuple] = {}
        self._failed_until: dict[str, float] = {}
        self._advertising = False
        # Two kinds of unhealthy, and they must not overwrite each other.
        # `_notice` is a standing condition (this robot can scan but cannot be
        # found) that a successful scan does not fix; `_scan_error` is transient
        # and clears on the next good round. An earlier version kept one string
        # and told them apart with startswith(), which lost the standing notice
        # the first time a scan succeeded.
        self._notice = ''
        self._scan_error = ''

    @property
    def last_error(self) -> str:
        """Surfaced through registry.provider_status(). A provider whose loop is
        alive but whose every scan raises must not read as healthy."""
        return self._scan_error or self._notice

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        try:
            import bleak  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                'BLE discovery unavailable (missing bleak). Reinstall dependencies, '
                'or use mDNS / a static peer list instead.'
            ) from e

        self._running = True
        self._notice = self._scan_error = ''
        # Advertising runs on *this* loop, which lives as long as the app does.
        # The earlier implementation started it with run_until_complete on a
        # throwaway loop in a worker thread: registration succeeded, the loop
        # then stopped, and the GATT server had nothing to serve reads with —
        # so the robot advertised a service that answered nothing.
        self._advertising = await self._start_advertising()
        self._scan_task = asyncio.ensure_future(self._scan_loop())

    async def stop(self) -> None:
        self._running = False
        if self._scan_task is not None:
            self._scan_task.cancel()
            self._scan_task = None
        self._identities.clear()
        self._failed_until.clear()
        if self._advertising:
            from peer import ble_advertiser
            try:
                await ble_advertiser.stop_advertising()
            except Exception as e:
                print(f'[peer] ble advertiser stop failed: {type(e).__name__}: {e}')
            self._advertising = False

    async def _start_advertising(self) -> bool:
        """Bring up the peripheral half; never fatal.

        A robot that can only scan is still useful — it finds peers, it just
        cannot be found. Reporting that in last_error beats refusing to start.
        """
        from peer import ble_advertiser
        if not ble_advertiser.is_available():
            self._notice = ('scan only — bluez_peripheral not installed, so this robot '
                            'is not discoverable over BLE (pip install bluez-peripheral)')
            print(f'[peer] ble: {self._notice}')
            return False
        try:
            await ble_advertiser.start_advertising()
            return True
        except Exception as e:
            self._notice = f'advertising failed: {type(e).__name__}: {e}'
            print(f'[peer] ble {self._notice}')
            return False

    # ── scanning ─────────────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                return
            except Exception as e:
                # Typical causes are worth distinguishing for the operator, and
                # all of them look identical from Python: adapter soft-blocked by
                # rfkill, bluetoothd not running, no D-Bus socket in the
                # container. The exception text is the only clue there is, so
                # surface it rather than collapsing it to "BLE failed".
                self._scan_error = f'scan failed: {type(e).__name__}: {e}'
                print(f'[peer] ble {self._scan_error}')
            try:
                await asyncio.sleep(SCAN_INTERVAL_S)
            except asyncio.CancelledError:
                return

    async def _scan_once(self) -> None:
        from bleak import BleakScanner

        # return_adv is required, not a nicety: bleak 3.x removed BLEDevice.rssi,
        # and rssi is the only proximity signal a human has when two robots in
        # the same room both appear in the list.
        found = await BleakScanner.discover(
            timeout=SCAN_TIMEOUT_S, service_uuids=[SERVICE_UUID], return_adv=True
        )
        # A successful round clears a transient failure, so the provider recovers
        # on its own once someone unblocks the radio — no settings save needed.
        # The standing notice (scan-only) deliberately survives.
        self._scan_error = ''

        now = time.time()
        my_peer_id = identity.peer_id()

        for address, (device, adv) in found.items():
            cached = self._identities.get(address)
            if cached and now - cached[4] < IDENTITY_TTL_S:
                self._emit(cached, adv.rssi, now)
                continue
            if now < self._failed_until.get(address, 0):
                continue
            try:
                record = await self._read_identity(device)
            except Exception as e:
                self._failed_until[address] = now + FAILURE_BACKOFF_S
                print(f'[peer] ble read failed for {address}: {type(e).__name__}: {e}')
                continue
            if record is None or record[0] == my_peer_id:
                # Our own advert comes back on adapters that hear themselves.
                self._failed_until[address] = now + FAILURE_BACKOFF_S
                continue
            self._identities[address] = record
            self._failed_until.pop(address, None)
            self._emit(record, adv.rssi, now)
            print(f'[peer] ble discovered {record[0][:12]} at {record[3] or "no endpoint"}')

    async def _read_identity(self, device) -> tuple | None:
        """Open a GATT connection and read the identity characteristics."""
        from bleak import BleakClient

        async with BleakClient(device, timeout=CONNECT_TIMEOUT_S) as client:
            raw_key = (await client.read_gatt_char(CHAR_PUBLIC_KEY)).decode('utf-8').strip()
            try:
                key_bytes = base64.b64decode(raw_key, validate=True)
            except (ValueError, TypeError):
                return None
            if len(key_bytes) != 32:
                return None
            peer_id = identity.fingerprint(key_bytes)

            endpoints = []
            try:
                blob = (await client.read_gatt_char(CHAR_ENDPOINTS)).decode('utf-8')
                parsed = json.loads(blob)
                if isinstance(parsed, list):
                    endpoints = [e for e in parsed if isinstance(e, str) and e]
            except Exception:
                # A peer with no IP at all is a legitimate state, not an error —
                # it is discoverable and simply cannot be paired with yet.
                pass

            display_name = ''
            try:
                display_name = (await client.read_gatt_char(CHAR_NAME)).decode('utf-8').strip()
            except Exception:
                pass  # older peers do not expose this characteristic

            return (peer_id, raw_key, display_name, endpoints, time.time())

    def _emit(self, record: tuple, rssi, now: float) -> None:
        peer_id, public_key, display_name, endpoints, _read_at = record
        self._on_advert(PeerAdvert(
            peer_id=peer_id,
            # Left empty when the peer did not supply one. The registry lets a
            # later non-empty scalar win, so inventing a placeholder here would
            # overwrite the real name mDNS already found with "BLE-a1b2c3d4".
            display_name=display_name,
            public_key=public_key,
            endpoints=list(endpoints),
            source=self.name,
            rssi=rssi,
            last_seen=now,
        ))


def local_endpoints() -> list[str]:
    """Reachable URLs to hand a BLE-discovered peer so it can dial back.

    Shared with the advertiser. Uses the default-route interface rather than
    enumerating every address: listing them all would advertise docker0 and veth
    addresses no peer can reach, and `netifaces` — the usual way to enumerate —
    is an unmaintained C extension that routinely fails to build on ARM64.
    """
    from peer.discovery.mdns import MdnsProvider

    settings = config.main.get('peer_settings', {}) or {}
    configured = (settings.get('advertise_url') or '').strip()
    if configured:
        return [configured]
    ip = MdnsProvider._primary_ip()
    if not ip or ip.startswith('127.'):
        return []
    return [f'https://{ip}:15678']


def local_display_name() -> str:
    settings = config.main.get('peer_settings', {}) or {}
    return settings.get('display_name') or socket.gethostname()
