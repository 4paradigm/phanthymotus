"""
peer/ble_bootstrap.py — BLE-based pairing bootstrap (full implementation).

Advertises peer_id as a BLE service with GATT characteristics containing:
  - public_key: base64 Ed25519 public key
  - endpoints: JSON list of reachable URLs

Scans for other peers advertising the same service and reads their characteristics,
then pushes them into the registry with source='ble'.

Once discovered via BLE, the normal HTTPS pairing handshake completes the trust
establishment (SAS code comparison on both dashboards).

## Service UUID Design

SERVICE_UUID: `12345678-1234-5678-1234-56789abcdef0` (Motus peer discovery)
  Characteristic CHAR_PUBLIC_KEY: `...def1` (read-only, base64 Ed25519 public key)
  Characteristic CHAR_ENDPOINTS: `...def2` (read-only, JSON array of URLs)

## Security

BLE advertisements are unauthenticated. The peer_id in the service name is just
a discovery hint; the actual identity verification happens during the HTTPS
pairing handshake (Ed25519 signature + SAS code comparison).
"""

import asyncio
import base64
import json
import os
import time
from typing import Optional

try:
    from bleak import BleakScanner, BleakClient
    from bleak.backends.service import BleakGATTServiceCollection
    from bleak.backends.characteristic import BleakGATTCharacteristic
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False


# Motus peer discovery service
SERVICE_UUID = '12345678-1234-5678-1234-56789abcdef0'
CHAR_PUBLIC_KEY = '12345678-1234-5678-1234-56789abcdef1'
CHAR_ENDPOINTS = '12345678-1234-5678-1234-56789abcdef2'

_running = False
_task: Optional[asyncio.Task] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


def is_available() -> bool:
    """Check if BLE is available on this system."""
    if not BLEAK_AVAILABLE:
        return False
    # bleak will fail gracefully if no adapter; we just check import succeeded
    return True


def start():
    """Start BLE bootstrap advertising and scanning."""
    global _running, _task, _loop
    if not is_available():
        print('[peer] BLE bootstrap disabled: bleak unavailable')
        return

    import config
    ble_config = config.main.get('peer_settings', {}).get('discovery', {}).get('ble', {})
    if not ble_config.get('enabled', False):
        print('[peer] BLE bootstrap disabled in config')
        return

    if _running:
        return

    _running = True

    # Start advertising (if bluez_peripheral is available)
    from peer import ble_advertiser
    if ble_advertiser.is_available():
        import threading
        def _start_adv():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(ble_advertiser.start_advertising())
        threading.Thread(target=_start_adv, daemon=True, name='ble_advertiser').start()

    # Start scanning in a background thread
    import threading
    def _thread_main():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        try:
            _loop.run_until_complete(_run_loop())
        except Exception as e:
            print(f'[peer] BLE bootstrap loop error: {e}')
        finally:
            _loop.close()
            _loop = None

    threading.Thread(target=_thread_main, daemon=True, name='ble_bootstrap').start()
    print('[peer] BLE bootstrap started')


def stop():
    """Stop BLE bootstrap."""
    global _running, _task
    _running = False
    if _task and not _task.done():
        _task.cancel()

    # Stop advertising
    from peer import ble_advertiser
    if ble_advertiser.is_available():
        import threading
        def _stop_adv():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(ble_advertiser.stop_advertising())
            loop.close()
        t = threading.Thread(target=_stop_adv, daemon=False, name='ble_stop')
        t.start()
        t.join(timeout=2.0)

    print('[peer] BLE bootstrap stopped')


async def _run_loop():
    """Main BLE loop: scan for peers periodically."""
    # Note: BLE advertising with GATT server is complex and platform-dependent.
    # bleak is primarily a client library. For advertising, we'd need:
    #   - Linux: use bluez D-Bus API directly (complex)
    #   - macOS/Windows: not easily supported for peripheral mode
    #
    # For now, we implement scanning only — this is the more critical half
    # (discovering peers). Advertising would require platform-specific code or
    # a dedicated BLE advertising library.
    #
    # In a real deployment, you'd run a separate advertiser (e.g., using
    # `bluez_peripheral` on Linux) that exposes the GATT service.

    print('[peer] BLE bootstrap: scanning mode only (advertising requires platform-specific setup)')

    while _running:
        try:
            await _scan_once()
        except Exception as e:
            print(f'[peer] BLE scan error: {e}')
        await asyncio.sleep(10)


async def _scan_once():
    """Scan for Motus peer services and read their characteristics."""
    from peer import identity
    from peer.registry import registry
    from peer.discovery.base import PeerAdvert

    my_peer_id = identity.peer_id()

    # Scan for devices advertising our service UUID
    devices = await BleakScanner.discover(timeout=5.0, service_uuids=[SERVICE_UUID])

    for device in devices:
        try:
            async with BleakClient(device) as client:
                # Read public key characteristic
                pub_key_bytes = await client.read_gatt_char(CHAR_PUBLIC_KEY)
                pub_key_b64 = pub_key_bytes.decode('utf-8')

                # Compute peer_id from public key
                pub_key_raw = base64.b64decode(pub_key_b64)
                peer_id = identity.fingerprint(pub_key_raw)

                # Skip self
                if peer_id == my_peer_id:
                    continue

                # Read endpoints characteristic
                endpoints_bytes = await client.read_gatt_char(CHAR_ENDPOINTS)
                endpoints_json = endpoints_bytes.decode('utf-8')
                endpoints = json.loads(endpoints_json)

                # Push to registry
                advert = PeerAdvert(
                    peer_id=peer_id,
                    display_name=f'BLE-{peer_id[:8]}',
                    endpoints=endpoints,
                    source='ble',
                    last_seen=time.time(),
                )
                registry.observe(advert)
                print(f'[peer] BLE discovered peer {peer_id[:12]} at {endpoints}')

        except Exception as e:
            # Device might not have our characteristics or disconnected
            continue


# ── Advertising (platform-specific, not implemented in bleak) ─────────────────
#
# To advertise, you need:
#
# Linux (BlueZ):
#   Use `bluez_peripheral` library or D-Bus directly to create a GATT server:
#
#   from bluez_peripheral.gatt.service import Service
#   from bluez_peripheral.gatt.characteristic import characteristic, CharacteristicFlags
#
#   class MotusPeerService(Service):
#       def __init__(self):
#           super().__init__(SERVICE_UUID, True)
#
#       @characteristic(CHAR_PUBLIC_KEY, CharacteristicFlags.READ)
#       def public_key(self, options):
#           from peer import identity
#           return identity.public_key_b64().encode('utf-8')
#
#       @characteristic(CHAR_ENDPOINTS, CharacteristicFlags.READ)
#       def endpoints(self, options):
#           endpoints = [f'https://{get_local_ip()}:15678']
#           return json.dumps(endpoints).encode('utf-8')
#
#   Then register the service with BlueZ and start advertising.
#
# macOS/Windows:
#   Peripheral mode is restricted. You'd need a separate device or use a
#   third-party library that wraps CoreBluetooth (macOS) or WinRT (Windows).
#
# For simplicity, this implementation focuses on scanning. Advertising can be
# added later with platform detection.
