"""
peer/ble_advertiser.py — BLE GATT server for Linux (optional, uses bluez_peripheral).

Exposes a GATT service with our peer identity for nearby robots to discover.
This is Linux-specific and requires `bluez_peripheral` library + BlueZ >= 5.43.

On platforms without BlueZ (macOS, Windows, non-BLE hardware), this silently
does nothing — the scanning half in ble_bootstrap.py still works.
"""

import asyncio
import json
import os

try:
    from bluez_peripheral.gatt.service import Service
    from bluez_peripheral.gatt.characteristic import characteristic, CharacteristicFlags
    from bluez_peripheral.advert import Advertisement
    from bluez_peripheral.agent import NoIoAgent
    BLUEZ_PERIPHERAL_AVAILABLE = True
except ImportError:
    BLUEZ_PERIPHERAL_AVAILABLE = False
    # Dummy base class for when bluez_peripheral is unavailable
    class Service:
        pass
    def characteristic(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    class CharacteristicFlags:
        READ = None

from peer.ble_bootstrap import SERVICE_UUID, CHAR_PUBLIC_KEY, CHAR_ENDPOINTS


_adapter = None
_service = None
_advertisement = None
_running = False


def is_available() -> bool:
    """Check if bluez_peripheral is available."""
    return BLUEZ_PERIPHERAL_AVAILABLE


async def start_advertising():
    """Start BLE advertising with GATT service."""
    global _adapter, _service, _advertisement, _running

    if not is_available():
        print('[peer] BLE advertising disabled: bluez_peripheral unavailable')
        return

    if _running:
        return

    try:
        from bluez_peripheral.util import get_message_bus
        bus = await get_message_bus()

        from peer import identity
        my_peer_id = identity.peer_id()

        # Create GATT service
        service = MotusPeerService()
        await service.register(bus)

        # Create advertisement
        adapter_name = os.environ.get('BLE_ADAPTER', 'hci0')
        advertisement = Advertisement(
            SERVICE_UUID,
            'peripheral',
            adapter_name,
        )
        await advertisement.register(bus)

        _service = service
        _advertisement = advertisement
        _running = True
        print(f'[peer] BLE advertising peer_id={my_peer_id[:12]} on {adapter_name}')

    except Exception as e:
        print(f'[peer] BLE advertising failed: {e}')
        _running = False


async def stop_advertising():
    """Stop BLE advertising."""
    global _running, _service, _advertisement

    if not _running:
        return

    _running = False

    try:
        if _advertisement:
            await _advertisement.unregister()
        if _service:
            await _service.unregister()
    except Exception as e:
        print(f'[peer] BLE stop error: {e}')

    _advertisement = None
    _service = None
    print('[peer] BLE advertising stopped')


class MotusPeerService(Service):
    """GATT service exposing peer identity and endpoints."""

    def __init__(self):
        super().__init__(SERVICE_UUID, True)

    @characteristic(CHAR_PUBLIC_KEY, CharacteristicFlags.READ)
    def public_key(self, options):
        """Return base64 Ed25519 public key."""
        from peer import identity
        return identity.public_key_b64().encode('utf-8')

    @characteristic(CHAR_ENDPOINTS, CharacteristicFlags.READ)
    def endpoints(self, options):
        """Return JSON list of reachable URLs."""
        endpoints = _get_local_endpoints()
        return json.dumps(endpoints).encode('utf-8')


def _get_local_endpoints() -> list[str]:
    """Get local HTTP endpoints for this peer.

    Returns all non-loopback IP addresses on port 15678.
    """
    import socket
    import netifaces

    endpoints = []
    try:
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info.get('addr')
                    if ip and not ip.startswith('127.'):
                        endpoints.append(f'https://{ip}:15678')
    except Exception:
        # Fallback: use hostname
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip and not ip.startswith('127.'):
                endpoints.append(f'https://{ip}:15678')
        except Exception:
            pass

    return endpoints
