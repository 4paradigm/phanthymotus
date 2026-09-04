"""
peer/ble_advertiser.py — the peripheral half of BLE discovery (Linux/BlueZ only).

Exposes a GATT service carrying this robot's identity so that a peer scanning
nearby (peer/discovery/ble.py) can read it without any network in common.

## Why this is a hand-installed dependency

`bluez_peripheral` is Linux-only and pre-1.0, and `uv lock` resolves every
declared extra when building the lockfile — declaring it would break the image
build on every platform, including the ARM64 robots. So it is imported lazily
and its absence degrades this robot to scan-only: it still finds peers, it just
cannot be found. See docs/PEER_SETUP.md § "Optional: BLE Advertising".

## Host prerequisites that are not Python's problem

BlueZ has to be reachable and the radio has to be on. In a container that means
`/var/run/dbus/system_bus_socket` mounted (agent-core's compose already does)
and, on the host, `rfkill unblock bluetooth` plus a running `bluetooth.service`.
Jetsons ship with the adapter soft-blocked, which surfaces here as a D-Bus error
with no obvious connection to Bluetooth at all.
"""

import os

try:
    from bluez_peripheral.gatt.service import Service
    from bluez_peripheral.gatt.characteristic import characteristic, CharacteristicFlags
    from bluez_peripheral.advert import Advertisement
    from bluez_peripheral.util import get_message_bus, Adapter
    BLUEZ_PERIPHERAL_AVAILABLE = True
except ImportError:
    BLUEZ_PERIPHERAL_AVAILABLE = False

    class Service:  # placeholder so the module still imports
        pass

    def characteristic(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator

    class CharacteristicFlags:
        READ = None

import json

from peer.discovery.ble import (
    SERVICE_UUID, CHAR_PUBLIC_KEY, CHAR_ENDPOINTS, CHAR_NAME,
    local_endpoints, local_display_name,
)


# BLE advertising packets are 31 bytes. A 128-bit service UUID eats 16 of them,
# so the local name has to be short or BlueZ silently drops it — the identity
# that matters is served over GATT, this is only what shows in a scanner.
_MAX_LOCAL_NAME = 8
# BlueZ treats 0 as "no timeout": advertise until unregistered. Anything else is
# capped at 180s by the adapter and would need a re-registration loop.
_ADVERT_TIMEOUT_S = 0
_APPEARANCE = 0  # unknown; robots have no assigned BLE appearance code

_bus = None
_service = None
_running = False


def is_available() -> bool:
    """Whether the peripheral stack is importable. Says nothing about whether an
    adapter exists or is unblocked — that only shows up on registration."""
    return BLUEZ_PERIPHERAL_AVAILABLE


def is_running() -> bool:
    return _running


async def start_advertising() -> None:
    """Register the GATT service and start advertising on the caller's loop.

    Must be awaited on a loop that keeps running: BlueZ serves every
    characteristic read back over D-Bus, so a loop that stops after registration
    leaves an advertised service that answers nothing.
    """
    global _bus, _service, _running

    if not is_available():
        raise RuntimeError('bluez_peripheral not installed')
    if _running:
        return

    bus = await get_message_bus()
    try:
        adapter = await _pick_adapter(bus)
        # An adapter that is present but powered off accepts registration and
        # then radiates nothing, which looks exactly like "no peers nearby".
        await adapter.set_powered(True)

        service = MotusPeerService()
        await service.register(bus, adapter=adapter)

        advert = Advertisement(
            localName=_short_name(),
            serviceUUIDs=[SERVICE_UUID],
            appearance=_APPEARANCE,
            timeout=_ADVERT_TIMEOUT_S,
        )
        await advert.register(bus, adapter=adapter)
    except Exception:
        bus.disconnect()
        raise

    _bus, _service, _running = bus, service, True
    from peer import identity
    print(f'[peer] ble advertising {identity.peer_id()[:12]} '
          f'on {await adapter.get_name()}')


async def stop_advertising() -> None:
    """Unregister and drop the bus.

    Disconnecting is what actually stops the advert: bluez_peripheral exposes no
    unregister for Advertisement, and BlueZ releases both the advert and the
    GATT application when the owning D-Bus connection goes away.
    """
    global _bus, _service, _running

    if not _running:
        return
    _running = False
    service, bus = _service, _bus
    _service, _bus = None, None

    if service is not None:
        try:
            await service.unregister()
        except Exception as e:
            print(f'[peer] ble service unregister failed: {type(e).__name__}: {e}')
    if bus is not None:
        try:
            bus.disconnect()
        except Exception as e:
            print(f'[peer] ble bus disconnect failed: {type(e).__name__}: {e}')
    print('[peer] ble advertising stopped')


async def _pick_adapter(bus):
    """The adapter named by $BLE_ADAPTER, else the first one BlueZ reports."""
    wanted = (os.environ.get('BLE_ADAPTER') or '').strip()
    if wanted:
        for adapter in await Adapter.get_all(bus):
            if await adapter.get_name() == wanted:
                return adapter
        raise RuntimeError(f'BLE adapter {wanted!r} not found')
    adapter = await Adapter.get_first(bus)
    if adapter is None:
        raise RuntimeError('no BLE adapter present')
    return adapter


def _short_name() -> str:
    return f'M-{local_display_name()}'[:_MAX_LOCAL_NAME]


class MotusPeerService(Service):
    """Identity, served over GATT.

    Read-only and unauthenticated by design — this is the same information mDNS
    puts in a TXT record for anyone on the LAN. It gets a peer as far as the
    pairing screen and no further.
    """

    def __init__(self):
        super().__init__(SERVICE_UUID, True)

    @characteristic(CHAR_PUBLIC_KEY, CharacteristicFlags.READ)
    def public_key(self, options):
        from peer import identity
        return identity.public_key_b64().encode('utf-8')

    @characteristic(CHAR_ENDPOINTS, CharacteristicFlags.READ)
    def endpoints(self, options):
        return json.dumps(local_endpoints()).encode('utf-8')

    @characteristic(CHAR_NAME, CharacteristicFlags.READ)
    def display_name(self, options):
        return local_display_name().encode('utf-8')
