"""
network.py — 网络配置管理（WiFi 扫描/连接/断开，接口状态）。

通过 Python dbus 直接与 NetworkManager D-Bus API 通信。
容器需要 network_mode: host + /var/run/dbus/system_bus_socket 挂载。
"""

import asyncio
import socket
import sys
import urllib.parse
import zlib

import fastapi
from pydantic import BaseModel

import config

# dbus-python 安装在系统 dist-packages，若 venv 无 system.pth 则手动加入
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.append('/usr/lib/python3/dist-packages')

router = fastapi.APIRouter(prefix='/network', tags=['network'])

_WIFI_STORE_KEY = 'wifi_saved'

# ── DB persistence ────────────────────────────────────────────────────────────


def _get_wifi_store() -> dict:
    """获取已保存的 WiFi 配置 {ssid: {password, auto_connect}}。"""
    return config.main.get(_WIFI_STORE_KEY, {})


def _save_wifi_entry(ssid: str, password: str, auto_connect: bool):
    """持久化 WiFi 凭据到数据库。"""
    store = _get_wifi_store()
    store[ssid] = {'password': password, 'auto_connect': auto_connect}
    config.main[_WIFI_STORE_KEY] = store


def _remove_wifi_entry(name: str):
    """从数据库中删除 WiFi 凭据。"""
    store = _get_wifi_store()
    store.pop(name, None)
    config.main[_WIFI_STORE_KEY] = store


# ── NetworkManager D-Bus helpers ──────────────────────────────────────────────

NM_IFACE = 'org.freedesktop.NetworkManager'
NM_PATH = '/org/freedesktop/NetworkManager'
NM_SETTINGS_PATH = '/org/freedesktop/NetworkManager/Settings'
NM_SETTINGS_IFACE = 'org.freedesktop.NetworkManager.Settings'
NM_CONN_IFACE = 'org.freedesktop.NetworkManager.Settings.Connection'
NM_DEVICE_IFACE = 'org.freedesktop.NetworkManager.Device'
NM_WIRELESS_IFACE = 'org.freedesktop.NetworkManager.Device.Wireless'
NM_AP_IFACE = 'org.freedesktop.NetworkManager.AccessPoint'
NM_ACTIVE_IFACE = 'org.freedesktop.NetworkManager.Connection.Active'
NM_IP4_IFACE = 'org.freedesktop.NetworkManager.IP4Config'
DBUS_PROPS_IFACE = 'org.freedesktop.DBus.Properties'

# Device type constants
NM_DEVICE_TYPE_ETHERNET = 1
NM_DEVICE_TYPE_WIFI = 2

# Device state constants
NM_DEVICE_STATE_ACTIVATED = 100
NM_DEVICE_STATE_DISCONNECTED = 30
NM_DEVICE_STATE_UNAVAILABLE = 20
NM_DEVICE_STATE_UNMANAGED = 10

_DEVICE_TYPE_NAMES = {
    NM_DEVICE_TYPE_ETHERNET: 'ethernet',
    NM_DEVICE_TYPE_WIFI: 'wifi',
    13: 'bridge',  # NM_DEVICE_TYPE_BRIDGE
    14: 'generic',  # NM_DEVICE_TYPE_GENERIC
}

# Device types to exclude from the interface list
_DEVICE_TYPE_HIDDEN = {13, 14, 22}  # bridge, generic, dummy

_DEVICE_STATE_NAMES = {
    NM_DEVICE_STATE_ACTIVATED: 'connected',
    NM_DEVICE_STATE_DISCONNECTED: 'disconnected',
    NM_DEVICE_STATE_UNAVAILABLE: 'unavailable',
    NM_DEVICE_STATE_UNMANAGED: 'unmanaged',
}


def _get_bus():
    import dbus
    return dbus.SystemBus()


def _get_prop(bus, obj_path, iface, prop):
    import dbus
    obj = bus.get_object(NM_IFACE, obj_path)
    props = dbus.Interface(obj, DBUS_PROPS_IFACE)
    return props.Get(iface, prop)


def _get_all_props(bus, obj_path, iface):
    import dbus
    obj = bus.get_object(NM_IFACE, obj_path)
    props = dbus.Interface(obj, DBUS_PROPS_IFACE)
    return props.GetAll(iface)


def _get_connection_settings(bus, settings_path: str) -> dict:
    """GetSettings() of a Settings.Connection object, as a plain dict."""
    import dbus
    conn_obj = bus.get_object(NM_IFACE, settings_path)
    conn_iface = dbus.Interface(conn_obj, NM_CONN_IFACE)
    return conn_iface.GetSettings()


def _bytes_to_ssid(ssid_bytes) -> str:
    """Convert dbus byte array to string."""
    return bytes(ssid_bytes).decode('utf-8', errors='replace')


def _get_devices_sync() -> list[dict]:
    """Get all network devices with status, including IP/mask/gateway/MAC."""
    bus = _get_bus()
    devices_paths = _get_prop(bus, NM_PATH, NM_IFACE, 'Devices')
    results = []
    for dev_path in devices_paths:
        props = _get_all_props(bus, dev_path, NM_DEVICE_IFACE)
        dev_type = int(props.get('DeviceType', 0))
        if dev_type in _DEVICE_TYPE_HIDDEN:
            continue
        if dev_type not in _DEVICE_TYPE_NAMES:
            continue
        state = int(props.get('State', 0))
        iface_name = str(props.get('Interface', ''))
        mac = str(props.get('HwAddress', ''))
        # Fallback: read MAC from sysfs (NM < 1.24 may return empty HwAddress)
        if not mac:
            try:
                with open(f'/sys/class/net/{iface_name}/address') as f:
                    mac = f.read().strip()
            except Exception:
                pass

        # Get active connection name
        connection_name = ''
        settings_path = ''
        active_conn_path = str(props.get('ActiveConnection', ''))
        if active_conn_path and active_conn_path != '/':
            try:
                conn_props = _get_all_props(bus, active_conn_path, NM_ACTIVE_IFACE)
                connection_name = str(conn_props.get('Id', ''))
                settings_path = str(conn_props.get('Connection', ''))
            except Exception:
                pass

        # Policy routing — whether replies from this device are forced back out the
        # same device instead of following the lowest-metric default route. The
        # source-based `ip rule` is what makes that happen, so that is what we key
        # the state off. `route-table` is also accepted so profiles written by the
        # older scheme (which relocated the routes instead of copying them, see
        # `_set_policy_route_sync`) still read as on and can be toggled off to clean up.
        policy_route = False
        if settings_path and settings_path != '/':
            try:
                ipv4_settings = _get_connection_settings(bus, settings_path).get('ipv4', {})
                policy_route = (bool(ipv4_settings.get('routing-rules'))
                                or int(ipv4_settings.get('route-table', 0)) != 0)
            except Exception:
                pass

        # Get IP info if activated
        ip = ''
        mask = ''
        gateway = ''
        if state == NM_DEVICE_STATE_ACTIVATED:
            ip4_path = str(props.get('Ip4Config', ''))
            if ip4_path and ip4_path != '/':
                try:
                    ip4_props = _get_all_props(bus, ip4_path, NM_IP4_IFACE)
                    addresses = ip4_props.get('AddressData', [])
                    if addresses:
                        addr = addresses[0]
                        ip = str(addr.get('address', ''))
                        prefix = int(addr.get('prefix', 0))
                        # Convert prefix to subnet mask
                        mask = _prefix_to_mask(prefix)
                    gateway = str(ip4_props.get('Gateway', ''))
                except Exception:
                    pass

        results.append({
            'device': iface_name,
            'type': _DEVICE_TYPE_NAMES.get(dev_type, 'unknown'),
            'state': _DEVICE_STATE_NAMES.get(state, f'unknown({state})'),
            'connection': connection_name,
            'mac': mac,
            'ip': ip,
            'mask': mask,
            'gateway': gateway,
            'policy_route': policy_route,
        })
    return results


def _prefix_to_mask(prefix: int) -> str:
    """Convert CIDR prefix length to dotted subnet mask."""
    bits = (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF
    return f"{(bits >> 24) & 0xFF}.{(bits >> 16) & 0xFF}.{(bits >> 8) & 0xFF}.{bits & 0xFF}"


def _get_wifi_device_path(bus) -> str | None:
    """Find the first WiFi device path."""
    devices_paths = _get_prop(bus, NM_PATH, NM_IFACE, 'Devices')
    for dev_path in devices_paths:
        dev_type = int(_get_prop(bus, dev_path, NM_DEVICE_IFACE, 'DeviceType'))
        if dev_type == NM_DEVICE_TYPE_WIFI:
            return str(dev_path)
    return None


def _scan_wifi_sync() -> list[dict]:
    """Scan and list WiFi access points."""
    import dbus
    bus = _get_bus()
    wifi_path = _get_wifi_device_path(bus)
    if not wifi_path:
        raise RuntimeError('未找到 WiFi 适配器')

    # Request scan
    wifi_obj = bus.get_object(NM_IFACE, wifi_path)
    wifi_iface = dbus.Interface(wifi_obj, NM_WIRELESS_IFACE)
    try:
        wifi_iface.RequestScan({})
        # Give scan a moment
        import time
        time.sleep(2)
    except Exception:
        pass  # Scan may already be in progress

    # Get current active AP
    active_ap = ''
    try:
        active_ap = str(_get_prop(bus, wifi_path, NM_WIRELESS_IFACE, 'ActiveAccessPoint'))
    except Exception:
        pass

    # Get access points
    aps = wifi_iface.GetAccessPoints()
    networks = []
    seen_ssids = set()
    for ap_path in aps:
        ap_props = _get_all_props(bus, str(ap_path), NM_AP_IFACE)
        ssid_bytes = ap_props.get('Ssid', [])
        ssid = _bytes_to_ssid(ssid_bytes)
        if not ssid or ssid in seen_ssids:
            continue
        seen_ssids.add(ssid)

        signal = int(ap_props.get('Strength', 0))
        flags = int(ap_props.get('Flags', 0))
        wpa_flags = int(ap_props.get('WpaFlags', 0))
        rsn_flags = int(ap_props.get('RsnFlags', 0))

        # Determine security type
        security = ''
        if rsn_flags != 0:
            security = 'WPA2'
        elif wpa_flags != 0:
            security = 'WPA'
        elif flags & 0x1:  # NM_802_11_AP_FLAGS_PRIVACY
            security = 'WEP'

        in_use = (str(ap_path) == active_ap)
        networks.append({
            'ssid': ssid,
            'signal': signal,
            'security': security,
            'in_use': in_use,
        })

    networks.sort(key=lambda x: (-x['in_use'], -x['signal']))
    return networks


def _connect_wifi_sync(ssid: str, password: str, auto_connect: bool) -> str:
    """Connect to a WiFi network. Returns success message or raises."""
    import dbus
    bus = _get_bus()
    wifi_path = _get_wifi_device_path(bus)
    if not wifi_path:
        raise RuntimeError('未找到 WiFi 适配器')

    # Find the AP with matching SSID
    wifi_obj = bus.get_object(NM_IFACE, wifi_path)
    wifi_iface = dbus.Interface(wifi_obj, NM_WIRELESS_IFACE)
    aps = wifi_iface.GetAccessPoints()
    target_ap = None
    for ap_path in aps:
        ap_ssid = _bytes_to_ssid(_get_prop(bus, str(ap_path), NM_AP_IFACE, 'Ssid'))
        if ap_ssid == ssid:
            target_ap = str(ap_path)
            break

    if not target_ap:
        raise RuntimeError(f'未找到网络 "{ssid}"，请先扫描')

    # Build connection settings
    conn_settings = dbus.Dictionary({
        'connection': dbus.Dictionary({
            'id': ssid,
            'type': '802-11-wireless',
            'autoconnect': dbus.Boolean(auto_connect),
        }),
        '802-11-wireless': dbus.Dictionary({
            'ssid': dbus.ByteArray(ssid.encode('utf-8')),
            'mode': 'infrastructure',
        }),
    })

    if password:
        conn_settings['802-11-wireless-security'] = dbus.Dictionary({
            'key-mgmt': 'wpa-psk',
            'psk': password,
        })
        conn_settings['802-11-wireless']['security'] = '802-11-wireless-security'

    # Use AddAndActivateConnection
    nm_obj = bus.get_object(NM_IFACE, NM_PATH)
    nm_iface = dbus.Interface(nm_obj, NM_IFACE)
    nm_iface.AddAndActivateConnection(conn_settings, dbus.ObjectPath(wifi_path), dbus.ObjectPath(target_ap))

    return f'已连接到 {ssid}'


def _disconnect_wifi_sync():
    """Disconnect the WiFi device."""
    import dbus
    bus = _get_bus()
    wifi_path = _get_wifi_device_path(bus)
    if not wifi_path:
        raise RuntimeError('未找到 WiFi 接口')

    nm_obj = bus.get_object(NM_IFACE, NM_PATH)
    nm_iface = dbus.Interface(nm_obj, NM_IFACE)
    nm_iface.DeactivateConnection(
        _get_prop(bus, wifi_path, NM_DEVICE_IFACE, 'ActiveConnection')
    )


def _policy_route_table_for(device: str) -> int:
    """Deterministic private routing table number for a device (200-249)."""
    return 200 + (zlib.crc32(device.encode()) % 50)


def _subnet_network(ip: str, prefix: int) -> str:
    """Network address (not host address) for `ip`/`prefix`, e.g. 10.100.129.141/19 -> 10.100.128.0."""
    a, b, c, d = (int(p) for p in ip.split('.'))
    ip_int = (a << 24) | (b << 16) | (c << 8) | d
    mask = (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF if prefix else 0
    network = ip_int & mask
    return f'{(network >> 24) & 0xFF}.{(network >> 16) & 0xFF}.{(network >> 8) & 0xFF}.{network & 0xFF}'


def _find_device_path_sync(bus, device: str) -> str | None:
    devices_paths = _get_prop(bus, NM_PATH, NM_IFACE, 'Devices')
    for dev_path in devices_paths:
        if str(_get_prop(bus, dev_path, NM_DEVICE_IFACE, 'Interface')) == device:
            return str(dev_path)
    return None


def _policy_route_data(table: int, network: str, prefix: int, gateway: str) -> list:
    """The routes to copy into private table `table` for a device on `network`/`prefix`.

    Two entries, both pinned to the table via the per-route `table` attribute:

    - the device's own subnet, kept on-link. Without it the default route below
      would swallow same-subnet destinations too (a table with a default route
      never falls through to main), so replies to a peer on this very subnet would
      take a needless detour through the gateway — and break outright behind AP
      client isolation.
    - a default route via `gateway`, for replies to everything else.

    NetworkManager adds the link route for `gateway` itself to the same table, so
    the default route is self-sufficient; that is not our job here.
    """
    import dbus
    entries = [
        dbus.Dictionary({
            'dest': dbus.String(network),
            'prefix': dbus.UInt32(prefix),
            'table': dbus.UInt32(table),
        }, signature='sv'),
    ]
    if gateway:
        entries.append(dbus.Dictionary({
            'dest': dbus.String('0.0.0.0'),
            'prefix': dbus.UInt32(0),
            'next-hop': dbus.String(gateway),
            'table': dbus.UInt32(table),
        }, signature='sv'))
    return entries


def _without_policy_routes(route_data, table: int) -> list:
    """`route_data` minus the entries we own, so a user's own static routes survive
    both enabling and disabling."""
    return [r for r in route_data if int(r.get('table', 0)) != table]


def _set_policy_route_sync(device: str, enable: bool):
    """Enable/disable policy routing on a device's active connection so replies
    to addresses outside its own subnet are still sent back out through it,
    instead of following another interface's lower-metric default route
    (asymmetric routing on multi-homed hosts).

    Implemented by *copying* the device's routes into a private table and pointing
    a source-based `ip rule` at it, leaving the main table alone.

    Do NOT reach for `ipv4.route-table` here, however natural it looks. It does not
    copy the connection's routes into the private table, it *relocates* them — the
    DHCP default route included. The device then stops being usable as a general
    egress path at all, and NetworkManager additionally reports the link to
    systemd-resolved as not-a-default-route, which silently drops its DNS servers
    (`resolvectl status` shows `Current Scopes: none`). Bumi 2026-09-02: the toggle
    was on for wifi, so every outbound connection fell to a dead 10.42.0.1 default
    route on another NIC and resolution rode on that same dead link. `ip route add
    default … dev wifi` could not even be added by hand — with the subnet route
    relocated too, the main table had no route to the wifi gateway, so the kernel
    rejected the nexthop as invalid.

    `ipv4.routing-rules` is required and separate: setting a route table alone does
    not install the matching `ip rule`. The rule is keyed on the device's current
    *subnet* rather than its exact address so it survives a DHCP renewal handing out
    a different address in the same subnet (the common case). The copied routes pin
    the current gateway, so a renewal into a different subnet or gateway needs this
    setting re-toggled; until then only reply symmetry is stale — the main table is
    still whatever DHCP says, so the host keeps its connectivity.

    Applied with Device.Reapply, which on NM 1.36.6 picks up routes, routing rules
    and route-table without touching the link (verified on Bumi — an SSH session
    over the very device being reconfigured survived). Older NM may not, so a full
    deactivate+reactivate remains as fallback; that one does drop the link briefly.
    """
    import dbus
    import time
    bus = _get_bus()
    dev_path = _find_device_path_sync(bus, device)
    if not dev_path:
        raise RuntimeError(f'未找到设备 "{device}"')

    active_conn_path = str(_get_prop(bus, dev_path, NM_DEVICE_IFACE, 'ActiveConnection'))
    if not active_conn_path or active_conn_path == '/':
        raise RuntimeError(f'设备 "{device}" 当前未连接')
    settings_path = str(_get_prop(bus, active_conn_path, NM_ACTIVE_IFACE, 'Connection'))

    table = _policy_route_table_for(device)
    conn_obj = bus.get_object(NM_IFACE, settings_path)
    conn_iface = dbus.Interface(conn_obj, NM_CONN_IFACE)
    settings = conn_iface.GetSettings()
    ipv4 = settings.setdefault('ipv4', dbus.Dictionary({}, signature='sv'))
    kept_routes = _without_policy_routes(ipv4.get('route-data', []), table)

    if enable:
        ip4_path = str(_get_prop(bus, dev_path, NM_DEVICE_IFACE, 'Ip4Config'))
        if not ip4_path or ip4_path == '/':
            raise RuntimeError(f'设备 "{device}" 当前没有 IP 地址')
        ip4_props = _get_all_props(bus, ip4_path, NM_IP4_IFACE)
        addresses = ip4_props.get('AddressData', [])
        if not addresses:
            raise RuntimeError(f'设备 "{device}" 当前没有 IP 地址')
        prefix = int(addresses[0].get('prefix', 32))
        network = _subnet_network(str(addresses[0].get('address', '')), prefix)
        gateway = str(ip4_props.get('Gateway', ''))
        new_routes = kept_routes + _policy_route_data(table, network, prefix, gateway)
        # Wire format is aa{sv}, not a plain string list — confirmed by round-tripping
        # a rule set via `nmcli ... +ipv4.routing-rules "priority ... from ... table ..."`
        # through GetSettings() and inspecting the actual dbus.Dictionary it produced.
        new_rules = [
            dbus.Dictionary({
                'family': dbus.Int32(socket.AF_INET),
                'priority': dbus.UInt32(table),
                'from': dbus.String(network),
                'from-len': dbus.Byte(prefix),
                'table': dbus.UInt32(table),
            }, signature='sv'),
        ]
    else:
        new_routes = kept_routes
        new_rules = []

    # Always cleared, both to undo profiles written by the older scheme and to keep
    # our own `table=` route attributes from being overridden by a connection-wide table.
    ipv4['route-table'] = dbus.UInt32(0)
    ipv4['route-data'] = dbus.Array(new_routes, signature='a{sv}')
    ipv4['routing-rules'] = dbus.Array(new_rules, signature='a{sv}')
    # Legacy alias for route-data; leaving a stale copy behind lets the two disagree.
    ipv4.pop('routes', None)

    conn_iface.Update(settings)

    try:
        dev_iface = dbus.Interface(bus.get_object(NM_IFACE, dev_path), NM_DEVICE_IFACE)
        # Empty connection + version_id 0 = "reapply the saved profile", which is what
        # `nmcli device reapply` sends and what we just wrote via Update().
        dev_iface.Reapply(dbus.Dictionary({}, signature='sa{sv}'), dbus.UInt64(0), dbus.UInt32(0))
        return
    except Exception:
        pass

    nm_obj = bus.get_object(NM_IFACE, NM_PATH)
    nm_iface = dbus.Interface(nm_obj, NM_IFACE)
    nm_iface.DeactivateConnection(dbus.ObjectPath(active_conn_path))
    time.sleep(1)
    nm_iface.ActivateConnection(
        dbus.ObjectPath(settings_path), dbus.ObjectPath(dev_path), dbus.ObjectPath('/'))




def _get_saved_wifi_sync() -> list[dict]:
    """List saved WiFi connections from NetworkManager."""
    import dbus
    bus = _get_bus()
    settings_obj = bus.get_object(NM_IFACE, NM_SETTINGS_PATH)
    settings_iface = dbus.Interface(settings_obj, NM_SETTINGS_IFACE)
    connections = settings_iface.ListConnections()

    results = []
    for conn_path in connections:
        conn_obj = bus.get_object(NM_IFACE, str(conn_path))
        conn_iface = dbus.Interface(conn_obj, NM_CONN_IFACE)
        settings = conn_iface.GetSettings()
        conn_type = str(settings.get('connection', {}).get('type', ''))
        if conn_type == '802-11-wireless':
            name = str(settings.get('connection', {}).get('id', ''))
            results.append({'name': name})
    return results


def _delete_connection_sync(name: str):
    """Delete a saved connection by name."""
    import dbus
    bus = _get_bus()
    settings_obj = bus.get_object(NM_IFACE, NM_SETTINGS_PATH)
    settings_iface = dbus.Interface(settings_obj, NM_SETTINGS_IFACE)
    connections = settings_iface.ListConnections()

    for conn_path in connections:
        conn_obj = bus.get_object(NM_IFACE, str(conn_path))
        conn_iface = dbus.Interface(conn_obj, NM_CONN_IFACE)
        settings = conn_iface.GetSettings()
        conn_id = str(settings.get('connection', {}).get('id', ''))
        if conn_id == name:
            conn_iface.Delete()
            return
    raise RuntimeError(f'未找到连接 "{name}"')


def _get_status_sync() -> list[dict]:
    """Get IP/gateway/DNS for active connections."""
    bus = _get_bus()
    devices_paths = _get_prop(bus, NM_PATH, NM_IFACE, 'Devices')
    connections = []
    for dev_path in devices_paths:
        dev_props = _get_all_props(bus, dev_path, NM_DEVICE_IFACE)
        state = int(dev_props.get('State', 0))
        if state != NM_DEVICE_STATE_ACTIVATED:
            continue
        dev_type = int(dev_props.get('DeviceType', 0))
        if dev_type not in _DEVICE_TYPE_NAMES:
            continue

        iface_name = str(dev_props.get('Interface', ''))
        ip4_path = str(dev_props.get('Ip4Config', ''))
        if not ip4_path or ip4_path == '/':
            continue

        try:
            ip4_props = _get_all_props(bus, ip4_path, NM_IP4_IFACE)
            addresses = ip4_props.get('AddressData', [])
            gateway = str(ip4_props.get('Gateway', ''))
            dns_data = ip4_props.get('NameserverData', [])

            ip_str = ''
            if addresses:
                addr = addresses[0]
                ip_str = f"{addr.get('address', '')}/{addr.get('prefix', '')}"

            dns_list = [str(d.get('address', '')) for d in dns_data]

            connections.append({
                'device': iface_name,
                'ip': ip_str,
                'gateway': gateway,
                'dns': dns_list,
            })
        except Exception:
            pass
    return connections


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get('/interfaces')
async def get_interfaces():
    """列出所有网络接口及连接状态。"""
    try:
        loop = asyncio.get_event_loop()
        interfaces = await loop.run_in_executor(None, _get_devices_sync)
        return {'code': 200, 'data': {'interfaces': interfaces}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e)}}


@router.get('/status')
async def get_status():
    """获取当前活跃连接的 IP/网关/DNS 信息。"""
    try:
        loop = asyncio.get_event_loop()
        connections = await loop.run_in_executor(None, _get_status_sync)
        return {'code': 200, 'data': {'connections': connections}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e)}}


@router.get('/wifi/scan')
async def wifi_scan():
    """扫描可用 WiFi 网络。"""
    try:
        loop = asyncio.get_event_loop()
        networks = await loop.run_in_executor(None, _scan_wifi_sync)
        return {'code': 200, 'data': {'networks': networks}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e)}}


class WifiConnectRequest(BaseModel):
    ssid: str
    password: str = ''
    auto_connect: bool = True


@router.post('/wifi/connect')
async def wifi_connect(req: WifiConnectRequest):
    """连接到指定 WiFi 网络。"""
    try:
        loop = asyncio.get_event_loop()
        msg = await loop.run_in_executor(
            None, _connect_wifi_sync, req.ssid, req.password, req.auto_connect)
        # 持久化到数据库
        _save_wifi_entry(req.ssid, req.password, req.auto_connect)
        return {'code': 200, 'data': {'success': True, 'message': msg}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e), 'success': False}}


@router.post('/wifi/disconnect')
async def wifi_disconnect():
    """断开当前 WiFi 连接。"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _disconnect_wifi_sync)
        return {'code': 200, 'data': {'success': True, 'message': 'WiFi 已断开'}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e)}}


@router.get('/wifi/saved')
async def wifi_saved():
    """列出已保存的 WiFi 连接（合并 NM 与数据库）。"""
    try:
        loop = asyncio.get_event_loop()
        connections = await loop.run_in_executor(None, _get_saved_wifi_sync)
        nm_names = {c['name'] for c in connections}

        # 补充数据库中有但 NM 中没有的（例如容器重建后）
        store = _get_wifi_store()
        for ssid in store:
            if ssid not in nm_names:
                connections.append({'name': ssid, 'db_only': True})

        return {'code': 200, 'data': {'connections': connections}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e)}}


@router.delete('/wifi/saved/{name:path}')
async def wifi_forget(name: str):
    """删除已保存的 WiFi 连接。"""
    name = urllib.parse.unquote(name)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _delete_connection_sync, name)
    except Exception:
        pass  # NM 中可能不存在（db_only 条目）
    # 同时从数据库删除
    _remove_wifi_entry(name)
    return {'code': 200, 'data': {'success': True, 'message': f'已删除 {name}'}}


class PolicyRouteRequest(BaseModel):
    device: str
    enable: bool


@router.post('/policy-route')
async def set_policy_route(req: PolicyRouteRequest):
    """开启/关闭指定接口的策略路由（防止多网卡时回包走了错误的默认路由）。"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _set_policy_route_sync, req.device, req.enable)
        return {'code': 200, 'data': {'success': True}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e), 'success': False}}
