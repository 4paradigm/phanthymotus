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
_UPLINK_KEY = 'preferred_uplink'

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


def _default_routes() -> dict:
    """`{interface: metric}` for every default route the kernel currently holds.

    NetworkManager's per-device `Gateway` property says a device *has* a gateway
    configured. It does not say that gateway is the one carrying traffic, and on
    a robot those are routinely different things: the wired port goes to the
    robot's own body network, which has a static gateway that routes nowhere,
    while WiFi is the only way out. Both devices then report a gateway and look
    equally healthy.

    Which one wins is decided by metric, and the default metrics are against us:
    NetworkManager gives Ethernet 100 and WiFi 600, so the body link beats the
    office WiFi on every robot wired this way, and nothing in this API used to
    say so. Measured on r1_sz: `default via 192.168.123.1 dev eth10 metric
    20100` ahead of `default via 10.100.128.1 dev wlan0 metric 20600`, with
    192.168.123.1 not answering so much as a ping — no DNS, no image pulls, and
    a driver marketplace that reported "no drivers found".

    Read from /proc/net/route rather than by running `ip`: agent-core runs with
    `network_mode: host`, so this is the host's real table, and it needs no
    binary that may not be in the image.
    """
    routes: dict = {}
    try:
        with open('/proc/net/route') as handle:
            next(handle, None)                      # header
            for line in handle:
                fields = line.split()
                if len(fields) < 7:
                    continue
                iface, destination, _gw, _flags, _rc, _use, metric = fields[:7]
                # Destination 00000000 is 0.0.0.0/0 — a default route.
                if destination != '00000000':
                    continue
                value = int(metric)
                # Lowest metric wins, so a device with several keeps the best.
                if iface not in routes or value < routes[iface]:
                    routes[iface] = value
    except Exception:
        return {}
    return routes


def _internet_device() -> str:
    """The interface that currently carries traffic to the internet, or ''."""
    routes = _default_routes()
    return min(routes, key=routes.get) if routes else ''


def _get_devices_sync() -> list[dict]:
    """Get all network devices with status, including IP/mask/gateway/MAC."""
    bus = _get_bus()
    # Read once, outside the loop: the table cannot change meaningfully within
    # one enumeration, and re-reading per device would let two devices disagree
    # about which of them holds the default route.
    default_routes = _default_routes()
    winner = min(default_routes, key=default_routes.get) if default_routes else ''
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
            # Which default route this device holds, and whether it is the one
            # actually used. `gateway` alone cannot answer that, and on a
            # multi-homed robot the difference is the whole question — see
            # `_default_routes`.
            'route_metric': default_routes.get(iface_name),
            'default_route': iface_name in default_routes,
            'carries_internet': bool(winner) and iface_name == winner,
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

    wifi_name = str(_get_prop(bus, wifi_path, NM_DEVICE_IFACE, 'Interface') or '')
    return f'已连接到 {ssid}' + _uplink_warning(wifi_name)


def _uplink_warning(wifi_iface: str, settle_s: float = 6.0) -> str:
    """Whether associating with this AP actually gave the machine a way out.

    Associating and having internet are different facts, and reporting only the
    first is how a machine ends up looking configured while nothing can reach
    it. The wired port on a robot usually goes to the robot's own body network,
    whose gateway routes nowhere; NetworkManager's default metrics (Ethernet
    100, WiFi 600) hand that link the default route, so WiFi connects, reports
    success, and changes nothing. Seen on r1_sz, where the symptom that finally
    got noticed was an empty driver marketplace.

    Returns '' when WiFi did win, so the ordinary case reads exactly as before.

    Deliberately a **warning, not a repair**. Lowering the WiFi metric here
    would silently demote the wired uplink, and on a robot that has a real
    wired uplink that is the wrong answer — there is no way to tell the two
    apart from inside this function. Naming the interface that holds the route
    is enough for an operator to decide.
    """
    import time as _time

    if not wifi_iface:
        return ''
    # DHCP has to finish before the route exists; polling beats one fixed sleep
    # because the usual case resolves in well under a second.
    # Read first, then poll — a do-while, not a while. With `settle_s` already
    # elapsed the loop body would never run, and the function would report "no
    # default route at all" no matter what the table says.
    deadline = _time.monotonic() + settle_s
    routes = _default_routes()
    while wifi_iface not in routes and _time.monotonic() < deadline:
        _time.sleep(0.5)
        routes = _default_routes()

    if not routes:
        return '（注意：这台机器没有任何默认路由，仍然无法上网）'
    winner = min(routes, key=routes.get)
    if winner == wifi_iface:
        return ''
    if wifi_iface not in routes:
        return (f'（注意：WiFi 已关联但没有拿到默认路由，出网仍然走 {winner}；'
                f'如果 {winner} 接的是机器人本体，这台机器上不了网）')
    return (f'（注意：出网仍然走 {winner}，它的路由优先级更高'
            f'（metric {routes[winner]} < WiFi 的 {routes[wifi_iface]}）。'
            f'如果 {winner} 接的是机器人本体，那条路由不通，'
            f'请去掉它的默认网关或调高它的 metric）')


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


_POLICY_TABLE_MIN = 200
_POLICY_TABLE_MAX = 249


def _policy_route_table_for(device: str) -> int:
    """Preferred private routing table number for a device (200-249).

    Only a starting point — `_allocate_policy_table` may hand out a different one to
    avoid a collision, or reuse whatever the connection already has.
    """
    return _POLICY_TABLE_MIN + (zlib.crc32(device.encode()) % 50)


def _tables_of(ipv4: dict) -> set:
    """Private-range table numbers an ipv4 setting refers to, via either its routing
    rules or the `table` attribute of its routes."""
    tables = set()
    for entry in list(ipv4.get('routing-rules', [])) + list(ipv4.get('route-data', [])):
        try:
            table = int(entry.get('table', 0))
        except (TypeError, ValueError):
            continue
        if _POLICY_TABLE_MIN <= table <= _POLICY_TABLE_MAX:
            tables.add(table)
    return tables


def _allocate_policy_table(own_ipv4: dict, other_ipv4s: list, preferred: int) -> int:
    """Pick the private table to use for one connection.

    Reuse whatever this connection already refers to first. That keeps the number
    stable across a WiFi card swap: the device name is MAC-derived, so a new card
    renames the interface and `_policy_route_table_for` would hand out a different
    number, stranding the routes and rule written for the old one — exactly what
    happened on Bumi 2026-09-02 when a dead dongle was replaced (205 -> 218).

    Otherwise take `preferred` if free, else the lowest free number in the range.
    Deriving it from the device name alone collides: 50 slots and a crc32 means two
    NICs on one host can land on the same table, and then their rules quietly share
    one table and fight over it.
    """
    in_use = set()
    for ipv4 in other_ipv4s:
        in_use |= _tables_of(ipv4)

    mine = _tables_of(own_ipv4)
    reusable = sorted(mine - in_use)
    if reusable:
        return reusable[0]

    if preferred not in in_use:
        return preferred
    for table in range(_POLICY_TABLE_MIN, _POLICY_TABLE_MAX + 1):
        if table not in in_use:
            return table
    raise RuntimeError(
        f'策略路由表号已用尽（{_POLICY_TABLE_MIN}-{_POLICY_TABLE_MAX}）')


def _other_connection_ipv4s(bus, own_settings_path: str) -> list:
    """ipv4 settings of every saved connection except `own_settings_path`."""
    import dbus
    settings_obj = bus.get_object(NM_IFACE, NM_SETTINGS_PATH)
    settings_iface = dbus.Interface(settings_obj, NM_SETTINGS_IFACE)
    results = []
    for conn_path in settings_iface.ListConnections():
        if str(conn_path) == own_settings_path:
            continue
        try:
            results.append(_get_connection_settings(bus, str(conn_path)).get('ipv4', {}))
        except Exception:
            pass  # a connection we cannot read cannot be reasoned about; skip it
    return results


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


def _without_policy_routes(route_data) -> list:
    """`route_data` minus every entry we own, so a user's own static routes survive
    both enabling and disabling.

    Keyed on the whole private range rather than the one table currently in play:
    a renumbering — a WiFi card swap renames the interface, so the old code derived
    a different table — would otherwise strand the previous table's copies in the
    profile forever, one stale pair per swap.
    """
    return [r for r in route_data if not _tables_of({'route-data': [r]})]


def _set_policy_route_sync(device: str, enable: bool):
    """Enable/disable policy routing on a device's active connection so replies
    to addresses outside its own subnet are still sent back out through it,
    instead of following another interface's lower-metric default route
    (asymmetric routing on multi-homed hosts).

    Implemented by *copying* the device's routes into a private table and pointing
    a source-based `ip rule` at it, leaving the main table alone. The table number
    comes from `_allocate_policy_table`, which reuses whatever this connection already
    refers to — the device name is MAC-derived, so deriving the number from it alone
    renumbers on every card swap and strands the old copies.

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

    conn_obj = bus.get_object(NM_IFACE, settings_path)
    conn_iface = dbus.Interface(conn_obj, NM_CONN_IFACE)
    settings = conn_iface.GetSettings()
    ipv4 = settings.setdefault('ipv4', dbus.Dictionary({}, signature='sv'))
    table = _allocate_policy_table(
        ipv4, _other_connection_ipv4s(bus, settings_path), _policy_route_table_for(device))
    kept_routes = _without_policy_routes(ipv4.get('route-data', []))

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




def _set_route_priority_sync(device: str, prefer: bool):
    """Make `device` win (or stop winning) the race for the default route.

    The repair for the situation `_uplink_warning` reports. It exists as a
    button because the people who hit this are not the people who can edit
    `/etc/netplan` over SSH: on r1_sz the wired port went to the robot's body
    network, whose static gateway answers nothing, and NetworkManager's default
    metrics (Ethernet 100, WiFi 600) handed that dead link every packet the
    machine sent. WiFi was associated and healthy the whole time.

    Implemented as `ipv4.route-metric` on **this** connection only — the
    narrowest change that fixes it:

    * it touches one profile, the one the operator is looking at;
    * it is fully reversible (`prefer=False` restores NetworkManager's default,
      `-1`), and reversible *without* knowing what the value used to be;
    * it leaves the other interface's profile alone. Setting `never-default` on
      the wired side would be the more correct statement about a body link, but
      it requires being sure the wired port *is* a body link, and nothing here
      can tell that apart from a real wired uplink.

    Deliberately **not** automatic — see `_uplink_warning`. A robot with a
    genuine wired uplink should keep it, and only a person knows which kind of
    robot this is.

    The new metric is taken below whatever currently wins rather than fixed at
    some small constant, so this works regardless of what the other side is
    using, and floors at 1 because 0 means "let the kernel decide".
    """
    import dbus
    import time
    bus = _get_bus()
    dev_path = _find_device_path_sync(bus, device)
    if not dev_path:
        raise RuntimeError(f'未找到设备 "{device}"')

    active_conn_path = str(_get_prop(bus, dev_path, NM_DEVICE_IFACE, 'ActiveConnection'))
    if not active_conn_path or active_conn_path == '/':
        raise RuntimeError(f'设备 "{device}" 当前没有活动连接')
    settings_path = str(_get_prop(bus, active_conn_path, NM_ACTIVE_IFACE, 'Connection'))

    conn_obj = bus.get_object(NM_IFACE, settings_path)
    conn_iface = dbus.Interface(conn_obj, NM_CONN_IFACE)
    settings = conn_iface.GetSettings()
    ipv4 = settings.setdefault('ipv4', dbus.Dictionary({}, signature='sv'))

    if prefer:
        routes = _default_routes()
        others = {k: v for k, v in routes.items() if k != device}
        # Nothing else is competing: whatever NM chose is already fine, and
        # writing a metric would be a change with no effect to undo later.
        if not others:
            raise RuntimeError(
                f'没有别的接口在争默认路由，"{device}" 已经是出网接口了')
        target = max(1, min(others.values()) - 50)
        ipv4['route-metric'] = dbus.Int64(target)
    else:
        # -1 is NetworkManager's "use the built-in default for this device type",
        # which is the state before anyone pressed this button. Removing the key
        # would work too, but writing -1 says so explicitly in the profile.
        ipv4['route-metric'] = dbus.Int64(-1)

    conn_iface.Update(settings)

    try:
        dev_iface = dbus.Interface(bus.get_object(NM_IFACE, dev_path), NM_DEVICE_IFACE)
        dev_iface.Reapply(dbus.Dictionary({}, signature='sa{sv}'), dbus.UInt64(0), dbus.UInt32(0))
    except Exception:
        # Older NM may not pick up route-metric from Reapply. Costs a brief link
        # drop, which is acceptable for a button the operator just pressed.
        nm_iface = dbus.Interface(bus.get_object(NM_IFACE, NM_PATH), NM_IFACE)
        nm_iface.DeactivateConnection(dbus.ObjectPath(active_conn_path))
        time.sleep(1)
        nm_iface.ActivateConnection(
            dbus.ObjectPath(settings_path), dbus.ObjectPath(dev_path), dbus.ObjectPath('/'))

    # Remember the choice, so a reboot does not undo it. This is the half that
    # was missing: the metric itself lives in the NetworkManager profile and
    # does persist, but netplan regenerates the *other* interface's static
    # default route at boot, and on r1_sz that route came back with a lower
    # metric and took the uplink with it. An operator who pressed this button
    # once should not have to press it after every restart — see
    # `ensure_preferred_uplink`.
    config.main[_UPLINK_KEY] = device if prefer else ''

    # Report what actually happened rather than what was asked for: Reapply can
    # return cleanly and still leave the table unchanged, and "已设置" on a
    # machine that still cannot reach anything is the failure this whole change
    # is about.
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if (_internet_device() == device) == prefer:
            break
        time.sleep(0.5)
    return {'device': device, 'carries_internet': _internet_device() == device,
            'routes': _default_routes()}


def ensure_preferred_uplink() -> dict:
    """Re-assert the operator's uplink choice at startup.

    Called once from the lifespan. Does nothing unless somebody has actually
    chosen a device, so a machine that never had this problem is untouched.

    It exists because the choice does not survive a reboot on its own. The
    metric written into the NetworkManager profile does persist — but netplan
    regenerates the *other* interface's static default route at boot, and if
    that one wins again the machine comes back with no way out. On r1_sz that
    interface pointed at the robot's own body network, whose gateway answers
    nothing, so the symptom was every LLM call timing out after 5 s while WiFi
    showed as connected and healthy.

    Failures are reported, never raised: this runs during startup, and a robot
    that will not boot because its network could not be tidied up is a much
    worse outcome than one with the wrong default route.
    """
    device = str(config.main.get(_UPLINK_KEY) or '').strip()
    if not device:
        return {'checked': False}
    try:
        current = _internet_device()
        if current == device:
            return {'checked': True, 'device': device, 'changed': False}
        print(f'[network] uplink should be {device} but {current or "nothing"} '
              f'holds the default route — re-applying')
        _set_route_priority_sync(device, True)
        ok = _internet_device() == device
        print(f'[network] uplink re-applied: carries_internet={ok}')
        return {'checked': True, 'device': device, 'changed': True, 'ok': ok}
    except Exception as error:                      # noqa: BLE001
        print(f'[network] could not re-apply uplink preference: {error}')
        return {'checked': True, 'device': device, 'error': str(error)}


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


class RoutePriorityRequest(BaseModel):
    device: str
    prefer: bool


@router.post('/route-priority')
async def set_route_priority(req: RoutePriorityRequest):
    """让指定接口优先出网（prefer=true），或恢复系统默认优先级（false）。

    多网口机器人上，接机器人本体的那张网卡常常带着一个不通的静态网关，而
    NetworkManager 默认给以太网的 metric 比 WiFi 低，于是本体那条线赢走所有出
    网流量 —— WiFi 显示已连接，机器却上不了网。这个接口就是那种情况的修复入口，
    存在的理由是：碰到这个问题的人未必能去 SSH 改 netplan。
    """
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, _set_route_priority_sync, req.device, req.prefer)
        return {'code': 200, 'data': {'success': True, **(data or {})}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e), 'success': False}}


@router.post('/policy-route')
async def set_policy_route(req: PolicyRouteRequest):
    """开启/关闭指定接口的策略路由（防止多网卡时回包走了错误的默认路由）。"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _set_policy_route_sync, req.device, req.enable)
        return {'code': 200, 'data': {'success': True}}
    except Exception as e:
        return {'code': 500, 'data': {'error': str(e), 'success': False}}


# ── 本机可达地址 ──────────────────────────────────────────────────────────────
#
# 给「扫码上手机」用：浏览器只知道自己是怎么连上来的，而那常常是
# `localhost`——机器上本地开的浏览器、SSH 端口转发、反向代理，都会让
# `location.host` 变成一个手机拨不通的地址。所以地址必须由服务端给。
#
# 刻意不走 NetworkManager：`/network/interfaces` 依赖 D-Bus，开发机上没有，
# 而这个接口在开发机上也要能用。

_VIRTUAL_IFACE_PREFIXES = ('lo', 'docker', 'br-', 'veth', 'virbr', 'tun', 'tap')

# 网卡名到人话。Linux 的可预测命名里 en* 是以太网、wl* 是无线；
# eth*/wlan* 是旧式名字，机器人上两种都见得到。
_IFACE_KINDS = (
    (('wl',), 'wifi'),
    (('en', 'eth', 'end'), 'ethernet'),
)


def _iface_kind(name: str) -> str:
    for prefixes, kind in _IFACE_KINDS:
        if name.startswith(prefixes):
            return kind
    return 'other'


def _primary_ipv4() -> str:
    """默认路由会走的那个地址。UDP connect 不发包，只是让内核挑一次路由。

    `gethostname()` 解析不能替代它：容器里通常解析成回环或容器内部地址，
    那是个手机连不上的地址。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except OSError:
        return ''
    finally:
        s.close()


def _ipv4_addresses() -> list[dict]:
    """枚举各网卡的 IPv4 地址。

    Linux 走 ioctl（容器是 host 网络，看到的就是宿主机的网卡）；其它平台
    没有 SIOCGIFADDR，退回只报默认路由那一个地址。
    """
    found = []
    try:
        import fcntl
        import struct
        SIOCGIFADDR = 0x8915
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _, name in socket.if_nameindex():
                if name.startswith(_VIRTUAL_IFACE_PREFIXES):
                    continue
                try:
                    packed = fcntl.ioctl(
                        sock.fileno(), SIOCGIFADDR,
                        struct.pack('256s', name[:15].encode()))
                    ip = socket.inet_ntoa(packed[20:24])
                except OSError:
                    continue            # 网卡在但没拿到地址
                found.append({'device': name, 'ip': ip, 'kind': _iface_kind(name)})
        finally:
            sock.close()
    except (ImportError, AttributeError, OSError):
        pass
    return found


@router.get('/reachable')
async def reachable_addresses():
    """手机能用来访问这台 Agent Core 的地址，默认路由那个排在最前。"""
    primary = _primary_ipv4()
    addresses = [
        a for a in _ipv4_addresses()
        # 169.254/16 是拿不到 DHCP 时自己编的地址，贴出来只会让人白扫一次
        if not a['ip'].startswith(('127.', '169.254.'))
    ]
    if primary and not any(a['ip'] == primary for a in addresses):
        addresses.append({'device': '', 'ip': primary, 'kind': 'other'})
    for a in addresses:
        a['primary'] = a['ip'] == primary
    addresses.sort(key=lambda a: (not a['primary'], a['kind'] != 'wifi', a['device']))
    return {'code': 200, 'data': {'addresses': addresses}}
