"""
test_peer_ble.py — BLE 发现 provider。

前一版只断言 `is_available()` 返回 bool、`start()` 不崩 —— 那三条在 BLE 完全不
工作时也全绿，而它当时确实完全不工作：config 里根本没有 `discovery.ble` 这个键，
`start()` 每次都在第二行就 return 了。所以这里测的是**行为**：

  * provider 只有在配置打开时才被注册（这是当初的真实缺陷）
  * 扫到的记录能变成一条 peer_id 由公钥推导出来的 advert
  * 损坏的 base64、长度不对的公钥不会进 registry
  * 自己的广播会被丢掉
  * BLE 不会用占位名覆盖 mDNS 已经拿到的真实名字

不需要蓝牙硬件：GATT 读取被打桩，跑的是 provider 自己的逻辑。
"""

import asyncio
import base64
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import config  # noqa: E402
from peer import identity  # noqa: E402
from peer.discovery.ble import (  # noqa: E402
    BleProvider, CHAR_PUBLIC_KEY, CHAR_ENDPOINTS, CHAR_NAME, local_endpoints,
)
from peer.discovery.base import PeerAdvert  # noqa: E402
from peer.registry import registry  # noqa: E402


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeClient:
    """Stands in for BleakClient — serves whatever bytes the test supplies."""

    def __init__(self, chars: dict):
        self._chars = chars

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def read_gatt_char(self, uuid):
        if uuid not in self._chars:
            raise RuntimeError('characteristic not found')
        return self._chars[uuid]


def _chars(public_key_b64: str, endpoints=None, name=None) -> dict:
    out = {CHAR_PUBLIC_KEY: public_key_b64.encode()}
    if endpoints is not None:
        out[CHAR_ENDPOINTS] = json.dumps(endpoints).encode()
    if name is not None:
        out[CHAR_NAME] = name.encode()
    return out


class BleProviderTest(unittest.TestCase):
    def setUp(self):
        self._db = os.path.join(tempfile.mkdtemp(), 'peers.db')
        patch = mock.patch.object(config, 'DB_PATH', self._db)
        patch.start()
        self.addCleanup(patch.stop)
        identity.reset_cache()
        identity.ensure_identity()
        registry.reset()
        self.addCleanup(registry.reset)
        self.adverts: list[PeerAdvert] = []
        self.provider = BleProvider(self.adverts.append)

    def _scan(self, devices: dict, clients: dict):
        """Run one scan round with stubbed bleak."""
        scanner = mock.MagicMock()
        scanner.discover = mock.AsyncMock(return_value=devices)
        module = mock.MagicMock(
            BleakScanner=scanner,
            BleakClient=lambda device, timeout=None: _FakeClient(clients[device]),
        )
        with mock.patch.dict(sys.modules, {'bleak': module}):
            _run(self.provider._scan_once())

    @staticmethod
    def _device(addr, rssi=-50):
        adv = mock.MagicMock()
        adv.rssi = rssi
        return {addr: (addr, adv)}

    @staticmethod
    def _key() -> str:
        return base64.b64encode(os.urandom(32)).decode()

    # ── the defect that made every earlier test green ────────────────────────

    def test_default_config_has_ble_key(self):
        """默认配置里必须有这个键，否则界面上的开关存不下来，provider 永远起不来。"""
        self.assertIn('ble', config._DB_DEFAULTS['peer_settings']['discovery'])

    def test_registry_skips_ble_when_config_off(self):
        config.main['peer_settings'] = {'enabled': True, 'discovery': {'mdns': False}}
        _run(registry.start())
        self.assertNotIn('ble', [p['name'] for p in registry.provider_status()])

    def test_registry_starts_ble_when_config_on(self):
        """打开开关就要真的把 provider 拉起来 —— 这正是之前缺的那段接线。"""
        config.main['peer_settings'] = {
            'enabled': True, 'discovery': {'mdns': False, 'ble': True},
        }
        with mock.patch.object(BleProvider, '_start_advertising',
                               new=mock.AsyncMock(return_value=False)), \
                mock.patch.object(BleProvider, '_scan_loop', new=mock.AsyncMock()):
            _run(registry.start())
            self.assertIn('ble', [p['name'] for p in registry.provider_status()])
            _run(registry.stop())

    # ── scanning ─────────────────────────────────────────────────────────────

    def test_valid_peer_becomes_advert(self):
        other = self._key()
        self._scan(self._device('AA:BB', rssi=-42),
                   {'AA:BB': _chars(other, ['https://10.0.0.5:15678'], 'Orin6')})

        self.assertEqual(len(self.adverts), 1)
        a = self.adverts[0]
        self.assertEqual(a.peer_id, identity.fingerprint(base64.b64decode(other)))
        self.assertEqual(a.public_key, other)
        self.assertEqual(a.endpoints, ['https://10.0.0.5:15678'])
        self.assertEqual(a.display_name, 'Orin6')
        self.assertEqual(a.source, 'ble')
        self.assertEqual(a.rssi, -42)

    def test_own_advert_ignored(self):
        self._scan(self._device('AA:BB'),
                   {'AA:BB': _chars(identity.public_key_b64(), [])})
        self.assertEqual(self.adverts, [])

    def test_malformed_key_rejected(self):
        """坏 base64 和长度不对的公钥都不能进 registry。"""
        for bad in ('not-base64!!', base64.b64encode(b'short').decode()):
            with self.subTest(bad=bad):
                self.adverts.clear()
                self.provider._identities.clear()
                self.provider._failed_until.clear()
                self._scan(self._device('AA:BB'), {'AA:BB': _chars(bad, [])})
                self.assertEqual(self.adverts, [])

    def test_missing_endpoints_is_not_fatal(self):
        """完全离网的 peer 没有 endpoint —— 合法状态，不是错误。"""
        self._scan(self._device('AA:BB'), {'AA:BB': _chars(self._key())})
        self.assertEqual(len(self.adverts), 1)
        self.assertEqual(self.adverts[0].endpoints, [])

    def test_no_name_leaves_display_name_empty(self):
        """不编造 'BLE-a1b2c3d4'：registry 里后来的非空值会覆盖旧值，
        占位名会把 mDNS 已经拿到的真名冲掉。"""
        self._scan(self._device('AA:BB'), {'AA:BB': _chars(self._key(), [])})
        ble_advert = self.adverts[0]
        self.assertEqual(ble_advert.display_name, '')

        registry.observe(PeerAdvert(peer_id=ble_advert.peer_id,
                                    display_name='Orin6', source='mdns'))
        registry.observe(ble_advert)
        self.assertEqual(registry.get(ble_advert.peer_id).display_name, 'Orin6')

    def test_second_round_uses_cache_not_a_new_connection(self):
        """已知设备不再重连 —— 连接慢，而且会挤掉别的 central。"""
        chars = {'AA:BB': _chars(self._key(), [])}
        self._scan(self._device('AA:BB'), chars)
        with mock.patch.object(BleProvider, '_read_identity',
                               new=mock.AsyncMock(side_effect=AssertionError('reconnected'))):
            self._scan(self._device('AA:BB', rssi=-70), chars)
        self.assertEqual(len(self.adverts), 2)
        self.assertEqual(self.adverts[1].rssi, -70)  # 缓存的身份 + 新的信号强度

    def test_unreadable_device_is_backed_off(self):
        """广播了同一个 UUID 但读不出特征的设备，不该每轮都重试。"""
        self._scan(self._device('AA:BB'), {'AA:BB': _chars('not-base64!!', [])})
        self.assertGreater(self.provider._failed_until.get('AA:BB', 0), 0)
        with mock.patch.object(BleProvider, '_read_identity',
                               new=mock.AsyncMock(side_effect=AssertionError('retried'))):
            self._scan(self._device('AA:BB'), {'AA:BB': _chars('not-base64!!', [])})

    # ── health reporting ─────────────────────────────────────────────────────

    def test_scan_failure_surfaces_in_provider_status(self):
        """扫描一直失败时不能显示成绿的 —— rfkill 挡住电台就是这个样子。"""
        self.provider._running = True
        self.provider._scan_error = 'scan failed: BleakDBusError: rfkill'
        registry._providers = [self.provider]
        status = registry.provider_status()[0]
        self.assertFalse(status['running'])
        self.assertIn('rfkill', status['error'])

    def test_successful_scan_clears_previous_error(self):
        """有人解开 rfkill 之后要能自愈，不必去设置页再保存一次。"""
        self.provider._advertising = True
        self.provider._scan_error = 'scan failed: BleakDBusError: rfkill'
        self._scan(self._device('AA:BB'), {'AA:BB': _chars(self._key(), [])})
        self.assertEqual(self.provider.last_error, '')

    def test_scan_only_notice_survives_a_good_scan(self):
        """只能扫、不能被扫，是个持续状态，不该被一次成功扫描抹掉。"""
        self.provider._advertising = False
        self.provider._notice = 'scan only — bluez_peripheral not installed'
        self.provider._scan_error = 'scan failed: BleakDBusError: rfkill'
        self._scan(self._device('AA:BB'), {'AA:BB': _chars(self._key(), [])})
        self.assertEqual(self.provider.last_error,
                         'scan only — bluez_peripheral not installed')

    def test_start_without_bleak_raises_with_a_reason(self):
        """缺依赖时要抛出可读的原因，registry 会把它显示出来。"""
        real_import = __import__

        def _no_bleak(name, *args, **kwargs):
            if name == 'bleak':
                raise ImportError('No module named bleak')
            return real_import(name, *args, **kwargs)

        with mock.patch('builtins.__import__', side_effect=_no_bleak):
            with self.assertRaises(RuntimeError) as ctx:
                _run(self.provider.start())
        self.assertIn('bleak', str(ctx.exception))

    # ── endpoints advertised to peers ────────────────────────────────────────

    def test_local_endpoints_prefers_configured_url(self):
        config.main['peer_settings'] = {'advertise_url': 'https://1.2.3.4:15678'}
        self.assertEqual(local_endpoints(), ['https://1.2.3.4:15678'])

    def test_local_endpoints_drops_loopback(self):
        """回环地址对 peer 毫无用处，宁可给空清单也别给一个拨不通的地址。"""
        config.main['peer_settings'] = {'advertise_url': ''}
        with mock.patch('peer.discovery.mdns.MdnsProvider._primary_ip',
                        return_value='127.0.0.1'):
            self.assertEqual(local_endpoints(), [])


if __name__ == '__main__':
    unittest.main()
