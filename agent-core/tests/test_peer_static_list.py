"""
test_peer_static_list.py — 手动地址（跨子网发现的兜底）。

mDNS 走链路本地多播，不跨路由 —— 实测同一栋楼里 10.100.121.0/24 与 10.100.128.0/19
之间互相 ping 得通（2ms、HTTPS 200），但彼此发现不到。手动地址是那种情况下唯一的路，
现在从界面可配，所以入参会是人手打的。

规范化因此必须存在：这个列表原来是**原样写入**的，少个协议头（"10.100.121.14:15678"）
会存下一条谁都连不上的 advert，而且没有任何报错 —— peer 只是永远不出现。
"""

import os
import pathlib
import sys
import tempfile
import unittest

import fastapi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api.peer import _normalize_static, _DEFAULT_PEER_PORT  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_bare_host_gets_scheme_and_the_agent_core_port(self):
        """人手打的多半是 IP，不是完整 URL。"""
        self.assertEqual(_normalize_static(['10.100.121.14']),
                         [{'url': f'https://10.100.121.14:{_DEFAULT_PEER_PORT}'}])

    def test_host_with_port_keeps_the_port(self):
        self.assertEqual(_normalize_static(['10.100.121.14:9000']),
                         [{'url': 'https://10.100.121.14:9000'}])

    def test_full_url_is_kept_including_http(self):
        self.assertEqual(_normalize_static(['http://box.local:8080']),
                         [{'url': 'http://box.local:8080'}])

    def test_object_form_keeps_the_display_name(self):
        got = _normalize_static([{'url': '10.0.0.5', 'display_name': 'Tianyi'}])
        self.assertEqual(got, [{'url': f'https://10.0.0.5:{_DEFAULT_PEER_PORT}',
                                'display_name': 'Tianyi'}])

    def test_duplicates_collapse_after_normalisation(self):
        """"10.0.0.5" 和 "https://10.0.0.5:15678" 是同一台机器。"""
        got = _normalize_static(['10.0.0.5', f'https://10.0.0.5:{_DEFAULT_PEER_PORT}'])
        self.assertEqual(len(got), 1)

    def test_blank_entries_are_dropped_not_rejected(self):
        """界面上的空输入不该变成一次报错。"""
        self.assertEqual(_normalize_static(['', '   ', {'url': ''}]), [])

    def test_garbage_is_refused_with_a_usable_message(self):
        for bad in ('http://', 'ftp://box', ':::'):
            with self.assertRaises(fastapi.HTTPException) as cm:
                _normalize_static([bad])
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn('host', cm.exception.detail)

    def test_a_wrong_shaped_entry_is_refused(self):
        with self.assertRaises(fastapi.HTTPException):
            _normalize_static([123])

    def test_order_is_preserved(self):
        got = _normalize_static(['10.0.0.2', '10.0.0.1'])
        self.assertEqual([g['url'].split('//')[1].split(':')[0] for g in got],
                         ['10.0.0.2', '10.0.0.1'])


class TestProviderReadsWhatWeWrite(unittest.TestCase):
    """规范化后的形状必须正是 StaticProvider 会去读的那一种。

    两边各写一套键名，症状是"存进去了但永远发现不到" —— 没有报错，最难查。所以这里
    真的跑一遍 provider 的 refresh()，而不是只比对字段。
    """

    def test_a_normalized_entry_becomes_a_discoverable_advert(self):
        import asyncio
        from unittest import mock

        import config
        from peer.discovery.static import StaticProvider, PROVISIONAL_PREFIX

        entries = _normalize_static([{'url': '10.100.121.14', 'display_name': 'Orin5'}])
        seen = []
        provider = StaticProvider(on_advert=seen.append)

        with mock.patch.object(config, 'main',
                               {'peer_settings': {'discovery': {'static': entries}}}):
            asyncio.run(provider.start())

        self.assertEqual(len(seen), 1, '规范化后的条目没有产出 advert')
        advert = seen[0]
        self.assertEqual(advert.endpoints, [f'https://10.100.121.14:{_DEFAULT_PEER_PORT}'])
        self.assertEqual(advert.display_name, 'Orin5')
        self.assertEqual(advert.source, 'static')
        self.assertTrue(advert.peer_id.startswith(PROVISIONAL_PREFIX),
                        '手动地址还没有指纹，配对完成时才会换成真的')


if __name__ == '__main__':
    unittest.main()
