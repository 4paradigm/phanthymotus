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


class TestStaticAdvertsStayFresh(unittest.IsolatedAsyncioTestCase):
    """静态地址不会过期消失。

    真机上看到的：配置里地址还在、provider 徽章还是绿的、"发现到的机器人"却是空的。
    因为 start() 只 emit 一次，而 registry 会剪掉超过 STALE_AFTER_S(300s) 的 advert ——
    所以静态地址在保存 5 分钟后消失，只有再次重启发现层才回来。配置不是"看见过一次"，
    它不该会过期。mDNS 早前踩过同一个坑并加了刷新循环，静态这条漏了。
    """

    async def test_the_provider_keeps_re_emitting(self):
        import asyncio
        from unittest import mock

        import config
        from peer.discovery import static as static_mod

        entries = _normalize_static(['10.100.121.14'])
        seen = []
        provider = static_mod.StaticProvider(on_advert=seen.append)

        with mock.patch.object(config, 'main',
                               {'peer_settings': {'discovery': {'static': entries}}}), \
             mock.patch.object(static_mod, 'REFRESH_INTERVAL_S', 0.02):
            await provider.start()
            await asyncio.sleep(0.12)
            await provider.stop()

        self.assertGreater(len(seen), 1,
                           'start() 之后再也没有 emit —— 300 秒后就会被剪掉')

    async def test_stop_ends_the_loop(self):
        import asyncio
        from unittest import mock

        import config
        from peer.discovery import static as static_mod

        seen = []
        provider = static_mod.StaticProvider(on_advert=seen.append)
        with mock.patch.object(config, 'main',
                               {'peer_settings': {'discovery': {'static': _normalize_static(['10.0.0.9'])}}}), \
             mock.patch.object(static_mod, 'REFRESH_INTERVAL_S', 0.02):
            await provider.start()
            await asyncio.sleep(0.05)
            await provider.stop()
            after_stop = len(seen)
            await asyncio.sleep(0.08)

        self.assertEqual(len(seen), after_stop, 'stop() 之后还在 emit')

    async def test_a_failing_round_does_not_kill_the_loop(self):
        import asyncio
        from unittest import mock

        from peer.discovery import static as static_mod

        provider = static_mod.StaticProvider(on_advert=lambda a: None)
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError('config unavailable')

        with mock.patch.object(provider, 'refresh', side_effect=boom), \
             mock.patch.object(static_mod, 'REFRESH_INTERVAL_S', 0.02):
            await provider.start()
            await asyncio.sleep(0.09)
            await provider.stop()

        self.assertGreater(len(calls), 2, '一轮异常把循环带走了')


if __name__ == '__main__':
    unittest.main()


class TestPairingAcceptsAProvisionalId(unittest.IsolatedAsyncioTestCase):
    """手动地址必须配得上，而真正的指纹不匹配必须仍然致命。

    真机报错：`peer public_key fingerprint mismatch: advertised
    static:https://10.100.121.14:15678, key hashes to dd398c73…`。手动地址天生只有
    URL 没有指纹，static provider 因此发一个 provisional id（`static:<url>`），
    而 static.py 的注释写着"配对时会换成真指纹" —— 那一步从来没实现，于是 mDNS 覆盖
    不到的那个场合恰好也配不上对。

    放宽只针对 provisional：一个正常发现的 peer 如果公钥哈希不等于它自报的 id，那正是
    这道检查要抓的东西，必须继续报错。
    """

    async def _start(self, advertised_id, remote_key_b64, remote_id):
        from unittest import mock
        import fastapi as _fastapi
        from api import peer as peer_api
        from peer.discovery.base import PeerAdvert

        advert = PeerAdvert(peer_id=advertised_id, display_name='',
                            endpoints=['https://10.100.121.14:15678'], source='static')
        observed, forgotten = [], []

        async def _post(eps, path, payload, **kw):
            return {'nonce': 'n' * 22, 'public_key': remote_key_b64,
                    'display_name': 'Orin5'}, ''

        with mock.patch.object(peer_api.registry, 'get', return_value=advert), \
             mock.patch.object(peer_api.registry, 'endpoints_for',
                               return_value=['https://10.100.121.14:15678']), \
             mock.patch.object(peer_api.registry, 'observe', side_effect=observed.append), \
             mock.patch.object(peer_api.registry, 'forget', side_effect=forgotten.append), \
             mock.patch('peer.transport.post_json', side_effect=_post), \
             mock.patch('peer.identity.fingerprint', return_value=remote_id):
            try:
                out = await peer_api.start_pairing(
                    peer_api.StartPairingReq(peer_id=advertised_id))
            except _fastapi.HTTPException as e:
                return None, e, observed, forgotten
        return out, None, observed, forgotten

    async def test_a_provisional_id_is_replaced_by_the_real_fingerprint(self):
        import base64
        real = 'dd398c73177aa3487e7c695f4b19dfe5'
        key = base64.b64encode(b'k' * 32).decode()
        out, exc, observed, forgotten = await self._start(
            'static:https://10.100.121.14:15678', key, real)
        self.assertIsNone(exc, f'手动地址配对被拒: {exc.detail if exc else ""}')
        self.assertEqual(out['peer_id'], real, '会话必须以真指纹为键')
        self.assertEqual([a.peer_id for a in observed], [real],
                         '真身份没有被重新登记，后续 endpoints_for 会查不到')
        self.assertEqual(forgotten, ['static:https://10.100.121.14:15678'],
                         'provisional 那条没删，同一台机器会重复出现一次且配不上')

    async def test_a_real_mismatch_is_still_fatal(self):
        import base64
        key = base64.b64encode(b'k' * 32).decode()
        out, exc, observed, _ = await self._start(
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', key, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        self.assertIsNone(out)
        self.assertEqual(exc.status_code, 502)
        self.assertIn('mismatch', exc.detail)
        self.assertEqual(observed, [], '不匹配时不该把它登记进 registry')
