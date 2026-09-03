"""
test_peer_pair_handshake.py — 配对握手的双端一致性。

存在的原因：真机联调时发现，发起方拿到了短码 002136，接收方
`/api/peer/pair/active` 却是空的 —— 接收端的 inbox_pair_request 生成 nonce、
回给对方、然后就丢掉了，既没保存自己的 nonce，也没建 session。

后果不是"少个 UI"，而是 SAS 整个失效：它的全部安全性就来自「人在两块屏幕上
比对同一串数字」。只有一端能看到码时，配对退化成"谁先发起谁说了算"，中间人
攻击完全不受阻拦。

单元测试当初测的是 sas_code() 这个纯函数（对称性没问题），但没有人测过
「两端各自跑完握手后，是否真的都持有同一个码」。
"""

import base64
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

from peer import identity, pairing  # noqa: E402


def _keypair():
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv, pub


class TestPairHandshake(unittest.TestCase):
    def setUp(self):
        identity.reset_cache()
        identity.ensure_identity()
        pairing.clear()

    def test_both_sides_derive_the_same_code(self):
        """双方各自建 session 后，必须持有同一个短码。

        模拟真实握手：发起方生成 nonce_a 发出去，接收方生成 nonce_b 并
        *保存自己的 session*，双方各用「本地私钥 + 对端公钥 + 两个 nonce」
        独立推导。
        """
        _, pub_a = _keypair()
        _, pub_b = _keypair()
        nonce_a = pairing.new_nonce()
        nonce_b = pairing.new_nonce()

        # 发起方（A）视角
        sess_a = pairing.PairingSession(
            peer_id=identity.fingerprint(pub_b), peer_public_key=pub_b,
            display_name='B', endpoints=[], local_nonce=nonce_a,
            remote_nonce=nonce_b, local_public_key=pub_a,
        )
        # 接收方（B）视角 —— 本地/远端完全颠倒
        sess_b = pairing.PairingSession(
            peer_id=identity.fingerprint(pub_a), peer_public_key=pub_a,
            display_name='A', endpoints=[], local_nonce=nonce_b,
            remote_nonce=nonce_a, local_public_key=pub_b,
        )

        self.assertEqual(sess_a.code, sess_b.code,
                         'two ends derived different SAS codes — nothing to compare')
        self.assertEqual(len(sess_a.code), 6)

    def test_receiver_stores_a_session(self):
        """接收方必须把 session 存下来，否则操作员无从确认。

        直接断言 put/get 语义 —— 这正是 inbox_pair_request 漏掉的那一步。
        """
        _, pub_peer = _keypair()
        _, pub_local = _keypair()
        peer_id = identity.fingerprint(pub_peer)
        sess = pairing.PairingSession(
            peer_id=peer_id, peer_public_key=pub_peer, display_name='peer',
            endpoints=[], local_nonce=pairing.new_nonce(),
            remote_nonce=pairing.new_nonce(), local_public_key=pub_local,
        )
        pairing.put(sess)

        self.assertIsNotNone(pairing.get(peer_id))
        self.assertIn(peer_id, [s['peer_id'] for s in pairing.active()],
                      'session not visible to the dashboard')

    def test_mitm_produces_different_codes(self):
        """中间人各持一把不同的密钥时，两端的码必须不同。

        这就是人工比对能挡住攻击的原因；如果这条不成立，比对毫无意义。
        """
        _, pub_a = _keypair()
        _, pub_b = _keypair()
        _, pub_evil = _keypair()
        nonce_a, nonce_b = pairing.new_nonce(), pairing.new_nonce()

        # A 以为在和 B 配对，实际拿到的是攻击者的公钥
        code_a = pairing.sas_code(pub_a, pub_evil, nonce_a, nonce_b)
        # B 那边看到的是真实的 A ↔ B 组合
        code_b = pairing.sas_code(pub_b, pub_a, nonce_b, nonce_a)

        self.assertNotEqual(code_a, code_b,
                            'MITM produced matching codes — SAS gives no protection')

    def test_display_name_helpers_never_empty(self):
        """两端展示名不能为空 —— 光看指纹没法判断在跟哪台机器配对。"""
        import api.peer as ap
        self.assertTrue(ap._local_display_name(),
                        'local display name empty; operator sees a bare fingerprint')
        self.assertTrue(ap._peer_label('8bd2bf8fe8f872361f8f3212dc7e4279'))


if __name__ == '__main__':
    unittest.main()
