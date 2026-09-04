"""
test_peer_identity_pairing.py — P0 身份与配对。

验证节点（按计划）：
1. identity 幂等：ensure_identity() 多次调用返回相同 peer_id
2. 指纹稳定：public_key ↔ peer_id 往返不变
3. ConfigDB 往返：存进 config.main、重启进程后仍能读出
4. pairing SAS：双方算出同一个 6 位短码
5. 公钥篡改后短码不匹配
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

import config  # noqa: E402
from peer import identity, pairing  # noqa: E402


class TestIdentity(unittest.TestCase):
    def setUp(self):
        identity.reset_cache()
        # tempfile DB 每个进程都是全新的，不需要显式清理

    def test_ensure_identity_is_idempotent(self):
        """多次调用返回相同 peer_id（节点 1）"""
        id1 = identity.ensure_identity()
        id2 = identity.ensure_identity()
        self.assertEqual(id1['peer_id'], id2['peer_id'])
        self.assertEqual(id1['private_key'], id2['private_key'])
        self.assertEqual(id1['public_key'], id2['public_key'])

    def test_fingerprint_stable(self):
        """public_key ↔ peer_id 往返不变（节点 2）"""
        ident = identity.ensure_identity()
        raw = identity.public_key_raw()
        self.assertEqual(len(raw), 32)
        recomputed = identity.fingerprint(raw)
        self.assertEqual(recomputed, ident['peer_id'])

    def test_configdb_roundtrip(self):
        """存进 ConfigDB 后仍能读出（节点 3）"""
        id1 = identity.ensure_identity()
        stored = config.main.get('peer_identity')
        self.assertIsNotNone(stored)
        self.assertEqual(stored['peer_id'], id1['peer_id'])

        # 模拟重启：清 cache，从 DB 重读
        identity.reset_cache()
        id2 = identity.ensure_identity()
        self.assertEqual(id1['peer_id'], id2['peer_id'])
        self.assertEqual(id1['private_key'], id2['private_key'])

    def test_sign_and_verify(self):
        """签名与校验"""
        identity.ensure_identity()
        payload = b'test payload'
        sig = identity.sign(payload)
        self.assertTrue(identity.verify(identity.public_key_raw(), sig, payload))
        self.assertFalse(identity.verify(identity.public_key_raw(), sig, b'wrong'))


class TestPairing(unittest.TestCase):
    def test_symmetric_sas(self):
        """双方算出同一个 6 位短码（节点 4）"""
        import base64
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization

        # 两个独立密钥对
        priv_a = ed25519.Ed25519PrivateKey.generate()
        pub_a = priv_a.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        priv_b = ed25519.Ed25519PrivateKey.generate()
        pub_b = priv_b.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        nonce_a = pairing.new_nonce()
        nonce_b = pairing.new_nonce()

        # A 视角
        code_a = pairing.sas_code(pub_a, pub_b, nonce_a, nonce_b)
        # B 视角（参数顺序相反）
        code_b = pairing.sas_code(pub_b, pub_a, nonce_b, nonce_a)

        self.assertEqual(code_a, code_b)
        self.assertEqual(len(code_a), 6)
        self.assertTrue(code_a.isdigit())

    def test_sas_changed_on_key_tamper(self):
        """公钥篡改后短码不匹配（节点 5）"""
        import base64
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization

        priv_a = ed25519.Ed25519PrivateKey.generate()
        pub_a = priv_a.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        priv_b = ed25519.Ed25519PrivateKey.generate()
        pub_b = priv_b.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        nonce_a = pairing.new_nonce()
        nonce_b = pairing.new_nonce()

        code_good = pairing.sas_code(pub_a, pub_b, nonce_a, nonce_b)

        # 篡改一个 bit
        pub_b_tampered = bytearray(pub_b)
        pub_b_tampered[0] ^= 0x01
        code_bad = pairing.sas_code(pub_a, bytes(pub_b_tampered), nonce_a, nonce_b)

        self.assertNotEqual(code_good, code_bad)

    def test_pairing_session_expired(self):
        """过期检测"""
        import time
        pairing.clear()
        sess = pairing.PairingSession(
            peer_id='test_peer',
            peer_public_key=b'\x00' * 32,
            display_name='Test',
            endpoints=['https://test'],
            local_nonce='ln',
            remote_nonce='rn',
            local_public_key=b'\x01' * 32,
        )
        self.assertFalse(sess.expired)
        sess.created_at = time.time() - 400
        self.assertTrue(sess.expired)


if __name__ == '__main__':
    unittest.main()
