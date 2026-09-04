"""
test_peer_transport.py — 签名传输与重放防护。

验证节点（按计划）：
7. 重放签名被拒：同一个 nonce 第二次校验失败
8. 过期签名被拒：时间窗外的签名校验失败
"""

import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

import config  # noqa: E402
from peer import identity, store, transport  # noqa: E402


class TestTransport(unittest.TestCase):
    def setUp(self):
        identity.reset_cache()
        transport.clear_nonces()
        # 把本机配置为一个 peer（测试用，实际场景里自己不在 peers 表）
        identity.ensure_identity()
        my_id = identity.peer_id()
        my_pubkey = identity.public_key_b64()
        store.upsert(my_id, my_pubkey, 'Self', role='viewer')

    def test_replay_rejected(self):
        """重放签名被拒（节点 7）"""
        method = 'POST'
        path = '/api/peer/inbox/ping'
        body = b'{"test": true}'
        headers = transport.sign_headers(method, path, body)

        # 第一次校验通过
        peer_id, reason = transport.verify_signed_request(method, path, headers, body)
        self.assertEqual(peer_id, identity.peer_id())
        self.assertEqual(reason, '')

        # 重放同一个 nonce
        peer_id2, reason2 = transport.verify_signed_request(method, path, headers, body)
        self.assertEqual(peer_id2, '')
        self.assertEqual(reason2, 'replayed_nonce')

    def test_expired_timestamp_rejected(self):
        """过期签名被拒（节点 8）"""
        method = 'POST'
        path = '/api/peer/inbox/ping'
        body = b'{"test": true}'
        from peer.pairing import new_nonce
        nonce = new_nonce()
        ts = str(int(time.time() - 200))  # 200s 前，超出默认 120s 窗口
        payload = transport.canonical_payload(method, path, ts, nonce, body)
        sig = identity.sign(payload)

        headers = {
            transport.HEADER_PEER_ID: identity.peer_id(),
            transport.HEADER_TIMESTAMP: ts,
            transport.HEADER_NONCE: nonce,
            transport.HEADER_SIGNATURE: sig,
        }
        peer_id, reason = transport.verify_signed_request(method, path, headers, body)
        self.assertEqual(peer_id, '')
        self.assertEqual(reason, 'timestamp_outside_window')

    def test_bad_signature_rejected(self):
        """签名不匹配时拒绝"""
        method = 'POST'
        path = '/api/peer/inbox/ping'
        body = b'{"test": true}'
        headers = transport.sign_headers(method, path, body)
        # 篡改 body
        bad_body = b'{"test": false}'
        peer_id, reason = transport.verify_signed_request(method, path, headers, bad_body)
        self.assertEqual(peer_id, '')
        self.assertEqual(reason, 'bad_signature')

    def test_unknown_peer_rejected(self):
        """未配对的 peer 被拒（require_paired=True）"""
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
        import base64

        # 生成一个新密钥，不在 peers 表里
        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        pub_b64 = base64.b64encode(pub).decode()
        unknown_peer_id = identity.fingerprint(pub)

        method = 'POST'
        path = '/api/peer/inbox/ping'
        body = b'{}'
        from peer.pairing import new_nonce
        nonce = new_nonce()
        ts = str(int(time.time()))
        payload = transport.canonical_payload(method, path, ts, nonce, body)
        sig_raw = priv.sign(payload)
        sig = base64.b64encode(sig_raw).decode()

        headers = {
            transport.HEADER_PEER_ID: unknown_peer_id,
            transport.HEADER_TIMESTAMP: ts,
            transport.HEADER_NONCE: nonce,
            transport.HEADER_SIGNATURE: sig,
        }
        peer_id, reason = transport.verify_signed_request(
            method, path, headers, body, require_paired=True
        )
        self.assertEqual(peer_id, '')
        self.assertEqual(reason, 'unknown_peer')

    def test_unpaired_with_consistent_key(self):
        """require_paired=False 时，自洽的签名通过（配对握手场景）"""
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
        import base64

        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        pub_b64 = base64.b64encode(pub).decode()
        peer_id = identity.fingerprint(pub)

        method = 'POST'
        path = '/api/peer/inbox/pair_request'
        body = b'{}'
        from peer.pairing import new_nonce
        nonce = new_nonce()
        ts = str(int(time.time()))
        payload = transport.canonical_payload(method, path, ts, nonce, body)
        sig_raw = priv.sign(payload)
        sig = base64.b64encode(sig_raw).decode()

        headers = {
            transport.HEADER_PEER_ID: peer_id,
            transport.HEADER_TIMESTAMP: ts,
            transport.HEADER_NONCE: nonce,
            transport.HEADER_SIGNATURE: sig,
            'x-motus-public-key': pub_b64,
        }
        verified_id, reason = transport.verify_signed_request(
            method, path, headers, body, require_paired=False
        )
        self.assertEqual(verified_id, peer_id)
        self.assertEqual(reason, '')


if __name__ == '__main__':
    unittest.main()
