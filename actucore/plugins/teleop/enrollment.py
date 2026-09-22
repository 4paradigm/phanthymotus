"""Bounded, two-sided enrollment; discovery never grants trust or motion authority."""
import asyncio
import base64
import json
import hashlib
import hmac
import secrets
import time
import uuid
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding, load_der_public_key
from cryptography.hazmat.primitives.asymmetric import ec

from .capture import CaptureError


class Enrollment:
    def __init__(self, capture, certificate_base64, clock=time.monotonic, public_wss_url=None):
        self.capture, self.clock = capture, clock
        cert = x509.load_pem_x509_certificate(base64.b64decode(certificate_base64))
        self.certificate = cert.public_bytes(Encoding.DER)
        self.device_id = hashlib.sha256(self.certificate).hexdigest()
        self.deadline = 0
        self.pending = None
        self.last_request = -float('inf')
        self.public_wss_url = public_wss_url
        self.invitation = None
        self._invite_lock = asyncio.Lock()

    def status(self):
        now = self.clock()
        if self.invitation and now >= self.invitation['deadline']:
            self.invitation = None
        if self.pending and now >= self.pending['deadline']:
            self.pending = None
        return {'invitation': ({'invitation_id': self.invitation['invitation_id'],
                    'expires_in_seconds': max(0, int(self.invitation['deadline']-now))}
                    if self.invitation else None), 'device_id': self.device_id, 'window_open': now < self.deadline,
                'expires_in_seconds': max(0, int(self.deadline-now)),
                'pending': ({k: self.pending[k] for k in
                    ('request_id', 'device_name', 'fingerprint', 'approved', 'confirmed')}
                    if self.pending else None)}

    async def open(self):
        self.revoke_invitation()
        if (await self.capture.status())['paired_devices']:
            raise CaptureError('revoke_existing_headset_first', status=409)
        self.pending = None
        self.deadline = self.clock()+120
        return self.status()

    async def request(self, data):
        self.status()
        if self.clock() >= self.deadline:
            raise CaptureError('pairing_window_closed', status=403)
        if self.pending or self.clock()-self.last_request < 2:
            raise CaptureError('pairing_busy', status=429)
        self.last_request = self.clock()
        if (await self.capture.status())['paired_devices']:
            raise CaptureError('revoke_existing_headset_first', status=409)
        try:
            if set(data) != {'device_name', 'public_key', 'nonce'}:
                raise ValueError()
            name = data['device_name']
            if not isinstance(name, str) or not 1 <= len(name) <= 64 or any(ord(c)<32 for c in name):
                raise ValueError()
            key = base64.b64decode(data['public_key'], validate=True)
            public = load_der_public_key(key)
            if not isinstance(public, ec.EllipticCurvePublicKey) or not isinstance(public.curve, ec.SECP256R1):
                raise ValueError()
            nonce = base64.b64decode(data['nonce'], validate=True)
            if len(nonce) != 32 or len(key) > 256:
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            raise CaptureError('pairing_request_invalid') from None
        server_nonce = secrets.token_bytes(32)
        # Fixed-width hashes keep this transcript unambiguous across Java/Python.
        transcript = b'motus-enrollment-v1\0'+hashlib.sha256(self.certificate).digest()+hashlib.sha256(key).digest()+nonce+server_nonce
        fingerprint = hashlib.sha256(transcript).hexdigest()[:32].upper()
        ticket = secrets.token_urlsafe(32)
        self.pending = dict(request_id=str(uuid.uuid4()), device_name=name,
            fingerprint=fingerprint, ticket_digest=hashlib.sha256(ticket.encode()).digest(),
            deadline=self.deadline, approved=False, confirmed=False)
        return {'request_id': self.pending['request_id'], 'ticket': ticket,
                'server_nonce': base64.b64encode(server_nonce).decode(),
                'fingerprint': fingerprint, 'expires_in_seconds': self.status()['expires_in_seconds']}

    def decide(self, request_id, fingerprint, approve):
        self.status()
        p = self.pending
        if not p or p['request_id'] != request_id or p['fingerprint'] != fingerprint:
            raise CaptureError('pairing_request_changed', status=409)
        if approve:
            p['approved'] = True
        else:
            self.pending = None
            self.deadline = 0
        return self.status()

    async def poll(self, data):
        self.status()
        p = self.pending
        ticket = data.get('ticket')
        if (not p or not isinstance(ticket, str) or len(ticket)>128 or
            data.get('request_id') != p['request_id'] or
            not hmac.compare_digest(hashlib.sha256(ticket.encode()).digest(), p['ticket_digest'])):
            raise CaptureError('pairing_request_expired', status=403)
        if data.get('confirm') is True:
            if data.get('fingerprint') != p['fingerprint']:
                raise CaptureError('pairing_fingerprint_mismatch', status=403)
            p['confirmed'] = True
        if p['approved'] and p['confirmed']:
            result = await self.capture.create_pairing()
            self.pending = None
            self.deadline = 0
            return {'state': 'approved', **result}
        return {'state': 'pending', 'approved': p['approved'], 'confirmed': p['confirmed']}

    async def create_invitation(self):
        """Called only by the authenticated management card; grants enrollment, never motion."""
        async with self._invite_lock:
            if (await self.capture.status())['paired_devices']:
                raise CaptureError('revoke_existing_headset_first', status=409)
            url = urlsplit(self.public_wss_url or '')
            if url.scheme != 'wss' or not url.hostname or not url.port:
                raise CaptureError('onboarding_endpoint_unavailable', status=503)
            token = secrets.token_urlsafe(32)
            self.pending = None
            self.deadline = 0
            self.invitation = {'invitation_id': str(uuid.uuid4()),
                'digest': hashlib.sha256(token.encode()).digest(), 'deadline': self.clock()+900}
            payload = {'schema': 'motus.teleop.invitation.v1',
                'invitation_id': self.invitation['invitation_id'], 'token': token,
                'device_id': self.device_id, 'certificate_sha256': self.device_id,
                'endpoint': url.netloc}
            encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode().rstrip('=')
            return {**payload, 'expires_in_seconds': 900,
                'deep_link': 'motus-teleop://connect#'+encoded}

    def revoke_invitation(self):
        self.invitation = None
        return {'revoked': True}

    async def redeem_invitation(self, data):
        async with self._invite_lock:
            self.status()
            invitation = self.invitation
            if not isinstance(data, dict) or set(data) != {'invitation_id', 'token', 'device_id', 'device_name'}:
                raise CaptureError('invitation_request_invalid')
            token, name = data.get('token'), data.get('device_name')
            if not isinstance(token, str) or not 32 <= len(token) <= 128 or not isinstance(name, str) or not 1 <= len(name) <= 64 or any(ord(c)<32 for c in name):
                raise CaptureError('invitation_request_invalid')
            if (not invitation or data['invitation_id'] != invitation['invitation_id']
                    or data['device_id'] != self.device_id
                    or not hmac.compare_digest(hashlib.sha256(token.encode()).digest(), invitation['digest'])):
                raise CaptureError('invitation_expired_or_invalid', status=403)
            # Consume before yielding: parallel requests can never mint two credentials.
            self.invitation = None
            # CaptureManager checks existing paired devices under the same lock as
            # its WebSocket pair/credential handshake. A separate status precheck
            # here cannot enforce this race boundary and must not replace that gate.
            result = await self.capture.create_pairing()
            return {'state': 'approved', 'device_id': self.device_id, **result}
