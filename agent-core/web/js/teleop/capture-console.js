const CAPTURE_PROTOCOL = 'motus.teleop.capture.v1';
const FRAME_PROTOCOL = 'motus.teleop.rtc-frame.v1';
const SAFE_CAPTURE_PATH = '/ws/teleop-capture';
const BASE64_PATTERN = /^[A-Za-z0-9+/]+={0,2}$/;

function stringField(value) {
  return typeof value === 'string' ? value : '';
}

export function captureBootstrapFields(pairing, locationLike) {
  const path = stringField(pairing?.websocket_path);
  const host = stringField(locationLike?.host);
  const pairingId = stringField(pairing?.pairing_id);
  const pairingCode = stringField(pairing?.pairing_code);
  const caCertificateBase64 = stringField(pairing?.ca_certificate_base64);
  const expiresAt = Number(pairing?.expires_at);
  if (
    path !== SAFE_CAPTURE_PATH
    || !host
    || host.includes('/')
    || !pairingId
    || pairingCode.length < 32
    || pairingCode.length > 128
    || caCertificateBase64.length === 0
    || caCertificateBase64.length > 48 * 1024
    || caCertificateBase64.length % 4 !== 0
    || !BASE64_PATTERN.test(caCertificateBase64)
    || !Number.isFinite(expiresAt)
    || expiresAt <= 0
  ) throw new Error('capture bootstrap response is invalid');
  return {
    coreWssUrl: `wss://${host}${path}`,
    pairingId,
    pairingCode,
    caCertificateBase64,
    expiresAt,
  };
}

export function captureAttachEligibility(session, capture) {
  if (!session || typeof session !== 'object') {
    return { allowed: false, code: 'capture_session_required' };
  }
  if (session.owned_by_client !== true) {
    return { allowed: false, code: 'session_client_mismatch' };
  }
  if (session.state === 'awaiting_confirmation') {
    return { allowed: false, code: 'capture_live_confirmation_required' };
  }
  if (session.state !== 'active') {
    return { allowed: false, code: 'capture_session_not_active' };
  }
  if (session.mode === 'live' && session.live_confirmed !== true) {
    return { allowed: false, code: 'capture_live_confirmation_required' };
  }
  if (!['shadow', 'live'].includes(session.mode)) {
    return { allowed: false, code: 'capture_session_contract_invalid' };
  }
  if (
    typeof session.profile_id !== 'string'
    || session.profile_id.length === 0
    || !/^[0-9a-f]{64}$/.test(stringField(session.capability_digest))
  ) return { allowed: false, code: 'capture_session_contract_invalid' };
  if (!capture || typeof capture !== 'object') {
    return { allowed: false, code: 'capture_device_required' };
  }
  if (capture.capture_protocol !== CAPTURE_PROTOCOL || capture.frame_protocol !== FRAME_PROTOCOL) {
    return { allowed: false, code: 'capture_protocol_mismatch' };
  }
  if (capture.connected !== true || capture.observed_state !== 'xr_standby') {
    return { allowed: false, code: 'capture_not_ready' };
  }
  if (capture.assignment !== null && capture.assignment !== undefined) {
    return { allowed: false, code: 'capture_assignment_conflict' };
  }
  return { allowed: true, code: 'ready' };
}

export function captureAssignmentForSession(captures, sessionId) {
  if (!Array.isArray(captures) || !sessionId) return null;
  for (const capture of captures) {
    if (capture?.assignment?.session_id === sessionId) return capture.assignment;
  }
  return null;
}

export const CAPTURE_CONSOLE_PROTOCOLS = Object.freeze({
  capture: CAPTURE_PROTOCOL,
  frame: FRAME_PROTOCOL,
});
