import assert from 'node:assert/strict';

import {
  captureAssignmentForSession,
  captureAttachEligibility,
  captureBootstrapFields,
} from '../../web/js/teleop/capture-console.js';

const DIGEST = '0123456789abcdef'.repeat(4);
const readySession = {
  id: 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3',
  owned_by_client: true,
  state: 'active',
  mode: 'shadow',
  live_confirmed: true,
  profile_id: 'recording',
  capability_digest: DIGEST,
};
const readyCapture = {
  id: '79def5a3-85e9-48b0-9220-8e730e2944c1',
  connected: true,
  observed_state: 'xr_standby',
  capture_protocol: 'motus.teleop.capture.v1',
  frame_protocol: 'motus.teleop.rtc-frame.v1',
  assignment: null,
};

assert.deepEqual(
  captureAttachEligibility(readySession, readyCapture),
  { allowed: true, code: 'ready' },
);
assert.deepEqual(
  captureAttachEligibility(
    {
      ...readySession,
      mode: 'live',
      state: 'awaiting_confirmation',
      live_confirmed: false,
    },
    readyCapture,
  ),
  { allowed: false, code: 'capture_live_confirmation_required' },
);
assert.deepEqual(
  captureAttachEligibility(
    { ...readySession, mode: 'live', live_confirmed: false },
    readyCapture,
  ),
  { allowed: false, code: 'capture_live_confirmation_required' },
);
assert.deepEqual(
  captureAttachEligibility(readySession, { ...readyCapture, observed_state: 'streaming' }),
  { allowed: false, code: 'capture_not_ready' },
);
assert.deepEqual(
  captureAttachEligibility(readySession, {
    ...readyCapture,
    assignment: { session_id: readySession.id },
  }),
  { allowed: false, code: 'capture_assignment_conflict' },
);

const bootstrap = captureBootstrapFields({
  websocket_path: '/ws/teleop-capture',
  pairing_id: '9878dc06-cfa9-40f2-9739-62db01165ce9',
  pairing_code: 'one-time-secret-that-is-never-stored',
  ca_certificate_base64: 'TUFNQ0FRRT0=',
  expires_at: 1_800_000_000,
}, { host: 'core.lab.example:15678' });
assert.equal(bootstrap.coreWssUrl, 'wss://core.lab.example:15678/ws/teleop-capture');
assert.equal(bootstrap.pairingCode, 'one-time-secret-that-is-never-stored');
assert.throws(() => captureBootstrapFields({
  websocket_path: '/ws/teleop-capture?pairing_code=secret',
  pairing_id: 'x',
  pairing_code: 'y',
  ca_certificate_base64: 'TUFNQ0FRRT0=',
}, { host: 'core.lab.example' }));

const assignment = { session_id: readySession.id, state: 'negotiated' };
assert.equal(
  captureAssignmentForSession([{ ...readyCapture, assignment }], readySession.id),
  assignment,
);
assert.equal(captureAssignmentForSession([], readySession.id), null);

console.log('capture-console contract tests passed');
