import {
  MAX_SAFE_WIRE_INTEGER,
  RTC_FRAME_PROTOCOL,
  buildRtcFrameV1 as buildConfiguredRtcFrameV1,
  createRtcFrameGenerator,
  createRtcFrameState,
  isDualSqueezeDeadmanRequested,
  markRtcFrameDeadmanReleased,
  performanceMillisecondsToNanoseconds,
  xrPoseToWirePose,
  xrStandardInputToWireController,
} from '../../web/js/teleop/webxr-frame.js';

const BASE_CAPABILITIES = {
  profile_id: 'fixture_base_profile_v1',
  input_bindings: {
    base_twist: {
      linear_x: {
        hand: 'left', axis: 3, scale: 0.5, deadzone: 0.2, direction: -1,
      },
      linear_y: {
        hand: 'left', axis: 2, scale: 0.3, deadzone: 0.2, direction: -1,
      },
      angular_z: {
        hand: 'right', axis: 2, scale: 0.6, deadzone: 0.2, direction: -1,
      },
    },
  },
  outputs: {
    base: { enabled: true },
    hands: { enabled: false },
  },
  effectors: ['base'],
};

const BASE_FRAME_CONFIGURATION = {
  mode: 'shadow',
  capabilities: BASE_CAPABILITIES,
};

const ARM_ONLY_FRAME_CONFIGURATION = {
  mode: 'live',
  capabilities: {
    profile_id: 'dual_arm_fixture_v1',
    input_bindings: {
      head: { required: true, role: 'reference' },
      left_controller: { required: true, role: 'left_end_effector' },
      right_controller: { required: true, role: 'right_end_effector' },
    },
    outputs: {
      dual_arm: { enabled: true, joint_count: 10 },
      base: { enabled: false },
      hands: { enabled: false },
    },
    effectors: ['dual_arm'],
  },
};

function buildRtcFrameV1(state, sample = {}, configuration = BASE_FRAME_CONFIGURATION) {
  return buildConfiguredRtcFrameV1(state, sample, configuration);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertEqual(actual, expected, message) {
  const left = JSON.stringify(actual);
  const right = JSON.stringify(expected);
  assert(left === right, `${message}: expected ${right}, got ${left}`);
}

function assertNear(actual, expected, message, tolerance = 1e-12) {
  assert(Math.abs(actual - expected) <= tolerance,
    `${message}: expected ${expected}, got ${actual}`);
}

function assertThrows(callback, message) {
  let threw = false;
  try {
    callback();
  } catch {
    threw = true;
  }
  assert(threw, message);
}

function pose({
  position = [0.1, 1.2, -0.3],
  orientation = [0, 0, 0, 0.999],
} = {}) {
  return {
    transform: {
      position: { x: position[0], y: position[1], z: position[2] },
      orientation: {
        x: orientation[0],
        y: orientation[1],
        z: orientation[2],
        w: orientation[3],
      },
    },
  };
}

function button(value, pressed = value >= 0.75) {
  return { value, pressed, touched: pressed };
}

function inputSource({
  handedness = 'left',
  targetRayMode = 'tracked-pointer',
  gripSpace = {},
  mapping = 'xr-standard',
  axes = [0, 0, 0, 0],
  buttons = [button(0, false), button(0.9, true)],
} = {}) {
  return {
    handedness,
    targetRayMode,
    gripSpace,
    gamepad: { mapping, axes, buttons },
  };
}

function validSample(overrides = {}) {
  return {
    monotonicTimeMs: 12.345678,
    headPose: pose(),
    leftGripPose: pose({ position: [-0.2, 1.1, -0.4] }),
    rightGripPose: pose({ position: [0.2, 1.1, -0.4] }),
    leftInputSource: inputSource(),
    rightInputSource: inputSource({ handedness: 'right' }),
    deadman: false,
    ...overrides,
  };
}

function assertNoPrivateAuthority(value) {
  const forbidden = new Set(['boot_id', 'session_id', 'epoch', 'fence']);
  if (Array.isArray(value)) {
    value.forEach(assertNoPrivateAuthority);
    return;
  }
  if (!value || typeof value !== 'object') return;
  for (const [key, nested] of Object.entries(value)) {
    assert(!forbidden.has(key), `public RTC frame disclosed ${key}`);
    assertNoPrivateAuthority(nested);
  }
}

assert(RTC_FRAME_PROTOCOL === 'motus.teleop.rtc-frame.v1', 'protocol name must match Driver');

// Exact public frame schema, normalized poses, default-off deadman, and no private identity.
let result = buildRtcFrameV1(createRtcFrameState(), validSample());
assertEqual(Object.keys(result.frame).sort(), [
  'schema_version',
  'sequence',
  'client_monotonic_ns',
  'mode',
  'deadman',
  'clutch_sequence',
  'tracking',
  'head',
  'left_controller',
  'right_controller',
  'controllers',
  'base_twist',
].sort(), 'frame must contain exactly the Driver public RTC fields');
assertNoPrivateAuthority(result.frame);
assert(result.frame.schema_version === 1, 'schema_version must be 1');
assert(result.frame.mode === 'shadow', 'mode must be shadow');
assert(result.frame.sequence === 0, 'first sequence must be zero');
assert(result.frame.deadman === false, 'deadman must default false');
assert(result.frame.clutch_sequence === 0, 'default-off deadman must not advance clutch');
assertEqual(result.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'default-off deadman must emit an explicit six-dimensional zero base_twist');
assertEqual(result.frame.tracking, {
  head: true,
  left_controller: true,
  right_controller: true,
}, 'valid poses must have matching tracking flags');
assert(Math.abs(Math.hypot(...result.frame.head.orientation) - 1) < 1e-12,
  'wire quaternion must be strictly normalized');

const armOnlyLive = buildRtcFrameV1(
  createRtcFrameState({ rearmRequired: false }),
  validSample({ deadman: true }),
  ARM_ONLY_FRAME_CONFIGURATION,
);
assert(armOnlyLive.frame.mode === 'live', 'frame mode must come from the live session contract');
assert(!Object.hasOwn(armOnlyLive.frame, 'base_twist'),
  'base-disabled capabilities must omit base_twist instead of sending a fixed zero output');
assert(!Object.hasOwn(armOnlyLive.frame, 'hands'),
  'hands-disabled capabilities must not invent a hands output field');
assertThrows(
  () => buildConfiguredRtcFrameV1(
    createRtcFrameState(),
    validSample(),
    { ...BASE_FRAME_CONFIGURATION, mode: 'invalid' },
  ),
  'an RTC frame without an exact session mode must fail closed',
);
assertThrows(
  () => buildConfiguredRtcFrameV1(
    createRtcFrameState(),
    validSample(),
    {
      mode: 'live',
      capabilities: {
        ...ARM_ONLY_FRAME_CONFIGURATION.capabilities,
        input_bindings: BASE_CAPABILITIES.input_bindings,
      },
    },
  ),
  'a base binding cannot bypass capabilities.outputs.base.enabled=false',
);

// Explicit dual-squeeze policy is centralized and fails closed.
const left = inputSource();
const right = inputSource({ handedness: 'right' });
const releasedLeft = inputSource({ buttons: [button(0, false), button(0, false)] });
const releasedRight = inputSource({
  handedness: 'right',
  buttons: [button(0, false), button(0, false)],
});
assert(isDualSqueezeDeadmanRequested(left, right), 'both valid squeezes must request deadman');
assert(!isDualSqueezeDeadmanRequested(
  left,
  inputSource({
    handedness: 'right',
    buttons: [button(0, false), button(0.9, false)],
  }),
), 'pressed=false must reject squeeze even at a high value');
assert(!isDualSqueezeDeadmanRequested(
  left,
  inputSource({
    handedness: 'right',
    buttons: [button(0, false), button(Number.NaN, true)],
  }),
), 'non-finite squeeze must fail closed');
assert(!isDualSqueezeDeadmanRequested(left, inputSource({ handedness: 'right', mapping: '' })),
  'non-xr-standard mapping must fail closed');
assert(!isDualSqueezeDeadmanRequested(left, inputSource()),
  'duplicate left-handed sources must fail closed');
assert(!isDualSqueezeDeadmanRequested(left, inputSource({
  handedness: 'right',
  targetRayMode: 'gaze',
})), 'non tracked-pointer sources must fail closed');
assert(!isDualSqueezeDeadmanRequested(left, new Proxy({}, {
  get() { throw new Error('malformed input source'); },
})), 'throwing input source must fail closed');
const rejectedDeadman = buildRtcFrameV1(result.state, validSample({
  leftInputSource: inputSource({ buttons: [button(0, false), button(0.5, true)] }),
  deadman: true,
}));
assert(rejectedDeadman.frame.deadman === false,
  'frame reducer must independently reject deadman without both squeeze thresholds');
assert(rejectedDeadman.state.rearmRequired === true,
  'pressed/value disagreement must not masquerade as a physical release');
const partialSqueeze = buildRtcFrameV1(
  createRtcFrameState({ rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ buttons: [button(0, false), button(0.5, true)] }),
    rightInputSource: inputSource({
      handedness: 'right',
      buttons: [button(0, false), button(0.5, true)],
    }),
    deadman: false,
  }),
);
assert(partialSqueeze.frame.deadman === false && partialSqueeze.state.rearmRequired === false,
  'a gradual squeeze below threshold must not engage or erase a prior valid rearm');

// Sequence is strictly increasing; clutch advances only on safe false -> true edges.
const requested = isDualSqueezeDeadmanRequested(left, right);
const armed = buildRtcFrameV1(result.state, validSample({
  leftInputSource: releasedLeft,
  rightInputSource: releasedRight,
  deadman: false,
}));
assert(armed.frame.deadman === false && armed.state.rearmRequired === false,
  'a valid observed squeeze release must arm the next deliberate engagement');
result = buildRtcFrameV1(armed.state, validSample({ deadman: requested }));
assert(result.frame.sequence === 2, 'sequence must advance on every frame');
assert(result.frame.deadman === true, 'explicit valid request must engage deadman');
assert(result.frame.clutch_sequence === 1, 'first engagement must advance clutch');
const held = buildRtcFrameV1(result.state, validSample({
  monotonicTimeMs: 12.345678,
  deadman: requested,
}));
assert(held.frame.sequence === 3, 'held frame sequence must advance');
assert(held.frame.clutch_sequence === 1, 'held deadman must not advance clutch twice');
assert(held.frame.client_monotonic_ns > result.frame.client_monotonic_ns,
  'equal performance timestamps must still produce increasing integer nanoseconds');
const open = buildRtcFrameV1(held.state, validSample({
  leftInputSource: releasedLeft,
  rightInputSource: releasedRight,
  deadman: false,
}));
const reengaged = buildRtcFrameV1(open.state, validSample({ deadman: true }));
assert(open.frame.clutch_sequence === 1 && reengaged.frame.clutch_sequence === 2,
  'release then engagement must advance clutch exactly once');

// RTC/XR reconnect persists the stream counters and explicitly clears the edge detector.
const persisted = markRtcFrameDeadmanReleased(reengaged.state);
assert(persisted.nextSequence === reengaged.state.nextSequence,
  'reconnect must preserve next sequence');
assert(persisted.clutchSequence === reengaged.state.clutchSequence,
  'reconnect must preserve clutch generation');
assert(persisted.lastMonotonicNs === reengaged.state.lastMonotonicNs,
  'reconnect must preserve monotonic watermark');
assert(persisted.deadmanActive === false, 'stream end must explicitly release deadman edge state');
assert(persisted.rearmRequired === true, 'stream end must require an observed physical release');
const heldAcrossReconnect = buildRtcFrameV1(persisted, validSample({
  leftInputSource: inputSource({ axes: [0, 0, -1, -1] }),
  rightInputSource: inputSource({ handedness: 'right', axes: [0, 0, -1, 0] }),
  deadman: true,
}));
assert(heldAcrossReconnect.frame.sequence === reengaged.state.nextSequence,
  'reconnected RTC stream must continue rather than reset sequence');
assert(heldAcrossReconnect.frame.deadman === false
  && heldAcrossReconnect.frame.clutch_sequence === reengaged.state.clutchSequence,
  'held squeezes across reconnect must not automatically recover motion');
assertEqual(heldAcrossReconnect.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'rearm latch must keep base_twist strictly zero');
const reconnectReleasedWithHeldSticks = buildRtcFrameV1(
  heldAcrossReconnect.state,
  validSample({
    leftInputSource: inputSource({
      axes: [0, 0, -1, -1],
      buttons: [button(0, false), button(0, false)],
    }),
    rightInputSource: inputSource({
      handedness: 'right',
      axes: [0, 0, -1, 0],
      buttons: [button(0, false), button(0, false)],
    }),
    deadman: false,
  }),
);
assert(reconnectReleasedWithHeldSticks.state.rearmRequired === true,
  'squeeze release with locomotion sticks held must remain re-arm latched');
const reconnectNeutralBeforePreload = buildRtcFrameV1(
  reconnectReleasedWithHeldSticks.state,
  validSample({
    leftInputSource: inputSource({
      axes: [0, 0, 0, 0],
      buttons: [button(0, false), button(0, false)],
    }),
    rightInputSource: inputSource({
      handedness: 'right',
      axes: [0, 0, 0, 0],
      buttons: [button(0, false), button(0, false)],
    }),
    deadman: false,
  }),
);
assert(reconnectNeutralBeforePreload.state.rearmRequired === false,
  'neutral squeeze release must clear the re-arm latch before the preload test');
const reconnectAttemptedAtFullScale = buildRtcFrameV1(
  reconnectNeutralBeforePreload.state,
  validSample({
    leftInputSource: inputSource({ axes: [0, 0, -1, -1] }),
    rightInputSource: inputSource({ handedness: 'right', axes: [0, 0, -1, 0] }),
    deadman: true,
  }),
);
assert(reconnectAttemptedAtFullScale.frame.deadman === false
  && reconnectAttemptedAtFullScale.state.rearmRequired === true,
  'first re-squeeze with non-neutral sticks must not preload a motion command');
assertEqual(reconnectAttemptedAtFullScale.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'non-neutral engagement attempt must remain six-dimensional zero');
const reconnectReleased = buildRtcFrameV1(reconnectAttemptedAtFullScale.state, validSample({
  leftInputSource: inputSource({
    axes: [0, 0, 0, 0],
    buttons: [button(0, false), button(0, false)],
  }),
  rightInputSource: inputSource({
    handedness: 'right',
    axes: [0, 0, 0, 0],
    buttons: [button(0, false), button(0, false)],
  }),
  deadman: false,
}));
const reconnected = buildRtcFrameV1(reconnectReleased.state, validSample({
  leftInputSource: inputSource({ axes: [0, 0, 0, 0] }),
  rightInputSource: inputSource({ handedness: 'right', axes: [0, 0, 0, 0] }),
  deadman: true,
}));
assert(reconnected.frame.clutch_sequence === reengaged.state.clutchSequence + 1,
  'neutral release then neutral re-squeeze after reconnect must start a new clutch generation');
assertEqual(reconnected.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'neutral engagement must start from zero twist');
const reconnectedMotion = buildRtcFrameV1(reconnected.state, validSample({
  leftInputSource: inputSource({ axes: [0, 0, -1, -1] }),
  rightInputSource: inputSource({ handedness: 'right', axes: [0, 0, -1, 0] }),
  deadman: true,
}));
assertEqual(reconnectedMotion.frame.base_twist, {
  linear: [0.5, 0.3, 0],
  angular: [0, 0, 0.6],
}, 'stick motion after neutral engagement must re-enable the mapped twist');

// Public base_twist uses only descriptor-declared xr-standard axes/scales.
// Per-axis deadzone remapping is deterministic and gated by final deadman.
const fullPositiveTwist = buildRtcFrameV1(
  createRtcFrameState({ deadmanActive: true, rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ axes: [0, 0, -1, -1] }),
    rightInputSource: inputSource({ handedness: 'right', axes: [0, 0, -1, 0] }),
    deadman: true,
  }),
);
assert(fullPositiveTwist.frame.deadman === true,
  'valid dual squeeze and tracked inputs must admit the twist sample');
assertEqual(fullPositiveTwist.frame.base_twist, {
  linear: [0.5, 0.3, 0],
  angular: [0, 0, 0.6],
}, 'negative xr-standard axes must map to full-scale +forward/+left/+CCW');

const reboundConfiguration = {
  mode: 'live',
  capabilities: {
    ...BASE_CAPABILITIES,
    input_bindings: {
      base_twist: {
        linear_x: {
          hand: 'right', axis: 0, scale: 2, deadzone: 0, direction: 1,
        },
        linear_y: {
          hand: 'left', axis: 1, scale: 1, deadzone: 0, direction: -1,
        },
        angular_z: {
          hand: 'right', axis: 1, scale: 3, deadzone: 0, direction: -1,
        },
      },
    },
  },
};
const rebound = buildRtcFrameV1(
  createRtcFrameState({ deadmanActive: true, rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ axes: [0, -0.25, -1, -1] }),
    rightInputSource: inputSource({ handedness: 'right', axes: [0.5, -0.5, -1, 0] }),
    deadman: true,
  }),
  reboundConfiguration,
);
assertEqual(rebound.frame.base_twist, {
  linear: [1, 0.25, 0],
  angular: [0, 0, 1.5],
}, 'base mapping must consume the capability binding rather than fixed browser axes/speeds');

const fullNegativeTwist = buildRtcFrameV1(
  createRtcFrameState({ deadmanActive: true, rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ axes: [0, 0, 1, 1] }),
    rightInputSource: inputSource({ handedness: 'right', axes: [0, 0, 1, 0] }),
    deadman: true,
  }),
);
assertEqual(fullNegativeTwist.frame.base_twist, {
  linear: [-0.5, -0.3, 0],
  angular: [0, 0, -0.6],
}, 'positive xr-standard axes must map to full-scale -forward/-left/-CCW');

const remappedDeadzone = buildRtcFrameV1(
  createRtcFrameState({ deadmanActive: true, rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ axes: [0, 0, -0.6, -0.2] }),
    rightInputSource: inputSource({ handedness: 'right', axes: [0, 0, 0.19, 0] }),
    deadman: true,
  }),
);
assert(remappedDeadzone.frame.base_twist.linear[0] === 0,
  'axis at the inclusive 0.2 deadzone boundary must be zero');
assertNear(remappedDeadzone.frame.base_twist.linear[1], 0.15,
  'axis travel after the deadzone must be remapped to the remaining full range');
assert(remappedDeadzone.frame.base_twist.angular[2] === 0,
  'axis inside the deadzone must be zero');

const ignoredSecondarySlots = buildRtcFrameV1(
  createRtcFrameState({ deadmanActive: true, rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ axes: [1, -1, 0, 0] }),
    rightInputSource: inputSource({ handedness: 'right', axes: [-1, 1, 0, 0] }),
    deadman: true,
  }),
);
assertEqual(ignoredSecondarySlots.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'axes[0]/axes[1] must never drive generic base_twist');

const missingPrimaryAxes = buildRtcFrameV1(
  createRtcFrameState({ rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ axes: [0, 0] }),
    rightInputSource: inputSource({ handedness: 'right', axes: [0, 0] }),
    deadman: true,
  }),
);
assert(missingPrimaryAxes.frame.deadman === false,
  'a descriptor-bound axis must exist before deadman can become true');
assert(missingPrimaryAxes.state.rearmRequired === true,
  'a missing descriptor-bound axis must latch re-arm');
assertEqual(missingPrimaryAxes.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'missing primary thumbstick slots must fail closed to zero base_twist');

const malformedPrimaryAxes = buildRtcFrameV1(
  createRtcFrameState({ deadmanActive: true, rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ axes: [0, 0, Number.NaN, -1] }),
    rightInputSource: inputSource({
      handedness: 'right',
      axes: [0, 0, Number.POSITIVE_INFINITY, 0],
    }),
    deadman: true,
  }),
);
assert(malformedPrimaryAxes.frame.deadman === false,
  'non-finite primary locomotion axes must fail closed before mapping');
assertEqual(malformedPrimaryAxes.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'non-finite selected axes must emit only finite zeros');

const canonicalNegativeZero = buildRtcFrameV1(
  createRtcFrameState({ deadmanActive: true, rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({ axes: [0, 0, -0, -0] }),
    rightInputSource: inputSource({ handedness: 'right', axes: [0, 0, -0, 0] }),
    deadman: true,
  }),
);
assert(
  [...canonicalNegativeZero.frame.base_twist.linear,
    ...canonicalNegativeZero.frame.base_twist.angular]
    .every((value) => value === 0 && !Object.is(value, -0)),
  'negative-zero stick samples must serialize as canonical positive zeros',
);

const releasedWithFullAxes = buildRtcFrameV1(
  createRtcFrameState({ deadmanActive: true, rearmRequired: false }),
  validSample({
    leftInputSource: inputSource({
      axes: [0, 0, -1, -1],
      buttons: [button(0, false), button(0, false)],
    }),
    rightInputSource: inputSource({
      handedness: 'right',
      axes: [0, 0, -1, 0],
      buttons: [button(0, false), button(0, false)],
    }),
    deadman: false,
  }),
);
assert(releasedWithFullAxes.frame.deadman === false,
  'physical squeeze release must clear the final deadman');
assertEqual(releasedWithFullAxes.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'released deadman must zero all six twist components despite full stick travel');

// Invalid pose data becomes null and tracking=false; it can never retain deadman.
const badPose = pose({ position: [Number.NaN, 1, 0] });
const untracked = buildRtcFrameV1(reconnected.state, validSample({
  headPose: badPose,
  leftGripPose: null,
  deadman: true,
}));
assert(untracked.frame.head === null && untracked.frame.tracking.head === false,
  'NaN head pose must become consistently untracked');
assert(untracked.frame.left_controller === null
  && untracked.frame.tracking.left_controller === false,
  'missing grip pose must become consistently untracked');
assert(untracked.frame.right_controller !== null
  && untracked.frame.tracking.right_controller === true,
  'independent valid grip tracking must remain visible');
assert(untracked.frame.deadman === false, 'tracking loss must force deadman false');
assert(xrPoseToWirePose(pose({ orientation: [0, 0, 0, 0] })) === null,
  'zero quaternion must be rejected');
assert(xrPoseToWirePose(pose({ orientation: [0, 0, 0, 2] })) === null,
  'malformed quaternion magnitude must be rejected');
assert(xrPoseToWirePose(pose({ position: [101, 0, 0] })) === null,
  'out-of-workspace pose must be rejected');
assert(xrPoseToWirePose({ ...pose(), emulatedPosition: true }) === null,
  'emulated poses must be rejected as untracked');

// Controller arrays are bounded and clamped, while malformed values disable deadman.
const malformedController = inputSource({
  axes: [-2, Number.NaN, 2, 0, 0, 0, 0, 0, 0],
  buttons: Array.from({ length: 17 }, (_, index) => button(index === 0 ? -1 : 2, true)),
});
const normalizedController = xrStandardInputToWireController(malformedController);
assert(normalizedController.axes.length === 8, 'axes must be capped at 8');
assert(normalizedController.buttons.length === 16, 'buttons must be capped at 16');
assertEqual(normalizedController.axes.slice(0, 3), [-1, 0, 1],
  'axes must clamp and replace NaN with a safe zero');
assertEqual(normalizedController.buttons.slice(0, 2), [0, 1],
  'buttons must clamp to [0,1]');
const malformedFrame = buildRtcFrameV1(untracked.state, validSample({
  leftInputSource: malformedController,
  deadman: true,
}));
assert(malformedFrame.frame.deadman === false,
  'truncated, clamped, or non-finite controller data must force deadman false');
assertEqual(malformedFrame.frame.base_twist, {
  linear: [0, 0, 0],
  angular: [0, 0, 0],
}, 'malformed controller input must fail closed to zero base_twist');
assertNoPrivateAuthority(malformedFrame.frame);
const missingSqueeze = buildRtcFrameV1(malformedFrame.state, validSample({
  leftInputSource: inputSource({ buttons: [button(0, false)] }),
  deadman: true,
}));
assert(missingSqueeze.frame.deadman === false && missingSqueeze.state.rearmRequired === true,
  'a controller without the xr-standard squeeze slot must remain unarmed');

// A sample Proxy that throws is data loss, not a control-path exception.
const throwingSample = new Proxy({}, {
  get() { throw new Error('bad WebXR sample getter'); },
});
const failSafe = buildRtcFrameV1(malformedFrame.state, throwingSample);
assertEqual(failSafe.frame.tracking, {
  head: false,
  left_controller: false,
  right_controller: false,
}, 'throwing sample must become fully untracked');
assert(failSafe.frame.deadman === false, 'throwing sample must force deadman false');
assert(failSafe.frame.head === null
  && failSafe.frame.left_controller === null
  && failSafe.frame.right_controller === null,
  'throwing sample must emit null poses');

// JS-safe bounds are explicit; internal counter exhaustion fails before a frame is emitted.
assert(performanceMillisecondsToNanoseconds(1.25) === 1_250_000,
  'performance milliseconds must convert to integer nanoseconds');
assert(performanceMillisecondsToNanoseconds(Number.NaN) === null,
  'NaN performance time must be rejected');
assert(performanceMillisecondsToNanoseconds(-1) === null,
  'negative performance time must be rejected');
assert(performanceMillisecondsToNanoseconds((MAX_SAFE_WIRE_INTEGER / 1_000_000) + 1) === null,
  'unsafe performance nanoseconds must be rejected');
assertThrows(
  () => createRtcFrameState({ nextSequence: MAX_SAFE_WIRE_INTEGER + 1 }),
  'unsafe sequence state must be rejected',
);
assertThrows(
  () => buildRtcFrameV1(
    createRtcFrameState({ nextSequence: MAX_SAFE_WIRE_INTEGER }),
    validSample(),
  ),
  'exhausted sequence must fail before emitting a repeated/unsafe next state',
);
assertThrows(
  () => buildRtcFrameV1(
    createRtcFrameState({
      clutchSequence: MAX_SAFE_WIRE_INTEGER,
      rearmRequired: false,
    }),
    validSample({ deadman: true }),
  ),
  'exhausted clutch counter must fail before engagement',
);

// The convenience generator has instance-local state and preserves the same reducer rules.
const generator = createRtcFrameGenerator({
  nextSequence: 10,
  clutchSequence: 4,
  rearmRequired: false,
}, BASE_FRAME_CONFIGURATION);
const generatedA = generator.next(validSample({ deadman: true }));
const generatedB = generator.next(validSample({ deadman: true }));
assert(generatedA.sequence === 10 && generatedB.sequence === 11,
  'generator sequence must be monotonic');
assert(generatedA.clutch_sequence === 5 && generatedB.clutch_sequence === 5,
  'generator clutch edge must be monotonic and stable while held');
assert(generator.snapshot().nextSequence === 12, 'generator snapshot must expose resumable state');

console.log('webxr-frame: all tests passed');
