// Pure WebXR -> motus.teleop.rtc-frame.v1 mapping.
//
// The browser wire contract intentionally has no session-authority fields.
// Core and Driver bind boot_id/session_id/epoch/fence server-side after the
// WebRTC offer ticket has been verified.

export const RTC_FRAME_PROTOCOL = 'motus.teleop.rtc-frame.v1';
export const MAX_SAFE_WIRE_INTEGER = Number.MAX_SAFE_INTEGER;

const MAX_AXES = 8;
const MAX_BUTTONS = 16;
const MAX_POSITION_METRES = 100;
const QUATERNION_NORM_MIN = 0.5;
const QUATERNION_NORM_MAX = 1.5;
const QUATERNION_COMPONENT_LIMIT = 1.000001;
const FRAME_MODES = new Set(['shadow', 'live']);
const BASE_TWIST_AXES = ['linear_x', 'linear_y', 'angular_z'];

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function requireSafeCounter(value, name, { allowMaximum = true } = {}) {
  const upper = allowMaximum ? MAX_SAFE_WIRE_INTEGER : MAX_SAFE_WIRE_INTEGER - 1;
  if (!Number.isSafeInteger(value) || value < 0 || value > upper) {
    throw new RangeError(`${name} must be a non-negative JavaScript safe integer <= ${upper}`);
  }
  return value;
}

function component(object, name) {
  if (!object || typeof object !== 'object') return null;
  try {
    const value = object[name];
    return isFiniteNumber(value) ? value : null;
  } catch {
    return null;
  }
}

function property(object, name) {
  if (!object || typeof object !== 'object') return undefined;
  try {
    return object[name];
  } catch {
    return undefined;
  }
}

function arrayLikeLength(value) {
  if (!value || typeof value === 'string') return null;
  try {
    const length = value.length;
    if (!Number.isSafeInteger(length) || length < 0) return null;
    return length;
  } catch {
    return null;
  }
}

function normalizedScalar(value, minimum, maximum) {
  if (!isFiniteNumber(value)) return { value: 0, valid: false };
  return {
    value: clamp(value, minimum, maximum),
    valid: value >= minimum && value <= maximum,
  };
}

function isQualifiedGripSource(inputSource, expectedHandedness) {
  if (!expectedHandedness) return true;
  try {
    return Boolean(
      inputSource
      && typeof inputSource === 'object'
      && inputSource.handedness === expectedHandedness
      && inputSource.targetRayMode === 'tracked-pointer'
      && inputSource.gripSpace,
    );
  } catch {
    return false;
  }
}

function normalizeXrStandardInput(inputSource, expectedHandedness = null) {
  let gamepad;
  try {
    gamepad = inputSource && typeof inputSource === 'object' ? inputSource.gamepad : null;
  } catch {
    gamepad = null;
  }

  let mapping;
  let axesSource;
  let buttonsSource;
  try {
    mapping = gamepad?.mapping;
    axesSource = gamepad?.axes;
    buttonsSource = gamepad?.buttons;
  } catch {
    mapping = null;
  }

  const axesLength = arrayLikeLength(axesSource);
  const buttonsLength = arrayLikeLength(buttonsSource);
  let valid = mapping === 'xr-standard'
    && isQualifiedGripSource(inputSource, expectedHandedness)
    && axesLength !== null
    && buttonsLength !== null
    && axesLength <= MAX_AXES
    && buttonsLength >= 2
    && buttonsLength <= MAX_BUTTONS;

  const axes = [];
  for (let index = 0; index < Math.min(axesLength ?? 0, MAX_AXES); index += 1) {
    let raw;
    try {
      raw = axesSource[index];
    } catch {
      raw = null;
    }
    const normalized = normalizedScalar(raw, -1, 1);
    axes.push(normalized.value);
    valid = valid && normalized.valid;
  }

  const buttons = [];
  for (let index = 0; index < Math.min(buttonsLength ?? 0, MAX_BUTTONS); index += 1) {
    let raw;
    try {
      const button = buttonsSource[index];
      raw = button && typeof button === 'object' ? button.value : null;
    } catch {
      raw = null;
    }
    const normalized = normalizedScalar(raw, 0, 1);
    buttons.push(normalized.value);
    valid = valid && normalized.valid;
  }

  return {
    controller: { axes, buttons },
    valid,
  };
}

function zeroBaseTwist() {
  return {
    linear: [0, 0, 0],
    angular: [0, 0, 0],
  };
}

function remapBoundAxis(value, deadzone) {
  if (!isFiniteNumber(value) || value < -1 || value > 1) return null;
  const magnitude = Math.abs(value);
  if (magnitude <= deadzone) return 0;
  return Math.sign(value)
    * ((magnitude - deadzone) / (1 - deadzone));
}

function scaledBoundAxis(value, binding) {
  const remapped = remapBoundAxis(value, binding.deadzone);
  if (remapped === null) return null;
  if (remapped === 0) return 0;
  return remapped * binding.scale * binding.direction;
}

function normalizeAxisBinding(value, name) {
  if (!value || typeof value !== 'object') {
    throw new TypeError(`${name} must be an object`);
  }
  const keys = Object.keys(value).sort();
  if (JSON.stringify(keys) !== JSON.stringify([
    'axis', 'deadzone', 'direction', 'hand', 'scale',
  ])) {
    throw new TypeError(`${name} has an invalid shape`);
  }
  const hand = property(value, 'hand');
  const axis = property(value, 'axis');
  const scale = property(value, 'scale');
  const deadzone = property(value, 'deadzone');
  const direction = property(value, 'direction');
  if (
    !['left', 'right'].includes(hand)
    || !Number.isSafeInteger(axis)
    || axis < 0
    || axis >= MAX_AXES
    || !isFiniteNumber(scale)
    || scale < 0
    || scale > 10
    || !isFiniteNumber(deadzone)
    || deadzone < 0
    || deadzone > 0.95
    || ![-1, 1].includes(direction)
  ) throw new TypeError(`${name} is invalid`);
  return Object.freeze({ hand, axis, scale, deadzone, direction });
}

function normalizeFrameConfiguration(configuration) {
  if (!configuration || typeof configuration !== 'object') {
    throw new TypeError('frame configuration is required');
  }
  const mode = property(configuration, 'mode');
  if (!FRAME_MODES.has(mode)) throw new TypeError('frame mode must be shadow or live');
  const capabilities = property(configuration, 'capabilities');
  if (!capabilities || typeof capabilities !== 'object') {
    throw new TypeError('frame capabilities are required');
  }
  const outputs = property(capabilities, 'outputs');
  const inputBindings = property(capabilities, 'input_bindings');
  if (!outputs || typeof outputs !== 'object' || !inputBindings || typeof inputBindings !== 'object') {
    throw new TypeError('frame capabilities are invalid');
  }
  const baseEnabled = property(property(outputs, 'base'), 'enabled') === true;
  const rawBaseTwist = property(inputBindings, 'base_twist');
  if (baseEnabled !== Boolean(rawBaseTwist && typeof rawBaseTwist === 'object')) {
    throw new TypeError('base output and base_twist binding must be enabled together');
  }
  let baseTwistBinding = null;
  if (baseEnabled) {
    if (
      JSON.stringify(Object.keys(rawBaseTwist).sort())
      !== JSON.stringify([...BASE_TWIST_AXES].sort())
    ) throw new TypeError('base_twist binding is invalid');
    baseTwistBinding = Object.freeze(Object.fromEntries(BASE_TWIST_AXES.map((name) => [
      name,
      normalizeAxisBinding(property(rawBaseTwist, name), `base_twist.${name}`),
    ])));
  }
  return Object.freeze({ mode, baseTwistBinding });
}

function boundAxisValue(binding, leftInput, rightInput) {
  const controller = binding.hand === 'left' ? leftInput?.controller : rightInput?.controller;
  const axes = controller?.axes;
  if (!Array.isArray(axes) || binding.axis >= axes.length) return null;
  return axes[binding.axis];
}

function boundMotionInputsNeutral(leftInput, rightInput, baseTwistBinding) {
  if (!baseTwistBinding) return true;
  return BASE_TWIST_AXES.every((name) => {
    const binding = baseTwistBinding[name];
    const value = boundAxisValue(binding, leftInput, rightInput);
    return value !== null && remapBoundAxis(value, binding.deadzone) === 0;
  });
}

function boundMotionInputsAvailable(leftInput, rightInput, baseTwistBinding) {
  if (!baseTwistBinding) return true;
  return BASE_TWIST_AXES.every((name) => (
    boundAxisValue(baseTwistBinding[name], leftInput, rightInput) !== null
  ));
}

function buildBaseTwist(deadman, inputsValid, leftInput, rightInput, binding) {
  const zero = zeroBaseTwist();
  if (deadman !== true || inputsValid !== true || !binding) return zero;
  const linearX = scaledBoundAxis(
    boundAxisValue(binding.linear_x, leftInput, rightInput),
    binding.linear_x,
  );
  const linearY = scaledBoundAxis(
    boundAxisValue(binding.linear_y, leftInput, rightInput),
    binding.linear_y,
  );
  const angularZ = scaledBoundAxis(
    boundAxisValue(binding.angular_z, leftInput, rightInput),
    binding.angular_z,
  );
  if (linearX === null || linearY === null || angularZ === null) return zero;

  return {
    linear: [linearX, linearY, 0],
    angular: [0, 0, angularZ],
  };
}

/**
 * Convert an XRPose (or its XRRigidTransform) into the Driver pose shape.
 * Invalid, untracked, non-finite, out-of-workspace, or malformed poses become
 * null so the caller can keep tracking and pose fields consistent.
 */
export function xrPoseToWirePose(xrPose) {
  if (!xrPose || typeof xrPose !== 'object') return null;

  let transform;
  let position;
  let orientation;
  try {
    if (xrPose.emulatedPosition === true) return null;
    transform = xrPose.transform && typeof xrPose.transform === 'object'
      ? xrPose.transform
      : xrPose;
    position = transform.position;
    orientation = transform.orientation;
  } catch {
    return null;
  }

  const positionValues = [
    component(position, 'x'),
    component(position, 'y'),
    component(position, 'z'),
  ];
  if (positionValues.some((value) => value === null || Math.abs(value) > MAX_POSITION_METRES)) {
    return null;
  }

  const quaternion = [
    component(orientation, 'x'),
    component(orientation, 'y'),
    component(orientation, 'z'),
    component(orientation, 'w'),
  ];
  if (quaternion.some((value) => (
    value === null || Math.abs(value) > QUATERNION_COMPONENT_LIMIT
  ))) {
    return null;
  }

  const norm = Math.hypot(...quaternion);
  if (!Number.isFinite(norm) || norm < QUATERNION_NORM_MIN || norm > QUATERNION_NORM_MAX) {
    return null;
  }
  const normalized = quaternion.map((value) => value / norm);
  const normalizedNorm = Math.hypot(...normalized);
  if (!Number.isFinite(normalizedNorm) || normalizedNorm === 0) return null;
  const strictOrientation = normalized.map((value) => clamp(value / normalizedNorm, -1, 1));
  if (strictOrientation.some((value) => !Number.isFinite(value))) return null;

  return {
    position: positionValues,
    orientation: strictOrientation,
  };
}

/** Return only the public controller payload; validity remains an internal safety input. */
export function xrStandardInputToWireController(inputSource) {
  return normalizeXrStandardInput(inputSource).controller;
}

function xrStandardSqueezeState(inputSource) {
  try {
    const gamepad = inputSource?.gamepad;
    if (!gamepad || gamepad.mapping !== 'xr-standard') return 'invalid';
    const squeeze = gamepad.buttons?.[1];
    if (
      !squeeze
      || typeof squeeze !== 'object'
      || typeof squeeze.pressed !== 'boolean'
      || !isFiniteNumber(squeeze.value)
      || squeeze.value < 0
      || squeeze.value > 1
    ) return 'invalid';
    if (squeeze.pressed === true && squeeze.value >= 0.75) return 'pressed';
    if (squeeze.pressed === false && squeeze.value < 0.75) return 'released';
    return 'transition';
  } catch {
    return 'invalid';
  }
}

/** Quest/xr-standard deadman policy: both explicit squeeze buttons are held. */
export function isDualSqueezeDeadmanRequested(leftInputSource, rightInputSource) {
  return leftInputSource !== rightInputSource
    && isQualifiedGripSource(leftInputSource, 'left')
    && isQualifiedGripSource(rightInputSource, 'right')
    && xrStandardSqueezeState(leftInputSource) === 'pressed'
    && xrStandardSqueezeState(rightInputSource) === 'pressed';
}

/**
 * Convert a performance.now()/XR predictedDisplayTime value to bounded integer
 * nanoseconds. Null means the sample was unsafe; the reducer will then advance
 * from its last known monotonic value instead of emitting invalid JSON.
 */
export function performanceMillisecondsToNanoseconds(milliseconds) {
  if (!isFiniteNumber(milliseconds) || milliseconds < 0) return null;
  const nanoseconds = Math.round(milliseconds * 1_000_000);
  if (!Number.isSafeInteger(nanoseconds) || nanoseconds < 0) return null;
  return nanoseconds;
}

/**
 * State is explicit and serializable. nextSequence is the value the next frame
 * will emit; clutchSequence is the last emitted clutch generation; and
 * rearmRequired latches entry/tracking/reconnect safety until a valid physical
 * squeeze release with every available locomotion axis neutral is observed.
 */
export function createRtcFrameState({
  nextSequence = 0,
  clutchSequence = 0,
  deadmanActive = false,
  rearmRequired = true,
  lastMonotonicNs = 0,
} = {}) {
  requireSafeCounter(nextSequence, 'nextSequence');
  requireSafeCounter(clutchSequence, 'clutchSequence');
  requireSafeCounter(lastMonotonicNs, 'lastMonotonicNs');
  if (typeof deadmanActive !== 'boolean') {
    throw new TypeError('deadmanActive must be a boolean');
  }
  if (typeof rearmRequired !== 'boolean') {
    throw new TypeError('rearmRequired must be a boolean');
  }
  return Object.freeze({
    nextSequence,
    clutchSequence,
    deadmanActive,
    rearmRequired,
    lastMonotonicNs,
  });
}

/**
 * Persist this state when an XR/RTC stream ends. Counters and the monotonic
 * watermark are deliberately retained across reconnects. Deadman is cleared
 * and re-arm is latched so holding squeeze across the boundary cannot recover.
 */
export function markRtcFrameDeadmanReleased(previousState) {
  const state = createRtcFrameState(previousState);
  if (!state.deadmanActive && state.rearmRequired) return state;
  return createRtcFrameState({
    nextSequence: state.nextSequence,
    clutchSequence: state.clutchSequence,
    deadmanActive: false,
    rearmRequired: true,
    lastMonotonicNs: state.lastMonotonicNs,
  });
}

function nextMonotonicNanoseconds(previous, milliseconds) {
  const candidate = performanceMillisecondsToNanoseconds(milliseconds);
  if (candidate !== null && candidate > previous) return candidate;
  if (previous >= MAX_SAFE_WIRE_INTEGER) {
    throw new RangeError('client_monotonic_ns exhausted the JavaScript safe integer range');
  }
  return previous + 1;
}

/**
 * Pure reducer from one explicit state plus one WebXR sample to a strict public
 * RTC frame. `deadman` is opt-in: only the literal boolean true can request it,
 * and malformed tracking/controller data forces it back to false.
 */
export function buildRtcFrameV1(previousState, sample = {}, configuration = {}) {
  const state = createRtcFrameState(previousState);
  const source = sample && typeof sample === 'object' ? sample : {};
  const frameConfiguration = normalizeFrameConfiguration(configuration);
  requireSafeCounter(state.nextSequence, 'nextSequence', { allowMaximum: false });

  const head = xrPoseToWirePose(property(source, 'headPose'));
  const leftController = xrPoseToWirePose(property(source, 'leftGripPose'));
  const rightController = xrPoseToWirePose(property(source, 'rightGripPose'));
  const leftInputSource = property(source, 'leftInputSource');
  const rightInputSource = property(source, 'rightInputSource');
  const leftInput = normalizeXrStandardInput(leftInputSource, 'left');
  const rightInput = normalizeXrStandardInput(rightInputSource, 'right');

  const tracking = {
    head: head !== null,
    left_controller: leftController !== null,
    right_controller: rightController !== null,
  };
  const allTracked = tracking.head && tracking.left_controller && tracking.right_controller;
  const callerRequestedDeadman = property(source, 'deadman') === true;
  const leftSqueezeState = xrStandardSqueezeState(leftInputSource);
  const rightSqueezeState = xrStandardSqueezeState(rightInputSource);
  const squeezeRequested = isDualSqueezeDeadmanRequested(leftInputSource, rightInputSource);
  const inputsValid = leftInput.valid
    && rightInput.valid
    && leftInputSource !== rightInputSource
    && boundMotionInputsAvailable(
      leftInput,
      rightInput,
      frameConfiguration.baseTwistBinding,
    );
  const explicitReleaseObserved = inputsValid
    && leftSqueezeState !== 'invalid'
    && rightSqueezeState !== 'invalid'
    && (leftSqueezeState === 'released' || rightSqueezeState === 'released');
  const motionInputsNeutral = boundMotionInputsNeutral(
    leftInput,
    rightInput,
    frameConfiguration.baseTwistBinding,
  );
  const nonNeutralEngagementAttempt = !state.deadmanActive
    && callerRequestedDeadman
    && squeezeRequested
    && !motionInputsNeutral;
  let rearmRequired = state.rearmRequired;
  if (!allTracked || !inputsValid) {
    rearmRequired = true;
  } else if (explicitReleaseObserved) {
    // Re-arm requires both a physical squeeze release and neutral locomotion
    // sticks. A held stick cannot preload the first command after reconnect.
    rearmRequired = !motionInputsNeutral;
  } else if (
    (state.deadmanActive && !squeezeRequested)
    || (!callerRequestedDeadman && squeezeRequested)
    || nonNeutralEngagementAttempt
  ) {
    rearmRequired = true;
  }
  const deadman = callerRequestedDeadman
    && allTracked
    && inputsValid
    && squeezeRequested
    && !rearmRequired;

  let clutchSequence = state.clutchSequence;
  if (deadman && !state.deadmanActive) {
    if (clutchSequence >= MAX_SAFE_WIRE_INTEGER) {
      throw new RangeError('clutch_sequence exhausted the JavaScript safe integer range');
    }
    clutchSequence += 1;
  }
  const clientMonotonicNs = nextMonotonicNanoseconds(
    state.lastMonotonicNs,
    property(source, 'monotonicTimeMs'),
  );
  const frame = {
    schema_version: 1,
    sequence: state.nextSequence,
    client_monotonic_ns: clientMonotonicNs,
    mode: frameConfiguration.mode,
    deadman,
    clutch_sequence: clutchSequence,
    tracking,
    head,
    left_controller: leftController,
    right_controller: rightController,
    controllers: {
      left: leftInput.controller,
      right: rightInput.controller,
    },
  };
  if (frameConfiguration.baseTwistBinding) {
    frame.base_twist = buildBaseTwist(
      deadman,
      inputsValid,
      leftInput,
      rightInput,
      frameConfiguration.baseTwistBinding,
    );
  }

  return {
    frame,
    state: createRtcFrameState({
      nextSequence: state.nextSequence + 1,
      clutchSequence,
      deadmanActive: deadman,
      rearmRequired,
      lastMonotonicNs: clientMonotonicNs,
    }),
  };
}

/** Stateful convenience wrapper; all state remains scoped to this instance. */
export function createRtcFrameGenerator(options = {}, configuration = {}) {
  let state = createRtcFrameState(options);
  return Object.freeze({
    next(sample = {}) {
      const result = buildRtcFrameV1(state, sample, configuration);
      state = result.state;
      return result.frame;
    },
    snapshot() {
      return state;
    },
  });
}
