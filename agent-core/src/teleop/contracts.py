from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


SHADOW_MODE = 'shadow'
LIVE_MODE = 'live'
TELEOP_MODES = frozenset({SHADOW_MODE, LIVE_MODE})
PROTOCOL_BY_MODE = {
    SHADOW_MODE: 'motus.teleop.shadow.v1',
    LIVE_MODE: 'motus.teleop.live.v1',
}
DISPATCH_CONTRACT_BY_MODE = {
    SHADOW_MODE: 'motus.teleop.dispatch.recording.v1',
    LIVE_MODE: 'motus.teleop.dispatch.hardware.v1',
}
PREPARE_ACTION_BY_MODE = {
    SHADOW_MODE: 'prepare_shadow',
    LIVE_MODE: 'prepare_live',
}
WEBRTC_SIGNALING_PROTOCOL = 'motus.teleop.webrtc-offer-answer.v1'
WEBRTC_SIGNALING_PATH = '/offer'
WEBRTC_SIGNALING_ACCESS = 'authenticated-core-proxy-only'
WEBRTC_SIGNALING_AUDIENCES = frozenset({'teleop-shadow-rtc', 'motus-teleop-rtc'})

_CAPABILITY_DIGEST_RE = re.compile(r'^[0-9a-f]{64}$')
_SAFE_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')
_BASE_TWIST_AXES = ('linear_x', 'linear_y', 'angular_z')
_BASE_TWIST_BINDING_KEYS = frozenset({'hand', 'axis', 'scale', 'deadzone', 'direction'})
_ROLE_BINDING_KEYS = frozenset({'required', 'role'})
_OUTPUT_KEYS = frozenset({'enabled', 'joint_count'})
_CAPABILITY_KEYS = frozenset({'profile_id', 'input_bindings', 'outputs', 'effectors'})
_DESCRIPTOR_KEYS = frozenset({
    'protocol',
    'driver_id',
    'driver_name',
    'robot_id',
    'profile_id',
    'mode',
    'actuation_enabled',
    'capability_digest',
    'dispatch_contract',
    'signaling',
    'capabilities',
    'dry_run_profile',
})
_COMMON_ACTIONS = frozenset({'heartbeat', 'pause', 'release', 'soft_stop', 'status', 'stop'})
_IDENTITY_PARAMS = ['boot_id', 'session_id', 'epoch', 'fence']


class TeleopContractError(ValueError):
    pass


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise TeleopContractError(f'{field} is invalid')
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TeleopContractError(f'{field} is invalid')
    return value


def _finite_number(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeleopContractError(f'{field} is invalid')
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise TeleopContractError(f'{field} is invalid')
    return number


def _strict_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TeleopContractError(f'{field} is invalid')
    return value


def _project_axis_binding(value: object, *, field: str) -> dict[str, object]:
    binding = _strict_mapping(value, field=field)
    if set(binding) != _BASE_TWIST_BINDING_KEYS:
        raise TeleopContractError(f'{field} is invalid')
    hand = binding.get('hand')
    axis = binding.get('axis')
    direction = binding.get('direction')
    if hand not in {'left', 'right'}:
        raise TeleopContractError(f'{field}.hand is invalid')
    if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 7:
        raise TeleopContractError(f'{field}.axis is invalid')
    if direction not in {-1, 1} or isinstance(direction, bool):
        raise TeleopContractError(f'{field}.direction is invalid')
    return {
        'hand': hand,
        'axis': axis,
        'scale': _finite_number(
            binding.get('scale'),
            field=f'{field}.scale',
            minimum=0.0,
            maximum=10.0,
        ),
        'deadzone': _finite_number(
            binding.get('deadzone'),
            field=f'{field}.deadzone',
            minimum=0.0,
            maximum=0.95,
        ),
        'direction': direction,
    }


def _project_input_bindings(value: object) -> dict[str, object]:
    bindings = _strict_mapping(value, field='capabilities.input_bindings')
    if len(bindings) > 16:
        raise TeleopContractError('capabilities.input_bindings is invalid')
    result: dict[str, object] = {}
    for raw_name, raw_binding in bindings.items():
        name = _safe_id(raw_name, field='capabilities.input_bindings key')
        if name == 'base_twist':
            base_twist = _strict_mapping(raw_binding, field='input_bindings.base_twist')
            if set(base_twist) != set(_BASE_TWIST_AXES):
                raise TeleopContractError('input_bindings.base_twist is invalid')
            result[name] = {
                axis_name: _project_axis_binding(
                    base_twist[axis_name],
                    field=f'input_bindings.base_twist.{axis_name}',
                )
                for axis_name in _BASE_TWIST_AXES
            }
            continue
        binding = _strict_mapping(raw_binding, field=f'input_bindings.{name}')
        if set(binding) != _ROLE_BINDING_KEYS:
            raise TeleopContractError(f'input_bindings.{name} is invalid')
        result[name] = {
            'required': _strict_bool(
                binding.get('required'),
                field=f'input_bindings.{name}.required',
            ),
            'role': _safe_id(binding.get('role'), field=f'input_bindings.{name}.role'),
        }
    return result


def _project_outputs(value: object) -> dict[str, dict[str, object]]:
    outputs = _strict_mapping(value, field='capabilities.outputs')
    if not outputs or len(outputs) > 16:
        raise TeleopContractError('capabilities.outputs is invalid')
    result: dict[str, dict[str, object]] = {}
    for raw_name, raw_output in outputs.items():
        name = _safe_id(raw_name, field='capabilities.outputs key')
        output = _strict_mapping(raw_output, field=f'capabilities.outputs.{name}')
        if not set(output).issubset(_OUTPUT_KEYS) or 'enabled' not in output:
            raise TeleopContractError(f'capabilities.outputs.{name} is invalid')
        projected: dict[str, object] = {
            'enabled': _strict_bool(
                output.get('enabled'),
                field=f'capabilities.outputs.{name}.enabled',
            ),
        }
        if 'joint_count' in output:
            joint_count = output.get('joint_count')
            if (
                isinstance(joint_count, bool)
                or not isinstance(joint_count, int)
                or not 0 <= joint_count <= 128
            ):
                raise TeleopContractError(
                    f'capabilities.outputs.{name}.joint_count is invalid',
                )
            projected['joint_count'] = joint_count
        result[name] = projected
    return result


def project_capabilities(value: object, *, profile_id: str) -> dict[str, object]:
    capabilities = _strict_mapping(value, field='capabilities')
    if set(capabilities) - _CAPABILITY_KEYS:
        raise TeleopContractError('capabilities contains unsupported fields')
    if capabilities.get('profile_id') != profile_id:
        raise TeleopContractError('capabilities.profile_id does not match profile_id')
    input_bindings = _project_input_bindings(capabilities.get('input_bindings'))
    outputs = _project_outputs(capabilities.get('outputs'))
    base_enabled = outputs.get('base', {}).get('enabled') is True
    if base_enabled != ('base_twist' in input_bindings):
        raise TeleopContractError(
            'base output and input_bindings.base_twist must be enabled together',
        )

    effectors_raw = capabilities.get('effectors')
    if effectors_raw is None:
        effectors = [name for name, output in outputs.items() if output['enabled']]
    else:
        if not isinstance(effectors_raw, list) or len(effectors_raw) > 16:
            raise TeleopContractError('capabilities.effectors is invalid')
        effectors = [
            _safe_id(item, field='capabilities.effectors item')
            for item in effectors_raw
        ]
        if len(effectors) != len(set(effectors)):
            raise TeleopContractError('capabilities.effectors contains duplicates')
        enabled_outputs = {name for name, output in outputs.items() if output['enabled']}
        if set(effectors) != enabled_outputs:
            raise TeleopContractError('capabilities.effectors does not match enabled outputs')
    return {
        'profile_id': profile_id,
        'input_bindings': input_bindings,
        'outputs': outputs,
        'effectors': effectors,
    }


def project_teleop_descriptor(
    tool: object,
    *,
    expected_driver_id: str | None = None,
) -> dict[str, object]:
    if not isinstance(tool, Mapping) or tool.get('type') != 'actuator':
        raise TeleopContractError('teleop_session tool is invalid')
    descriptor = _strict_mapping(tool.get('x-teleop'), field='x-teleop')
    mode = descriptor.get('mode')
    if mode not in TELEOP_MODES:
        raise TeleopContractError('x-teleop.mode is invalid')
    driver_id = _safe_id(descriptor.get('driver_id'), field='x-teleop.driver_id')
    robot_id = _safe_id(descriptor.get('robot_id'), field='x-teleop.robot_id')
    legacy_shadow = mode == SHADOW_MODE and 'profile_id' not in descriptor
    if set(descriptor) - _DESCRIPTOR_KEYS:
        raise TeleopContractError('x-teleop contains unsupported fields')
    if legacy_shadow:
        if descriptor.get('dry_run_profile', 'recording') != 'recording':
            raise TeleopContractError('legacy Shadow profile is invalid')
        profile_id = 'recording'
    else:
        if 'dry_run_profile' in descriptor:
            raise TeleopContractError('dry_run_profile is legacy Shadow-only')
        profile_id = _safe_id(descriptor.get('profile_id'), field='x-teleop.profile_id')
    if expected_driver_id is not None and driver_id != expected_driver_id:
        raise TeleopContractError('x-teleop.driver_id does not match')
    expected_actuation = mode == LIVE_MODE
    if descriptor.get('actuation_enabled') is not expected_actuation:
        raise TeleopContractError('x-teleop.actuation_enabled is invalid')
    if descriptor.get('protocol') != PROTOCOL_BY_MODE[mode]:
        raise TeleopContractError('x-teleop.protocol is invalid')
    if descriptor.get('dispatch_contract') != DISPATCH_CONTRACT_BY_MODE[mode]:
        raise TeleopContractError('x-teleop.dispatch_contract is invalid')
    digest = descriptor.get('capability_digest')
    if not isinstance(digest, str) or not _CAPABILITY_DIGEST_RE.fullmatch(digest):
        raise TeleopContractError('x-teleop.capability_digest is invalid')

    signaling = _strict_mapping(descriptor.get('signaling'), field='x-teleop.signaling')
    signaling_audience = signaling.get('audience')
    if legacy_shadow and signaling_audience is None:
        signaling_audience = 'teleop-shadow-rtc'
    expected_signaling_keys = {'protocol', 'path', 'access', 'audience'}
    if legacy_shadow and 'audience' not in signaling:
        expected_signaling_keys.remove('audience')
    if (
        set(signaling) != expected_signaling_keys
        or signaling.get('protocol') != WEBRTC_SIGNALING_PROTOCOL
        or signaling.get('path') != WEBRTC_SIGNALING_PATH
        or signaling.get('access') != WEBRTC_SIGNALING_ACCESS
        or signaling_audience not in WEBRTC_SIGNALING_AUDIENCES
        or (mode == LIVE_MODE and signaling_audience != 'motus-teleop-rtc')
    ):
        raise TeleopContractError('x-teleop.signaling is invalid')

    input_schema = _strict_mapping(tool.get('inputSchema'), field='inputSchema')
    properties = _strict_mapping(input_schema.get('properties'), field='inputSchema.properties')
    action_schema = _strict_mapping(properties.get('action'), field='inputSchema.action')
    action_enum = action_schema.get('enum')
    action_params = input_schema.get('x-action-params')
    required_actions = _COMMON_ACTIONS | {PREPARE_ACTION_BY_MODE[mode]}
    if (
        not isinstance(action_enum, list)
        or not isinstance(action_params, Mapping)
        or any(not isinstance(item, str) or not item for item in action_enum)
        or len(action_enum) != len(set(action_enum))
        or not set(action_enum).issubset(action_params)
        or not required_actions.issubset(action_enum)
        or not required_actions.issubset(action_params)
    ):
        raise TeleopContractError('teleop_session actions are invalid')

    expected_action_params = {
        PREPARE_ACTION_BY_MODE[mode]: ['session_id', 'epoch', 'fence'],
        'heartbeat': _IDENTITY_PARAMS,
        'pause': _IDENTITY_PARAMS,
        'release': _IDENTITY_PARAMS,
        'soft_stop': _IDENTITY_PARAMS,
        'status': [],
        'stop': [],
    }
    for action, expected_params in expected_action_params.items():
        declaration = action_params.get(action)
        if (
            not isinstance(declaration, Mapping)
            or declaration.get('params') != expected_params
        ):
            raise TeleopContractError(f'teleop_session action {action} is invalid')

    if legacy_shadow and descriptor.get('capabilities') is None:
        capabilities = {
            'profile_id': profile_id,
            'input_bindings': {},
            'outputs': {'recording': {'enabled': False}},
            'effectors': [],
        }
    else:
        capabilities = project_capabilities(
            descriptor.get('capabilities'),
            profile_id=profile_id,
        )
    driver_name = descriptor.get('driver_name', driver_id)
    if not isinstance(driver_name, str) or not 1 <= len(driver_name) <= 256:
        raise TeleopContractError('x-teleop.driver_name is invalid')
    return {
        'protocol': PROTOCOL_BY_MODE[mode],
        'driver_id': driver_id,
        'driver_name': driver_name,
        'robot_id': robot_id,
        'profile_id': profile_id,
        'mode': mode,
        'actuation_enabled': expected_actuation,
        'capability_digest': digest,
        'dispatch_contract': DISPATCH_CONTRACT_BY_MODE[mode],
        'signaling': {
            'protocol': WEBRTC_SIGNALING_PROTOCOL,
            'path': WEBRTC_SIGNALING_PATH,
            'access': WEBRTC_SIGNALING_ACCESS,
            'audience': signaling_audience,
        },
        'capabilities': capabilities,
    }


def valid_teleop_descriptor(
    tool: object,
    *,
    expected_driver_id: str | None = None,
) -> bool:
    try:
        project_teleop_descriptor(tool, expected_driver_id=expected_driver_id)
    except TeleopContractError:
        return False
    return True
