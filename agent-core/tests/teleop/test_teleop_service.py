from __future__ import annotations

import asyncio
import base64
import json
import threading
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

import aiohttp
import pytest

import auth
import config
import mcp_client
from api import mcp_manage, motus_stream, teleop
from teleop import audit, authority_guard
from teleop.command_broker import CommandBroker, TeleopCommandBlocked
from teleop.service import (
    TeleopCoordinator,
    TeleopServiceError,
    _project_driver_snapshot,
)
from teleop.session_manager import (
    MAX_DRIVER_EPOCH,
    SessionConflict,
    SessionNotFound,
    ShadowSessionManager,
)

DRIVER_ID = 'teleop-shadow-driver'
ROBOT_ID = 'robot-fixture'
DIGEST = '0123456789abcdef' * 4
BOOT_ID = 'f77787b9-c5d2-465a-b4b4-e74f61f35e30'
DRIVER_TOKEN = 'driver-token-must-never-leak'
CLIENT_ID = '7dbabfca-15c1-43ca-b600-75e7682c21d0'
_REAL_AUDIT_EMIT = audit.emit
LIVE_PROFILE_ID = 'dual_arm_profile_v1'
LIVE_CAPABILITIES = {
    'profile_id': LIVE_PROFILE_ID,
    'input_bindings': {
        'head': {'required': True, 'role': 'reference'},
        'left_controller': {'required': True, 'role': 'left_end_effector'},
        'right_controller': {'required': True, 'role': 'right_end_effector'},
    },
    'outputs': {
        'dual_arm': {'enabled': True, 'joint_count': 10},
        'base': {'enabled': False},
        'hands': {'enabled': False},
    },
    'effectors': ['dual_arm'],
}
G1_PROFILE_ID = 'unitree_g1_23_dual_arm_controller_v1'
G1_CAPABILITIES = deepcopy(LIVE_CAPABILITIES)
G1_CAPABILITIES['profile_id'] = G1_PROFILE_ID
G1_DRIVER_ID = 'unitree-g1'
G1_SHADOW_DIGEST = (
    '3a333966ddb1c146c3852e02e90b59825'
    'e6844d6fbd9937502741af3b96a0757'
)
G1_SIGNALING_AUDIENCE = 'motus-teleop-rtc'


def _latency_sample(value: float = 1.0, count: int = 1) -> dict[str, Any]:
    if count == 0:
        return {'last': None, 'p50': None, 'p95': None, 'p99': None, 'count': 0}
    return {
        'last': value,
        'p50': value,
        'p95': value + 1.0,
        'p99': value + 2.0,
        'count': count,
    }


class MutableClock:
    def __init__(self) -> None:
        self.monotonic = 100.0
        self.wall = 1_000.0

    def monotonic_now(self) -> float:
        return self.monotonic

    def wall_now(self) -> float:
        return self.wall

    def advance(self, seconds: float, *, wall_seconds: float | None = None) -> None:
        self.monotonic += seconds
        self.wall += seconds if wall_seconds is None else wall_seconds


class FakeD1Caller:
    """Stateful Shadow producer double with the recording-v1 snapshot shape."""

    def __init__(self, *, lease_seconds: float = 1.0, epoch: int = 7) -> None:
        self.driver_id = DRIVER_ID
        self.boot_id = BOOT_ID
        self.robot_id = ROBOT_ID
        self.capability_digest = DIGEST
        self.lease_seconds = lease_seconds
        self.epoch = epoch
        self.session_id: str | None = None
        self.fence: str | None = None
        self.state = 'idle'
        self.reason: str | None = None
        self.authority_valid = False
        self.dispatch_generation = 0
        self.dispatch_last_admitted: int | None = None
        self.prepare_timeout_after_apply = False
        self.calls: list[dict[str, Any]] = []
        self.script: dict[str, deque[Any]] = defaultdict(deque)
        self._aiohttp_session: aiohttp.ClientSession | None = None

    def queue(self, action: str, *outcomes: Any) -> None:
        self.script[action].extend(outcomes)

    def count(self, action: str) -> int:
        return sum(call['action'] == action for call in self.calls)

    def _assert_identity(self, arguments: dict[str, Any] | None) -> None:
        assert arguments == {
            'boot_id': self.boot_id,
            'session_id': self.session_id,
            'epoch': self.epoch,
            'fence': self.fence,
        }

    def snapshot(self, **overrides: Any) -> dict[str, Any]:
        effective_state = overrides.get('state', self.state)
        effective_reason = overrides.get('reason', self.reason)
        dispatch_state = {
            'idle': 'safe_unarmed',
            'prepared_shadow': 'safe_waiting_frame',
            'active_shadow': 'motion_eligible',
            'paused': 'safe_latched',
            'released': 'safe_revoked',
        }.get(effective_state) if isinstance(effective_state, str) else 'safe_unarmed'
        if effective_state == 'hold':
            if effective_reason == 'soft_stop':
                dispatch_state = 'safe_latched'
            elif effective_reason == 'lease_timeout':
                dispatch_state = 'safe_revoked'
            else:
                dispatch_state = 'safe_reclutch_required'
        if effective_state == 'idle':
            dispatch_decision = 'startup_safe_ack'
        elif effective_state == 'prepared_shadow':
            dispatch_decision = 'prepared_after_stop_ack'
        elif effective_state == 'active_shadow':
            dispatch_decision = 'admitted'
        else:
            dispatch_decision = f'would_stop:{effective_reason}'
        if effective_state == 'active_shadow' and self.dispatch_last_admitted is None:
            self.dispatch_last_admitted = 1
        active_sequence = (
            None
            if isinstance(effective_state, str)
            and effective_state in {'idle', 'prepared_shadow'}
            else self.dispatch_last_admitted
        )
        snapshot = {
            'driver': self.driver_id,
            'driver_id': self.driver_id,
            'driver_name': 'Generic Teleop Shadow Diagnostics',
            'driver_type': 'teleop-shadow',
            'robot_id': self.robot_id,
            'mode': 'shadow',
            'actuation_enabled': False,
            'boot_id': self.boot_id,
            'session_id': self.session_id,
            'epoch': self.epoch,
            'state': self.state,
            'reason': self.reason,
            'authority_valid': self.authority_valid,
            'capability_digest': self.capability_digest,
            'capabilities': {
                'pose_transport': ['webrtc-datachannel'],
                'control_transport': ['mcp'],
            },
            'lease': {
                'source': 'agent-core-mcp-heartbeat-only',
                'timeout_ms': round(self.lease_seconds * 1_000),
                'age_ms': 0.0 if self.authority_valid else None,
                'fresh': self.authority_valid,
                'authority_valid': self.authority_valid,
                'expired_latched': False,
            },
            'pose': {
                'timeout_ms': 250.0,
                'age_ms': None,
                'fresh': False,
                'latest_sequence': None,
                'latest': None,
            },
            'rtc': {
                'connected': False,
                'channels': {
                    'teleop-control': False,
                    'teleop-pose': False,
                },
                'renews_lease': False,
            },
            'dispatch': {
                'kind': 'recording',
                'state': dispatch_state,
                'ready': True,
                'generation': self.dispatch_generation,
                'mailbox_depth': 0,
                'stop_queue_depth': 0,
                'last_admitted_sequence': active_sequence,
                'last_would_apply_sequence': None,
                'last_decision': dispatch_decision,
                'stop_acknowledged': True,
                'fault_code': None,
                'io_inflight': None,
                'counters': {
                    'startup_safe_acks': 1,
                    'stop_acks': self.dispatch_generation + 1,
                },
                'adapter': {
                    'kind': 'recording',
                    'closed': False,
                    'current': {'kind': 'safe', 'reason': 'not_started'},
                    'records': [{'private': 'must-not-be-projected'}],
                },
            },
            'counters': {'lease_heartbeats': self.count('heartbeat')},
        }
        snapshot.update(overrides)
        return snapshot

    async def __call__(
        self,
        driver_id: str,
        action: str,
        arguments: dict[str, Any] | None,
        *,
        timeout_seconds: float,
        session: aiohttp.ClientSession,
    ) -> dict[str, Any]:
        assert driver_id == self.driver_id
        assert isinstance(session, aiohttp.ClientSession)
        assert not session.closed
        if self._aiohttp_session is None:
            self._aiohttp_session = session
        else:
            assert session is self._aiohttp_session
        self.calls.append({
            'action': action,
            'arguments': deepcopy(arguments),
            'timeout_seconds': timeout_seconds,
        })

        if self.script[action]:
            outcome = self.script[action].popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                outcome = outcome(self, arguments)
                if asyncio.iscoroutine(outcome):
                    outcome = await outcome
                if isinstance(outcome, BaseException):
                    raise outcome
            return deepcopy(outcome)

        if action == 'status':
            return self.snapshot()
        if action == 'prepare_shadow':
            assert arguments is not None
            assert set(arguments) == {'session_id', 'epoch', 'fence'}
            assert str(uuid.UUID(arguments['session_id'])) == arguments['session_id']
            assert arguments['epoch'] > self.epoch
            self.session_id = arguments['session_id']
            self.epoch = arguments['epoch']
            self.fence = arguments['fence']
            self.state = 'prepared_shadow'
            self.reason = None
            self.authority_valid = True
            self.dispatch_generation += 1
            self.dispatch_last_admitted = None
            if self.prepare_timeout_after_apply:
                self.prepare_timeout_after_apply = False
                raise mcp_client.TrustedShadowTransportError('timeout')
            return self.snapshot()
        if action == 'heartbeat':
            self._assert_identity(arguments)
            return self.snapshot()
        if action == 'pause':
            self._assert_identity(arguments)
            self.state = 'paused'
            self.reason = 'operator_pause'
            self.dispatch_generation += 1
            return self.snapshot()
        if action == 'soft_stop':
            self._assert_identity(arguments)
            self.state = 'hold'
            self.reason = 'soft_stop'
            self.dispatch_generation += 1
            return self.snapshot()
        if action == 'release':
            self._assert_identity(arguments)
            self.session_id = None
            self.fence = None
            self.state = 'released'
            self.reason = 'operator_release'
            self.authority_valid = False
            self.dispatch_generation += 1
            return self.snapshot()
        if action == 'stop':
            assert arguments in (None, {})
            self.session_id = None
            self.fence = None
            self.state = 'released'
            self.reason = 'lifecycle_stop'
            self.authority_valid = False
            self.dispatch_generation += 1
            return self.snapshot()
        raise AssertionError(f'unexpected action: {action}')


class FakeG1ShadowCaller(FakeD1Caller):
    """Stateful double using the adapter-neutral G1 Shadow wire contract."""

    def __init__(self, *, lease_seconds: float = 10.0, epoch: int = 7) -> None:
        super().__init__(lease_seconds=lease_seconds, epoch=epoch)
        self.driver_id = G1_DRIVER_ID
        self.robot_id = G1_DRIVER_ID
        self.capability_digest = G1_SHADOW_DIGEST
        self.rtc_connected = False
        self.frames_received = 0
        self.latest_sequence: int | None = None

    def receive_direct_rtc_frames(self, sequence: int = 3) -> None:
        """Model Browser -> Driver frames without creating a Core RPC action."""

        self.rtc_connected = True
        self.frames_received = sequence
        self.latest_sequence = sequence
        self.dispatch_last_admitted = sequence
        self.state = 'active_shadow'

    def latch_projection_fault(self) -> None:
        self.rtc_connected = False
        self.session_id = None
        self.fence = None
        self.state = 'fault'
        self.reason = 'dispatch_fault'
        self.authority_valid = False
        self.dispatch_generation += 1

    def snapshot(self, **overrides: Any) -> dict[str, Any]:
        raw = super().snapshot(**overrides)
        state = raw['state']
        rtc_open = self.rtc_connected and state == 'active_shadow'
        latency_count = 1 if self.frames_received else 0
        raw.update({
            'driver': self.driver_id,
            'driver_id': self.driver_id,
            'driver_name': 'Unitree G1 Bundle',
            'driver_type': 'teleop',
            'robot_id': self.robot_id,
            'profile_id': G1_PROFILE_ID,
            'capability_digest': self.capability_digest,
            'capabilities': deepcopy(G1_CAPABILITIES),
            'diagnostics': {
                'transport': {
                    'rtc_rtt_ms': None,
                    'pose_age_ms': 4.0 if rtc_open else None,
                    'frame_rate_hz': 60.0 if rtc_open else 0.0,
                    'frames_received': self.frames_received,
                    'frames_rejected': 0,
                    'sequence_gaps': 0,
                    'mailbox_replacements': 0,
                },
                'latency_ms': {
                    'receive_to_admit': _latency_sample(1.0, latency_count),
                    'mailbox_wait': _latency_sample(2.0, latency_count),
                    'ik': _latency_sample(3.0, latency_count),
                    'adapter_apply': _latency_sample(4.0, latency_count),
                    'robot_follow': _latency_sample(count=0),
                },
            },
            'output': {
                'profile_id': G1_PROFILE_ID,
                'hardware_output': False,
                'state': 'tracking' if state == 'active_shadow' else state,
                'target_joint_positions_rad': [0.1] * 10,
                'measured_joint_positions_rad': [0.09] * 10,
                'max_abs_error_rad': 0.01 if self.frames_received else 0.0,
                'arm_sdk_weight': None,
                'command_age_ms': 4.0 if rtc_open else None,
                'fault_reason': None,
            },
        })
        raw['rtc'].update({
            'connected': rtc_open,
            'channels': {
                'teleop-control': rtc_open,
                'teleop-pose': rtc_open,
            },
        })
        raw['pose'].update({
            'age_ms': 4.0 if rtc_open else None,
            'fresh': rtc_open,
            'latest_sequence': self.latest_sequence if rtc_open else None,
        })
        adapter = raw['dispatch']['adapter']
        adapter.update({
            'hardware_output': False,
            'actuation_enabled': False,
        })
        if state == 'active_shadow':
            raw['dispatch'].update({
                'last_would_apply_sequence': self.latest_sequence,
                'last_decision': 'would_apply',
            })
            adapter['current'] = {'kind': 'would_apply'}
        elif state in {'hold', 'paused', 'released'}:
            raw['dispatch']['last_would_apply_sequence'] = self.latest_sequence
            adapter['current'] = {'kind': 'would_stop'}
        else:
            adapter['current'] = {'kind': 'safe'}
        if state == 'fault':
            raw['lease'].update({
                'age_ms': None,
                'fresh': False,
                'authority_valid': False,
                'expired_latched': False,
            })
            raw['dispatch'].update({
                'state': 'fault_latched',
                'ready': False,
                'mailbox_depth': 0,
                'stop_queue_depth': 0,
                'last_would_apply_sequence': self.latest_sequence,
                'last_decision': 'async_fault:projection_or_ik_failed',
                'stop_acknowledged': False,
                'fault_code': 'projection_or_ik_failed',
                'io_inflight': None,
            })
            adapter['current'] = {'kind': 'would_stop'}
            raw['output'].update({
                'state': 'fault',
                'fault_reason': 'projection_or_ik_failed',
            })
        return raw


def _live_active_snapshot(
    fake: FakeD1Caller,
    *,
    published_sequence: int | None = 9,
) -> dict[str, Any]:
    fake.state = 'active_shadow'
    fake.authority_valid = True
    fake.session_id = str(uuid.uuid4())
    fake.dispatch_generation = 3
    fake.dispatch_last_admitted = 9
    raw = fake.snapshot()
    raw.update({
        'driver_type': 'teleop',
        'mode': 'live',
        'actuation_enabled': True,
        'state': 'active_live',
        'diagnostics': {
            'transport': {
                'rtc_rtt_ms': 8.0,
                'pose_age_ms': 4.0,
                'frame_rate_hz': 60.0,
                'frames_received': 100,
                'frames_rejected': 2,
                'sequence_gaps': 1,
                'mailbox_replacements': 3,
            },
            'latency_ms': {
                'receive_to_admit': _latency_sample(1.0),
                'mailbox_wait': _latency_sample(2.0),
                'ik': _latency_sample(3.0),
                'adapter_apply': _latency_sample(4.0),
                'robot_follow': _latency_sample(5.0),
            },
        },
        'output': {
            'profile_id': LIVE_PROFILE_ID,
            'hardware_output': True,
            'state': 'tracking',
            'target_joint_positions_rad': [0.1] * 10,
            'measured_joint_positions_rad': [0.09] * 10,
            'max_abs_error_rad': 0.01,
            'arm_sdk_weight': 1.0,
            'command_age_ms': 5.0,
            'fault_reason': None,
        },
    })
    raw['dispatch'].update({
        'kind': 'hardware',
        'last_published_sequence': published_sequence,
        'last_decision': 'published',
    })
    raw['dispatch'].pop('last_would_apply_sequence')
    raw['dispatch'].pop('adapter')
    return raw


def _live_fault_snapshot(fake: FakeD1Caller) -> dict[str, Any]:
    """Mirror the bounded terminal status emitted by the G1 live runtime."""

    raw = _live_active_snapshot(fake, published_sequence=7)
    raw.update({
        'session_id': None,
        'state': 'fault',
        'reason': 'dispatch_fault',
        'authority_valid': False,
    })
    raw['lease'].update({
        'age_ms': None,
        'fresh': False,
        'authority_valid': False,
        'expired_latched': False,
    })
    raw['pose'].update({'age_ms': None, 'fresh': False, 'latest_sequence': None})
    raw['rtc'].update({
        'connected': False,
        'channels': {'teleop-control': False, 'teleop-pose': False},
    })
    raw['dispatch'].update({
        'state': 'fault_latched',
        'ready': False,
        'generation': 4,
        'mailbox_depth': 0,
        'stop_queue_depth': 0,
        'last_admitted_sequence': 9,
        'last_published_sequence': 7,
        'last_decision': 'async_fault:arm_sdk_async_fault',
        'stop_acknowledged': False,
        'fault_code': 'arm_sdk_async_fault',
        'io_inflight': None,
    })
    raw['output'].update({
        'state': 'fault',
        'arm_sdk_weight': 0.0,
        'fault_reason': 'arm_sdk_async_fault',
    })
    return raw


def _g1_shadow_snapshot(state: str, reason: str | None) -> dict[str, Any]:
    """Adapter-neutral shape of the G1 Shadow runtime across lifecycle states."""

    raw = _live_active_snapshot(FakeD1Caller(), published_sequence=7)
    authority = state in {'prepared_shadow', 'active_shadow', 'hold', 'paused'}
    terminal = state in {'idle', 'released', 'fault'}
    raw.update({
        'mode': 'shadow',
        'actuation_enabled': False,
        'session_id': None if terminal else raw['session_id'],
        'state': state,
        'reason': reason,
        'authority_valid': authority,
    })
    raw['lease'].update({
        'age_ms': 0.0 if authority else None,
        'fresh': authority,
        'authority_valid': authority,
        'expired_latched': False,
    })
    if terminal:
        raw['pose'].update({'age_ms': None, 'fresh': False, 'latest_sequence': None})
        raw['rtc'].update({
            'connected': False,
            'channels': {'teleop-control': False, 'teleop-pose': False},
        })
    dispatch_state = {
        'idle': 'safe_unarmed',
        'prepared_shadow': 'safe_waiting_frame',
        'active_shadow': 'motion_eligible',
        'hold': 'safe_reclutch_required',
        'paused': 'safe_latched',
        'released': 'safe_revoked',
        'fault': 'fault_latched',
    }[state]
    decision = {
        'idle': 'startup_safe_ack',
        'prepared_shadow': 'prepared_after_stop_ack',
        'active_shadow': 'would_apply',
        'hold': f'would_stop:{reason}',
        'paused': 'would_stop:operator_pause',
        'released': 'would_stop:operator_release',
        'fault': 'async_fault:projection_or_ik_failed',
    }[state]
    has_sequence = state in {'active_shadow', 'hold', 'paused', 'released', 'fault'}
    raw['dispatch'].update({
        'kind': 'recording',
        'state': dispatch_state,
        'ready': state != 'fault',
        'last_admitted_sequence': 9 if has_sequence else None,
        'last_would_apply_sequence': 7 if has_sequence else None,
        'last_decision': decision,
        'stop_acknowledged': state != 'fault',
        'fault_code': 'projection_or_ik_failed' if state == 'fault' else None,
        'io_inflight': None,
        'mailbox_depth': 0,
        'stop_queue_depth': 0,
        'adapter': {
            'kind': 'recording',
            'closed': False,
            'hardware_output': False,
            'actuation_enabled': False,
            'current': {
                'kind': (
                    'would_apply'
                    if state == 'active_shadow'
                    else ('would_stop' if state in {'hold', 'paused', 'released', 'fault'} else 'safe')
                ),
            },
            'records': [],
        },
    })
    raw['dispatch'].pop('last_published_sequence')
    raw['output'].update({
        'profile_id': G1_PROFILE_ID,
        'hardware_output': False,
        'state': 'fault' if state == 'fault' else state,
        'arm_sdk_weight': None,
        'fault_reason': 'projection_or_ik_failed' if state == 'fault' else None,
    })
    return raw


def _project_live(raw: dict[str, Any]) -> dict[str, Any]:
    projected, _ = _project_driver_snapshot(
        raw,
        driver_id=DRIVER_ID,
        robot_id=ROBOT_ID,
        capability_digest=DIGEST,
        action='status',
        expected_mode='live',
        expected_profile_id=LIVE_PROFILE_ID,
        expected_capabilities=LIVE_CAPABILITIES,
    )
    return projected


def _project_g1_shadow(raw: dict[str, Any]) -> dict[str, Any]:
    projected, _ = _project_driver_snapshot(
        raw,
        driver_id=DRIVER_ID,
        robot_id=ROBOT_ID,
        capability_digest=DIGEST,
        action='status',
        expected_mode='shadow',
        expected_profile_id=G1_PROFILE_ID,
        expected_capabilities=G1_CAPABILITIES,
    )
    return projected


@pytest.fixture(autouse=True)
def quiet_audit(monkeypatch: pytest.MonkeyPatch):
    async def emit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(audit, 'emit', emit)


def _install_descriptor(shadow_session_tool: dict[str, Any]) -> None:
    tool = deepcopy(shadow_session_tool)
    tool['x-teleop'].update({
        'driver_id': DRIVER_ID,
        'capability_digest': DIGEST,
    })
    config.main['services'] = {'mcp': [{
        'id': DRIVER_ID,
        'name': 'Generic Teleop Shadow Diagnostics',
        'url': 'http://teleop-shadow.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'reported_robot_id': ROBOT_ID,
        'authority_domain': ROBOT_ID,
        'tools': [tool],
    }, {
        'id': ROBOT_ID,
        'name': 'Robot Fixture Actuator',
        'url': 'http://robot-fixture.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [{
            'name': 'locomotion',
            'type': 'actuator',
            'inputSchema': {'type': 'object'},
        }],
    }]}


def _install_g1_shadow_descriptor() -> tuple[dict[str, Any], dict[str, Any]]:
    """Install the authority-relevant fields emitted by the G1 Driver."""

    identity = {
        'boot_id': {'type': 'string', 'format': 'uuid'},
        'session_id': {'type': 'string', 'format': 'uuid'},
        'epoch': {'type': 'integer', 'minimum': 1},
        'fence': {'type': 'string', 'minLength': 24},
    }
    identity_params = list(identity)
    actions = {
        'stop': {'params': []},
        'prepare_shadow': {'params': ['session_id', 'epoch', 'fence']},
        'heartbeat': {'params': identity_params},
        'pause': {'params': identity_params},
        'release': {'params': identity_params},
        'soft_stop': {'params': identity_params},
        'status': {'params': []},
    }
    descriptor = {
        'protocol': 'motus.teleop.shadow.v1',
        'mode': 'shadow',
        'profile_id': G1_PROFILE_ID,
        'capabilities': deepcopy(G1_CAPABILITIES),
        'dispatch_contract': 'motus.teleop.dispatch.recording.v1',
        'signaling': {
            'protocol': 'motus.teleop.webrtc-offer-answer.v1',
            'path': '/offer',
            'access': 'authenticated-core-proxy-only',
            'audience': G1_SIGNALING_AUDIENCE,
        },
        'driver_id': G1_DRIVER_ID,
        'driver_name': 'Unitree G1 Bundle',
        'robot_id': G1_DRIVER_ID,
        'actuation_enabled': False,
        'capability_digest': G1_SHADOW_DIGEST,
    }
    session_tool = {
        'name': 'teleop_session',
        'type': 'actuator',
        'multiInstance': False,
        'description': 'Unitree G1_23 dual-arm controller teleoperation.',
        'annotations': {'destructiveHint': False, 'idempotentHint': False},
        'inputSchema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                'action': {'type': 'string', 'enum': list(actions)},
                **identity,
            },
            'required': ['action'],
            'x-action-params': actions,
        },
        'x-teleop': deepcopy(descriptor),
    }
    state_tool = {
        'name': 'teleop_state',
        'type': 'resource',
        'multiInstance': False,
        'readOnly': True,
        'description': 'Read-only G1 teleoperation state and diagnostics.',
        'annotations': {
            'readOnlyHint': True,
            'destructiveHint': False,
            'idempotentHint': True,
        },
        'inputSchema': {
            'type': 'object',
            'properties': {},
            'additionalProperties': False,
        },
        'x-teleop': deepcopy(descriptor),
    }
    record = {
        'id': G1_DRIVER_ID,
        'name': 'Unitree G1 Bundle',
        'server_name': 'unitree-g1',
        'url': 'https://unitree-g1.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'reported_robot_id': G1_DRIVER_ID,
        'authority_domain': G1_DRIVER_ID,
        'tools': [session_tool, state_tool],
    }
    config.main['services'] = {'mcp': [record]}
    mcp_client.registry[G1_DRIVER_ID] = {
        'online': True,
        'trusted': True,
        'url': record['url'],
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(session_tool),
    }
    return record, session_tool


def _coordinator(
    fake: FakeD1Caller,
    clock: MutableClock,
    *,
    command_broker: CommandBroker | None = None,
) -> TeleopCoordinator:
    manager = ShadowSessionManager(
        monotonic=clock.monotonic_now,
        wall_clock=clock.wall_now,
    )
    return TeleopCoordinator(
        session_manager=manager,
        caller=fake,
        monotonic=clock.monotonic_now,
        wall_clock=clock.wall_now,
        command_broker=command_broker or CommandBroker(),
    )


async def _finish(coordinator: TeleopCoordinator) -> None:
    """Release live fixtures before stop so cleanup does not add test calls."""

    sessions = await coordinator.manager.list_visible('', owner=True)
    for session in sessions:
        if session.state in {'preparing', 'active', 'paused', 'hold'}:
            try:
                await coordinator.release(
                    session.id,
                    session.principal_id,
                    CLIENT_ID,
                    owner=False,
                )
            except Exception:  # noqa: BLE001, S110 -- teardown must close aiohttp
                pass
    await coordinator.stop()


async def _crash_without_authority_cleanup(
    coordinator: TeleopCoordinator,
    fake: FakeD1Caller,
) -> None:
    """Drop process-owned tasks/transports without release or guard deletion."""

    tasks = [
        *coordinator._heartbeat_tasks.values(),
        *coordinator._supervisor_tasks,
        *coordinator._safety_tasks,
    ]
    if coordinator._reaper_task is not None:
        tasks.append(coordinator._reaper_task)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    coordinator._heartbeat_tasks.clear()
    coordinator._supervisor_tasks.clear()
    coordinator._safety_tasks.clear()
    coordinator._reaper_task = None
    if coordinator._http_session is not None:
        await coordinator._http_session.close()
        coordinator._http_session = None
    fake._aiohttp_session = None


async def _wait_until(predicate, *, timeout: float = 0.8) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError('condition was not reached before the test deadline')
        await asyncio.sleep(0.01)


def test_acquire_uses_driver_epoch_floor_immediate_heartbeat_and_safe_status(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(epoch=41, lease_seconds=10.0)
        coordinator = _coordinator(fake, clock)
        try:
            result = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            public = coordinator.public_session(result.session)

            assert result.disposition == 'created'
            assert result.session.driver_id == DRIVER_ID
            assert result.session.robot_id == ROBOT_ID
            assert result.session.epoch >= 42
            assert public['configured_dry_run_profile'] == 'recording'
            assert [call['action'] for call in fake.calls[:3]] == [
                'status', 'prepare_shadow', 'heartbeat',
            ]
            assert fake.calls[1]['arguments']['epoch'] == result.session.epoch
            assert fake.calls[2]['arguments']['session_id'] == result.session.id
            assert public['driver']['lease']['source'] == (
                'agent-core-mcp-heartbeat-only'
            )
            assert public['driver']['rtc']['renews_lease'] is False
            assert public['driver']['dispatch'] == {
                'contract': 'motus.teleop.dispatch.recording.v1',
                'kind': 'recording',
                'state': 'safe_waiting_frame',
                'ready': True,
                'generation': 1,
                'mailbox_depth': 0,
                'stop_queue_depth': 0,
                'last_admitted_sequence': None,
                'last_would_apply_sequence': None,
                'last_decision': 'prepared_after_stop_ack',
                'stop_acknowledged': True,
                'fault_code': None,
                'io_inflight': None,
                'counters': {
                    'startup_safe_acks': 1,
                    'stop_acks': 2,
                },
                'dry_run': {
                    'profile': 'recording',
                    'hardware_output': False,
                    'actuation_enabled': False,
                    'operation': 'recording',
                    'sequence': None,
                    'requested': None,
                    'effective': None,
                    'clamped_axes': [],
                    'stop_reason': None,
                    'gait_policy': None,
                },
            }
            assert 'adapter' not in public['driver']['dispatch']
            assert 'must-not-be-projected' not in repr(public)
            assert public['driver_heartbeat']['state'] == 'healthy'
            assert result.session.fence not in repr(public)
            assert 'fence' not in repr(public).lower()
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_trusted_g1_shadow_directory_to_rtc_hold_and_fault_stays_generic():
    async def scenario() -> None:
        record, session_tool = _install_g1_shadow_descriptor()
        directory_entry = teleop._robot_view(record, [record])
        declared_actions = session_tool['inputSchema']['properties']['action']['enum']

        assert directory_entry['teleop_ready'] is True
        assert directory_entry['robot_id'] == G1_DRIVER_ID
        assert directory_entry['tools'] == ['teleop_session', 'teleop_state']
        assert directory_entry['teleop']['mode'] == 'shadow'
        assert directory_entry['teleop']['profile_id'] == G1_PROFILE_ID
        assert directory_entry['teleop']['capabilities']['effectors'] == ['dual_arm']
        assert directory_entry['teleop']['capabilities']['outputs']['base']['enabled'] is False
        assert directory_entry['teleop']['capabilities']['outputs']['hands']['enabled'] is False
        assert directory_entry['teleop']['signaling']['audience'] == G1_SIGNALING_AUDIENCE
        assert 'submit_shadow_frame' not in declared_actions

        clock = MutableClock()
        fake = FakeG1ShadowCaller()
        observed_offer: dict[str, Any] = {}

        async def signaler(
            driver_id: str,
            offer: dict[str, Any],
            ticket: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            observed_offer.update({
                'driver_id': driver_id,
                'offer': deepcopy(offer),
                'ticket': ticket,
                'kwargs': kwargs,
            })
            fake.receive_direct_rtc_frames()
            return {
                'sdp': 'v=0\r\na=g1-shadow-answer',
                'type': 'answer',
                'boot_id': fake.boot_id,
                'session_id': fake.session_id,
                'epoch': fake.epoch,
                'capability_digest': fake.capability_digest,
                'mode': 'shadow',
                'actuation_enabled': False,
            }

        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(
                monotonic=clock.monotonic_now,
                wall_clock=clock.wall_now,
            ),
            caller=fake,
            signaler=signaler,
            monotonic=clock.monotonic_now,
            wall_clock=clock.wall_now,
            command_broker=CommandBroker(),
        )
        try:
            acquired = await coordinator.acquire(
                G1_DRIVER_ID,
                'alice',
                CLIENT_ID,
                mode=directory_entry['teleop']['mode'],
            )
            session = acquired.session
            public = coordinator.public_session(session)

            assert acquired.disposition == 'created'
            assert [call['action'] for call in fake.calls[:3]] == [
                'status',
                'prepare_shadow',
                'heartbeat',
            ]
            assert public['mode'] == 'shadow'
            assert public['profile_id'] == G1_PROFILE_ID
            assert public['effectors'] == ['dual_arm']
            assert public['driver']['profile_id'] == G1_PROFILE_ID
            assert public['driver']['output']['hardware_output'] is False
            assert len(public['driver']['output']['target_joint_positions_rad']) == 10

            fingerprint = mcp_client.teleop_tool_fingerprint(session_tool)
            assert fingerprint is not None
            target = mcp_client.TrustedShadowTarget(
                mcp_id=G1_DRIVER_ID,
                url=record['url'],
                capability_digest=G1_SHADOW_DIGEST,
                descriptor_fingerprint=fingerprint,
                actions=frozenset(declared_actions),
            )
            coordinator._pinned_targets[session.id] = target
            offer = {'type': 'offer', 'sdp': 'v=0\r\no=quest-3-g1-shadow'}
            answer = await coordinator.signaling_offer(
                session.id,
                'alice',
                CLIENT_ID,
                offer,
            )

            assert answer == {'sdp': 'v=0\r\na=g1-shadow-answer', 'type': 'answer'}
            assert observed_offer['driver_id'] == G1_DRIVER_ID
            assert observed_offer['offer'] == offer
            assert observed_offer['kwargs']['target'] == target
            ticket_payload = observed_offer['ticket'].split('.', 1)[0]
            ticket_claims = json.loads(base64.urlsafe_b64decode(
                ticket_payload + '=' * (-len(ticket_payload) % 4),
            ))
            assert ticket_claims['aud'] == G1_SIGNALING_AUDIENCE
            assert ticket_claims['capability_digest'] == G1_SHADOW_DIGEST

            rtc_status = await coordinator.status(session.id, 'alice', owner=False)
            driver = rtc_status['driver']
            assert driver['state'] == 'active_shadow'
            assert driver['rtc'] == {
                'connected': True,
                'channels': {
                    'teleop-control': True,
                    'teleop-pose': True,
                },
                'renews_lease': False,
            }
            assert driver['diagnostics']['transport']['frames_received'] == 3
            assert driver['diagnostics']['latency_ms']['ik']['p95'] == 4.0
            assert driver['dispatch']['last_would_apply_sequence'] == 3
            assert driver['output']['hardware_output'] is False
            assert all(call['action'] != 'submit_shadow_frame' for call in fake.calls)
            assert all(
                'frame' not in (call['arguments'] or {})
                for call in fake.calls
            )

            held = await coordinator.soft_stop(
                session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            held_public = coordinator.public_session(held)
            assert held_public['state'] == 'hold'
            assert held_public['driver']['state'] == 'hold'
            assert held_public['driver']['reason'] == 'soft_stop'
            assert held_public['driver']['rtc']['connected'] is False
            assert held_public['driver']['dispatch']['state'] == 'safe_latched'
            assert held_public['driver']['output']['hardware_output'] is False

            fake.latch_projection_fault()
            faulted = await coordinator.status(session.id, 'alice', owner=False)
            assert faulted['state'] == 'faulted'
            assert faulted['driver_heartbeat']['state'] == 'faulted'
            assert faulted['driver']['state'] == 'fault'
            assert faulted['driver']['reason'] == 'dispatch_fault'
            assert faulted['driver']['dispatch']['state'] == 'fault_latched'
            assert faulted['driver']['dispatch']['fault_code'] == (
                'projection_or_ik_failed'
            )
            assert faulted['driver']['output']['fault_reason'] == (
                'projection_or_ik_failed'
            )
            public_surface = repr({
                'directory': directory_entry,
                'answer': answer,
                'status': faulted,
            })
            assert observed_offer['ticket'] not in public_surface
            assert session.fence not in public_surface
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_acquire_rejects_descriptor_without_lifecycle_stop_contract(
    shadow_session_tool,
):
    async def scenario() -> None:
        incomplete = deepcopy(shadow_session_tool)
        incomplete['inputSchema']['properties']['action']['enum'].remove('stop')
        incomplete['inputSchema']['x-action-params'].pop('stop')
        _install_descriptor(incomplete)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert raised.value.code == 'driver_not_ready'
            assert fake.calls == []
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_acquire_rejects_legacy_vendor_profile_before_driver_contact(
    shadow_session_tool,
):
    async def scenario() -> None:
        descriptor = deepcopy(shadow_session_tool)
        descriptor['x-teleop']['dry_run_profile'] = 'vendor_specific_profile'
        _install_descriptor(descriptor)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert raised.value.code == 'driver_not_ready'
            assert fake.calls == []
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_live_output_and_latency_are_projected_without_adapter_internals():
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake)
    raw['private_adapter_state'] = {
        'sdk_call': 'must-not-be-projected',
        'joint_mapping': 'must-not-be-projected',
    }

    projected = _project_live(raw)

    assert projected['mode'] == 'live'
    assert projected['actuation_enabled'] is True
    assert projected['dispatch']['contract'] == 'motus.teleop.dispatch.hardware.v1'
    assert projected['dispatch']['kind'] == 'hardware'
    assert projected['dispatch']['last_admitted_sequence'] == 9
    assert projected['dispatch']['last_published_sequence'] == 9
    assert projected['output']['profile_id'] == LIVE_PROFILE_ID
    assert projected['output']['hardware_output'] is True
    assert len(projected['output']['target_joint_positions_rad']) == 10
    assert projected['diagnostics']['latency_ms']['ik'] == _latency_sample(3.0)
    serialized = repr(projected)
    for private_value in (
        'must-not-be-projected',
        'sdk_call',
        'joint_mapping',
    ):
        assert private_value not in serialized


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        (('output', 'hardware_output'), False),
        (('output', 'profile_id'), 'different_profile'),
        (('output', 'state'), '<script>unsafe</script>'),
        (('output', 'target_joint_positions_rad', 0), float('nan')),
        (('output', 'measured_joint_positions_rad'), [0.1] * 9),
        (('output', 'max_abs_error_rad'), 2_000.0),
        (('output', 'arm_sdk_weight'), None),
        (('diagnostics', 'transport', 'pose_age_ms'), -1.0),
        (('diagnostics', 'latency_ms', 'ik', 'p50'), 10.0),
        (('diagnostics', 'latency_ms', 'ik', 'count'), 0),
    ],
    ids=[
        'hardware-output',
        'profile',
        'unsafe-state',
        'joint-nan',
        'joint-count-mismatch',
        'error-out-of-range',
        'missing-live-arm-weight',
        'negative-pose-age',
        'percentiles-out-of-order',
        'zero-count-with-samples',
    ],
)
def test_live_output_and_diagnostics_fail_closed(path, value):
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake)
    target: Any = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(TeleopServiceError) as raised:
        _project_live(raw)

    assert raised.value.code == 'driver_response_invalid'


@pytest.mark.parametrize('joint_count', [0, 9, 11])
def test_live_authority_output_requires_exact_capability_joint_count(joint_count):
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake)
    raw['output']['target_joint_positions_rad'] = [0.1] * joint_count
    raw['output']['measured_joint_positions_rad'] = [0.09] * joint_count

    with pytest.raises(TeleopServiceError) as raised:
        _project_live(raw)

    assert raised.value.code == 'driver_response_invalid'


@pytest.mark.parametrize('joint_count', [0, 10])
def test_live_idle_output_allows_empty_or_declared_joint_vector(joint_count):
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake)
    raw.update({
        'session_id': None,
        'state': 'idle',
        'reason': None,
        'authority_valid': False,
    })
    raw['lease'].update({'age_ms': None, 'fresh': False, 'authority_valid': False})
    raw['dispatch'].update({
        'state': 'safe_unarmed',
        'last_admitted_sequence': None,
        'last_published_sequence': None,
        'last_decision': 'startup_safe_ack',
    })
    raw['output'].update({
        'state': 'idle',
        'target_joint_positions_rad': [0.1] * joint_count,
        'measured_joint_positions_rad': [0.09] * joint_count,
        'max_abs_error_rad': None,
        'arm_sdk_weight': 0.0,
        'command_age_ms': None,
    })

    projected = _project_live(raw)

    assert len(projected['output']['target_joint_positions_rad']) == joint_count


@pytest.mark.parametrize(
    ('state', 'reason', 'dispatch_state', 'decision'),
    [
        ('prepared_live', None, 'safe_waiting_frame', 'prepared_after_stop_ack'),
        ('paused', 'operator_pause', 'safe_latched', 'would_stop:operator_pause'),
        ('hold', 'deadman_released', 'safe_reclutch_required', 'would_stop:deadman_released'),
        ('hold', 'command_timeout', 'safe_reclutch_required', 'would_stop:command_timeout'),
    ],
)
def test_every_live_authority_state_requires_declared_joint_vector(
    state,
    reason,
    dispatch_state,
    decision,
):
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake)
    raw.update({'state': state, 'reason': reason})
    raw['dispatch'].update({
        'state': dispatch_state,
        'last_decision': decision,
        'last_admitted_sequence': None if state == 'prepared_live' else 9,
        'last_published_sequence': None if state == 'prepared_live' else 9,
    })
    raw['output'].update({
        'state': state,
        'target_joint_positions_rad': [],
        'measured_joint_positions_rad': [],
    })

    with pytest.raises(TeleopServiceError) as raised:
        _project_live(raw)
    assert raised.value.code == 'driver_response_invalid'

    raw['output']['target_joint_positions_rad'] = [0.1] * 10
    raw['output']['measured_joint_positions_rad'] = [0.09] * 10
    projected = _project_live(raw)
    assert len(projected['output']['target_joint_positions_rad']) == 10


def test_live_hold_is_visible_without_joint_semantics():
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake)
    raw.update({'state': 'hold', 'reason': 'deadman_released'})
    raw['dispatch'].update({
        'state': 'safe_reclutch_required',
        'last_decision': 'would_stop:deadman_released',
    })
    raw['output'].update({
        'state': 'hold',
        'fault_reason': 'deadman_released',
        'command_age_ms': None,
    })

    projected = _project_live(raw)

    assert projected['state'] == 'hold'
    assert projected['dispatch']['state'] == 'safe_reclutch_required'
    assert projected['output']['state'] == 'hold'
    assert projected['output']['fault_reason'] == 'deadman_released'
    assert projected['output']['hardware_output'] is True


def test_g1_live_terminal_dispatch_fault_fixture_is_projected_exactly():
    raw = _live_fault_snapshot(FakeD1Caller())

    projected = _project_live(raw)

    assert projected['state'] == 'fault'
    assert projected['reason'] == 'dispatch_fault'
    assert projected['authority_valid'] is False
    assert projected['session_id'] is None
    assert projected['lease']['fresh'] is False
    assert projected['dispatch']['state'] == 'fault_latched'
    assert projected['dispatch']['ready'] is False
    assert projected['dispatch']['stop_acknowledged'] is False
    assert projected['dispatch']['fault_code'] == 'arm_sdk_async_fault'
    assert projected['dispatch']['last_published_sequence'] == 7
    assert projected['output']['state'] == 'fault'
    assert projected['output']['arm_sdk_weight'] == 0.0
    assert projected['output']['fault_reason'] == 'arm_sdk_async_fault'


def test_g1_live_terminal_fault_accepts_confirmed_safe_stop_variant():
    raw = _live_fault_snapshot(FakeD1Caller())
    raw['dispatch'].update({
        'stop_acknowledged': True,
        'last_decision': 'would_stop:adapter_fault',
    })

    projected = _project_live(raw)

    assert projected['dispatch']['stop_acknowledged'] is True
    assert projected['dispatch']['fault_code'] == 'arm_sdk_async_fault'


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        (('reason',), 'lease_timeout'),
        (('session_id',), str(uuid.uuid4())),
        (('authority_valid',), True),
        (('lease', 'age_ms'), 1.0),
        (('dispatch', 'state'), 'safe_revoked'),
        (('dispatch', 'ready'), True),
        (('dispatch', 'stop_acknowledged'), True),
        (('dispatch', 'fault_code'), None),
        (('output', 'state'), 'stopped'),
        (('output', 'arm_sdk_weight'), 0.1),
        (('output', 'fault_reason'), None),
        (('output', 'target_joint_positions_rad'), []),
        (('pose', 'latest_sequence'), 9),
        (('rtc', 'connected'), True),
    ],
)
def test_g1_live_terminal_dispatch_fault_rejects_contradictory_shapes(path, value):
    raw = _live_fault_snapshot(FakeD1Caller())
    target: Any = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path == ('authority_valid',):
        raw['lease']['authority_valid'] = value
    if path == ('rtc', 'connected'):
        raw['rtc']['channels'] = {'teleop-control': True, 'teleop-pose': True}

    with pytest.raises(TeleopServiceError) as raised:
        _project_live(raw)
    assert raised.value.code == 'driver_response_invalid'


@pytest.mark.parametrize(
    ('state', 'reason'),
    [
        ('idle', None),
        ('prepared_shadow', None),
        ('active_shadow', None),
        ('hold', 'deadman_released'),
        ('hold', 'command_timeout'),
        ('hold', 'intent_expired'),
        ('paused', 'operator_pause'),
        ('released', 'operator_release'),
        ('fault', 'dispatch_fault'),
    ],
)
def test_g1_shadow_profile_is_orthogonal_to_standard_recording_envelope(
    state,
    reason,
):
    projected = _project_g1_shadow(_g1_shadow_snapshot(state, reason))

    assert projected['profile_id'] == G1_PROFILE_ID
    assert projected['dispatch']['kind'] == 'recording'
    assert projected['dispatch']['dry_run']['profile'] == 'recording'
    assert projected['output']['profile_id'] == G1_PROFILE_ID
    assert projected['output']['hardware_output'] is False
    if state == 'fault':
        assert projected['dispatch']['state'] == 'fault_latched'
        assert projected['dispatch']['ready'] is False
        assert projected['dispatch']['fault_code'] == 'projection_or_ik_failed'
        assert projected['output']['state'] == 'fault'
        assert projected['output']['arm_sdk_weight'] is None
        assert projected['output']['fault_reason'] == 'projection_or_ik_failed'


def test_g1_shadow_terminal_fault_accepts_confirmed_recording_stop_variant():
    raw = _g1_shadow_snapshot('fault', 'dispatch_fault')
    raw['dispatch'].update({
        'stop_acknowledged': True,
        'last_decision': 'would_stop:adapter_fault',
    })

    projected = _project_g1_shadow(raw)

    assert projected['dispatch']['stop_acknowledged'] is True
    assert projected['dispatch']['fault_code'] == 'projection_or_ik_failed'


def test_g1_shadow_rejects_hardware_arm_weight_in_non_fault_state():
    raw = _g1_shadow_snapshot('active_shadow', None)
    raw['output']['arm_sdk_weight'] = 0.5

    with pytest.raises(TeleopServiceError) as raised:
        _project_g1_shadow(raw)
    assert raised.value.code == 'driver_response_invalid'


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        (('dispatch', 'state'), 'safe_revoked'),
        (('dispatch', 'ready'), True),
        (('dispatch', 'fault_code'), None),
        (('dispatch', 'last_decision'), 'would_stop:adapter_fault'),
        (('output', 'state'), 'released'),
        (('output', 'arm_sdk_weight'), 0.0),
        (('output', 'fault_reason'), 'different_fault'),
    ],
)
def test_g1_shadow_terminal_fault_rejects_contradictory_shapes(path, value):
    raw = _g1_shadow_snapshot('fault', 'dispatch_fault')
    target: Any = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(TeleopServiceError) as raised:
        _project_g1_shadow(raw)
    assert raised.value.code == 'driver_response_invalid'


def test_adapter_snapshot_failure_is_visible_without_forwarding_raw_details():
    fake = FakeD1Caller()
    raw = fake.snapshot()
    raw['dispatch']['adapter'] = {
        'kind': 'unavailable',
        'reason': 'adapter_snapshot_exception',
        'exception': '<script>private</script>',
    }
    projected, _ = _project_driver_snapshot(
        raw,
        driver_id=DRIVER_ID,
        robot_id=ROBOT_ID,
        capability_digest=DIGEST,
        action='status',
    )
    dry_run = projected['dispatch']['dry_run']
    assert dry_run['profile'] == 'unavailable'
    assert dry_run['operation'] == 'unavailable'
    assert dry_run['stop_reason'] == 'adapter_snapshot_exception'
    assert dry_run['hardware_output'] is None
    assert dry_run['actuation_enabled'] is None
    assert '<script>' not in repr(projected)

    raw['dispatch']['adapter']['reason'] = 'arbitrary_exception_text'
    with pytest.raises(TeleopServiceError) as raised:
        _project_driver_snapshot(
            raw,
            driver_id=DRIVER_ID,
            robot_id=ROBOT_ID,
            capability_digest=DIGEST,
            action='status',
        )
    assert raised.value.code == 'driver_response_invalid'


@pytest.mark.parametrize(
    'adapter',
    [
        {'kind': 'recording'},
        {
            'kind': 'recording',
            'closed': True,
            'current': {'kind': 'safe'},
            'records': [],
        },
        {
            'kind': 'recording',
            'closed': False,
            'current': {'kind': 'live_apply'},
            'records': [],
        },
        {
            'kind': 'recording',
            'closed': False,
            'current': {'kind': {}},
            'records': [],
        },
        {
            'kind': 'recording',
            'closed': False,
            'current': {'kind': 'safe'},
            'records': [{}] * 65,
        },
        {
            'kind': 'recording',
            'closed': False,
            'hardware_output': True,
            'current': {'kind': 'safe'},
            'records': [],
        },
        {
            'kind': 'recording',
            'closed': False,
            'actuation_enabled': True,
            'current': {'kind': 'safe'},
            'records': [],
        },
    ],
    ids=[
        'incomplete',
        'closed',
        'unknown-current',
        'unhashable-current',
        'unbounded-history',
        'contradictory-hardware-output',
        'contradictory-actuation',
    ],
)
def test_recording_adapter_projection_requires_bounded_zero_output_shape(adapter):
    fake = FakeD1Caller()
    raw = fake.snapshot()
    raw['dispatch']['adapter'] = adapter
    with pytest.raises(TeleopServiceError) as raised:
        _project_driver_snapshot(
            raw,
            driver_id=DRIVER_ID,
            robot_id=ROBOT_ID,
            capability_digest=DIGEST,
            action='status',
        )
    assert raised.value.code == 'driver_response_invalid'


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('last_published_sequence', 10),
        ('last_admitted_sequence', None),
        ('last_published_sequence', -1),
    ],
    ids=['applied-after-admitted', 'applied-without-admission', 'negative-applied'],
)
def test_hardware_dispatch_sequence_evidence_fails_closed(field, value):
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake, published_sequence=8)
    raw['dispatch'][field] = value

    with pytest.raises(TeleopServiceError) as raised:
        _project_live(raw)
    assert raised.value.code == 'driver_response_invalid'


@pytest.mark.parametrize('published_sequence', [None, 7, 9])
def test_hardware_dispatch_accepts_bounded_publish_lag(
    published_sequence,
):
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake, published_sequence=published_sequence)

    projected = _project_live(raw)

    assert projected['dispatch']['last_admitted_sequence'] == 9
    assert projected['dispatch']['last_published_sequence'] == published_sequence


@pytest.mark.parametrize('spoofed_key', ['last_applied_sequence', 'last_would_apply_sequence'])
def test_hardware_dispatch_rejects_non_publisher_sequence_aliases(spoofed_key):
    fake = FakeD1Caller()
    raw = _live_active_snapshot(fake, published_sequence=7)
    raw['dispatch'][spoofed_key] = 7

    with pytest.raises(TeleopServiceError) as raised:
        _project_live(raw)
    assert raised.value.code == 'driver_response_invalid'


def test_guard_write_failure_never_reaches_prepare_shadow(
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        broker = CommandBroker()
        coordinator = _coordinator(fake, MutableClock(), command_broker=broker)

        def fail_guard_write(_guard):
            raise OSError('simulated guard database failure')

        monkeypatch.setattr(authority_guard, 'create_guard', fail_guard_write)
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert raised.value.code == 'authority_guard_persistence_error'
            assert [call['action'] for call in fake.calls] == ['status']
            assert await broker.authority_for(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_cancelled_guard_commit_is_removed_before_command_gate_reopens(
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller(lease_seconds=10.0)
        broker = CommandBroker()
        coordinator = _coordinator(
            fake,
            MutableClock(),
            command_broker=broker,
        )
        committed = threading.Event()
        allow_return = threading.Event()
        real_create = authority_guard.create_guard

        def commit_then_wait(guard):
            result = real_create(guard)
            committed.set()
            if not allow_return.wait(timeout=1.0):
                raise AssertionError('test did not release committed guard write')
            return result

        monkeypatch.setattr(authority_guard, 'create_guard', commit_then_wait)
        acquire_task = asyncio.create_task(
            coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID),
        )
        try:
            await _wait_until(committed.is_set)
            acquire_task.cancel()
            await asyncio.sleep(0)
            assert not acquire_task.done()
            allow_return.set()
            with pytest.raises(asyncio.CancelledError):
                await acquire_task

            assert [call['action'] for call in fake.calls] == ['status']
            assert authority_guard.get_guard(ROBOT_ID) is None
            assert coordinator.authority_guard_for_robot(ROBOT_ID) is None
            assert await broker.authority_for(ROBOT_ID) is None
            async with broker.ordinary_command(
                ROBOT_ID,
                read_only=False,
                source='test',
                tool='locomotion',
                action='move',
                tool_verified=True,
                action_verified=True,
            ):
                pass
        finally:
            allow_return.set()
            await asyncio.gather(acquire_task, return_exceptions=True)
            await coordinator.stop()

    asyncio.run(scenario())


def test_guard_commit_wins_race_with_target_delete_and_delete_is_rejected(
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller(lease_seconds=10.0)
        coordinator = _coordinator(fake, MutableClock())
        create_entered = threading.Event()
        allow_create = threading.Event()
        real_create = authority_guard.create_guard

        def delayed_create(guard):
            create_entered.set()
            if not allow_create.wait(timeout=1.0):
                raise AssertionError('test did not release guard commit')
            return real_create(guard)

        monkeypatch.setattr(authority_guard, 'create_guard', delayed_create)
        acquire_task = asyncio.create_task(
            coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID),
        )
        try:
            await _wait_until(create_entered.is_set)
            delete_task = asyncio.create_task(mcp_manage.mcp_delete(DRIVER_ID))
            await asyncio.sleep(0)
            assert not delete_task.done()

            allow_create.set()
            acquired = await acquire_task
            delete_response = await delete_task
            assert delete_response.status_code == 409
            assert json.loads(delete_response.body)['data'] == {
                'code': 'authority_target_locked',
                'reason': 'persistent_authority_guard_requires_stable_target',
                'project_state': 'running',
                'mcp_ids': [DRIVER_ID],
            }
            assert authority_guard.get_guard(ROBOT_ID).session_id == acquired.session.id
            assert [call['action'] for call in fake.calls[:3]] == [
                'status',
                'prepare_shadow',
                'heartbeat',
            ]
        finally:
            allow_create.set()
            await asyncio.gather(acquire_task, return_exceptions=True)
            await _finish(coordinator)

    asyncio.run(scenario())


def test_target_change_before_guard_commit_rejects_without_prepare(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller(lease_seconds=10.0)
        status_entered = asyncio.Event()
        allow_status = asyncio.Event()

        async def delayed_status(producer, _arguments):
            status_entered.set()
            await allow_status.wait()
            return producer.snapshot()

        fake.queue('status', delayed_status)
        broker = CommandBroker()
        coordinator = _coordinator(
            fake,
            MutableClock(),
            command_broker=broker,
        )
        acquire_task = asyncio.create_task(
            coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID),
        )
        try:
            await asyncio.wait_for(status_entered.wait(), timeout=0.5)
            changed_services = deepcopy(config.main['services'])
            changed_services['mcp'][0]['tools'][0]['x-teleop'][
                'capability_digest'
            ] = 'f' * 64
            config.main['services'] = changed_services
            allow_status.set()

            with pytest.raises(TeleopServiceError) as raised:
                await acquire_task
            assert raised.value.code == 'driver_not_ready'
            assert [call['action'] for call in fake.calls] == ['status']
            assert authority_guard.get_guard(ROBOT_ID) is None
            assert await broker.authority_for(ROBOT_ID) is None
        finally:
            allow_status.set()
            await asyncio.gather(acquire_task, return_exceptions=True)
            await coordinator.stop()

    asyncio.run(scenario())


def test_core_crash_restores_robot_gate_and_owner_stop_clears_it(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=10.0)
        crashed = _coordinator(fake, clock, command_broker=CommandBroker())
        acquired = await crashed.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        stored = authority_guard.get_guard(ROBOT_ID)
        assert stored is not None
        assert stored.phase == 'active'
        assert stored.session_id == acquired.session.id
        await _crash_without_authority_cleanup(crashed, fake)

        recovered_broker = CommandBroker()
        recovered = _coordinator(fake, clock, command_broker=recovered_broker)
        await recovered.start()
        try:
            assert await recovered.manager.list_visible('', owner=True) == []
            recovered_guard = recovered.authority_guard_for_robot(ROBOT_ID)
            assert recovered_guard == {
                'state': 'recovery_required',
                'phase': 'recovery_required',
                'driver_id': DRIVER_ID,
                'robot_id': ROBOT_ID,
                'retryable': True,
                'created_at': stored.created_at,
                'updated_at': recovered_guard['updated_at'],
            }
            assert recovered_guard['updated_at'] >= stored.updated_at
            with pytest.raises(TeleopCommandBlocked) as blocked:
                async with recovered_broker.ordinary_command(
                    ROBOT_ID,
                    read_only=False,
                    source='test',
                    tool='locomotion',
                    action='move',
                    tool_verified=True,
                    action_verified=True,
                ):
                    raise AssertionError('guarded robot write must not be admitted')
            assert blocked.value.public_detail()['state'] == 'recovery_required'
            async with recovered_broker.ordinary_command(
                'robot-b',
                read_only=False,
                source='test',
                tool='locomotion',
                action='move',
                tool_verified=True,
                action_verified=True,
            ):
                pass
            with pytest.raises(TeleopServiceError) as acquire_blocked:
                await recovered.acquire(DRIVER_ID, 'bob', CLIENT_ID)
            assert acquire_blocked.value.code == 'robot_recovery_required'

            result = await recovered.reconcile_authority_guard(
                ROBOT_ID,
                principal_id='owner:legacy',
            )
            assert result == {
                'state': 'clear',
                'robot_id': ROBOT_ID,
                'driver_id': DRIVER_ID,
                'old_session_restored': False,
                'reacquire_required': True,
            }
            assert fake.count('stop') == 1
            assert authority_guard.get_guard(ROBOT_ID) is None
            assert recovered.authority_guard_for_robot(ROBOT_ID) is None
            assert await recovered.manager.get(acquired.session.id) is None
            async with recovered_broker.ordinary_command(
                ROBOT_ID,
                read_only=False,
                source='test',
                tool='locomotion',
                action='move',
                tool_verified=True,
                action_verified=True,
            ):
                pass
        finally:
            await recovered.stop()

    asyncio.run(scenario())


def test_new_driver_boot_safe_unarmed_clears_guard_without_lifecycle_stop(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=10.0)
        crashed = _coordinator(fake, clock, command_broker=CommandBroker())
        await crashed.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        await _crash_without_authority_cleanup(crashed, fake)

        fake.boot_id = 'f2b909a5-119c-4328-9213-4c8da7293188'
        fake.epoch = 1
        fake.session_id = None
        fake.fence = None
        fake.state = 'idle'
        fake.reason = None
        fake.authority_valid = False
        fake.dispatch_generation = 0
        fake.dispatch_last_admitted = None

        recovered = _coordinator(fake, clock, command_broker=CommandBroker())
        await recovered.start()
        try:
            result = await recovered.reconcile_authority_guard(
                ROBOT_ID,
                principal_id='owner:legacy',
            )
            assert result['state'] == 'clear'
            assert result['old_session_restored'] is False
            assert result['reacquire_required'] is True
            assert fake.count('stop') == 0
            assert authority_guard.get_guard(ROBOT_ID) is None
        finally:
            await recovered.stop()

    asyncio.run(scenario())


def test_new_driver_boot_with_reset_generation_can_be_stopped_and_cleared(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=10.0)
        crashed = _coordinator(fake, clock, command_broker=CommandBroker())
        await crashed.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        stored = authority_guard.get_guard(ROBOT_ID)
        assert stored is not None
        assert stored.dispatch_generation > 0
        await _crash_without_authority_cleanup(crashed, fake)

        fake.boot_id = 'ad401205-82aa-4a55-8bff-3c59ef543ce5'
        fake.epoch = 1
        fake.session_id = None
        fake.fence = None
        fake.state = 'released'
        fake.reason = 'lifecycle_stop'
        fake.authority_valid = False
        fake.dispatch_generation = 0
        fake.dispatch_last_admitted = None

        recovered = _coordinator(fake, clock, command_broker=CommandBroker())
        await recovered.start()
        try:
            result = await recovered.reconcile_authority_guard(
                ROBOT_ID,
                principal_id='owner:legacy',
            )
            assert result['state'] == 'clear'
            assert fake.count('stop') == 1
            assert authority_guard.get_guard(ROBOT_ID) is None
        finally:
            await recovered.stop()

    asyncio.run(scenario())


def test_recovery_target_change_keeps_persistent_and_runtime_gates(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=10.0)
        crashed = _coordinator(fake, clock, command_broker=CommandBroker())
        acquired = await crashed.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        await _crash_without_authority_cleanup(crashed, fake)

        changed_services = deepcopy(config.main['services'])
        changed_services['mcp'][0]['tools'][0]['x-teleop'][
            'capability_digest'
        ] = 'f' * 64
        config.main['services'] = changed_services
        broker = CommandBroker()
        recovered = _coordinator(fake, clock, command_broker=broker)
        await recovered.start()
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await recovered.reconcile_authority_guard(
                    ROBOT_ID,
                    principal_id='owner:legacy',
                )
            assert raised.value.code == 'authority_guard_target_changed'
            persisted = authority_guard.get_guard(ROBOT_ID)
            assert persisted is not None
            assert persisted.session_id == acquired.session.id
            assert recovered.authority_guard_for_robot(ROBOT_ID) is not None
            with pytest.raises(TeleopCommandBlocked):
                async with broker.ordinary_command(
                    ROBOT_ID,
                    read_only=False,
                    source='test',
                    tool='locomotion',
                    action='move',
                    tool_verified=True,
                    action_verified=True,
                ):
                    raise AssertionError('changed target must remain quarantined')
            assert fake.count('stop') == 0
        finally:
            await recovered.stop()

    asyncio.run(scenario())


def test_recovery_authority_root_change_keeps_guard_without_stop(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=10.0)
        crashed = _coordinator(fake, clock, command_broker=CommandBroker())
        acquired = await crashed.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        await _crash_without_authority_cleanup(crashed, fake)

        changed_services = deepcopy(config.main['services'])
        changed_services['mcp'][1]['url'] = 'http://changed-root.invalid/mcp'
        config.main['services'] = changed_services
        recovered = _coordinator(fake, clock, command_broker=CommandBroker())
        await recovered.start()
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await recovered.reconcile_authority_guard(
                    ROBOT_ID,
                    principal_id='owner:legacy',
                )
            assert raised.value.code == 'authority_guard_target_changed'
            persisted = authority_guard.get_guard(ROBOT_ID)
            assert persisted is not None
            assert persisted.session_id == acquired.session.id
            assert fake.count('stop') == 0
        finally:
            await recovered.stop()

    asyncio.run(scenario())


def test_guard_delete_failure_keeps_gate_and_is_retryable(
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=10.0)
        crashed = _coordinator(fake, clock, command_broker=CommandBroker())
        await crashed.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        await _crash_without_authority_cleanup(crashed, fake)

        broker = CommandBroker()
        recovered = _coordinator(fake, clock, command_broker=broker)
        await recovered.start()
        real_delete = authority_guard.delete_guard

        def fail_delete(_robot_id: str, _session_id: str) -> bool:
            raise OSError('simulated guard delete failure')

        monkeypatch.setattr(authority_guard, 'delete_guard', fail_delete)
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await recovered.reconcile_authority_guard(
                    ROBOT_ID,
                    principal_id='owner:legacy',
                )
            assert raised.value.code == 'authority_guard_persistence_error'
            assert fake.count('stop') == 1
            assert authority_guard.get_guard(ROBOT_ID) is not None
            assert recovered.authority_guard_for_robot(ROBOT_ID) is not None
            with pytest.raises(TeleopCommandBlocked):
                async with broker.ordinary_command(
                    ROBOT_ID,
                    read_only=False,
                    source='test',
                    tool='locomotion',
                    action='move',
                    tool_verified=True,
                    action_verified=True,
                ):
                    raise AssertionError('failed delete must keep admission closed')

            monkeypatch.setattr(authority_guard, 'delete_guard', real_delete)
            result = await recovered.reconcile_authority_guard(
                ROBOT_ID,
                principal_id='owner:legacy',
            )
            assert result['state'] == 'clear'
            assert fake.count('stop') == 1
            assert authority_guard.get_guard(ROBOT_ID) is None
        finally:
            await recovered.stop()

    asyncio.run(scenario())


def test_cancelled_reconcile_finishes_guard_delete_and_broker_release(
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=10.0)
        crashed = _coordinator(fake, clock, command_broker=CommandBroker())
        await crashed.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        await _crash_without_authority_cleanup(crashed, fake)

        broker = CommandBroker()
        recovered = _coordinator(fake, clock, command_broker=broker)
        await recovered.start()
        deleted = threading.Event()
        allow_return = threading.Event()
        real_delete = authority_guard.delete_guard

        def delete_then_wait(robot_id: str, session_id: str) -> bool:
            result = real_delete(robot_id, session_id)
            deleted.set()
            if not allow_return.wait(timeout=1.0):
                raise AssertionError('test did not release committed guard delete')
            return result

        monkeypatch.setattr(authority_guard, 'delete_guard', delete_then_wait)
        reconcile_task = asyncio.create_task(recovered.reconcile_authority_guard(
            ROBOT_ID,
            principal_id='owner:legacy',
        ))
        try:
            await _wait_until(deleted.is_set)
            reconcile_task.cancel()
            await asyncio.sleep(0)
            assert not reconcile_task.done()
            allow_return.set()
            with pytest.raises(asyncio.CancelledError):
                await reconcile_task

            assert fake.count('stop') == 1
            assert authority_guard.get_guard(ROBOT_ID) is None
            assert recovered.authority_guard_for_robot(ROBOT_ID) is None
            assert await broker.authority_for(ROBOT_ID) is None
            repeated = await recovered.reconcile_authority_guard(
                ROBOT_ID,
                principal_id='owner:legacy',
            )
            assert repeated['already_clear'] is True
        finally:
            allow_return.set()
            await asyncio.gather(reconcile_task, return_exceptions=True)
            await recovered.stop()

    asyncio.run(scenario())


def test_acquire_is_idempotent_for_same_operator_and_conflicts_for_another(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        coordinator = _coordinator(FakeD1Caller(), MutableClock())
        try:
            created = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            call_count = len(coordinator._caller.calls)
            repeated = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert repeated.disposition == 'existing'
            assert repeated.session is created.session
            assert len(coordinator._caller.calls) == call_count

            with pytest.raises(SessionConflict) as raised:
                await coordinator.acquire(DRIVER_ID, 'bob', CLIENT_ID)
            assert raised.value.session is created.session
            assert len(coordinator._caller.calls) == call_count
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_random_unknown_ids_cannot_grow_lock_storage(shadow_session_tool):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        coordinator = _coordinator(FakeD1Caller(), MutableClock())
        operation_lock_ids = tuple(map(id, coordinator._operation_locks))
        acquire_lock_ids = tuple(map(id, coordinator._acquire_locks))
        try:
            for _ in range(300):
                with pytest.raises(SessionNotFound):
                    await coordinator.status(str(uuid.uuid4()), 'alice', owner=False)
            for index in range(50):
                with pytest.raises(TeleopServiceError) as raised:
                    await coordinator.acquire(
                        f'unknown-driver-{index}',
                        'alice',
                        CLIENT_ID,
                    )
                assert raised.value.code == 'driver_not_found'

            assert tuple(map(id, coordinator._operation_locks)) == operation_lock_ids
            assert tuple(map(id, coordinator._acquire_locks)) == acquire_lock_ids
            assert len(coordinator._operation_locks) == 256
            assert len(coordinator._acquire_locks) == 256
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_prepare_timeout_reconciles_exact_same_session_and_epoch(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        fake.prepare_timeout_after_apply = True
        coordinator = _coordinator(fake, MutableClock())
        try:
            result = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert result.disposition == 'created'
            assert [call['action'] for call in fake.calls[:4]] == [
                'status', 'prepare_shadow', 'status', 'heartbeat',
            ]
            reconciled_status = fake.calls[2]
            assert reconciled_status['arguments'] is None
            assert fake.session_id == result.session.id
            assert fake.epoch == result.session.epoch
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_owner_release_cannot_pass_an_inflight_prepare(shadow_session_tool):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        prepare_started = asyncio.Event()
        allow_prepare = asyncio.Event()

        async def delayed_prepare(
            producer: FakeD1Caller,
            arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            assert arguments is not None
            prepare_started.set()
            await allow_prepare.wait()
            producer.session_id = arguments['session_id']
            producer.epoch = arguments['epoch']
            producer.fence = arguments['fence']
            producer.state = 'prepared_shadow'
            producer.reason = None
            producer.authority_valid = True
            producer.dispatch_generation += 1
            return producer.snapshot()

        fake.queue('prepare_shadow', delayed_prepare)
        acquire = asyncio.create_task(coordinator.acquire(
            DRIVER_ID,
            'alice',
            CLIENT_ID,
        ))
        try:
            await asyncio.wait_for(prepare_started.wait(), timeout=0.8)
            visible = await coordinator.manager.list_visible('', owner=True)
            assert len(visible) == 1
            release = asyncio.create_task(coordinator.release(
                visible[0].id,
                'owner',
                CLIENT_ID,
                owner=True,
            ))
            await asyncio.sleep(0)
            assert release.done() is False

            allow_prepare.set()
            acquired = await acquire
            released, acknowledged = await release
            actions = [call['action'] for call in fake.calls]

            assert acquired.session is released
            assert released.state == 'released'
            assert acknowledged is True
            assert actions.index('release') > actions.index('heartbeat')
            assert fake.count('prepare_shadow') == 1
            assert fake.count('release') == 1
            assert fake.session_id is None
        finally:
            allow_prepare.set()
            if not acquire.done():
                acquire.cancel()
                await asyncio.gather(acquire, return_exceptions=True)
            await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'malformation',
    [
        {'actuation_enabled': True},
        {'capability_digest': 'f' * 64},
        {'authority_valid': 'yes'},
        {'lease': {'source': 'browser', 'timeout_ms': 1_000}},
        {'rtc': {'connected': False, 'channels': {'teleop-pose': False}}},
    ],
    ids=['actuation', 'digest', 'authority-type', 'lease-source', 'rtc-channels'],
)
def test_acquire_fails_closed_on_nonconforming_driver_response(
    shadow_session_tool,
    malformation,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        fake.queue('status', fake.snapshot(**malformation))
        coordinator = _coordinator(fake, MutableClock())
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert raised.value.code == 'driver_response_invalid'
            assert raised.value.status_code == 502
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        (('dispatch',), None),
        (('dispatch', 'kind'), 'direct'),
        (('dispatch', 'ready'), False),
        (('dispatch', 'generation'), True),
        (('dispatch', 'generation'), 2**63),
        (('dispatch', 'mailbox_depth'), 2),
        (('dispatch', 'stop_queue_depth'), 1),
        (('dispatch', 'stop_acknowledged'), False),
        (('dispatch', 'fault_code'), 'adapter_fault'),
        (('dispatch', 'last_decision'), 'unknown'),
        (('dispatch', 'io_inflight'), 'safe_stop'),
        (('dispatch', 'last_would_apply_sequence'), 1),
    ],
    ids=[
        'missing',
        'wrong-kind',
        'not-ready',
        'bool-generation',
        'generation-overflow',
        'mailbox-overflow',
        'stop-pending',
        'stop-unacknowledged',
        'faulted',
        'unknown-decision',
        'stop-inflight',
        'applied-without-admission',
    ],
)
def test_acquire_requires_stable_recording_dispatch_status(
    shadow_session_tool,
    path,
    value,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        raw = fake.snapshot()
        if len(path) == 1:
            raw.pop(path[0])
        else:
            raw[path[0]][path[1]] = value
        fake.queue('status', raw)
        broker = CommandBroker()
        coordinator = _coordinator(fake, MutableClock(), command_broker=broker)
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert raised.value.code == 'driver_response_invalid'
            assert coordinator._command_claims == {}
            assert await broker.authority_for(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_post_prepare_numeric_overflow_revokes_driver_authority(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()

        def overflowing_prepare(
            producer: FakeD1Caller,
            arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            assert arguments is not None
            producer.session_id = arguments['session_id']
            producer.epoch = arguments['epoch']
            producer.fence = arguments['fence']
            producer.state = 'prepared_shadow'
            producer.reason = None
            producer.authority_valid = True
            producer.dispatch_generation += 1
            snapshot = producer.snapshot()
            snapshot['lease']['timeout_ms'] = 10**400
            return snapshot

        fake.queue('prepare_shadow', overflowing_prepare)
        coordinator = _coordinator(fake, MutableClock())
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)

            assert raised.value.code == 'driver_response_invalid'
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
            assert fake.count('release') == 1
            assert fake.state == 'released'
            assert fake.authority_valid is False
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('reported_epoch', 'expected_code'),
    [
        (MAX_DRIVER_EPOCH, 'driver_epoch_exhausted'),
        (MAX_DRIVER_EPOCH + 1, 'driver_response_invalid'),
    ],
)
def test_acquire_rejects_driver_epoch_exhaustion_without_preparing(
    shadow_session_tool,
    reported_epoch,
    expected_code,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller(epoch=reported_epoch)
        coordinator = _coordinator(fake, MutableClock())
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert raised.value.code == expected_code
            assert fake.count('prepare_shadow') == 0
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_supervisor_runs_within_250ms_and_keeps_paused_and_hold_alive(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller(lease_seconds=10.0)
        coordinator = _coordinator(fake, MutableClock())
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            initial = fake.count('heartbeat')
            started = asyncio.get_running_loop().time()
            await _wait_until(lambda: fake.count('heartbeat') > initial, timeout=0.35)
            assert asyncio.get_running_loop().time() - started < 0.33

            paused = await coordinator.pause(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            paused_count = fake.count('heartbeat')
            await _wait_until(lambda: fake.count('heartbeat') > paused_count)
            assert paused.state == 'paused'
            assert fake.state == 'paused'

            # Driver-side active_shadow is a valid heartbeat state.  Enter HOLD
            # through the public operation from a newly active fake state.
            paused.state = 'active'
            fake.state = 'active_shadow'
            fake.reason = None
            held = await coordinator.soft_stop(
                paused.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            held_count = fake.count('heartbeat')
            await _wait_until(lambda: fake.count('heartbeat') > held_count)
            assert held.state == 'hold'
            assert fake.state == 'hold'
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('action', 'malformation', 'expected_code'),
    [
        ('pause', {'state': {}}, 'driver_pause_rejected'),
        ('soft_stop', {'state': 'hold', 'reason': []}, 'driver_soft_stop_rejected'),
    ],
)
def test_unhashable_control_projection_faults_and_revokes(
    shadow_session_tool,
    action,
    malformation,
    expected_code,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            fake.queue(action, fake.snapshot(**malformation))
            operation = getattr(coordinator, action)
            with pytest.raises(TeleopServiceError) as raised:
                await operation(
                    acquired.session.id,
                    'alice',
                    CLIENT_ID,
                    owner=False,
                )

            assert raised.value.code == expected_code
            assert acquired.session.state == 'faulted'
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
            assert fake.count('release') == 1
            assert fake.state == 'released'
            assert fake.authority_valid is False
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize('action', ['pause', 'soft_stop'])
def test_cancelled_control_faults_and_revokes_ambiguous_driver_result(
    shadow_session_tool,
    action,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        driver_applied = asyncio.Event()

        async def ambiguous_control(
            producer: FakeD1Caller,
            arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            producer._assert_identity(arguments)
            if action == 'pause':
                producer.state = 'paused'
                producer.reason = 'operator_pause'
            else:
                producer.state = 'hold'
                producer.reason = 'soft_stop'
            producer.dispatch_generation += 1
            driver_applied.set()
            await asyncio.Future()
            raise AssertionError('unreachable')

        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            fake.queue(action, ambiguous_control)
            operation = asyncio.create_task(getattr(coordinator, action)(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            ))
            await asyncio.wait_for(driver_applied.wait(), timeout=0.8)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation

            assert acquired.session.state == 'faulted'
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
            assert fake.count('release') == 1
            assert fake.state == 'released'
            assert fake.authority_valid is False
            assert coordinator._safety_tasks == set()
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize('failure_mode', ['three_failures', 'lease_age'])
def test_retryable_heartbeat_faults_after_three_failures_or_75_percent_lease(
    shadow_session_tool,
    failure_mode,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=0.8)
        coordinator = _coordinator(fake, clock)
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            timeout = mcp_client.TrustedShadowTransportError('timeout')
            if failure_mode == 'three_failures':
                fake.queue('heartbeat', timeout, timeout, timeout)
                await _wait_until(
                    lambda: acquired.session.state == 'faulted',
                    timeout=1.2,
                )
                assert coordinator._heartbeat_health[acquired.session.id].consecutive_failures == 3
            else:
                clock.advance(0.61)
                fake.queue('heartbeat', timeout)
                await asyncio.sleep(0.25)

            await _wait_until(lambda: fake.count('release') == 1, timeout=0.8)
            assert acquired.session.state == 'faulted'
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
            assert fake.count('soft_stop') == 1
            assert fake.count('release') == 1
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_terminal_rpc_faults_on_first_failed_supervisor_call(shadow_session_tool):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller(lease_seconds=0.8)
        coordinator = _coordinator(fake, MutableClock())
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            heartbeat_task = coordinator._heartbeat_tasks[acquired.session.id]
            fake.queue(
                'heartbeat',
                mcp_client.TrustedShadowTransportError(
                    'rpc_error',
                    rpc_code=-32602,
                    rpc_data_code='session_expired',
                ),
            )
            before = fake.count('heartbeat')
            # Local authority is revoked before remote soft-stop/release finish.
            # The worker itself, rather than a wall-clock sleep, is the cleanup
            # completion signal for this terminal heartbeat path.
            await asyncio.wait_for(asyncio.shield(heartbeat_task), timeout=1.0)
            assert fake.count('heartbeat') == before + 1
            assert acquired.session.state == 'faulted'
            assert fake.count('soft_stop') == 1
            assert fake.count('release') == 1
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_obsolete_worker_cannot_fault_a_replacement_session(shadow_session_tool):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        try:
            old = (await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)).session
            await coordinator.release(old.id, 'alice', CLIENT_ID, owner=False)
            replacement = (
                await coordinator.acquire(DRIVER_ID, 'bob', CLIENT_ID)
            ).session

            await coordinator._fault_session(old, 'late_old_worker')

            assert old.state == 'released'
            assert replacement.state == 'active'
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is replacement
            assert fake.session_id == replacement.id
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_browser_lease_uses_monotonic_heartbeat_then_expires_and_drains(
    shadow_session_tool,
    monkeypatch,
):
    async def scenario() -> None:
        monkeypatch.setenv('MOTUS_TELEOP_LEASE_SECONDS', '5')
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=10.0)
        coordinator = _coordinator(fake, clock)
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            original_deadline = acquired.session.deadline_monotonic
            clock.advance(14.5, wall_seconds=-50_000)
            renewed = await coordinator.heartbeat(
                acquired.session.id,
                'alice',
                CLIENT_ID,
            )
            assert renewed.deadline_monotonic > original_deadline

            clock.advance(0.6, wall_seconds=-50_000)
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is renewed

            clock.advance(14.5, wall_seconds=-50_000)
            await _wait_until(lambda: fake.count('release') == 1)
            assert renewed.state == 'expired'
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
            assert fake.count('soft_stop') == 1
            assert fake.count('release') == 1
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_release_is_idempotent_and_only_calls_driver_once(shadow_session_tool):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            first, first_ack = await coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            second, second_ack = await coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            assert first is second
            assert first.state == 'released'
            assert first_ack is True
            assert second_ack is True
            assert fake.count('release') == 1
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_concurrent_release_waits_for_one_shared_driver_ack(shadow_session_tool):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        cancellation_seen = asyncio.Event()
        allow_cancel = asyncio.Event()

        async def heartbeat_cancel_barrier() -> None:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await allow_cancel.wait()
                raise

        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            original = coordinator._heartbeat_tasks.pop(acquired.session.id)
            original.cancel()
            await asyncio.gather(original, return_exceptions=True)
            coordinator._heartbeat_tasks[acquired.session.id] = asyncio.create_task(
                heartbeat_cancel_barrier()
            )

            first = asyncio.create_task(coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            ))
            await asyncio.wait_for(cancellation_seen.wait(), timeout=0.5)
            second = asyncio.create_task(coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            ))
            await asyncio.sleep(0)
            assert second.done() is False

            allow_cancel.set()
            (first_session, first_ack), (second_session, second_ack) = await asyncio.gather(
                first,
                second,
            )

            assert first_session is second_session
            assert first_ack is True
            assert second_ack is True
            assert fake.count('release') == 1
        finally:
            allow_cancel.set()
            await coordinator.stop()

    asyncio.run(scenario())


def test_terminal_cleanup_keeps_the_owned_operation_lock(
    shadow_session_tool,
    monkeypatch,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        cleanup_reached = asyncio.Event()
        allow_cleanup_return = asyncio.Event()
        original_cleanup = coordinator._cleanup_terminal_runtime

        async def delayed_cleanup(session_id: str) -> None:
            await original_cleanup(session_id)
            cleanup_reached.set()
            await allow_cleanup_return.wait()

        monkeypatch.setattr(coordinator, '_cleanup_terminal_runtime', delayed_cleanup)
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            session_id = acquired.session.id
            original_lock = coordinator._session_lock(session_id)
            fake.queue('pause', fake.snapshot(state='invalid-state'))
            pause = asyncio.create_task(coordinator.pause(
                session_id,
                'alice',
                CLIENT_ID,
                owner=False,
            ))
            await asyncio.wait_for(cleanup_reached.wait(), timeout=0.8)

            assert coordinator._session_lock(session_id) is original_lock
            assert original_lock.locked() is True
            release = asyncio.create_task(coordinator.release(
                session_id,
                'alice',
                CLIENT_ID,
                owner=False,
            ))
            await asyncio.sleep(0)
            assert release.done() is False

            allow_cleanup_return.set()
            with pytest.raises(TeleopServiceError) as raised:
                await pause
            released, acknowledged = await release
            assert raised.value.code == 'driver_pause_rejected'
            assert released.state == 'faulted'
            assert acknowledged is False
            assert fake.count('soft_stop') == 1
            assert fake.count('release') == 1
        finally:
            allow_cleanup_return.set()
            await coordinator.stop()

    asyncio.run(scenario())


def test_cancelled_release_finishes_revocation_and_preserves_result(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        release_started = asyncio.Event()
        allow_release = asyncio.Event()

        async def delayed_release(
            producer: FakeD1Caller,
            arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            producer._assert_identity(arguments)
            release_started.set()
            await allow_release.wait()
            producer.session_id = None
            producer.fence = None
            producer.state = 'released'
            producer.reason = 'operator_release'
            producer.authority_valid = False
            producer.dispatch_generation += 1
            return producer.snapshot()

        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            fake.queue('release', delayed_release)
            first = asyncio.create_task(coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            ))
            await asyncio.wait_for(release_started.wait(), timeout=0.8)
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
            await asyncio.sleep(0)
            assert first.done() is False
            allow_release.set()
            with pytest.raises(asyncio.CancelledError):
                await first

            released, acknowledged = await coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            assert released.state == 'released'
            assert acknowledged is True
            assert fake.count('release') == 1
            assert coordinator._safety_tasks == set()
        finally:
            allow_release.set()
            await coordinator.stop()

    asyncio.run(scenario())


def test_double_cancelled_acquire_still_revokes_ambiguous_prepare(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        prepare_applied = asyncio.Event()
        release_started = asyncio.Event()
        allow_release = asyncio.Event()

        async def ambiguous_prepare(
            producer: FakeD1Caller,
            arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            assert arguments is not None
            producer.session_id = arguments['session_id']
            producer.epoch = arguments['epoch']
            producer.fence = arguments['fence']
            producer.state = 'prepared_shadow'
            producer.reason = None
            producer.authority_valid = True
            producer.dispatch_generation += 1
            prepare_applied.set()
            await asyncio.Future()
            raise AssertionError('unreachable')

        async def delayed_release(
            producer: FakeD1Caller,
            arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            producer._assert_identity(arguments)
            release_started.set()
            await allow_release.wait()
            producer.session_id = None
            producer.fence = None
            producer.state = 'released'
            producer.reason = 'operator_release'
            producer.authority_valid = False
            producer.dispatch_generation += 1
            return producer.snapshot()

        fake.queue('prepare_shadow', ambiguous_prepare)
        fake.queue('release', delayed_release)
        acquire = asyncio.create_task(coordinator.acquire(
            DRIVER_ID,
            'alice',
            CLIENT_ID,
        ))
        try:
            await asyncio.wait_for(prepare_applied.wait(), timeout=0.8)
            acquire.cancel()
            await asyncio.wait_for(release_started.wait(), timeout=0.8)
            acquire.cancel()
            await asyncio.sleep(0)
            assert acquire.done() is False

            allow_release.set()
            with pytest.raises(asyncio.CancelledError):
                await acquire

            assert fake.state == 'released'
            assert fake.authority_valid is False
            assert fake.count('release') == 1
            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
            assert coordinator._safety_tasks == set()
        finally:
            allow_release.set()
            await coordinator.stop()

    asyncio.run(scenario())


def test_shutdown_releases_driver_and_leaves_no_background_tasks(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)

        await coordinator.stop()

        assert acquired.session.state == 'released'
        assert fake.count('release') == 1
        assert coordinator._reaper_task is None
        assert coordinator._heartbeat_tasks == {}
        assert coordinator._http_session is None
        assert coordinator._stopping is True
        assert await coordinator.manager.list_visible('', owner=True) == []

    asyncio.run(scenario())


def test_shutdown_drains_an_acquire_that_has_not_reserved_yet(shadow_session_tool):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        status_started = asyncio.Event()
        allow_status = asyncio.Event()

        async def delayed_status(
            producer: FakeD1Caller,
            _arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            status_started.set()
            await allow_status.wait()
            return producer.snapshot()

        fake.queue('status', delayed_status)
        acquire = asyncio.create_task(coordinator.acquire(
            DRIVER_ID,
            'alice',
            CLIENT_ID,
        ))
        await asyncio.wait_for(status_started.wait(), timeout=0.8)
        stop = asyncio.create_task(coordinator.stop())
        await asyncio.sleep(0)
        assert stop.done() is False

        allow_status.set()
        acquired = await acquire
        await stop

        assert acquired.session.state == 'released'
        assert fake.state == 'released'
        assert fake.authority_valid is False
        assert fake.count('prepare_shadow') == 1
        assert fake.count('release') == 1
        assert coordinator._acquire_tasks == set()
        assert coordinator._heartbeat_tasks == {}
        assert await coordinator.manager.list_visible('', owner=True) == []

        with pytest.raises(TeleopServiceError) as raised:
            await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        assert raised.value.code == 'coordinator_stopping'

    asyncio.run(scenario())


def test_real_fence_and_driver_token_never_reach_public_error_audit_or_activity(
    shadow_session_tool,
    monkeypatch,
):
    async def scenario() -> None:
        monkeypatch.setattr(audit, 'emit', _REAL_AUDIT_EMIT)
        auth.init({
            'ACCESS_TOKEN': 'owner-token',
                'MOTUS_OPERATOR_TOKENS': '{"alice":"operator-token"}',
                'MOTUS_DRIVER_TOKEN': DRIVER_TOKEN,
                'MOTUS_TELEOP_TICKET_SECRET': 'test-service-ticket-secret-000001',
            })
        events: list[dict[str, Any]] = []

        async def capture_event(event: dict[str, Any]) -> None:
            events.append(deepcopy(event))

        monkeypatch.setattr(motus_stream, 'push_event', capture_event)
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            fence = acquired.session.fence
            public = coordinator.public_session(acquired.session)

            reflected = fake.snapshot(driver_name=fence)
            fake.queue('status', reflected)
            safe_status = await coordinator.status(
                acquired.session.id,
                'alice',
                owner=False,
            )
            assert safe_status['driver']['driver_name'] == DRIVER_ID

            await coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            await audit.flush()
            with config._get_conn() as conn:
                rows = conn.execute(
                    'SELECT event_type, session_id, robot_id, principal_id, '
                    'source, decision, reason, tool, action, details '
                    'FROM teleop_audit ORDER BY created_at, id'
                ).fetchall()

            columns = (
                'event_type', 'session_id', 'robot_id', 'principal_id',
                'source', 'decision', 'reason', 'tool', 'action', 'details',
            )
            audit_payload = [dict(zip(columns, row, strict=True)) for row in rows]
            scan = json.dumps(
                    {
                        'public': public,
                        'status': safe_status,
                        'audit': audit_payload,
                    'activity': events,
                },
                sort_keys=True,
            )
            assert fence not in scan
            assert DRIVER_TOKEN not in scan
            assert 'fence' not in json.dumps(public).lower()
            assert events
            assert audit_payload
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_acquire_waits_for_an_admitted_write_and_blocks_following_writes(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()
        clock = MutableClock()
        coordinator = _coordinator(fake, clock, command_broker=broker)
        write_entered = asyncio.Event()
        finish_write = asyncio.Event()

        async def admitted_write() -> None:
            async with broker.ordinary_command(
                ROBOT_ID,
                read_only=False,
                source='test',
                tool='locomotion',
                action='move',
            ):
                write_entered.set()
                await finish_write.wait()

        write_task = asyncio.create_task(admitted_write())
        try:
            await asyncio.wait_for(write_entered.wait(), timeout=0.5)
            acquire_task = asyncio.create_task(
                coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID),
            )
            for _ in range(100):
                if await broker.authority_for(ROBOT_ID) is not None:
                    break
                await asyncio.sleep(0.005)
            else:
                raise AssertionError('Acquire did not publish its command claim')

            assert fake.calls == []
            with pytest.raises(TeleopCommandBlocked):
                async with broker.ordinary_command(
                    ROBOT_ID,
                    read_only=False,
                    source='test',
                    tool='locomotion',
                    action='move',
                ):
                    raise AssertionError('blocked command entered the transport')

            # Admission is exact per Driver, not a global robot stop.
            async with broker.ordinary_command(
                'another-driver',
                read_only=False,
                source='test',
                tool='locomotion',
                action='move',
            ):
                pass

            finish_write.set()
            acquired = await asyncio.wait_for(acquire_task, timeout=0.8)
            assert acquired.session.state == 'active'
            assert [call['action'] for call in fake.calls[:3]] == [
                'status', 'prepare_shadow', 'heartbeat',
            ]
        finally:
            finish_write.set()
            await asyncio.gather(write_task, return_exceptions=True)
            await _finish(coordinator)

    asyncio.run(scenario())


def test_release_keeps_write_gate_closed_until_driver_cleanup_finishes(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()
        clock = MutableClock()
        coordinator = _coordinator(fake, clock, command_broker=broker)
        release_started = asyncio.Event()
        finish_release = asyncio.Event()

        async def delayed_release(caller, arguments):
            caller._assert_identity(arguments)
            release_started.set()
            await finish_release.wait()
            caller.session_id = None
            caller.fence = None
            caller.state = 'released'
            caller.reason = 'operator_release'
            caller.authority_valid = False
            caller.dispatch_generation += 1
            return caller.snapshot()

        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            fake.queue('release', delayed_release)
            release_task = asyncio.create_task(coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            ))
            await asyncio.wait_for(release_started.wait(), timeout=0.5)

            assert await coordinator.manager.active_for_robot(ROBOT_ID) is None
            claim = await broker.authority_for(ROBOT_ID)
            assert claim is not None
            assert claim.state == 'releasing'
            with pytest.raises(TeleopCommandBlocked):
                async with broker.ordinary_command(
                    ROBOT_ID,
                    read_only=False,
                    source='test',
                    tool='locomotion',
                    action='move',
                ):
                    raise AssertionError('write escaped during Driver release')

            finish_release.set()
            released, acknowledged = await asyncio.wait_for(release_task, timeout=0.8)
            assert released.state == 'released'
            assert acknowledged is True
            assert await broker.authority_for(ROBOT_ID) is None
            async with broker.ordinary_command(
                ROBOT_ID,
                read_only=False,
                source='test',
                tool='locomotion',
                action='move',
            ):
                pass
        finally:
            finish_release.set()
            await _finish(coordinator)

    asyncio.run(scenario())


def test_unconfirmed_release_stays_quarantined_until_a_retry_is_acknowledged(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()
        clock = MutableClock()
        coordinator = _coordinator(fake, clock, command_broker=broker)
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            fake.queue(
                'release',
                mcp_client.TrustedShadowTransportError('timeout'),
            )
            released, acknowledged = await coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )

            assert released.state == 'released'
            assert acknowledged is False
            assert await broker.authority_for(ROBOT_ID) is not None
            assert released.id in coordinator._release_reconcile_tasks
            with pytest.raises(TeleopCommandBlocked):
                async with broker.ordinary_command(
                    ROBOT_ID,
                    read_only=False,
                    source='test',
                    tool='locomotion',
                    action='move',
                ):
                    raise AssertionError('unconfirmed release opened the gate')

            retried, retry_acknowledged = await coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            assert retried is released
            assert retry_acknowledged is True
            assert fake.count('release') == 2
            assert await broker.authority_for(ROBOT_ID) is None
            assert released.id not in coordinator._release_reconcile_tasks
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_double_cancelled_pre_reservation_acquire_cannot_orphan_command_claim(
    shadow_session_tool,
):
    class DelayedReleaseBroker(CommandBroker):
        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.finish_release = asyncio.Event()

        async def release_authority(self, robot_id: str, token: str) -> bool:
            self.release_started.set()
            await self.finish_release.wait()
            return await super().release_authority(robot_id, token)

    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = DelayedReleaseBroker()
        fake = FakeD1Caller()
        status_started = asyncio.Event()

        async def delayed_status(_caller, _arguments):
            status_started.set()
            await asyncio.Event().wait()

        fake.queue('status', delayed_status)
        coordinator = _coordinator(fake, MutableClock(), command_broker=broker)
        acquire = asyncio.create_task(
            coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID),
        )
        await asyncio.wait_for(status_started.wait(), timeout=0.5)
        acquire.cancel()
        await asyncio.wait_for(broker.release_started.wait(), timeout=0.5)
        acquire.cancel()
        await asyncio.sleep(0)
        assert acquire.done() is False
        broker.finish_release.set()
        with pytest.raises(asyncio.CancelledError):
            await acquire
        assert await broker.authority_for(ROBOT_ID) is None
        assert coordinator._command_claims == {}
        await coordinator.stop()

    asyncio.run(scenario())


def test_cancel_during_broker_begin_return_cannot_orphan_untracked_claim(
    shadow_session_tool,
):
    class DelayedBeginBroker(CommandBroker):
        def __init__(self) -> None:
            super().__init__()
            self.claim_created = asyncio.Event()
            self.release_started = asyncio.Event()
            self.finish_release = asyncio.Event()

        async def begin_authority(self, robot_id: str, token: str, **kwargs):
            claim = await super().begin_authority(robot_id, token, **kwargs)
            self.claim_created.set()
            await asyncio.Event().wait()
            return claim

        async def release_authority(self, robot_id: str, token: str) -> bool:
            self.release_started.set()
            await self.finish_release.wait()
            return await super().release_authority(robot_id, token)

    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = DelayedBeginBroker()
        coordinator = _coordinator(
            FakeD1Caller(),
            MutableClock(),
            command_broker=broker,
        )
        acquire = asyncio.create_task(
            coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID),
        )
        await asyncio.wait_for(broker.claim_created.wait(), timeout=0.5)
        assert await broker.authority_for(ROBOT_ID) is not None
        assert coordinator._command_claims == {}

        acquire.cancel()
        await asyncio.wait_for(broker.release_started.wait(), timeout=0.5)
        acquire.cancel()
        await asyncio.sleep(0)
        assert acquire.done() is False
        broker.finish_release.set()
        with pytest.raises(asyncio.CancelledError):
            await acquire

        assert await broker.authority_for(ROBOT_ID) is None
        assert coordinator._command_claims == {}
        await coordinator.stop()

    asyncio.run(scenario())


def test_cross_id_acquire_requires_descriptor_to_echo_authority_root(
    shadow_session_tool,
):
    async def scenario() -> None:
        tool = deepcopy(shadow_session_tool)
        tool['x-teleop'].pop('robot_id')
        _install_descriptor(tool)
        coordinator = _coordinator(FakeD1Caller(), MutableClock())
        try:
            with pytest.raises(TeleopServiceError) as error:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert error.value.code == 'driver_not_ready'
            assert coordinator._command_claims == {}
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_cancel_after_driver_activation_revokes_authority_and_command_claim(
    shadow_session_tool,
):
    class ActiveUpdateBroker(CommandBroker):
        def __init__(self) -> None:
            super().__init__()
            self.active_update_started = asyncio.Event()

        async def update_authority(self, robot_id: str, token: str, **kwargs):
            if kwargs.get('state') == 'active':
                self.active_update_started.set()
                await asyncio.Event().wait()
            return await super().update_authority(robot_id, token, **kwargs)

    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = ActiveUpdateBroker()
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock(), command_broker=broker)
        acquire = asyncio.create_task(
            coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID),
        )
        await asyncio.wait_for(broker.active_update_started.wait(), timeout=0.8)
        assert fake.authority_valid is True
        acquire.cancel()
        with pytest.raises(asyncio.CancelledError):
            await acquire
        assert fake.authority_valid is False
        assert fake.state == 'released'
        assert await broker.authority_for(ROBOT_ID) is None
        visible = await coordinator.manager.list_visible('', owner=True)
        assert visible and visible[0].state == 'released'
        await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize('malformation', ['missing-ack', 'same-generation'])
def test_prepare_requires_new_recording_stop_generation_before_activation(
    shadow_session_tool,
    malformation,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()

        def unacknowledged_prepare(
            producer: FakeD1Caller,
            arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            assert arguments is not None
            producer.session_id = arguments['session_id']
            producer.epoch = arguments['epoch']
            producer.fence = arguments['fence']
            producer.state = 'prepared_shadow'
            producer.reason = None
            producer.authority_valid = True
            if malformation == 'missing-ack':
                producer.dispatch_generation += 1
            raw = producer.snapshot()
            if malformation == 'missing-ack':
                raw['dispatch']['stop_acknowledged'] = False
            return raw

        fake.queue('prepare_shadow', unacknowledged_prepare)
        coordinator = _coordinator(
            fake,
            MutableClock(),
            command_broker=broker,
        )
        try:
            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            assert raised.value.code == 'driver_response_invalid'
            assert fake.count('release') == 1
            assert fake.authority_valid is False
            assert coordinator._command_claims == {}
            assert await broker.authority_for(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('action', 'state', 'reason', 'malformation', 'expected_code'),
    [
        (
            'pause', 'paused', 'operator_pause', 'missing-ack',
            'driver_pause_rejected',
        ),
        (
            'pause', 'paused', 'operator_pause', 'same-generation',
            'driver_pause_rejected',
        ),
        (
            'soft_stop', 'hold', 'soft_stop', 'missing-ack',
            'driver_soft_stop_rejected',
        ),
        (
            'soft_stop', 'hold', 'soft_stop', 'same-generation',
            'driver_soft_stop_rejected',
        ),
    ],
)
def test_control_stop_requires_recording_ack_or_faults_and_revokes(
    shadow_session_tool,
    action,
    state,
    reason,
    malformation,
    expected_code,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()
        coordinator = _coordinator(
            fake,
            MutableClock(),
            command_broker=broker,
        )
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)

            def unacknowledged_stop(
                producer: FakeD1Caller,
                arguments: dict[str, Any] | None,
            ) -> dict[str, Any]:
                producer._assert_identity(arguments)
                saved = (
                    producer.state,
                    producer.reason,
                    producer.dispatch_generation,
                )
                producer.state = state
                producer.reason = reason
                if malformation == 'missing-ack':
                    producer.dispatch_generation += 1
                raw = producer.snapshot()
                producer.state, producer.reason, producer.dispatch_generation = saved
                if malformation == 'missing-ack':
                    raw['dispatch']['stop_acknowledged'] = False
                return raw

            fake.queue(action, unacknowledged_stop)
            method = getattr(coordinator, action)
            with pytest.raises(TeleopServiceError) as raised:
                await method(
                    acquired.session.id,
                    'alice',
                    CLIENT_ID,
                    owner=False,
                )

            assert raised.value.code == expected_code
            assert acquired.session.state == 'faulted'
            assert fake.count('release') == 1
            assert fake.authority_valid is False
            assert await broker.authority_for(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize('malformation', ['missing-ack', 'same-generation'])
def test_unproven_released_snapshot_keeps_broker_quarantined(
    shadow_session_tool,
    malformation,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()
        coordinator = _coordinator(
            fake,
            MutableClock(),
            command_broker=broker,
        )
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)

            def unacknowledged_release(
                producer: FakeD1Caller,
                arguments: dict[str, Any] | None,
            ) -> dict[str, Any]:
                producer._assert_identity(arguments)
                saved = (
                    producer.session_id,
                    producer.fence,
                    producer.state,
                    producer.reason,
                    producer.authority_valid,
                    producer.dispatch_generation,
                )
                producer.session_id = None
                producer.fence = None
                producer.state = 'released'
                producer.reason = 'operator_release'
                producer.authority_valid = False
                if malformation == 'missing-ack':
                    producer.dispatch_generation += 1
                raw = producer.snapshot()
                (
                    producer.session_id,
                    producer.fence,
                    producer.state,
                    producer.reason,
                    producer.authority_valid,
                    producer.dispatch_generation,
                ) = saved
                if malformation == 'missing-ack':
                    raw['dispatch']['stop_acknowledged'] = False
                return raw

            fake.queue('release', unacknowledged_release)
            released, acknowledged = await coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )

            assert released.state == 'released'
            assert acknowledged is False
            assert await broker.authority_for(ROBOT_ID) is not None
            with pytest.raises(TeleopCommandBlocked):
                async with broker.ordinary_command(
                    ROBOT_ID,
                    read_only=False,
                    source='test',
                    tool='locomotion',
                    action='move',
                ):
                    raise AssertionError('unacknowledged release opened the gate')

            _, retry_acknowledged = await coordinator.release(
                acquired.session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            assert retry_acknowledged is True
            assert await broker.authority_for(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize('malformation', ['missing-dispatch', 'sequence-rollback'])
def test_status_protocol_downgrade_or_sequence_rollback_revokes_authority(
    shadow_session_tool,
    malformation,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()
        coordinator = _coordinator(
            fake,
            MutableClock(),
            command_broker=broker,
        )
        try:
            acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            await coordinator._cancel_heartbeat(acquired.session.id, origin_task=None)
            if malformation == 'missing-dispatch':
                raw = fake.snapshot()
                raw.pop('dispatch')
            else:
                fake.state = 'active_shadow'
                fake.dispatch_last_admitted = 8
                await coordinator.status(
                    acquired.session.id,
                    'alice',
                    owner=False,
                )
                raw = fake.snapshot()
                raw['dispatch']['last_admitted_sequence'] = 7
            fake.queue('status', raw)

            with pytest.raises(TeleopServiceError) as raised:
                await coordinator.status(
                    acquired.session.id,
                    'alice',
                    owner=False,
                )
            assert raised.value.code == 'driver_session_lost'
            assert acquired.session.state == 'faulted'
            assert fake.authority_valid is False
            assert await broker.authority_for(ROBOT_ID) is None
        finally:
            await coordinator.stop()

    asyncio.run(scenario())


def test_status_exposes_stop_pending_without_treating_it_as_stop_proof(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()
        coordinator = _coordinator(
            fake,
            MutableClock(),
            command_broker=broker,
        )
        try:
            session = (
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            ).session
            await coordinator._cancel_heartbeat(session.id, origin_task=None)
            pending = fake.snapshot(
                state='hold',
                reason='deadman_released',
            )
            pending['dispatch'].update({
                'state': 'safe_reclutch_required',
                'generation': fake.dispatch_generation + 1,
                'stop_queue_depth': 1,
                'last_decision': 'stop_requested:deadman_released',
                'stop_acknowledged': False,
            })
            fake.queue('status', pending)

            public_pending = await coordinator.status(
                session.id,
                'alice',
                owner=False,
            )
            assert session.state == 'active'
            assert public_pending['driver']['state'] == 'hold'
            assert public_pending['driver']['dispatch']['stop_acknowledged'] is False
            assert public_pending['driver']['dispatch']['stop_queue_depth'] == 1
            assert await broker.authority_for(ROBOT_ID) is not None

            fake.state = 'hold'
            fake.reason = 'deadman_released'
            fake.dispatch_generation += 1
            public_stable = await coordinator.status(
                session.id,
                'alice',
                owner=False,
            )
            assert public_stable['driver']['dispatch']['stop_acknowledged'] is True
            assert public_stable['driver']['dispatch']['stop_queue_depth'] == 0
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_delayed_heartbeat_snapshot_cannot_overwrite_newer_pause_proof(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller(lease_seconds=0.8)
        coordinator = _coordinator(fake, MutableClock())
        heartbeat_started = asyncio.Event()
        allow_heartbeat_return = asyncio.Event()
        heartbeat_returned = asyncio.Event()

        async def delayed_old_heartbeat(
            producer: FakeD1Caller,
            arguments: dict[str, Any] | None,
        ) -> dict[str, Any]:
            producer._assert_identity(arguments)
            captured = producer.snapshot()
            heartbeat_started.set()
            await allow_heartbeat_return.wait()
            heartbeat_returned.set()
            return captured

        try:
            session = (
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            ).session
            fake.queue('heartbeat', delayed_old_heartbeat)
            await asyncio.wait_for(heartbeat_started.wait(), timeout=0.6)

            paused = await coordinator.pause(
                session.id,
                'alice',
                CLIENT_ID,
                owner=False,
            )
            allow_heartbeat_return.set()
            await asyncio.wait_for(heartbeat_returned.wait(), timeout=0.5)
            await asyncio.sleep(0.02)

            cached = coordinator._driver_snapshots[session.id]
            assert paused.state == 'paused'
            assert cached['state'] == 'paused'
            assert cached['dispatch']['generation'] == 2
            assert cached['dispatch']['stop_acknowledged'] is True
            assert coordinator._heartbeat_health[session.id].state == 'healthy'
        finally:
            allow_heartbeat_return.set()
            await _finish(coordinator)

    asyncio.run(scenario())


def test_delayed_pending_heartbeat_cannot_replace_stable_stop_or_mark_health(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        clock = MutableClock()
        fake = FakeD1Caller(lease_seconds=0.8)
        coordinator = _coordinator(fake, clock)
        heartbeat_started = asyncio.Event()
        allow_heartbeat_return = asyncio.Event()

        try:
            session = (
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            ).session
            initial_confirmation = coordinator._heartbeat_health[
                session.id
            ].last_confirmed_monotonic
            pending = fake.snapshot(state='paused', reason='operator_pause')
            pending['dispatch'].update({
                'state': 'safe_latched',
                'generation': fake.dispatch_generation + 1,
                'stop_queue_depth': 1,
                'last_decision': 'stop_requested:operator_pause',
                'stop_acknowledged': False,
            })

            async def delayed_pending_heartbeat(
                producer: FakeD1Caller,
                arguments: dict[str, Any] | None,
            ) -> dict[str, Any]:
                producer._assert_identity(arguments)
                heartbeat_started.set()
                await allow_heartbeat_return.wait()
                return pending

            fake.queue('heartbeat', delayed_pending_heartbeat)
            await asyncio.wait_for(heartbeat_started.wait(), timeout=0.6)

            stable = deepcopy(pending)
            stable['dispatch'].update({
                'stop_queue_depth': 0,
                'last_decision': 'would_stop:operator_pause',
                'stop_acknowledged': True,
            })
            fake.queue('status', stable)
            await coordinator.status(session.id, 'alice', owner=False)
            clock.advance(0.1)
            allow_heartbeat_return.set()
            await asyncio.sleep(0.02)
            await coordinator._cancel_heartbeat(session.id, origin_task=None)

            cached = coordinator._driver_snapshots[session.id]
            assert cached['dispatch']['stop_acknowledged'] is True
            assert cached['dispatch']['stop_queue_depth'] == 0
            assert coordinator._heartbeat_health[
                session.id
            ].last_confirmed_monotonic == initial_confirmation

            fake.state = 'paused'
            fake.reason = 'operator_pause'
            fake.dispatch_generation += 1
        finally:
            allow_heartbeat_return.set()
            await _finish(coordinator)

    asyncio.run(scenario())


def test_terminal_status_requires_same_epoch_revocation_or_restart_startup_ack(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        fake = FakeD1Caller()
        coordinator = _coordinator(fake, MutableClock())
        try:
            session = (
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            ).session

            def projected_terminal(
                *,
                state: str,
                reason: str | None,
                boot_id: str,
                epoch: int,
                pending: bool = False,
            ) -> dict[str, Any]:
                raw = fake.snapshot()
                raw.update({
                    'boot_id': boot_id,
                    'session_id': None,
                    'epoch': epoch,
                    'state': state,
                    'reason': reason,
                    'authority_valid': False,
                })
                raw['lease'].update({
                    'age_ms': None,
                    'fresh': False,
                    'authority_valid': False,
                    'expired_latched': reason == 'lease_timeout',
                })
                raw['dispatch'].update({
                    'state': (
                        'safe_unarmed' if state == 'idle' else 'safe_revoked'
                    ),
                    'generation': fake.dispatch_generation + 1,
                    'mailbox_depth': 0,
                    'stop_queue_depth': int(pending),
                    'last_decision': (
                        'startup_safe_ack'
                        if state == 'idle'
                        else f'stop_requested:{reason}'
                        if pending
                        else f'would_stop:{reason}'
                    ),
                    'stop_acknowledged': not pending,
                    'fault_code': None,
                    'io_inflight': None,
                })
                projected, _ = _project_driver_snapshot(
                    raw,
                    driver_id=DRIVER_ID,
                    robot_id=ROBOT_ID,
                    capability_digest=DIGEST,
                    action='status',
                    allow_stop_pending=pending,
                )
                return projected

            same_boot_idle = projected_terminal(
                state='idle',
                reason=None,
                boot_id=session.boot_id,
                epoch=session.epoch,
            )
            same_boot_released = projected_terminal(
                state='released',
                reason='operator_release',
                boot_id=session.boot_id,
                epoch=session.epoch,
            )
            wrong_epoch_released = projected_terminal(
                state='released',
                reason='operator_release',
                boot_id=session.boot_id,
                epoch=session.epoch + 1,
            )
            restarted_idle = projected_terminal(
                state='idle',
                reason=None,
                boot_id=str(uuid.uuid4()),
                epoch=0,
            )
            same_boot_release_pending = projected_terminal(
                state='released',
                reason='operator_release',
                boot_id=session.boot_id,
                epoch=session.epoch,
                pending=True,
            )

            assert coordinator._terminal_status_proves_safe(
                session,
                same_boot_idle,
            ) is False
            assert coordinator._terminal_status_proves_safe(
                session,
                same_boot_released,
            ) is True
            assert coordinator._terminal_status_proves_safe(
                session,
                wrong_epoch_released,
            ) is False
            assert coordinator._terminal_status_proves_safe(
                session,
                restarted_idle,
            ) is True
            assert coordinator._terminal_status_proves_safe(
                session,
                same_boot_release_pending,
            ) is False
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_terminal_cleanup_rejects_unproven_unlock(shadow_session_tool):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        coordinator = _coordinator(
            FakeD1Caller(),
            MutableClock(),
            command_broker=broker,
        )
        try:
            session = (
                await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
            ).session
            with pytest.raises(RuntimeError, match='acknowledged safety proof'):
                await coordinator._cleanup_terminal_runtime(session.id)
            assert await broker.authority_for(ROBOT_ID) is not None
        finally:
            await _finish(coordinator)

    asyncio.run(scenario())


def test_terminal_cleanup_retains_quarantined_snapshot_baseline():
    async def scenario() -> None:
        coordinator = _coordinator(FakeD1Caller(), MutableClock())
        coordinator._terminal_cleanup_done.add('completed-session')
        coordinator._driver_snapshots.update({
            'quarantined-session': {'dispatch': {'generation': 9}},
            'unretained-session': {'dispatch': {'generation': 4}},
        })
        coordinator._driver_snapshot_revisions.update({
            'quarantined-session': 3,
            'unretained-session': 2,
        })
        coordinator._command_claims['quarantined-session'] = (
            ROBOT_ID,
            'quarantine-token',
        )

        await coordinator._cleanup_terminal_runtime('completed-session')

        assert 'quarantined-session' in coordinator._driver_snapshots
        assert 'quarantined-session' in coordinator._driver_snapshot_revisions
        assert 'unretained-session' not in coordinator._driver_snapshots
        assert 'unretained-session' not in coordinator._driver_snapshot_revisions

    asyncio.run(scenario())


def test_shutdown_keeps_unconfirmed_driver_release_quarantined(
    shadow_session_tool,
):
    async def scenario() -> None:
        _install_descriptor(shadow_session_tool)
        broker = CommandBroker()
        fake = FakeD1Caller()
        coordinator = _coordinator(
            fake,
            MutableClock(),
            command_broker=broker,
        )
        acquired = await coordinator.acquire(DRIVER_ID, 'alice', CLIENT_ID)
        fake.queue(
            'release',
            mcp_client.TrustedShadowTransportError('timeout'),
        )

        await coordinator.stop()

        claim = await broker.authority_for(ROBOT_ID)
        assert claim is not None
        assert claim.state == 'releasing'
        assert acquired.session.id in coordinator._command_claims
        assert coordinator._http_session is None
        with pytest.raises(TeleopCommandBlocked):
            async with broker.ordinary_command(
                ROBOT_ID,
                read_only=False,
                source='test',
                tool='locomotion',
                action='move',
            ):
                raise AssertionError('shutdown opened an unconfirmed write gate')

    asyncio.run(scenario())
