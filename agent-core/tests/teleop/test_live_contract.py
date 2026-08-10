from __future__ import annotations

import asyncio
import threading
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

import aiohttp
import pytest

import config
import mcp_client
from api import teleop
from teleop import authority_guard
from teleop.command_broker import CommandBroker
from teleop.service import TeleopCoordinator, TeleopServiceError
from teleop.session_manager import SessionStateConflict, ShadowSessionManager

CLIENT_ID = '7dbabfca-15c1-43ca-b600-75e7682c21d0'


def _live_tool(shadow_session_tool: dict) -> dict:
    tool = deepcopy(shadow_session_tool)
    actions = tool['inputSchema']['x-action-params']
    prepare = actions.pop('prepare_shadow')
    actions['prepare_live'] = prepare
    action_enum = tool['inputSchema']['properties']['action']['enum']
    action_enum[action_enum.index('prepare_shadow')] = 'prepare_live'
    descriptor = tool['x-teleop']
    descriptor.pop('dry_run_profile')
    descriptor.update({
        'protocol': 'motus.teleop.live.v1',
        'mode': 'live',
        'actuation_enabled': True,
        'dispatch_contract': 'motus.teleop.dispatch.hardware.v1',
        'profile_id': 'dual_arm_profile_v1',
        'capabilities': {
            'profile_id': 'dual_arm_profile_v1',
            'input_bindings': {
                'head': {'required': True, 'role': 'reference'},
                'left_controller': {
                    'required': True,
                    'role': 'left_end_effector',
                },
                'right_controller': {
                    'required': True,
                    'role': 'right_end_effector',
                },
            },
            'outputs': {
                'dual_arm': {'enabled': True, 'joint_count': 10},
                'base': {'enabled': False},
                'hands': {'enabled': False},
            },
            'effectors': ['dual_arm'],
        },
    })
    descriptor['signaling']['audience'] = 'motus-teleop-rtc'
    return tool


def _trusted_record(tool: dict) -> dict:
    tool = deepcopy(tool)
    record = {
        'id': 'live-teleop-driver',
        'name': 'Live Teleop Driver',
        'server_name': 'teleop',
        'url': 'http://127.0.0.1:15711/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [tool],
    }
    tool['x-teleop']['driver_id'] = record['id']
    tool['x-teleop']['robot_id'] = record['id']
    return record


def _install_live(tool: dict) -> dict:
    record = _trusted_record(tool)
    config.main['services'] = {'mcp': [record]}
    mcp_client.registry[record['id']] = {
        'online': True,
        'trusted': True,
        'url': record['url'],
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(record['tools'][0]),
    }
    return record


def _empty_latency() -> dict[str, Any]:
    return {'last': None, 'p50': None, 'p95': None, 'p99': None, 'count': 0}


class _LiveCaller:
    def __init__(self, driver_id: str):
        self.driver_id = driver_id
        self.robot_id = driver_id
        self.boot_id = 'f77787b9-c5d2-465a-b4b4-e74f61f35e30'
        self.epoch = 7
        self.session_id: str | None = None
        self.fence: str | None = None
        self.state = 'idle'
        self.reason: str | None = None
        self.authority_valid = False
        self.dispatch_generation = 0
        self.calls: list[dict[str, Any]] = []
        self.script: dict[str, deque[Any]] = defaultdict(deque)

    def queue(self, action: str, *outcomes: Any) -> None:
        self.script[action].extend(outcomes)

    def count(self, action: str) -> int:
        return sum(call['action'] == action for call in self.calls)

    def latch_live_fault(self) -> None:
        self.session_id = None
        self.fence = None
        self.state = 'fault'
        self.reason = 'dispatch_fault'
        self.authority_valid = False
        self.dispatch_generation += 1

    def snapshot(self) -> dict[str, Any]:
        dispatch_state = {
            'idle': 'safe_unarmed',
            'prepared_live': 'safe_waiting_frame',
            'active_live': 'motion_eligible',
            'paused': 'safe_latched',
            'released': 'safe_revoked',
            'fault': 'fault_latched',
        }.get(self.state, 'safe_reclutch_required')
        if self.state == 'hold' and self.reason == 'soft_stop':
            dispatch_state = 'safe_latched'
        decision = {
            'idle': 'startup_safe_ack',
            'prepared_live': 'prepared_after_stop_ack',
            'active_live': 'motion_committed',
            'fault': 'async_fault:arm_sdk_async_fault',
        }.get(self.state, f'would_stop:{self.reason}')
        admitted = 1 if self.state in {'active_live', 'fault'} else None
        joints = [0.0] * 10 if self.authority_valid or self.state == 'fault' else []
        return {
            'driver_id': self.driver_id,
            'driver_type': 'teleop',
            'robot_id': self.robot_id,
            'mode': 'live',
            'actuation_enabled': True,
            'boot_id': self.boot_id,
            'session_id': self.session_id,
            'epoch': self.epoch,
            'state': self.state,
            'reason': self.reason,
            'authority_valid': self.authority_valid,
            'capability_digest': '0123456789abcdef' * 4,
            'lease': {
                'source': 'agent-core-mcp-heartbeat-only',
                'timeout_ms': 1_000.0,
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
            },
            'rtc': {
                'connected': False,
                'channels': {'teleop-control': False, 'teleop-pose': False},
                'renews_lease': False,
            },
            'dispatch': {
                'kind': 'hardware',
                'state': dispatch_state,
                'ready': self.state != 'fault',
                'generation': self.dispatch_generation,
                'mailbox_depth': 0,
                'stop_queue_depth': 0,
                'last_admitted_sequence': admitted,
                'last_published_sequence': admitted,
                'last_decision': decision,
                'stop_acknowledged': self.state != 'fault',
                'fault_code': 'arm_sdk_async_fault' if self.state == 'fault' else None,
                'io_inflight': None,
                'counters': {'startup_safe_acks': 1, 'stop_acks': 1},
            },
            'diagnostics': {
                'transport': {
                    'rtc_rtt_ms': None,
                    'pose_age_ms': None,
                    'frame_rate_hz': None,
                    'frames_received': 0,
                    'frames_rejected': 0,
                    'sequence_gaps': 0,
                    'mailbox_replacements': 0,
                },
                'latency_ms': {
                    stage: _empty_latency()
                    for stage in (
                        'receive_to_admit', 'mailbox_wait', 'ik',
                        'adapter_apply', 'robot_follow',
                    )
                },
            },
            'output': {
                'profile_id': 'dual_arm_profile_v1',
                'hardware_output': True,
                'state': self.state,
                'target_joint_positions_rad': joints,
                'measured_joint_positions_rad': list(joints),
                'max_abs_error_rad': None,
                'arm_sdk_weight': 0.0,
                'command_age_ms': None,
                'fault_reason': 'arm_sdk_async_fault' if self.state == 'fault' else None,
            },
            'counters': {},
        }

    async def __call__(
        self,
        driver_id: str,
        action: str,
        arguments: dict[str, Any] | None,
        *,
        timeout_seconds: float,
        session: aiohttp.ClientSession,
        target: mcp_client.TrustedShadowTarget | None = None,
    ) -> dict[str, Any]:
        assert driver_id == self.driver_id
        assert timeout_seconds > 0
        assert not session.closed
        self.calls.append({
            'action': action,
            'arguments': deepcopy(arguments),
            'target': target,
        })
        if self.script[action]:
            outcome = self.script[action].popleft()
            if callable(outcome):
                outcome = outcome(self, arguments)
                if asyncio.iscoroutine(outcome):
                    outcome = await outcome
            if isinstance(outcome, BaseException):
                raise outcome
            return deepcopy(outcome)
        if action == 'status':
            return self.snapshot()
        if self.state == 'fault' and action in {'heartbeat', 'release', 'soft_stop'}:
            raise RuntimeError('fault-latched Driver requires process restart')
        if action == 'prepare_live':
            assert arguments is not None
            self.session_id = arguments['session_id']
            self.epoch = arguments['epoch']
            self.fence = arguments['fence']
            self.state = 'prepared_live'
            self.authority_valid = True
            self.dispatch_generation += 1
            return self.snapshot()
        if action == 'heartbeat':
            assert arguments is not None and arguments['fence'] == self.fence
            return self.snapshot()
        if action == 'release':
            self.session_id = None
            self.fence = None
            self.state = 'released'
            self.reason = 'operator_release'
            self.authority_valid = False
            self.dispatch_generation += 1
            return self.snapshot()
        if action == 'soft_stop':
            self.state = 'hold'
            self.reason = 'soft_stop'
            self.dispatch_generation += 1
            return self.snapshot()
        raise AssertionError(f'unexpected live action {action}')


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError('condition was not reached before the test deadline')
        await asyncio.sleep(0.01)


def _pin_injected_live_target(
    coordinator: TeleopCoordinator,
    record: dict,
    session_id: str,
    capability_digest: str,
) -> mcp_client.TrustedShadowTarget:
    target = mcp_client.TrustedShadowTarget(
        mcp_id=record['id'],
        url=record['url'],
        capability_digest=capability_digest,
        descriptor_fingerprint=mcp_client.teleop_tool_fingerprint(
            record['tools'][0],
        ),
        actions=frozenset(
            record['tools'][0]['inputSchema']['properties']['action']['enum'],
        ),
    )
    coordinator._pinned_targets[session_id] = target
    coordinator._uses_pinned_targets = True
    return target


def test_live_descriptor_is_explicit_and_adapter_neutral(shadow_session_tool):
    tool = _live_tool(shadow_session_tool)

    assert teleop._valid_teleop_descriptor(tool) is True
    assert teleop._valid_shadow_descriptor(tool) is False

    for path, value in (
        (('actuation_enabled',), False),
        (('protocol',), 'motus.teleop.shadow.v1'),
        (('dispatch_contract',), 'motus.teleop.dispatch.recording.v1'),
        (('signaling', 'audience'), 'teleop-shadow-rtc'),
    ):
        malformed = deepcopy(tool)
        target = malformed['x-teleop']
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert teleop._valid_teleop_descriptor(malformed) is False


@pytest.mark.parametrize('malformation', ['duplicate', 'non_string', 'undeclared'])
def test_live_descriptor_rejects_ambiguous_action_declarations(
    shadow_session_tool,
    malformation,
):
    tool = _live_tool(shadow_session_tool)
    action_enum = tool['inputSchema']['properties']['action']['enum']
    if malformation == 'duplicate':
        action_enum.append(action_enum[0])
    elif malformation == 'non_string':
        action_enum.append(7)
    else:
        action_enum.append('adapter_private_action')

    assert teleop._valid_teleop_descriptor(tool) is False


def test_base_speed_can_only_come_from_bounded_capability_binding(
    shadow_session_tool,
):
    tool = _live_tool(shadow_session_tool)
    descriptor = tool['x-teleop']
    descriptor['capabilities']['outputs']['base']['enabled'] = True
    descriptor['capabilities']['input_bindings']['base_twist'] = {
        'linear_x': {
            'hand': 'left', 'axis': 3, 'scale': 0.5,
            'deadzone': 0.2, 'direction': -1,
        },
        'linear_y': {
            'hand': 'left', 'axis': 2, 'scale': 0.3,
            'deadzone': 0.2, 'direction': -1,
        },
        'angular_z': {
            'hand': 'right', 'axis': 2, 'scale': 0.6,
            'deadzone': 0.2, 'direction': -1,
        },
    }
    descriptor['capabilities']['effectors'].append('base')
    assert teleop._valid_teleop_descriptor(tool) is True

    tool['x-teleop']['capabilities']['input_bindings']['base_twist'][
        'linear_x'
    ]['scale'] = 100
    assert teleop._valid_teleop_descriptor(tool) is False


def test_live_manager_defers_identity_and_requires_one_confirmation():
    async def scenario():
        manager = ShadowSessionManager()
        capabilities = {
            'profile_id': 'dual_arm_profile_v1',
            'input_bindings': {},
            'outputs': {'dual_arm': {'enabled': True, 'joint_count': 10}},
            'effectors': ['dual_arm'],
        }
        session = await manager.reserve(
            'robot-live',
            'operator:alice',
            driver_id='driver-live',
            boot_id='',
            capability_digest='a' * 64,
            client_id=CLIENT_ID,
            mode='live',
            profile_id='dual_arm_profile_v1',
            capabilities=capabilities,
            effectors=['dual_arm'],
            signaling_audience='motus-teleop-rtc',
            defer_identity=True,
        )
        capabilities['outputs']['dual_arm']['joint_count'] = 99
        assert session.state == 'awaiting_confirmation'
        assert session.epoch == 0
        assert session.boot_id == ''
        assert session.live_confirmed is False
        assert session.capabilities['outputs']['dual_arm']['joint_count'] == 10

        confirmed = await manager.confirm_live_identity(
            session.id,
            'operator:alice',
            CLIENT_ID,
            boot_id='driver-boot-live',
            minimum_epoch=4,
        )
        assert confirmed.state == 'preparing'
        assert confirmed.epoch >= 4
        assert confirmed.live_confirmed is True
        with pytest.raises(SessionStateConflict):
            await manager.confirm_live_identity(
                session.id,
                'operator:alice',
                CLIENT_ID,
                boot_id='driver-boot-live',
                minimum_epoch=5,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'signaling_audience': 'teleop-shadow-rtc'}, 'signaling_audience'),
        ({'profile_id': 'other_profile'}, 'capabilities.profile_id'),
        ({'effectors': []}, 'effectors'),
    ],
)
def test_live_manager_rejects_contradictory_contract_projection(override, message):
    async def scenario():
        manager = ShadowSessionManager()
        capabilities = {
            'profile_id': 'dual_arm_profile_v1',
            'input_bindings': {},
            'outputs': {'dual_arm': {'enabled': True, 'joint_count': 10}},
            'effectors': ['dual_arm'],
        }
        arguments = {
            'driver_id': 'driver-live',
            'boot_id': '',
            'capability_digest': 'a' * 64,
            'client_id': CLIENT_ID,
            'mode': 'live',
            'profile_id': 'dual_arm_profile_v1',
            'capabilities': capabilities,
            'effectors': ['dual_arm'],
            'signaling_audience': 'motus-teleop-rtc',
            'defer_identity': True,
        }
        arguments.update(override)
        with pytest.raises(ValueError, match=message):
            await manager.reserve(
                'robot-live',
                'operator:alice',
                **arguments,
            )
        assert await manager.active_for_robot('robot-live') is None

    asyncio.run(scenario())


def test_live_acquire_and_release_never_contact_driver_before_confirmation(
    shadow_session_tool,
):
    async def scenario():
        record = _trusted_record(_live_tool(shadow_session_tool))
        config.main['services'] = {'mcp': [record]}
        mcp_client.registry[record['id']] = {
            'online': True,
            'trusted': True,
            'url': record['url'],
            'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(
                record['tools'][0],
            ),
        }
        calls = []

        async def caller(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError('Driver must not be contacted before live confirmation')

        broker = CommandBroker()
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=broker,
        )
        result = await coordinator.acquire(
            record['id'],
            'operator:alice',
            CLIENT_ID,
            mode='live',
        )
        assert result.disposition == 'confirmation_required'
        assert result.session.state == 'awaiting_confirmation'
        assert calls == []
        assert coordinator.list_authority_guards() == []
        assert authority_guard.get_guard(record['id']) is None
        assert record['id'] not in coordinator._authority_guards
        claim = await broker.authority_for(record['id'])
        assert claim is not None
        assert claim.state == 'awaiting_confirmation'

        released, acknowledged = await coordinator.release(
            result.session.id,
            'operator:alice',
            CLIENT_ID,
            owner=False,
        )
        assert acknowledged is True
        assert released.state == 'released'
        assert calls == []
        assert await broker.authority_for(record['id']) is None

    asyncio.run(scenario())


def test_confirm_live_happy_path_is_ordered_and_idempotent(shadow_session_tool):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        broker = CommandBroker()
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=broker,
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        assert caller.calls == []

        confirmed = await coordinator.confirm_live(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        )
        assert confirmed.disposition == 'created'
        assert confirmed.session.state == 'active'
        assert confirmed.session.live_confirmed is True
        assert [call['action'] for call in caller.calls[:3]] == [
            'status', 'prepare_live', 'heartbeat',
        ]
        assert coordinator.public_session(confirmed.session)['driver']['mode'] == 'live'
        assert coordinator.public_session(confirmed.session)['driver'][
            'output'
        ]['hardware_output'] is True
        persisted_guard = authority_guard.get_guard(record['id'])
        assert persisted_guard is not None
        assert persisted_guard.session_id == confirmed.session.id
        assert persisted_guard.phase == 'active'
        assert coordinator._authority_guards[record['id']].phase == 'active'

        prepare_count = caller.count('prepare_live')
        retried = await coordinator.confirm_live(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        )
        assert retried.disposition == 'existing'
        assert caller.count('prepare_live') == prepare_count

        await coordinator.release(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            owner=False,
        )
        await coordinator.stop()

    asyncio.run(scenario())


def test_live_status_preserves_strict_driver_fault_evidence_for_console(
    shadow_session_tool,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=CommandBroker(),
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        confirmed = await coordinator.confirm_live(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        )
        caller.latch_live_fault()

        public = await coordinator.status(
            confirmed.session.id,
            'operator:alice',
            owner=False,
        )

        assert public['state'] == 'faulted'
        assert public['driver']['state'] == 'fault'
        assert public['driver']['reason'] == 'dispatch_fault'
        assert public['driver']['authority_valid'] is False
        assert public['driver']['session_id'] is None
        assert public['driver']['dispatch']['state'] == 'fault_latched'
        assert public['driver']['dispatch']['fault_code'] == 'arm_sdk_async_fault'
        assert public['driver']['dispatch']['stop_acknowledged'] is False
        assert public['driver']['output']['state'] == 'fault'
        assert public['driver']['output']['arm_sdk_weight'] == 0.0
        assert authority_guard.get_guard(record['id']) is not None
        await coordinator.stop()

    asyncio.run(scenario())


def test_terminal_heartbeat_rpc_captures_one_strict_fault_status_for_console(
    shadow_session_tool,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=CommandBroker(),
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        confirmed = await coordinator.confirm_live(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        )
        status_before = caller.count('status')
        heartbeat_before = caller.count('heartbeat')
        pinned_target = _pin_injected_live_target(
            coordinator,
            record,
            confirmed.session.id,
            confirmed.session.capability_digest,
        )
        caller.latch_live_fault()
        caller.queue(
            'heartbeat',
            mcp_client.TrustedShadowTransportError(
                'rpc_error',
                rpc_code=-32602,
                rpc_data_code='session_inactive',
            ),
        )

        await _wait_until(lambda: confirmed.session.state == 'faulted')

        public = coordinator.public_session(confirmed.session)
        assert caller.count('heartbeat') == heartbeat_before + 1
        assert caller.count('status') == status_before + 1
        terminal_status = [
            call for call in caller.calls if call['action'] == 'status'
        ][-1]
        assert terminal_status['arguments'] is None
        assert terminal_status['target'] is pinned_target
        assert public['state'] == 'faulted'
        assert public['driver']['state'] == 'fault'
        assert public['driver']['dispatch']['fault_code'] == 'arm_sdk_async_fault'
        assert public['driver']['output']['fault_reason'] == 'arm_sdk_async_fault'
        assert public['driver_heartbeat']['state'] == 'faulted'
        assert authority_guard.get_guard(record['id']) is not None
        await coordinator.stop()

    asyncio.run(scenario())


def test_terminal_heartbeat_status_failure_keeps_original_fail_closed_path(
    shadow_session_tool,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=CommandBroker(),
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        confirmed = await coordinator.confirm_live(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        )
        healthy = deepcopy(coordinator.public_session(confirmed.session)['driver'])
        status_before = caller.count('status')
        heartbeat_before = caller.count('heartbeat')
        heartbeat_task = coordinator._heartbeat_tasks[confirmed.session.id]
        _pin_injected_live_target(
            coordinator,
            record,
            confirmed.session.id,
            confirmed.session.capability_digest,
        )
        caller.latch_live_fault()
        caller.queue(
            'heartbeat',
            mcp_client.TrustedShadowTransportError(
                'rpc_error',
                rpc_code=-32602,
                rpc_data_code='session_inactive',
            ),
        )
        caller.queue(
            'status',
            mcp_client.TrustedShadowTransportError('timeout'),
        )

        # ``faulted`` publishes Core's local revoke immediately; the background
        # heartbeat task remains the completion signal for remote safety cleanup.
        await asyncio.wait_for(asyncio.shield(heartbeat_task), timeout=1.0)

        public = coordinator.public_session(confirmed.session)
        assert caller.count('heartbeat') == heartbeat_before + 1
        assert caller.count('status') == status_before + 1
        assert caller.count('soft_stop') == 1
        assert caller.count('release') == 1
        assert public['state'] == 'faulted'
        assert public['driver'] == healthy
        assert public['driver']['dispatch']['fault_code'] is None
        assert public['driver_heartbeat']['state'] == 'faulted'
        assert authority_guard.get_guard(record['id']) is not None
        await coordinator.stop()

    asyncio.run(scenario())


def test_confirm_live_rejects_wrong_profile_and_tab_without_driver_contact(
    shadow_session_tool,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=CommandBroker(),
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        with pytest.raises(TeleopServiceError) as wrong_profile:
            await coordinator.confirm_live(
                acquired.session.id,
                'operator:alice',
                CLIENT_ID,
                profile_id='different_profile',
            )
        assert wrong_profile.value.code == 'live_confirmation_mismatch'
        with pytest.raises(Exception) as wrong_tab:
            await coordinator.confirm_live(
                acquired.session.id,
                'operator:alice',
                '68991413-d37a-4603-9f07-3e25219d6d96',
                profile_id='dual_arm_profile_v1',
            )
        assert wrong_tab.type.__name__ == 'SessionClientMismatch'
        assert caller.calls == []
        await coordinator.release(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            owner=False,
        )

    asyncio.run(scenario())


def test_waiting_live_blocks_every_driver_facing_operation(shadow_session_tool):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=CommandBroker(),
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        operations = (
            coordinator.heartbeat(
                acquired.session.id, 'operator:alice', CLIENT_ID,
            ),
            coordinator.status(
                acquired.session.id, 'operator:alice', owner=False,
            ),
            coordinator.signaling_offer(
                acquired.session.id,
                'operator:alice',
                CLIENT_ID,
                {'type': 'offer', 'sdp': 'v=0'},
            ),
            coordinator.pause(
                acquired.session.id,
                'operator:alice',
                CLIENT_ID,
                owner=False,
            ),
            coordinator.soft_stop(
                acquired.session.id,
                'operator:alice',
                CLIENT_ID,
                owner=False,
            ),
        )
        for operation in operations:
            with pytest.raises(SessionStateConflict):
                await operation
        assert caller.calls == []
        await coordinator.release(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            owner=False,
        )

    asyncio.run(scenario())


def test_unconfirmed_live_expiry_never_contacts_driver(shadow_session_tool):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        broker = CommandBroker()
        manager = ShadowSessionManager()
        coordinator = TeleopCoordinator(
            session_manager=manager,
            caller=caller,
            command_broker=broker,
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        acquired.session.deadline_monotonic = 0.0
        expired = await manager.expire_due()
        assert expired == [acquired.session]
        await coordinator._complete_expiry_cleanup(
            acquired.session,
            origin_task=asyncio.current_task(),
        )
        assert caller.calls == []
        assert await broker.authority_for(record['id']) is None

    asyncio.run(scenario())


def test_live_guard_failure_releases_memory_authority_without_driver_prepare(
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        broker = CommandBroker()
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=broker,
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )

        def reject_guard(_guard):
            raise OSError('simulated durable guard failure')

        monkeypatch.setattr(authority_guard, 'create_guard', reject_guard)
        with pytest.raises(TeleopServiceError) as raised:
            await coordinator.confirm_live(
                acquired.session.id,
                'operator:alice',
                CLIENT_ID,
                profile_id='dual_arm_profile_v1',
            )
        assert raised.value.code == 'authority_guard_persistence_error'
        assert caller.count('status') == 1
        assert caller.count('prepare_live') == 0
        assert acquired.session.state == 'faulted'
        assert await broker.authority_for(record['id']) is None
        assert coordinator.list_authority_guards() == []
        assert authority_guard.get_guard(record['id']) is None
        assert record['id'] not in coordinator._authority_guards
        await coordinator.stop()

    asyncio.run(scenario())


def test_cancelled_live_status_keeps_confirmation_pending_and_driver_unarmed(
    shadow_session_tool,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        broker = CommandBroker()
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=broker,
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        status_started = asyncio.Event()

        async def slow_status(_producer, _arguments):
            status_started.set()
            await asyncio.Event().wait()

        caller.queue('status', slow_status)
        confirmation = asyncio.create_task(coordinator.confirm_live(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        ))
        await asyncio.wait_for(status_started.wait(), timeout=0.8)
        confirmation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await confirmation

        assert acquired.session.state == 'awaiting_confirmation'
        assert acquired.session.live_confirmed is False
        assert acquired.session.epoch == 0
        assert caller.count('prepare_live') == 0
        assert authority_guard.get_guard(record['id']) is None
        claim = await broker.authority_for(record['id'])
        assert claim is not None and claim.state == 'awaiting_confirmation'
        await coordinator.release(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            owner=False,
        )
        assert await broker.authority_for(record['id']) is None
        await coordinator.stop()

    asyncio.run(scenario())


def test_cancel_after_live_identity_commit_abandons_reservation_without_prepare(
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        broker = CommandBroker()
        manager = ShadowSessionManager()
        coordinator = TeleopCoordinator(
            session_manager=manager,
            caller=caller,
            command_broker=broker,
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        original = manager.confirm_live_identity

        async def commit_then_cancel(*args, **kwargs):
            await original(*args, **kwargs)
            raise asyncio.CancelledError

        monkeypatch.setattr(manager, 'confirm_live_identity', commit_then_cancel)
        with pytest.raises(asyncio.CancelledError):
            await coordinator.confirm_live(
                acquired.session.id,
                'operator:alice',
                CLIENT_ID,
                profile_id='dual_arm_profile_v1',
            )

        assert acquired.session.live_confirmed is True
        assert acquired.session.state == 'faulted'
        assert caller.count('status') == 1
        assert caller.count('prepare_live') == 0
        assert await broker.authority_for(record['id']) is None
        assert authority_guard.get_guard(record['id']) is None
        assert coordinator._command_claims == {}
        await coordinator.stop()

    asyncio.run(scenario())


def test_cancel_after_live_guard_commit_discards_guard_without_driver_prepare(
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        broker = CommandBroker()
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=broker,
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        guard_started = threading.Event()
        allow_guard = threading.Event()
        original = authority_guard.create_guard

        def commit_guard_then_wait(guard):
            created = original(guard)
            guard_started.set()
            assert allow_guard.wait(timeout=2.0)
            return created

        monkeypatch.setattr(authority_guard, 'create_guard', commit_guard_then_wait)
        confirmation = asyncio.create_task(coordinator.confirm_live(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        ))
        await asyncio.wait_for(
            asyncio.to_thread(guard_started.wait, 0.8),
            timeout=1.0,
        )
        confirmation.cancel()
        allow_guard.set()
        with pytest.raises(asyncio.CancelledError):
            await confirmation

        assert acquired.session.state == 'faulted'
        assert caller.count('prepare_live') == 0
        assert authority_guard.get_guard(record['id']) is None
        assert record['id'] not in coordinator._authority_guards
        assert await broker.authority_for(record['id']) is None
        await coordinator.stop()

    asyncio.run(scenario())


def test_cancel_after_live_prepare_applies_revokes_driver_and_guard(
    shadow_session_tool,
):
    async def scenario():
        record = _install_live(_live_tool(shadow_session_tool))
        caller = _LiveCaller(record['id'])
        broker = CommandBroker()
        coordinator = TeleopCoordinator(
            session_manager=ShadowSessionManager(),
            caller=caller,
            command_broker=broker,
        )
        acquired = await coordinator.acquire(
            record['id'], 'operator:alice', CLIENT_ID, mode='live',
        )
        prepare_applied = asyncio.Event()

        async def apply_then_wait(producer, arguments):
            assert arguments is not None
            producer.session_id = arguments['session_id']
            producer.epoch = arguments['epoch']
            producer.fence = arguments['fence']
            producer.state = 'prepared_live'
            producer.authority_valid = True
            producer.dispatch_generation += 1
            prepare_applied.set()
            await asyncio.Event().wait()

        caller.queue('prepare_live', apply_then_wait)
        confirmation = asyncio.create_task(coordinator.confirm_live(
            acquired.session.id,
            'operator:alice',
            CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        ))
        await asyncio.wait_for(prepare_applied.wait(), timeout=0.8)
        confirmation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await confirmation

        assert acquired.session.state == 'faulted'
        assert caller.count('prepare_live') == 1
        assert caller.count('release') == 1
        assert caller.authority_valid is False
        assert caller.session_id is None
        assert authority_guard.get_guard(record['id']) is None
        assert record['id'] not in coordinator._authority_guards
        assert await broker.authority_for(record['id']) is None
        await coordinator.stop()

    asyncio.run(scenario())
