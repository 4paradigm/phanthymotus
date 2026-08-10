from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import fastapi
import httpx
import pytest

import client as client_mod
import config
import mcp_client
from api import canvas, config as config_api, mcp_manage
from teleop import audit
from teleop.command_broker import (
    CommandBroker,
    CommandDrainTimeout,
    TeleopCommandBlocked,
    classify_tool_access,
)
from teleop.service import coordinator


@pytest.mark.parametrize(
    ('tool_type', 'annotations', 'action', 'action_declared', 'read_only'),
    [
        ('sensor', {}, None, False, True),
        ('resource', {}, None, False, True),
        ('actuator', {'readOnlyHint': True}, None, False, True),
        ('actuator', {'readOnlyHint': True}, 'move', True, False),
        ('processor', {}, 'status', True, True),
        ('processor', {}, 'STATUS', True, False),
        ('actuator', {}, 'info', True, True),
        ('actuator', {}, 'info', False, False),
        ('actuator', {}, 'get_up', True, False),
        ('processor', {}, 'move', True, False),
        ('sensor', {}, 'calibrate', True, False),
        ('resource', {}, 'delete', True, False),
        (None, {}, None, False, False),
        ('sensor', {}, 'start', True, False),
        ('resource', {'readOnlyHint': True}, 'config', True, False),
    ],
)
def test_tool_access_classifier_is_fail_closed(
    tool_type,
    annotations,
    action,
    action_declared,
    read_only,
):
    access = classify_tool_access(
        tool_type=tool_type,
        annotations=annotations,
        action=action,
        action_declared=action_declared,
    )

    assert access.read_only is read_only


def test_authority_timeout_removes_provisional_claim():
    async def scenario() -> None:
        broker = CommandBroker()
        write_entered = asyncio.Event()
        finish_write = asyncio.Event()

        async def write() -> None:
            async with broker.ordinary_command(
                'robot-a',
                read_only=False,
                source='test',
                tool='drive',
                action='move',
            ):
                write_entered.set()
                await finish_write.wait()

        write_task = asyncio.create_task(write())
        await write_entered.wait()
        try:
            with pytest.raises(CommandDrainTimeout):
                await broker.begin_authority(
                    'robot-a',
                    'claim-timeout',
                    drain_timeout_seconds=0,
                )
            assert await broker.authority_for('robot-a') is None
        finally:
            finish_write.set()
            await write_task

    asyncio.run(scenario())


def test_same_token_authority_waits_for_the_original_command_drain():
    async def scenario() -> None:
        broker = CommandBroker()
        write_entered = asyncio.Event()
        finish_write = asyncio.Event()

        async def write() -> None:
            async with broker.ordinary_command(
                'robot-a',
                read_only=False,
                source='test',
                tool='drive',
                action='move',
            ):
                write_entered.set()
                await finish_write.wait()

        write_task = asyncio.create_task(write())
        await write_entered.wait()
        first = asyncio.create_task(broker.begin_authority('robot-a', 'same-token'))
        while await broker.authority_for('robot-a') is None:
            await asyncio.sleep(0)
        second = asyncio.create_task(broker.begin_authority('robot-a', 'same-token'))
        await asyncio.sleep(0)
        assert first.done() is False
        assert second.done() is False

        finish_write.set()
        first_claim, second_claim = await asyncio.gather(first, second)
        assert first_claim == second_claim
        assert first_claim.state == 'acquiring'
        await broker.release_authority('robot-a', 'same-token')
        await write_task

    asyncio.run(scenario())


def test_broker_rebinds_conditions_between_event_loops_after_state_drains():
    broker = CommandBroker()

    async def contention_cycle(token: str) -> None:
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def write() -> None:
            async with broker.ordinary_command(
                'robot-loop',
                read_only=False,
                source='test',
                tool='drive',
                action='move',
            ):
                entered.set()
                await finish.wait()

        writer = asyncio.create_task(write())
        await entered.wait()
        claim = asyncio.create_task(broker.begin_authority('robot-loop', token))
        await asyncio.sleep(0)
        finish.set()
        await claim
        await writer
        assert await broker.release_authority('robot-loop', token) is True

    asyncio.run(contention_cycle('loop-one'))
    asyncio.run(contention_cycle('loop-two'))


def test_double_cancelled_ordinary_command_cannot_leak_inflight_admission():
    async def scenario() -> None:
        broker = CommandBroker()
        entered = asyncio.Event()
        cleanup_started = asyncio.Event()
        finish_cleanup = asyncio.Event()
        original_finish = broker._finish_ordinary

        async def delayed_finish(robot_id: str) -> None:
            cleanup_started.set()
            await finish_cleanup.wait()
            await original_finish(robot_id)

        broker._finish_ordinary = delayed_finish  # type: ignore[method-assign]

        async def write() -> None:
            async with broker.ordinary_command(
                'robot-cancel',
                read_only=False,
                source='test',
                tool='drive',
                action='move',
            ):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(write())
        await entered.wait()
        task.cancel()
        await cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        claim = await broker.begin_authority(
            'robot-cancel',
            'after-cancel',
            drain_timeout_seconds=0.1,
        )
        assert claim.token == 'after-cancel'
        await broker.release_authority('robot-cancel', 'after-cancel')

    asyncio.run(scenario())


def test_blocked_audit_redacts_unverified_tool_and_action(monkeypatch):
    async def scenario() -> None:
        broker = CommandBroker()
        events = []

        async def capture(event_type, **fields):
            events.append({'event_type': event_type, **fields})

        monkeypatch.setattr(audit, 'emit', capture)
        await broker.begin_authority('robot-secret', 'claim')
        await broker.update_authority(
            'robot-secret',
            'claim',
            session_id='safe-session',
            state='active',
        )
        secret = 'fence-token-must-not-be-audited'
        with pytest.raises(TeleopCommandBlocked):
            async with broker.ordinary_command(
                'robot-secret',
                read_only=False,
                source='test',
                tool=secret,
                action=secret,
            ):
                raise AssertionError('blocked call entered')
        serialized = json.dumps(events)
        assert secret not in serialized
        assert events[0]['tool'] == '<unverified>'
        assert events[0]['action'] == '<unverified>'
        await broker.release_authority('robot-secret', 'claim')

    asyncio.run(scenario())


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __await__(self):
        async def resolve():
            return self

        return resolve().__await__()

    async def json(self, content_type=None):
        return deepcopy(self._payload)


class _RecordingSession:
    def __init__(self, records: list[dict], *args, **kwargs):
        self.records = records

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, json, headers, **kwargs):
        self.records.append(deepcopy(json))
        if json['method'] == 'initialize':
            result = {'serverInfo': {'name': 'broker-test-driver'}}
        elif json['method'] == 'tools/call':
            result = {
                'content': [{
                    'type': 'text',
                    'text': '{"state":"observed"}',
                }],
            }
        else:
            raise AssertionError(f'unexpected method: {json["method"]}')
        payload = {'jsonrpc': '2.0', 'id': json['id'], 'result': result}
        return _FakeResponse(payload)


def _session_factory(records: list[dict]):
    def factory(*args, **kwargs):
        return _RecordingSession(records, *args, **kwargs)

    return factory


def _tool_result_session_factory(records: list[dict], tool_result: dict):
    class ScriptedSession(_RecordingSession):
        def post(self, url, *, json, headers, **kwargs):
            self.records.append(deepcopy(json))
            if json['method'] == 'initialize':
                result = {'serverInfo': {'name': 'broker-test-driver'}}
            elif json['method'] == 'tools/call':
                result = deepcopy(tool_result)
            else:
                raise AssertionError(f'unexpected method: {json["method"]}')
            payload = {'jsonrpc': '2.0', 'id': json['id'], 'result': result}
            return _FakeResponse(payload)

    def factory(*args, **kwargs):
        return ScriptedSession(records, *args, **kwargs)

    return factory


def _ordinary_control_tool() -> dict:
    return {
        'name': 'robot_control',
        'type': 'actuator',
        'annotations': {'destructiveHint': True},
        'configSchema': {'type': 'object', 'properties': {}},
        'inputSchema': {
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'string',
                    'enum': ['config', 'info', 'move', 'status'],
                },
            },
            'required': ['action'],
        },
    }


def test_active_authority_blocks_every_core_write_lane_but_allows_diagnostics(
    monkeypatch,
):
    async def scenario() -> None:
        driver_id = 'broker-entry-driver'
        session_id = 'e9b5fd2a-e24e-46da-bf98-fe10dbbff862'
        claim_token = 'broker-entry-claim'
        tool = _ordinary_control_tool()
        full_name = f'mcp__{driver_id}__{tool["name"]}'
        record = {
            'id': driver_id,
            'name': 'Broker Entry Driver',
            'transport': 'http',
            'url': 'http://broker-entry.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }
        config.main['services'] = {'mcp': [record]}
        config.main[f'tool_config:{driver_id}:{tool["name"]}'] = {
            'configured': True,
        }
        mcp_client.registry[driver_id] = {
            'name': record['name'],
            'url': record['url'],
            'online': True,
            'trusted': True,
            'transport': 'http',
            'schemas': {},
            'split_map': {},
            'input_schemas': {full_name: tool['inputSchema']},
            'tool_meta': {
                full_name: {
                    'type': 'actuator',
                    'action_enum': ['config', 'info', 'move', 'status'],
                    'annotations': tool['annotations'],
                    'has_config_schema': True,
                },
            },
        }

        audit_events: list[dict] = []

        async def capture_audit(event_type, **fields):
            audit_events.append({'event_type': event_type, **deepcopy(fields)})
            return audit_events[-1]

        monkeypatch.setattr(audit, 'emit', capture_audit)
        rpc_records: list[dict] = []

        async def fake_jrpc(
            session, url, method, params, *, trusted=False, driver_id='',
        ):
            rpc_records.append({
                'method': method,
                'params': deepcopy(params),
                'trusted': trusted,
            })
            return {
                'content': [{
                    'type': 'text',
                    'text': '{"state":"observed"}',
                }],
            }

        monkeypatch.setattr(mcp_client, '_jrpc', fake_jrpc)
        monkeypatch.setattr(
            mcp_client.aiohttp,
            'ClientSession',
            _session_factory([]),
        )
        api_records: list[dict] = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _session_factory(api_records),
        )

        await coordinator.command_broker.begin_authority(driver_id, claim_token)
        await coordinator.command_broker.update_authority(
            driver_id,
            claim_token,
            session_id=session_id,
            principal_id='operator:alice',
            state='active',
        )
        try:
            llm_result = await mcp_client.call_tool(full_name, {'action': 'move'})
            assert json.loads(llm_result) == {
                'error': {
                    'code': 'teleop_command_blocked',
                    'reason': 'teleop_session_active',
                    'robot_id': driver_id,
                    'session_id': session_id,
                    'state': 'active',
                },
            }
            assert rpc_records == []

            with pytest.raises(fastapi.HTTPException) as blocked_api:
                await mcp_manage.mcp_call_tool(
                    driver_id,
                    mcp_manage.MCPCallRequest(
                        tool=tool['name'],
                        arguments={'action': 'move'},
                    ),
                )
            assert blocked_api.value.status_code == 409
            assert blocked_api.value.detail['code'] == 'teleop_command_blocked'
            assert api_records == []

            deferred = await canvas.save_tool_config(
                driver_id,
                tool['name'],
                {'gain': 0.5},
            )
            deferred_body = json.loads(deferred.body)
            assert deferred.status_code == 202
            assert deferred_body['data'] == {
                'saved': True,
                'applied': False,
                'deferred': True,
                'reason': 'teleop_session_active',
            }
            deferred_instance = await canvas.save_instance_config(
                driver_id,
                tool['name'],
                'canvas-card-1',
                {'gain': 0.75},
            )
            assert deferred_instance.status_code == 202
            assert json.loads(deferred_instance.body)['data']['deferred'] is True
            assert api_records == []

            config.main['canvas_layout'] = {
                'cards': [{
                    'id': 'canvas-card-1',
                    'mcpId': driver_id,
                    'toolName': tool['name'],
                }],
                'connections': [],
                'revision': 7,
            }
            assert (await canvas.claim_edit({'session_id': 'canvas-editor'}))['code'] == 200
            project_start = await config_api.api_start_project(
                config_api.ProjectStartRequest(
                    session_id='canvas-editor',
                    layout_revision=7,
                ),
            )
            project_body = json.loads(project_start.body)
            assert project_start.status_code == 409
            assert project_body['detail']['code'] == 'teleop_command_blocked'
            assert project_body['detail']['session_id'] == session_id
            assert api_records == []

            config_api._set_project_state('running')
            project_stop = await config_api.api_stop_project()
            stop_body = json.loads(project_stop.body)
            assert project_stop.status_code == 409
            assert stop_body['detail']['code'] == 'teleop_command_blocked'
            assert stop_body['detail']['session_id'] == session_id
            assert api_records == []

            await mcp_manage._restore_saved_configs(
                driver_id,
                record['url'],
                [tool],
                trusted=True,
            )
            assert api_records == []

            diagnostic = await mcp_client.call_tool(full_name, {'action': 'info'})
            assert json.loads(diagnostic) == {'state': 'observed'}
            assert len(rpc_records) == 1
            assert rpc_records[0]['params']['arguments'] == {'action': 'info'}

            api_diagnostic = await mcp_manage.mcp_call_tool(
                driver_id,
                mcp_manage.MCPCallRequest(
                    tool=tool['name'],
                    arguments={'action': 'info'},
                ),
            )
            assert api_diagnostic['code'] == 200
            assert [record['method'] for record in api_records] == [
                'initialize', 'tools/call',
            ]
            api_status = await mcp_manage.mcp_call_tool(
                driver_id,
                mcp_manage.MCPCallRequest(
                    tool=tool['name'],
                    arguments={'action': 'status'},
                ),
            )
            assert api_status['code'] == 200
            assert [record['method'] for record in api_records] == [
                'initialize', 'tools/call', 'initialize', 'tools/call',
            ]
            assert [
                record['params']['arguments']['action']
                for record in api_records
                if record['method'] == 'tools/call'
            ] == ['info', 'status']

            blocked_events = [
                event for event in audit_events
                if event['event_type'] == 'teleop.command.blocked'
            ]
            assert {event['source'] for event in blocked_events} == {
                'config_restore', 'mcp_api', 'mcp_client',
            }
            assert all(event['session_id'] == session_id for event in blocked_events)
            assert all(event['tool'] == tool['name'] for event in blocked_events)
            assert {event['action'] for event in blocked_events} == {
                '<unverified>', 'config', 'move',
            }
            scan = json.dumps(blocked_events, sort_keys=True)
            assert 'arguments' not in scan
            assert 'fence' not in scan
        finally:
            await coordinator.command_broker.release_authority(
                driver_id,
                claim_token,
            )
        try:
            resumed = await mcp_client.call_tool(full_name, {'action': 'move'})
            assert json.loads(resumed) == {'state': 'observed'}
            assert len(rpc_records) == 2
        finally:
            mcp_client.registry.pop(driver_id, None)

    asyncio.run(scenario())


def test_owner_bound_adapter_and_actuator_share_one_robot_gate(monkeypatch):
    async def scenario() -> None:
        robot_id = 'g1-lab-a'
        adapter_id = 'teleop-shadow-lab-a'
        session_id = 'a9abdaaf-318f-4a52-849c-f55a77b0c1bd'
        tool = _ordinary_control_tool()
        full_name = f'mcp__{robot_id}__{tool["name"]}'
        config.main['services'] = {'mcp': [{
            'id': adapter_id,
            'name': 'Standalone Teleop Adapter',
            'transport': 'http',
            'category': 'driver',
            'url': 'http://teleop-adapter.invalid/mcp',
            'trust_state': 'trusted',
            'reported_robot_id': robot_id,
            'authority_domain': robot_id,
            'authority_binding_required': True,
            'tools': [{
                'name': 'teleop_session',
                'type': 'actuator',
                'x-teleop': {'robot_id': robot_id},
            }],
        }, {
            'id': robot_id,
            'name': 'G1 Actuator',
            'transport': 'http',
            'category': 'driver',
            'url': 'http://g1.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }]}
        mcp_client.registry[robot_id] = {
            'name': 'G1 Actuator',
            'url': 'http://g1.invalid/mcp',
            'online': True,
            'trusted': True,
            'schemas': {},
            'split_map': {},
            'input_schemas': {full_name: tool['inputSchema']},
            'tool_meta': {
                full_name: {
                    'type': 'actuator',
                    'action_enum': ['config', 'info', 'move', 'status'],
                    'annotations': {},
                },
            },
        }
        calls = []

        async def fake_jrpc(
            session, url, method, params, *, trusted=False, driver_id='',
        ):
            calls.append(deepcopy(params))
            return {
                'content': [{
                    'type': 'text',
                    'text': '{"state":"observed"}',
                }],
            }

        monkeypatch.setattr(mcp_client, '_jrpc', fake_jrpc)
        await coordinator.command_broker.begin_authority(robot_id, 'adapter-claim')
        await coordinator.command_broker.update_authority(
            robot_id,
            'adapter-claim',
            session_id=session_id,
            state='active',
        )
        try:
            blocked = json.loads(await mcp_client.call_tool(
                full_name,
                {'action': 'move'},
            ))
            assert blocked['error']['code'] == 'teleop_command_blocked'
            assert blocked['error']['robot_id'] == robot_id
            assert calls == []

            diagnostic = await mcp_client.call_tool(full_name, {'action': 'status'})
            assert json.loads(diagnostic) == {'state': 'observed'}
            assert len(calls) == 1
        finally:
            await coordinator.command_broker.release_authority(
                robot_id,
                'adapter-claim',
            )
            mcp_client.registry.pop(robot_id, None)

    asyncio.run(scenario())


def test_reserved_and_unknown_tools_fail_before_network_with_stale_registry(
    monkeypatch,
):
    async def scenario() -> None:
        driver_id = 'stale-registry-driver'
        config.main['services'] = {'mcp': [{
            'id': driver_id,
            'name': 'Stale Registry Driver',
            'transport': 'http',
            'url': 'http://stale.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [],
        }]}
        mcp_client.registry[driver_id] = {
            'name': 'Stale Registry Driver',
            'url': 'http://stale.invalid/mcp',
            'online': True,
            'trusted': True,
            'schemas': {},
            'split_map': {},
            'tool_meta': {
                f'mcp__{driver_id}__unknown_tool': {
                    'type': 'actuator',
                    'action_enum': ['move'],
                    'annotations': {},
                },
            },
        }
        network = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _session_factory(network),
        )

        reserved = await mcp_client.call_tool(
            f'mcp__{driver_id}__teleop_session',
            {'action': 'status'},
        )
        assert reserved == 'Teleop tools are reserved for the dedicated teleop API'
        unknown = await mcp_client.call_tool(
            f'mcp__{driver_id}__unknown_tool',
            {'action': 'move'},
        )
        assert unknown == 'MCP tool not found'

        with pytest.raises(fastapi.HTTPException) as reserved_api:
            await mcp_manage.mcp_call_tool(
                driver_id,
                mcp_manage.MCPCallRequest(
                    tool='teleop_session',
                    arguments={'action': 'status'},
                ),
            )
        assert reserved_api.value.status_code == 403
        with pytest.raises(fastapi.HTTPException) as unknown_api:
            await mcp_manage.mcp_call_tool(
                driver_id,
                mcp_manage.MCPCallRequest(
                    tool='unknown_tool',
                    arguments={'action': 'move'},
                ),
            )
        assert unknown_api.value.status_code == 404
        assert network == []
        mcp_client.registry.pop(driver_id, None)

    asyncio.run(scenario())


def test_broken_explicit_authority_binding_fails_closed_without_network(
    monkeypatch,
):
    async def scenario() -> None:
        mcp_id = 'bound-but-broken'
        tool = _ordinary_control_tool()
        full_name = f'mcp__{mcp_id}__{tool["name"]}'
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Broken Binding',
            'transport': 'http',
            'url': 'http://broken.invalid/mcp',
            'trust_state': 'trusted',
            'authority_domain': 'missing-root',
            'authority_binding_required': True,
            'tools': [deepcopy(tool)],
        }]}
        mcp_client.registry[mcp_id] = {
            'name': 'Broken Binding',
            'url': 'http://broken.invalid/mcp',
            'online': True,
            'trusted': True,
            'schemas': {},
            'split_map': {},
            'tool_meta': {
                full_name: {
                    'type': 'actuator',
                    'action_enum': ['move'],
                    'annotations': {},
                },
            },
        }
        records = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _session_factory(records),
        )

        llm_result = json.loads(await mcp_client.call_tool(
            full_name,
            {'action': 'move'},
        ))
        assert llm_result['error']['code'] == 'authority_binding_invalid'
        with pytest.raises(fastapi.HTTPException) as api_error:
            await mcp_manage.mcp_call_tool(
                mcp_id,
                mcp_manage.MCPCallRequest(
                    tool=tool['name'],
                    arguments={'action': 'move'},
                ),
            )
        assert api_error.value.status_code == 409
        assert api_error.value.detail['code'] == 'authority_binding_invalid'
        assert records == []
        mcp_client.registry.pop(mcp_id, None)

    asyncio.run(scenario())


def test_inflight_ping_cannot_restore_an_owner_retargeted_url(monkeypatch):
    async def scenario() -> None:
        mcp_id = 'retargeted-driver'
        old_url = 'http://old.invalid/mcp'
        new_url = 'http://new.invalid/mcp'
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Retargeted Driver',
            'transport': 'http',
            'url': old_url,
            'trust_state': 'trusted',
            'tools': [],
        }]}
        ping_entered = asyncio.Event()
        finish_ping = asyncio.Event()
        restores = []

        async def delayed_ping(url, *, trusted=False, driver_id=''):
            assert url == old_url
            ping_entered.set()
            await finish_ping.wait()
            return {
                'server_name': 'old-driver',
                'tools': [_ordinary_control_tool()],
                'resources': [],
                'device_type': '',
                'topic_out': [],
                'topic_in': [],
            }

        async def record_restore(*args, **kwargs):
            restores.append((args, kwargs))

        monkeypatch.setattr(mcp_manage, '_ping_mcp_http', delayed_ping)
        monkeypatch.setattr(mcp_manage, '_restore_saved_configs', record_restore)

        task = asyncio.create_task(mcp_manage._do_ping(mcp_id))
        await asyncio.wait_for(ping_entered.wait(), timeout=0.5)
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Retargeted Driver',
            'transport': 'http',
            'url': new_url,
            'trust_state': 'trusted',
            'tools': [],
            'resources': [],
            'server_name': '',
            'capability_refresh_required': True,
        }]}
        mcp_client.registry.pop(mcp_id, None)
        finish_ping.set()

        result = await asyncio.wait_for(task, timeout=0.5)
        await asyncio.sleep(0)

        persisted = config.main['services']['mcp'][0]
        assert result['error'] == 'ping target changed; stale result discarded'
        assert persisted['url'] == new_url
        assert persisted['tools'] == []
        assert persisted['capability_refresh_required'] is True
        assert mcp_id not in mcp_client.registry
        assert restores == []

    asyncio.run(scenario())


def test_background_config_restore_rechecks_target_before_network(monkeypatch):
    async def scenario() -> None:
        mcp_id = 'restore-retarget-driver'
        old_url = 'http://restore-old.invalid/mcp'
        new_url = 'http://restore-new.invalid/mcp'
        tool = _ordinary_control_tool()
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Restore Driver',
            'transport': 'http',
            'url': old_url,
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }]}
        config.main[f'tool_config:{mcp_id}:{tool["name"]}'] = {'gain': 0.4}
        session_entered = asyncio.Event()
        continue_restore = asyncio.Event()
        records = []

        class DelayedSession(_RecordingSession):
            async def __aenter__(self):
                session_entered.set()
                await continue_restore.wait()
                return self

        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            lambda *args, **kwargs: DelayedSession(records, *args, **kwargs),
        )

        task = asyncio.create_task(mcp_manage._restore_saved_configs(
            mcp_id,
            old_url,
            [tool],
            trusted=True,
        ))
        await asyncio.wait_for(session_entered.wait(), timeout=0.5)
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Restore Driver',
            'transport': 'http',
            'url': new_url,
            'trust_state': 'trusted',
            'tools': [],
            'capability_refresh_required': True,
        }]}
        continue_restore.set()
        await asyncio.wait_for(task, timeout=0.5)

        assert records == []

    asyncio.run(scenario())


def test_background_config_restore_rejects_same_url_capability_generation(
    monkeypatch,
):
    async def scenario() -> None:
        mcp_id = 'restore-capability-refresh-driver'
        url = 'http://restore-same-url.invalid/mcp'
        tool = _ordinary_control_tool()
        target = {
            'id': mcp_id,
            'name': 'Capability Refresh Driver',
            'transport': 'http',
            'url': url,
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }
        config.main['services'] = {'mcp': [target]}
        config.main[f'tool_config:{mcp_id}:{tool["name"]}'] = {'gain': 0.4}
        mcp_manage._ping_generations[mcp_id] = 1
        target_fingerprint = mcp_manage._ping_target_fingerprint(target)
        session_entered = asyncio.Event()
        continue_restore = asyncio.Event()
        records = []

        class DelayedSession(_RecordingSession):
            async def __aenter__(self):
                session_entered.set()
                await continue_restore.wait()
                return self

        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            lambda *args, **kwargs: DelayedSession(records, *args, **kwargs),
        )

        task = asyncio.create_task(mcp_manage._restore_saved_configs(
            mcp_id,
            url,
            [tool],
            trusted=True,
            target_fingerprint=target_fingerprint,
            ping_generation=1,
        ))
        await asyncio.wait_for(session_entered.wait(), timeout=0.5)

        reserved = deepcopy(tool)
        reserved['x-teleop'] = {
            'protocol': 'motus.teleop.shadow.v1',
            'mode': 'shadow',
        }
        config.main['services'] = {'mcp': [{**target, 'tools': [reserved]}]}
        mcp_manage._ping_generations[mcp_id] = 2
        continue_restore.set()
        await asyncio.wait_for(task, timeout=0.5)

        assert records == []

    asyncio.run(scenario())


def test_running_project_rejects_active_target_delete_and_can_still_stop(
    monkeypatch,
):
    async def scenario() -> None:
        mcp_id = 'active-delete-driver'
        tool = _ordinary_control_tool()
        target = {
            'id': mcp_id,
            'name': 'Active Delete Driver',
            'transport': 'http',
            'url': 'http://active-delete.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }
        card = {'id': 'card-delete', 'mcpId': mcp_id, 'toolName': tool['name']}
        config.main['services'] = {'mcp': [target]}
        config_api._set_project_state('running', cards=[card])
        records = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _session_factory(records),
        )

        response = await mcp_manage.mcp_delete(mcp_id)

        assert response.status_code == 409
        body = json.loads(response.body)
        assert body['data']['code'] == 'project_target_locked'
        assert body['data']['mcp_ids'] == [mcp_id]
        assert config.main['services']['mcp'][0]['id'] == mcp_id
        assert await config_api._do_stop_project() is True
        assert records[-1]['params']['arguments'] == {
            'action': 'stop',
            'instance_id': card['id'],
        }

    asyncio.run(scenario())


def test_running_project_rejects_retarget_with_zero_hidden_config_writes(
    monkeypatch,
):
    async def scenario() -> None:
        mcp_id = 'active-retarget-driver'
        old_url = 'http://active-retarget-old.invalid/mcp'
        tool = _ordinary_control_tool()
        target = {
            'id': mcp_id,
            'name': 'Active Retarget Driver',
            'transport': 'http',
            'url': old_url,
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }
        card = {'id': 'card-retarget', 'mcpId': mcp_id, 'toolName': tool['name']}
        config.main['services'] = {
            'llm': {'url': 'http://old-llm.invalid/v1', 'key': 'old', 'model': 'old'},
            'mcp': [target],
        }
        config.main['client'] = {'llm': [{
            'url': 'http://old-llm.invalid/v1',
            'key': 'old',
            'model': 'old',
        }]}
        config_api._set_project_state('running', cards=[card])
        before_services = config.main['services']
        before_client = config.main['client']
        runtime_client = client_mod.llm
        records = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _session_factory(records),
        )
        request = config_api.ConfigSaveRequest(
            services={'llm': {
                'url': 'http://hidden-new-llm.invalid',
                'key': 'hidden-new-key',
                'model': 'hidden-new-model',
            }},
            mcp_list=[{
                'id': mcp_id,
                'name': target['name'],
                'transport': 'http',
                'url': 'http://active-retarget-new.invalid/mcp',
            }],
        )

        with pytest.raises(fastapi.HTTPException) as error:
            await config_api.config_save(request)

        assert error.value.status_code == 409
        assert error.value.detail['code'] == 'project_target_locked'
        assert config.main['services'] == before_services
        assert config.main['client'] == before_client
        assert client_mod.llm is runtime_client
        assert await config_api._do_stop_project() is True
        assert records[-1]['params']['arguments']['action'] == 'stop'

    asyncio.run(scenario())


def test_running_project_rejects_same_url_capability_removal_and_can_stop(
    monkeypatch,
):
    async def scenario() -> None:
        mcp_id = 'active-capability-driver'
        tool = _ordinary_control_tool()
        target = {
            'id': mcp_id,
            'name': 'Active Capability Driver',
            'transport': 'http',
            'url': 'http://active-capability.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }
        card = {'id': 'card-capability', 'mcpId': mcp_id, 'toolName': tool['name']}
        config.main['services'] = {'mcp': [target]}
        config_api._set_project_state('running', cards=[card])
        mcp_client.registry[mcp_id] = {'online': True}

        async def removed_capabilities(url, *, trusted=False, driver_id=''):
            return {
                'server_name': 'active-capability-driver',
                'tools': [],
                'resources': [],
                'device_type': '',
                'topic_out': [],
                'topic_in': [],
            }

        monkeypatch.setattr(mcp_manage, '_ping_mcp_http', removed_capabilities)
        result = await mcp_manage._do_ping(mcp_id)

        assert result['detail']['code'] == 'project_target_locked'
        assert config.main['services']['mcp'][0]['tools'] == [tool]

        records = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _session_factory(records),
        )
        assert await config_api._do_stop_project() is True
        assert records[-1]['params']['arguments']['action'] == 'stop'

    asyncio.run(scenario())


def test_running_project_locks_authority_root_target(monkeypatch):
    async def scenario() -> None:
        adapter_id = 'active-alias-adapter'
        root_id = 'active-alias-root'
        tool = _ordinary_control_tool()
        adapter = {
            'id': adapter_id,
            'name': 'Active Alias Adapter',
            'transport': 'http',
            'url': 'http://active-alias.invalid/mcp',
            'category': 'driver',
            'trust_state': 'trusted',
            'authority_domain': root_id,
            'tools': [deepcopy(tool)],
        }
        root = {
            'id': root_id,
            'name': 'Active Alias Root',
            'transport': 'http',
            'url': 'http://active-root-old.invalid/mcp',
            'category': 'driver',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }
        card = {'id': 'card-alias', 'mcpId': adapter_id, 'toolName': tool['name']}
        config.main['services'] = {'mcp': [adapter, root]}
        config_api._set_project_state('running', cards=[card])
        request = config_api.ConfigSaveRequest(mcp_list=[{
            'id': adapter_id,
            'name': adapter['name'],
            'transport': 'http',
            'url': adapter['url'],
        }, {
            'id': root_id,
            'name': root['name'],
            'transport': 'http',
            'url': 'http://active-root-new.invalid/mcp',
        }])

        with pytest.raises(fastapi.HTTPException) as error:
            await config_api.config_save(request)

        assert error.value.status_code == 409
        assert error.value.detail['code'] == 'project_target_locked'
        assert error.value.detail['mcp_ids'] == [root_id]

        async def fake_call(mcp_id, call):
            return {'code': 200, 'data': {'state': 'idle'}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        assert await config_api._do_stop_project() is True

    asyncio.run(scenario())


def test_project_transition_rejects_target_delete_before_running_commit(
    monkeypatch,
):
    async def scenario() -> None:
        mcp_id = 'transition-delete-driver'
        tool = _ordinary_control_tool()
        target = {
            'id': mcp_id,
            'name': 'Transition Delete Driver',
            'transport': 'http',
            'url': 'http://transition-delete.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }
        card = {'id': 'card-transition', 'mcpId': mcp_id, 'toolName': tool['name']}
        config.main['services'] = {'mcp': [target]}
        config.main['canvas_layout'] = {'cards': [card], 'connections': []}
        start_entered = asyncio.Event()
        finish_start = asyncio.Event()

        async def delayed_call(driver_id, request):
            action = request.arguments.get('action')
            if action == 'start':
                start_entered.set()
                await finish_start.wait()
                return {'code': 200, 'data': {'state': 'running'}}
            if action == 'info':
                return {'code': 200, 'data': {}}
            return {'code': 200, 'data': {'state': 'idle'}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', delayed_call)
        start_task = asyncio.create_task(config_api._do_start_project())
        await asyncio.wait_for(start_entered.wait(), timeout=0.5)
        assert config_api._project_transition_lock.locked() is True

        response = await mcp_manage.mcp_delete(mcp_id)

        assert response.status_code == 409
        assert json.loads(response.body)['data']['project_state'] == 'transitioning'
        finish_start.set()
        assert await asyncio.wait_for(start_task, timeout=0.5) is True
        assert config.main['services']['mcp'][0]['id'] == mcp_id
        assert await config_api._do_stop_project() is True

    asyncio.run(scenario())


def test_rejected_duplicate_config_save_never_switches_hidden_llm_route():
    async def scenario() -> None:
        config.main['services'] = {
            'llm': {'url': 'http://old.invalid/v1', 'key': 'old', 'model': 'old'},
            'mcp': [],
        }
        config.main['client'] = {'llm': [{
            'url': 'http://old.invalid/v1',
            'key': 'old',
            'model': 'old',
        }]}
        before_services = config.main['services']
        before_client = config.main['client']
        runtime_client = client_mod.llm
        request = config_api.ConfigSaveRequest(
            services={'llm': {
                'url': 'http://new.invalid',
                'key': 'new-secret',
                'model': 'new-model',
            }},
            mcp_list=[
                {'id': 'duplicate', 'url': 'http://one.invalid/mcp'},
                {'id': 'duplicate', 'url': 'http://two.invalid/mcp'},
            ],
        )

        with pytest.raises(fastapi.HTTPException) as error:
            await config_api.config_save(request)

        assert error.value.status_code == 409
        assert config.main['services'] == before_services
        assert config.main['client'] == before_client
        assert client_mod.llm is runtime_client

    asyncio.run(scenario())


def test_config_save_persists_related_keys_atomically_before_runtime_swap(
    monkeypatch,
):
    async def scenario() -> None:
        old_values = {
            'services': {
                'llm': {'url': 'http://old.invalid/v1', 'key': 'old', 'model': 'old'},
                'mcp': [],
            },
            'client': {'llm': [{
                'url': 'http://old.invalid/v1',
                'key': 'old',
                'model': 'old',
            }]},
            'desktop_tools': {'search': {
                'type': 'none',
                'base_url': '',
                'api_key': '',
            }},
            'core': {'configured': False},
        }
        config.main.set_many(old_values)
        runtime_client = client_mod.llm
        real_get_conn = config._get_conn
        fault = {'enabled': True, 'writes': 0}

        class FaultyConnection:
            def __init__(self):
                self.connection = real_get_conn()

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                return self.connection.__exit__(exc_type, exc, tb)

            def execute(self, sql, parameters=()):
                if fault['enabled'] and sql.startswith('INSERT OR REPLACE'):
                    fault['writes'] += 1
                    if fault['writes'] == 3:
                        fault['enabled'] = False
                        raise OSError('simulated third config write failure')
                return self.connection.execute(sql, parameters)

            def commit(self):
                return self.connection.commit()

            def rollback(self):
                return self.connection.rollback()

        monkeypatch.setattr(config, '_get_conn', FaultyConnection)
        request = config_api.ConfigSaveRequest(
            services={
                'llm': {
                    'url': 'http://new.invalid',
                    'key': 'new-secret',
                    'model': 'new-model',
                },
                'search': {
                    'type': 'baidu_search',
                    'base_url': 'http://search.invalid',
                    'api_key': 'search-secret',
                },
            },
            mcp_list=[],
        )

        with pytest.raises(OSError, match='third config write failure'):
            await config_api.config_save(request)

        for key, value in old_values.items():
            assert config.main[key] == value
        assert client_mod.llm is runtime_client

    asyncio.run(scenario())


def test_invalid_config_runtime_candidate_is_zero_write_and_keeps_registry(
    monkeypatch,
):
    async def scenario() -> None:
        mcp_id = 'candidate-retarget-driver'
        old_values = {
            'services': {
                'llm': {'url': 'http://old.invalid/v1', 'key': 'old', 'model': 'old'},
                'mcp': [{
                    'id': mcp_id,
                    'name': 'Candidate Retarget Driver',
                    'transport': 'http',
                    'url': 'http://candidate-old.invalid/mcp',
                    'trust_state': 'trusted',
                    'tools': [deepcopy(_ordinary_control_tool())],
                }],
            },
            'client': {'llm': [{
                'url': 'http://old.invalid/v1',
                'key': 'old',
                'model': 'old',
            }]},
            'desktop_tools': {'search': {'type': 'none'}},
            'core': {'configured': False},
        }
        config.main.set_many(old_values)
        registry_entry = {'online': True, 'url': 'http://candidate-old.invalid/mcp'}
        mcp_client.registry[mcp_id] = deepcopy(registry_entry)
        runtime_client = client_mod.llm
        request = config_api.ConfigSaveRequest(
            services={'llm': {
                'url': '::::',
                'key': 'new-secret',
                'model': 'new-model',
            }},
            mcp_list=[{
                'id': mcp_id,
                'name': 'Candidate Retarget Driver',
                'transport': 'http',
                'url': 'http://candidate-new.invalid/mcp',
            }],
        )

        with pytest.raises(httpx.InvalidURL):
            await config_api.config_save(request)

        for key, value in old_values.items():
            assert config.main[key] == value
        assert client_mod.llm is runtime_client
        assert mcp_client.registry[mcp_id] == registry_entry

    asyncio.run(scenario())


def test_successful_config_commit_swaps_runtime_and_invalidates_target_registry(
    monkeypatch,
):
    async def scenario() -> None:
        mcp_id = 'candidate-success-driver'
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Candidate Success Driver',
            'transport': 'http',
            'url': 'http://candidate-success-old.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(_ordinary_control_tool())],
        }]}
        mcp_client.registry[mcp_id] = {'online': True}
        previous_runtime = client_mod.llm
        monkeypatch.setattr(client_mod, 'llm', previous_runtime)
        request = config_api.ConfigSaveRequest(
            services={'llm': {
                'url': 'http://candidate-llm.invalid',
                'key': 'candidate-secret',
                'model': 'candidate-model',
            }},
            mcp_list=[{
                'id': mcp_id,
                'name': 'Candidate Success Driver',
                'transport': 'http',
                'url': 'http://candidate-success-new.invalid/mcp',
            }],
        )

        response = await config_api.config_save(request)

        assert response['code'] == 200
        candidate_runtime = client_mod.llm
        assert candidate_runtime is not previous_runtime
        assert config.main['client']['llm'][0]['url'] == 'http://candidate-llm.invalid/v1'
        assert config.main['services']['mcp'][0]['tools'] == []
        assert mcp_id not in mcp_client.registry
        await candidate_runtime.aclose()

    asyncio.run(scenario())


def test_agentcore_invalid_runtime_candidate_is_atomic_zero_write():
    async def scenario() -> None:
        old_values = {
            'client': {'llm': [{
                'url': 'http://old-agentcore.invalid/v1',
                'key': 'old',
                'model': 'old',
            }]},
            'event': {
                'llm': {'trigger_interval_ms': 1000},
                'subscribe_topics': [],
            },
            'desktop_tools': {'search': {
                'type': 'none',
                'base_url': '',
                'api_key': '',
            }},
        }
        config.main.set_many(old_values)
        runtime_client = client_mod.llm
        request = mcp_manage.MCPCallRequest(
            tool='decision_core',
            arguments={
                'action': 'config',
                'llm_url': '::::',
                'llm_key': 'new-secret',
                'llm_model': 'new-model',
                'trigger_interval_ms': 25,
                'search_type': 'baidu_search',
                'search_base_url': 'http://new-search.invalid',
                'search_api_key': 'new-search-secret',
            },
        )

        with pytest.raises(httpx.InvalidURL):
            await mcp_manage._handle_agentcore_call(request)

        for key, value in old_values.items():
            assert config.main[key] == value
        assert client_mod.llm is runtime_client

    asyncio.run(scenario())


def test_degraded_persist_failure_latches_residual_targets_until_retry_stop(
    monkeypatch,
):
    async def scenario() -> None:
        tool = _ordinary_control_tool()
        cards = [{
            'id': 'residual-card-a',
            'mcpId': 'residual-driver-a',
            'toolName': tool['name'],
        }, {
            'id': 'residual-card-b',
            'mcpId': 'residual-driver-b',
            'toolName': tool['name'],
        }]
        config.main['services'] = {'mcp': [{
            'id': card['mcpId'],
            'name': card['mcpId'],
            'transport': 'http',
            'url': f'http://{card["mcpId"]}.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        } for card in cards]}
        config.main['canvas_layout'] = {'cards': cards, 'connections': []}
        calls = []
        fail_stops = {'enabled': True}

        async def fake_call(mcp_id, request):
            action = request.arguments['action']
            calls.append((mcp_id, action))
            if action == 'start' and mcp_id == 'residual-driver-b':
                return {'code': 500, 'message': 'start failed'}
            if action == 'info':
                return {'code': 200, 'data': {}}
            if action == 'stop' and fail_stops['enabled']:
                return {'code': 500, 'message': 'stop failed'}
            return {
                'code': 200,
                'data': {'state': 'running' if action == 'start' else 'idle'},
            }

        original_set_state = config_api._set_project_state

        def fail_degraded_commit(state, *, cards=None):
            if state == 'degraded':
                raise OSError('simulated degraded state commit failure')
            return original_set_state(state, cards=cards)

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        monkeypatch.setattr(config_api, '_set_project_state', fail_degraded_commit)

        result = await config_api._do_start_project()

        assert result['status_code'] == 500
        assert result['detail']['rollback_incomplete'] is True
        assert await config_api.get_project_running() == {
            'running': True,
            'state': 'degraded',
            'transitioning': False,
        }
        assert config.main['core'].get('active_project_cards') is None

        delete = await mcp_manage.mcp_delete('residual-driver-a')
        assert delete.status_code == 409
        assert json.loads(delete.body)['data']['project_state'] == 'degraded'
        restart = await config_api._do_start_project()
        assert restart['detail']['code'] == 'project_degraded_requires_stop'

        fail_stops['enabled'] = False
        assert await config_api._do_stop_project() is True
        assert calls.count(('residual-driver-a', 'stop')) == 2
        assert calls.count(('residual-driver-b', 'stop')) == 2
        assert await config_api.get_project_running() == {
            'running': False,
            'state': 'stopped',
            'transitioning': False,
        }
        assert config_api._project_residual_latched is False

    asyncio.run(scenario())


def test_canvas_config_response_confirms_driver_apply_before_success(monkeypatch):
    async def scenario() -> None:
        mcp_id = 'config-apply-driver'
        tool = _ordinary_control_tool()
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Config Apply Driver',
            'transport': 'http',
            'url': 'http://config-apply.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }]}
        records = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _session_factory(records),
        )

        response = await canvas.save_tool_config(
            mcp_id,
            tool['name'],
            {'gain': 0.5},
        )

        assert response == {
            'code': 200,
            'data': {'saved': True, 'applied': True, 'deferred': False},
        }
        assert [record['method'] for record in records] == [
            'initialize', 'tools/call',
        ]
        assert records[-1]['params']['arguments'] == {
            'action': 'config',
            'gain': 0.5,
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'tool_result',
    [
        {
            'isError': True,
            'content': [{'type': 'text', 'text': '{"message":"rejected"}'}],
        },
        {
            'content': [{'type': 'text', 'text': '{"adapter_ok":false}'}],
        },
        {
            'content': [{
                'type': 'text',
                'text': '{"error":"Missing input_topic"}',
            }],
        },
        {
            'content': [{}],
        },
        {
            'content': [{'type': 'text', 'text': 'configured'}],
        },
        {
            'content': [{
                'type': 'text',
                'text': '{"configured":true,"gain":NaN}',
            }],
        },
        {
            'content': [{
                'type': 'image',
                'data': 'aW1hZ2U=',
                'mimeType': 'image/png',
            }],
        },
        {
            'content': [],
        },
        {},
    ],
    ids=[
        'mcp-is-error',
        'adapter-rejected',
        'explicit-error-ack',
        'invalid-content-block',
        'unstructured-text-ack',
        'non-finite-json-ack',
        'image-only-ack',
        'empty-ack',
        'invalid-result',
    ],
)
def test_canvas_config_driver_rejection_is_saved_but_not_reported_applied(
    monkeypatch,
    tool_result,
):
    async def scenario() -> None:
        mcp_id = 'config-reject-driver'
        tool = _ordinary_control_tool()
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Config Reject Driver',
            'transport': 'http',
            'url': 'http://config-reject.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }]}
        records = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _tool_result_session_factory(records, tool_result),
        )

        response = await canvas.save_tool_config(
            mcp_id,
            tool['name'],
            {'gain': 0.5},
        )

        assert response.status_code == 502
        body = json.loads(response.body)
        assert body['data'] == {
            'saved': True,
            'applied': False,
            'deferred': False,
            'reason': 'driver_apply_failed',
        }
        assert config.main[f'tool_config:{mcp_id}:{tool["name"]}'] == {
            'gain': 0.5,
        }
        assert records[-1]['params']['arguments'] == {
            'gain': 0.5,
            'action': 'config',
        }

    asyncio.run(scenario())


def test_generic_mcp_call_accepts_a_valid_non_text_content_block(monkeypatch):
    async def scenario() -> None:
        mcp_id = 'image-result-driver'
        tool = _ordinary_control_tool()
        tool.pop('configSchema')
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Image Result Driver',
            'transport': 'http',
            'url': 'http://image-result.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }]}
        image_block = {
            'type': 'image',
            'data': 'aW1hZ2U=',
            'mimeType': 'image/png',
        }
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _tool_result_session_factory([], {'content': [image_block]}),
        )

        result = await mcp_manage.mcp_call_tool(
            mcp_id,
            mcp_manage.MCPCallRequest(
                tool=tool['name'],
                arguments={'action': 'info'},
            ),
        )

        assert result == {'code': 200, 'data': [image_block]}

    asyncio.run(scenario())


def test_llm_dispatch_requires_config_ack_but_keeps_generic_images(monkeypatch):
    async def scenario() -> None:
        mcp_id = 'llm-config-ack-driver'
        tool = _ordinary_control_tool()
        tool['inputSchema']['properties']['action']['enum'].append('start')
        full_name = f'mcp__{mcp_id}__{tool["name"]}'
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'LLM Config Ack Driver',
            'transport': 'http',
            'url': 'http://llm-config.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }]}
        config.main[f'tool_config:{mcp_id}:{tool["name"]}'] = {'gain': 0.5}
        mcp_client.registry[mcp_id] = {
            'name': 'LLM Config Ack Driver',
            'url': 'http://llm-config.invalid/mcp',
            'online': True,
            'trusted': True,
            'schemas': {},
            'split_map': {},
            'input_schemas': {full_name: tool['inputSchema']},
            'tool_meta': {
                full_name: {
                    'type': 'actuator',
                    'action_enum': tool['inputSchema']['properties']['action']['enum'],
                    'annotations': {},
                    'has_config_schema': True,
                },
            },
        }
        calls = []
        image_block = {
            'type': 'image',
            'data': 'aW1hZ2U=',
            'mimeType': 'image/png',
        }

        async def image_result(session, url, method, params, **kwargs):
            calls.append(deepcopy(params['arguments']))
            return {'content': [deepcopy(image_block)]}

        monkeypatch.setattr(mcp_client, '_jrpc', image_result)

        rejected = await mcp_client.call_tool(full_name, {'action': 'start'})
        assert rejected == f'[{tool["name"]}] Driver 拒绝配置，启动已取消。'
        assert calls == [{'gain': 0.5, 'action': 'config'}]

        observed = await mcp_client.call_tool(full_name, {'action': 'info'})
        assert observed == [{
            'type': 'image_url',
            'image_url': 'data:image/png;base64,aW1hZ2U=',
        }]
        assert calls[-1] == {'action': 'info'}

    asyncio.run(scenario())


def test_generic_result_can_observe_error_state_without_becoming_rpc_failure():
    result = {
        'content': [{
            'type': 'text',
            'text': '{"state":"error","message":"sensor unavailable"}',
        }],
    }

    assert mcp_manage._tool_result_error(result) is None
    assert mcp_manage._tool_result_error(
        result,
        require_structured_ack=True,
    ) == 'sensor unavailable'

    explicit_error = {
        'content': [{
            'type': 'text',
            'text': '{"error":"Missing input_topic"}',
        }],
    }
    assert mcp_manage._tool_result_error(explicit_error) is None
    assert mcp_manage._tool_result_error(
        explicit_error,
        require_structured_ack=True,
    ) == 'Missing input_topic'
    assert config_api._project_tool_ack_error({
        'code': 200,
        'data': explicit_error['content'],
    }) == 'Missing input_topic'


@pytest.mark.parametrize(
    'body',
    [
        {'action': 'move', 'gain': 0.5},
        {'instance_id': 'another-card', 'gain': 0.5},
        ['not', 'an', 'object'],
    ],
    ids=['action', 'instance-id', 'non-object'],
)
def test_canvas_config_rejects_protocol_owned_fields_before_persisting(body):
    async def scenario() -> None:
        key = 'tool_config:config-safe-driver:robot_control'
        with pytest.raises(fastapi.HTTPException) as error:
            await canvas.save_tool_config(
                'config-safe-driver',
                'robot_control',
                body,
            )
        assert error.value.status_code == 422
        assert config.main.get(key, None) is None

    asyncio.run(scenario())


def test_legacy_saved_config_cannot_override_restore_protocol_fields(monkeypatch):
    async def scenario() -> None:
        mcp_id = 'legacy-config-driver'
        tool = _ordinary_control_tool()
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Legacy Config Driver',
            'transport': 'http',
            'url': 'http://legacy-config.invalid/mcp',
            'trust_state': 'trusted',
            'tools': [deepcopy(tool)],
        }]}
        config.main[f'tool_config:{mcp_id}:{tool["name"]}'] = {
            'action': 'move',
            'instance_id': 'victim-card',
            'gain': 0.25,
        }
        records = []
        monkeypatch.setattr(
            mcp_manage.aiohttp,
            'ClientSession',
            _session_factory(records),
        )

        await mcp_manage._restore_saved_configs(
            mcp_id,
            'http://legacy-config.invalid/mcp',
            [tool],
            trusted=True,
        )

        assert records[-1]['params']['arguments'] == {
            'gain': 0.25,
            'action': 'config',
        }

    asyncio.run(scenario())


def test_empty_canvas_start_is_a_visible_conflict():
    async def scenario() -> None:
        assert (await canvas.claim_edit({'session_id': 'canvas-editor'}))['code'] == 200
        response = await config_api.api_start_project(
            config_api.ProjectStartRequest(
                session_id='canvas-editor',
                layout_revision=0,
            ),
        )

        assert response.status_code == 409
        body = json.loads(response.body)
        assert body['detail'] == {
            'code': 'project_empty',
            'reason': 'no_canvas_cards',
        }
        state = await config_api.get_project_running()
        assert state == {
            'running': False,
            'state': 'stopped',
            'transitioning': False,
        }

    asyncio.run(scenario())


def test_degraded_project_must_be_stopped_before_another_start():
    async def scenario() -> None:
        config_api._set_project_state('degraded', cards=[{
            'id': 'card-a',
            'mcpId': 'driver-a',
            'toolName': 'tool-a',
        }])

        result = await config_api._do_start_project()

        assert result == {
            'status_code': 409,
            'detail': {
                'code': 'project_degraded_requires_stop',
                'reason': 'residual_control_must_be_stopped_before_start',
                'project_state': 'degraded',
            },
            'errors': [],
        }

    asyncio.run(scenario())


def test_canvas_revision_cas_rejects_a_stale_visible_layout():
    async def scenario() -> None:
        assert (await canvas.claim_edit({'session_id': 'editor-a'}))['code'] == 200
        initial = await canvas.get_layout()
        assert initial['data']['revision'] == 0

        saved = await canvas.save_layout(canvas.CanvasLayout(
            cards=[{'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'}],
            connections=[],
            session_id='editor-a',
            revision=0,
        ))
        assert saved == {'code': 200, 'data': {'revision': 1}}

        stale = await canvas.save_layout(canvas.CanvasLayout(
            cards=[{'id': 'stale-card', 'mcpId': 'driver-b', 'toolName': 'tool-b'}],
            connections=[],
            session_id='editor-a',
            revision=0,
        ))
        assert stale.status_code == 409
        assert json.loads(stale.body)['detail'] == {
            'code': 'canvas_revision_conflict',
            'expected_revision': 0,
            'current_revision': 1,
        }
        assert config.main['canvas_layout']['cards'][0]['id'] == 'card-a'

    asyncio.run(scenario())


def test_project_start_requires_current_editor_and_visible_revision(monkeypatch):
    async def scenario() -> None:
        config.main['canvas_layout'] = {
            'cards': [{'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'}],
            'connections': [],
            'revision': 4,
        }
        assert (await canvas.claim_edit({'session_id': 'editor-a'}))['code'] == 200
        calls = []

        async def fake_call(mcp_id, request):
            calls.append((mcp_id, request.tool, request.arguments['action']))
            if request.arguments['action'] == 'info':
                return {'code': 200, 'data': {}}
            return {'code': 200, 'data': {'state': 'running'}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)

        wrong_editor = await config_api.api_start_project(
            config_api.ProjectStartRequest(
                session_id='editor-b',
                layout_revision=4,
            ),
        )
        assert wrong_editor.status_code == 409
        assert json.loads(wrong_editor.body)['detail']['code'] == 'project_editor_required'
        assert calls == []

        stale = await config_api.api_start_project(
            config_api.ProjectStartRequest(
                session_id='editor-a',
                layout_revision=3,
            ),
        )
        assert stale.status_code == 409
        assert json.loads(stale.body)['detail'] == {
            'code': 'project_layout_revision_conflict',
            'expected_revision': 3,
            'current_revision': 4,
        }
        assert calls == []

        started = await config_api.api_start_project(
            config_api.ProjectStartRequest(
                session_id='editor-a',
                layout_revision=4,
            ),
        )
        assert started == {'ok': True}
        assert calls == [
            ('driver-a', 'tool-a', 'start'),
            ('driver-a', 'tool-a', 'info'),
        ]

    asyncio.run(scenario())


def test_project_running_exposes_an_inflight_start_transition(monkeypatch):
    async def scenario() -> None:
        config.main['canvas_layout'] = {
            'cards': [{'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'}],
            'connections': [],
            'revision': 2,
        }
        assert (await canvas.claim_edit({'session_id': 'editor-a'}))['code'] == 200
        start_entered = asyncio.Event()
        finish_start = asyncio.Event()

        async def fake_call(mcp_id, request):
            if request.arguments['action'] == 'start':
                start_entered.set()
                await finish_start.wait()
                return {'code': 200, 'data': {'state': 'running'}}
            return {'code': 200, 'data': {}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        task = asyncio.create_task(config_api.api_start_project(
            config_api.ProjectStartRequest(
                session_id='editor-a',
                layout_revision=2,
            ),
        ))
        await asyncio.wait_for(start_entered.wait(), timeout=0.5)

        assert await config_api.get_project_running() == {
            'running': False,
            'state': 'stopped',
            'transitioning': True,
        }
        finish_start.set()
        assert await asyncio.wait_for(task, timeout=0.5) == {'ok': True}
        assert await config_api.get_project_running() == {
            'running': True,
            'state': 'running',
            'transitioning': False,
        }

    asyncio.run(scenario())


def test_project_start_reports_degraded_when_rollback_is_incomplete(monkeypatch):
    async def scenario() -> None:
        config.main['canvas_layout'] = {
            'cards': [
                {'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'},
                {'id': 'card-b', 'mcpId': 'driver-b', 'toolName': 'tool-b'},
            ],
            'connections': [],
        }
        calls = []

        async def fake_call(mcp_id, request):
            action = request.arguments['action']
            calls.append((mcp_id, request.tool, action))
            if action == 'info':
                return {'code': 200, 'data': {}}
            if action == 'start' and request.tool == 'tool-b':
                return {'code': 500, 'message': 'start rejected', 'data': None}
            if action == 'stop' and request.tool == 'tool-a':
                return {'code': 500, 'message': 'stop rejected', 'data': None}
            return {'code': 200, 'data': {'state': 'running'}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        result = await config_api._do_start_project()

        assert result['status_code'] == 500
        assert result['detail']['rollback_incomplete'] is True
        assert result['detail']['project_state'] == 'degraded'
        assert result['detail']['started_card_ids'] == ['card-a']
        assert result['detail']['attempted_card_ids'] == ['card-a', 'card-b']
        assert ('driver-a', 'tool-a', 'stop') in calls
        assert ('driver-b', 'tool-b', 'stop') in calls
        assert await config_api.get_project_running() == {
            'running': True,
            'state': 'degraded',
            'transitioning': False,
        }

    asyncio.run(scenario())


def test_project_start_never_marks_running_without_structured_driver_ack(monkeypatch):
    async def scenario() -> None:
        config.main['canvas_layout'] = {
            'cards': [{'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'}],
            'connections': [],
        }
        calls = []

        async def fake_call(mcp_id, request):
            action = request.arguments['action']
            calls.append(action)
            if action == 'start':
                return {
                    'code': 200,
                    'data': [{
                        'type': 'image',
                        'data': 'aW1hZ2U=',
                        'mimeType': 'image/png',
                    }],
                }
            return {'code': 200, 'data': {'state': 'idle'}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        result = await config_api._do_start_project()

        assert result['status_code'] == 500
        assert result['detail']['code'] == 'project_start_failed'
        assert result['detail']['rollback_incomplete'] is False
        assert calls == ['start', 'stop']
        assert await config_api.get_project_running() == {
            'running': False,
            'state': 'stopped',
            'transitioning': False,
        }

    asyncio.run(scenario())


def test_project_running_state_commit_failure_stops_started_drivers(monkeypatch):
    async def scenario() -> None:
        config.main['canvas_layout'] = {
            'cards': [{'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'}],
            'connections': [],
        }
        calls = []

        async def fake_call(mcp_id, request):
            action = request.arguments['action']
            calls.append(action)
            if action == 'info':
                return {'code': 200, 'data': {}}
            return {'code': 200, 'data': {'state': 'running' if action == 'start' else 'idle'}}

        original_set_state = config_api._set_project_state

        def fail_running_commit(state, *, cards=None):
            if state == 'running':
                raise OSError('simulated SQLite commit failure')
            return original_set_state(state, cards=cards)

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        monkeypatch.setattr(config_api, '_set_project_state', fail_running_commit)

        result = await config_api._do_start_project()

        assert result['status_code'] == 500
        assert result['detail']['code'] == 'project_state_commit_failed'
        assert result['detail']['rollback_incomplete'] is False
        assert calls == ['start', 'info', 'stop']
        assert config.main['core']['project_running'] is False

    asyncio.run(scenario())


def test_agentcore_decision_card_has_structured_project_lifecycle_ack():
    async def scenario() -> None:
        config.main['canvas_layout'] = {
            'cards': [{
                'id': 'decision-card',
                'mcpId': 'agentcore',
                'toolName': 'decision_core',
            }],
            'connections': [],
        }

        assert await config_api._do_start_project() is True
        assert config.main['core']['project_state'] == 'running'
        assert await config_api._do_stop_project() is True
        assert config.main['core']['project_state'] == 'stopped'

    asyncio.run(scenario())


def test_cancelled_project_start_finishes_rollback_despite_recancellation(monkeypatch):
    async def scenario() -> None:
        config.main['canvas_layout'] = {
            'cards': [
                {'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'},
                {'id': 'card-b', 'mcpId': 'driver-b', 'toolName': 'tool-b'},
            ],
            'connections': [],
        }
        calls = []
        second_start = asyncio.Event()
        rollback_started = asyncio.Event()
        finish_rollback = asyncio.Event()

        async def fake_call(mcp_id, request):
            action = request.arguments['action']
            calls.append((request.tool, action))
            if action == 'info':
                return {'code': 200, 'data': {}}
            if action == 'start' and request.tool == 'tool-b':
                second_start.set()
                await asyncio.Event().wait()
            if action == 'stop':
                rollback_started.set()
                await finish_rollback.wait()
                return {'code': 200, 'data': {'state': 'idle'}}
            return {'code': 200, 'data': {'state': 'running'}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        task = asyncio.create_task(config_api._do_start_project())
        await asyncio.wait_for(second_start.wait(), timeout=0.5)
        task.cancel()
        await asyncio.wait_for(rollback_started.wait(), timeout=0.5)
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        finish_rollback.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert ('tool-a', 'stop') in calls
        assert ('tool-b', 'stop') in calls
        assert await config_api.get_project_running() == {
            'running': False,
            'state': 'stopped',
            'transitioning': False,
        }

    asyncio.run(scenario())


def test_cancelled_project_stop_finishes_all_stops_before_propagating(monkeypatch):
    async def scenario() -> None:
        config.main['canvas_layout'] = {
            'cards': [
                {'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'},
                {'id': 'card-b', 'mcpId': 'driver-b', 'toolName': 'tool-b'},
                {'id': 'card-c', 'mcpId': 'driver-c', 'toolName': 'tool-c'},
            ],
            'connections': [],
        }
        config_api._set_project_state('running')
        calls = []
        second_stop = asyncio.Event()
        finish_second_stop = asyncio.Event()

        async def fake_call(mcp_id, request):
            calls.append((request.tool, request.arguments['action']))
            if request.tool == 'tool-b':
                second_stop.set()
                await finish_second_stop.wait()
            return {'code': 200, 'data': {'state': 'idle'}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        task = asyncio.create_task(config_api._do_stop_project())
        await asyncio.wait_for(second_stop.wait(), timeout=0.5)
        task.cancel()
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        finish_second_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert calls == [
            ('tool-a', 'stop'),
            ('tool-b', 'stop'),
            ('tool-c', 'stop'),
        ]
        assert await config_api.get_project_running() == {
            'running': False,
            'state': 'stopped',
            'transitioning': False,
        }

    asyncio.run(scenario())


def test_project_stop_uses_active_snapshot_even_if_canvas_layout_changes(monkeypatch):
    async def scenario() -> None:
        card = {'id': 'card-a', 'mcpId': 'driver-a', 'toolName': 'tool-a'}
        config.main['canvas_layout'] = {
            'cards': [deepcopy(card)],
            'connections': [],
        }
        calls = []

        async def fake_call(mcp_id, request):
            calls.append((request.tool, request.arguments['action']))
            if request.arguments['action'] == 'info':
                return {'code': 200, 'data': {}}
            return {'code': 200, 'data': {'state': 'running'}}

        monkeypatch.setattr(mcp_manage, 'mcp_call_tool', fake_call)
        assert await config_api._do_start_project() is True
        assert config.main['core']['active_project_cards'] == [card]

        rejected = await canvas.save_layout(canvas.CanvasLayout(
            cards=[],
            connections=[],
        ))
        assert rejected.status_code == 409
        assert json.loads(rejected.body)['detail']['code'] == 'project_topology_locked'

        # Even a stale/external write cannot change the target of the active
        # stop transaction because it uses the independent runtime snapshot.
        config.main['canvas_layout'] = {'cards': [], 'connections': []}
        assert await config_api._do_stop_project() is True
        assert ('tool-a', 'stop') in calls
        assert 'active_project_cards' not in config.main['core']
        assert await config_api.get_project_running() == {
            'running': False,
            'state': 'stopped',
            'transitioning': False,
        }

    asyncio.run(scenario())
