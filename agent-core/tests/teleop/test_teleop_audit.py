from __future__ import annotations

import asyncio
import json
import math
import threading

import config
from api import motus_stream
from teleop import audit


def _assert_no_secret_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = ''.join(character for character in key.lower() if character.isalnum())
            assert not any(
                secret in normalized
                for secret in (
                    'credential',
                    'fence',
                    'password',
                    'privatekey',
                    'secret',
                    'token',
                )
            )
            _assert_no_secret_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_secret_keys(item)


def test_audit_recursively_redacts_nested_secrets_and_bounds_payloads():
    async def scenario():
        cyclic: dict = {'safe': 'cycle-survives'}
        cyclic['self'] = cyclic
        details = {
            'state': 'active',
            'fence': 'top-level-fence',
            'nested': {
                'access_token': 'nested-token',
                'private-key': 'nested-private-key',
                'safe': [
                    {'password': 'nested-password', 'value': math.nan},
                    {'credentialId': 'credential', 'value': math.inf},
                ],
            },
            'long': 'x' * 2_000,
            'many': list(range(100)),
            'cycle': cyclic,
        }
        event = await audit.emit(
            'session.activated',
            robot_id='robot-1',
            details=details,
        )
        return event

    event = asyncio.run(scenario())
    _assert_no_secret_keys(event['details'])
    assert event['details']['nested']['safe'] == [
        {'value': None},
        {'value': None},
    ]
    assert len(event['details']['long']) == audit.MAX_STRING_LENGTH
    assert len(event['details']['many']) == audit.MAX_COLLECTION_ITEMS
    assert event['details']['cycle']['self'] is None
    assert json.dumps(event['details'], allow_nan=False)

    persisted = audit.list_events()
    assert len(persisted) == 1
    assert persisted[0]['details'] == event['details']
    _assert_no_secret_keys(persisted[0]['details'])


def test_audit_read_resanitizes_legacy_rows_and_tolerates_malformed_json():
    with config._get_conn() as conn:
        audit._ensure_table(conn)
        base = (
            1.0,
            'legacy',
            '',
            'robot-1',
            '',
            '',
            '',
            '',
            '',
            '',
        )
        conn.execute(
            '''INSERT INTO teleop_audit
               (id, created_at, event_type, session_id, robot_id, principal_id,
                source, decision, reason, tool, action, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            ('legacy-safe', *base, '{"nested":{"fence":"old","ok":1}}'),
        )
        conn.execute(
            '''INSERT INTO teleop_audit
               (id, created_at, event_type, session_id, robot_id, principal_id,
                source, decision, reason, tool, action, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            ('legacy-broken', *base, '{not-json'),
        )
        conn.commit()

    events = {event['id']: event for event in audit.list_events()}
    assert events['legacy-safe']['details'] == {'nested': {'ok': 1}}
    assert events['legacy-broken']['details'] == {}


def test_audit_write_and_broadcast_failures_never_break_control_flow(monkeypatch):
    def fail_connection():
        raise OSError('disk unavailable')

    async def fail_broadcast(_event):
        raise RuntimeError('no activity clients')

    monkeypatch.setattr(config, '_get_conn', fail_connection)
    monkeypatch.setattr(motus_stream, 'push_event', fail_broadcast)

    event = asyncio.run(audit.emit(
        'session.released',
        robot_id='robot-1',
        details={'safe': True, 'fence_token': 'must-not-leak'},
    ))
    assert event['details'] == {'safe': True}
    assert audit.list_events() == []


def test_audit_broadcast_receives_only_the_sanitized_event(monkeypatch):
    delivered = []

    async def capture(event):
        delivered.append(event)

    monkeypatch.setattr(motus_stream, 'push_event', capture)
    event = asyncio.run(audit.emit(
        'session.prepared',
        robot_id='robot-1',
        details={
            'nested': {
                'fence': 'fence-value',
                'api_secret': 'secret-value',
                'safe': 'visible',
            },
        },
    ))

    assert delivered == [{
        'type': 'teleop_activity',
        'mcp_id': 'robot-1',
        'payload': event,
    }]
    assert delivered[0]['payload']['details'] == {
        'nested': {'safe': 'visible'},
    }


def test_slow_audit_storage_never_blocks_the_control_fast_path(monkeypatch):
    entered = threading.Event()
    allow_persist = threading.Event()

    def slow_persist(_event):
        entered.set()
        allow_persist.wait(timeout=1.0)

    monkeypatch.setattr(audit, '_persist_event', slow_persist)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = loop.time()
        event = await audit.emit('session.nonblocking', robot_id='robot-1')
        elapsed = loop.time() - started
        worker_started = await asyncio.to_thread(entered.wait, 0.2)
        allow_persist.set()
        await audit.flush()
        return event, elapsed, worker_started

    try:
        event, elapsed, worker_started = asyncio.run(scenario())
    finally:
        allow_persist.set()

    assert event['event_type'] == 'session.nonblocking'
    assert worker_started is True
    assert elapsed < 0.1


def test_audit_storage_retention_is_bounded(monkeypatch):
    monkeypatch.setattr(audit, 'MAX_STORED_EVENTS', 3)

    async def scenario():
        for index in range(5):
            await audit.emit(f'session.event.{index}', robot_id='robot-1')
        await audit.flush()

    asyncio.run(scenario())

    events = audit.list_events(limit=10)
    assert len(events) == 3
    assert {event['event_type'] for event in events} == {
        'session.event.2',
        'session.event.3',
        'session.event.4',
    }
