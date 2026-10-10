"""QR data frames -> deduplicated Activity, without agent-loop or UI consumers."""
import asyncio
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from qr_activity import QrActivity

TOPIC = '/vision/qr_scan/result'


def frame(session='s1', sequence=1, text='U1-001', status='detected'):
    return json.dumps({'status': status, 'session_id': session,
                       'new_events': [{'sequence': sequence, 'text': text}]}).encode()


def test_success_duplicate_new_event_and_session():
    activity = QrActivity()
    assert activity.consume(TOPIC, frame(), 'data/json')[0]['payload']['text'] == 'U1-001'
    assert activity.consume(TOPIC, frame(), 'data/json') == []
    assert activity.consume(TOPIC, frame(sequence=2), 'data/json')
    assert activity.consume(TOPIC, frame(session='s2'), 'data/json')
    assert activity.consume('/other/qr_scan/result', frame(), 'data/json')


@pytest.mark.parametrize('data', [b'invalid', b'[]', b'null', b'\xff', b'x'*1_000_001,
    frame(status='stale'), frame(status='no_qr'), frame(sequence=True),
    frame(sequence=0), frame(session=''), frame(text=''),
    b'{"status":"detected","session_id":"s","new_events":"wrong"}',
    b'{"status":"detected","session_id":"s","new_events":[null]}'])
def test_invalid_and_non_success_do_not_log(data):
    assert QrActivity().consume(TOPIC, data, 'data/json') == []


def test_continuous_codes_without_new_events_do_not_log():
    data = json.dumps({'status': 'detected', 'session_id': 's1',
                       'codes': [{'text': 'A'}], 'new_events': []}).encode()
    assert QrActivity().consume(TOPIC, data, 'data/json') == []


def test_topic_format_and_limits():
    activity = QrActivity(max_seen=2)
    assert activity.consume('/other/result', frame(), 'data/json') == []
    assert activity.consume(TOPIC, frame(), 'image/jpeg') == []
    for i in range(1, 5):
        activity.consume(TOPIC, frame(sequence=i), 'data/json')
    assert len(activity._seen) == 2
    event = activity.consume(TOPIC, frame(sequence=5, text='字'*1000), 'data/json')[0]
    assert len(event['payload']['text']) == 512
    assert event['payload']['text_truncated'] is True


def test_primary_subscription_forwards_once_without_viewers(monkeypatch):
    from api import inspection, motus_stream
    activity = QrActivity()
    monkeypatch.setattr(inspection, '_qr_activity', activity)
    monkeypatch.setattr(inspection, '_topic_queues', {})
    monkeypatch.setattr(inspection, '_last_frame', {})
    monkeypatch.setattr(inspection, '_topic_registry', {TOPIC: {'mcp_id': 'u1'}})
    queue = asyncio.Queue()
    monkeypatch.setattr(motus_stream, '_clients', {queue})

    async def check():
        callback = inspection._push_factory(TOPIC)
        await callback(frame(), 'data/json')
        # Rebuilding the subscription should not reset de-duplication.
        await inspection._push_factory(TOPIC)(frame(), 'data/json')
        assert queue.qsize() == 1
        event = json.loads(queue.get_nowait())
        assert event['type'] == 'qr_detected'
        assert event['mcp_id'] == 'u1'
        assert event['payload']['text'] == 'U1-001'
        assert event['ts'] > 0
        assert inspection._last_frame[TOPIC] == frame()
    asyncio.run(check())
