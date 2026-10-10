"""Turn QR result frames into bounded, deduplicated Activity events.

This is display-only: QR content never becomes a user command or agent trigger.
"""
from collections import OrderedDict
import json


class QrActivity:
    def __init__(self, max_seen=1024):
        self._seen = OrderedDict()
        self.max_seen = max_seen

    def consume(self, topic, data, fmt):
        if fmt != 'data/json' or not topic.endswith('/qr_scan/result'):
            return []
        if len(data) > 1_000_000:
            return []
        try:
            result = json.loads(data)
        except (ValueError, TypeError, UnicodeError):
            return []
        if not isinstance(result, dict) or result.get('status') != 'detected':
            return []
        session = result.get('session_id')
        events = result.get('new_events')
        if not isinstance(session, str) or not 0 < len(session) <= 128:
            return []
        if not isinstance(events, list):
            return []
        output = []
        for event in events[:32]:
            if not isinstance(event, dict):
                continue
            sequence, text = event.get('sequence'), event.get('text')
            if type(sequence) is not int or sequence < 1:
                continue
            if not isinstance(text, str) or not text:
                continue
            key = (topic, session, sequence)
            if key in self._seen:
                self._seen.move_to_end(key)
                continue
            self._seen[key] = None
            while len(self._seen) > self.max_seen:
                self._seen.popitem(last=False)
            output.append({'type': 'qr_detected', 'payload': {
                'topic': topic, 'session_id': session, 'sequence': sequence,
                'text': text[:512], 'text_truncated': len(text) > 512,
            }})
        return output
