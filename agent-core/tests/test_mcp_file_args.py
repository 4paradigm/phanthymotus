"""
tests/test_mcp_file_args.py — `format: file` arguments are transferred, not guessed.

The browser has had this since `format: file` existed: the canvas renders a
picker and uploads through /api/mcp/<id>/file/upload. The LLM had no equivalent,
so it could only guess a path — and in production it guessed wrong twice, passing
`/work/dai_wenyuan_1.jpeg` for a file it had just downloaded to `/tmp`, then
falling back to a 43 800-character base64 that did not survive its own context.

The fix is not in the face card: it is in `mcp_client.call_tool`, the one point
every MCP call passes through, so any tool declaring `format: file` gets it —
including ones written later.

Run: python3 -m pytest agent-core/tests/test_mcp_file_args.py -q
"""
import asyncio
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# No module stubs here. An earlier revision installed fakes for aiohttp,
# jsonschema, config and auth into sys.modules — which leaked into the shared
# interpreter and broke tests/test_mcp_call_errors.py, whose own imports then got
# the fakes. These tests exercise _transfer_file_args, which touches only os.path
# and the injected _push_file, so the real modules import fine.
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402

FILE_SCHEMA = {
    'type': 'object',
    'properties': {
        'image_path': {'type': 'string', 'format': 'file', 'uploadTo': 'mcp'},
        'name': {'type': 'string'},
    },
}
NO_FILE_SCHEMA = {
    'type': 'object',
    'properties': {'text': {'type': 'string'}},
}


@pytest.fixture
def pushed(monkeypatch):
    """Record what would have been uploaded, without a network."""
    calls = []

    async def fake_push(mcp_id, url, local_path):
        calls.append({'mcp_id': mcp_id, 'url': url, 'local': local_path})
        return f'/models/uploads/2026-09-08/{os.path.basename(local_path)}'

    monkeypatch.setattr(mcp_client, '_push_file', fake_push)
    return calls


def _transfer(schema, args, pushed_url='http://localhost:15720/mcp'):
    """Sync wrapper: pytest-asyncio is not installed, and the suite's convention
    is asyncio.run inside a plain test."""
    return asyncio.run(
        mcp_client._transfer_file_args('mcp-p', pushed_url, schema, args))


# ── the transfer ──────────────────────────────────────────────────────────────

def test_a_local_file_becomes_the_targets_path(pushed, tmp_path):
    photo = tmp_path / 'dai_wenyuan.jpeg'
    photo.write_bytes(b'\xff\xd8jpeg')

    args, error = _transfer(FILE_SCHEMA,
                                  {'image_path': str(photo), 'name': '戴文渊'})

    assert error is None
    assert args['image_path'] == '/models/uploads/2026-09-08/dai_wenyuan.jpeg'
    assert args['name'] == '戴文渊', 'other arguments must pass through untouched'
    assert len(pushed) == 1
    assert pushed[0]['local'] == str(photo)


def test_a_tool_without_file_fields_is_not_touched(pushed, tmp_path):
    args, error = _transfer(NO_FILE_SCHEMA, {'text': str(tmp_path)})
    assert error is None and args == {'text': str(tmp_path)}
    assert pushed == [], 'must not upload for a tool that takes no file'


def test_a_path_that_is_not_local_is_left_for_the_tool(pushed):
    """Either it is already the target's path — the canvas picker's output, or a
    second call reusing an earlier result — or the model invented it. The tool
    knows its own filesystem; re-sending would fail here for the wrong reason."""
    args, error = _transfer(
        FILE_SCHEMA, {'image_path': '/models/uploads/2026-09-08/already.jpg'})
    assert error is None
    assert args['image_path'] == '/models/uploads/2026-09-08/already.jpg'
    assert pushed == []


def test_a_missing_file_field_is_not_invented(pushed):
    args, error = _transfer(FILE_SCHEMA, {'name': 'Alice'})
    assert error is None and 'image_path' not in args
    assert pushed == []


def test_a_non_string_value_is_ignored(pushed):
    args, error = _transfer(FILE_SCHEMA, {'image_path': 12345})
    assert error is None and args['image_path'] == 12345
    assert pushed == []


def test_every_file_field_is_transferred(pushed, tmp_path):
    """A tool may take more than one."""
    first, second = tmp_path / 'a.jpg', tmp_path / 'b.jpg'
    first.write_bytes(b'a')
    second.write_bytes(b'b')
    schema = {
        'type': 'object',
        'properties': {
            'front': {'type': 'string', 'format': 'file'},
            'side': {'type': 'string', 'format': 'file'},
        },
    }
    args, error = _transfer(schema,
                                  {'front': str(first), 'side': str(second)})
    assert error is None
    assert args['front'].endswith('/a.jpg') and args['side'].endswith('/b.jpg')
    assert len(pushed) == 2


# ── failure is reported, never passed through ─────────────────────────────────

def test_an_upload_failure_stops_the_call(monkeypatch, tmp_path):
    """It must not fall through: the tool would answer "cannot read <path>" and
    the model would retry with another path it invented — the exact loop this
    code exists to break."""
    async def boom(mcp_id, url, local_path):
        raise RuntimeError('Connection refused')

    monkeypatch.setattr(mcp_client, '_push_file', boom)
    photo = tmp_path / 'x.jpg'
    photo.write_bytes(b'x')

    args, error = _transfer(FILE_SCHEMA, {'image_path': str(photo)})
    assert error is not None
    assert 'could not send' in error
    assert 'Connection refused' in error
    # And it says what to check, because the usual cause is a stale image.
    assert '/file/upload' in error
    assert args['image_path'] == str(photo), 'unchanged when the transfer failed'


# ── it reaches every card, present and future ─────────────────────────────────

def test_format_file_survives_the_x_action_params_split():
    """The split rebuilds each action's `properties` from the parent schema. If
    it dropped `format`, the interceptor would never see a file field on any
    split tool — which is every action-based card."""
    tool = {
        'name': 'face_recognition',
        'description': 'Face recognition',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'action': {'type': 'string', 'enum': ['register_by_photo', 'stop']},
                'image_path': {'type': 'string', 'format': 'file', 'uploadTo': 'mcp'},
                'name': {'type': 'string'},
            },
            'x-action-params': {
                'register_by_photo': {'params': ['image_path', 'name'],
                                      'description': 'reg'},
                'stop': {'params': [], 'description': 'stop'},
            },
        },
    }
    schemas = mcp_client._to_openai_schema('mcp-x', tool)
    by_action = {s['name'].split('__')[-1]: s for s in schemas}

    reg = by_action['register_by_photo']['parameters']['properties']
    assert reg['image_path']['format'] == 'file'
    assert by_action['stop']['parameters']['properties'] == {}


def test_the_transfer_is_generic_not_face_specific(pushed, tmp_path):
    """Any tool declaring `format: file` is covered, including ones written
    later — that is the point of doing this in call_tool."""
    clip = tmp_path / 'greeting.wav'
    clip.write_bytes(b'RIFF....WAVE')
    schema = {
        'type': 'object',
        'properties': {'audio_file': {'type': 'string', 'format': 'file'}},
    }
    args, error = _transfer(schema, {'audio_file': str(clip)})
    assert error is None
    assert args['audio_file'].endswith('/greeting.wav')
