from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import tempfile

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


# config seeds SQLite at import time.  Point that bootstrap write at an isolated
# temporary location before importing Agent Core modules.
_AGENT_CORE = Path(__file__).resolve().parents[2]
_SRC = _AGENT_CORE / 'src'
sys.path.insert(0, str(_SRC))
_BOOTSTRAP = Path(tempfile.mkdtemp(prefix='motus-core-c1-bootstrap-'))
os.environ['DB_PATH'] = str(_BOOTSTRAP / 'data.db')

import auth  # noqa: E402
import config  # noqa: E402
import mcp_client  # noqa: E402
from api import canvas, config as config_api, inspection, mcp_manage, motus_stream, teleop  # noqa: E402
from teleop import authority_guard  # noqa: E402


HUMAN_AUTH = {
    'ACCESS_TOKEN': 'owner-token',
    'MOTUS_OPERATOR_TOKENS': '{"alice":"operator-token"}',
    'MOTUS_VIEWER_TOKENS': '{"auditor":"viewer-token"}',
    'MOTUS_DRIVER_TOKEN': 'test-only-legacy-driver-fallback',
    'MOTUS_TELEOP_TICKET_SECRET': 'test-only-ticket-secret-000000000',
}

TEST_CAPTURE_CA_CERTIFICATE_PEM = (
    '-----BEGIN CERTIFICATE-----\n'
    'MAMCAQE=\n'
    '-----END CERTIFICATE-----\n'
)


@pytest.fixture(autouse=True)
def isolated_core_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path / 'data.db'))
    config._seed_defaults()
    config.main['services'] = {'mcp': []}
    config.main['canvas_layout'] = {
        'cards': [], 'connections': [], 'execConnections': [], 'transform': {},
    }
    config.main['event'] = {'llm': {}, 'subscribe_topics': []}

    auth.init(HUMAN_AUTH)
    mcp_client.registry.clear()
    mcp_manage._last_tool_names.clear()
    mcp_manage._ping_generations.clear()
    target_mutation_lock = asyncio.Lock()
    monkeypatch.setattr(authority_guard, 'target_mutation_lock', target_mutation_lock)
    monkeypatch.setattr(mcp_manage, '_mcp_write_lock', target_mutation_lock)
    canvas._editor_session = None
    canvas._editor_last_seen = 0.0
    config_api._clear_project_residual()
    motus_stream._clients.clear()
    yield
    motus_stream._clients.clear()
    mcp_client.registry.clear()
    config_api._clear_project_residual()
    auth.init({})


@pytest.fixture
def client():
    api_app = FastAPI()
    api_app.include_router(mcp_manage.router)
    api_app.include_router(canvas.router)
    api_app.include_router(teleop.router)
    api_app.state.teleop_capture_ca_certificate_pem = (
        TEST_CAPTURE_CA_CERTIFICATE_PEM
    )

    @api_app.get('/auth/verify')
    async def auth_verify(request: Request):
        if not auth.is_enabled():
            return {'valid': True, 'auth_required': False, 'principal': None}
        principal = auth.authenticate(auth._extract_token(request))
        if principal:
            return {
                'valid': True,
                'auth_required': True,
                'principal': principal.as_dict(),
            }
        return JSONResponse(
            status_code=401,
            content={'valid': False, 'auth_required': True, 'principal': None},
        )

    app = FastAPI()
    app.middleware('http')(auth.auth_middleware)
    app.mount('/api', api_app)
    app.include_router(motus_stream.router)
    app.include_router(inspection.ws_router)
    app.include_router(teleop.ws_router)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    client_header = {
        'X-Motus-Teleop-Client': '7dbabfca-15c1-43ca-b600-75e7682c21d0',
    }
    return {
        'owner': {'Authorization': 'Bearer owner-token', **client_header},
        'operator': {'Authorization': 'Bearer operator-token', **client_header},
        'viewer': {'Authorization': 'Bearer viewer-token', **client_header},
    }


@pytest.fixture
def shadow_session_tool():
    """Producer-shaped fixture mirrored from Driver teleop_shadow.tool_definitions."""
    identity = {
        'boot_id': {'type': 'string', 'format': 'uuid'},
        'session_id': {'type': 'string', 'format': 'uuid'},
        'epoch': {'type': 'integer', 'minimum': 1},
        'fence': {'type': 'string', 'minLength': 24},
    }
    actions = {
        'start': {'params': [], 'description': 'Passive lifecycle readiness check'},
        'stop': {'params': [], 'description': 'Stop lifecycle and release Shadow'},
        'prepare_shadow': {
            'params': ['session_id', 'epoch', 'fence'],
            'description': 'Install a new Core-issued epoch/fence',
        },
        'heartbeat': {'params': list(identity), 'description': 'Renew lease'},
        'pause': {'params': list(identity), 'description': 'Pause diagnostics'},
        'release': {'params': list(identity), 'description': 'Release session'},
        'soft_stop': {'params': list(identity), 'description': 'Enter HOLD'},
        'status': {'params': [], 'description': 'Read diagnostics'},
        'submit_shadow_frame': {'params': ['frame'], 'description': 'Submit Frame v1'},
    }
    return {
        'name': 'teleop_session',
        'type': 'actuator',
        'multiInstance': False,
        'description': 'Robot-free Quest/WebRTC Shadow session diagnostics.',
        'annotations': {'destructiveHint': False, 'idempotentHint': False},
        'inputSchema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                'action': {'type': 'string', 'enum': list(actions)},
                **identity,
                'frame': {'type': 'object', 'description': 'Strict Teleop Frame v1 object'},
            },
            'required': ['action'],
            'x-action-params': actions,
        },
        'x-teleop': {
            'protocol': 'motus.teleop.shadow.v1',
            'driver_id': 'teleop-shadow-driver',
            'driver_name': 'Generic Teleop Shadow Diagnostics',
            'robot_id': 'robot-fixture',
            'mode': 'shadow',
            'actuation_enabled': False,
            'capability_digest': '0123456789abcdef' * 4,
            'dispatch_contract': 'motus.teleop.dispatch.recording.v1',
            'dry_run_profile': 'recording',
            'signaling': {
                'protocol': 'motus.teleop.webrtc-offer-answer.v1',
                'path': '/offer',
                'access': 'authenticated-core-proxy-only',
            },
        },
    }
