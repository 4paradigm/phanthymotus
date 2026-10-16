"""Operator setup and deletion use real route handlers/SQLite, no robot needed."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import auth
import personal_memory as pm
from api.personal_memory import router
from test_personal_memory import store


@pytest.fixture
def api(store, monkeypatch):
    monkeypatch.setattr(auth, '_auth_enabled', True)
    monkeypatch.setattr(auth, '_token', 'operator-test-token')
    app = FastAPI()
    app.include_router(router)
    with TestClient(app, headers={'Authorization': 'Bearer operator-test-token'}) as client:
        yield client


def test_operator_token_required_even_without_global_middleware(api):
    assert api.get('/personal-memory/users', headers={'Authorization': 'Bearer wrong'}).status_code == 401


def test_user_setup_binds_both_modalities_and_rejects_reassignment(api):
    a = api.post('/personal-memory/users', json={'name': 'Alice', 'remember': True}).json()
    b = api.post('/personal-memory/users', json={'name': 'Bob'}).json()
    assert a['id'] != b['id']
    for modality, external_id in [('face', 'p-1'), ('voice', 'p-8')]:
        body = {'modality': modality, 'external_id': external_id}
        assert api.post(f'/personal-memory/users/{a["id"]}/bindings', json=body).status_code == 200
        assert api.post(f'/personal-memory/users/{b["id"]}/bindings', json=body).status_code == 400
    assert len(api.get('/personal-memory/users').json()) == 2
    assert api.post(f'/personal-memory/users/{a["id"]}/bindings',
                    json={'modality': 'unknown', 'external_id': 'p-1'}).status_code == 422


def test_operator_can_review_public_candidates_and_delete_entire_user(api):
    a = api.post('/personal-memory/users', json={'name': 'Alice', 'remember': True}).json()
    pm.bind_context(a['id'])
    pm.save_memory('private', 'fact', 'origin')
    item = pm.propose_public('A generic exercise')
    assert len(api.get('/personal-memory/public-candidates').json()) == 1
    assert api.post(f'/personal-memory/public-candidates/{item["id"]}/approve').status_code == 200
    assert api.get('/personal-memory/public-candidates').json() == []
    assert api.delete(f'/personal-memory/users/{a["id"]}').status_code == 200
    assert pm.get_user(a['id']) is None
    assert pm.recall('') == []


def test_preferences_clear_data_and_unknown_user_is_not_created(api):
    a = api.post('/personal-memory/users', json={'name': 'Alice', 'remember': True}).json()
    pm.bind_context(a['id'])
    pm.save_memory('private', 'fact', 'origin')
    assert api.put(f'/personal-memory/users/{a["id"]}/preferences',
                   json={'remember': False, 'onboarding': False}).status_code == 200
    assert pm.recall('') == []
    assert api.put('/personal-memory/users/missing/preferences',
                   json={'remember': True, 'onboarding': False}).status_code == 400
