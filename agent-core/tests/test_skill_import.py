"""Local authoring uses the existing skill visibility and activation path."""
import asyncio
import os
from pathlib import Path
import sys
import tempfile

import fastapi
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'skill-import.db'))

import config
from api import skills

DEFINITION = {'slug': 'taught-123', 'name': '零件放入托盘', 'oneLiner': '把零件放到托盘里',
              'description': '固定位置抓取', 'instruction': '先预览，再确认本次接管，然后执行。',
              'category': 'robotics', 'version': '1.0.0', 'requiredTools': ['piper_teaching']}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, 'main', {'skills': {'installed': []}})
    app = fastapi.FastAPI()
    app.include_router(skills.router, prefix='/api')
    with TestClient(app) as client:
        yield client


def test_import_list_activate_and_retry_without_resource_center(client):
    import prompt
    assert DEFINITION['slug'] not in prompt._env_static({})
    response = client.post('/api/skills/import', json=DEFINITION)
    assert response.status_code == 200
    assert response.json()['data']['already'] is False
    result = client.get('/api/skills').json()['data']
    assert len(result) == 1 and result[0]['active']
    assert DEFINITION['slug'] in prompt._env_static({})
    assert DEFINITION['instruction'] not in prompt._env_static({})
    assert client.get('/api/skills/taught-123').json()['data']['instruction'] == DEFINITION['instruction']
    assert '已激活' in asyncio.run(skills._skills_mod.Tools().activate_skill('taught-123'))
    assert skills._skills_mod.get_active_skills()[0]['slug'] == 'taught-123'
    assert DEFINITION['instruction'] in prompt._env_static({})
    # Idempotent import must not reset a user's visibility setting.
    config.main['skills']['installed'][0]['active'] = False
    assert client.post('/api/skills/import', json=DEFINITION).json()['data']['already'] is True
    assert len(config.main['skills']['installed']) == 1
    assert not config.main['skills']['installed'][0]['active']
    skills._skills_mod._runtime_activated.discard('taught-123')


def test_conflict_never_overwrites_existing_skill(client):
    client.post('/api/skills/import', json=DEFINITION)
    response = client.post('/api/skills/import', json={**DEFINITION, 'instruction': 'changed'})
    assert response.status_code == 409
    assert config.main['skills']['installed'][0]['instruction'] == DEFINITION['instruction']


@pytest.mark.parametrize('change', [
    {'slug': '../outside'}, {'name': ''}, {'instruction': []}, {'requiredTools': 'piper_teaching'},
    {'requiredTools': ['../file']}, {'active': True}, {'icon': '<img onerror=alert(1)>'},
    {'name': 'a' * 101}, {'oneLiner': ' '}, {'configSchema': {'untrusted': True}},
])
def test_invalid_definitions_are_rejected_without_writes(client, change):
    assert client.post('/api/skills/import', json={**DEFINITION, **change}).status_code == 422
    assert config.main['skills']['installed'] == []


def test_bounded_payload(client):
    assert client.post('/api/skills/import', json={**DEFINITION, 'instruction': 'x'*33000}).status_code == 413


def test_independent_imports_keep_both_skills(client):
    async def imports():
        return await asyncio.gather(skills.import_local_skill(DEFINITION),
                                   skills.import_local_skill({**DEFINITION, 'slug': 'taught-other'}))
    assert len(asyncio.run(imports())) == 2
    assert {s['slug'] for s in config.main['skills']['installed']} == {'taught-123', 'taught-other'}
