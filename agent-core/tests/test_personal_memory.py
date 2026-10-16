"""Biometric OR routing and owner checks, using real SQLite storage."""
import json
import time

import pytest
import config
import personal_memory as pm


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path / 'personal.db'))
    monkeypatch.setattr(config, 'main', config.ConfigDB())
    monkeypatch.setattr(pm, 'resolver', pm.IdentityResolver())
    config.main['personal_memory'] = {
        'enabled': True, 'voice_sources': ['dds:/mic/asr'],
        'face_sources': ['dds:/camera/face'], 'face_max_age_s': 3,
        'output_tools': [],
    }
    token = pm.bind_context(None)
    turn_token = pm._turn.set('')
    yield
    pm._turn.reset(turn_token)
    pm.reset_context(token)


def event(source, **data):
    return {'source': source, 'text': json.dumps(data), 'ts': time.time()}


def test_face_and_voice_ids_are_not_implicitly_equal(store):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_identity(a['id'], 'voice', 'p-1')
    pm.bind_identity(b['id'], 'face', 'p-1')
    resolver = pm.IdentityResolver()
    assert resolver.resolve(event('dds:/mic/asr', speaker_id='p-1')) == a['id']
    assert resolver.resolve(event('dds:/camera/face', faces=[{'person_id': 'p-1', 'quality': 'ok'}])) == b['id']
    assert resolver.resolve(event('dds:/mic/asr', speaker_id='p-1')) is None


def test_either_modality_unlocks_same_explicitly_bound_user(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_identity(a['id'], 'voice', 'v-2')
    pm.bind_identity(a['id'], 'face', 'f-9')
    resolver = pm.IdentityResolver()
    assert resolver.resolve(event('dds:/mic/asr', speaker_id='v-2')) == a['id']
    assert resolver.resolve(event('dds:/camera/face', faces=[{'person_id': 'f-9', 'quality': 'ok'}])) == a['id']
    assert resolver.resolve(event('dds:/mic/asr', text='只刷脸也能聊')) == a['id']


def test_unknown_multi_and_remote_input_do_not_inherit_user(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_identity(a['id'], 'voice', 'v-1')
    resolver = pm.IdentityResolver()
    assert resolver.resolve(event('dds:/mic/asr', speaker_id='v-1')) == a['id']
    for ev in [event('dds:/remote_control/message', speaker_id='v-1'),
               event('dds:/mic/asr', speaker_id='unknown'),
               event('dds:/mic/asr', speaker_id='v-1', multi_speaker=True),
               event('dds:/mic/asr', text='no biometric')]:
        assert resolver.resolve(ev) is None


def test_stale_and_multiple_faces_do_not_unlock(store):
    a = pm.create_user('Alice', remember=True)
    pm.bind_identity(a['id'], 'face', 'f-1')
    resolver = pm.IdentityResolver()
    old = event('dds:/camera/face', faces=[{'person_id': 'f-1', 'quality': 'ok'}])
    old['ts'] -= 10
    assert resolver.resolve(old) is None
    assert resolver.resolve(event('dds:/mic/asr', text='hello')) is None
    assert resolver.resolve(event('dds:/camera/face', faces=[
        {'person_id': 'f-1', 'quality': 'ok'}, {'person_id': None}])) is None
    assert resolver.resolve(event('dds:/mic/asr', text='hello')) is None


def test_memory_and_history_are_scoped_and_public_needs_approval(store):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_context(a['id'])
    secret = pm.save_memory('Alice secret', 'fact', 'turn-a')
    pm.save_turn('turn-a', [{'role': 'user', 'content': 'Alice transcript'}])
    candidate = pm.propose_public('Generic exercise')
    pm.bind_context(b['id'])
    assert pm.recall('') == []
    assert pm.load_turns() == []
    assert pm.delete_memory(secret['id']) is False
    with pytest.raises(ValueError):
        pm.save_memory('diagnosis', 'knowledge', 'turn-b')
    pm.approve_public(candidate['id'])
    assert [m['content'] for m in pm.recall('')] == ['Generic exercise']
    pm.bind_context(a['id'])
    assert [m['content'] for m in pm.recall('secret')] == ['Alice secret']
    assert pm.load_turns()[0][0]['content'] == 'Alice transcript'
    assert pm.delete_memory(secret['id']) is True
    assert pm.recall('secret') == []


def test_opt_out_prevents_persistent_memory_and_history(store):
    a = pm.create_user('Alice', remember=False)
    pm.bind_context(a['id'])
    with pytest.raises(PermissionError):
        pm.save_memory('secret', 'fact', 'turn')
    pm.save_turn('turn', [{'role': 'user', 'content': 'secret'}])
    assert pm.load_turns() == []
    pm.bind_context(None)
    with pytest.raises(PermissionError):
        pm.save_memory('guest', 'fact', 'turn')


def test_identity_binding_cannot_silently_reassign_and_user_delete_cascades(store):
    a = pm.create_user('Alice', remember=True)
    b = pm.create_user('Bob', remember=True)
    pm.bind_identity(a['id'], 'voice', 'v-1')
    with pytest.raises(ValueError):
        pm.bind_identity(b['id'], 'voice', 'v-1')
    pm.bind_context(a['id'])
    pm.save_memory('secret', 'analysis', 'turn')
    pm.save_turn('turn', [{'role': 'user', 'content': 'secret'}])
    pm.propose_public('pending')
    pm.delete_user(a['id'])
    assert pm.IdentityResolver().resolve(event('dds:/mic/asr', speaker_id='v-1')) is None
    assert pm.load_turns() == []
    assert pm.recall('') == []
    assert pm.public_candidates() == []
