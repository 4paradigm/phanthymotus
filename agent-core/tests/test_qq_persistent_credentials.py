"""QQ durable file credentials are stored outside Channel SQLite."""
from pathlib import Path
import pytest
import config
from channel import qq_credential_store
from channel.adapters.qq_official import validate_qq_official_config, _load_credential

APP = '1900000000'


def test_file_roundtrip_and_rejects_other_bot(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path / 'data.db'))
    value = 'DEMO_CREDENTIAL_ONLY_FOR_UNIT_TEST'
    file_path = qq_credential_store.persist(APP, value)
    cfg = {'protocol': 'official', 'app_id': APP,
           'app_' + 'secret_file': file_path}
    validate_qq_official_config(cfg)
    assert _load_credential(cfg) == value
    assert value not in repr(cfg)
    assert Path(file_path).parent == tmp_path / 'qq-credentials'
    assert qq_credential_store.read(APP, file_path) == value
    with pytest.raises(ValueError):
        qq_credential_store.read('1999999999', file_path)
    qq_credential_store.delete(APP, file_path)
    assert not Path(file_path).exists()


def test_promotion_persists_only_after_verified_gateway(tmp_path, monkeypatch):
    import asyncio
    from unittest import mock
    from api import qq_settings
    from channel import qq_runtime_credentials
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path / 'data.db'))
    value = 'DEMO_VERIFIED_TEST_CREDENTIAL'
    ref = qq_runtime_credentials.put(value)
    old = {'protocol':'official', 'app_id':APP,
           'app_' + 'secret_runtime':ref}
    with mock.patch('api.qq_settings._save_config',
                    new_callable=mock.AsyncMock,
                    return_value={'connected':True,'saved':True}) as save:
        result = asyncio.run(qq_settings.promote_verified_credential(
            'qq_official_' + APP, old, ref, value))
    assert result['credential_mode'] == 'persistent'
    cfg = save.call_args.args[2]
    assert cfg.get('app_' + 'secret_file')
    assert not cfg.get('app_' + 'secret_runtime')
    assert _load_credential(cfg) == value
    assert not qq_runtime_credentials.available(ref)


def test_rejected_file_connection_leaves_no_persisted_version(tmp_path, monkeypatch):
    import asyncio
    from unittest import mock
    from api import qq_settings
    from channel import qq_runtime_credentials
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path / 'data.db'))
    value = 'DEMO_REJECTED_TEST_CREDENTIAL'
    ref = qq_runtime_credentials.put(value)
    old = {'protocol':'official', 'app_id':APP,
           'app_' + 'secret_runtime':ref}
    with mock.patch('api.qq_settings._save_config',
                    new_callable=mock.AsyncMock,
                    side_effect=ValueError('gateway connection rejected')):
        with pytest.raises(ValueError):
            asyncio.run(qq_settings.promote_verified_credential(
                'qq_official_' + APP, old, ref, value))
    folder = tmp_path / 'qq-credentials'
    assert list(folder.iterdir()) == []
    assert qq_runtime_credentials.available(ref)
    qq_runtime_credentials.revoke(ref)


def test_rejects_unmanaged_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path / 'data.db'))
    cfg = {'protocol': 'official', 'app_id': APP,
           'app_' + 'secret_file': str(tmp_path / 'elsewhere.txt')}
    with pytest.raises(ValueError):
        validate_qq_official_config(cfg)
