"""Site provisioning gates tool advertisement, never VLA or robot motion."""
import ast
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from plugins.teleop import site


def site_configuration(tmp_path, monkeypatch):
    """Real temporary certificate/key/profile files; no ROS or listener."""
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]), False)
            .sign(key, hashes.SHA256()))
    certificate = tmp_path / 'cert.pem'
    private = tmp_path / 'key.pem'
    certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                         serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
    private.chmod(0o600)
    management = tmp_path / 'management-key'
    management.write_text('test-management-key-' + 'a' * 32)
    management.chmod(0o600)
    monkeypatch.setenv('TELEOP_MANAGEMENT_KEY_FILE', str(management))
    monkeypatch.delenv('TELEOP_MANAGEMENT_URL', raising=False)
    model = tmp_path / 'synthetic.urdf'
    model.write_text('<robot name="site-fixture"><link name="torso"/></robot>')
    calibration = tmp_path / 'site.json'
    calibration.write_text(json.dumps({'schema': 'motus.tianyi-calibration.v1',
                                      'urdf_path': str(model),
                                      'urdf_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}))
    return {'enabled': True, 'mode': 'shadow', 'calibration_path': str(calibration),
            'capture': {'discovery_enabled': False, 'port': 15741,
                        'public_wss_url': 'wss://localhost:15741/ws/teleop-capture',
                        'tls_cert_file': str(certificate), 'tls_key_file': str(private),
                        'state_file': str(tmp_path / 'capture.json')}}


def bundle_type():
    # The bundle itself is real; its module's ROS process entry point is not
    # imported in these laptop-only tests, as in test_teleop's MCP contract test.
    source = ast.parse((ROOT / 'main.py').read_text())
    node = next(n for n in source.body if getattr(n, 'name', '') == 'ActuCoreBundle')
    namespace = {'log': logging.getLogger('site-test')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'main.py', 'exec'), namespace)
    return namespace['ActuCoreBundle']


def make_bundle(cfg):
    return bundle_type()({'plugins': {'teleop': cfg,
                                     'vla': {'enabled': True, 'provider': 'mock'}}}, None)


def assert_unavailable(cfg, field):
    bundle = make_bundle(cfg)
    assert [t['name'] for t in bundle.get_all_tools()] == ['vla']
    assert bundle.dispatch('vla', {'action': 'info'})['state'] == 'idle'
    unavailable = bundle.dispatch('teleop', {'action': 'info'})
    assert unavailable['error'] == 'required_site_config'
    assert unavailable['output_active'] is False
    assert field in {item['field'] for item in unavailable['required_site_config']}
    # Field names/codes only: no secrets, host paths or exception traceback.
    assert all(set(item) == {'field', 'code'} for item in unavailable['required_site_config'])
    assert 'test-management-key' not in json.dumps(unavailable)
    assert cfg.get('calibration_path', 'not-a-path') not in json.dumps(unavailable)
    return bundle


def test_complete_site_registers_without_starting_or_requiring_core_client_url(tmp_path, monkeypatch):
    cfg = site_configuration(tmp_path, monkeypatch)
    before = sorted(p.name for p in tmp_path.iterdir())
    assert site.required_site_config(cfg) == []
    bundle = make_bundle(cfg)
    assert {t['name'] for t in bundle.get_all_tools()} == {'teleop', 'vla'}
    assert bundle.required_site_config == {}
    teleop = bundle._plugins[1]
    assert teleop.runtime is None and teleop.link is None and teleop.server is None
    assert not teleop.info()['output_active']
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_remote_motion_control_registers_onboarding_without_robot_model(tmp_path,monkeypatch):
    cfg=site_configuration(tmp_path,monkeypatch)
    cfg.update(control_backend='motion_control',robot_profile='tianyi2')
    Path(cfg.pop('calibration_path')).unlink()
    (tmp_path/'synthetic.urdf').unlink()
    original=site.importlib.import_module
    def no_model_stack(name):
        assert not name.startswith(('pinocchio','casadi'))
        return original(name)
    monkeypatch.setattr(site.importlib,'import_module',no_model_stack)
    assert site.required_site_config(cfg)==[]
    bundle=make_bundle(cfg)
    tools=bundle.get_all_tools()
    teleop=next(t for t in tools if t['name']=='teleop')
    assert 'installation_info' in teleop['inputSchema']['properties']['action']['enum']
    assert bundle._plugins[1].runtime is None


def test_default_disabled_teleop_does_not_check_site_or_import_optional_stack(monkeypatch):
    def unexpected(*_):
        raise AssertionError('disabled card checked site')
    monkeypatch.setattr(site, 'required_site_config', unexpected)
    bundle = make_bundle({'enabled': False})
    assert [t['name'] for t in bundle.get_all_tools()] == ['vla']
    assert bundle.required_site_config == {}


def test_enabled_without_site_does_not_break_vla(monkeypatch):
    monkeypatch.delenv('TELEOP_MANAGEMENT_KEY_FILE', raising=False)
    bundle = assert_unavailable({'enabled': True}, 'TELEOP_MANAGEMENT_KEY_FILE')
    assert {'capture.state_file', 'calibration_path', 'capture.tls_key_file'} <= {
        item['field'] for item in bundle.required_site_config['teleop']}


@pytest.mark.parametrize('failure', ['unset', 'missing', 'short', 'non_ascii', 'newline', 'symlink'])
def test_bad_management_key_hides_card(tmp_path, monkeypatch, failure):
    cfg = site_configuration(tmp_path, monkeypatch)
    key = Path(os.environ['TELEOP_MANAGEMENT_KEY_FILE'])
    if failure == 'unset':
        monkeypatch.delenv('TELEOP_MANAGEMENT_KEY_FILE')
    elif failure == 'missing':
        key.unlink()
    elif failure == 'symlink':
        link = tmp_path / 'key-link'
        link.symlink_to(key)
        monkeypatch.setenv('TELEOP_MANAGEMENT_KEY_FILE', str(link))
    else:
        key.write_text({'short': 'bad', 'non_ascii': '密' * 64,
                        'newline': 'x' * 32 + '\n' + 'y' * 32}[failure])
    assert_unavailable(cfg, 'TELEOP_MANAGEMENT_KEY_FILE')


@pytest.mark.parametrize('failure', ['missing_cert', 'missing_key', 'invalid_cert', 'wrong_san', 'wrong_key'])
def test_real_tls_validation_hides_unusable_card(tmp_path, monkeypatch, failure):
    cfg = site_configuration(tmp_path, monkeypatch)
    capture = cfg['capture']
    if failure == 'missing_cert':
        Path(capture['tls_cert_file']).unlink()
    elif failure == 'missing_key':
        Path(capture['tls_key_file']).unlink()
    elif failure == 'invalid_cert':
        Path(capture['tls_cert_file']).write_text('invalid certificate')
    elif failure == 'wrong_san':
        capture['public_wss_url'] = 'wss://other.invalid:15741/ws/teleop-capture'
    else:
        key = ec.generate_private_key(ec.SECP256R1())
        Path(capture['tls_key_file']).write_bytes(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
    assert_unavailable(cfg, 'capture.tls')


@pytest.mark.parametrize('failure', ['missing_parent', 'readonly', 'bad_json', 'bad_schema', 'bad_mode', 'symlink'])
def test_state_must_be_persistable_and_parse_with_real_capture_parser(tmp_path, monkeypatch, failure):
    cfg = site_configuration(tmp_path, monkeypatch)
    state = Path(cfg['capture']['state_file'])
    if failure == 'missing_parent':
        cfg['capture']['state_file'] = str(tmp_path / 'missing' / 'capture.json')
    elif failure == 'readonly':
        def readonly(*args, **kwargs):
            raise OSError('read-only mount')
        monkeypatch.setattr(site.tempfile, 'TemporaryFile', readonly)
    elif failure == 'symlink':
        state.symlink_to(tmp_path / 'missing-target')
    else:
        state.write_text('bad json' if failure == 'bad_json' else json.dumps({
            'schema_version': 2 if failure == 'bad_schema' else 1, 'capture': None}))
        state.chmod(0o644 if failure == 'bad_mode' else 0o600)
    assert_unavailable(cfg, 'capture.state_file')


@pytest.mark.parametrize('failure', ['missing', 'bad_json', 'bad_schema', 'bad_hash', 'missing_model'])
def test_calibration_file_and_model_must_be_readable_and_correspond(tmp_path, monkeypatch, failure):
    cfg = site_configuration(tmp_path, monkeypatch)
    path = Path(cfg['calibration_path'])
    profile = json.loads(path.read_text())
    if failure == 'missing':
        path.unlink()
    elif failure == 'bad_json':
        path.write_text('not JSON')
    elif failure == 'missing_model':
        Path(profile['urdf_path']).unlink()
    else:
        profile['schema' if failure == 'bad_schema' else 'urdf_sha256'] = 'wrong'
        path.write_text(json.dumps(profile))
    assert_unavailable(cfg, 'calibration_path')


def test_missing_optional_runtime_dependencies_do_not_remove_vla(tmp_path, monkeypatch):
    cfg = site_configuration(tmp_path, monkeypatch)
    real_import = site.importlib.import_module
    def without_pinocchio(name, *args, **kwargs):
        if name == 'pinocchio':
            raise ImportError('optional stack is absent')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(site.importlib, 'import_module', without_pinocchio)
    assert_unavailable(cfg, 'runtime')


def test_readiness_preserves_saved_pairing_state(tmp_path, monkeypatch):
    cfg = site_configuration(tmp_path, monkeypatch)
    state = Path(cfg['capture']['state_file'])
    state.write_text(json.dumps({'schema_version': 1, 'capture': {
        'capture_id': 'b9923587-4ebd-4e13-8d71-3c2dc7b722b8',
        'credential_sha256': 'a' * 64, 'client_kind': 'native_openxr',
        'app_version': '0.3.11'}}))
    state.chmod(0o600)
    before = state.read_bytes()
    assert site.required_site_config(cfg) == []
    assert state.read_bytes() == before
    assert state.stat().st_mode & 0o777 == 0o600
