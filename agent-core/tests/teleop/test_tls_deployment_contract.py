from __future__ import annotations

from pathlib import Path

import yaml

CORE = Path(__file__).resolve().parents[2]
REPOSITORY = CORE.parent


def test_core_compose_passes_mounted_tls_paths_as_an_optional_pair():
    compose = yaml.safe_load(
        (CORE / 'deploy' / 'docker-compose.yml').read_text(encoding='utf-8')
    )
    core_service = compose['services']['agent-core']
    environment = {
        entry.split('=', 1)[0]: entry.split('=', 1)[1]
        for entry in core_service['environment']
    }

    assert '/opt/phanthy-motus:/opt/phanthy-motus' in core_service['volumes']
    assert environment['MOTUS_TLS_CERT_FILE'] == '${MOTUS_TLS_CERT_FILE:-}'
    assert environment['MOTUS_TLS_KEY_FILE'] == '${MOTUS_TLS_KEY_FILE:-}'

    # The self-update path parses and dumps this Compose document with PyYAML.
    # Keep the interpolation expressions intact across that real round trip.
    round_trip = yaml.safe_load(yaml.safe_dump(compose, sort_keys=False))
    round_trip_environment = {
        entry.split('=', 1)[0]: entry.split('=', 1)[1]
        for entry in round_trip['services']['agent-core']['environment']
    }
    assert round_trip_environment['MOTUS_TLS_CERT_FILE'] == '${MOTUS_TLS_CERT_FILE:-}'
    assert round_trip_environment['MOTUS_TLS_KEY_FILE'] == '${MOTUS_TLS_KEY_FILE:-}'


def test_deployment_env_example_uses_container_visible_tls_paths():
    example = (REPOSITORY / 'deploy' / '.env.example').read_text(encoding='utf-8')

    assert (
        'MOTUS_TLS_CERT_FILE=/opt/phanthy-motus/tls/fullchain.pem'
        in example
    )
    assert (
        'MOTUS_TLS_KEY_FILE=/opt/phanthy-motus/tls/privkey.pem'
        in example
    )
    assert 'Configure both paths or neither.' in example
    assert 'exact Quest Browser hostname in its SAN' in example
    assert 'chmod 600, or 400' in example


def test_start_uses_the_fail_closed_tls_resolver():
    source = (CORE / 'src' / 'start.py').read_text(encoding='utf-8')

    assert 'load_public_certificate_chain, resolve_tls_certificates' in source
    assert 'cert_file, key_file = resolve_tls_certificates()' in source
    assert 'app_api.state.teleop_capture_ca_certificate_pem' in source
    assert 'load_public_certificate_chain(cert_file)' in source
    assert 'ssl_certfile=cert_file, ssl_keyfile=key_file' in source
    assert '_ensure_ssl_certs' not in source


def test_readmes_require_browser_accepted_exact_host_certificates_for_quest():
    english = (REPOSITORY / 'README.md').read_text(encoding='utf-8')
    chinese = (REPOSITORY / 'README_zh.md').read_text(encoding='utf-8')

    assert 'http://<device-ip>:15678' not in english
    assert 'http://<\u8bbe\u5907IP>:15678' not in chinese
    assert 'Subject Alternative Name (SAN)' in english
    assert '完整证书链也必须被 Quest Browser 接受' in chinese
    assert 'does **not** guarantee browser trust' in english
    assert '**不保证**浏览器信任' in chinese
    assert 'window.isSecureContext=true' in english
    assert 'window.isSecureContext=true' in chinese
    assert 'Merely seeing an `https://` URL' in english
    assert '仅看到 `https://` URL' in chinese
    assert 'private key must be owner-only (`0600` or `0400`)' in english
    assert '私钥必须为 owner-only（`0600` 或 `0400`）' in chinese
    assert 'parseable' in english
    assert 'matching server pair' in english
    assert '可解析且相互匹配的服务端 pair' in chinese
