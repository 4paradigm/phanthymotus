from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import tls_config

_REAL_VALIDATE_CERTIFICATE_PAIR = tls_config._validate_certificate_pair


@pytest.fixture(autouse=True)
def _stub_certificate_pair_validation(monkeypatch: pytest.MonkeyPatch):
    """Keep path-policy tests small; dedicated tests exercise real OpenSSL."""

    monkeypatch.setattr(
        tls_config,
        '_validate_certificate_pair',
        lambda _cert_path, _key_path: None,
    )


def _write_pair(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    cert = directory / 'cert.pem'
    key = directory / 'key.pem'
    cert.write_text('certificate', encoding='utf-8')
    key.write_text('private-key', encoding='utf-8')
    key.chmod(0o600)
    return cert, key


def _unexpected_openssl(*args, **kwargs):
    raise AssertionError('openssl must not run for this TLS configuration')


def _minimal_public_certificate_pem() -> str:
    der = b'\x30\x03\x02\x01\x01'
    body = base64.b64encode(der).decode('ascii')
    return f'-----BEGIN CERTIFICATE-----\n{body}\n-----END CERTIFICATE-----\n'


def test_public_certificate_projection_rebuilds_certificate_blocks_only():
    projected = tls_config.project_public_certificate_chain(
        '\n' + _minimal_public_certificate_pem() + '\n'
    )

    assert projected == _minimal_public_certificate_pem()
    assert 'PRIVATE KEY' not in projected


@pytest.mark.parametrize('invalid_suffix', [
    'deployment comment',
    '-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----',
])
def test_public_certificate_projection_rejects_non_certificate_material(
    invalid_suffix: str,
):
    with pytest.raises(tls_config.TLSConfigurationError):
        tls_config.project_public_certificate_chain(
            _minimal_public_certificate_pem() + invalid_suffix
        )


def test_public_certificate_loader_is_bounded(tmp_path: Path):
    oversized = tmp_path / 'oversized.pem'
    oversized.write_bytes(
        b'-' * (tls_config.MAX_PUBLIC_CERTIFICATE_CHAIN_BYTES + 1)
    )

    with pytest.raises(tls_config.TLSConfigurationError, match='exceeds'):
        tls_config.load_public_certificate_chain(oversized)


def test_existing_compatibility_pair_is_reused_without_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cert, key = _write_pair(tmp_path / 'compatibility')
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    resolved = tls_config.resolve_tls_certificates(
        environ={},
        cert_dir=cert.parent,
    )

    assert resolved == (str(cert), str(key))
    assert cert.read_text(encoding='utf-8') == 'certificate'
    assert key.read_text(encoding='utf-8') == 'private-key'


def test_missing_compatibility_pair_generates_both_files_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    calls: list[list[str]] = []

    def fake_openssl(command, *, check, capture_output):
        calls.append(command)
        assert check is True
        assert capture_output is True
        cert = Path(command[command.index('-out') + 1])
        key = Path(command[command.index('-keyout') + 1])
        cert.write_text('generated-certificate', encoding='utf-8')
        key.write_text('generated-private-key', encoding='utf-8')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tls_config.subprocess, 'run', fake_openssl)
    cert_dir = tmp_path / 'generated'

    resolved = tls_config.resolve_tls_certificates(
        environ={},
        cert_dir=cert_dir,
    )

    assert resolved == (str(cert_dir / 'cert.pem'), str(cert_dir / 'key.pem'))
    assert (cert_dir / 'key.pem').stat().st_mode & 0o777 == 0o600
    assert len(calls) == 1
    assert calls[0][0:2] == ['openssl', 'req']
    output = capsys.readouterr().out
    assert 'self-signed compatibility fallback' in output
    assert 'does not establish browser trust or Quest WebXR readiness' in output


def test_generated_pair_is_rejected_if_both_paths_resolve_to_one_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_openssl(command, *, check, capture_output):
        assert check is True
        assert capture_output is True
        cert = Path(command[command.index('-out') + 1])
        key = Path(command[command.index('-keyout') + 1])
        cert.write_text('combined', encoding='utf-8')
        key.unlink()
        key.symlink_to(cert)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tls_config.subprocess, 'run', fake_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='different files'):
        tls_config.resolve_tls_certificates(
            environ={},
            cert_dir=tmp_path / 'generated',
        )


@pytest.mark.skipif(shutil.which('openssl') is None, reason='openssl is unavailable')
def test_real_generated_pair_is_parseable_matching_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        tls_config,
        '_validate_certificate_pair',
        _REAL_VALIDATE_CERTIFICATE_PAIR,
    )

    cert_file, key_file = tls_config.resolve_tls_certificates(
        environ={},
        cert_dir=tmp_path / 'real-pair',
    )

    assert Path(cert_file).is_file()
    assert Path(key_file).stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(shutil.which('openssl') is None, reason='openssl is unavailable')
def test_real_mismatched_explicit_pair_fails_before_server_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        tls_config,
        '_validate_certificate_pair',
        _REAL_VALIDATE_CERTIFICATE_PAIR,
    )
    cert_a, _key_a = tls_config.resolve_tls_certificates(
        environ={},
        cert_dir=tmp_path / 'pair-a',
    )
    _cert_b, key_b = tls_config.resolve_tls_certificates(
        environ={},
        cert_dir=tmp_path / 'pair-b',
    )

    with pytest.raises(tls_config.TLSConfigurationError, match='parseable matching'):
        tls_config.resolve_tls_certificates(
            environ={
                tls_config.TLS_CERT_FILE_ENV: cert_a,
                tls_config.TLS_KEY_FILE_ENV: key_b,
            },
            cert_dir=tmp_path / 'must-not-exist',
        )


def test_existing_compatibility_key_permissions_are_tightened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cert, key = _write_pair(tmp_path / 'compatibility')
    key.chmod(0o644)
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    resolved = tls_config.resolve_tls_certificates(environ={}, cert_dir=cert.parent)

    assert resolved == (str(cert), str(key))
    assert key.stat().st_mode & 0o777 == 0o600


def test_broad_managed_compatibility_symlink_fails_without_chmod_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    managed = tmp_path / 'managed'
    cert, key = _write_pair(managed)
    key.chmod(0o640)
    compatibility = tmp_path / 'compatibility'
    compatibility.mkdir()
    (compatibility / 'cert.pem').symlink_to(cert)
    (compatibility / 'key.pem').symlink_to(key)
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='linked compatibility'):
        tls_config.resolve_tls_certificates(environ={}, cert_dir=compatibility)

    assert key.stat().st_mode & 0o777 == 0o640


def test_broad_hard_linked_compatibility_key_fails_without_chmod_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    compatibility = tmp_path / 'compatibility'
    _cert, key = _write_pair(compatibility)
    key.chmod(0o640)
    other_link = tmp_path / 'shared-key.pem'
    os.link(key, other_link)
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='multiply-linked'):
        tls_config.resolve_tls_certificates(environ={}, cert_dir=compatibility)

    assert key.stat().st_mode & 0o777 == 0o640
    assert other_link.stat().st_mode & 0o777 == 0o640


def test_explicit_pair_is_reused_without_touching_files_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cert, key = _write_pair(tmp_path / 'deployment')
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    resolved = tls_config.resolve_tls_certificates(
        environ={
            tls_config.TLS_CERT_FILE_ENV: str(cert),
            tls_config.TLS_KEY_FILE_ENV: str(key),
        },
        cert_dir=tmp_path / 'must-not-exist',
    )

    assert resolved == (str(cert), str(key))
    assert cert.read_text(encoding='utf-8') == 'certificate'
    assert key.read_text(encoding='utf-8') == 'private-key'
    assert not (tmp_path / 'must-not-exist').exists()


def test_explicit_private_key_with_group_or_world_access_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cert, key = _write_pair(tmp_path / 'deployment')
    key.chmod(0o640)
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='mode to 0600 or 0400'):
        tls_config.resolve_tls_certificates(
            environ={
                tls_config.TLS_CERT_FILE_ENV: str(cert),
                tls_config.TLS_KEY_FILE_ENV: str(key),
            },
            cert_dir=tmp_path / 'must-not-exist',
        )

    assert key.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize('configured_name', [
    tls_config.TLS_CERT_FILE_ENV,
    tls_config.TLS_KEY_FILE_ENV,
])
def test_partial_explicit_pair_fails_closed(
    configured_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    configured_file = tmp_path / 'configured.pem'
    configured_file.write_text('configured', encoding='utf-8')
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='configured together'):
        tls_config.resolve_tls_certificates(
            environ={configured_name: str(configured_file)},
            cert_dir=tmp_path / 'must-not-exist',
        )

    assert not (tmp_path / 'must-not-exist').exists()


@pytest.mark.parametrize('bad_kind', ['missing', 'directory'])
@pytest.mark.parametrize('bad_name', [
    tls_config.TLS_CERT_FILE_ENV,
    tls_config.TLS_KEY_FILE_ENV,
])
def test_non_file_explicit_path_fails_closed(
    bad_kind: str,
    bad_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cert, key = _write_pair(tmp_path / 'deployment')
    bad_path = tmp_path / 'bad-path'
    if bad_kind == 'directory':
        bad_path.mkdir()
    environ = {
        tls_config.TLS_CERT_FILE_ENV: str(cert),
        tls_config.TLS_KEY_FILE_ENV: str(key),
    }
    environ[bad_name] = str(bad_path)
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='regular file'):
        tls_config.resolve_tls_certificates(
            environ=environ,
            cert_dir=tmp_path / 'must-not-exist',
        )

    assert not (tmp_path / 'must-not-exist').exists()


@pytest.mark.parametrize('present_name', ['cert.pem', 'key.pem'])
def test_partial_compatibility_pair_fails_without_overwriting(
    present_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cert_dir = tmp_path / 'partial'
    cert_dir.mkdir()
    present = cert_dir / present_name
    present.write_text('must-survive', encoding='utf-8')
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='incomplete'):
        tls_config.resolve_tls_certificates(environ={}, cert_dir=cert_dir)

    assert present.read_text(encoding='utf-8') == 'must-survive'
    assert len(list(cert_dir.iterdir())) == 1


def test_broken_default_symlinks_fail_without_running_openssl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cert_dir = tmp_path / 'broken-links'
    cert_dir.mkdir()
    cert_link = cert_dir / 'cert.pem'
    key_link = cert_dir / 'key.pem'
    cert_link.symlink_to(tmp_path / 'missing-cert.pem')
    key_link.symlink_to(tmp_path / 'missing-key.pem')
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='incomplete'):
        tls_config.resolve_tls_certificates(environ={}, cert_dir=cert_dir)

    assert cert_link.is_symlink()
    assert key_link.is_symlink()


def test_explicit_certificate_and_key_must_not_be_the_same_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    combined = tmp_path / 'combined.pem'
    combined.write_text('combined', encoding='utf-8')
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='different files'):
        tls_config.resolve_tls_certificates(
            environ={
                tls_config.TLS_CERT_FILE_ENV: str(combined),
                tls_config.TLS_KEY_FILE_ENV: str(combined),
            },
            cert_dir=tmp_path / 'must-not-exist',
        )


@pytest.mark.skipif(not hasattr(os, 'symlink'), reason='symlinks are unavailable')
def test_explicit_symlinks_to_distinct_regular_files_are_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cert, key = _write_pair(tmp_path / 'managed')
    cert_link = tmp_path / 'fullchain.pem'
    key_link = tmp_path / 'privkey.pem'
    cert_link.symlink_to(cert)
    key_link.symlink_to(key)
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    resolved = tls_config.resolve_tls_certificates(
        environ={
            tls_config.TLS_CERT_FILE_ENV: str(cert_link),
            tls_config.TLS_KEY_FILE_ENV: str(key_link),
        },
        cert_dir=tmp_path / 'must-not-exist',
    )

    assert resolved == (str(cert_link), str(key_link))


def test_distinct_symlinks_to_the_same_file_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    combined = tmp_path / 'combined.pem'
    combined.write_text('combined', encoding='utf-8')
    cert_link = tmp_path / 'cert-link.pem'
    key_link = tmp_path / 'key-link.pem'
    cert_link.symlink_to(combined)
    key_link.symlink_to(combined)
    monkeypatch.setattr(tls_config.subprocess, 'run', _unexpected_openssl)

    with pytest.raises(tls_config.TLSConfigurationError, match='different files'):
        tls_config.resolve_tls_certificates(
            environ={
                tls_config.TLS_CERT_FILE_ENV: str(cert_link),
                tls_config.TLS_KEY_FILE_ENV: str(key_link),
            },
            cert_dir=tmp_path / 'must-not-exist',
        )
