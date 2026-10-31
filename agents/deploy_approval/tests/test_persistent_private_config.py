"""Persistent private configuration tests for Deploy Approval.

Covers:
  - docker-compose.yml persistent volume mounts (not relative)
  - deploy.sh persistent directory and preflight command
  - Missing configuration rejection
  - Permission-denied rejection
  - Placeholder value rejection (machines.yaml and secrets.yaml)
  - Cross-SHA persistence (persistent dir independent of release dir)
  - Read-only mount verification
  - COS path protocol (phanthymotus_pr/<repo-dir>/<YYYY-MM>/<YYYY-MM-DD>/pr-<N>/evidence-<40hex>.log.gz)
  - Config regression (production COS guard, fake test values still work)
"""

from __future__ import annotations

import datetime
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent.parent
COMPOSE_PATH = ROOT / "deploy" / "deploy-approval" / "docker-compose.yml"
DEPLOY_SH_PATH = ROOT / "deploy" / "deploy-approval" / "deploy.sh"


def _compose():
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def _deploy_sh():
    return DEPLOY_SH_PATH.read_text(encoding="utf-8")


def _secrets_yaml(ca_pem, **overrides):
    """Return offline secrets with a test CA for each explicit token alias."""
    base = {
        "version": 1,
        "cos": {
            "region": "ap-beijing",
            "bucket": "testbucket123",
            "secret_id": "TEST_FAKE_SECRET_ID_AAAA",
            "secret_key": "TEST_FAKE_SECRET_KEY_AAAA",
        },
        "review_comment_trust": {
            "author_id": "7950763",
            "author_login": "kentcyq",
        },
        "agent_core_tokens": {
            "test-machine": "test-token",
        },
    }
    base.update(overrides)
    base.setdefault("agent_core_ca_pem", {
        alias: ca_pem for alias in base["agent_core_tokens"]
    })
    return yaml.dump(base, default_flow_style=False)


def _machines_yaml(**overrides):
    """Return a minimal valid machines.yaml with optional overrides."""
    base = {
        "version": 1,
        "machines": {
            "test-machine": {
                "node_id": "test-node-1",
                "owners": ["test-owner"],
                "node_host": "10.0.0.1",
                "targets": ["perception"],
                "platforms": ["linux/arm64"],
                "variants": ["1.0"],
                "driver_paths": [],
            },
        },
    }
    base.update(overrides)
    return yaml.dump(base, default_flow_style=False)


# ── 1. Compose persistent volume mounts ─────────────────────────────────

def test_compose_uses_persistent_volume_mounts():
    compose = _compose()
    svc = compose["services"]["deploy-approval"]
    volumes = svc.get("volumes", [])
    # After migration to long-form bind mounts, volumes are dicts.
    machine_vol = None
    secrets_vol = None
    for v in volumes:
        if isinstance(v, dict):
            src = str(v.get("source", ""))
            tgt = str(v.get("target", ""))
            if "machines.yaml" in tgt:
                machine_vol = v
            elif "secrets.yaml" in tgt:
                secrets_vol = v
    assert machine_vol is not None, "machines.yaml bind mount must exist"
    assert secrets_vol is not None, "secrets.yaml bind mount must exist"
    assert "DEPLOY_APPROVAL_CONFIG_DIR" in str(machine_vol["source"])
    assert str(machine_vol["target"]) == "/run/deploy-approval/machines.yaml"
    assert machine_vol.get("read_only") is True
    assert "DEPLOY_APPROVAL_CONFIG_DIR" in str(secrets_vol["source"])
    assert str(secrets_vol["target"]) == "/run/deploy-approval/secrets.yaml"
    assert secrets_vol.get("read_only") is True
    # Ensure no short-form relative mounts remain
    for v in volumes:
        if isinstance(v, str):
            assert not v.startswith("./machines.yaml")
            assert not v.startswith("./secrets.yaml")


def test_compose_volumes_are_read_only():
    compose = _compose()
    svc = compose["services"]["deploy-approval"]
    volumes = svc.get("volumes", [])
    # All config-related volumes must be long-form dicts with read_only=true
    config_vols = [v for v in volumes if isinstance(v, dict)]
    assert len(config_vols) >= 3, (
        f"Expected at least 3 long-form bind volumes, got {len(config_vols)}"
    )
    for v in config_vols:
        assert v.get("type") == "bind", f"Volume must be bind type: {v}"
        assert v.get("read_only") is True, (
            f"All bind volumes must have read_only=true: {v}"
        )
        bind_opts = v.get("bind", {})
        assert bind_opts.get("create_host_path") is False, (
            f"bind.create_host_path must be false: {v}"
        )


# ── 2. deploy.sh persistent directory and preflight ─────────────────────

def test_deploy_sh_uses_env_var_config_dir():
    text = _deploy_sh()
    assert "DEPLOY_APPROVAL_CONFIG_DIR" in text
    assert 'PERSISTENT_DIR="${DEPLOY_APPROVAL_CONFIG_DIR:' in text
    assert "/var/lib/deploy-approval" not in text
    assert '${PERSISTENT_DIR}/machines.yaml' in text
    assert '${PERSISTENT_DIR}/secrets.yaml' in text


def test_deploy_sh_has_preflight_command():
    text = _deploy_sh()
    assert "cmd_preflight" in text
    assert "preflight)" in text
    assert "preflight" in text.split("# Commands:")[1].split("\n")[1]


def test_deploy_preflight_checks_directory_exists():
    text = _deploy_sh()
    assert "Persistent directory does not exist" in text


def test_deploy_preflight_checks_symlink():
    text = _deploy_sh()
    assert "Persistent directory must not be a symlink" in text


def test_deploy_preflight_requires_config_dir():
    text = _deploy_sh()
    assert "DEPLOY_APPROVAL_CONFIG_DIR" in text
    assert "DEPLOY_APPROVAL_CONFIG_DIR is required" in text


# ── 3. Missing configuration rejection ─────────────────────────────────

def _extract_bodies(deploy_sh_text):
    """Extract require_secrets and require_machine_policy bodies."""
    m1 = re.search(r"require_secrets\(\)\s*\{(.*?)\n\}", deploy_sh_text, re.DOTALL)
    assert m1, "require_secrets must exist in deploy.sh"
    m2 = re.search(r"require_machine_policy\(\)\s*\{(.*?)\n\}", deploy_sh_text, re.DOTALL)
    assert m2, "require_machine_policy must exist in deploy.sh"
    return m1.group(1), m2.group(1)


def test_preflight_compatibility_without_ca(tmp_path, offline_ca_pem, monkeypatch):
    """Execute the actual preflight validators against local fixture files only."""
    secrets_body, machine_body = _extract_bodies(_deploy_sh())
    secrets = tmp_path / "secrets.yaml"
    machines = tmp_path / "machines.yaml"
    machines.write_text(_machines_yaml(), encoding="utf-8")
    wrapper = tmp_path / "preflight.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'SECRETS_FILE="' + str(secrets) + '"\n'
        'MACHINES_FILE="' + str(machines) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_machine_policy() {\n" + machine_body + "\n}\n"
        "require_secrets() {\n" + secrets_body + "\n}\n"
        "require_machine_policy\nrequire_secrets\n", encoding="utf-8",
    )
    for mode, ca, expected in (
        (None, {}, False),
        ("legacy", {}, True),
        ("strict", {}, False),
        ("strict", {"test-machine": offline_ca_pem}, True),
        ("strcit", {}, False),
    ):
        fixture = yaml.safe_load(_secrets_yaml(
            offline_ca_pem, agent_core_ca_pem=ca,
        ))
        if mode is None:
            monkeypatch.delenv("AGENT_CORE_TLS_MODE", raising=False)
        else:
            monkeypatch.setenv("AGENT_CORE_TLS_MODE", mode)
        secrets.write_text(yaml.safe_dump(fixture), encoding="utf-8")
        result = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True, timeout=30)
        assert (result.returncode == 0) is expected
        output = result.stdout + result.stderr
        if expected:
            assert "SECRETS_OK" in output
            assert f"TLS_MODE={mode or 'strict'}" in output
        elif mode in (None, "strict"):
            assert "CA_NOT_PROVISIONED" in output
        else:
            assert "AGENT_CORE_TLS_MODE" in output
        assert "test-token" not in output
        assert "TEST_FAKE_SECRET_KEY_AAAA" not in output

    monkeypatch.setenv("AGENT_CORE_TLS_MODE", "legacy")
    for host, tokens, error in (
        ("192.168.1.100", {"other-machine": "offline-other-token"}, "token required"),
        ("http://192.168.1.100:15678", {"test-machine": "test-token"}, "literal IPv4"),
    ):
        policy = yaml.safe_load(_machines_yaml())
        policy["machines"]["test-machine"]["node_host"] = host
        machines.write_text(yaml.safe_dump(policy), encoding="utf-8")
        secrets.write_text(_secrets_yaml(
            offline_ca_pem,
            agent_core_ca_pem={}, agent_core_tokens=tokens,
        ), encoding="utf-8")
        result = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert error in result.stdout + result.stderr


def _tls_preflight(tmp_path, mode):
    secrets_body, machine_body = _extract_bodies(_deploy_sh())
    secrets = tmp_path / "secrets.yaml"
    machines = tmp_path / "machines.yaml"
    # Deliberately omit the CA key entirely, matching the old server schema.
    data = yaml.safe_load(_secrets_yaml(None, agent_core_ca_pem={}))
    del data["agent_core_ca_pem"]
    secrets.write_text(yaml.safe_dump(data), encoding="utf-8")
    machines.write_text(_machines_yaml(), encoding="utf-8")
    wrapper = tmp_path / "tls-preflight.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'SECRETS_FILE="' + str(secrets) + '"\n'
        'MACHINES_FILE="' + str(machines) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_machine_policy() {\n" + machine_body + "\n}\n"
        "require_secrets() {\n" + secrets_body + "\n}\n"
        "require_machine_policy\nrequire_secrets\n", encoding="utf-8",
    )
    env = os.environ.copy()
    if mode is None:
        env.pop("AGENT_CORE_TLS_MODE", None)
    else:
        env["AGENT_CORE_TLS_MODE"] = mode
    return subprocess.run(["bash", str(wrapper)], env=env, capture_output=True, text=True, timeout=30)


def test_preflight_legacy_no_ca_passes(tmp_path):
    result = _tls_preflight(tmp_path, "legacy")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SECRETS_OK" in result.stdout and "MACHINE_POLICY_OK" in result.stdout
    assert "TLS_MODE=legacy" in result.stdout
    assert "LEGACY_TLS_UNVERIFIED=YES TLS_CERT_VERIFIED=NO" in result.stdout
    assert "test-token" not in result.stdout + result.stderr


def test_preflight_strict_no_ca_blocks(tmp_path):
    for mode in (None, "strict"):
        result = _tls_preflight(tmp_path, mode)
        assert result.returncode != 0
        assert "CA_NOT_PROVISIONED" in result.stdout + result.stderr
        assert "SECRETS_OK" not in result.stdout
        assert "LEGACY_TLS_UNVERIFIED" not in result.stdout
        assert "test-token" not in result.stdout + result.stderr


def test_compose_tls_mode_passthrough(tmp_path, monkeypatch):
    """Execute the env-file generator and real Config loader across the Compose key."""
    from .. import config as config_module

    script = _deploy_sh()
    functions = []
    for name in ("require_github_repos", "build_env_file"):
        match = re.search(rf"{name}\(\)\s*\{{(.*?)\n\}}", script, re.DOTALL)
        assert match
        functions.append(name + "() {\n" + match.group(1) + "\n}\n")
    wrapper = tmp_path / "generate-env.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nRUNTIME_UID=65532\nRUNTIME_GID=65532\n"
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        + "".join(functions) + "build_env_file\n", encoding="utf-8",
    )
    declared = _compose()["services"]["deploy-approval"]["environment"]["AGENT_CORE_TLS_MODE"]
    # Resolve only this scalar, using the variable and default declared by Compose.
    interpolation = re.fullmatch(r"\$\{([A-Z_]+)-([^}]+)\}", declared)
    assert interpolation and interpolation.groups() == ("AGENT_CORE_TLS_MODE", "strict")
    variable, default = interpolation.groups()
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(_secrets_yaml(None, agent_core_ca_pem={}), encoding="utf-8")
    loader = config_module._load_secrets_config
    monkeypatch.setattr(config_module, "_load_secrets_config", lambda path: loader(str(secrets)))
    for supplied in (None, "legacy", "strict", "", "compatibility"):
        env = os.environ.copy()
        env["TMPDIR"] = str(tmp_path)
        env["GITHUB_REPOS"] = "example/main,example/drivers"
        if supplied is None:
            env.pop(variable, None)
        else:
            env[variable] = supplied
        result = subprocess.run(["bash", str(wrapper)], env=env, capture_output=True, text=True, timeout=30)
        if supplied in ("", "compatibility"):
            assert result.returncode != 0
            assert "AGENT_CORE_TLS_MODE" in result.stderr
            continue
        assert result.returncode == 0, result.stderr
        generated = Path(result.stdout.strip())
        try:
            assert stat.S_IMODE(generated.stat().st_mode) == 0o600
            values = dict(line.split("=", 1) for line in generated.read_text().splitlines())
            effective = values.get(variable, default)
            assert effective == (supplied or "strict")
            assert values["DEPLOY_APPROVAL_RUNTIME_UID"] == "65532"
            monkeypatch.setenv(variable, effective)
            monkeypatch.setenv("GITHUB_REPOS", values["GITHUB_REPOS"])
            cfg = config_module.load_config()
            assert cfg.agent_core_tls_mode == effective
            assert cfg.agent_core_tokens == {"test-machine": "test-token"}
            assert cfg.github_repos == ["example/main", "example/drivers"]
            assert cfg.cos_secret_key == "TEST_FAKE_SECRET_KEY_AAAA"
        finally:
            generated.unlink()


def test_missing_machines_file_rejected(tmp_path, offline_ca_pem):
    """Missing machines.yaml must cause deploy.sh validation to fail."""
    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    ok_secrets = tmp_path / "secrets.yaml"
    ok_secrets.write_text(_secrets_yaml(offline_ca_pem))

    wrapper = tmp_path / "test_missing.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SECRETS_FILE="' + str(ok_secrets) + '"\n'
        'MACHINES_FILE="' + str(tmp_path / "nonexistent.yaml") + '"\n'
        'PERSISTENT_DIR="' + str(tmp_path) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n"
        + secrets_body + "\n}\n"
        "require_machine_policy() {\n"
        + machine_body + "\n}\n"
        "require_machine_policy\n"
        "require_secrets\n"
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    expected = f"machine policy file must be a regular file: {tmp_path / 'nonexistent.yaml'}"
    assert expected.lower() in combined
    assert "ca_not_provisioned" not in combined
    assert "SECRETS_OK" not in r.stdout


def test_missing_secrets_file_rejected(tmp_path):
    """Missing secrets.yaml must cause deploy.sh validation to fail."""
    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    ok_machines = tmp_path / "machines.yaml"
    ok_machines.write_text(_machines_yaml())

    wrapper = tmp_path / "test_missing.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SECRETS_FILE="' + str(tmp_path / "nonexistent.yaml") + '"\n'
        'MACHINES_FILE="' + str(ok_machines) + '"\n'
        'PERSISTENT_DIR="' + str(tmp_path) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n"
        + secrets_body + "\n}\n"
        "require_machine_policy() {\n"
        + machine_body + "\n}\n"
        "require_machine_policy\n"
        "require_secrets\n"
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "missing" in combined or "not a regular file" in combined


# ── 4. Permission-denied rejection ──────────────────────────────────────

def test_unreadable_file_rejected(tmp_path, offline_ca_pem):
    """Unreadable machines.yaml must be rejected."""
    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    bad = tmp_path / "machines.yaml"
    bad.write_text(_machines_yaml())
    bad.chmod(0o000)

    ok_secrets = tmp_path / "secrets.yaml"
    ok_secrets.write_text(_secrets_yaml(offline_ca_pem))

    wrapper = tmp_path / "test_perm.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SECRETS_FILE="' + str(ok_secrets) + '"\n'
        'MACHINES_FILE="' + str(bad) + '"\n'
        'PERSISTENT_DIR="' + str(tmp_path) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n"
        + secrets_body + "\n}\n"
        "require_machine_policy() {\n"
        + machine_body + "\n}\n"
        "require_machine_policy\n"
        "require_secrets\n"
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    bad.chmod(0o644)
    assert r.returncode != 0


# ── 5. Placeholder value rejection ─────────────────────────────────────

def test_placeholder_in_machines_node_id_rejected(tmp_path, offline_ca_pem):
    """machines.yaml with <placeholder> in node_id must be rejected."""
    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    bad = tmp_path / "machines.yaml"
    bad.write_text(
        "version: 1\n"
        "machines:\n"
        "  test-machine:\n"
        '    node_id: "<NODE_ID>"\n'
        "    owners: [\"owner\"]\n"
        "    node_host: 10.0.0.1\n"
        "    targets: [\"perception\"]\n"
        "    platforms: [\"linux/arm64\"]\n"
        "    variants: [\"1.0\"]\n"
        "    driver_paths: []\n"
    )
    ok_secrets = tmp_path / "secrets.yaml"
    ok_secrets.write_text(_secrets_yaml(offline_ca_pem))

    wrapper = tmp_path / "test_placeholder.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SECRETS_FILE="' + str(ok_secrets) + '"\n'
        'MACHINES_FILE="' + str(bad) + '"\n'
        'PERSISTENT_DIR="' + str(tmp_path) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n"
        + secrets_body + "\n}\n"
        "require_machine_policy() {\n"
        + machine_body + "\n}\n"
        "require_machine_policy\n"
        "require_secrets\n"
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "field 'node_id' contains an unreplaced placeholder" in combined
    assert "ca_not_provisioned" not in combined
    assert "SECRETS_OK" not in r.stdout


def test_placeholder_in_secrets_secret_id_rejected(tmp_path):
    """secrets.yaml with <placeholder> in secret_id must be rejected."""
    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    bad = tmp_path / "secrets.yaml"
    bad.write_text(
        "version: 1\n"
        "cos:\n"
        "  region: ap-beijing\n"
        "  bucket: testbucket123\n"
        '  secret_id: "<SECRET_ID>"\n'
        '  secret_key: "<SECRET_KEY>"\n'
        "review_comment_trust:\n"
        '  author_id: "7950763"\n'
        '  author_login: "kentcyq"\n'
        "agent_core_tokens:\n"
        "  m1: token\n"
    )

    ok_machines = tmp_path / "machines.yaml"
    ok_machines.write_text(_machines_yaml())

    wrapper = tmp_path / "test_placeholder.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SECRETS_FILE="' + str(bad) + '"\n'
        'MACHINES_FILE="' + str(ok_machines) + '"\n'
        'PERSISTENT_DIR="' + str(tmp_path) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n"
        + secrets_body + "\n}\n"
        "require_machine_policy() {\n"
        + machine_body + "\n}\n"
        "require_machine_policy\n"
        "require_secrets\n"
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode != 0
    assert "placeholder" in (r.stdout + r.stderr).lower()


# ── 6. Cross-SHA persistence ───────────────────────────────────────────

def test_persistent_dir_independent_of_release_dir(tmp_path):
    text = _deploy_sh()
    assert 'PERSISTENT_DIR="${DEPLOY_APPROVAL_CONFIG_DIR:' in text
    assert 'MACHINES_FILE="${PERSISTENT_DIR}/machines.yaml"' in text
    assert 'SECRETS_FILE="${PERSISTENT_DIR}/secrets.yaml"' in text

    compose = _compose()
    svc = compose["services"]["deploy-approval"]
    for v in svc.get("volumes", []):
        if isinstance(v, str) and "machines.yaml" in v:
            assert "DEPLOY_APPROVAL_CONFIG_DIR" in v
        if isinstance(v, str) and "secrets.yaml" in v:
            assert "DEPLOY_APPROVAL_CONFIG_DIR" in v
    # Must not contain hardcoded server path
    compose_text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "/var/lib/deploy-approval" not in compose_text


# ── 7. Read-only mount verification ────────────────────────────────────

def test_compose_read_only_flag():
    compose = _compose()
    svc = compose["services"]["deploy-approval"]
    assert svc.get("read_only") is True


def test_no_write_mounts_for_config():
    compose = _compose()
    svc = compose["services"]["deploy-approval"]
    volumes = svc.get("volumes", [])
    for v in volumes:
        if isinstance(v, str) and ("/run/deploy-approval/" in v):
            assert v.endswith(":ro")


# ── 8. COS path protocol ───────────────────────────────────────────────

def test_cos_path_protocol_phanthymotus():
    from agents.deploy_approval.cos_client import CosClient
    from agents.deploy_approval.config import Config

    cfg = Config()
    client = CosClient(cfg, _fake=True)
    head_sha = "a" * 40
    key = client.build_object_key(
        "4paradigm/phanthymotus", 42, head_sha,
        now=datetime.datetime(2026, 9, 17, tzinfo=datetime.timezone.utc).astimezone(
            datetime.timezone(datetime.timedelta(hours=8))
        ),
    )
    assert key == "phanthymotus_pr/phanthymotus/2026-09/2026-09-17/pr-42/evidence-" + head_sha + ".log.gz"


def test_cos_path_protocol_phanthymotus_driver():
    from agents.deploy_approval.cos_client import CosClient
    from agents.deploy_approval.config import Config

    cfg = Config()
    client = CosClient(cfg, _fake=True)
    head_sha = "b" * 40
    key = client.build_object_key(
        "4paradigm/phanthymotus-driver", 7, head_sha,
        now=datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc).astimezone(
            datetime.timezone(datetime.timedelta(hours=8))
        ),
    )
    assert key == "phanthymotus_pr/phanthymotus-driver/2026-10/2026-10-01/pr-7/evidence-" + head_sha + ".log.gz"


def test_cos_path_rejects_unsupported_repo():
    from agents.deploy_approval.cos_client import CosClient
    from agents.deploy_approval.config import Config

    cfg = Config()
    client = CosClient(cfg, _fake=True)
    head_sha = "c" * 40
    with pytest.raises(ValueError, match="not configured for COS evidence"):
        client.build_object_key("other/repo", 1, head_sha)


# ── 9. Config regression — production COS guard ────────────────────────

def test_config_rejects_placeholder_in_cos_region():
    from agents.deploy_approval.config import Config, validate_config

    cfg = Config(cos_region="<REGION>", agent_core_tokens={"test": "token"})
    with pytest.raises(ValueError, match="placeholder"):
        validate_config(cfg)


def test_config_rejects_placeholder_in_cos_bucket():
    from agents.deploy_approval.config import Config, validate_config

    cfg = Config(cos_region="ap-beijing", cos_bucket="<BUCKET>",
                 cos_secret_id="TEST_FAKE_SECRET_ID_AAAA",
                 cos_secret_key="TEST_FAKE_SECRET_KEY_AAAA",
                 agent_core_tokens={"test": "token"})
    with pytest.raises(ValueError, match="placeholder"):
        validate_config(cfg)


def test_config_rejects_placeholder_in_cos_secret_id():
    from agents.deploy_approval.config import Config, validate_config

    cfg = Config(cos_region="ap-beijing", cos_bucket="testbucket123",
                 cos_secret_id="<SECRET_ID>",
                 cos_secret_key="TEST_FAKE_SECRET_KEY_AAAA",
                 agent_core_tokens={"test": "token"})
    with pytest.raises(ValueError, match="placeholder"):
        validate_config(cfg)


def test_config_rejects_placeholder_in_cos_secret_key():
    from agents.deploy_approval.config import Config, validate_config

    cfg = Config(cos_region="ap-beijing", cos_bucket="testbucket123",
                 cos_secret_id="TEST_FAKE_SECRET_ID_AAAA",
                 cos_secret_key="<SECRET_KEY>",
                 agent_core_tokens={"test": "token"})
    with pytest.raises(ValueError, match="placeholder"):
        validate_config(cfg)


def test_config_rejects_short_fake_cos_values():
    """Short fake values like 'test' must also be rejected after removing len-based bypass."""
    from agents.deploy_approval.config import Config, validate_config

    cfg = Config(
        cos_region="test",
        cos_bucket="test",
        cos_secret_id="test",
        cos_secret_key="test",
        agent_core_tokens={"test": "token"},
    )
    with pytest.raises(ValueError, match="invalid format"):
        validate_config(cfg)


def test_config_rejects_long_invalid_cos_region():
    from agents.deploy_approval.config import Config, validate_config

    cfg = Config(cos_region="invalid-region-name", cos_bucket="testbucket123",
                 cos_secret_id="TEST_FAKE_SECRET_ID_AAAA",
                 cos_secret_key="TEST_FAKE_SECRET_KEY_AAAA",
                 agent_core_tokens={"test": "token"})
    with pytest.raises(ValueError, match="invalid format"):
        validate_config(cfg)


# ── 10. deploy.sh preflight validation ─────────────────────────────────

def test_preflight_command_exists_in_case():
    text = _deploy_sh()
    lines = text.split("\n")
    in_case = False
    for line in lines:
        if 'case "${1:-up}"' in line:
            in_case = True
        if in_case and "preflight)" in line:
            return
    assert False, "preflight must be in the case dispatch"


def test_preflight_validates_both_config_files():
    text = _deploy_sh()
    assert "PREFLIGHT_MACHINES" in text
    assert "PREFLIGHT_SECRETS" in text
    assert "require_machine_policy" in text
    assert "require_secrets" in text


def test_preflight_matches_compose_mount_paths():
    text = _deploy_sh()
    assert "/run/deploy-approval/machines.yaml" in text
    assert "/run/deploy-approval/secrets.yaml" in text


# ── 11. Symlink rejection ──────────────────────────────────────────────

def test_symlink_machines_yaml_rejected(tmp_path, offline_ca_pem):
    """machines.yaml as symlink must be rejected."""
    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    real_file = tmp_path / "machines_real.yaml"
    real_file.write_text(_machines_yaml())
    link = tmp_path / "machines.yaml"
    link.symlink_to(real_file)

    ok_secrets = tmp_path / "secrets.yaml"
    ok_secrets.write_text(_secrets_yaml(offline_ca_pem))

    wrapper = tmp_path / "test_symlink.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SECRETS_FILE="' + str(ok_secrets) + '"\n'
        'MACHINES_FILE="' + str(link) + '"\n'
        'PERSISTENT_DIR="' + str(tmp_path) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n"
        + secrets_body + "\n}\n"
        "require_machine_policy() {\n"
        + machine_body + "\n}\n"
        "require_machine_policy\n"
        "require_secrets\n"
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode != 0
    assert link.is_symlink() and link.is_file()
    combined = r.stdout + r.stderr
    assert f"Machine policy file must be a regular file: {link}" in combined
    assert "CA_NOT_PROVISIONED" not in combined
    assert "SECRETS_OK" not in r.stdout


# ── 12. New release SHA preservation ───────────────────────────────────

def test_config_paths_survive_release_directory_change():
    deploy_text = _deploy_sh()
    compose_text = COMPOSE_PATH.read_text()

    machines_line = deploy_text.split("MACHINES_FILE=")[1].split("\n")[0]
    secrets_line = deploy_text.split("SECRETS_FILE=")[1].split("\n")[0]
    assert "SCRIPT_DIR" not in machines_line
    assert "SCRIPT_DIR" not in secrets_line

    assert "./machines.yaml" not in compose_text
    assert "./secrets.yaml" not in compose_text
    assert "DEPLOY_APPROVAL_CONFIG_DIR" in compose_text
    assert "/var/lib/deploy-approval" not in compose_text


# ── 13. Different config directories ──────────────────────────────────

def test_two_different_config_directories(tmp_path, offline_ca_pem):
    """Two different config directories can each hold valid independent YAML."""
    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    dir1 = tmp_path / "config_dir_1"
    dir1.mkdir()
    (dir1 / "machines.yaml").write_text(_machines_yaml())
    (dir1 / "secrets.yaml").write_text(_secrets_yaml(offline_ca_pem))

    dir2 = tmp_path / "config_dir_2"
    dir2.mkdir()
    (dir2 / "machines.yaml").write_text(
        _machines_yaml(machines={"other-machine": {
            "node_id": "other-node", "owners": ["other-owner"],
            "node_host": "10.0.0.2", "targets": ["core"],
            "platforms": ["linux/amd64"], "variants": ["6.1"],
            "driver_paths": [],
        }})
    )
    (dir2 / "secrets.yaml").write_text(
        _secrets_yaml(offline_ca_pem, agent_core_tokens={"other-machine": "other-test-token"}, cos={
            "region": "ap-guangzhou",
            "bucket": "otherbucket456",
            "secret_id": "TEST_FAKE_SECRET_ID_BBBB",
            "secret_key": "TEST_FAKE_SECRET_KEY_BBBB",
        })
    )

    # dir1 should pass
    wrapper1 = tmp_path / "test_dir1.sh"
    wrapper1.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'SECRETS_FILE="' + str(dir1 / "secrets.yaml") + '"\n'
        'MACHINES_FILE="' + str(dir1 / "machines.yaml") + '"\n'
        'PERSISTENT_DIR="' + str(dir1) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n" + secrets_body + "\n}\n"
        "require_machine_policy() {\n" + machine_body + "\n}\n"
        "require_machine_policy\nrequire_secrets\n"
    )
    wrapper1.chmod(0o755)
    r1 = subprocess.run(["bash", str(wrapper1)], capture_output=True, text=True)
    assert r1.returncode == 0

    # dir2 should pass with different values
    wrapper2 = tmp_path / "test_dir2.sh"
    wrapper2.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'SECRETS_FILE="' + str(dir2 / "secrets.yaml") + '"\n'
        'MACHINES_FILE="' + str(dir2 / "machines.yaml") + '"\n'
        'PERSISTENT_DIR="' + str(dir2) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n" + secrets_body + "\n}\n"
        "require_machine_policy() {\n" + machine_body + "\n}\n"
        "require_machine_policy\nrequire_secrets\n"
    )
    wrapper2.chmod(0o755)
    r2 = subprocess.run(["bash", str(wrapper2)], capture_output=True, text=True)
    assert r2.returncode == 0
    assert "ap-guangzhou" in r2.stdout or "SECRETS_OK" in r2.stdout


# ── 14. DEPLOY_APPROVAL_CONFIG_DIR missing rejects ───────────────────

def test_missing_config_dir_env_rejects_in_deploy_sh():
    """deploy.sh must fail when DEPLOY_APPROVAL_CONFIG_DIR is unset."""
    deploy_text = DEPLOY_SH_PATH.read_text(encoding="utf-8")
    # PERSISTENT_DIR must use :? syntax to fail on unset
    assert 'PERSISTENT_DIR="${DEPLOY_APPROVAL_CONFIG_DIR:' in deploy_text
    assert "?DEPLOY_APPROVAL_CONFIG_DIR" in deploy_text


def test_preflight_requires_config_dir(tmp_path):
    """preflight must fail when DEPLOY_APPROVAL_CONFIG_DIR is not set."""
    deploy_text = _deploy_sh()
    # The preflight function should reference DEPLOY_APPROVAL_CONFIG_DIR
    # and use :? or equivalent to fail when missing
    preflight_section = deploy_text.split("cmd_preflight()")[1].split("\n\ncmd_")[0]
    assert "DEPLOY_APPROVAL_CONFIG_DIR" in preflight_section
    assert ":?" in preflight_section or "is required" in preflight_section


def test_preflight_missing_env_var(tmp_path):
    """Running preflight without DEPLOY_APPROVAL_CONFIG_DIR must fail."""
    wrapper = tmp_path / "test_preflight_missing.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "unset DEPLOY_APPROVAL_CONFIG_DIR 2>/dev/null || true\n"
        'source "' + str(DEPLOY_SH_PATH) + '"\n'
        'cmd_preflight\n'
    )
    wrapper.chmod(0o755)
    # Run with no DEPLOY_APPROVAL_CONFIG_DIR set
    env = os.environ.copy()
    env.pop("DEPLOY_APPROVAL_CONFIG_DIR", None)
    r = subprocess.run(
        ["bash", str(wrapper)], capture_output=True, text=True, env=env,
    )
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "required" in combined


# ── 15. Relative path rejects ────────────────────────────────────────

def test_relative_config_dir_rejected_in_preflight(tmp_path, offline_ca_pem):
    """preflight must reject relative DEPLOY_APPROVAL_CONFIG_DIR even if the directory exists."""
    # Create a real relative directory
    rel_dir = tmp_path / "relative_config"
    rel_dir.mkdir()
    (rel_dir / "machines.yaml").write_text(_machines_yaml())
    (rel_dir / "machines.yaml").chmod(0o600)
    (rel_dir / "secrets.yaml").write_text(_secrets_yaml(offline_ca_pem))
    (rel_dir / "secrets.yaml").chmod(0o600)

    # Use a relative path (relative to tmp_path)
    rel_path = "relative_config"

    wrapper = tmp_path / "test_relative.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'export DEPLOY_APPROVAL_CONFIG_DIR="' + str(rel_path) + '"\n'
        'export GITHUB_APP_ID="12345"\n'
        'export GITHUB_INSTALLATION_ID="67890"\n'
        'export GITHUB_APP_PRIVATE_KEY_FILE="' + str(tmp_path / "key.pem") + '"\n'
        "set -- preflight\n"
        'source "' + str(DEPLOY_SH_PATH) + '"\n'
        'cmd_preflight\n'
    )
    wrapper.chmod(0o755)

    # Write a fake key file
    key_file = tmp_path / "key.pem"
    key_file.write_text("-----BEGIN PRIVATE KEY-----\nFAKE\n-----END PRIVATE KEY-----\n")
    key_file.chmod(0o600)

    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode != 0, (
        f"preflight must reject relative path even if directory exists: {r.stdout}{r.stderr}"
    )
    combined = (r.stdout + r.stderr).lower()
    assert "absolute" in combined or "relative" in combined or "required" in combined, (
        f"Error must mention absolute/relative path: {r.stdout}{r.stderr}"
    )


# ── 16. Non-existent directory rejects ───────────────────────────────

def test_nonexistent_config_dir_rejected_in_preflight(tmp_path):
    """preflight must fail when DEPLOY_APPROVAL_CONFIG_DIR points to missing dir."""
    wrapper = tmp_path / "test_nonexistent.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'export DEPLOY_APPROVAL_CONFIG_DIR="' + str(tmp_path / "nonexistent_xyz") + '"\n'
        'source "' + str(DEPLOY_SH_PATH) + '"\n'
        'cmd_preflight\n'
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "does not exist" in combined or "required" in combined


# ── 17. preflight failure does zero Docker modification ──────────────

def test_preflight_failure_zero_docker_modifications(tmp_path):
    """If preflight fails, no Docker containers or images are affected."""
    wrapper = tmp_path / "test_preflight_fail.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'export DEPLOY_APPROVAL_CONFIG_DIR="' + str(tmp_path / "nonexistent") + '"\n'
        'source "' + str(DEPLOY_SH_PATH) + '"\n'
        'cmd_preflight\n'
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode != 0
    # The preflight function must NOT call compose/build/up
    preflight_section = _deploy_sh().split("cmd_preflight()")[1].split("\n\ncmd_")[0]
    lines_with_compose_cmd = [
        l for l in preflight_section.splitlines()
        if "$COMPOSE" in l or "docker " in l or "docker-compose" in l
    ]
    assert len(lines_with_compose_cmd) == 0, \
        f"Preflight must not invoke docker: {lines_with_compose_cmd}"


# ── 18. Dev/test/prod config isolation ───────────────────────────────

def test_config_isolation_between_environments(tmp_path, offline_ca_pem):
    """Dev, test, and prod configs are independent and don't leak."""
    dev_dir = tmp_path / "dev"
    dev_dir.mkdir()
    (dev_dir / "machines.yaml").write_text(
        _machines_yaml(machines={"dev-machine": {
            "node_id": "dev-node", "owners": ["dev-owner"],
            "node_host": "192.168.1.1", "targets": ["perception"],
            "platforms": ["linux/arm64"], "variants": ["1.0"],
            "driver_paths": [],
        }})
    )
    (dev_dir / "secrets.yaml").write_text(
        _secrets_yaml(offline_ca_pem, agent_core_tokens={"dev-machine": "dev-test-token"},
                      cos={"region": "ap-beijing", "bucket": "dev-bucket-123",
                           "secret_id": "TEST_FAKE_SECRET_ID_CCCC", "secret_key": "TEST_FAKE_SECRET_KEY_CCCC"})
    )

    prod_dir = tmp_path / "prod"
    prod_dir.mkdir()
    (prod_dir / "machines.yaml").write_text(
        _machines_yaml(machines={"prod-machine": {
            "node_id": "prod-node", "owners": ["prod-owner"],
            "node_host": "10.0.0.100", "targets": ["core"],
            "platforms": ["linux/amd64"], "variants": ["6.1"],
            "driver_paths": [],
        }})
    )
    (prod_dir / "secrets.yaml").write_text(
        _secrets_yaml(offline_ca_pem, agent_core_tokens={"prod-machine": "prod-test-token"},
                      cos={"region": "ap-shanghai", "bucket": "prod-bucket-456",
                           "secret_id": "TEST_FAKE_SECRET_ID_DDDD", "secret_key": "TEST_FAKE_SECRET_KEY_DDDD"})
    )

    # Each directory validates independently
    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    for label, d in [("dev", dev_dir), ("prod", prod_dir)]:
        wrapper = tmp_path / f"test_{label}.sh"
        wrapper.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'SECRETS_FILE="' + str(d / "secrets.yaml") + '"\n'
            'MACHINES_FILE="' + str(d / "machines.yaml") + '"\n'
            'PERSISTENT_DIR="' + str(d) + '"\n'
            'die() { echo "ERROR: $*" >&2; exit 1; }\n'
            "require_secrets() {\n" + secrets_body + "\n}\n"
            "require_machine_policy() {\n" + machine_body + "\n}\n"
            "require_machine_policy\nrequire_secrets\n"
        )
        wrapper.chmod(0o755)
        r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
        assert r.returncode == 0, f"{label} config should be valid"


# ── 19. GitHub App key no leak ───────────────────────────────────────

def test_github_app_key_not_leaked_in_preflight_output(tmp_path, offline_ca_pem):
    """Preflight must not print GitHub App private key content."""
    wrapper = tmp_path / "test_key_leak.sh"
    key_file = tmp_path / "key.pem"
    key_file.write_text(
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\n"
        "fake-key-content-that-should-not-be-leaked\n"
        "-----END PRIVATE KEY-----\n"
    )
    key_file.chmod(0o600)

    dir_path = tmp_path / "config_key"
    dir_path.mkdir()
    (dir_path / "machines.yaml").write_text(_machines_yaml())
    (dir_path / "machines.yaml").chmod(0o600)
    (dir_path / "secrets.yaml").write_text(_secrets_yaml(offline_ca_pem))
    (dir_path / "secrets.yaml").chmod(0o600)

    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'export DEPLOY_APPROVAL_CONFIG_DIR="' + str(dir_path) + '"\n'
        'export GITHUB_APP_ID="12345"\n'
        'export GITHUB_INSTALLATION_ID="67890"\n'
        'export GITHUB_APP_PRIVATE_KEY_FILE="' + str(key_file) + '"\n'
        "set -- preflight\n"
        'source "' + str(DEPLOY_SH_PATH) + '"\n'
        'cmd_preflight\n'
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode == 0
    assert "MACHINE_POLICY_OK" in r.stdout
    assert "SECRETS_OK" in r.stdout
    assert "Preflight PASSED" in r.stdout
    # Neither part of the synthetic private-key payload may appear in output.
    combined = r.stdout + r.stderr
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcw" not in combined
    assert "fake-key-content-that-should-not-be-leaked" not in combined


# ── 20. Agent Core Token no leak ─────────────────────────────────────

def test_agent_core_token_not_leaked_in_output(tmp_path, offline_ca_pem):
    """Preflight and validation must not print Agent Core Token values."""
    dir_path = tmp_path / "config_token"
    dir_path.mkdir()
    machines = _machines_yaml(machines={"secret-machine": {
        "node_id": "secret-node", "owners": ["secret-owner"],
        "node_host": "10.0.0.50", "targets": ["actucore"],
        "platforms": ["linux/arm64"], "variants": ["5.11"],
        "driver_paths": [],
    }})
    (dir_path / "machines.yaml").write_text(machines)
    (dir_path / "secrets.yaml").write_text(
        _secrets_yaml(
            offline_ca_pem,
            cos={"region": "ap-beijing", "bucket": "tokentest123",
                 "secret_id": "TEST_FAKE_SECRET_ID_EEEE", "secret_key": "TEST_FAKE_SECRET_KEY_EEEE"},
            agent_core_tokens={"secret-machine": "SUPER-SECRET-TOKEN-VALUE-12345"},
        )
    )

    deploy_sh = _deploy_sh()
    secrets_body, machine_body = _extract_bodies(deploy_sh)

    wrapper = tmp_path / "test_token_leak.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'SECRETS_FILE="' + str(dir_path / "secrets.yaml") + '"\n'
        'MACHINES_FILE="' + str(dir_path / "machines.yaml") + '"\n'
        'PERSISTENT_DIR="' + str(dir_path) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n" + secrets_body + "\n}\n"
        "require_machine_policy() {\n" + machine_body + "\n}\n"
        "require_machine_policy\nrequire_secrets\n"
    )
    wrapper.chmod(0o755)
    r = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
    assert r.returncode == 0
    assert "SUPER-SECRET-TOKEN-VALUE-12345" not in (r.stdout + r.stderr)
    assert "MACHINE_POLICY_OK" in r.stdout
    assert "SECRETS_OK" in r.stdout


# ── 21. Compose validation with env var ──────────────────────────────

def test_compose_validates_with_config_dir_set():
    """docker-compose config must parse successfully with DEPLOY_APPROVAL_CONFIG_DIR."""
    import subprocess as sp
    r = sp.run(
        ["docker", "compose", "-f", str(COMPOSE_PATH), "config", "-q"],
        capture_output=True, text=True,
        env={**os.environ,
             "DEPLOY_APPROVAL_CONFIG_DIR": "/tmp/fake-config",
             "GITHUB_APP_PRIVATE_KEY_FILE": "/tmp/fake-key.pem"},
    )
    # Must parse successfully — no OR conditions that allow failure
    assert r.returncode == 0, (
        f"docker compose config failed: {r.stderr}"
    )


# ── 22. Source code does not auto-create directories ─────────────────

def test_source_does_not_auto_create_config_dir():
    """Source code must not auto-create the config directory on the host."""
    deploy_text = _deploy_sh()
    assert "mkdir -p" not in deploy_text or "preflight" in deploy_text.split("mkdir")[0]
    # The deploy.sh preflight checks for directory existence but doesn't create it
    preflight_section = deploy_text.split("cmd_preflight()")[1].split("\n\ncmd_")[0]
    assert "mkdir" not in preflight_section


def test_compose_port_loopback_default():
    """Inspect parsed production Compose and retain the existing hardening."""
    svc = _compose()["services"]["deploy-approval"]
    assert svc["ports"] == ["${DEPLOY_APPROVAL_BIND_IP:-127.0.0.1}:25001:25001"]
    assert svc["user"] == "${DEPLOY_APPROVAL_RUNTIME_UID:-65532}:${DEPLOY_APPROVAL_RUNTIME_GID:-65532}"
    assert svc["read_only"] is True
    assert "no-new-privileges:true" in svc["security_opt"]
    assert svc["cap_drop"] == ["ALL"]
    assert "network_mode" not in svc
    assert "deploy" not in svc or svc["deploy"].get("replicas", 1) == 1
    assert all(v.get("read_only") is True for v in svc["volumes"])
    assert all("docker.sock" not in str(v) for v in svc["volumes"])
    assert {v["target"] for v in svc["volumes"]} >= {
        "/run/deploy-approval/machines.yaml", "/run/deploy-approval/secrets.yaml"}
