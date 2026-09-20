"""Config tests (final alignment)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ..config import (Config,
    DEFAULT_GITHUB_REPOS,
    load_config,
    validate_config,
    _env_bool,
    _env_float,
    _env_int,
)


def test_config_defaults():
    c = Config(github_repos=["org/repo"], review_comment_author_id="7950763")
    assert c.host == "0.0.0.0"
    assert c.port == 25001
    assert c.poll_enabled is True
    assert c.machine_owners_file == "/run/deploy-approval/machines.yaml"


def test_validate_config_no_longer_requires_github_token():
    """GITHUB_TOKEN is no longer a config field; token comes from GitHub App provider."""
    validate_config(
        Config(
            github_repos=[
                "4paradigm/phanthymotus",
            ],
            registry="ccr.ccs.tencentyun.com",
            review_comment_author_id="7950763",
        )
    )


def test_validate_config_default_repos():
    """Default github_repos is the single production repository."""
    c = Config(review_comment_author_id="7950763")
    assert "4paradigm/phanthymotus" in c.github_repos
    # Explicit empty overrides defaults — must fail closed
    with pytest.raises(ValueError, match="GITHUB_REPOS is required"):
        validate_config(Config(github_repos=[], registry="ccr.ccs.tencentyun.com"))


@pytest.mark.parametrize(
    "repos,should_pass",
    [
        (list(DEFAULT_GITHUB_REPOS), True),
        ([DEFAULT_GITHUB_REPOS[0], "some/fork-repo"], False),
        ([DEFAULT_GITHUB_REPOS[0], DEFAULT_GITHUB_REPOS[0]], False),
        (["some/other"], False),
        ([], False),
        ([*DEFAULT_GITHUB_REPOS, "some/other"], False),
    ],
)
def test_github_repos_requires_exact_production_set(repos, should_pass):
    cfg = Config(
        github_repos=repos,
        registry="ccr.ccs.tencentyun.com",
        review_comment_author_id="7950763",
    )
    if should_pass:
        validate_config(cfg)
    else:
        with pytest.raises(ValueError):
            validate_config(cfg)


def test_validate_config_requires_webhook_secret():
    with pytest.raises(ValueError, match="GITHUB_WEBHOOK_SECRET is empty"):
        validate_config(
            Config(
                github_repos=[
                    "4paradigm/phanthymotus",
                ],
                webhook_enabled=True,
                review_comment_author_id="7950763",
            )
        )


def test_validate_config_requires_polling():
    with pytest.raises(ValueError, match="requires polling"):
        validate_config(
            Config(
                github_repos=[
                    "4paradigm/phanthymotus",
                ],
                webhook_enabled=True,
                poll_enabled=False,
                github_webhook_secret="secret",
                review_comment_author_id="7950763",
            )
        )


def test_env_int():
    os.environ["TEST_INT"] = "42"
    assert _env_int("TEST_INT", 1) == 42
    os.environ["TEST_INT"] = "0"
    with pytest.raises(ValueError):
        _env_int("TEST_INT", 1)
    del os.environ["TEST_INT"]


def test_env_bool():
    os.environ["TEST_BOOL"] = "true"
    assert _env_bool("TEST_BOOL") is True
    os.environ["TEST_BOOL"] = "false"
    assert _env_bool("TEST_BOOL") is False
    os.environ["TEST_BOOL"] = "invalid"
    with pytest.raises(ValueError):
        _env_bool("TEST_BOOL")
    del os.environ["TEST_BOOL"]


def test_env_float():
    os.environ["TEST_FLOAT"] = "3.5"
    assert _env_float("TEST_FLOAT", 1.0) == 3.5
    os.environ["TEST_FLOAT"] = "0"
    with pytest.raises(ValueError):
        _env_float("TEST_FLOAT", 1.0)
    del os.environ["TEST_FLOAT"]


def test_config_env_override(monkeypatch, tmp_path):
    """Explicit GITHUB_REPOS overrides the default."""
    monkeypatch.setenv("GITHUB_REPOS", "4paradigm/phanthymotus")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text('version: 1\ncos:\n  region: test\n  bucket: test\n  secret_id: test\n  secret_key: test\nreview_comment_trust:\n  author_id: "7950763"\n  author_login: "review-agent-bot"\n')
    import agents.deploy_approval.config as config_mod
    orig = config_mod._load_secrets_config
    def patched(path):
        return orig(str(secrets))
    monkeypatch.setattr(config_mod, "_load_secrets_config", patched)
    cfg = load_config()
    assert cfg.github_repos == ["4paradigm/phanthymotus"]



def test_config_env_empty_fails_closed(monkeypatch):
    """Explicit empty GITHUB_REPOS must fail closed."""
    monkeypatch.setenv("GITHUB_REPOS", "")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    with pytest.raises(ValueError, match="GITHUB_REPOS is required"):
        load_config()


def test_config_env_unset_uses_default(monkeypatch, tmp_path):
    """GITHUB_REPOS unset uses DEFAULT_GITHUB_REPOS."""
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    monkeypatch.delenv("GITHUB_REPOS", raising=False)
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text('version: 1\ncos:\n  region: test\n  bucket: test\n  secret_id: test\n  secret_key: test\nreview_comment_trust:\n  author_id: "7950763"\n  author_login: "review-agent-bot"\n')
    import agents.deploy_approval.config as config_mod
    orig = config_mod._load_secrets_config
    def patched(path):
        return orig(str(secrets))
    monkeypatch.setattr(config_mod, "_load_secrets_config", patched)
    cfg = load_config()
    assert "4paradigm/phanthymotus" in cfg.github_repos
    assert "4paradigm/phanthymotus-driver" not in cfg.github_repos


# ── migrated from test_v8_contract.py ──────────────────────────────────────

def test_default_supported_repo_is_production_phanthymotus():
    cfg = Config()
    assert cfg.github_repos == ["4paradigm/phanthymotus"]


def test_perception_review_variant_511_matches_canonical_machine_variant(tmp_path):
    from ..policy import load_machines
    import textwrap
    path = tmp_path / "machines.yaml"
    path.write_text(textwrap.dedent("""\
        version: 1
        machines:
          m1:
            node_id: node-1
            node_host: 127.0.0.1
            tls_peer_cert_file: /run/deploy-approval/certs/test-agent-core.pem
            owners: [owner1]
            targets: [perception]
            platforms: [linux/arm64]
            variants: [jetson-jp5.11]
    """))
    machines = load_machines(str(path))
    assert machines["m1"].variants == ["5.11"]


def test_perception_review_variant_61_matches_canonical_machine_variant(tmp_path):
    from ..policy import load_machines
    import textwrap
    path = tmp_path / "machines.yaml"
    path.write_text(textwrap.dedent("""\
        version: 1
        machines:
          m1:
            node_id: node-1
            node_host: 127.0.0.1
            tls_peer_cert_file: /run/deploy-approval/certs/test-agent-core.pem
            owners: [owner1]
            targets: [perception]
            platforms: [linux/arm64]
            variants: [jetson-jp6.1]
    """))
    machines = load_machines(str(path))
    assert machines["m1"].variants == ["6.1"]


def test_legacy_jetson_variant_is_normalized_once_or_rejected_explicitly(tmp_path):
    from ..policy import load_machines
    import textwrap
    path = tmp_path / "machines.yaml"
    path.write_text(textwrap.dedent("""\
        version: 1
        machines:
          m1:
            node_id: node-1
            node_host: 127.0.0.1
            tls_peer_cert_file: /run/deploy-approval/certs/test-agent-core.pem
            owners: [owner1]
            targets: [perception]
            platforms: [linux/arm64]
            variants: [jetson-jp5.11, jetson-jp5.11]
    """))
    machines = load_machines(str(path))
    assert machines["m1"].variants == ["5.11"]


def test_validate_hidden_state_accepts_approval_revoked():
    from ..github_state_proxy import _validate_hidden_state
    state = {
        "version": 1,
        "head_sha": "a" * 40,
        "status": "review-required",
        "review_evidence": {"build_comment_id": 1, "build_comment_updated_at": "2026-09-18T00:00:00Z", "commit_prefix": "abc1234", "resolved_head_sha": "a" * 40, "test_comment_id": 2, "code_review_comment_id": 3, "review_author_id": "7950763"},
        "components": [],
        "deployments": [],
        "approve_attempts": [],
        "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "case_results": {},
        "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "command": {"comment_id": 1, "kind": "approve_deploy", "phase": "completed", "args": {"machine": "m1", "actor": "owner1"}},
        "last_processed_comment_id": 1,
        "approval_revoked": True,
    }
    result = _validate_hidden_state(state)
    assert result["approval_revoked"] is True


def test_driver_paths_required_for_driver_machine(config):
    from ..policy import Policy
    import textwrap
    from pathlib import Path
    tmp_path = Path("/tmp/test_driver_paths_required")
    tmp_path.mkdir(exist_ok=True)
    machines_file = tmp_path / "machines.yaml"
    machines_file.write_text(textwrap.dedent("""\
        version: 1
        machines:
          m1:
            node_id: node-1
            node_host: 127.0.0.1
            tls_peer_cert_file: /run/deploy-approval/certs/test-agent-core.pem
            owners: [owner1]
            targets: [driver]
            platforms: [linux/arm64]
    """))
    # machine m1 has targets=[driver] but no driver_paths,
    # so load_machines should raise MachineLoadError
    from ..policy import MachineLoadError, load_machines
    with pytest.raises(MachineLoadError, match="driver_paths"):
        load_machines(str(machines_file))


def test_driver_paths_reject_string_scalar(config):
    from ..policy import MachineLoadError, load_machines
    import textwrap
    from pathlib import Path
    tmp_path = Path("/tmp/test_driver_paths_string")
    tmp_path.mkdir(exist_ok=True)
    machines_file = tmp_path / "machines.yaml"
    machines_file.write_text(textwrap.dedent("""\
        version: 1
        machines:
          m1:
            node_id: node-1
            node_host: 127.0.0.1
            tls_peer_cert_file: /run/deploy-approval/certs/test-agent-core.pem
            owners: [owner1]
            targets: [driver]
            platforms: [linux/arm64]
            driver_paths: "unitree/g1"
    """))
    with pytest.raises(MachineLoadError, match="driver_paths"):
        load_machines(str(machines_file))


def test_driver_paths_reject_absolute_parent_backslash_and_empty_segments(config):
    from ..policy import MachineLoadError, load_machines
    import textwrap
    from pathlib import Path
    tmp_path = Path("/tmp/test_driver_paths_bad")
    tmp_path.mkdir(exist_ok=True)
    machines_file = tmp_path / "machines.yaml"
    machines_file.write_text(textwrap.dedent("""\
        version: 1
        machines:
          m1:
            node_id: node-1
            node_host: 127.0.0.1
            tls_peer_cert_file: /run/deploy-approval/certs/test-agent-core.pem
            owners: [owner1]
            targets: [driver]
            platforms: [linux/arm64]
            driver_paths:
              - "unitree/g1"
              - "/absolute/path"
              - "../parent"
              - "path\\backslash"
              - "a//double"
              - ""
    """))
    with pytest.raises(MachineLoadError, match="driver_paths"):
        load_machines(str(machines_file))


def test_driver_paths_are_trimmed_deduped_and_exact_case_preserved(config):
    from ..policy import load_machines
    import textwrap
    from pathlib import Path
    tmp_path = Path("/tmp/test_driver_paths_trim")
    tmp_path.mkdir(exist_ok=True)
    machines_file = tmp_path / "machines.yaml"
    machines_file.write_text(textwrap.dedent("""\
        version: 1
        machines:
          m1:
            node_id: node-1
            node_host: 127.0.0.1
            tls_peer_cert_file: /run/deploy-approval/certs/test-agent-core.pem
            owners: [owner1]
            targets: [driver]
            platforms: [linux/arm64]
            driver_paths:
              - "  unitree/g1  "
              - "unitree/g1"
    """))
    p = load_machines(str(machines_file))
    assert p["m1"].driver_paths == ["unitree/g1"]
