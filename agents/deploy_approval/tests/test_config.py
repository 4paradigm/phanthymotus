"""Config tests (final alignment)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ..config import (
    Config,
    FORK_TEST_GITHUB_REPOS,
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
                "4paradigm/phanthymotus-driver",
            ],
            deploy_approval_public_base_url="https://deploy.example",
            github_oauth_client_id="test-client",
            github_oauth_client_secret="test-secret",
            registry="ccr.ccs.tencentyun.com",
        )
    )


def test_validate_config_default_repos():
    """Default github_repos includes the two official repos."""
    c = Config(review_comment_author_id="7950763")
    assert "4paradigm/phanthymotus" in c.github_repos
    assert "4paradigm/phanthymotus-driver" in c.github_repos
    # Explicit empty overrides defaults — must fail closed
    with pytest.raises(ValueError, match="GITHUB_REPOS is required"):
        validate_config(Config(github_repos=[], registry="ccr.ccs.tencentyun.com"))


@pytest.mark.parametrize(
    "repos,fork_test_mode,should_pass",
    [
        (list(DEFAULT_GITHUB_REPOS), False, True),
        (list(FORK_TEST_GITHUB_REPOS), True, True),
        (list(FORK_TEST_GITHUB_REPOS), False, False),
        (list(DEFAULT_GITHUB_REPOS), True, False),
        ([DEFAULT_GITHUB_REPOS[0], "some/fork-repo"], False, False),
        ([DEFAULT_GITHUB_REPOS[0], DEFAULT_GITHUB_REPOS[0]], False, False),
        (["some/other"], False, False),
        ([], False, False),
        ([*DEFAULT_GITHUB_REPOS, "some/other"], False, False),
    ],
)
def test_fork_test_mode_requires_exact_repo_pair(repos, fork_test_mode, should_pass):
    cfg = Config(
        github_repos=repos,
        fork_test_mode=fork_test_mode,
        deploy_approval_public_base_url="https://deploy.example",
        github_oauth_client_id="test-client",
        github_oauth_client_secret="test-secret",
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
                    "4paradigm/phanthymotus-driver",
                ],
                deploy_approval_public_base_url="https://deploy.example",
                github_oauth_client_id="test-client",
                github_oauth_client_secret="test-secret",
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
                    "4paradigm/phanthymotus-driver",
                ],
                deploy_approval_public_base_url="https://deploy.example",
                github_oauth_client_id="test-client",
                github_oauth_client_secret="test-secret",
                poll_enabled=False,
                webhook_enabled=True,
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


def test_config_env_override(monkeypatch):
    """Explicit GITHUB_REPOS overrides the default."""
    monkeypatch.setenv("GITHUB_REPOS", "4paradigm/phanthymotus,4paradigm/phanthymotus-driver")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    cfg = load_config()
    assert cfg.github_repos == ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]


def test_config_env_fork_test_mode_requires_fork_pair(monkeypatch):
    """Fork test mode only allows exact Haohao-end/phanthymotus — no driver."""
    monkeypatch.setenv("GITHUB_REPOS", "Haohao-end/phanthymotus")
    monkeypatch.setenv("DEPLOY_APPROVAL_FORK_TEST_MODE", "true")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    cfg = load_config()
    assert cfg.fork_test_mode is True
    assert cfg.github_repos == ["Haohao-end/phanthymotus"]


def test_config_env_empty_fails_closed(monkeypatch):
    """Explicit empty GITHUB_REPOS must fail closed."""
    monkeypatch.setenv("GITHUB_REPOS", "")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    with pytest.raises(ValueError, match="GITHUB_REPOS is required"):
        load_config()


def test_config_env_unset_uses_default(monkeypatch):
    """GITHUB_REPOS unset uses DEFAULT_GITHUB_REPOS."""
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    monkeypatch.delenv("GITHUB_REPOS", raising=False)
    cfg = load_config()
    assert "4paradigm/phanthymotus" in cfg.github_repos
    assert "4paradigm/phanthymotus-driver" in cfg.github_repos


# ── migrated from test_v8_contract.py ──────────────────────────────────────

def test_default_supported_repos_include_both_real_repositories():
    cfg = Config()
    assert cfg.github_repos == ["4paradigm/phanthymotus", "4paradigm/phanthymotus-driver"]


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
    p = Policy(config)
    p.machines = {"m1": p._load_machine_info(str(tmp_path), "m1")}
    with pytest.raises(ValueError, match="driver_paths"):
        p.validate_machine_for_targets("m1", {"driver"})


def test_driver_paths_reject_string_scalar(config):
    from ..policy import Policy
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
    p = Policy(config)
    with pytest.raises(ValueError, match="driver_paths"):
        p.load_machines(str(machines_file))


def test_driver_paths_reject_absolute_parent_backslash_and_empty_segments(config):
    from ..policy import Policy
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
    p = Policy(config)
    with pytest.raises(ValueError, match="driver_paths"):
        p.load_machines(str(machines_file))


def test_driver_paths_are_trimmed_deduped_and_exact_case_preserved(config):
    from ..policy import Policy
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
    p = Policy(config)
    p.load_machines(str(machines_file))
    assert p.machines["m1"].driver_paths == ["unitree/g1"]
