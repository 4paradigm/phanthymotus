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
    _env_int,
)


def test_config_defaults():
    c = Config(github_repos=["org/repo"], review_comment_author_id="7950763", review_comment_author_login="kentcyq")
    assert c.host == "0.0.0.0"
    assert c.port == 25001
    assert c.poll_enabled is True
    assert c.machine_owners_file == "/run/deploy-approval/machines.yaml"


def test_validate_config_no_longer_requires_github_token():
    """GITHUB_TOKEN is no longer a config field; token comes from GitHub App provider."""
    validate_config(
        Config(
            github_repos=list(DEFAULT_GITHUB_REPOS),
            review_comment_author_id="7950763",
            review_comment_author_login="kentcyq",
            agent_core_tokens={"test": "token"},
            cos_region="ap-beijing",
            cos_bucket="testbucket123",
            cos_secret_id="TEST_FAKE_SECRET_ID_AAAA",
            cos_secret_key="TEST_FAKE_SECRET_KEY_AAAA",
        )
    )


def test_validate_config_default_repos():
    """Default github_repos contains both production repositories."""
    c = Config(review_comment_author_id="7950763", review_comment_author_login="kentcyq", agent_core_tokens={"test": "token"})
    assert "4paradigm/phanthymotus" in c.github_repos
    assert "4paradigm/phanthymotus-driver" in c.github_repos
    # Explicit empty overrides defaults — must fail closed
    with pytest.raises(ValueError, match="GITHUB_REPOS is required"):
        validate_config(Config(github_repos=[], agent_core_tokens={"test": "token"}))


@pytest.mark.parametrize(
    "repos,should_pass",
    [
        (list(DEFAULT_GITHUB_REPOS), True),
        (list(reversed(DEFAULT_GITHUB_REPOS)), True),
        (["4paradigm/phanthymotus"], True),
        (["4paradigm/phanthymotus-driver"], True),
        (["4paradigm/phanthymotus", "4paradigm/phanthymotus"], False),
        (["4paradigm/phanthymotus", "some/fork-repo"], True),
        (["some/other"], True),
        ([], False),
        ([*DEFAULT_GITHUB_REPOS, "some/other"], True),
    ],
)
def test_github_repos_validates_configured_runtime_repo_set(repos, should_pass):
    cfg = Config(
        github_repos=repos,
        agent_core_tokens={"test": "token"},
        review_comment_author_id="7950763",
        review_comment_author_login="kentcyq",
        cos_region="ap-beijing",
        cos_bucket="testbucket123",
        cos_secret_id="TEST_FAKE_SECRET_ID_AAAA",
        cos_secret_key="TEST_FAKE_SECRET_KEY_AAAA",
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
                github_repos=list(DEFAULT_GITHUB_REPOS),
                webhook_enabled=True,
                review_comment_author_id="7950763",
                review_comment_author_login="kentcyq",
                agent_core_tokens={"test-machine": "test-token"},
            )
        )


def test_validate_config_requires_polling():
    with pytest.raises(ValueError, match="requires polling"):
        validate_config(
            Config(
                github_repos=list(DEFAULT_GITHUB_REPOS),
                webhook_enabled=True,
                poll_enabled=False,
                github_webhook_secret="secret",
                review_comment_author_id="7950763",
                review_comment_author_login="kentcyq",
                agent_core_tokens={"test-machine": "test-token"},
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


def test_config_env_override(monkeypatch, tmp_path):
    """Explicit GITHUB_REPOS overrides the default."""
    monkeypatch.setenv("GITHUB_REPOS", "4paradigm/phanthymotus,4paradigm/phanthymotus-driver")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text('version: 1\ncos:\n  region: ap-beijing\n  bucket: testbucket123\n  secret_id: TEST_FAKE_SECRET_ID_AAAA\n  secret_key: TEST_FAKE_SECRET_KEY_AAAA\nreview_comment_trust:\n  author_id: "7950763"\n  author_login: "kentcyq"\nagent_core_tokens:\n  test-machine: test-token\n')
    import agents.deploy_approval.config as config_mod
    orig = config_mod._load_secrets_config
    def patched(path):
        return orig(str(secrets))
    monkeypatch.setattr(config_mod, "_load_secrets_config", patched)
    cfg = load_config()
    assert cfg.github_repos == list(DEFAULT_GITHUB_REPOS)



def test_config_env_empty_fails_closed(monkeypatch):
    """Explicit empty GITHUB_REPOS must fail closed."""
    monkeypatch.setenv("GITHUB_REPOS", "")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    with pytest.raises(ValueError, match="GITHUB_REPOS is required"):
        load_config()


def test_config_env_unset_uses_default(monkeypatch, tmp_path):
    """GITHUB_REPOS unset uses DEFAULT_GITHUB_REPOS."""
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.delenv("GITHUB_REPOS", raising=False)
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text('version: 1\ncos:\n  region: ap-beijing\n  bucket: testbucket123\n  secret_id: TEST_FAKE_SECRET_ID_AAAA\n  secret_key: TEST_FAKE_SECRET_KEY_AAAA\nreview_comment_trust:\n  author_id: "7950763"\n  author_login: "kentcyq"\nagent_core_tokens:\n  test-machine: test-token\n')
    import agents.deploy_approval.config as config_mod
    orig = config_mod._load_secrets_config
    def patched(path):
        return orig(str(secrets))
    monkeypatch.setattr(config_mod, "_load_secrets_config", patched)
    cfg = load_config()
    assert "4paradigm/phanthymotus" in cfg.github_repos
    assert "4paradigm/phanthymotus-driver" in cfg.github_repos


# ── migrated from test_v8_contract.py ──────────────────────────────────────

def test_default_supported_repos_are_production_main_and_driver():
    cfg = Config()
    assert cfg.github_repos == list(DEFAULT_GITHUB_REPOS)


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
            owners: [owner1]
            targets: [perception]
            platforms: [linux/arm64]
            variants: [jetson-jp5.11, jetson-jp5.11]
    """))
    machines = load_machines(str(path))
    assert machines["m1"].variants == ["5.11"]


def test_driver_paths_required_for_driver_machine(config):
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
            owners: [owner1]
            targets: [driver]
            platforms: [linux/arm64]
            driver_paths:
              - "  unitree/g1  "
              - "unitree/g1"
    """))
    p = load_machines(str(machines_file))
    assert p["m1"].driver_paths == ["unitree/g1"]


# ── Review Agent canonical identity validation tests ───────────────────

def test_validate_config_accepts_canonical_identity():
    """A. canonical author_id=7950763, author_login=kentcyq => PASS."""
    validate_config(
        Config(
            github_repos=list(DEFAULT_GITHUB_REPOS),
            review_comment_author_id="7950763",
            review_comment_author_login="kentcyq",
            agent_core_tokens={"test": "token"},
            cos_region="ap-beijing",
            cos_bucket="testbucket123",
            cos_secret_id="TEST_FAKE_SECRET_ID_AAAA",
            cos_secret_key="TEST_FAKE_SECRET_KEY_AAAA",
        )
    )


def test_runtime_review_rejects_unconfigured_author():
    """Actual comments must match BOTH fields of the configured identity."""
    from ..review_comment_parser import extract_review_evidence
    from .test_review_agent_contract import _make_comment, _BUILD_COMMENT_BODY, _CODE_REVIEW_BODY

    def evidence_for(author_id, author_login):
        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, author_id=author_id, author_login=author_login),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z",
                          author_id=author_id, author_login=author_login),
        ]
        return extract_review_evidence(comments, "7950763", "kentcyq")

    assert evidence_for("7950763", "kentcyq") is not None
    assert evidence_for("184792454", "Haohao-end") is None


def test_runtime_review_rejects_correct_id_wrong_login():
    """Actual comments must match BOTH fields of the configured identity."""
    from ..review_comment_parser import extract_review_evidence
    from .test_review_agent_contract import _make_comment, _BUILD_COMMENT_BODY, _CODE_REVIEW_BODY

    def evidence_for(author_id, author_login):
        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, author_id=author_id, author_login=author_login),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z",
                          author_id=author_id, author_login=author_login),
        ]
        return extract_review_evidence(comments, "7950763", "kentcyq")

    assert evidence_for("7950763", "kentcyq") is not None
    assert evidence_for("7950763", "wrong-login") is None


def test_runtime_review_rejects_wrong_id_correct_login():
    """Actual comments must match BOTH fields of the configured identity."""
    from ..review_comment_parser import extract_review_evidence
    from .test_review_agent_contract import _make_comment, _BUILD_COMMENT_BODY, _CODE_REVIEW_BODY

    def evidence_for(author_id, author_login):
        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, author_id=author_id, author_login=author_login),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z",
                          author_id=author_id, author_login=author_login),
        ]
        return extract_review_evidence(comments, "7950763", "kentcyq")

    assert evidence_for("7950763", "kentcyq") is not None
    assert evidence_for("12345", "kentcyq") is None


def test_load_config_reads_canonical_secrets(tmp_path, monkeypatch):
    """E. load_config reads 7950763 / kentcyq from secrets.yaml."""
    import agents.deploy_approval.config as config_mod
    orig = config_mod._load_secrets_config


    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(
        'version: 1\n'
        'cos:\n'
        '  region: ap-beijing\n'
        '  bucket: testbucket123\n'
        '  secret_id: TEST_FAKE_SECRET_ID_AAAA\n'
        '  secret_key: TEST_FAKE_SECRET_KEY_AAAA\n'
        'review_comment_trust:\n'
        '  author_id: "7950763"\n'
        '  author_login: "kentcyq"\n'
        'agent_core_tokens:\n'
        '  test-machine: test-token\n'
    )

    def patched(path):
        return orig(str(secrets))

    config_mod._load_secrets_config = patched
    try:
        cfg = load_config()
        assert cfg.review_comment_author_id == "7950763"
        assert cfg.review_comment_author_login == "kentcyq"
    finally:
        config_mod._load_secrets_config = orig


def test_no_env_var_override_for_review_trust():
    """F. No REVIEW_COMMENT_AUTHOR_ID / REVIEW_COMMENT_AUTHOR_LOGIN env contract.

    Review Agent trust is ONLY read from secrets.yaml review_comment_trust.
    There must be no env var override mechanism.
    """
    import agents.deploy_approval.config as config_mod

    # The config module must NOT define REVIEW_COMMENT_AUTHOR_ID or
    # REVIEW_COMMENT_AUTHOR_LOGIN as env var overrides.
    assert not hasattr(config_mod, "REVIEW_COMMENT_AUTHOR_ID") or not isinstance(
        getattr(config_mod, "REVIEW_COMMENT_AUTHOR_ID", None), str
    )
    # More precisely: check that load_config does not read these env vars.
    # We verify by checking the source code doesn't contain os.getenv calls
    # for these variable names in the review trust path.
    import inspect
    source = inspect.getsource(config_mod.load_config)
    assert "REVIEW_COMMENT_AUTHOR_ID" not in source
    assert "REVIEW_COMMENT_AUTHOR_LOGIN" not in source


# ── deploy.sh review_comment_trust contract regression ─────────────────

def test_deploy_sh_validates_runtime_review_agent_trust():
    """deploy.sh validates protected runtime identity without a hardcoded user."""
    ROOT = Path(__file__).parent.parent.parent.parent
    deploy_sh = ROOT / "deploy" / "deploy-approval" / "deploy.sh"
    text = deploy_sh.read_text(encoding="utf-8")
    assert "review_comment_trust.author_id must be a positive decimal string" in text
    assert "review_comment_trust.author_login must be a valid GitHub login" in text
    assert '"7950763"' not in text and '"kentcyq"' not in text


def test_deploy_sh_rejects_wrong_trust(tmp_path):
    """7. deploy.sh wrong ID / wrong login must fail closed."""
    import subprocess
    ROOT = Path(__file__).parent.parent.parent.parent
    deploy_sh = ROOT / "deploy" / "deploy-approval" / "deploy.sh"
    text = deploy_sh.read_text()
    import re
    m = re.search(r'require_secrets\(\)\s*\{(.*?)\n\}', text, re.DOTALL)
    assert m, "require_secrets function must exist in deploy.sh"
    body = m.group(1)

    bad_secrets = tmp_path / "secrets.yaml"
    bad_secrets.write_text(
        "version: 1\n"
        "cos:\n"
        "  region: ap-beijing\n"
        "  bucket: testbucket123\n"
        "  secret_id: TEST_FAKE_SECRET_ID_AAAA\n"
        "  secret_key: TEST_FAKE_SECRET_KEY_AAAA\n"
        "review_comment_trust:\n"
        "  author_id: \"not-a-numeric-id\"\n"
        "  author_login: \"Haohao-end\"\n"
        "agent_core_tokens:\n"
        "  m1: token\n"
    )

    # Write a wrapper that defines require_secrets inline from deploy.sh body
    wrapper = tmp_path / "test_wrapper.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'set -euo pipefail\n'
        'SECRETS_FILE="' + str(bad_secrets) + '"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        "require_secrets() {\n"
        + body
        + "\n}\n"
        "require_secrets\n"
    )
    wrapper.chmod(0o755)

    r = subprocess.run(
        ["bash", str(wrapper)],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "author_id must be a positive decimal string" in (r.stdout + r.stderr)


def test_service_has_no_self_approval_prohibition():
    """8. service.py must not re-appear with self-approval prohibition."""
    service_py = Path(__file__).parent.parent / "service.py"
    text = service_py.read_text(encoding="utf-8")
    assert "_is_self_approval" not in text, "service.py must not contain _is_self_approval"
    assert "PR author cannot approve their own deployment" not in text
    assert "A different Machine Owner or authorized collaborator must approve" not in text
    assert "No-self-approval" not in text


def test_docs_have_no_self_approval_forbidden():
    """9. docs must not declare self-approval forbidden."""
    ROOT = Path(__file__).parent.parent.parent.parent
    agent_md = ROOT / "DEPLOY_APPROVAL_AGENT.md"
    text = agent_md.read_text(encoding="utf-8")
    assert "Self-approval is FORBIDDEN" not in text, "DEPLOY_APPROVAL_AGENT.md must not say Self-approval is FORBIDDEN"
    assert "PR author must not approve their own deployment" not in text

    arch_md = ROOT / "docs" / "deploy-approval-github-driven-architecture.md"
    arch_text = arch_md.read_text(encoding="utf-8")
    assert "Self-approval is FORBIDDEN" not in arch_text


# ── agent_core_tokens fail-closed validation ────────────────────────

def test_agent_core_tokens_empty_mapping_fails():
    """agent_core_tokens={} -> ValueError."""
    with pytest.raises(ValueError, match="agent_core_tokens must be non-empty"):
        validate_config(
            Config(
                github_repos=list(DEFAULT_GITHUB_REPOS),
                review_comment_author_id="7950763",
                review_comment_author_login="kentcyq",
                agent_core_tokens={},
            )
        )


def test_agent_core_tokens_non_mapping_fails():
    """agent_core_tokens='not-a-mapping' -> ValueError."""
    with pytest.raises(ValueError, match="agent_core_tokens must be a mapping"):
        validate_config(
            Config(
                github_repos=list(DEFAULT_GITHUB_REPOS),
                review_comment_author_id="7950763",
                review_comment_author_login="kentcyq",
                agent_core_tokens="not-a-mapping",
            )
        )


def test_agent_core_tokens_empty_key_fails():
    """agent_core_tokens={"": "token"} -> ValueError."""
    with pytest.raises(ValueError, match="agent_core_tokens keys must be non-empty strings"):
        validate_config(
            Config(
                github_repos=list(DEFAULT_GITHUB_REPOS),
                review_comment_author_id="7950763",
                review_comment_author_login="kentcyq",
                agent_core_tokens={"": "token"},
            )
        )


def test_agent_core_tokens_empty_value_fails():
    """agent_core_tokens={"machine": ""} -> ValueError."""
    with pytest.raises(ValueError, match="agent_core_tokens values must be non-empty strings"):
        validate_config(
            Config(
                github_repos=list(DEFAULT_GITHUB_REPOS),
                review_comment_author_id="7950763",
                review_comment_author_login="kentcyq",
                agent_core_tokens={"machine": ""},
            )
        )


def test_agent_core_tokens_valid_mapping_passes():
    """Valid mapping -> PASS."""
    validate_config(
        Config(
            github_repos=list(DEFAULT_GITHUB_REPOS),
            review_comment_author_id="7950763",
            review_comment_author_login="kentcyq",
            agent_core_tokens={"machine1": "token1", "machine2": "token2"},
            cos_region="ap-beijing",
            cos_bucket="testbucket123",
            cos_secret_id="TEST_FAKE_SECRET_ID_AAAA",
            cos_secret_key="TEST_FAKE_SECRET_KEY_AAAA",
        )
    )


def test_load_config_ignores_legacy_registry_environment(tmp_path, monkeypatch):
    """Prove legacy REGISTRY env vars are inert at runtime."""
    import agents.deploy_approval.config as config_mod
    orig = config_mod._load_secrets_config

    # Set all legacy registry env vars
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    monkeypatch.setenv("REGISTRY_USER", "some-user")
    monkeypatch.setenv("REGISTRY_PASSWORD", "some-password")

    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(
        'version: 1\n'
        'cos:\n'
        '  region: ap-beijing\n'
        '  bucket: testbucket123\n'
        '  secret_id: TEST_FAKE_SECRET_ID_AAAA\n'
        '  secret_key: TEST_FAKE_SECRET_KEY_AAAA\n'
        'review_comment_trust:\n'
        '  author_id: "7950763"\n'
        '  author_login: "kentcyq"\n'
        'agent_core_tokens:\n'
        '  test-machine: test-token\n'
    )

    def patched(path):
        return orig(str(secrets))

    config_mod._load_secrets_config = patched
    try:
        cfg = config_mod.load_config()
    finally:
        config_mod._load_secrets_config = orig

    # Config must not have any registry attribute
    assert not hasattr(cfg, "registry"), "Config must not have registry attribute"
    assert not hasattr(cfg, "registry_auth_host_allowlist"), "Config must not have registry_auth_host_allowlist"

    # Normal canonical config must load successfully
    assert cfg.cos_region == "ap-beijing"
    assert cfg.cos_bucket == "testbucket123"
    assert cfg.review_comment_author_id == "7950763"
    assert cfg.review_comment_author_login == "kentcyq"
    assert "test-machine" in cfg.agent_core_tokens


def test_config_runtime_repos_dynamic_and_validated(monkeypatch):
    """The server allowlist defines the primary; malformed entries never disappear."""
    import agents.deploy_approval.config as config_mod
    from ..models import BuildInfo
    from ..service import _is_deployable_build

    monkeypatch.setattr(config_mod, "_load_secrets_config", lambda _path: {
        "cos": {"region": "ap-beijing", "bucket": "offline-bucket-123",
                "secret_id": "OFFLINE_ID", "secret_key": "OFFLINE_KEY"},
        "review_comment_trust": {"author_id": "123456", "author_login": "ci-review"},
        "agent_core_tokens": {"test-machine": "OFFLINE_TOKEN"}, "agent_core_ca_pem": {},
    })
    for raw, expected in (
        ("4paradigm/phanthymotus,4paradigm/phanthymotus-driver", list(DEFAULT_GITHUB_REPOS)),
        ("Example-Team/Main, example-team/drivers", ["example-team/main", "example-team/drivers"]),
    ):
        monkeypatch.setenv("GITHUB_REPOS", raw)
        cfg = load_config()
        assert cfg.github_repos == expected and cfg.primary_repo == expected[0]
        assert cfg.active_repos == [] and cfg.auth_valid is False
        for target in ("core", "perception", "actucore"):
            build = BuildInfo(target=target, driver_path="", variant="", success=True,
                              image_tag="registry/component:v1", deployable=False)
            assert _is_deployable_build(expected[0], build, cfg.github_repos)
            assert not _is_deployable_build("outside/unconfigured", build, cfg.github_repos)
        driver = BuildInfo(target="driver", driver_path="unitree/g1", variant="",
                           success=True, image_tag="registry/driver:v1", deployable=False)
        assert not _is_deployable_build(expected[0], driver, cfg.github_repos)
        assert _is_deployable_build(expected[1], driver, cfg.github_repos)
    for raw in ("", "team/main,", ",team/main", "team/main,,team/drivers",
                "team/main,TEAM/Main", "team/../main", "../main",
                "team/..", "team/main?extra", "https://github.com/team/main", "team main/repo"):
        monkeypatch.setenv("GITHUB_REPOS", raw)
        with pytest.raises(ValueError, match="GITHUB_REPOS"):
            load_config()
