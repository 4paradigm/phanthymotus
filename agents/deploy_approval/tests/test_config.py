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
            github_repos=list(DEFAULT_GITHUB_REPOS),
            registry="ccr.ccs.tencentyun.com",
            review_comment_author_id="7950763",
            agent_core_tokens={"test": "token"},
        )
    )


def test_validate_config_default_repos():
    """Default github_repos contains both production repositories."""
    c = Config(review_comment_author_id="7950763", agent_core_tokens={"test": "token"})
    assert "4paradigm/phanthymotus" in c.github_repos
    assert "4paradigm/phanthymotus-driver" in c.github_repos
    # Explicit empty overrides defaults — must fail closed
    with pytest.raises(ValueError, match="GITHUB_REPOS is required"):
        validate_config(Config(github_repos=[], registry="ccr.ccs.tencentyun.com", agent_core_tokens={"test": "token"}))


@pytest.mark.parametrize(
    "repos,should_pass",
    [
        (list(DEFAULT_GITHUB_REPOS), True),
        (list(reversed(DEFAULT_GITHUB_REPOS)), True),
        (["4paradigm/phanthymotus"], False),
        (["4paradigm/phanthymotus-driver"], False),
        (["4paradigm/phanthymotus", "4paradigm/phanthymotus"], False),
        (["4paradigm/phanthymotus", "some/fork-repo"], False),
        (["some/other"], False),
        ([], False),
        ([*DEFAULT_GITHUB_REPOS, "some/other"], False),
    ],
)
def test_github_repos_requires_exact_runtime_repo_set(repos, should_pass):
    cfg = Config(
        github_repos=repos,
        registry="ccr.ccs.tencentyun.com",
        agent_core_tokens={"test": "token"},
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
                github_repos=list(DEFAULT_GITHUB_REPOS),
                webhook_enabled=True,
                review_comment_author_id="7950763",
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


def test_env_float():
    os.environ["TEST_FLOAT"] = "3.5"
    assert _env_float("TEST_FLOAT", 1.0) == 3.5
    os.environ["TEST_FLOAT"] = "0"
    with pytest.raises(ValueError):
        _env_float("TEST_FLOAT", 1.0)
    del os.environ["TEST_FLOAT"]


def test_config_env_override(monkeypatch, tmp_path):
    """Explicit GITHUB_REPOS overrides the default."""
    monkeypatch.setenv("GITHUB_REPOS", "4paradigm/phanthymotus,4paradigm/phanthymotus-driver")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text('version: 1\ncos:\n  region: test\n  bucket: test\n  secret_id: test\n  secret_key: test\nreview_comment_trust:\n  author_id: "7950763"\n  author_login: "kentcyq"\nagent_core_tokens:\n  test-machine: test-token\n')
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
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    with pytest.raises(ValueError, match="GITHUB_REPOS is required"):
        load_config()


def test_config_env_unset_uses_default(monkeypatch, tmp_path):
    """GITHUB_REPOS unset uses DEFAULT_GITHUB_REPOS."""
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")
    monkeypatch.delenv("GITHUB_REPOS", raising=False)
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text('version: 1\ncos:\n  region: test\n  bucket: test\n  secret_id: test\n  secret_key: test\nreview_comment_trust:\n  author_id: "7950763"\n  author_login: "kentcyq"\nagent_core_tokens:\n  test-machine: test-token\n')
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
    from ..config import REVIEW_AGENT_GITHUB_USER_ID, REVIEW_AGENT_GITHUB_LOGIN

    assert REVIEW_AGENT_GITHUB_USER_ID == "7950763"
    assert REVIEW_AGENT_GITHUB_LOGIN == "kentcyq"

    validate_config(
        Config(
            github_repos=list(DEFAULT_GITHUB_REPOS),
            registry="ccr.ccs.tencentyun.com",
            review_comment_author_id=REVIEW_AGENT_GITHUB_USER_ID,
            review_comment_author_login=REVIEW_AGENT_GITHUB_LOGIN,
            agent_core_tokens={"test": "token"},
        )
    )


def test_validate_config_rejects_haohao_end_as_review_agent():
    """B. author_id=184792454, author_login=Haohao-end => fail closed."""
    with pytest.raises(ValueError, match="author_id must be"):
        validate_config(
            Config(
                github_repos=list(DEFAULT_GITHUB_REPOS),
                registry="ccr.ccs.tencentyun.com",
                review_comment_author_id="184792454",
                review_comment_author_login="Haohao-end",
                agent_core_tokens={"test": "token"},
            )
        )


def test_validate_config_rejects_correct_id_wrong_login():
    """C. author_id correct but login wrong => fail closed."""
    with pytest.raises(ValueError, match="author_login must be"):
        validate_config(
            Config(
                github_repos=list(DEFAULT_GITHUB_REPOS),
                registry="ccr.ccs.tencentyun.com",
                review_comment_author_id="7950763",
                review_comment_author_login="wrong-login",
                agent_core_tokens={"test": "token"},
            )
        )


def test_validate_config_rejects_wrong_id_correct_login():
    """D. author_id wrong but login correct => fail closed."""
    with pytest.raises(ValueError, match="author_id must be"):
        validate_config(
            Config(
                github_repos=list(DEFAULT_GITHUB_REPOS),
                registry="ccr.ccs.tencentyun.com",
                review_comment_author_id="12345",
                review_comment_author_login="kentcyq",
                agent_core_tokens={"test": "token"},
            )
        )


def test_load_config_reads_canonical_secrets(tmp_path, monkeypatch):
    """E. load_config reads 7950763 / kentcyq from secrets.yaml."""
    import agents.deploy_approval.config as config_mod
    orig = config_mod._load_secrets_config

    monkeypatch.setenv("REGISTRY", "ccr.ccs.tencentyun.com")

    secrets = tmp_path / "secrets.yaml"
    secrets.write_text(
        'version: 1\n'
        'cos:\n'
        '  region: test\n'
        '  bucket: test\n'
        '  secret_id: test\n'
        '  secret_key: test\n'
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

def test_deploy_sh_validates_canonical_review_agent_trust():
    """6. deploy.sh require_secrets contains and enforces canonical review_comment_trust."""
    ROOT = Path(__file__).parent.parent.parent.parent
    deploy_sh = ROOT / "deploy" / "deploy-approval" / "deploy.sh"
    text = deploy_sh.read_text(encoding="utf-8")
    assert '"7950763"' in text, "deploy.sh must validate author_id == 7950763"
    assert '"kentcyq"' in text, "deploy.sh must validate author_login == kentcyq"


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
        "  region: test\n"
        "  bucket: test\n"
        "  secret_id: test\n"
        "  secret_key: test\n"
        "review_comment_trust:\n"
        "  author_id: \"184792454\"\n"
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
    assert "7950763" in (r.stdout + r.stderr)


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
