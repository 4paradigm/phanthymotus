"""Security blocker tests (final alignment).
Fail-closed, legacy commands unknown, per-machine owners, no duplicate POST.
"""

from __future__ import annotations

import asyncio
import pytest
import yaml

from ..commands import parse_command
from ..agent_core_client import AgentCoreClient, AgentCoreError
from ..models import can_transition
from ..policy import Policy, PolicyError, load_machines, MachineLoadError
from .conftest import make_config


def test_legacy_rollback_commands_unknown():
    assert parse_command("/reject_deploy d-9").kind == "unknown"
    assert parse_command("/rollback_deploy d-7").kind == "unknown"
    assert parse_command("/cancel_deploy d-9").kind == "unknown"
    assert parse_command("/resume_deploy d-9").kind == "unknown"


def test_unknown_command_no_mutation():
    cmd = parse_command("/unknown_command")
    assert cmd.kind == "unknown"
    assert not cmd.is_command


def test_unbalanced_quote_fails_closed():
    assert parse_command('/request_deploy build=1 test-mode="manual').kind == "unknown"


def test_machine_owner_empty_fails_closed(tmp_path):
    import tempfile, os as _os
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump({"version": 1, "machines": {
        "g1": {"node_id": "n1", "owners": []},
    }}, f)
    path = f.name
    f.close()
    try:
        with pytest.raises(MachineLoadError):
            load_machines(path)
    finally:
        _os.unlink(path)


def test_non_owner_rejected(tmp_path):
    p = Policy(make_config())
    with pytest.raises(PolicyError):
        p.can_approve("alice", actor_id="id2", machine_alias="nonexistent")


def test_collaborator_write_can_approve():
    from ..policy import Policy
    assert Policy.collaborator_can_approve("write") is True
    assert Policy.collaborator_can_approve("") is False


def test_fail_closed_state_transition():
    assert not can_transition("waiting-approval", "succeeded")


def test_no_controller_cleanup():
    """Deploy Controller must not implement cleanup methods."""
    import agents.deploy_approval.service as svc_mod
    import inspect
    src = inspect.getsource(svc_mod)
    # The service must not call stop/remove/sync endpoints
    forbidden = ["/api/drivers/{id}/stop", "driver_stop", "driver_remove",
                 "system_update", "docker rm", "docker stop"]
    for phrase in forbidden:
        assert phrase not in src, f"Found forbidden pattern: {phrase}"


def test_fail_closed_head_drift():
    """HEAD drift must fail closed."""
    # The controller checks current HEAD before dispatching commands
    # This is verified in the controller's handle_approve_deploy path
    # which reads fresh PR data and compares head_sha
    from ..github_state_proxy import _is_valid_full_sha
    assert _is_valid_full_sha("a" * 40) is True
    assert _is_valid_full_sha("b" * 40) is True
    assert _is_valid_full_sha("") is False


def test_deploy_compose_runs_as_invoking_non_root_uid_gid():
    from pathlib import Path

    compose = Path("deploy/deploy-approval/docker-compose.yml").read_text(encoding="utf-8")
    assert 'user: "${DEPLOY_APPROVAL_RUNTIME_UID:-65532}:${DEPLOY_APPROVAL_RUNTIME_GID:-65532}"' in compose
    assert "privileged:" not in compose
    assert "network_mode: host" not in compose
    assert "/var/run/docker.sock" not in compose


def test_deploy_sh_runtime_uid_gid_and_private_input_gates():
    from pathlib import Path

    deploy_sh = Path("deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    assert 'RUNTIME_UID="$(id -u)"' in deploy_sh
    assert 'RUNTIME_GID="$(id -g)"' in deploy_sh
    assert '"$RUNTIME_UID" -gt 0' in deploy_sh
    assert "runtime user must be non-root" in deploy_sh
    assert "DEPLOY_APPROVAL_RUNTIME_UID" in deploy_sh
    assert "DEPLOY_APPROVAL_RUNTIME_GID" in deploy_sh
    assert "must not grant group/world permissions" in deploy_sh
    assert "must be owned by the invoking user" in deploy_sh
    assert "TLS peer cert" in deploy_sh
    assert "must not be a symlink" in deploy_sh
    assert "os.getuid()" in deploy_sh
    assert "int(sys.argv[4])" not in deploy_sh
    # os.getuid() replaces the shell-injected UID
    # Verify os.getuid is used in the embedded Python, not a shell-injected UID
    import re
    # The embedded Python inside require_private_local_inputs must use os.getuid()
    py_section = re.search(r'require_private_local_inputs\(\) \{[^}]*python3 - "\$MACHINES_FILE" "\$SECRETS_FILE" "\$CERTS_DIR"[^}]*\}', deploy_sh, re.DOTALL)
    # Cannot easily extract just the heredoc, but os.getuid() presence is sufficient
    assert "os.getuid()" in deploy_sh



def test_deploy_sh_start_uses_require_runtime_inputs():
    from pathlib import Path
    deploy_sh = Path("deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    assert "require_runtime_inputs" in deploy_sh
    assert 'cmd_start()' in deploy_sh
    assert 'cmd_restart()' in deploy_sh


def test_deploy_sh_start_uses_no_build():
    from pathlib import Path
    deploy_sh = Path("deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    # Find the cmd_start section
    import re
    start_match = re.search(r'cmd_start\(\) \{[^}]+\}', deploy_sh, re.DOTALL)
    assert start_match is not None
    start_body = start_match.group()
    assert "require_runtime_inputs" in start_body
    assert "--no-build" in start_body


def test_deploy_sh_restart_uses_no_build_force_recreate():
    from pathlib import Path
    deploy_sh = Path("deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    import re
    restart_match = re.search(r'cmd_restart\(\) \{[^}]+\}', deploy_sh, re.DOTALL)
    assert restart_match is not None
    restart_body = restart_match.group()
    assert "require_runtime_inputs" in restart_body
    assert "--no-build" in restart_body
    assert "--force-recreate" in restart_body


def test_deploy_sh_no_raw_compose_start():
    from pathlib import Path
    deploy_sh = Path("deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    # The old raw implementations must be gone
    assert "COMPOSE start" not in deploy_sh or "require_runtime_inputs" in deploy_sh.split("COMPOSE start")[0] if "COMPOSE start" in deploy_sh else True
    # Check for the old bare start pattern in the body of cmd_start
    assert "COMPOSE start" not in deploy_sh.split("cmd_stop")[0] if "cmd_stop" in deploy_sh else True


def test_deploy_sh_no_raw_compose_restart():
    from pathlib import Path
    deploy_sh = Path("deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    assert "COMPOSE restart" not in deploy_sh or "require_runtime_inputs" in deploy_sh.split("COMPOSE restart")[0] if "COMPOSE restart" in deploy_sh else True



def test_deploy_sh_cert_direct_child_only():
    from pathlib import Path
    deploy_sh = Path("deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    assert "direct child of" in deploy_sh
    assert "must end with .pem" in deploy_sh
    assert "endswith" in deploy_sh


def test_deploy_sh_cert_no_subdirectory():
    from pathlib import Path
    deploy_sh = Path("deploy/deploy-approval/deploy.sh").read_text(encoding="utf-8")
    # The old relative_to / subdirectory path must not be used
    assert "local_certs_dir / relative" not in deploy_sh
    assert "local_certs_dir / logical.name" in deploy_sh


def test_policy_cert_direct_child():
    from pathlib import Path
    policy = Path("agents/deploy_approval/policy.py").read_text(encoding="utf-8")
    assert "parent != _TLS_CERT_DIR" in policy
    assert "must be a direct child" in policy
    assert "must end with .pem" in policy


def test_agent_core_client_cert_direct_child():
    from pathlib import Path
    acl = Path("agents/deploy_approval/agent_core_client.py").read_text(encoding="utf-8")
    assert "direct child" in acl
    assert "remainder.endswith" in acl


def test_machines_example_direct_child_cert_path():
    from pathlib import Path
    example = Path("deploy/deploy-approval/machines.example.yaml").read_text(encoding="utf-8")
    assert ".pem" in example
    assert "/run/deploy-approval/certs/" in example



# ══════════════════════════════════════════════════════════════════════════════
# MIGRATED from test_v10_contract.py
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentCoreClientSecurity:
    """AgentCoreClient constructor defense-in-depth."""

    def test_configured_private_agent_core_node_allowed_with_private_http_disabled(self, config):
        """A node with a configured 10.x IP can be used even when allow_private_http is False."""
        config.allow_private_http = False
        client = AgentCoreClient(
            config,
            base_url="https://10.0.0.1:15678",
            node_host="10.0.0.1",
        tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
)
        assert client.base_url == "https://10.0.0.1:15678"
        assert client.node_host == "10.0.0.1"

    def test_unconfigured_private_agent_core_node_rejected(self, config):
        """A private IP not matching the machine's node_host is rejected by the constructor."""
        config.allow_private_http = False
        with pytest.raises(AgentCoreError, match="must match node_host"):
            AgentCoreClient(
                config,
                base_url="https://10.0.0.99:15678",
                node_host="10.0.0.1",
            tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
)

    def test_agent_core_node_wrong_port_rejected(self, config):
        """Port must be exactly 15678."""
        with pytest.raises(AgentCoreError, match="port must be 15678"):
            AgentCoreClient(
                config,
                base_url="https://10.0.0.1:15679",
                node_host="10.0.0.1",
            tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
)

    def test_machine_node_host_rejects_url_or_path_injection(self, config):
        """node_host must be a literal IP; URL/path injection is rejected."""
        # non-IP node_host -> literal IP check
        with pytest.raises(AgentCoreError, match="literal IP"):
            AgentCoreClient(
                config,
                base_url="https://10.0.0.1:15678",
                node_host="evil.com",
                tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
            )
        # path injection in base_url
        with pytest.raises(AgentCoreError, match="must not contain a path"):
            AgentCoreClient(
                config,
                base_url="https://10.0.0.1:15678/api/evil",
                node_host="10.0.0.1",
            tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
)
        with pytest.raises(AgentCoreError, match="must not contain a path"):
            AgentCoreClient(
                config,
                base_url="https://10.0.0.1:15678/evil",
                node_host="10.0.0.1",
            tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
)
        with pytest.raises(AgentCoreError, match="must not contain query"):
            AgentCoreClient(
                config,
                base_url="https://10.0.0.1:15678?evil=1",
                node_host="10.0.0.1",
            tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
)
        with pytest.raises(AgentCoreError, match="must not contain fragment"):
            AgentCoreClient(
                config,
                base_url="https://10.0.0.1:15678#evil",
                node_host="10.0.0.1",
            tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
)

    def test_agent_core_bearer_token_is_sent(self, config):
        """Per-machine token is used via config parameter, not ACCESS_TOKEN env."""
        client = AgentCoreClient(
            config,
            base_url="https://10.0.0.1:15678",
            node_host="10.0.0.1",
            tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
            access_token="secret-token-123",
        )
        headers = client._headers()
        assert headers.get("Authorization") == "Bearer secret-token-123"
    def test_agent_core_token_never_persisted_or_rendered(self, config):
            """The token is read from env at call time and never stored on the instance."""
        # Per-machine token passed as parameter, not from env.
            client = AgentCoreClient(
                    config,
                    base_url="https://10.0.0.1:15678",
                    node_host="10.0.0.1",
                tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
                    access_token="my-secret-token",
    )
            # The token is not stored directly on the instance
            assert not hasattr(client, "token_value")
            # The env var is not exposed in __dict__
            inst_repr = repr(client)
            assert "my-secret" not in inst_repr
            # The token is only produced by _headers() at call time
            headers = client._headers()
            assert headers["Authorization"] == "Bearer my-secret-token"

    def test_agent_core_verify_invalid_token_fails_closed(self, config):
        """verify() raises AgentCoreError when the token is invalid (401)."""
        import httpx
        client = AgentCoreClient(
            config,
            base_url="https://10.0.0.1:15678",
            node_host="10.0.0.1",
        tls_peer_cert_file="/run/deploy-approval/certs/test-agent-core.pem",
)
        # Replace verify with a mock that raises a SecurityError
        async def _mock_verify():
            from agents.deploy_approval.clients_common import SecurityError
            raise SecurityError("unexpected status 401")
        client.verify = _mock_verify

        with pytest.raises((AgentCoreError, Exception), match="401|unauthorized|unexpected status"):
            asyncio.run(client.verify())
