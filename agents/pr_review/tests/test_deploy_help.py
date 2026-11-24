"""The "Try it" block gives every target the same run-pr-image.sh command."""

from pathlib import Path

from agents.pr_review.builder import container_name_from_service_yaml, service_yaml_path
from agents.pr_review.comments import _deploy_help
from agents.pr_review.models import BuildResult, BuildTarget

REPO = Path(__file__).resolve().parents[3]
RUN = "curl -fsSL https://motus.phanthy.com/run-pr-image.sh | bash -s -- "


def _built(target, driver_path=None):
    rel = service_yaml_path(target, driver_path)
    name = container_name_from_service_yaml((REPO / rel).read_text())
    return BuildResult(target=target, driver_path=driver_path, success=True,
                       image_tag=f"reg/phanthy-motus/{target.value}:release.1",
                       log_tail="", container_name=name)


def test_core_declares_the_agent_core_container():
    assert _built(BuildTarget.CORE).container_name == "phanthy-motus-agent-core-1"


def test_core_gets_the_same_command_as_perception():
    out = _deploy_help([_built(BuildTarget.CORE), _built(BuildTarget.PERCEPTION)])
    assert RUN + "reg/phanthy-motus/core:release.1" in out
    assert RUN + "reg/phanthy-motus/perception:release.1" in out
    assert "--down agent-core" in out
    assert "web console's deploy panel" not in out


def test_a_full_compose_file_and_a_fragment_both_name_their_container():
    assert container_name_from_service_yaml(
        "services:\n  agent-core:\n    container_name: c1\n") == "c1"
    assert container_name_from_service_yaml("svc:\n  image: x\n") == "svc"
