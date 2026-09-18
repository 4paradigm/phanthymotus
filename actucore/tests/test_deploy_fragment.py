"""What the compose fragment shipped inside the image has to say.

agent-core reads `/deploy/service.yml` out of the image and merges it into the
host's docker-compose.yml, so this file *is* the deployment. A line missing
here is a line missing on every robot, and the failures it causes do not look
like deployment failures.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_deploy_fragment.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

FRAGMENT = Path(__file__).resolve().parents[1] / "deploy" / "service.yml"


@pytest.fixture(scope="module")
def service():
    return yaml.safe_load(FRAGMENT.read_text())["actucore"]


def test_the_container_asks_for_the_nvidia_runtime(service):
    """Without it there is no GPU, whatever `privileged` and /dev:/dev suggest.

    The L4T base ships zero-length placeholders for the driver libraries and
    only the nvidia runtime bind-mounts the host's real ones over them. Missing,
    `torch.cuda.is_available()` is False and a local VLA provider falls back to
    CPU — announced in a single warning line that nobody reads, after which a
    450M-parameter policy runs on an Orin's CPU at a rate no arm can use.
    Measured on a Tianyi before this line existed.
    """
    assert service.get("runtime") == "nvidia"


def test_the_dds_profile_is_mounted(service):
    """A DDS container without it isolates itself: the robot hears nothing."""
    mounts = service.get("volumes") or []
    assert any("dds-local.xml" in m for m in mounts)


def test_the_model_directory_is_mounted(service):
    """Weights are fetched at runtime; without this they live in the container
    and are re-downloaded — gigabytes — on every deploy."""
    mounts = service.get("volumes") or []
    assert any(m.split(":")[1].rstrip("/") == "/models"
               for m in mounts if ":" in m)


def test_the_image_is_the_placeholder_the_console_substitutes(service):
    assert service["image"] == "__IMAGE__"
