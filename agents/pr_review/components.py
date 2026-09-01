"""Which rules, docs and reference implementation apply to a given PR.

Keeps the mapping in one place so the loop does not grow a pile of conditionals,
and so adding a component means adding a row and a markdown file.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .models import BuildTarget

logger = logging.getLogger(__name__)

RULES_DIR = Path(__file__).parent / "rules"

# Reference drivers by shape. A new driver is best judged against the closest
# existing one, and "closest" is not always the same driver — the repo has clear
# exemplars for different kinds of work.
#
# Deliberately absent: deep_robotics/lynx_m20 (empty plugins, no service.yml,
# ships a committed .zip) and unitree/g1/device.py (spec-canonical for IDs and
# ports, but 3,400 lines in one file — not a structural model).
DRIVER_REFERENCES = [
    ("unitree/go1",
     "the declared blueprint: cleanest module split, and the only driver with an "
     "authoring tutorial (unitree/go1/CONTRIBUTING.md)"),
    ("robotera/q5_bundle",
     "best decomposition — one module per capability, plus sensor/control "
     "contract abstractions"),
    ("x-humanoid/tianyi2.0",
     "largest complete bundle (30 plugins), and has tests"),
    ("unitree/r1",
     "compact and conventional — the model for a small new driver"),
]

# Extra references picked when the PR's shape suggests them.
DRIVER_REFERENCE_HINTS = [
    (("dji/",), "dji/mavic3e", "native-SDK C bridge (psdk_bridge) pattern"),
    (("grpc", "proto"), "pndbotics/adam", "gRPC vendor-SDK pattern"),
    (("lidar", "slam", "pointcloud", "icp", "mapping"),
     "unitree/go2", "SLAM / spatial reference"),
]


@dataclass
class ComponentContext:
    """Everything component-specific the loop needs for one review."""

    name: str
    rules: str = ""
    # Which rule files were concatenated into `rules`. Recorded in the review
    # trace so the dashboard can show which standards the review was held to.
    rule_files: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    references: list[tuple[str, str]] = field(default_factory=list)


def _read_rules(*names: str) -> str:
    """Concatenate rule files, skipping any that are missing."""
    parts = []
    for n in names:
        p = RULES_DIR / n
        try:
            parts.append(p.read_text())
        except OSError as e:
            # A missing rule file degrades the review rather than breaking it.
            logger.warning(f"Rule file {n} unreadable: {e}")
    return "\n\n---\n\n".join(parts)


def build_context(
    repo_full_name: str,
    targets: list[BuildTarget],
    driver_paths: list[str],
    changed_files: list[str],
) -> ComponentContext:
    """Pick rules, docs and references for this PR."""
    repo = repo_full_name.split("/")[-1]

    if repo == "phanthymotus-driver":
        return _driver_context(driver_paths, changed_files)

    # phanthymotus: core, perception and actucore can all appear in one PR.
    touches_core = any(f.startswith("agent-core/") for f in changed_files)
    touches_perc = any(f.startswith("perception/") for f in changed_files)
    touches_actu = any(f.startswith("actucore/") for f in changed_files)

    names, docs, label = ["common.md"], [], []
    if touches_core:
        names.append("core.md")
        label.append("agent-core")
        # No agent-core/README.md exists — these are the real references.
        docs += ["CONTRIBUTING.md", "README.md"]
    if touches_perc:
        names.append("perception.md")
        label.append("perception")
        docs.append("perception/README.md")
    if touches_actu:
        names.append("actucore.md")
        label.append("actucore")
        docs.append("actucore/README.md")

    if not label:
        label = ["phanthymotus"]
        docs = ["CONTRIBUTING.md"]

    return ComponentContext(
        name=" + ".join(label),
        rules=_read_rules(*names),
        rule_files=list(names),
        docs=_dedupe(docs),
        references=[],
    )


def _driver_context(
    driver_paths: list[str], changed_files: list[str]
) -> ComponentContext:
    joined = " ".join(changed_files).lower()

    refs = list(DRIVER_REFERENCES)
    for needles, path, why in DRIVER_REFERENCE_HINTS:
        if any(n in joined for n in needles):
            refs.insert(0, (path, why))

    # Never suggest comparing a driver against itself.
    refs = [(p, why) for p, why in refs if p not in driver_paths]

    return ComponentContext(
        name=", ".join(driver_paths) if driver_paths else "phanthymotus-driver",
        rules=_read_rules("common.md", "driver.md"),
        rule_files=["common.md", "driver.md"],
        docs=_dedupe([
            "README_dev.md",                  # the authoritative spec
            "README.md",                      # catalogue + rendering tables
            "unitree/go1/CONTRIBUTING.md",    # the card-authoring tutorial
        ]),
        references=refs[:4],
    )


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
