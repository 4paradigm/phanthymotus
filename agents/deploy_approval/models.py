"""Data models for the Deploy Approval Agent (stateless GitHub persistence).

The ONLY persistence is the GitHub lifecycle comment hidden state.
No SQLite, no DeploymentStore, no deployment DB rows.

State machine (hidden state only):
  review-required -> reviewing -> deploy-ready -> deploy-requested
  -> testing -> succeeded | failed

Canonical lifecycle:
  review-required -> reviewing -> deploy-ready -> deploy-requested
  -> testing -> succeeded | failed

Deployment execution and validation failures may transition:
  deploy-requested -> failed
  testing -> failed
failed is terminal for that request.

Actually removed (not present in contract):
  legacy rollout / cleanup / rollback / intermediate machine-clean states
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


def utc_now() -> float:
    return _dt.datetime.now(_dt.timezone.utc).timestamp()


def new_id() -> str:
    return uuid.uuid4().hex


# Deployment state machine (pre-merge validation only)
DeploymentStatus = Literal[
    "review-required",
    "reviewing",
    "deploy-ready",
    "deploy-requested",
    "testing",
    "succeeded",
    "failed",
]


def terminal_deployment(status: str) -> bool:
    return status in {
        "succeeded",
        "failed",
    }


# Allowed transitions. A transition not present here is rejected before any
# side effect runs.
_TRANSITIONS: dict[str, set[str]] = {
    "review-required": {"reviewing"},
    "reviewing": {"deploy-ready", "review-required"},
    "deploy-ready": {"deploy-requested", "review-required"},
    "deploy-requested": {"testing", "failed", "review-required"},
    "testing": {"succeeded", "failed", "review-required"},
    "succeeded": set(),
    "failed": set(),
}

ALL_STATUSES = tuple(_TRANSITIONS.keys())


def can_transition(current: str, target: str) -> bool:
    return target in _TRANSITIONS.get(current, set())


# Machine owners configuration
@dataclass
class MachineInfo:
    alias: str
    node_id: str
    owners: list[str]  # GitHub logins (case-insensitive)
    node_host: str = ""
    tls_peer_cert_file: str = ""
    targets: list[str] | None = None  # allowed targets (None = all)
    platforms: list[str] | None = None  # allowed platforms (None = all)
    variants: list[str] | None = None  # allowed variants (None = all)
    driver_paths: list[str] | None = None  # allowed driver paths (None = all)


@dataclass
class BuildInfo:
    """One build result from the Review Agent, as used in the lifecycle comment."""
    idx: int
    target: str
    driver_path: str
    variant: str
    success: bool
    image_tag: str
    deployable: bool  # computed by Controller: target=perception|actucore|driver
    component_id: str = ""


@dataclass
class HiddenState:
    """The complete hidden state schema (version 1)."""
    version: int = 1
    head_sha: str = ""
    status: str = "review-required"
    review_evidence: dict = field(default_factory=dict)
    components: list[dict] = field(default_factory=list)
    deployments: list[dict] = field(default_factory=list)
    case_results: dict[str, str] = field(default_factory=dict)
    test_result: str = ""  # pass/fail
    cos: dict = field(default_factory=lambda: {"object_key": "", "sha256": "", "size": 0})
    approve_attempts: list[dict] = field(default_factory=list)
    approve_attempts_total: int = 0
    approve_attempts_truncated: bool = False
    command: dict = field(default_factory=lambda: {
        "comment_id": 0, "kind": "", "phase": "completed", "args": {}
    })
    last_processed_comment_id: int = 0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "head_sha": self.head_sha,
            "status": self.status,
            "review_evidence": dict(self.review_evidence),
            "components": self.components,
            "deployments": self.deployments,
            "case_results": dict(self.case_results),
            "test_result": self.test_result,
            "cos": dict(self.cos),
            "approve_attempts": list(self.approve_attempts),
            "approve_attempts_total": self.approve_attempts_total,
            "approve_attempts_truncated": self.approve_attempts_truncated,
            "command": dict(self.command),
            "last_processed_comment_id": self.last_processed_comment_id,
        }


__all__ = [
    "MachineInfo",
    "BuildInfo",
    "HiddenState",
    "DeploymentStatus",
    "ALL_STATUSES",
    "terminal_deployment",
    "can_transition",
    "utc_now",
    "new_id",
]
