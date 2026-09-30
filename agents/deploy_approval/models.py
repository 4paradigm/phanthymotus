"""Data models for the Deploy Approval Agent (stateless GitHub persistence).

The ONLY persistence is the GitHub lifecycle comment hidden state.
No SQLite, no DeploymentStore, no deployment DB rows.

The hidden-state schema and its status transitions are owned exclusively by
github_state_proxy._validate_hidden_state (the production validator).
"""

from __future__ import annotations

from dataclasses import dataclass


# Machine owners configuration
@dataclass
class MachineInfo:
    alias: str
    node_id: str
    owners: list[str]  # GitHub logins (case-insensitive)
    node_host: str = ""
    targets: list[str] | None = None  # allowed targets (None = all)
    platforms: list[str] | None = None  # allowed platforms (None = all)
    variants: list[str] | None = None  # allowed variants (None = all)
    driver_paths: list[str] | None = None  # allowed driver paths (None = all)


@dataclass
class BuildInfo:
    """One build result from the Review Agent, as used in the lifecycle comment."""
    target: str
    driver_path: str
    variant: str
    success: bool
    image_tag: str
    deployable: bool  # computed by Controller: target=core|perception|actucore|driver
    component_id: str = ""


__all__ = [
    "MachineInfo",
    "BuildInfo",
]
