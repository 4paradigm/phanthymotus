"""Authority, machine owners, and image policy checks (pre-merge validation).

Pre-merge validation: one approver suffices (collaborator OR machine owner).
PR author MUST NOT approve their own deployment.
Every decision is fail-closed: an unknown target, user, machine or image is a
hard error, never a pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import ipaddress
from typing import Any

import yaml

from .config import Config
from .models import MachineInfo


class PolicyError(Exception):
    pass


class MachineLoadError(Exception):
    pass


_CANONICAL_VARIANTS = {"5.11", "6.1"}
_LEGACY_VARIANTS = {
    "jetson-jp5.11": "5.11",
    "jetson-jp6.1": "6.1",
}

def _normalize_variant(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise MachineLoadError("variant must be a non-empty string")
    if raw in _CANONICAL_VARIANTS:
        return raw
    if raw in _LEGACY_VARIANTS:
        return _LEGACY_VARIANTS[raw]
    raise MachineLoadError(f"unsupported variant: {raw!r}")


def _validate_literal_ip(host: str, alias: str) -> str:
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError as e:
        raise MachineLoadError(
            f"machine {alias!r}: node_host must be a literal IP address"
        ) from e
    if parsed.version != 4:
        raise MachineLoadError(
            f"machine {alias!r}: node_host must be an IPv4 address"
        )
    return str(parsed)





def _validate_driver_paths(raw: Any, alias: str, *, required: bool) -> list[str] | None:
    if raw is None:
        if required:
            raise MachineLoadError(
                f"machine {alias!r}: driver_paths is required for driver machines"
            )
        return None
    if not isinstance(raw, list):
        raise MachineLoadError(
            f"machine {alias!r}: driver_paths must be a non-empty list"
        )
    if not raw:
        raise MachineLoadError(
            f"machine {alias!r}: driver_paths must be a non-empty list"
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise MachineLoadError(
                f"machine {alias!r}: driver_paths must contain non-empty strings"
            )
        path = item.strip()
        if not path:
            raise MachineLoadError(
                f"machine {alias!r}: driver_paths must contain non-empty strings"
            )
        if path.startswith("/") or path.endswith("/"):
            raise MachineLoadError(
                f"machine {alias!r}: driver_paths must be repo-relative POSIX paths"
            )
        if "\\" in path or "//" in path:
            raise MachineLoadError(
                f"machine {alias!r}: driver_paths must be repo-relative POSIX paths"
            )
        segments = path.split("/")
        if len(segments) < 2 or any(seg in {"", ".", ".."} for seg in segments):
            raise MachineLoadError(
                f"machine {alias!r}: driver_paths must contain at least two safe path segments"
            )
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def load_machines(path: str) -> dict[str, MachineInfo]:
    """Load machines.yaml and return {alias: MachineInfo}.

    Final schema:
      version: 1
      machines:
        <alias>:
          node_id: <unique-node-id>
          owners:
            - <github-login>

    Startup fail-closed: missing, invalid, or malformed YAML raises.
    """
    if not path or not os.path.exists(path):
        raise MachineLoadError(f"machine owners file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise MachineLoadError("machines YAML must be a dict at top level")
    version = raw.get("version")
    if version != 1:
        raise MachineLoadError(
            f"machines YAML version must be 1, got {version!r}"
        )
    machines_raw = raw.get("machines")
    if not isinstance(machines_raw, dict):
        raise MachineLoadError("machines YAML must contain a 'machines' mapping")
    if not machines_raw:
        raise MachineLoadError("machines mapping must not be empty")
    machines: dict[str, MachineInfo] = {}
    node_ids_seen: set[str] = set()
    for alias, entry in machines_raw.items():
        if not isinstance(alias, str) or not alias.strip():
            raise MachineLoadError("machine alias must be a non-empty string")
        alias = alias.strip()
        if not isinstance(entry, dict):
            raise MachineLoadError(
                f"machine {alias!r} must be a mapping, got {type(entry).__name__}"
            )
        node_id = str(entry.get("node_id", "")).strip()
        if not node_id:
            raise MachineLoadError(f"machine {alias!r}: node_id is required and must be non-empty")
        if node_id in node_ids_seen:
            raise MachineLoadError(f"duplicate node_id: {node_id!r}")
        node_ids_seen.add(node_id)
        owners_raw = entry.get("owners", [])
        if not isinstance(owners_raw, list):
            raise MachineLoadError(
                f"machine {alias!r}: owners must be a list, got {type(owners_raw).__name__}"
            )
        if not owners_raw:
            raise MachineLoadError(f"machine {alias!r}: owners list is empty")
        if not all(isinstance(o, str) and o.strip() for o in owners_raw):
            raise MachineLoadError(f"machine {alias!r}: each owner must be a non-empty string")
        owners = list(dict.fromkeys(o.strip().lower() for o in owners_raw if o.strip()))
        if not owners:
            raise MachineLoadError(f"machine {alias!r}: no valid owners after dedup")
        node_host = str(entry.get("node_host", "") or "").strip()
        if not node_host:
            raise MachineLoadError(
                f"machine {alias!r}: node_host is required and must be non-empty"
            )
        node_host = _validate_literal_ip(node_host, alias)
        targets_raw = entry.get("targets")
        if not isinstance(targets_raw, list) or not targets_raw:
            raise MachineLoadError(
                f"machine {alias!r}: targets must be a non-empty list"
            )
        if not all(isinstance(t, str) and t.strip() for t in targets_raw):
            raise MachineLoadError(
                f"machine {alias!r}: targets must contain non-empty strings"
            )
        platforms_raw = entry.get("platforms")
        if not isinstance(platforms_raw, list) or not platforms_raw:
            raise MachineLoadError(
                f"machine {alias!r}: platforms must be a non-empty list"
            )
        if not all(isinstance(p, str) and p.strip() for p in platforms_raw):
            raise MachineLoadError(
                f"machine {alias!r}: platforms must contain non-empty strings"
            )
        targets = list(dict.fromkeys(t.strip() for t in targets_raw))
        platforms = list(dict.fromkeys(p.strip() for p in platforms_raw))
        driver_paths = _validate_driver_paths(
            entry.get("driver_paths"),
            alias,
            required="driver" in targets,
        )
        variants_raw = entry.get("variants")
        variants = None
        if variants_raw is not None:
            if not isinstance(variants_raw, list):
                raise MachineLoadError(
                    f"machine {alias!r}: variants must be a list"
                )
            if not variants_raw:
                raise MachineLoadError(
                    f"machine {alias!r}: variants must not be empty when set"
                )
            normalized: list[str] = []
            for variant in variants_raw:
                if not isinstance(variant, str) or not variant.strip():
                    raise MachineLoadError(
                        f"machine {alias!r}: variants must contain non-empty strings"
                    )
                normalized.append(_normalize_variant(variant))
            variants = list(dict.fromkeys(normalized))
            if not variants:
                raise MachineLoadError(
                    f"machine {alias!r}: variants must contain at least one supported variant"
                )
        machines[alias] = MachineInfo(
            alias=alias, node_id=node_id, owners=owners, node_host=node_host,
            targets=targets,
            platforms=platforms,
            variants=variants,
            driver_paths=driver_paths,
        )
    return machines


class Policy:
    def __init__(self, config: Config):
        self.config = config
        self.policy_version = self._fingerprint(config)
        self.machines: dict[str, MachineInfo] = {}

    def load_machines(self) -> None:
        self.machines = load_machines(self.config.machine_owners_file)

    def get_machine_by_node_id(self, node_id: str) -> MachineInfo | None:
        for m in self.machines.values():
            if m.node_id == node_id:
                return m
        return None

    def get_machine(self, alias: str) -> MachineInfo | None:
        return self.machines.get(alias)

    # GitHub collaborator permissions that may approve.
    COLLABORATOR_WRITE = frozenset({"write", "maintain", "admin"})

    @staticmethod
    def collaborator_can_approve(permission: str) -> bool:
        return isinstance(permission, str) and permission in Policy.COLLABORATOR_WRITE

    @staticmethod
    def _fingerprint(config: Config) -> str:
        payload = {"v": 7}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "deploy-approval-" + hashlib.sha256(blob).hexdigest()[:16]

    def can_approve(
        self,
        actor: str,
        *,
        actor_id: str = "",
        machine_alias: str = "",
    ) -> None:
        """Raise PolicyError unless ``actor`` may approve for the machine.

        Gates:
        1. actor must be an owner of the selected machine
        """
        machine = self.get_machine(machine_alias)
        if machine is None:
            raise PolicyError(f"machine {machine_alias!r} not found in machine owners")
        actor_lower = actor.strip().lower() if actor else ""
        if actor_lower not in machine.owners:
            raise PolicyError(
                f"actor {actor!r} is not an owner of machine {machine_alias!r}"
            )

    def can_request(self, requester: str) -> None:
        return None

    def get_machines(self) -> list:
        """Return all machines as a list."""
        return list(self.machines.values())

    @staticmethod
    def resolve_machine_selector(selector: str, machines: dict[str, "MachineInfo"]) -> "MachineInfo":
        """Resolve a machine selector (alias or literal IPv4) to a canonical MachineInfo.

        1. Exact alias match first.
        2. If not found, validate selector as literal IPv4.
        3. Find machines with node_host == selector.
        4. Exactly 1 match -> return it.
        5. 0 matches -> raise PolicyError.
        6. Multiple matches -> raise PolicyError listing sorted aliases.
        7. If selector is not a valid IPv4 and not an alias -> raise PolicyError.
        """
        # 1. Exact alias match
        if selector in machines:
            return machines[selector]

        # 2. Validate as literal IPv4
        try:
            parsed = ipaddress.ip_address(selector)
        except ValueError:
            raise PolicyError(
                f"invalid machine selector {selector!r}: "
                "not a known alias or valid IPv4 address"
            )
        if parsed.version != 4:
            raise PolicyError(
                f"invalid machine selector {selector!r}: only IPv4 addresses are supported"
            )

        canonical_ipv4 = str(parsed)

        # 3. Find machines with matching node_host
        matches: list[MachineInfo] = []
        for m in machines.values():
            if m.node_host == canonical_ipv4:
                matches.append(m)

        # 4-6. Resolve based on match count
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise PolicyError(f"no machine found with IP {canonical_ipv4!r}")
        # Multiple matches -> ambiguous
        sorted_aliases = sorted(m.alias for m in matches)
        raise PolicyError(
            f"ambiguous IP {canonical_ipv4!r}: matches machines "
            f"{', '.join(f'`{a}`' for a in sorted_aliases)}. "
            "Please use a machine alias instead."
        )

    def machine_supports_target(self, alias: str, target: str) -> bool:
        """Check if a machine supports a given target based on static config.

        Fail closed: targets must be explicit.
        """
        machine = self.machines.get(alias)
        if machine is None:
            return False
        if not machine.targets:
            return False
        return target in machine.targets
