"""Static physical-machine inventory, separate from repository authorization.

This file contains identity and verified descriptive metadata only. The supplied
driver-directory snapshot reference is 814b3ca953d33eba0aca9a61dcefa773f5da483b.
This file must not be used to authorize deployments or hold credentials. The former
``p200pro-011`` record was removed and is intentionally not assigned to a
current machine; the old ``x2-012`` and ``rm75-014`` identities map to their
current physical-machine aliases below.
"""

from __future__ import annotations

import copy as _copy
import ipaddress as _ipaddress

__all__ = ("get_machine_inventory",)

_OWNER_SOURCE_V16 = "V16 legacy machines.yaml SHA16=d4157deaf4b8bb86"
_DRIVER_PATH_SNAPSHOT = "814b3ca953d33eba0aca9a61dcefa773f5da483b"
_PLATFORM_LINUX_ARM64 = "linux/arm64"
_PLATFORM_SOURCE_EXPLICIT = (
    "PHANTHYMOTUS_V44U explicit machine platform verification evidence"
)
_PLATFORM_SOURCE_RM75 = (
    f"{_OWNER_SOURCE_V16}: rm75-014 platform=linux/arm64; migrated to rm75-013 "
    "through the retained legacy_aliases mapping and matching driver_path="
    "realman/rm75_6f_v"
)


def _record(
    number: str,
    alias: str,
    brand: str,
    model: str,
    primary_ip: str,
    *,
    secondary_ips: tuple[str, ...] = (),
    owners: tuple[str, ...] = (),
    owner_status: str = "pending",
    owner_source: str = "pending",
    legacy_aliases: tuple[str, ...] = (),
    driver_path: str | None = None,
    driver_path_verified: bool = False,
    platforms: tuple[str, ...] = (),
    platform_status: str = "unverified",
    platform_source: str | None = None,
) -> dict:
    """Build one fresh, JSON-compatible record from immutable literals."""
    return {
        "number": number,
        "alias": alias,
        "brand": brand,
        "model": model,
        "primary_ip": primary_ip,
        "secondary_ips": list(secondary_ips),
        "owners": list(owners),
        "owner_status": owner_status,
        "owner_source": owner_source,
        "legacy_aliases": list(legacy_aliases),
        "driver_path": driver_path,
        "driver_path_verified": driver_path_verified,
        "platforms": list(platforms),
        "platform_status": platform_status,
        "platform_source": platform_source,
    }


_VERIFIED_OWNER = ("haohao-end",)

_INVENTORY = (
    _record("001", "beijing-g1-001", "宇树", "G1北京", "10.100.129.168",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="unitree/g1", driver_path_verified=True,
            platforms=(_PLATFORM_LINUX_ARM64,), platform_status="verified",
            platform_source=f"{_PLATFORM_SOURCE_EXPLICIT}: beijing-g1-001={_PLATFORM_LINUX_ARM64}"),
    _record("002", "unitree-r1-002", "宇树", "R1", "10.100.130.6",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="unitree/r1", driver_path_verified=True),
    _record("003", "unitree-go2-003", "宇树", "Go2", "10.100.129.166",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="unitree/go2", driver_path_verified=True),
    _record("004", "unitree-go1-004", "宇树", "Go1", "10.100.130.4",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="unitree/go1", driver_path_verified=True),
    _record("005", "tianyi2-005", "北京人形", "天轶2.0", "10.100.129.72",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="x-humanoid/tianyi2.0", driver_path_verified=True,
            platforms=(_PLATFORM_LINUX_ARM64,), platform_status="verified",
            platform_source=f"{_PLATFORM_SOURCE_EXPLICIT}: tianyi2-005={_PLATFORM_LINUX_ARM64}"),
    _record("006", "bumi-006", "松延动力", "Bumi-Edu-Max", "10.100.128.245",
            secondary_ips=("192.168.55.101",),
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="noetix/bumi", driver_path_verified=True),
    _record("007", "m20pro-007", "云深处", "M20 Pro", "10.110.15.219",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="deep_robotics/lynx_m20", driver_path_verified=True),
    _record("008", "t800-008", "众擎", "T800", "10.110.10.96",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="engineai/t800", driver_path_verified=True),
    _record("009", "shanghai-g1-009", "宇树", "G1上海", "10.110.12.110",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="unitree/g1", driver_path_verified=True),
    _record("010", "q5-010", "星动纪元", "Q5", "10.100.128.225",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            driver_path="robotera/q5_bundle", driver_path_verified=True,
            platforms=(_PLATFORM_LINUX_ARM64,), platform_status="verified",
            platform_source=f"{_PLATFORM_SOURCE_EXPLICIT}: q5-010={_PLATFORM_LINUX_ARM64}"),
    _record("011", "x2-011", "智元机器人", "灵犀X2旗舰版", "10.100.129.195",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            legacy_aliases=("x2-012",),
            driver_path="agibot/AimDK_X2", driver_path_verified=True),
    _record("012", "adam-pro-012", "柏奥尼克", "Adam Pro", "10.100.128.154",
            secondary_ips=("10.10.20.126",),
            driver_path="pndbotics/adam", driver_path_verified=True),
    _record("013", "rm75-013", "睿尔曼", "RM75-6f-v", "10.100.130.37",
            owners=_VERIFIED_OWNER, owner_status="verified", owner_source=_OWNER_SOURCE_V16,
            legacy_aliases=("rm75-014",),
            driver_path="realman/rm75_6f_v", driver_path_verified=True,
            platforms=(_PLATFORM_LINUX_ARM64,), platform_status="verified",
            platform_source=_PLATFORM_SOURCE_RM75),
    _record("014", "as2w-014", "宇树", "As2 W", "10.100.129.146",
            secondary_ips=("10.100.128.245",),
            driver_path="unitree/as2w", driver_path_verified=True),
    _record("015", "r1-blue-015", "宇树", "R1蓝白", "10.100.129.73",
            driver_path="unitree/r1", driver_path_verified=True),
    _record("016", "a3ultra-016", "智元机器人", "远征A3Ultra", "10.100.129.135",
            secondary_ips=("10.100.129.63",),
            driver_path="agibot/A3", driver_path_verified=True),
    _record("017", "u1pro-017", "优必选", "U1 Pro", "10.100.129.28",
            driver_path="ubtrobot/u1_pro", driver_path_verified=True),
    _record("018", "piper-018", "松灵AgileX", "Piper上海", "10.110.10.46",
            driver_path="agilex/piper", driver_path_verified=True),
    _record("019", "tron2-019", "逐际动力", "Tron2上海", "10.110.8.244",
            driver_path="limx/tron2", driver_path_verified=True),
    _record("020", "s10pro-020", "云深处", "山猫S10-pro", "10.100.128.127",
            driver_path="deep_robotics/s10_pro", driver_path_verified=True),
    _record("021", "c23-021", "越疆", "C23", "10.100.128.229"),
)

_REQUIRED_FIELDS = frozenset({
    "number", "alias", "brand", "model", "primary_ip", "secondary_ips",
    "owners", "owner_status", "owner_source", "legacy_aliases", "driver_path",
    "driver_path_verified", "platforms", "platform_status", "platform_source",
})


def _validate_static_inventory() -> None:
    """Fail closed if static inventory literals are malformed or incomplete."""
    if len(_INVENTORY) != 21:
        raise RuntimeError("physical machine inventory must contain exactly 21 records")
    verified_paths = [machine for machine in _INVENTORY
                      if machine["driver_path_verified"] is True]
    if len(verified_paths) != 20:
        raise RuntimeError("exactly 20 physical machines must have verified driver paths")
    verified_platforms = {
        machine["alias"] for machine in _INVENTORY
        if machine["platform_status"] == "verified"
    }
    if verified_platforms != {
        "beijing-g1-001", "tianyi2-005", "q5-010", "rm75-013",
    }:
        raise RuntimeError("only evidence-backed physical machines may have verified platforms")
    if sum(machine["owner_status"] == "verified" for machine in _INVENTORY) != 12:
        raise RuntimeError("exactly the 12 historical owners must remain verified")
    if sum(machine["owner_status"] == "pending" for machine in _INVENTORY) != 9:
        raise RuntimeError("the nine newly added machine owners must remain pending")
    numbers: set[str] = set()
    aliases: set[str] = set()
    for expected_number, machine in enumerate(_INVENTORY, start=1):
        if set(machine) != _REQUIRED_FIELDS:
            raise RuntimeError("physical machine inventory record has invalid fields")
        if machine["number"] != f"{expected_number:03d}":
            raise RuntimeError("physical machine inventory numbers must be ordered 001-021")
        if machine["number"] in numbers or machine["alias"] in aliases:
            raise RuntimeError("physical machine inventory numbers and aliases must be unique")
        numbers.add(machine["number"])
        aliases.add(machine["alias"])
        addresses = [machine["primary_ip"], *machine["secondary_ips"]]
        for address in addresses:
            try:
                parsed = _ipaddress.ip_address(address)
            except ValueError as exc:
                raise RuntimeError("physical machine inventory contains an invalid IP") from exc
            if parsed.version != 4:
                raise RuntimeError("physical machine inventory IPs must be IPv4")
        if len(addresses) != len(set(addresses)):
            raise RuntimeError("one machine record must not repeat an IP")
        if machine["owner_status"] not in {"verified", "pending"}:
            raise RuntimeError("physical machine owner status is invalid")
        if machine["platform_status"] not in {"verified", "unverified"}:
            raise RuntimeError("physical machine platform status is invalid")
        if type(machine["driver_path_verified"]) is not bool:
            raise RuntimeError("driver path verification status must be boolean")
        if machine["driver_path_verified"] and not machine["driver_path"]:
            raise RuntimeError("verified driver path must have a value")
        if machine["driver_path_verified"]:
            path = machine["driver_path"]
            if (not isinstance(path, str) or path.startswith("/")
                    or any(part in {"", ".", ".."} for part in path.split("/"))):
                raise RuntimeError("verified driver path must be repo-relative")
        if not machine["driver_path_verified"] and machine["driver_path"] is not None:
            raise RuntimeError("unverified driver path must not be guessed")
        if machine["platform_status"] == "verified" and not machine["platforms"]:
            raise RuntimeError("verified platform status must have at least one platform")
        if machine["platform_status"] == "verified" and (
                not isinstance(machine["platform_source"], str)
                or not machine["platform_source"].strip()):
            raise RuntimeError("verified platform must retain a traceable evidence source")
        if machine["platform_status"] == "unverified" and (
                machine["platforms"] or machine["platform_source"] is not None):
            raise RuntimeError("unverified platform must not include a value or fabricated source")
        if machine["owner_status"] == "verified" and not machine["owners"]:
            raise RuntimeError("verified owner status must have at least one owner")
        if machine["owner_status"] == "verified" and machine["owner_source"] != _OWNER_SOURCE_V16:
            raise RuntimeError("verified historical owner must retain its source")
        if machine["owner_status"] == "pending" and (
                machine["owners"] or machine["owner_source"] != "pending"):
            raise RuntimeError("pending machine owners must remain unassigned")

    by_alias = {machine["alias"]: machine for machine in _INVENTORY}
    if by_alias["x2-011"]["legacy_aliases"] != ["x2-012"]:
        raise RuntimeError("X2 historical alias must remain linked to x2-011")
    if (by_alias["rm75-013"]["legacy_aliases"] != ["rm75-014"]
            or by_alias["rm75-013"]["driver_path"] != "realman/rm75_6f_v"
            or by_alias["rm75-013"]["platform_source"] != _PLATFORM_SOURCE_RM75):
        raise RuntimeError("RM75 platform migration must stay bound to its verified legacy identity")
    if (by_alias["c23-021"]["driver_path"] is not None
            or by_alias["c23-021"]["driver_path_verified"] is not False):
        raise RuntimeError("C23 driver path must remain unverified")


def get_machine_inventory() -> list[dict]:
    """Return the 21 static physical machines as a fresh independent copy."""
    _validate_static_inventory()
    return _copy.deepcopy(list(_INVENTORY))
