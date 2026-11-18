"""Legacy QQ owner utility. Promotion has been retired.

Only --revoke is retained for cleaning up existing owner grants. QQ chat does
not require roles and is never permitted to execute robot/system tools.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def update_owner(*, revoke: bool = False) -> dict:
    if not revoke:
        raise ValueError("QQ administrator promotion is retired; use Dashboard to configure channels")
    import config
    from channel import acl
    from channel.tool_policy import _qq_owner_dm_identity
    from channel.permissions import CHAT_FILTER

    channel_configs = config.main.get("channel_configs", [])
    active_app_ids = {
        str(ch.get("config", {}).get("app_id", ""))
        for ch in channel_configs
        if ch.get("enabled") and ch.get("platform") == "qq"
        and ch.get("config", {}).get("protocol") == "official"
    }
    if len(active_app_ids) != 1:
        raise ValueError("Expected exactly one active official QQ AppID")
    appid = next(iter(active_app_ids))
    candidates = [
        x for x in acl.list_users("qq")
        if _qq_owner_dm_identity(x.get("user_id", ""))
        and x["user_id"].split(":")[1] == appid
    ]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one registered private QQ user")
    user = candidates[0]
    if user["role"] == "blocked":
        raise ValueError("Refusing to promote a blocked account")
    # Revoke before demoting: never leave a window of elevated ACL state.
    config.main["qq_c2c_owner_grant"] = {"enabled": False}
    updated = acl.upsert_user(
        "qq", user["user_id"], display_name=user.get("display_name", ""),
        role="viewer", tool_filter=CHAT_FILTER,
    )
    return {
        "role": updated["role"],
        "privileged_tools_enabled": False,
        "principal_hash": hashlib.sha256(user["user_id"].encode()).hexdigest()[:12],
        "is_private_chat": True,
        "other_qq_users_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--promote", action="store_true", help="retired (always refused)")
    group.add_argument("--revoke", action="store_true")
    args = parser.parse_args()
    try:
        result = update_owner(revoke=args.revoke)
    except ValueError as e:
        print("QQ_C2C_OWNER_UPDATE_FAILED", str(e))
        return 1
    print("QQ_C2C_OWNER_UPDATED", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
