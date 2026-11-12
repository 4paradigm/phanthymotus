"""Local administrator utility: promote only the unique registered QQ C2C principal.

No API endpoint, no QQ message command, no implicit grant to groups. Restore
viewer with --revoke. The platform AppID must match the active official adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

sys.path.insert(0, "/work/src")


def update_owner(*, revoke: bool = False) -> dict:
    import config
    from channel import acl
    from channel.tool_policy import _qq_owner_dm_identity

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
    new_role = "viewer" if revoke else "owner"
    updated = acl.upsert_user(
        "qq", user["user_id"], display_name=user.get("display_name", ""),
        role=new_role,
        tool_filter="*",
    )
    return {
        "role": updated["role"],
        "principal_hash": hashlib.sha256(user["user_id"].encode()).hexdigest()[:12],
        "is_private_chat": True,
        "other_qq_users_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--promote", action="store_true")
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
