"""Offline QQ capability preflight: never reads or prints credential contents.

Reports configuration readiness, not Tencent account entitlement or live delivery.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys


FEATURE_FLAGS = (
    "enable_inbound_media", "enable_outbound_media",
    "enable_markdown", "enable_typing",
    "enable_session_resume", "enable_interactions",
    "enable_approvals", "enable_voice_transcription",
    "enable_guild_plain_messages",
)


def inspect_database(path: Path, channel_id: str) -> dict:
    if not path.is_file():
        raise ValueError("Core database not found")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
        def load(key, default):
            row = database.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
            ).fetchone()
            return json.loads(row[0]) if row else default

        channels = load("channel_configs", [])
        channel = next((
            c for c in channels
            if c.get("id") == channel_id and c.get("platform") == "qq"
        ), None)
        if channel is None:
            raise ValueError("QQ channel not found")
        cfg = channel.get("config") or {}
        event = load("event", {})
        scopes = cfg.get("enabled_scopes", ["c2c"])
        sdk_version = None
        try:
            sdk_version = importlib.metadata.version("qqbot-agent-sdk")
        except importlib.metadata.PackageNotFoundError:
            pass
        vision = bool(event.get("llm", {}).get("vision_input", False))
        result = {
            "channel_enabled": bool(channel.get("enabled", False)),
            "sdk_version": sdk_version,
            "scopes": scopes,
            "features_enabled": {
                flag: cfg.get(flag, False) is True for flag in FEATURE_FLAGS
            },
            "vision_input_enabled": vision,
            "credential_reference_present": bool(
                cfg.get("app_secret_file") or cfg.get("app_secret_env")
            ),
            "ready_for_platform_test": [],
            "not_ready_or_unverified": [],
        }
        enabled = result["features_enabled"]
        if not sdk_version:
            result["not_ready_or_unverified"].append("SDK not installed")
        if "c2c" not in scopes:
            result["not_ready_or_unverified"].append("C2C scope disabled")
        if "group" not in scopes:
            result["not_ready_or_unverified"].append("Group scope disabled")
        if enabled["enable_inbound_media"]:
            result["ready_for_platform_test"].append("media download")
        else:
            result["not_ready_or_unverified"].append("media download disabled")
        if enabled["enable_outbound_media"]:
            result["ready_for_platform_test"].append("media upload")
        else:
            result["not_ready_or_unverified"].append("media upload disabled")
        if enabled["enable_voice_transcription"]:
            result["ready_for_platform_test"].append("voice transcription")
        else:
            result["not_ready_or_unverified"].append("voice transcription disabled")
        if not vision:
            result["not_ready_or_unverified"].append("image understanding requires VLM")
        else:
            result["not_ready_or_unverified"].append("VLM/provider/image bridge not verified")
        for flag in ("enable_markdown", "enable_typing", "enable_session_resume",
                     "enable_interactions", "enable_approvals",
                     "enable_guild_plain_messages"):
            if enabled[flag]:
                result["ready_for_platform_test"].append(flag)
            else:
                result["not_ready_or_unverified"].append(flag + " disabled")
        result["not_ready_or_unverified"].append(
            "QQ platform entitlements require real Gateway/message checks")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("/work/resource/data.db"))
    parser.add_argument("--channel", default="qq_official_v2_acceptance")
    args = parser.parse_args(argv)
    try:
        report = inspect_database(args.database, args.channel)
    except (sqlite3.Error, ValueError, OSError, json.JSONDecodeError) as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
