"""Admin authorization scoped to one registered official QQ private principal."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from channel.tool_policy import _qq_owner_dm_identity, allowed_for_principals
# Load the production policy dependency before patching config.main in tests.
import event.llm  # noqa: F401

_OWNER = "official:1900000000:A1234567890BCDEF1234567890ABCDEF"
_GROUP = "official:1900000000:group:GroupOpenID12345:MemberOpenID12345"
_TOOLS = (
    "Bash", "Read", "Write", "PythonExec", "mcp__bumi__loco",
    "mcp__channel__channel_reply", "finish"
)


class QQOwnerACLTests(unittest.TestCase):
    def check(self, user_id, role, tool, filter_="*", grant=None):
        if grant is None:
            grant = {"enabled": True, "principal": _OWNER}
        with (mock.patch("channel.acl.get_user", return_value={
                "role": role, "tool_filter": filter_}),
              mock.patch("config.main", {"qq_c2c_owner_grant": grant})):
            return allowed_for_principals([("qq", user_id)], tool)

    def test_default_denies_privileged_owner_without_local_grant(self):
        for grant in ({}, {"enabled": False, "principal": _OWNER},
                      {"enabled": True, "principal": "official:1900000000:OtherOwner123456789"}):
            with self.subTest(grant=grant):
                self.assertFalse(self.check(_OWNER, "owner", "Bash", grant=grant))
                self.assertTrue(self.check(_OWNER, "owner", "mcp__channel__channel_reply", grant=grant))

    def test_owner_c2c_can_call_all_tools(self):
        self.assertTrue(_qq_owner_dm_identity(_OWNER))
        for tool in _TOOLS:
            with self.subTest(tool=tool):
                self.assertTrue(self.check(_OWNER, "owner", tool))

    def test_group_owner_does_not_override_platform_cap(self):
        self.assertFalse(_qq_owner_dm_identity(_GROUP))
        for tool in ("Bash", "Read", "Write", "mcp__bumi__loco"):
            with self.subTest(tool=tool):
                self.assertFalse(self.check(_GROUP, "owner", tool))
        # New QQ groups require explicit binding; even ordinary replies are
        # denied before approval, regardless of a fabricated Owner role.
        self.assertFalse(self.check(_GROUP, "owner", "mcp__channel__channel_reply"))

    def test_regular_qq_viewer_and_operator_are_restricted(self):
        for role in ("viewer", "operator"):
            with self.subTest(role=role):
                self.assertFalse(self.check(_OWNER, role, "Bash"))
                self.assertFalse(self.check(_OWNER, role, "mcp__perception__tts"))
                self.assertTrue(self.check(_OWNER, role, "mcp__channel__channel_reply"))

    def test_cannot_forge_owner_through_other_principal_syntax(self):
        malformed = (
            "official:1900000000:group:GroupOpenID12345:Someone",
            "official:1900000000:guild_dm:AnId:Someone",
            "official:1900000000:abc",
            "official:notdigits:A1234567890BCDEF1234567890ABCDEF",
            "other:1900000000:A1234567890BCDEF1234567890ABCDEF",
        )
        for name in malformed:
            with self.subTest(name=name):
                self.assertFalse(self.check(name, "owner", "Bash"))

    def test_explicit_tool_filter_still_applies(self):
        self.assertFalse(self.check(_OWNER, "owner", "Bash", "Read"))
        self.assertTrue(self.check(_OWNER, "owner", "Read", "Read"))

    def test_blocked_owner_and_missing_user_fail_closed(self):
        self.assertFalse(self.check(_OWNER, "blocked", "Read"))
        with mock.patch("channel.acl.get_user", return_value=None):
            self.assertFalse(allowed_for_principals(
                [("qq", _OWNER)], "mcp__channel__channel_reply"))

    def test_mixed_viewer_and_owner_batch_stays_restricted(self):
        with mock.patch("channel.acl.get_user", side_effect=lambda p, u: {
                "role": "owner" if u == _OWNER else "viewer",
                "tool_filter": "*"}):
            self.assertFalse(allowed_for_principals(
                [("qq", _OWNER), ("qq", _GROUP)], "Bash"))


if __name__ == "__main__":
    unittest.main()
