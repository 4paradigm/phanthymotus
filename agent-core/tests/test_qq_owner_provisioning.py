"""Privileged QQ access must be local opt-in, scoped and revocable."""
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from qq_promote_c2c_owner import update_owner
from channel.tool_policy import allowed_for_principals
# Load the production policy dependency before patching config.main in tests.
import event.llm  # noqa: F401

UID = "official:1900000000:A1234567890BCDEF1234567890ABCDEF"
OTHER = "official:1900000000:BB234567890BCDEF1234567890ABCDEF"


def config_data():
    return {"channel_configs": [{
        "id": "qq", "platform": "qq", "enabled": True,
        "config": {"protocol": "official", "app_id": "1900000000"}}]}


class QQOwnerProvisioningTests(unittest.TestCase):
    def provision(self, *, revoke=False, users=None):
        cfg = config_data()
        if users is None:
            users = [{"user_id": UID, "role": "viewer", "display_name": "tester"}]
        def fake_upsert(platform, identity, **kw):
            return {"role": kw["role"]}
        with (mock.patch("config.main", cfg),
              mock.patch("channel.acl.list_users", return_value=users),
              mock.patch("channel.acl.upsert_user", side_effect=fake_upsert)):
            response = update_owner(revoke=revoke)
        return response, cfg

    def test_local_provision_explicit_exact_user(self):
        response, cfg = self.provision()
        self.assertEqual(response["role"], "owner")
        self.assertTrue(response["privileged_tools_enabled"])
        self.assertEqual(cfg["qq_c2c_owner_grant"], {
            "enabled": True, "principal": UID,
        })
        self.assertNotIn(UID, repr(response))

    def test_revocation_turns_off_privilege_first(self):
        response, cfg = self.provision(revoke=True)
        self.assertEqual(response["role"], "viewer")
        self.assertFalse(response["privileged_tools_enabled"])
        self.assertEqual(cfg["qq_c2c_owner_grant"], {"enabled": False})

    def test_multiple_identities_fail_without_grant(self):
        users = [{"user_id": UID, "role": "viewer"},
                 {"user_id": OTHER, "role": "viewer"}]
        with (mock.patch("config.main", config_data()),
              mock.patch("channel.acl.list_users", return_value=users),
              mock.patch("channel.acl.upsert_user") as upsert):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                update_owner()
            upsert.assert_not_called()

    def test_grant_without_matching_acl_owner_does_not_allow_shell(self):
        cfg = config_data()
        cfg["qq_c2c_owner_grant"] = {"enabled": True, "principal": UID}
        with (mock.patch("config.main", cfg),
              mock.patch("channel.acl.get_user", return_value={
                  "role": "viewer", "tool_filter": "*" })):
            self.assertFalse(allowed_for_principals([("qq", UID)], "Bash"))
        with (mock.patch("config.main", cfg),
              mock.patch("channel.acl.get_user", return_value={
                  "role": "owner", "tool_filter": "*" })):
            self.assertTrue(allowed_for_principals([("qq", UID)], "Bash"))
            self.assertFalse(allowed_for_principals([("qq", OTHER)], "Bash"))


if __name__ == "__main__":
    unittest.main()
