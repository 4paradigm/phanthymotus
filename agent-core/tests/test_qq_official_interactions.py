"""QQ interactive callback and prompt isolation, without a live Gateway."""
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from channel.adapters.qq_official_interactions import QQInteractionRegistry


def _raw_event(button, user="PersonOpenId1", chat="", group="", member=""):
    return {"event": types.SimpleNamespace(
        id="interaction-id", user_openid=user,
        group_openid=group, group_member_openid=member,
        data=types.SimpleNamespace(resolved=types.SimpleNamespace(button_data=button)),
    )}


class QQInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = QQInteractionRegistry()
        self.api = types.SimpleNamespace(
            acknowledge_interaction=mock.AsyncMock(return_value=None),
        )
        self.fake = types.SimpleNamespace(
            parse_interaction_event=lambda raw: raw["event"],
            parse_approval_button_data=lambda data:
                tuple(data.split(":")[1:]) if data.startswith("approve:") else None,
        )
        m = mock.patch.dict(sys.modules, {"qqbot_agent_sdk": self.fake})
        m.start()
        self.addCleanup(m.stop)

    async def test_private_callback_exact_actor_and_single_use(self):
        token = self.registry.issue("approval", "c2c", "PersonOpenId1",
                                    "PersonOpenId1", ("allow-once", "allow-always", "deny"))
        message = _raw_event("approve:" + token + ":allow-once")
        self.assertTrue(await self.registry.handle(self.api, "INTERACTION_CREATE", message))
        self.assertEqual(self.registry.take(token), "allow-once")
        self.assertIsNone(self.registry.take(token))
        self.assertFalse(await self.registry.handle(self.api, "INTERACTION_CREATE", message))
        self.api.acknowledge_interaction.assert_awaited_once_with(
            "interaction-id", code=0)

    async def test_group_callback_cannot_be_approved_by_other_member(self):
        token = self.registry.issue("approval", "group", "GroupOpenId1",
                                    "PersonOpenId1", ("allow-once", "allow-always", "deny"))
        msg = _raw_event("approve:" + token + ":allow-once", group="GroupOpenId1",
                         member="WrongMember")
        self.assertFalse(await self.registry.handle(self.api, "INTERACTION_CREATE", msg))
        self.assertIsNone(self.registry.take(token))
        msg = _raw_event("approve:" + token + ":allow-once", group="GroupOpenId1",
                         member="PersonOpenId1")
        self.assertTrue(await self.registry.handle(self.api, "INTERACTION_CREATE", msg))

    async def test_sdk_always_button_is_not_permanent_grant(self):
        token = self.registry.issue("approval", "c2c", "PersonOpenId1",
                                    "PersonOpenId1", ("allow-once", "allow-always", "deny"))
        msg = _raw_event("approve:" + token + ":allow-always")
        self.assertTrue(await self.registry.handle(self.api, "INTERACTION_CREATE", msg))
        self.assertEqual(self.registry.take(token), "allow-once")

    async def test_unknown_and_wrong_scope_callbacks_rejected(self):
        token = self.registry.issue("choice", "c2c", "PersonOpenId1",
                                    "PersonOpenId1", ("yes", "no"))
        self.assertFalse(await self.registry.handle(
            self.api, "INTERACTION_CREATE", _raw_event("qqchoice:bad:yes")))
        self.assertFalse(await self.registry.handle(
            self.api, "INTERACTION_CREATE",
            _raw_event("qqchoice:" + token + ":yes", user="WrongPerson1")))
        self.api.acknowledge_interaction.assert_not_called()

    async def test_unregistered_sdk_event_ignored(self):
        self.assertFalse(await self.registry.handle(self.api, "UNKNOWN", {}))
        self.api.acknowledge_interaction.assert_not_called()


if __name__ == "__main__":
    unittest.main()
