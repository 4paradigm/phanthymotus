from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from execution_control import ManualExecutionController  # noqa: E402


PROPOSAL_TOPIC = "/ubuntu/navigation/nav2/velocity_proposal"
PROPOSAL_SCHEMA = "phanthy.navigation.velocity_proposal.v1"


def _content(**payload) -> dict:
    return {
        "code": 200,
        "data": [
            {
                "type": "text",
                "text": json.dumps(payload),
            }
        ],
    }


def _mcps() -> list:
    return [
        {
            "id": "driver-perception",
            "tools": [
                {
                    "name": "controlled_semantic_spatial",
                    "x-execution-control": {
                        "version": 2,
                        "proposal_schema": PROPOSAL_SCHEMA,
                        "output_port": "velocity_proposal",
                        "target_tool": "loco",
                        "lease_argument": "_control_nav_id",
                        "authorize_action": "authorize_navigation",
                        "revoke_action": "revoke_navigation",
                        "nav_id_argument": "nav_id",
                        "proposal_topic_argument": "proposal_topic",
                        "proposal_schema_argument": "proposal_schema",
                        "start_actions": ["navigate_to_pose"],
                        "stop_actions": ["stop_nav"],
                    },
                    "topic_out": [
                        {
                            "port": "velocity_proposal",
                            "topic": PROPOSAL_TOPIC,
                            "schema": PROPOSAL_SCHEMA,
                        }
                    ],
                }
            ],
        },
        {
            "id": "driver-unitree-g1",
            "tools": [
                {
                    "name": "loco",
                    "topic_in": [
                        {
                            "port": "velocity_proposal",
                            "topic": PROPOSAL_TOPIC,
                            "schema": PROPOSAL_SCHEMA,
                        }
                    ],
                }
            ],
        },
    ]


class ManualExecutionControllerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.controller = ManualExecutionController()
        self.calls = []
        self.authorize_error = None
        self.source_error = None
        self.driver_active_nav_id = None

    async def _raw_call(self, mcp_id: str, tool: str, arguments: dict) -> dict:
        self.calls.append((mcp_id, tool, dict(arguments)))
        action = arguments.get("action")
        if action == "authorize_navigation":
            if self.authorize_error or self.driver_active_nav_id:
                return _content(
                    status="error",
                    error_code="navigation_active",
                    error=self.authorize_error
                    or "another navigation is still active",
                )
            self.driver_active_nav_id = arguments["nav_id"]
            return _content(status="authorized", nav_id=arguments["nav_id"])
        if action == "revoke_navigation":
            if self.driver_active_nav_id == arguments["nav_id"]:
                self.driver_active_nav_id = None
            return _content(status="revoked", nav_id=arguments["nav_id"])
        if action == "navigate_to_pose":
            if self.source_error:
                return _content(
                    status="error",
                    error_code="navigation_not_ready",
                    error=self.source_error,
                )
            return _content(
                status="navigating",
                nav_id=arguments["_control_nav_id"],
            )
        if action == "stop_nav":
            return _content(status="stopped")
        return _content(status="ok")

    async def _navigate(self) -> dict:
        return await self.controller.call_manual(
            mcps=_mcps(),
            source_mcp_id="driver-perception",
            source_tool_name="controlled_semantic_spatial",
            arguments={
                "action": "navigate_to_pose",
                "x": 1.0,
                "y": 2.0,
                "yaw": 0.0,
                "speed": 0.5,
            },
            raw_call=self._raw_call,
        )

    async def test_each_manual_navigation_gets_a_new_authorization(self) -> None:
        first = await self._navigate()
        self.driver_active_nav_id = None  # previous task reached a terminal state
        second = await self._navigate()

        self.assertEqual(first["code"], 200)
        self.assertEqual(second["code"], 200)
        authorize_calls = [
            call
            for call in self.calls
            if call[2].get("action") == "authorize_navigation"
        ]
        navigate_calls = [
            call for call in self.calls if call[2].get("action") == "navigate_to_pose"
        ]
        self.assertEqual(len(authorize_calls), 2)
        self.assertEqual(len(navigate_calls), 2)
        first_nav_id = authorize_calls[0][2]["nav_id"]
        second_nav_id = authorize_calls[1][2]["nav_id"]
        self.assertNotEqual(first_nav_id, second_nav_id)
        self.assertEqual(navigate_calls[0][2]["_control_nav_id"], first_nav_id)
        self.assertEqual(navigate_calls[1][2]["_control_nav_id"], second_nav_id)
        self.assertEqual(
            authorize_calls[0][2]["proposal_topic"], PROPOSAL_TOPIC
        )
        self.assertEqual(
            authorize_calls[0][2]["proposal_schema"], PROPOSAL_SCHEMA
        )

    async def test_active_driver_rejection_does_not_replace_source_goal(self) -> None:
        self.authorize_error = "another navigation is still active"

        response = await self._navigate()

        payload = json.loads(response["data"][0]["text"])
        self.assertEqual(payload["error_code"], "execution_authorization_failed")
        self.assertEqual(
            [call[2]["action"] for call in self.calls],
            ["authorize_navigation"],
        )

    async def test_source_rejection_revokes_the_new_authorization(self) -> None:
        self.source_error = "planner is not ready"

        response = await self._navigate()

        self.assertEqual(
            [call[2]["action"] for call in self.calls],
            ["authorize_navigation", "navigate_to_pose", "revoke_navigation"],
        )
        self.assertEqual(
            self.calls[0][2]["nav_id"],
            self.calls[2][2]["nav_id"],
        )
        payload = json.loads(response["data"][0]["text"])
        self.assertEqual(payload["error_code"], "navigation_not_ready")

    async def test_manual_stop_revokes_the_last_authorized_navigation(self) -> None:
        await self._navigate()
        self.calls.clear()

        response = await self.controller.call_manual(
            mcps=_mcps(),
            source_mcp_id="driver-perception",
            source_tool_name="controlled_semantic_spatial",
            arguments={"action": "stop_nav"},
            raw_call=self._raw_call,
        )

        self.assertEqual(response["code"], 200)
        self.assertEqual(
            [call[2]["action"] for call in self.calls],
            ["stop_nav", "revoke_navigation"],
        )

    async def test_non_controlled_action_remains_a_direct_call(self) -> None:
        await self.controller.call_manual(
            mcps=_mcps(),
            source_mcp_id="driver-perception",
            source_tool_name="controlled_semantic_spatial",
            arguments={"action": "info"},
            raw_call=self._raw_call,
        )

        self.assertEqual(
            [call[2]["action"] for call in self.calls],
            ["info"],
        )


if __name__ == "__main__":
    unittest.main()
