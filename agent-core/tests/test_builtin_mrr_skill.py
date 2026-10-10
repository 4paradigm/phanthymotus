"""The multi-robot registration runtime is shipped in the Agent Core image.

These tests protect the fast path: the model invokes a prebuilt script instead
of recreating or installing Python during a registration conversation.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import uuid


AGENT_CORE = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = AGENT_CORE / "skills/multi-robot-registration/scripts/mrr_runtime.py"
SKILL = AGENT_CORE / "skills/multi-robot-registration/SKILL.md"


def _run(root: pathlib.Path, command: str, payload: dict) -> dict:
    env = dict(os.environ, MRR_ROOT=str(root))
    result = subprocess.run(
        [sys.executable, str(RUNTIME), command],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


class BuiltinMrrSkillTests(unittest.TestCase):
    def test_runtime_is_baked_into_agent_core_image(self):
        dockerfile = (AGENT_CORE / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY skills/   /work/skills/", dockerfile)
        self.assertTrue(RUNTIME.is_file())
        self.assertTrue(SKILL.is_file())

    def test_runtime_reports_expected_version(self):
        result = subprocess.run(
            [sys.executable, str(RUNTIME), "--version"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "8.0")

    def test_prebuilt_runtime_handles_short_employee_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            round_id = uuid.uuid4().hex
            leader = {"peer_id": "g1", "name": "g1"}
            start = {
                "marker": "MRR_CONTROL_V1",
                "protocol": "mrr-v1",
                "command": "START",
                "request_id": uuid.uuid4().hex,
                "round_id": round_id,
                "leader": leader,
                "config_version": 1,
                "sent_at": "2026-10-10T00:00:00+00:00",
                "payload": {
                    "target": leader,
                    "team_members": [],
                    "config": {
                        "activity_name": "午餐登记",
                        "questions": [
                            {
                                "id": "meal",
                                "prompt": "请选择午餐",
                                "type": "single_choice",
                                "options": ["A", "B"],
                                "required": True,
                            }
                        ],
                        "duplicate_rule": "latest_success",
                    },
                },
            }

            started = _run(root, "control-leader", start)
            self.assertEqual(started["status"], "ok")

            begun = _run(root, "begin", {})
            self.assertEqual(begun["status"], "ok")
            self.assertEqual(begun["config"]["activity_name"], "午餐登记")

            committed = _run(
                root,
                "commit",
                {
                    "session_id": begun["session_id"],
                    "employee": {"stable_id": "employee-1", "name": "测试员工"},
                    "operation": "register",
                    "answers": {"meal": "A"},
                },
            )
            self.assertEqual(committed["status"], "ok")
            self.assertTrue(committed["event_id"])

    def test_skill_contains_no_install_workflow(self):
        text = SKILL.read_text(encoding="utf-8")
        self.assertNotIn("固定安装块", text)
        self.assertNotIn("mrr_runtime_guard", text)
        self.assertNotIn("systemd", text)
        self.assertNotIn(
            "/work/resource/multi-robot-registration/.runtime/mrr_runtime.py", text
        )
        self.assertIn(
            "/work/skills/multi-robot-registration/scripts/mrr_runtime.py", text
        )


if __name__ == "__main__":
    unittest.main()
