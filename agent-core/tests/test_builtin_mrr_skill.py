"""The multi-robot registration runtime is shipped in the Agent Core image.

These tests protect the fast path: the model invokes a prebuilt script instead
of recreating or installing Python during a registration conversation.
"""

import json
import fcntl
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid


AGENT_CORE = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = AGENT_CORE / "tools/multi-robot-registration/mrr_runtime.py"


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


def _envelope(command: str, round_id: str, target: dict, **payload) -> dict:
    return {
        "marker": "MRR_CONTROL_V1",
        "protocol": "mrr-v1",
        "command": command,
        "request_id": uuid.uuid4().hex,
        "round_id": round_id,
        "leader": {"peer_id": "g1", "name": "g1"},
        "config_version": 1,
        "sent_at": "2026-10-10T00:00:00+00:00",
        "payload": {"target": target, **payload},
    }


def _start(root: pathlib.Path, role: str = "leader") -> tuple[str, dict]:
    round_id = uuid.uuid4().hex
    target = {"peer_id": "g1" if role == "leader" else "g2", "name": role}
    request = _envelope(
        "START",
        round_id,
        target,
        team_members=[],
        config={
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
    )
    response = _run(root, f"control-{role}", request)
    assert response["status"] == "ok"
    return round_id, target


class BuiltinMrrSkillTests(unittest.TestCase):
    def test_runtime_is_baked_into_agent_core_image(self):
        dockerfile = (AGENT_CORE / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY tools/    /work/tools/", dockerfile)
        self.assertTrue(RUNTIME.is_file())

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
            _start(root)

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

    def test_protocol_ids_cannot_escape_the_registration_root(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = pathlib.Path(directory)
            root = parent / "registration"
            outside = parent / "outside"
            outside.mkdir()
            sentinel = outside / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            leader = {"peer_id": "g1", "name": "g1"}

            for bad_round_id in ("../outside", str(outside)):
                response = _run(
                    root,
                    "control-leader",
                    _envelope("START", bad_round_id, leader, team_members=[], config={}),
                )
                self.assertEqual(response["status"], "error")
                self.assertTrue(sentinel.is_file())

    def test_cleanup_refuses_a_round_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = pathlib.Path(directory)
            root = parent / "registration"
            round_id, member = _start(root, role="member")
            stopped = _run(
                root,
                "control-member",
                _envelope("STOP", round_id, member, mode="immediate"),
            )
            self.assertEqual(stopped["runtime_state"], "stopped")

            outside = parent / "outside"
            outside.mkdir()
            sentinel = outside / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            round_path = root / "rounds" / round_id
            shutil.rmtree(round_path)
            round_path.symlink_to(outside, target_is_directory=True)

            response = _run(
                root,
                "control-member",
                _envelope(
                    "CLEANUP",
                    round_id,
                    member,
                    archive_confirmed=True,
                    archive_digest="verified-by-leader",
                ),
            )
            self.assertEqual(response["status"], "error")
            self.assertTrue(sentinel.is_file())

    def test_persisted_config_path_cannot_redirect_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = pathlib.Path(directory)
            root = parent / "registration"
            _start(root)
            outside = parent / "outside.json"
            outside.write_text(json.dumps({"activity_name": "not this activity"}), encoding="utf-8")
            current_path = root / "current.json"
            current = json.loads(current_path.read_text(encoding="utf-8"))
            current["config_path"] = str(outside)
            current_path.write_text(json.dumps(current), encoding="utf-8")

            response = _run(root, "inspect", {})
            self.assertEqual(response["status"], "error")
            self.assertEqual(response["code"], "CONFIG_INVALID")

    def test_persisted_snapshot_filename_cannot_escape_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "registration"
            round_id, leader = _start(root)
            stopped = _run(
                root,
                "control-leader",
                _envelope("STOP", round_id, leader, mode="immediate"),
            )
            self.assertEqual(stopped["runtime_state"], "stopped")

            round_path = root / "rounds" / round_id
            secret = round_path / "secret.json"
            secret.write_text(
                json.dumps({"round_id": round_id, "event_id": "must-not-be-read"}),
                encoding="utf-8",
            )
            final_path = round_path / "final.json"
            final = json.loads(final_path.read_text(encoding="utf-8"))
            final["record_files"] = ["../secret.json"]
            final_path.write_text(json.dumps(final), encoding="utf-8")

            response = _run(
                root,
                "control-leader",
                _envelope(
                    "SNAPSHOT",
                    round_id,
                    leader,
                    snapshot_id=uuid.uuid4().hex,
                    cursor=0,
                    limit=50,
                    mode="final",
                ),
            )
            self.assertEqual(response["status"], "ok")
            self.assertEqual(response["returned_count"], 0)
            self.assertEqual(response["page_invalid_record_count"], 1)
            self.assertTrue(response["data_incomplete"])

    def test_concurrent_begin_creates_exactly_one_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "registration"
            round_id, _ = _start(root)
            lock_path = root.parent / f".{root.name}.runtime.lock"
            env = dict(os.environ, MRR_ROOT=str(root))

            with lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                processes = [
                    subprocess.Popen(
                        [sys.executable, str(RUNTIME), "begin"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                    )
                    for _ in range(8)
                ]
                time.sleep(0.1)
                self.assertTrue(all(process.poll() is None for process in processes))
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

            responses = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr or stdout)
                responses.append(json.loads(stdout))

            self.assertEqual(sum(x["status"] == "ok" for x in responses), 1)
            self.assertEqual(sum(x["status"] == "busy" for x in responses), 7)
            sessions = list(
                (root / "rounds" / round_id / "runtime" / "interactions").glob("*.json")
            )
            self.assertEqual(len(sessions), 1)

    def test_skill_definition_is_not_baked_into_the_image(self):
        self.assertFalse((AGENT_CORE / "skills/multi-robot-registration/SKILL.md").exists())


if __name__ == "__main__":
    unittest.main()
