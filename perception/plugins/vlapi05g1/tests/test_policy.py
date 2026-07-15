from __future__ import annotations

import base64
import copy
import json
import math
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


CARD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD_DIR.parent))

from vlapi05g1.plugin import VLAPi05G1Plugin
from vlapi05g1.policy import (
    ACTION_DIM,
    ACTION_SHAPE,
    ACTION_SPACE,
    ACTION_STEPS,
    CardError,
    ObservationSnapshot,
    build_action_proposal,
    build_policy_payload,
    parse_state_message,
    validate_jpeg,
    validate_policy_response,
    validate_seed,
)


def make_snapshot() -> ObservationSnapshot:
    return ObservationSnapshot(
        image_bytes=b"\xff\xd8recorded-jpeg\xff\xd9",
        image_topic="/camera/image/compressed",
        image_frame_id="g1_front_camera",
        image_stamp=100.25,
        image_stamp_source="header",
        image_received_at=101.0,
        image_age_at_request_s=0.05,
        state=tuple(float(index) / 100.0 for index in range(29)),
        state_topic="/robot/state/joints",
        state_received_at=101.1,
        state_age_at_request_s=0.02,
    )


def make_policy_response() -> dict:
    return {
        "ok": True,
        "action_shape": list(ACTION_SHAPE),
        "action_space": ACTION_SPACE,
        "fresh_inference_per_request": True,
        "num_inference_steps": 10,
        "seed": 0,
        "infer_seconds": 0.25,
        "action_chunk": [
            [float(row * ACTION_DIM + column) / 1000.0 for column in range(ACTION_DIM)]
            for row in range(ACTION_STEPS)
        ],
    }


class StateParsingTest(unittest.TestCase):
    def test_maps_indices_and_ignores_extra_joints(self):
        joints = [{"idx": index, "q": index + 0.5} for index in reversed(range(29))]
        joints.append({"idx": 34, "q": 9.0})

        state = parse_state_message(json.dumps({"joints": joints, "imu_quat": [1, 0, 0, 0]}))

        self.assertEqual(29, len(state))
        self.assertEqual(0.5, state[0])
        self.assertEqual(28.5, state[-1])

    def test_rejects_missing_duplicate_and_nonfinite_state(self):
        valid = [{"idx": index, "q": float(index)} for index in range(29)]
        cases = {
            "missing": valid[:-1],
            "duplicate": valid + [{"idx": 0, "q": 0.0}],
            "nan": [*valid[:-1], {"idx": 28, "q": math.nan}],
            "infinity": [*valid[:-1], {"idx": 28, "q": math.inf}],
        }
        for name, joints in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(CardError, "state|joint"):
                    parse_state_message(json.dumps({"joints": joints}))


class JpegValidationTest(unittest.TestCase):
    def test_accepts_exact_size_boundary(self):
        payload = b"\xff\xd8abc\xff\xd9"
        self.assertEqual(payload, validate_jpeg(payload, len(payload)))

    def test_rejects_empty_truncated_and_oversized_input(self):
        cases = (
            (b"", 10),
            (b"\xff\xd8truncated", 100),
            (b"not-jpeg", 100),
            (b"\xff\xd8abc\xff\xd9", 6),
        )
        for payload, limit in cases:
            with self.subTest(payload=payload, limit=limit):
                with self.assertRaisesRegex(CardError, "JPEG|payload"):
                    validate_jpeg(payload, limit)


class PolicyContractTest(unittest.TestCase):
    def test_payload_is_deterministic_and_preserves_inputs(self):
        snapshot = make_snapshot()

        first = build_policy_payload(snapshot, "task", 0)
        second = build_policy_payload(snapshot, "task", 0)

        self.assertEqual(first, second)
        self.assertEqual(snapshot.image_bytes, base64.b64decode(first["image_base64"]))
        self.assertEqual(list(snapshot.state), first["state"])
        self.assertEqual("task", first["task"])
        self.assertEqual(0, first["seed"])

    def test_seed_boundaries_are_strict_integers(self):
        self.assertEqual(0, validate_seed(0))
        self.assertEqual(2**31 - 1, validate_seed(2**31 - 1))
        for value in (-1, 2**31, True, 0.0, "0", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_seed(value)

    def test_validates_deterministic_positive_response(self):
        response = make_policy_response()

        first = validate_policy_response(copy.deepcopy(response), 0)
        second = validate_policy_response(copy.deepcopy(response), 0)

        self.assertEqual(first, second)
        self.assertEqual(ACTION_STEPS, len(first["action_chunk"]))
        self.assertTrue(all(len(row) == ACTION_DIM for row in first["action_chunk"]))
        self.assertEqual(0.25, first["policy_infer_seconds"])

    def test_rejects_empty_and_malformed_policy_results(self):
        cases = {}

        empty = make_policy_response()
        empty["action_chunk"] = []
        cases["empty"] = empty

        wrong_shape = make_policy_response()
        wrong_shape["action_shape"] = [1, 1, ACTION_DIM]
        cases["shape"] = wrong_shape

        wrong_space = make_policy_response()
        wrong_space["action_space"] = "normalized"
        cases["space"] = wrong_space

        wrong_seed = make_policy_response()
        wrong_seed["seed"] = 1
        cases["seed"] = wrong_seed

        nonfinite = make_policy_response()
        nonfinite["action_chunk"][0][0] = math.nan
        cases["nonfinite"] = nonfinite

        for name, response in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(CardError):
                    validate_policy_response(response, 0)

    def test_proposal_keeps_provenance_and_never_authorizes_execution(self):
        snapshot = make_snapshot()
        validated = validate_policy_response(make_policy_response(), 0)

        proposal = build_action_proposal(
            request_id="vlapi05g1-main-000001",
            created_at=102.0,
            snapshot=snapshot,
            task="task",
            seed=0,
            validated_response=validated,
            card_elapsed_seconds=0.3,
        )

        self.assertEqual("pi05.g1.action_chunk.v1", proposal["schema"])
        self.assertEqual(list(snapshot.state), proposal["observation"]["state"])
        self.assertEqual(snapshot.image_topic, proposal["observation"]["image_topic"])
        self.assertEqual(snapshot.image_frame_id, proposal["observation"]["image_frame_id"])
        self.assertEqual(snapshot.state_topic, proposal["observation"]["state_topic"])
        self.assertIs(False, proposal["execution_authorized"])
        json.dumps(proposal)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps(self.server.health_payload).encode("utf-8")
        self.send_response(self.server.health_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return


class _FakeExecutor:
    def add_node(self, _node):
        raise AssertionError("info must not create a ROS node")

    def remove_node(self, _node):
        raise AssertionError("info must not remove a ROS node")


class InfoHealthTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
        self.server.health_status = 200
        self.server.health_payload = {
            "ok": True,
            "action_dim": ACTION_DIM,
            "action_chunk_size": ACTION_STEPS,
            "action_space": ACTION_SPACE,
        }
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def make_plugin(self) -> VLAPi05G1Plugin:
        root_url = f"http://127.0.0.1:{self.server.server_port}"
        return VLAPi05G1Plugin(
            {
                "policy_url": f"{root_url}/predict",
                "health_url": f"{root_url}/health",
                "request_timeout_s": 1.0,
            },
            _FakeExecutor(),
        )

    def test_info_embeds_current_health_without_starting_ros(self):
        result = self.make_plugin().dispatch("vlapi05g1", {"action": "info"})

        self.assertEqual("idle", result["state"])
        self.assertEqual("ok", result["last_health"]["status"])
        self.assertEqual(ACTION_DIM, result["last_health"]["response"]["action_dim"])
        self.assertGreaterEqual(result["last_health"]["latency_s"], 0.0)

    def test_info_contains_health_error_instead_of_failing(self):
        self.server.health_status = 503
        self.server.health_payload = {"ok": False, "error": "loading"}

        result = self.make_plugin().dispatch("vlapi05g1", {"action": "info"})

        self.assertEqual("idle", result["state"])
        self.assertEqual("error", result["last_health"]["status"])
        self.assertEqual("policy_rejected", result["last_health"]["error_code"])


if __name__ == "__main__":
    unittest.main()
