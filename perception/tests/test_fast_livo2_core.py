from __future__ import annotations

import sys
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.fast_livo2.core import (  # noqa: E402
    FastLivo2BackendError,
    FastLivo2Core,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.stop_calls = 0

    def info(self) -> dict:
        return {"state": "ready", "backend": "fake"}

    def execute(self, action: str, args: dict) -> dict:
        self.calls.append((action, dict(args)))
        if action == "start_mapping":
            return {"status": "mapping", "map_name": args["map_name"]}
        return {
            "status": "saved",
            "map_name": args["map_name"],
            "global_relocalization_supported": False,
        }

    def stop(self) -> None:
        self.stop_calls += 1


class RejectingBackend(FakeBackend):
    def execute(self, action: str, args: dict) -> dict:
        del action, args
        raise FastLivo2BackendError("algorithm_unavailable", "algorithm is unavailable")


class FastLivo2CoreTest(unittest.TestCase):
    def test_mapping_lifecycle_and_idempotent_stop(self) -> None:
        backend = FakeBackend()
        core = FastLivo2Core(backend)

        started = core.dispatch({"action": "start_mapping", "map_name": "room-1"})
        self.assertEqual(started["status"], "mapping")
        self.assertEqual(core.info()["active_map"], "room-1")

        duplicate = core.dispatch({"action": "start_mapping", "map_name": "room-2"})
        self.assertEqual(duplicate["error_code"], "mapping_active")
        self.assertEqual(len(backend.calls), 1)

        stopped = core.dispatch({"action": "stop_mapping"})
        self.assertEqual(stopped["status"], "saved")
        self.assertFalse(stopped["global_relocalization_supported"])
        self.assertIsNone(core.info()["active_map"])

        idle = core.dispatch({"action": "stop_mapping"})
        self.assertEqual(idle["status"], "stopped")
        self.assertTrue(idle["already_idle"])

    def test_map_name_is_fail_closed(self) -> None:
        backend = FakeBackend()
        core = FastLivo2Core(backend)
        for value in (None, "", "../escape", "name with spaces", "x" * 65):
            with self.subTest(value=value):
                result = core.dispatch({"action": "start_mapping", "map_name": value})
                self.assertEqual(result["error_code"], "invalid_argument")
        self.assertEqual(backend.calls, [])

    def test_backend_error_is_preserved(self) -> None:
        core = FastLivo2Core(RejectingBackend())
        result = core.dispatch({"action": "start_mapping", "map_name": "room"})
        self.assertEqual(result["error_code"], "algorithm_unavailable")
        self.assertIsNone(core.info()["active_map"])


if __name__ == "__main__":
    unittest.main()
