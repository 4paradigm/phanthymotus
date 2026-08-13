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
        if action == "load_map":
            return {"status": "map_loaded", "map_name": args["map_name"]}
        if action == "relocalize":
            return {"status": "relocalized", "map_name": args["map_name"], "score": 0.8}
        if action == "unload_map":
            return {"status": "unloaded", "map_name": args["map_name"]}
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


class UnconfirmedLocalizationStopBackend(FakeBackend):
    def execute(self, action: str, args: dict) -> dict:
        if action == "unload_map":
            self.calls.append((action, dict(args)))
            return {"status": "running", "map_name": args["map_name"]}
        return super().execute(action, args)


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

    def test_load_relocalize_and_automatic_map_replacement(self) -> None:
        backend = FakeBackend()
        core = FastLivo2Core(backend)

        loaded = core.dispatch({"action": "load_map", "map_name": "office"})
        self.assertEqual(loaded["status"], "map_loaded")
        self.assertEqual(core.info()["loaded_map"], "office")

        blocked_mapping = core.dispatch({"action": "start_mapping", "map_name": "new"})
        self.assertEqual(blocked_mapping["error_code"], "localization_active")

        localized = core.dispatch(
            {
                "action": "relocalize",
                "initial_x": 1.0,
                "initial_y": -2.0,
                "initial_yaw": 0.5,
            }
        )
        self.assertEqual(localized["status"], "relocalized")
        self.assertEqual(backend.calls[-1][1]["map_name"], "office")
        self.assertEqual(backend.calls[-1][1]["search_xy_m"], 1.0)

        replaced = core.dispatch({"action": "load_map", "map_name": "warehouse"})
        self.assertEqual(replaced["status"], "map_loaded")
        self.assertEqual(replaced["replaced_map"], "office")
        self.assertEqual(core.info()["loaded_map"], "warehouse")
        self.assertEqual(
            [call[0] for call in backend.calls[-2:]],
            ["unload_map", "load_map"],
        )

        removed = core.dispatch({"action": "unload_map"})
        self.assertEqual(removed["error_code"], "unsupported_action")

    def test_load_map_keeps_old_map_when_private_stop_is_unconfirmed(self) -> None:
        backend = UnconfirmedLocalizationStopBackend()
        core = FastLivo2Core(backend)
        self.assertEqual(
            core.dispatch({"action": "load_map", "map_name": "office"})["status"],
            "map_loaded",
        )

        result = core.dispatch({"action": "load_map", "map_name": "warehouse"})

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "localization_stop_unconfirmed")
        self.assertEqual(core.info()["loaded_map"], "office")
        self.assertEqual(
            backend.calls,
            [
                ("load_map", {"map_name": "office"}),
                ("unload_map", {"map_name": "office"}),
            ],
        )

    def test_relocalize_requires_map_and_finite_pose(self) -> None:
        backend = FakeBackend()
        core = FastLivo2Core(backend)
        missing = core.dispatch(
            {"action": "relocalize", "initial_x": 0, "initial_y": 0, "initial_yaw": 0}
        )
        self.assertEqual(missing["error_code"], "map_not_loaded")
        core.dispatch({"action": "load_map", "map_name": "office"})
        for key, value in (("initial_x", float("nan")), ("search_xy_m", 10.0)):
            args = {
                "action": "relocalize",
                "initial_x": 0,
                "initial_y": 0,
                "initial_yaw": 0,
                key: value,
            }
            with self.subTest(key=key):
                self.assertEqual(core.dispatch(args)["error_code"], "invalid_argument")


if __name__ == "__main__":
    unittest.main()
