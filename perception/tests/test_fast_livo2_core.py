from __future__ import annotations

import sys
import threading
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


class RollbackReportingBackend(FakeBackend):
    def execute(self, action: str, args: dict) -> dict:
        if action == "load_map" and args["map_name"] == "warehouse":
            self.calls.append((action, dict(args)))
            raise FastLivo2BackendError(
                "algorithm_start_failed",
                "new runtime failed and old map was restored",
                details={
                    "loaded_map": "office",
                    "runtime_mode": "localization",
                    "replaced_map": "office",
                    "rollback_status": "restored",
                },
            )
        return super().execute(action, args)


class BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def execute(self, action: str, args: dict) -> dict:
        if not self.calls:
            self.entered.set()
            self.release.wait(timeout=2.0)
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
        self.assertEqual([call[0] for call in backend.calls[-1:]], ["load_map"])

        removed = core.dispatch({"action": "unload_map"})
        self.assertEqual(removed["error_code"], "unsupported_action")

    def test_load_map_reconciles_restored_map_after_transaction_failure(self) -> None:
        backend = RollbackReportingBackend()
        core = FastLivo2Core(backend)
        self.assertEqual(
            core.dispatch({"action": "load_map", "map_name": "office"})["status"],
            "map_loaded",
        )

        result = core.dispatch({"action": "load_map", "map_name": "warehouse"})

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "algorithm_start_failed")
        self.assertEqual(result["rollback_status"], "restored")
        self.assertEqual(core.info()["loaded_map"], "office")
        self.assertEqual(
            backend.calls,
            [
                ("load_map", {"map_name": "office"}),
                ("load_map", {"map_name": "warehouse"}),
            ],
        )

    def test_runtime_and_collection_lifecycles_are_serialized(self) -> None:
        backend = BlockingBackend()
        core = FastLivo2Core(backend)
        results = []
        first = threading.Thread(
            target=lambda: results.append(
                core.dispatch({"action": "start_mapping", "map_name": "office"})
            )
        )
        second = threading.Thread(
            target=lambda: results.append(
                core.dispatch({"action": "start_mapping", "map_name": "warehouse"})
            )
        )
        first.start()
        self.assertTrue(backend.entered.wait(timeout=1.0))
        second.start()
        self.assertEqual(backend.calls, [])
        backend.release.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual({result["status"] for result in results}, {"mapping", "error"})

        collection_backend = BlockingBackend()
        collection_core = FastLivo2Core(collection_backend)
        first = threading.Thread(
            target=lambda: collection_core.configure_collection({"enabled": True})
        )
        second = threading.Thread(
            target=lambda: collection_core.configure_collection({"enabled": False})
        )
        first.start()
        self.assertTrue(collection_backend.entered.wait(timeout=1.0))
        second.start()
        collection_backend.release.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)
        self.assertEqual(
            [call[1]["enabled"] for call in collection_backend.calls],
            [True, False],
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
