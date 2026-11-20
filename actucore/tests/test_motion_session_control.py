"""Real physics verifies controller fencing across independent workbench views."""
from concurrent.futures import ThreadPoolExecutor
import time
import os
from pathlib import Path
import sys

import pytest

from test_motion_simulation import profile, physics_log_directory
from motion_service.simulation import SimulationManager
from plugins.motion.plugin import MotionPlugin
from plugins.motion.client import ServiceError


@pytest.fixture
def manager(profile, tmp_path):
    value = SimulationManager({"test": profile}, tmp_path)
    yield value
    value.close()


def create(manager, view="motion-view"):
    return manager.call("simulation_create", {"profile_id": "test", "control_mode": "pd_hold",
                                               "view_id": view, "request_id": "new-trial"})


def command(manager, result, action, view="motion-view", **extra):
    return manager.call("simulation_" + action,
                        {"session_id": result["session"]["id"], "view_id": view,
                         "control_epoch": result["control"]["epoch"], **extra})


def test_takeover_fences_old_commands_including_late_stop(manager):
    a = create(manager)
    sid = a["session"]["id"]
    watching = manager.call("simulation_inspect", {"session_id": sid, "view_id": "debug-view"})
    assert watching["control"]["is_controller"] is False
    assert a["control"]["is_controller"] is True
    with pytest.raises(ValueError, match="控制权"):
        manager.call("simulation_run", {"session_id": sid})
    with pytest.raises(ValueError, match="另一个窗口"):
        command(manager, watching, "claim", "debug-view")
    command(manager, a, "run")
    b = command(manager, watching, "claim", "debug-view", takeover=True)
    assert b["session"]["state"] == "paused"
    assert b["control"]["epoch"] > a["control"]["epoch"]
    command(manager, b, "run", "debug-view")
    for operation in ("run", "pause", "step", "reset", "close", "heartbeat", "release", "record", "perturb"):
        with pytest.raises(ValueError, match="控制权已变化"):
            command(manager, a, operation)
        assert manager.sessions[sid].state == "running"
    command(manager, b, "pause", "debug-view")
    stepped = command(manager, b, "step", "debug-view")
    assert stepped["snapshot"]["step"] > 0


def test_disconnect_expires_without_an_incoming_request_and_stale_heartbeat_cannot_revive(manager):
    manager._lease_seconds = .05
    a = create(manager)
    sid = a["session"]["id"]
    command(manager, a, "run")
    until = time.monotonic() + 2
    while manager.sessions[sid].state == "running" and time.monotonic() < until:
        time.sleep(.02)
    assert manager.sessions[sid].state == "paused"
    with pytest.raises(ValueError, match="控制权已变化"):
        command(manager, a, "heartbeat")
    state = manager.call("simulation_inspect", {"session_id": sid, "view_id": "new-view"})
    assert state["control"]["owner_view_id"] is None
    b = command(manager, state, "claim", "new-view")
    assert b["control"]["epoch"] > a["control"]["epoch"]


def test_release_requires_current_owner_and_claim_uses_observed_epoch(manager):
    a = create(manager)
    command(manager, a, "run")
    with pytest.raises(ValueError, match="控制权"):
        command(manager, a, "release", "viewer")
    assert manager.sessions[a["session"]["id"]].state == "running"
    released = command(manager, a, "release")
    assert released["session"]["state"] == "paused"
    assert released["control"]["owner_view_id"] is None
    with pytest.raises(ValueError, match="控制权已变化"):
        command(manager, a, "claim", "viewer", takeover=True)
    command(manager, released, "claim", "viewer")
    with pytest.raises(ValueError, match="控制权已变化"):
        command(manager, released, "claim", "late-viewer", takeover=True)


def test_concurrent_retry_creates_one_session_and_does_not_renew_or_rebind(manager):
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: create(manager), range(4)))
    assert len({r["session"]["id"] for r in outputs}) == 1
    assert len(manager.sessions) == 1
    a = outputs[0]
    b = command(manager, a, "claim", "debug-view", takeover=True)
    retry = create(manager)
    assert retry["control"]["owner_view_id"] == "debug-view"
    assert retry["control"]["epoch"] == b["control"]["epoch"]
    assert retry["control"]["is_controller"] is False
    with pytest.raises(ValueError, match="创建请求已用于"):
        manager.call("simulation_create", {"profile_id": "test", "control_mode": "free_physics",
                                            "view_id": "motion-view", "request_id": "new-trial"})


def test_two_simultaneous_takeovers_only_one_observed_epoch_wins(manager):
    a = create(manager)
    def acquire(view):
        try:
            return command(manager, a, "claim", view, takeover=True)
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, ["view-b", "view-c"]))
    assert sum(result is not None for result in results) == 1
    winner = next(result for result in results if result)
    assert winner["control"]["epoch"] == a["control"]["epoch"] + 1


def test_motion_and_driver_share_controller_but_keep_view_bindings(profile, tmp_path):
    driver = Path(os.environ.get("MOTION_TEST_DRIVER_ROOT") or
                  Path(__file__).resolve().parents[3] / "phanthymotus-driver")
    if not driver.exists():
        pytest.skip("sibling Driver checkout is needed for the cross-card contract")
    sys.path.insert(0, str(driver))
    from simulator.mujoco.plugin import SimulationPlugin
    from simulator.mujoco.client import ServiceError as DriverError

    class Controller:
        def reset(self): pass
        def step(self, observation): return [0.0]

    backend = SimulationManager({"test": profile}, tmp_path / "physics", model_resolver=lambda *args: Controller())
    class Client:
        configured, has_key, endpoint = True, True, "http://motion:15732"
        def call(self, action, **args):
            try:
                return backend.call(action, args)
            except ValueError as exc:
                raise ServiceError(str(exc)) from exc
    class DriverClient(Client):
        def call(self, action, **args):
            try:
                return super().call(action, **args)
            except ServiceError as exc:
                raise DriverError(str(exc)) from exc

    motion = MotionPlugin({"endpoint": Client.endpoint, "api_key": "private", "state_dir": str(tmp_path / "motion")})
    debug = SimulationPlugin({"endpoint": Client.endpoint, "api_key": "private", "state_dir": str(tmp_path / "debug")})
    motion._client, debug._client = Client(), DriverClient()
    def invoke(plugin, action, **args):
        result = plugin.dispatch(action, args)
        assert not result.get("error"), result
        return result
    try:
        a = invoke(motion, "simulation_create", profile_id="test", model_id="policy-1",
                   view_id="motion-a", request_id="new")
        sid = a["session"]["id"]
        assert a["session"]["origin"] == "motion"
        invoke(debug, "simulation_attach", session_id=sid, view_id="debug-b")
        b = invoke(debug, "simulation_claim", view_id="debug-b", control_epoch=a["control"]["epoch"], takeover=True)
        invoke(debug, "simulation_run", view_id="debug-b", control_epoch=b["control"]["epoch"])
        motion.stop()  # The Motion entry's remembered old grant cannot stop Driver.
        assert backend.sessions[sid].state == "running"
        invoke(motion, "simulation_attach", session_id=sid, view_id="motion-c", profile_id="test", model_id="policy-1")
        invoke(motion, "stop", view_id="motion-c")
        assert backend.sessions[sid].state == "running"
        invoke(motion, "simulation_detach", view_id="motion-c")
        assert backend.sessions[sid].state == "running"
        assert invoke(motion, "info", view_id="motion-c")["session_id"] is None
        assert invoke(motion, "info", view_id="motion-a")["session_id"] == sid
        invoke(debug, "simulation_attach", session_id=sid, view_id="debug-d")
        invoke(debug, "stop", view_id="debug-d")
        assert backend.sessions[sid].state == "running"
        invoke(debug, "simulation_detach", view_id="debug-d")
        assert invoke(debug, "info", view_id="debug-b")["session_id"] == sid
        assert backend.sessions[sid].state == "running"
        refused = motion.dispatch("simulation_attach", {"session_id": sid, "view_id": "wrong-model",
                                                          "profile_id": "test", "model_id": "policy-2"})
        assert refused["state"] == "error" and "不匹配" in refused["error"]
        invoke(debug, "simulation_close", view_id="debug-b", control_epoch=b["control"]["epoch"])
        debug.stop()
        assert backend.sessions[sid].state == "closed"
        created = invoke(debug, "simulation_create", profile_id="test", view_id="debug-origin", request_id="new-debug", origin="forged")
        assert created["session"]["origin"] == "simulation"
    finally:
        backend.close()


def test_readonly_polling_does_not_extend_control_but_heartbeat_does(manager):
    now = [100.0]
    manager._clock = lambda: now[0]
    a = create(manager)
    sid = a["session"]["id"]
    now[0] += 15
    state = manager.call("simulation_inspect", {"session_id": sid, "view_id": "motion-view"})
    assert state["control"]["expires_in_s"] == 5
    heartbeat = command(manager, a, "heartbeat")
    assert heartbeat["control"]["expires_in_s"] == 20
    now[0] += 21
    listed = manager.call("simulation_list", {"view_id": "motion-view"})["sessions"][0]
    assert listed["control"]["owner_view_id"] is None
    assert listed["state"] == "paused"
