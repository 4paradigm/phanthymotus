"""Environment switching uses server control leases, not stale iframe bindings."""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import pytest
from test_motion_simulation import profile, physics_log_directory
from motion_service.simulation import SimulationManager
import plugins.motion.plugin as motion_module
from plugins.motion.client import ServiceError as MotionError


@pytest.fixture(params=["motion", "simulation"])
def setup(request, profile, tmp_path, monkeypatch):
    kind = request.param
    if kind == "simulation":
        driver = Path(os.environ.get("MOTION_TEST_DRIVER_ROOT") or
                      Path(__file__).resolve().parents[3] / "phanthymotus-driver")
        if not driver.exists():
            pytest.skip("sibling Driver checkout is needed")
        sys.path.insert(0, str(driver))
        import simulator.mujoco.plugin as module
        from simulator.mujoco.client import ServiceError
    else:
        module, ServiceError = motion_module, MotionError

    class Controller:
        def reset(self): pass
        def step(self, observation): return [0.0]

    servers = {endpoint: SimulationManager({"test": profile}, tmp_path / endpoint.split("//")[1],
                 model_resolver=lambda *args: Controller()) for endpoint in ("http://old:15732", "http://new:15732")}
    clock = [100.]
    for server in servers.values(): server._clock = lambda: clock[0]
    clients = {}
    class Client:
        configured = has_key = True
        def __init__(self, endpoint="", key=""):
            self.endpoint, self.calls, self.offline = endpoint, [], False
            clients[endpoint] = self
        def health(self): return {"state": "ready"}
        def call(self, action, **args):
            self.calls.append((action, args))
            if self.offline: raise ServiceError("网络暂时不可用")
            try: return servers[self.endpoint].call(action, args)
            except ValueError as exc: raise ServiceError(str(exc)) from exc
    monkeypatch.setattr(module, "MotionClient", Client)
    cfg = {"endpoint": "http://old:15732", "api_key": "private", "state_dir": str(tmp_path / "state")}
    plugin = module.MotionPlugin(cfg) if kind == "motion" else module.SimulationPlugin(cfg)
    def call(action, **args): return plugin.dispatch(action, args)
    created = call("simulation_create", profile_id="test", model_id="policy-1", view_id="view-a", request_id="create-a")
    assert not created.get("error"), created
    yield SimpleNamespace(kind=kind, plugin=plugin, call=call, created=created, servers=servers,
                          clients=clients, clock=clock, monkeypatch=monkeypatch)
    for server in servers.values(): server.close()


def test_closed_window_stale_binding_does_not_block_after_lease_expiry(setup):
    s = setup;sid = s.created["session"]["id"]
    # A's iframe disappears without a close callback. B reopens and leaves.
    assert s.call("info", view_id="view-b")["session_id"] == sid
    assert not s.call("simulation_detach", view_id="view-b").get("error")
    # Even paused physics must not be abandoned while A's control lease is valid.
    blocked = s.call("config", endpoint="http://new:15732")
    assert blocked["state"] == "error" and "释放控制" in blocked["error"]
    s.clock[0] += 21
    switched = s.call("config", endpoint="http://new:15732")
    assert switched["state"] == "ready", switched
    assert switched["config_values"]["endpoint"] == "http://new:15732"
    original = s.servers["http://old:15732"].sessions[sid]
    assert original.state == "paused"  # The remote session remains available to viewers.
    assert all(action != "simulation_close" for action, _ in s.clients["http://old:15732"].calls)

    # Deliberately collide IDs across servers: old calls must still be rejected
    # by the plugin before even reaching the new service's epoch checks.
    s.monkeypatch.setattr("motion_service.simulation.uuid.uuid4", lambda: SimpleNamespace(hex=sid))
    current = s.call("simulation_create", profile_id="test", model_id="policy-1", view_id="view-b", request_id="create-b")
    assert current["session"]["id"] == sid
    new_client = s.clients["http://new:15732"]
    before = len(new_client.calls)
    for supplied in ({}, {"session_id": sid}):
        for action in ("simulation_run", "simulation_pause", "simulation_close", "simulation_claim", "simulation_heartbeat"):
            result = s.call(action, view_id="view-a", control_epoch=1, takeover=True, **supplied)
            assert result["state"] == "error", result
    retry = s.call("simulation_create", profile_id="test", model_id="policy-1", view_id="view-a", request_id="create-a")
    assert retry["state"] == "error" and "之前的服务器" in retry["error"]
    assert len(new_client.calls) == before
    assert s.call("simulation_step", view_id="view-b", control_epoch=current["control"]["epoch"])["snapshot"]["step"] == 1


def test_other_entry_controller_keeps_running_when_a_viewer_changes_server(setup):
    s = setup;sid = s.created["session"]["id"]
    old = s.servers["http://old:15732"]
    taken = old.call("simulation_claim", {"session_id": sid, "view_id": "other-entry", "control_epoch":1, "takeover":True})
    old.call("simulation_run", {"session_id":sid,"view_id":"other-entry","control_epoch":taken["control"]["epoch"]})
    s.call("info", view_id="view-b")
    s.call("simulation_detach", view_id="view-b")
    result = s.call("config", endpoint="http://new:15732")
    assert result["state"] == "ready", result
    assert old.sessions[sid].state == "running"
    assert old.call("simulation_inspect", {"session_id":sid})["control"]["owner_view_id"] == "other-entry"


def test_detach_does_not_hide_a_still_owned_lease_and_offline_is_not_proof_of_release(setup):
    s = setup
    s.call("simulation_detach", view_id="view-a")
    assert s.call("config", endpoint="http://new:15732")["state"] == "error"
    old_client = s.clients["http://old:15732"]
    old_client.offline = True
    s.clock[0] += 21
    result = s.call("config", endpoint="http://new:15732")
    assert result["state"] == "error" and "网络暂时不可用" in result["error"]
    assert s.plugin._client is old_client
