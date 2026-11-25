"""Workbench ownership and credential boundaries across reopening and restart."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.motion import MotionPlugin
from plugins.motion.client import MotionClient
from plugins.motion.client import ServiceError


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MOTION_WORKBENCH_STATE_DIR", str(tmp_path))
    config = {"endpoint": "http://motion:15732", "api_key": "motion-secret",
              "environment_endpoint": "http://bootstrap:15734", "environment_token": "manager-secret"}
    calls, sessions = [], {"other-card": {"id": "other-card", "state": "paused"}}

    def call(client, action, **args):
        calls.append((client.endpoint, action, args))
        if action == "simulation_list":
            return {"sessions": list(sessions.values())}
        if action == "simulation_create":
            sessions["owned"] = {"id": "owned", "state": "paused", **args}
            return {"session": sessions["owned"]}
        if action.startswith("simulation_"):
            sid = args["session_id"]
            assert sid == "owned", "must never control the other card's session"
            session = sessions[sid]
            if action == "simulation_close":
                session = sessions.pop(sid)
                session["state"] = "closed"
            elif action == "simulation_run":
                session["state"] = "running"
            elif action == "simulation_pause":
                session["state"] = "paused"
            return {"session": dict(session)}
        if action == "connection":
            return {"connection": {"endpoint": "http://installed:15732", "api_key": "installed-secret"}}
        return {"action": action, **args}

    monkeypatch.setattr(MotionClient, "call", call)
    monkeypatch.setattr(MotionClient, "health", lambda _: {"state": "ready"})
    return MotionPlugin(config), config, calls, sessions


def invoke(plugin, action, **args):
    return plugin.dispatch("motion", {"action": action, **args})


def test_only_owned_session_is_controllable_and_duplicate_create_is_idempotent(setup):
    plugin, _, calls, sessions = setup
    assert invoke(plugin, "simulation_run", session_id="other-card")["state"] == "error"
    assert invoke(plugin, "simulation_create", profile_id="g1", model_id="model-1")["session"]["id"] == "owned"
    invoke(plugin, "simulation_create", profile_id="g1", model_id="model-1")
    assert sum(action == "simulation_create" for _, action, _ in calls) == 1
    assert invoke(plugin, "simulation_create", profile_id="g1", model_id="model-2")["state"] == "error"
    assert invoke(plugin, "simulation_run", session_id="other-card")["state"] == "error"
    invoke(plugin, "simulation_run")
    assert sessions["owned"]["state"] == "running"
    invoke(plugin, "stop")
    assert sessions["owned"]["state"] == "paused"
    assert sessions["other-card"]["state"] == "paused"
    assert invoke(plugin, "info")["session_id"] == "owned"


def test_reopen_and_backend_restart_restore_only_own_session(setup):
    plugin, config, calls, sessions = setup
    invoke(plugin, "simulation_create", profile_id="g1", model_id="model-1")
    restored = MotionPlugin(config)
    assert invoke(restored, "info")["session_id"] == "owned"
    invoke(restored, "inspect", mode="simulation_frame", session_id="owned", width=640, height=360, action_override="run")
    assert calls[-1][1:] == ("simulation_render", {"session_id": "owned", "width": 640, "height": 360})
    sessions.pop("owned")  # Motion service restarted; an unrelated session remains.
    assert invoke(restored, "info")["session_id"] is None
    assert invoke(restored, "simulation_run")["state"] == "error"
    assert "other-card" in sessions


def test_live_session_blocks_server_switch_until_closed(setup):
    plugin, _, _, _ = setup
    invoke(plugin, "simulation_create", profile_id="g1", model_id="model-1")
    assert invoke(plugin, "env_connect", job_id="job-1")["state"] == "error"
    assert invoke(plugin, "config", endpoint="http://another:15732")["state"] == "error"
    invoke(plugin, "simulation_close")
    result = invoke(plugin, "env_connect", job_id="job-1")
    assert result["config_values"]["endpoint"] == "http://installed:15732"
    assert not any(secret in json.dumps(result) for secret in ("installed-secret", "manager-secret", "motion-secret"))


def test_environment_connection_is_backend_only_durable_and_manual_config_can_clear_it(setup):
    plugin, config, calls, _ = setup
    assert invoke(plugin, "env_connect", job_id="job-1")["state"] == "ready"
    assert plugin._binding_path.stat().st_mode & 0o777 == 0o600
    restored = MotionPlugin(config)
    result = invoke(restored, "info")
    assert result["config_values"]["environment_job"] == "job-1"
    assert restored._client.endpoint == "http://installed:15732"
    # Core restoring its older endpoint must not silently undo a managed binding.
    invoke(restored, "config", endpoint=config["endpoint"], api_key=config["api_key"])
    assert restored._client.endpoint == "http://installed:15732"
    invoke(restored, "config", environment_job="", endpoint=config["endpoint"], api_key=config["api_key"])
    assert restored._client.endpoint == config["endpoint"]
    assert not restored._binding_path.exists()
    assert "installed-secret" not in json.dumps(invoke(restored, "info"))


def test_readonly_inspection_cannot_start_install_or_run_simulation(setup):
    plugin, _, calls, _ = setup
    before = len(calls)
    for mode in ("install", "env_install", "simulation_run", "connection"):
        assert invoke(plugin, "inspect", mode=mode)["state"] == "error"
    assert len(calls) == before
    invoke(plugin, "inspect", mode="env_status", job_id="job-1", cursor=7, command="evil")
    assert calls[-1] == ("http://bootstrap:15734", "status", {"job_id": "job-1", "cursor": 7})
    invoke(plugin, "env_install", target_id="server-1", recipe_id="supermimic-g1-v1", mode="isolated", command="evil")
    assert calls[-1] == ("http://bootstrap:15734", "install", {"target_id": "server-1", "recipe_id": "supermimic-g1-v1", "mode": "isolated"})


def test_reference_preview_is_readonly_and_filters_arbitrary_routing(setup):
    plugin, _, calls, _ = setup
    invoke(plugin, "inspect", mode="dataset_render", dataset_id="data-1", offset=10, width=640,
           endpoint="http://evil", model_id="wrong", command="evil")
    assert calls[-1] == ("http://motion:15732", "dataset_render", {"dataset_id": "data-1", "offset": 10, "width": 640})


def test_training_monitor_routes_real_replay_and_selected_sample_with_bounded_dimensions(setup):
    plugin, _, calls, _ = setup
    invoke(plugin, "inspect", mode="training_preview", job_id="job-1", width=960, height=240)
    assert calls[-1] == ("http://motion:15732", "training_preview", {"job_id": "job-1", "width": 960, "height": 240})
    before = len(calls)
    for dimensions in ({"width": 99999}, {"height": 120}, {"width": True}, {"height": "240"}):
        assert invoke(plugin, "inspect", mode="training_preview", job_id="job-1", **dimensions)["state"] == "error"
    assert len(calls) == before
    invoke(plugin, "inspect", mode="training_preview", job_id="job-1", frame=3, sample=7, command="evil")
    assert calls[-1] == ("http://motion:15732", "training_preview", {"job_id": "job-1", "frame": 3, "sample": 7, "width": 896, "height": 448})


def test_environment_can_open_when_previous_motion_session_is_unreachable(setup, monkeypatch):
    plugin, _, _, _ = setup
    invoke(plugin, "simulation_create", profile_id="g1", model_id="model-1")
    def offline(*args, **kwargs):
        raise ServiceError("无法连接服务器或请求超时")
    monkeypatch.setattr(plugin._client, "call", offline)
    result = invoke(plugin, "info")
    assert result["environment_available"] is True
    assert result["session_reachable"] is False
    assert result["session_id"] == "owned"
    assert "error" not in result
    assert invoke(plugin, "stop")["state"] == "error"
    assert plugin._session.id == "owned"  # A lost response never proves a pause.


def test_core_saved_environment_job_replay_keeps_new_binding(setup):
    plugin, config, calls, _ = setup
    saved = {"endpoint": config["endpoint"], "api_key": config["api_key"], "environment_job": ""}
    invoke(plugin, "config", **saved)
    # The UI uses host config: apply, then persist only the nonsecret job ref.
    desired = {**saved, "environment_job": "job-1"}
    assert invoke(plugin, "config", **desired)["config_values"]["environment_job"] == "job-1"
    saved = desired
    for action in ("catalog", "resources"):
        invoke(plugin, "config", **saved)
        invoke(plugin, action)
        assert calls[-1][0] == "http://installed:15732"
    assert sum(action == "connection" for _, action, _ in calls) == 1
    # Switching from a previously managed job also survives Core reapplication.
    desired = {**saved, "environment_job": "job-2"}
    invoke(plugin, "config", **desired)
    invoke(plugin, "config", **desired)
    assert invoke(plugin, "info")["config_values"]["environment_job"] == "job-2"
    assert sum(action == "connection" for _, action, _ in calls) == 2
