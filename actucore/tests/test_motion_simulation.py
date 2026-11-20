"""Real MuJoCo dynamics and session lifecycle; no hardware or graphics context."""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("mujoco")
from motion_service.simulation import MAX_SESSIONS, SimulationManager


@pytest.fixture(autouse=True)
def physics_log_directory(tmp_path, monkeypatch):
    # MuJoCo writes diagnostics for the deliberate instability test to cwd.
    monkeypatch.chdir(tmp_path)


MJCF = """<mujoco model="test_robot">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1" rgba=".2 .3 .4 1"/>
    <body name="robot" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size=".1" mass="1"/>
      <body name="arm" pos="0 0 .15">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 .2 0 0" size=".03" mass=".1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="motor" joint="hinge" gear="2" ctrllimited="true" ctrlrange="-20 20"/></actuator>
</mujoco>"""


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text(MJCF)
    return {"id": "test", "label": "Test Robot", "robot_type": "fixture", "model_version": "1",
            "model_path": str(path), "joint_names": ["hinge"], "control_hz": 50,
            "physics_dt": .002, "default_joint_pos": [0.0], "kp": [20.0], "kd": [1.0],
            "joint_limits": {"lower": [-1.5], "upper": [1.5]}, "quaternion_order": "wxyz"}


@pytest.fixture
def manager(profile, tmp_path):
    value = SimulationManager({"test": profile}, tmp_path)
    yield value
    value.close()


def create(manager, **kwargs):
    return manager.call("simulation_create", {"profile_id": "test", **kwargs})


def test_sessions_can_be_discovered_after_client_loses_its_binding(manager):
    first = create(manager)
    second = create(manager)
    sessions = manager.call("simulation_list", {})["sessions"]
    assert {s["id"] for s in sessions} == {first["session"]["id"], second["session"]["id"]}
    call(manager, first["session"]["id"], "close")
    assert [s["id"] for s in manager.call("simulation_list", {})["sessions"]] == [second["session"]["id"]]


@pytest.mark.parametrize("background", [False, True])
def test_completed_policy_stops_physics_recording_and_requires_reset(profile, tmp_path, background):
    class FiniteController:
        def reset(self):
            self.calls = 0
            self.reference_complete = False

        def step(self, observation):
            self.calls += 1
            self.reference_complete = self.calls == 2
            return [0.]

    controller = FiniteController()
    manager = SimulationManager({"test": profile}, tmp_path, model_resolver=lambda *_: controller)
    try:
        sid = create(manager, model_id="finite")["session"]["id"]
        call(manager, sid, "record")
        call(manager, sid, "perturb", force=[1., 0., 0.], duration_s=1.)
        if background:
            call(manager, sid, "run")
            deadline = time.monotonic() + 3
            while call(manager, sid, "inspect")["session"]["state"] == "running" and time.monotonic() < deadline:
                time.sleep(.02)
        else:
            call(manager, sid, "step", steps=100)
        out = call(manager, sid, "inspect")
        assert out["session"]["state"] == "completed"
        assert out["session"]["recording"] is False
        assert controller.calls == 2
        assert out["snapshot"]["time"] == pytest.approx(.04)
        assert manager.sessions[sid].data.xfrc_applied.sum() == 0
        for action in ("run", "step", "perturb", "record"):
            with pytest.raises(ValueError, match="reset"):
                call(manager, sid, action)
        call(manager, sid, "reset")
        assert call(manager, sid, "step")["session"]["state"] == "paused"
        assert controller.calls == 1
    finally:
        manager.close()


def call(manager, session_id, action, **kwargs):
    return manager.call("simulation_" + action, {"session_id": session_id, **kwargs})


def test_real_gravity_step_and_reset(manager):
    initial = create(manager)
    sid = initial["session"]["id"]
    assert initial["session"]["engine"] == "native_mujoco"
    assert initial["session"]["control_mode"] == "free_physics"
    assert initial["session"]["state"] == "paused"
    stepped = call(manager, sid, "step", steps=10)
    assert stepped["snapshot"]["time"] == pytest.approx(.2)
    assert stepped["snapshot"]["root_pose"]["pos"][2] < initial["snapshot"]["root_pose"]["pos"][2] - .1
    assert stepped["snapshot"]["root_lin_vel_w"][2] < -1
    reset = call(manager, sid, "reset")
    assert reset["snapshot"]["time"] == 0
    assert reset["snapshot"]["qpos"] == pytest.approx(initial["snapshot"]["qpos"])
    assert reset["snapshot"]["joint_vel"] == [0.0]


def test_snapshot_has_real_geometry_contact_and_json_payload(manager):
    state = create(manager)
    sid = state["session"]["id"]
    out = call(manager, sid, "step", steps=60)
    assert out["snapshot"]["contacts"]
    assert {g["type"] for g in out["snapshot"]["geoms"]} == {"plane", "sphere", "capsule"}
    assert out["snapshot"]["bodies"][0]["name"] == "robot"
    assert len(out["snapshot"]["geoms"][0]["mat"]) == 9
    json.dumps(out, allow_nan=False)


def test_background_run_pause_and_close_release_resources(manager):
    sid = create(manager)["session"]["id"]
    call(manager, sid, "run")
    deadline = time.monotonic() + 2
    while call(manager, sid, "inspect")["snapshot"]["time"] < .04:
        assert time.monotonic() < deadline
        time.sleep(.005)
    with pytest.raises(ValueError, match="paused"):
        call(manager, sid, "step")
    paused = call(manager, sid, "pause")
    time.sleep(.05)
    assert call(manager, sid, "inspect")["snapshot"]["time"] == paused["snapshot"]["time"]
    closed = call(manager, sid, "close")
    assert closed["session"]["state"] == "closed"
    assert manager.sessions[sid].data is None
    assert manager.sessions[sid].model is None
    assert not manager.sessions[sid]._thread.is_alive()
    assert call(manager, sid, "close")["session"]["state"] == "closed"
    with pytest.raises(ValueError, match="closed"):
        call(manager, sid, "run")


@pytest.mark.parametrize("changes, message", [
    ({"joint_names": ["missing"]}, "absent"),
    ({"joint_names": ["hinge", "hinge"]}, "unique"),
    ({"physics_dt": .003}, "timestep"),
    ({"control_hz": 60}, "integer multiple"),
    ({"quaternion_order": "xyzw"}, "wxyz"),
    ({"default_joint_pos": [2.0]}, "limits"),
    ({"kp": [-1.0]}, "nonnegative"),
])
def test_bad_robot_profiles_are_rejected(profile, tmp_path, changes, message):
    bad = {**profile, **changes}
    manager = SimulationManager({"test": bad}, tmp_path)
    with pytest.raises(ValueError, match=message):
        create(manager)
    assert not manager.sessions


def test_pd_motor_mapping_and_policy_loop(profile, tmp_path):
    class Controller:
        initial_state = {"joint_pos": [.3], "root_pos": [0, 0, 1.1], "root_quat": [1, 0, 0, 0]}

        def __init__(self):
            self.calls, self.resets, self.closed = [], 0, False

        def reset(self):
            self.resets += 1

        def step(self, observation):
            self.calls.append(copy.deepcopy(observation))
            return [.3]

        def close(self):
            self.closed = True

    controller = Controller()
    manager = SimulationManager({"test": profile}, tmp_path, model_resolver=lambda mid, prof: controller)
    try:
        out = create(manager, model_id="policy-1")
        sid = out["session"]["id"]
        assert out["session"]["control_mode"] == "policy"
        assert out["snapshot"]["joint_pos"] == pytest.approx([.3])
        assert out["snapshot"]["root_pose"]["pos"] == pytest.approx([0, 0, 1.1])
        call(manager, sid, "step", steps=5)
        assert len(controller.calls) == 5
        assert controller.calls[0]["dt"] == .02
        assert controller.calls[-1]["step"] == 4
        assert controller.calls[0]["bodies"][0]["name"] == "robot"
        assert manager.sessions[sid].actuators == [0]
        assert manager.sessions[sid].gains.tolist() == [2.0]
        assert abs(manager.sessions[sid].data.ctrl[0]) <= 20
        call(manager, sid, "reset")
        assert controller.resets == 2
    finally:
        manager.close()
    assert controller.closed


@pytest.mark.parametrize("target", [float("nan"), float("inf"), 2.0])
def test_policy_bad_target_fails_session_instead_of_reusing_action(profile, tmp_path, target):
    class Controller:
        def reset(self):
            pass

        def step(self, observation):
            return [target]

    manager = SimulationManager({"test": profile}, tmp_path, model_resolver=lambda *a: Controller())
    try:
        sid = create(manager, model_id="bad")["session"]["id"]
        with pytest.raises(ValueError):
            call(manager, sid, "step")
        out = call(manager, sid, "inspect")
        assert out["session"]["state"] == "error"
        assert out["snapshot"]["time"] == 0
        with pytest.raises(ValueError, match="reset"):
            call(manager, sid, "run")
    finally:
        manager.close()


def test_pd_rejects_mismatched_position_servo_gains(profile, tmp_path):
    path = Path(profile["model_path"])
    path.write_text(MJCF.replace('<motor name="motor" joint="hinge" gear="2" ctrllimited="true" ctrlrange="-20 20"/>',
                                 '<position name="motor" joint="hinge" kp="20"/>'))
    manager = SimulationManager({"test": profile}, tmp_path)
    with pytest.raises(ValueError, match="gains do not match"):
        create(manager, control_mode="pd_hold")
    manager.close()


def test_native_position_servo_receives_target_without_second_pd(profile, tmp_path):
    path = Path(profile["model_path"])
    path.write_text(MJCF.replace('<motor name="motor" joint="hinge" gear="2" ctrllimited="true" ctrlrange="-20 20"/>',
                                 '<position name="motor" joint="hinge" kp="20" kv="1"/>'))
    profile["default_joint_pos"] = [.3]
    manager = SimulationManager({"test": profile}, tmp_path)
    try:
        sid = create(manager, control_mode="pd_hold")["session"]["id"]
        session = manager.sessions[sid]
        assert session.position_servos.tolist() == [True]
        assert session.data.ctrl[0] == pytest.approx(.3)
        call(manager, sid, "step", steps=5)
        assert session.data.ctrl[0] == pytest.approx(.3)
        free_id = create(manager)["session"]["id"]
        import mujoco
        assert manager.sessions[free_id].model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        assert not session.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    finally:
        manager.close()


@pytest.fixture
def bounded_position_profile(profile):
    path = Path(profile["model_path"])
    path.write_text(MJCF.replace('<freejoint name="root"/>', '')
        .replace('gravity="0 0 -9.81"', 'gravity="0 0 -9.81" integrator="implicitfast"')
        .replace('axis="0 1 0"', 'axis="0 0 1"')
        .replace('<motor name="motor" joint="hinge" gear="2" ctrllimited="true" ctrlrange="-20 20"/>',
                 '<position name="motor" joint="hinge" kp="20" kv="1" '
                 'ctrllimited="false" forcelimited="true" forcerange="-.5 .5"/>'))
    profile["joint_limits"] = {"lower": [-math.pi / 2], "upper": [math.pi / 2]}
    return profile


@pytest.mark.parametrize("control_limit", [None, 1.6])
def test_bounded_position_policy_matches_direct_physics_beyond_joint_target(
        bounded_position_profile, tmp_path, monkeypatch, control_limit):
    """Compare the same state/actions to MuJoCo's native position contract.

    The desired reference crosses the mechanical limit; actual joint-limit
    constraints must activate and every physics substep remains force bounded.
    """
    import mujoco
    import numpy as np
    from motion_service.contracts import model_fingerprint

    profile = bounded_position_profile
    path = Path(profile["model_path"])
    if control_limit is not None:
        path.write_text(path.read_text().replace('ctrllimited="false"',
                        f'ctrllimited="true" ctrlrange="-{control_limit} {control_limit}"'))
    targets = [2.0, 2.2, 1.8] * 10

    class Controller:
        initial_state = {"joint_pos": [1.55], "joint_vel": [0.]}

        def reset(self):
            self.index = 0

        def step(self, observation):
            target = targets[self.index]
            self.index += 1
            return [target]

    manager = SimulationManager({"test": profile}, tmp_path, model_resolver=lambda *_: Controller())
    try:
        sid = create(manager, model_id="position-contract")["session"]["id"]
        session = manager.sessions[sid]
        reference_model = mujoco.MjModel.from_xml_path(str(path))
        reference = mujoco.MjData(reference_model)
        reference.qpos[:] = session.data.qpos
        reference.qvel[:] = session.data.qvel
        reference.ctrl[:] = session.data.ctrl
        mujoco.mj_forward(reference_model, reference)
        fingerprint = model_fingerprint(reference_model)
        assert model_fingerprint(session.model) == fingerprint
        assert session.force_limited_position_servos.tolist() == [True]
        substep_forces, actual_positions, constraints = [], [], []
        direct_step = mujoco.mj_step

        def check_physics_step(model, data):
            direct_step(model, data)
            if data is session.data:
                substep_forces.append(float(data.actuator_force[0]))
                actual_positions.append(float(data.qpos[session.qadr[0]]))
                constraints.append(bool(np.any(data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)))

        monkeypatch.setattr(mujoco, "mj_step", check_physics_step)
        for target in targets:
            # Supply raw setpoints directly; MuJoCo itself applies any
            # actuator control/force bounds in this independent reference.
            reference.ctrl[:] = target
            for _ in range(session.substeps):
                direct_step(reference_model, reference)
            mujoco.mj_forward(reference_model, reference)
            call(manager, sid, "step")
            np.testing.assert_allclose(session.data.qpos, reference.qpos, rtol=0, atol=1e-12)
            np.testing.assert_allclose(session.data.qvel, reference.qvel, rtol=0, atol=1e-12)
            np.testing.assert_allclose(session.data.actuator_force, reference.actuator_force, rtol=0, atol=1e-12)
            assert session.data.ctrl[0] == pytest.approx(min(target, control_limit) if control_limit else target)
        assert len(substep_forces) == len(targets) * session.substeps
        assert max(abs(force) for force in substep_forces) <= .5 + 1e-12
        assert any(constraints)
        # MuJoCo's joint-limit constraints are soft; they constrain the actual
        # state near the mechanical stop rather than snapping it to the target.
        assert max(actual_positions) < min(targets)
        assert actual_positions[-1] < math.pi / 2 + .01
        assert model_fingerprint(session.model) == fingerprint
        assert session.state == "paused"
    finally:
        manager.close()


@pytest.mark.parametrize("protection", ["no_force_limit", "no_joint_limit", "disabled_joint_limit",
                                        "disabled_constraints"])
def test_position_target_guard_remains_without_native_force_and_joint_protection(
        bounded_position_profile, tmp_path, protection):
    profile = bounded_position_profile
    path = Path(profile["model_path"])
    xml = path.read_text()
    if protection == "no_force_limit":
        xml = xml.replace('forcelimited="true"', 'forcelimited="false"')
    elif protection == "no_joint_limit":
        xml = xml.replace('range="-90 90"', 'range="-90 90" limited="false"')
    else:
        flag = "limit" if protection == "disabled_joint_limit" else "constraint"
        xml = xml.replace('integrator="implicitfast"/>',
                          f'integrator="implicitfast"><flag {flag}="disable"/></option>')
    path.write_text(xml)

    class Controller:
        def reset(self):
            pass

        def step(self, observation):
            return [2.]

    manager = SimulationManager({"test": profile}, tmp_path, model_resolver=lambda *_: Controller())
    try:
        sid = create(manager, model_id="unprotected-position")["session"]["id"]
        with pytest.raises(ValueError, match="target exceeds robot joint limits"):
            call(manager, sid, "step")
        assert manager.sessions[sid].data.time == 0
        assert manager.sessions[sid].state == "error"
    finally:
        manager.close()


@pytest.mark.parametrize("invalid", ["initial_pose", "nonfinite_target"])
def test_bounded_position_servo_still_rejects_invalid_policy_state(
        bounded_position_profile, tmp_path, invalid):
    class Controller:
        initial_state = {"joint_pos": [2.]} if invalid == "initial_pose" else {}

        def reset(self):
            pass

        def step(self, observation):
            return [float("nan")]

    manager = SimulationManager({"test": bounded_position_profile}, tmp_path, model_resolver=lambda *_: Controller())
    try:
        if invalid == "initial_pose":
            with pytest.raises(ValueError, match="initial joint position exceeds limits"):
                create(manager, model_id="invalid-initial-position")
        else:
            sid = create(manager, model_id="nonfinite-target")["session"]["id"]
            with pytest.raises(ValueError, match="policy target"):
                call(manager, sid, "step")
            assert manager.sessions[sid].data.time == 0
    finally:
        manager.close()


def test_perturbation_has_finite_duration(manager):
    sid = create(manager)["session"]["id"]
    call(manager, sid, "perturb", force=[20, 0, 0], duration_s=.04)
    call(manager, sid, "step", steps=2)
    session = manager.sessions[sid]
    assert session.data.xfrc_applied.sum() == 0
    assert session.data.qvel[0] > 0
    with pytest.raises(ValueError):
        call(manager, sid, "perturb", force=[math.inf, 0, 0])


def test_record_replay_does_not_mutate_live_physics(manager, tmp_path):
    sid = create(manager)["session"]["id"]
    call(manager, sid, "record")
    live = call(manager, sid, "step", steps=5)
    stop = call(manager, sid, "record_stop")
    assert stop["session"]["recorded_frames"] == 6
    assert stop["session"]["recording"] is False
    path = tmp_path / "recordings" / (stop["session"]["record_id"] + ".jsonl")
    assert len(path.read_text().splitlines()) == 7
    replay = call(manager, sid, "replay", offset=0, limit=2)
    assert replay["frames"][0]["time"] == 0
    assert replay["frames"][0]["step"] == 0
    assert replay["frames"][1]["time"] == pytest.approx(.02)
    assert replay["frames"][1]["step"] == 1
    assert replay["next_offset"] == 2
    assert replay["total"] == 6
    assert call(manager, sid, "inspect")["snapshot"] == live["snapshot"]


def test_session_capacity_releases_on_close(manager):
    ids = [create(manager)["session"]["id"] for _ in range(MAX_SESSIONS)]
    with pytest.raises(ValueError, match="maximum"):
        create(manager)
    call(manager, ids[0], "close")
    create(manager)


def test_no_silent_fake_policy_or_dataset_rebinding(manager):
    with pytest.raises(ValueError, match="not configured"):
        create(manager, model_id="not-an-installed-policy")
    with pytest.raises(ValueError, match="bundled reference"):
        create(manager, dataset_id="different-reference")


def test_mjlab_export_namespaces_are_resolved_explicitly(profile, tmp_path):
    path = Path(profile["model_path"])
    path.write_text(MJCF.replace('name="hinge"', 'name="robot/hinge"')
                   .replace('joint="hinge"', 'joint="robot/hinge"')
                   .replace('name="robot"', 'name="entity/robot"'))
    profile["root_body_name"] = "robot"
    manager = SimulationManager({"test": profile}, tmp_path)
    try:
        out = create(manager, control_mode="pd_hold")
        assert out["snapshot"]["joint_names"] == ["hinge"]
        assert out["snapshot"]["bodies"][0]["name"] == "robot"
        assert out["snapshot"]["bodies"][0]["model_name"] == "entity/robot"
    finally:
        manager.close()


def test_binary_model_can_be_loaded(profile, tmp_path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(profile["model_path"])
    target = tmp_path / "robot.mjb"
    mujoco.mj_saveModel(model, str(target))
    profile["model_path"] = str(target)
    manager = SimulationManager({"test": profile}, tmp_path)
    try:
        out = create(manager)
        sid = out["session"]["id"]
        assert call(manager, sid, "step")["snapshot"]["time"] > 0
    finally:
        manager.close()


@pytest.mark.parametrize("outside_pd_target", [False, True])
def test_http_service_runs_real_onnx_in_real_physics(tmp_path, outside_pd_target):
    """Exercise the producer/consumer boundary, with an executable test graph."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from test_motion_policy import make_assets
    from motion_service.server import MotionService, make_handler
    from motion_service.store import atomic_json, file_sha256
    from http.server import ThreadingHTTPServer
    import base64
    import shutil
    import threading
    import urllib.request

    path, profile, reference, manifest = make_assets(tmp_path, frames=3)
    if outside_pd_target:
        import numpy as np
        import onnx
        # Raw action 12 maps to 2.5 radians, outside the 2.094-radian
        # mechanical limit. This must reach the bounded native PD servo.
        model = onnx.load(path)
        model.graph.node[0].CopyFrom(onnx.helper.make_node("Constant", [], ["actions"],
            value=onnx.numpy_helper.from_array(np.array([[12., 0.]], dtype=np.float32))))
        onnx.checker.check_model(model)
        onnx.save(model, path)
    scene = tmp_path / "scene.xml"
    scene.write_text('''<mujoco><option timestep=".002" integrator="implicitfast"/>
      <worldbody><geom type="plane" size="2 2 .1"/>
        <body name="robot/pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
          <body name="robot/arm" pos="0 0 .15"><joint name="robot/joint_a" axis="0 1 0" range="-120 120"/>
            <geom type="capsule" fromto="0 0 0 .15 0 0" size=".03" mass=".1"/>
            <body name="robot/hand" pos=".15 0 0"><joint name="robot/joint_b" axis="0 1 0" range="-120 120"/>
              <geom type="sphere" size=".03" mass=".1"/>
            </body></body></body></worldbody>
      <actuator><position joint="robot/joint_a" kp="20" kv="1" forcerange="-1 1"/>
        <position joint="robot/joint_b" kp="30" kv="2" forcerange="-1 1"/></actuator></mujoco>''')
    profile["model_path"] = str(scene)
    service = MotionService(tmp_path / "service", [profile])
    token = "test-motion-api-token"
    http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, token))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()

    def request(action, **args):
        request = urllib.request.Request(f"http://127.0.0.1:{http.server_port}/v1/call",
            data=json.dumps({"action": action, **args}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    try:
        content = reference.read_bytes()
        upload = request("upload_begin", name="reference.npz", size=len(content), profile_id=profile["id"])
        request("upload_chunk", upload_id=upload["upload_id"], offset=0, data=base64.b64encode(content).decode())
        dataset = request("upload_finish", upload_id=upload["upload_id"])["dataset"]
        job = service.store.root / "jobs" / "test-model"
        job.mkdir()
        shutil.copyfile(path, job / path.name)
        atomic_json(job / "run.json", {"test_fixture": True})
        record = {**manifest, "id": "test-model", "job_id": "test-model", "onnx": path.name,
                  "profile_hash": service.profiles[profile["id"]]["profile_hash"],
                  "profile_id": profile["id"], "dataset_id": dataset["id"],
                  "artifact_sha256": {name: file_sha256(job / name) for name in (path.name, "run.json")}}
        atomic_json(service.store.root / "models" / "test-model.json", record)
        initial = request("simulation_create", profile_id=profile["id"], model_id="test-model")
        assert initial["session"]["control_mode"] == "policy"
        sid = initial["session"]["id"]
        result = request("simulation_step", session_id=sid, steps=3)
        assert result["snapshot"]["time"] == pytest.approx(.06)
        assert result["session"]["reference_complete"] is True
        assert result["session"]["reference_frame"] == 2
        assert result["snapshot"]["root_pose"]["pos"][2] < 1
        assert result["snapshot"]["joint_pos"] != initial["snapshot"]["joint_pos"]
        if outside_pd_target:
            session = service.simulations.sessions[sid]
            assert session.data.ctrl[0] == pytest.approx(2.5)
            assert abs(session.data.actuator_force[0]) <= 1.
        assert request("simulation_close", session_id=sid)["session"]["state"] == "closed"
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
        service.close()


def test_shutdown_rejects_new_sessions(manager):
    manager.close()
    with pytest.raises(ValueError, match="closing"):
        create(manager)


@pytest.mark.parametrize("steps", [0, -1, 101, True, 1.1])
def test_step_count_is_bounded(manager, steps):
    sid = create(manager)["session"]["id"]
    with pytest.raises(ValueError, match="integer"):
        call(manager, sid, "step", steps=steps)
