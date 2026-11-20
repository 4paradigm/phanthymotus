"""Executable ONNX contracts and observation mathematics, without learned weights."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_service.contracts import profile_hash
from motion_service.policy import SuperMimicPolicy, ACTOR_TERMS, GOAL_TERMS


def _csv(values):
    def scalar(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
    return ",".join(";".join(scalar(v) for v in value) if isinstance(value, list)
                    else scalar(value) for value in values)


def make_assets(root, *, frames=61, history=2, metadata_updates=None, reference_mismatch=False):
    p = {"id": "g1-fixture", "label": "Math fixture", "robot_type": "unitree_g1", "model_version": "test",
         "joint_names": ["joint_a", "joint_b"], "control_hz": 50, "physics_dt": .002,
         "history_length": history, "quaternion_order": "wxyz", "default_joint_pos": [.1, -.2],
         "action_scale": [.2, .3], "kp": [20., 30.], "kd": [1., 2.],
         "joint_limits": {"lower": [-2., -2.], "upper": [2., 2.]}}
    t = np.arange(frames)
    q = np.stack((t*.001, t*.002), axis=1).astype(np.float32)
    dq = np.repeat(np.array([[.05, .1]], dtype=np.float32), frames, axis=0)
    pos = np.repeat(np.array([[0., 0., 1.]], dtype=np.float32), frames, axis=0)
    quat = np.stack((np.cos(t*.01/2), np.zeros(frames), np.zeros(frames), np.sin(t*.01/2)), axis=1).astype(np.float32)
    linear = np.zeros((frames, 3), dtype=np.float32)
    angular = np.repeat(np.array([[0., 0., .5]], dtype=np.float32), frames, axis=0)
    reference = root / "reference.npz"
    np.savez(reference, joint_names=np.asarray(p["joint_names"]), joint_pos=q, joint_vel=dq,
             root_pos_w=pos, root_quat_w=quat, root_lin_vel_w=linear, root_ang_vel_w=angular, fps=50.)
    obs_width = 12*history + 100
    # Actions expose the current first joint observation and first joint target
    # from the NEXT future window frame. This makes input/layout errors visible.
    nodes = [helper.make_node("Gather", ["obs", "action_indices"], ["actions"], axis=1),
             helper.make_node("Cast", ["time_step"], ["frame_int"], to=TensorProto.INT64),
             helper.make_node("Squeeze", ["frame_int", "squeeze_axis"], ["frame"])]
    initializers = [numpy_helper.from_array(np.array([6*history + 2*(history-1), 12*history + 4], dtype=np.int64), "action_indices"),
                    numpy_helper.from_array(np.array([1], dtype=np.int64), "squeeze_axis")]
    tables = {"joint_pos": q.copy(), "joint_vel": dq, "body_pos_w": pos[:, None],
              "body_quat_w": quat[:, None], "body_lin_vel_w": linear[:, None],
              "body_ang_vel_w": angular[:, None]}
    if reference_mismatch:
        tables["joint_pos"][0, 0] += .2
    outputs = [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])]
    for name, values in tables.items():
        initializers.append(numpy_helper.from_array(values, name + "_table"))
        nodes.append(helper.make_node("Gather", [name + "_table", "frame"], [name], axis=0))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, *values.shape[1:]]))
    graph = helper.make_graph(nodes, "test_motion_contract",
                              [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_width]),
                               helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])],
                              outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    metadata = {"joint_names": _csv(p["joint_names"]), "observation_names": _csv([*ACTOR_TERMS, *GOAL_TERMS]),
                "command_names": "motion", "anchor_body_name": "pelvis", "body_names": "pelvis",
                "observation_history_lengths": _csv([history]*5 + [1, 1]),
                "observation_terms_scale": _csv([1.]*5),
                "observation_terms_flatten_history_dim": _csv([True]*5),
                "observation_terms_history_length": _csv([0]*5),
                "observation_terms_clip": _csv([[-math.inf, math.inf]]*5),
                "joint_stiffness": _csv(p["kp"]), "joint_damping": _csv(p["kd"]),
                "default_joint_pos": _csv(p["default_joint_pos"]), "action_scale": _csv(p["action_scale"])}
    metadata.update(metadata_updates or {})
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    path = root / "policy_motion.onnx"
    onnx.save(model, path)
    manifest = {"adapter": "supermimic-motion-v1", "profile_hash": profile_hash(p),
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "control_hz": 50, "goal_frames": 10, "goal_stride": 5}
    return path, p, reference, manifest


def observation(step=0, q=(.5, -.1), quat=(1., 0., 0., 0.), angular=(0., 0., 0.)):
    from motion_service.policy import _rotation
    root = {"pos": [0., 0., 1.], "quat": list(quat)}
    return {"step": step, "time": step*.02, "dt": .02, "root_pose": root,
            "joint_pos": list(q), "joint_vel": [.3, -.4], "root_ang_vel_w": list(angular),
            "root_ang_vel_b": (_rotation(quat).T @ angular).tolist(),
            "bodies": [{"name": "pelvis", **deepcopy(root)}]}


def test_executable_graph_receives_sensor_terms_and_future_reference(tmp_path):
    policy = SuperMimicPolicy(*make_assets(tmp_path))
    target = policy.step(observation())
    assert target == pytest.approx([.18, -.1985])
    obs = policy.last_observation
    assert obs[:6] == pytest.approx([0, 0, -1, 0, 0, -1])
    assert obs[12:16] == pytest.approx([.4, .1, .4, .1])
    assert obs[16:20] == pytest.approx([.3, -.4, .3, -.4])
    assert obs[20:24] == pytest.approx([0, 0, 0, 0])
    # Each goal frame is [position0, position1, velocity0, velocity1].
    assert obs[24:32] == pytest.approx([0, 0, .05, .1, .005, .01, .05, .1])
    assert policy.initial_state["root_pos"] == [0., 0., 1.]


def test_history_is_term_major_chronological_and_uses_raw_previous_action(tmp_path):
    policy = SuperMimicPolicy(*make_assets(tmp_path))
    policy.step(observation())
    policy.step(observation(1, q=(.7, .2)))
    obs = policy.last_observation
    assert obs[12:16] == pytest.approx([.4, .1, .6, .4])
    assert obs[20:24] == pytest.approx([0, 0, .4, .005])
    policy.reset()
    policy.step(observation())
    assert policy.last_observation[20:24] == pytest.approx([0, 0, 0, 0])


def test_gravity_and_relative_rotation_use_current_pelvis_frame(tmp_path):
    policy = SuperMimicPolicy(*make_assets(tmp_path))
    half = math.sqrt(.5)
    policy.step(observation(quat=(half, 0., half, 0.), angular=(1., 2., 3.)))
    obs = policy.last_observation
    assert obs[:3] == pytest.approx([1., 0., 0.], abs=1e-12)
    assert obs[6:9] == pytest.approx([-3., 2., 1.])
    # R(pelvis)^T R(reference), first two columns flattened BY ROW.
    assert obs[64:70] == pytest.approx([0, 0, 0, 1, 1, 0], abs=1e-12)


def test_future_window_clamps_to_single_clip_end(tmp_path):
    policy = SuperMimicPolicy(*make_assets(tmp_path, frames=3))
    for i in range(5):
        policy.step(observation(i))
    assert policy.reference_frame == 2
    assert policy.reference_complete is True
    assert policy.last_observation[24:64].reshape(10, 4) == pytest.approx(
        np.repeat([[.002, .004, .05, .1]], 10, axis=0))
    policy.reset()
    assert policy.reference_complete is False


@pytest.mark.parametrize("updates,match", [
    ({"observation_names": "root_pos,root_quat"}, "observation terms"),
    ({"anchor_body_name": "torso_link"}, "anchor_body_name"),
    ({"command_names": "twist"}, "motion command"),
    ({"joint_names": "joint_b,joint_a"}, "joint names/order"),
    ({"observation_history_lengths": _csv([10]*5 + [1, 1])}, "history"),
    ({"observation_terms_scale": _csv([1., 1., 2., 1., 1.])}, "scaled proprio"),
    ({"observation_terms_clip": _csv([[-1., 1.]]*5)}, "clipped proprio"),
    ({"joint_stiffness": "21.000,30.000"}, "joint_stiffness"),
])
def test_unsupported_metadata_is_rejected_before_execution(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        SuperMimicPolicy(*make_assets(tmp_path, metadata_updates=updates))


def test_metadata_rounding_does_not_replace_precise_profile_parameters(tmp_path):
    path, profile, reference, manifest = make_assets(tmp_path, metadata_updates={"default_joint_pos": ".100,-.200"})
    profile["default_joint_pos"][0] = .10049
    manifest["profile_hash"] = profile_hash(profile)
    policy = SuperMimicPolicy(path, profile, reference, manifest)
    target = policy.step(observation())
    assert target[0] == pytest.approx((.5-.10049)*.2 + .10049)
    profile["default_joint_pos"][0] = .101
    manifest["profile_hash"] = profile_hash(profile)
    with pytest.raises(ValueError, match="default_joint_pos"):
        SuperMimicPolicy(path, profile, reference, manifest)


@pytest.mark.parametrize("field,value", [
    ("adapter", "another_adapter"), ("reference_sha256", "wrong"),
    ("profile_hash", "wrong"), ("control_hz", 100), ("goal_stride", 1),
])
def test_manifest_binds_profile_reference_and_timing(tmp_path, field, value):
    args = list(make_assets(tmp_path))
    args[3][field] = value
    with pytest.raises(ValueError):
        SuperMimicPolicy(*args)


def test_embedded_reference_must_match_bound_npz(tmp_path):
    with pytest.raises(ValueError, match="embedded joint_pos"):
        SuperMimicPolicy(*make_assets(tmp_path, reference_mismatch=True))


@pytest.mark.parametrize("mutation", [
    lambda o: o.update(step=1),
    lambda o: o.update(dt=.01),
    lambda o: o.update(time=1.),
    lambda o: o.update(joint_pos=[float("nan"), 0.]),
    lambda o: o.update(root_ang_vel_b=[1., 0., 0.]),
    lambda o: o["bodies"][0].update(name="torso"),
    lambda o: o["root_pose"].update(quat=[0., 0., 0., 0.]),
])
def test_invalid_observation_does_not_advance_policy_state(tmp_path, mutation):
    policy = SuperMimicPolicy(*make_assets(tmp_path))
    value = observation()
    mutation(value)
    with pytest.raises(ValueError):
        policy.step(value)
    assert policy.step(observation()) == pytest.approx([.18, -.1985])
