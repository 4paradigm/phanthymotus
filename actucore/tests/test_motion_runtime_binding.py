"""Training/evaluation must use the same compiled baseline as native sessions."""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

mujoco = pytest.importorskip("mujoco")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_service.contracts import model_fingerprint
from motion_service.worker import _validate_environment


@pytest.fixture
def runtime(monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".005"/>'
        '<worldbody><body><joint name="hinge"/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody>'
        '<actuator><motor joint="hinge"/></actuator></mujoco>')
    profile = {"joint_names": ["hinge"], "control_hz": 50, "physics_dt": .005,
               "default_joint_pos": [0.], "action_scale": [.1], "kp": [10.], "kd": [1.],
               "mujoco_version": mujoco.__version__,
               "compiled_model_sha256": model_fingerprint(model)}
    metadata = {"joint_names": ["hinge"], "default_joint_pos": [0.],
                "action_scale": [.1], "joint_stiffness": [10.], "joint_damping": [1.]}
    exporter = ModuleType("mjlab.rl.exporter_utils")
    exporter.get_base_metadata = lambda env, name: metadata
    for name in ("mjlab", "mjlab.rl"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "mjlab.rl.exporter_utils", exporter)
    env = SimpleNamespace(step_dt=.02, sim=SimpleNamespace(mj_model=model))
    return env, profile, metadata


def test_matching_runtime_baseline_is_admitted(runtime):
    env, profile, _ = runtime
    _validate_environment(env, profile)


def test_per_world_randomization_does_not_change_baseline_binding(runtime):
    env, profile, _ = runtime
    # mjlab's DR writes sim.model (the Warp bridge), not sim.mj_model.
    env.sim.model = SimpleNamespace(body_mass=[[1.2], [.8]],
                                    geom_friction=[[[.3, .005, .0001]], [[1.2, .005, .0001]]])
    _validate_environment(env, profile)


@pytest.mark.parametrize("field", ["mass", "friction", "solver"])
def test_changed_physics_is_rejected_with_unchanged_metadata(runtime, field):
    env, profile, _ = runtime
    if field == "mass":
        env.sim.mj_model.body_mass[1] *= 2
    elif field == "friction":
        env.sim.mj_model.geom_friction[0, 0] *= 2
    else:
        env.sim.mj_model.opt.iterations += 1
    with pytest.raises(ValueError, match="运行模型资产或物理配置"):
        _validate_environment(env, profile)


def test_different_mujoco_version_is_rejected(runtime):
    env, profile, _ = runtime
    profile["mujoco_version"] = "different-runtime"
    with pytest.raises(ValueError, match="MuJoCo 版本"):
        _validate_environment(env, profile)


@pytest.mark.parametrize("field", ["mujoco_version", "compiled_model_sha256"])
def test_missing_runtime_binding_is_rejected(runtime, field):
    env, profile, _ = runtime
    del profile[field]
    with pytest.raises(ValueError, match="缺少完整模型绑定"):
        _validate_environment(env, profile)


def test_matching_model_still_requires_matching_controller_metadata(runtime):
    env, profile, metadata = runtime
    metadata["joint_stiffness"] = [20.]
    with pytest.raises(ValueError, match="上游任务 kp"):
        _validate_environment(env, profile)
