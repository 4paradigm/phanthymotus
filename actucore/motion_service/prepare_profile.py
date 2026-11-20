"""Extract a G1 profile and portable scene from the installed official task."""
from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import TASKS, validate_profile
from .store import atomic_json


def export_runtime_scene(base, directory):
    """Keep editable assets, but execute the configured runtime model.

    mjlab configures physics after compiling its MjSpec. scene.write() alone
    therefore does not preserve the runtime timestep, solver or other options.
    MJB includes those settings and the resolved assets; the profile also binds
    its MuJoCo version, so loading it in a different runtime is rejected.
    """
    import mujoco

    directory = Path(directory)
    base.scene.write(directory)
    model_path = directory / "scene.mjb"
    mujoco.mj_saveModel(base.sim.mj_model, str(model_path), None)
    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=sorted(TASKS), default="SuperMimic-GenCoord-G1")
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if target.exists():
        parser.error("输出目录已存在，请指定新目录")
    import supermimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.rl.exporter_utils import get_base_metadata
    from supermimic.scripts.train_utils import create_env, setup_device
    cfg = load_env_cfg(args.task, play=True)
    cfg.commands["motion"].motion_file = str(Path(args.motion_file).resolve(strict=True))
    env = create_env(cfg, 1, setup_device())
    try:
        base = env.unwrapped
        meta = get_base_metadata(base, "profile")
        model_path = export_runtime_scene(base, target / "scene")
        limits = base.scene["robot"].data.default_joint_pos_limits[0].cpu().tolist()
        scale = meta["action_scale"]
        if isinstance(scale, (int, float)):
            scale = [scale] * len(meta["joint_names"])
        p = {"id": "g1-" + str(len(meta["joint_names"])) + "dof-v1", "label": "G1 · 官方任务配置",
             "robot_type": "unitree_g1", "model_version": args.task,
             "task_id": args.task, "model_path": str(model_path),
             "joint_names": meta["joint_names"], "control_hz": 1 / base.step_dt,
             "physics_dt": float(base.sim.mj_model.opt.timestep), "history_length": 10,
             "quaternion_order": "wxyz", "default_joint_pos": meta["default_joint_pos"],
             "action_scale": scale, "kp": meta["joint_stiffness"], "kd": meta["joint_damping"],
             "joint_limits": {"lower": [x[0] for x in limits], "upper": [x[1] for x in limits]}}
        atomic_json(target / "profiles.json", [validate_profile(p)])
        print(target / "profiles.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
