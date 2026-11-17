"""Create and step the official GPU G1 task using a temporary static diagnostic.

This does not train a policy and never registers a dataset with the Motion API.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("--job-id", required=True)
parser.add_argument("--profiles", required=True)
args = parser.parse_args()
if not re.fullmatch(r"[a-f0-9]{32}", args.job_id):
    raise ValueError("invalid probe job id")
try:
    os.setsid()
except PermissionError:
    pass
prefix = Path("/tmp/phanthymotus-environment-probe-" + args.job_id)
fd = os.open(str(prefix) + ".pid", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write(str(os.getpid()))

import mujoco
import numpy as np
import torch
import supermimic.tasks
from mjlab.tasks.registry import load_env_cfg
from supermimic.scripts.train_utils import create_env

profile = json.loads(Path(args.profiles).read_text())[0]
if mujoco.__version__ != profile["mujoco_version"]:
    raise RuntimeError("GPU runtime MuJoCo version differs from robot package")
model = mujoco.MjModel.from_binary_path(profile["model_path"])
motion = Path(str(prefix) + ".npz")
count = len(profile["joint_names"])
np.savez_compressed(motion, joint_names=np.array(profile["joint_names"]),
                    joint_pos=np.tile(profile["default_joint_pos"], (150, 1)),
                    joint_vel=np.zeros((150, count)),
                    root_pos_w=np.tile(model.qpos0[:3], (150, 1)),
                    root_quat_w=np.tile(model.qpos0[3:7], (150, 1)),
                    root_lin_vel_w=np.zeros((150, 3)), root_ang_vel_w=np.zeros((150, 3)),
                    fps=np.array(profile["control_hz"]))
env = None
try:
    cfg = load_env_cfg(profile["task_id"], play=True)
    cfg.commands["motion"].motion_file = str(motion)
    cfg.commands["motion"].sampling_mode = "start"
    env = create_env(cfg, 1, "cuda:0")
    baseline = env.unwrapped.sim.mj_model
    content = np.zeros(mujoco.mj_sizeModel(baseline), dtype=np.uint8)
    mujoco.mj_saveModel(baseline, None, content)
    digest = hashlib.sha256(content).hexdigest()
    if digest != profile["compiled_model_sha256"]:
        raise RuntimeError("Official GPU task model does not match the robot package")
    env.reset()
    for _ in range(2):
        env.step(torch.zeros((1, count), device="cuda:0"))
    state = env.unwrapped.sim.data.qpos.detach().cpu().numpy()
    if not np.isfinite(state).all():
        raise RuntimeError("Official GPU physics produced nonfinite state")
    print(json.dumps({"mjlab_gpu_physics": True, "num_envs": 1, "steps": 2,
                      "compiled_model_sha256": digest, "diagnostic_motion": True,
                      "dataset_registered": False, "training_started": False}), flush=True)
finally:
    if env is not None:
        env.close()
    motion.unlink(missing_ok=True)
