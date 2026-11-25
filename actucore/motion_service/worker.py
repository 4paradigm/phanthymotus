"""Real upstream training/evaluation in a separate Python/CUDA process.

Uses the researched SuperMimic API; local run metadata replaces W&B lookup for
evaluation. W&B can still be enabled on the server, but credentials never travel
through the card and offline checkpoints remain local.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import os
from pathlib import Path
import sys
import threading
import time

from .store import atomic_json, read_json, file_sha256, verify_artifacts


def _parent_watchdog():
    parent = int(os.environ.get("MOTION_PARENT_PID", "0"))
    if not parent:
        return
    launcher = os.getppid()
    while True:
        # torchrun is the direct parent of ranks; MOTION_PARENT_PID remains the
        # service. Detect either disappearing, without rejecting valid ranks.
        try:
            os.kill(parent, 0)
        except ProcessLookupError:
            os._exit(75)
        if os.getppid() != launcher:
            os._exit(75)
        time.sleep(1)


def _versions():
    return {name: importlib.metadata.version(name)
            for name in ("supermimic", "mjlab", "mujoco", "mujoco-warp", "torch")}


def compare_onnx_actions(exported, reference):
    import numpy as np
    if exported.shape != reference.shape or not np.isfinite(exported).all():
        raise ValueError("ONNX 策略输出维度错误或包含非有限数值")
    difference = float(np.max(np.abs(exported - reference)))
    if not np.isfinite(difference) or difference > 1e-3:
        raise ValueError(f"ONNX 与原策略输出不一致：最大误差 {difference:.6g}")
    return difference


def _environment(request, play=False, topology=None):
    import torch
    import supermimic.tasks  # noqa: F401 — registry producer
    from mjlab.tasks.registry import load_env_cfg
    from supermimic.scripts.train_utils import create_env, setup_device
    if not torch.cuda.is_available():
        raise RuntimeError("SuperMimic 训练与正式评测需要可用的 NVIDIA CUDA GPU")
    device = setup_device(topology.device if topology else "cuda:0")
    torch.cuda.set_device(device)
    p = request["profile"]
    if file_sha256(request["motion_file"]) != request["dataset"]["sha256"]:
        raise ValueError("动作数据在任务提交后已改变或损坏")
    cfg = load_env_cfg(p["task_id"], play=play)
    if topology is not None:
        # Different rollouts and replay draws, shared parameters after broadcast.
        cfg.seed = 42 + topology.rank
    command = cfg.commands["motion"]
    command.motion_file = request["motion_file"]
    command.sampling_mode = "start" if play else "external"
    if play:
        cfg.observations["actor"].enable_corruption = False
    env = create_env(cfg, 1 if play else (topology.envs_per_rank if topology else request["num_envs"]), device)
    try:
        _validate_environment(env.unwrapped, p)
    except Exception:
        env.close()
        raise
    return env, device


def _validate_environment(env, profile):
    import mujoco
    import numpy as np
    from .contracts import model_fingerprint

    if not profile.get("compiled_model_sha256") or not profile.get("mujoco_version"):
        raise ValueError("机器人配置缺少完整模型绑定，请重新生成配置")
    if profile["mujoco_version"] != mujoco.__version__:
        raise ValueError("上游 MuJoCo 版本与机器人配置不一致")
    # Official G1 randomization changes per-world Warp arrays, while mj_model
    # remains the compiled baseline exported by prepare_profile in both modes.
    if model_fingerprint(env.sim.mj_model) != profile["compiled_model_sha256"]:
        raise ValueError("上游运行模型资产或物理配置与机器人配置不一致")

    from mjlab.rl.exporter_utils import get_base_metadata
    meta = get_base_metadata(env, "local")
    if meta["joint_names"] != profile["joint_names"]:
        raise ValueError("上游任务的实际关节顺序与机器人配置不一致")
    if abs(1 / env.step_dt - profile["control_hz"]) > 1e-6:
        raise ValueError("上游任务控制频率与机器人配置不一致")
    if abs(float(env.sim.mj_model.opt.timestep) - profile["physics_dt"]) > 1e-8:
        raise ValueError("上游物理步长与机器人配置不一致")
    for field, key in (("default_joint_pos", "default_joint_pos"), ("action_scale", "action_scale"),
                       ("kp", "joint_stiffness"), ("kd", "joint_damping")):
        if not np.allclose(profile[field], meta[key], rtol=1e-5, atol=1e-6):
            raise ValueError(f"上游任务 {field} 与机器人配置不一致")


def train(request):
    from .distributed import Topology, Synchronizer, RankProgress, run_loop, split_batches
    from .training_preview import PreviewRecorder
    topology = Topology.from_request(request)
    # All-reduce executes eagerly at each real optimizer step. The upstream
    # CapturedUpdate is separately bypassed by Synchronizer for multiple GPUs.
    if request.get("preset", "small") == "small" or topology.world_size > 1:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    updates = {"world_model": 0, "policy": 0}
    progress = RankProgress(topology, request["max_iterations"])
    progress.write("initializing", updates, force=True)
    sync, env, runner, preview = None, None, None, None
    try:
        # A cold torch import can take tens of seconds on the server. Publish
        # the real rank PID/initializing state first, and report import failures.
        import torch
        from supermimic.runners import supermimic_runner as runners
        from supermimic.runners.exporter import export_actor_as_onnx
        sync = Synchronizer(topology)
        env, device = _environment(request, topology=topology)
        cfg = runners.SuperMimicRunnerCfg()
        cfg.actor_obs = "proprio"
        cfg.history_length = request["profile"]["history_length"]
        cfg.max_iterations = request["max_iterations"]
        if request.get("preset", "small") == "small":
            cfg.wm.num_batches = 1
            cfg.wm.batch_size = 64
            cfg.policy.batch_size = 64
            cfg.policy.horizon = 8
        batches = split_batches(cfg, topology)
        cfg.save_interval = max(1, cfg.max_iterations // 10)
        cfg.log_interval = min(10, cfg.max_iterations)
        cfg.representation = "maxcoord" if "MaxCoord" in request["profile"]["task_id"] else "gencoord"
        cfg.run_name = "motus-" + Path.cwd().name
        run = {"runner_cfg": dataclasses.asdict(cfg), "profile": request["profile"],
               "dataset": request["dataset"], "versions": _versions(),
               "training_engine": "mjlab_mujoco_warp", "preset": request.get("preset", "small"),
               "torch_compile": os.environ.get("TORCHDYNAMO_DISABLE") != "1",
               "distributed": {**topology.metadata(), "batches": batches,
                               "cuda_graph_updates": topology.world_size == 1,
                               "normalization": "global_empirical_statistics"}}
        runner = runners.SuperMimicRunner(env=env, cfg=cfg,
                                         task_id=request["profile"]["task_id"],
                                         registry_name=None, device=device)
        modules = sync.prepare(runner)
        run["distributed"]["learner_method_sha256"] = sync.method_hashes
        if topology.primary:
            atomic_json("run.json", run)

        def save_checkpoint(folder):
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(runner._wm.bundle(), folder / "world_model.pt")
            torch.save(runner.policy_learner.actor.state_dict(), folder / "actor.pt")
            critic = getattr(runner.policy_learner, "critic", None)
            if critic is not None:
                torch.save(critic.state_dict(), folder / "critic.pt")

        preview = PreviewRecorder(Path.cwd().resolve(), topology.rank, topology.world_size,
                                  sample_hz=float(os.environ.get("MOTION_TRAINING_PREVIEW_HZ", "4")))
        run_loop(runner, topology, progress, updates, save_checkpoint, preview)
        preview.close()
        if not all(updates.values()):
            raise RuntimeError("迭代次数不足，尚未发生世界模型与策略更新；请增加迭代次数或使用小规模预设")
        progress.write("verifying", updates, force=True)
        ranks = sync.verify(modules, updates)
        if topology.primary:
            folder = Path("final")
            save_checkpoint(folder)
            progress.write("exporting", updates, force=True)
            if runner.motion.time_step_total <= 15000:
                export_actor_as_onnx(actor=runner.policy_learner.actor, env=env.unwrapped,
                                     path=str(folder), run_name=cfg.run_name,
                                     term_widths=runner._representation.proprio_term_widths(runner._num_actions))
            onnx = folder / "policy_motion.onnx"
            result = {"actor": str(folder / "actor.pt"), "world_model": str(folder / "world_model.pt"),
                      "onnx": str(onnx) if onnx.exists() else None,
                      "adapter": "supermimic-motion-v1", "reference_sha256": request["dataset"]["sha256"],
                      "control_hz": request["profile"]["control_hz"], "goal_frames": 10, "goal_stride": 5,
                      "versions": run["versions"], "updates": updates,
                      "preset": run["preset"], "torch_compile": run["torch_compile"],
                      "distributed": {**topology.metadata(), "backend": "nccl" if sync.initialized else None,
                                      "synchronized": True, "ranks": ranks}}
            atomic_json("result.json", result)
        progress.write("succeeded", updates, force=True)
    except BaseException as exc:
        progress.write("failed", updates, force=True, error=exc)
        raise
    finally:
        if preview is not None:
            preview.close()
        if topology.primary and runner is not None:
            import wandb
            wandb.finish()
        if env is not None:
            env.close()
        if sync is not None:
            sync.close()


def evaluate(request):
    folder = Path(request["model_dir"])
    verify_artifacts(folder, request["model"].get("artifact_sha256"))
    run = read_json(folder / "run.json")
    if not run.get("torch_compile", True):
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import numpy as np
    import torch
    from supermimic.runners.run_config import load_runner_cfg, check_representation_matches
    from supermimic.runners.supermimic_runner import build_actor, build_world_model
    from supermimic.tasks.task_spec import build_actor_obs_td
    from supermimic.representations import extract_actor_goal

    cfg = load_runner_cfg(run)
    check_representation_matches(request["profile"]["task_id"], cfg.representation)
    env, device = _environment(request, play=True)
    try:
        wm = build_world_model(cfg, env, device)
        model = request["model"]
        wm.load_bundle(torch.load(folder / model["world_model"], map_location=device, weights_only=True))
        wm.dynamics.eval()
        actor = build_actor(cfg, wm, device)
        actor.load_state_dict(torch.load(folder / model["actor"], map_location=device, weights_only=True))
        actor.eval()
        actor.set_slot_mix({"motion": 1.0})
        rep = wm.representation
        context = wm.sim_context.read(env.unwrapped)
        session = None
        if model.get("onnx"):
            import onnxruntime as ort
            session = ort.InferenceSession(str(folder / model["onnx"]), providers=["CPUExecutionProvider"])
        obs, _ = env.reset()
        history = None
        squared_error, samples, terminated, timeouts, maximum_difference = 0.0, 0, 0, 0, 0.0
        frames = []
        with torch.inference_mode():
            for step in range(request["max_steps"]):
                sensor = rep.compute_actor_sensor(rep.represent(obs["full_state"]), context,
                                                 sim_actor_obs=obs["actor"])
                history = (sensor.unsqueeze(1).repeat(1, cfg.history_length, 1) if history is None
                           else torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1))
                goal = extract_actor_goal(obs["goal"])
                actions = actor(build_actor_obs_td({"motion": goal}, history))
                if not torch.isfinite(actions).all():
                    raise ValueError("策略输出非有限数值")
                motion = env.unwrapped.command_manager.get_term("motion")
                if session:
                    widths = rep.proprio_term_widths(actions.shape[-1])
                    chunks = torch.split(history, widths, dim=-1)
                    term_major = torch.cat([c.flatten(start_dim=1) for c in chunks], dim=-1)
                    feed = {"obs": torch.cat([term_major, goal], dim=-1).cpu().numpy().astype(np.float32),
                            "time_step": motion.time_steps.reshape(1, 1).cpu().numpy().astype(np.float32)}
                    exported = session.run(["actions"], feed)[0]
                    difference = compare_onnx_actions(exported, actions.cpu().numpy())
                    maximum_difference = max(maximum_difference, difference)
                    # Evaluate the deployment graph's action, including its float32 behavior.
                    actions = torch.as_tensor(exported, device=device)
                robot = env.unwrapped.scene["robot"]
                error = robot.data.joint_pos - motion.joint_pos
                squared_error += float(torch.sum(error * error))
                samples += error.numel()
                if step % max(1, request["max_steps"] // 300) == 0:
                    frames.append({"step": step, "time": step * env.unwrapped.step_dt,
                                   "joint_pos": robot.data.joint_pos[0].cpu().tolist(),
                                   "reference_joint_pos": motion.joint_pos[0].cpu().tolist(),
                                   "body_pos": robot.data.body_link_pos_w[0].cpu().tolist(),
                                   "body_names": list(robot.body_names)})
                obs, _, dones, _ = env.step(actions)
                terminated += int(env.unwrapped.termination_manager.terminated.sum())
                timeouts += int(env.unwrapped.termination_manager.time_outs.sum())
                if bool(dones.any()):
                    history = None
        report = {"engine": "mjlab_mujoco_warp", "steps": request["max_steps"],
                  "dataset_id": request["dataset"]["id"], "profile_hash": request["profile"]["profile_hash"],
                  "joint_rmse_rad": (squared_error / max(samples, 1)) ** .5,
                  "early_terminations": terminated, "timeouts": timeouts,
                  "onnx_checked": session is not None,
                  "onnx_max_abs_error": maximum_difference if session else None,
                  "qualification": "evaluated", "versions": _versions()}
        atomic_json("evaluation.json", report)
        atomic_json("rollout.json", {"frames": frames})
        atomic_json("result.json", {"evaluation": "evaluation.json"})
    finally:
        env.close()


def main():
    threading.Thread(target=_parent_watchdog, daemon=True).start()
    request = read_json(sys.argv[1])
    if request["kind"] == "train":
        # Each rank has independently readable diagnostics; the inherited
        # launcher log still receives all output for an overall failure report.
        rank = int(os.environ.get("RANK", "0"))
        folder = Path("ranks") / f"rank-{rank}"
        folder.mkdir(parents=True, exist_ok=True)
        stream = (folder / "worker.log").open("a", buffering=1)
        class Tee:
            def __init__(self, target):
                self.target = target
            def write(self, value):
                stream.write(value)
                return self.target.write(value)
            def flush(self):
                stream.flush()
                self.target.flush()
            def isatty(self):
                return False
            def __getattr__(self, name):
                return getattr(self.target, name)
        sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
    {"train": train, "evaluate": evaluate}[request["kind"]](request)


if __name__ == "__main__":
    main()
