"""Validated, versioned inputs shared by jobs and interactive sessions."""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path

PROTOCOL = "motus.motion/1"
ID = re.compile(r"[A-Za-z0-9_-]{1,96}\Z")
TASKS = {"SuperMimic-GenCoord-G1", "SuperMimic-MaxCoord-G1"}
_MODEL_LOAD_LOCK = threading.RLock()


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("无效资源标识")
    return value


def integer(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} 必须是 {low}–{high} 的整数")
    return value


def vector(value, width, name, nonnegative=False):
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{name} 维度必须为 {width}")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
           or (nonnegative and v < 0) for v in value):
        raise ValueError(f"{name} 包含无效数值")
    return [float(v) for v in value]


def profile_hash(profile):
    public = {k: v for k, v in profile.items() if k not in {"model_path", "profile_hash"}}
    return hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_fingerprint(model):
    """Hash the complete compiled model, including its resolved asset content.

    MuJoCo's MJB representation contains the model arrays, meshes and textures;
    hashing only the source XML would miss changed include/mesh files. The
    caller must invoke this before mutating the model for a simulation session.
    Fingerprints are bound to ``mujoco.__version__`` in the profile because MJB
    is a versioned MuJoCo format, not a cross-version serialization protocol.
    """
    import mujoco
    import numpy as np

    # Zero-initialize in case a future writer leaves alignment padding unused.
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(buffer).hexdigest()


def load_model_fresh(path):
    """Load the actual asset contents, bypassing MuJoCo's timestamp cache.

    Replacing an external mesh within the same timestamp resolution can leave
    the compiler cache serving stale geometry. Both profile admission and
    session creation use this loader, so content binding never depends on that
    cache. Clearing cached assets does not change its configured capacity or
    the arrays of any already compiled simulation model.
    """
    import mujoco

    path = Path(path)
    if path.suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(path))
    if path.suffix.lower() != ".xml":
        raise ValueError("模型必须为 MJCF 或 MJB")
    with _MODEL_LOAD_LOCK:
        cache = mujoco.mj_getCache()
        mujoco.mj_clearCache(cache)
        return mujoco.MjModel.from_xml_path(str(path))


def validate_profile(raw):
    p = dict(raw)
    identifier(p.get("id"))
    for key in ("label", "robot_type", "model_version"):
        if not isinstance(p.get(key), str) or not 1 <= len(p[key]) <= 128:
            raise ValueError(f"机器人配置缺少 {key}")
    names = p.get("joint_names")
    if (not isinstance(names, list) or not 1 <= len(names) <= 64
            or any(not isinstance(n, str) or not n or len(n) > 100 for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("joint_names 必须是有序且不重复的关节名称")
    n = len(names)
    hz = p.get("control_hz")
    dt = p.get("physics_dt")
    if (isinstance(hz, bool) or not isinstance(hz, (float, int)) or not 1 <= hz <= 1000
            or isinstance(dt, bool) or not isinstance(dt, (float, int)) or not 0 < dt <= 1 / hz):
        raise ValueError("control_hz 或 physics_dt 无效")
    if abs(1 / hz / dt - round(1 / hz / dt)) > 1e-6:
        raise ValueError("控制周期必须是物理步长的整数倍")
    for key in ("default_joint_pos", "action_scale", "kp", "kd"):
        p[key] = vector(p.get(key), n, key, key in {"kp", "kd"})
    lim = p.get("joint_limits", {})
    low, high = vector(lim.get("lower"), n, "lower"), vector(lim.get("upper"), n, "upper")
    if any(a >= b or not a <= q <= b for a, b, q in zip(low, high, p["default_joint_pos"])):
        raise ValueError("关节限位或默认位置无效")
    p["joint_limits"] = {"lower": low, "upper": high}
    integer(p.get("history_length", 10), 1, 64, "history_length")
    p.setdefault("history_length", 10)
    if p.get("quaternion_order") != "wxyz":
        raise ValueError("四元数顺序必须显式声明 wxyz")
    if p.get("task_id") is not None and p["task_id"] not in TASKS:
        raise ValueError("当前仅支持官方 G1 GenCoord / MaxCoord 训练任务")
    if p.get("model_path"):
        path = Path(p["model_path"]).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix not in {".xml", ".mjb"}:
            raise ValueError("model_path 必须指向 MJCF 或编译模型")
        p["model_path"] = str(path)
        p["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        import mujoco
        model = load_model_fresh(path)
        p["compiled_model_sha256"] = model_fingerprint(model)
        p["mujoco_version"] = mujoco.__version__
    p["profile_hash"] = profile_hash(p)
    return p


def public_profile(p):
    return {k: v for k, v in p.items() if k != "model_path"}


def validate_motion(path, profile):
    """Reject ambiguous data rather than silently reordering/resampling it."""
    import zipfile
    import numpy as np

    required = {"joint_names", "joint_pos", "joint_vel", "root_pos_w", "root_quat_w",
                "root_lin_vel_w", "root_ang_vel_w", "fps"}
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        if len(entries) > 32 or len({i.filename for i in entries}) != len(entries):
            raise ValueError("动作文件包含重复或过多数组")
        if sum(i.file_size for i in entries) > 128 * 1024 * 1024:
            raise ValueError("动作解压后超过 128 MiB")
    with np.load(path, allow_pickle=False) as data:
        if not required <= set(data.files):
            raise ValueError("动作缺少字段：" + ", ".join(sorted(required - set(data.files))))
        names = data["joint_names"]
        if names.ndim != 1 or names.dtype.kind != "U" or names.tolist() != profile["joint_names"]:
            raise ValueError("动作关节名称或顺序与机器人配置不匹配")
        q = data["joint_pos"]
        if q.ndim != 2 or q.shape[1] != len(names) or not 2 <= len(q) <= 1_000_000:
            raise ValueError("joint_pos 帧数或维度无效")
        frames = len(q)
        for key, width in (("joint_pos", len(names)), ("joint_vel", len(names)),
                           ("root_pos_w", 3), ("root_quat_w", 4),
                           ("root_lin_vel_w", 3), ("root_ang_vel_w", 3)):
            a = data[key]
            if a.shape != (frames, width) or a.dtype.kind not in "fi" or not np.isfinite(a).all():
                raise ValueError(f"{key} 维度或数值无效")
        fps_array = data["fps"]
        if fps_array.size != 1 or fps_array.dtype.kind not in "fi":
            raise ValueError("fps 无效")
        fps = float(fps_array.reshape(-1)[0])
        if not math.isfinite(fps) or abs(fps - profile["control_hz"]) > 1e-6:
            raise ValueError("动作采样率与机器人控制频率不匹配")
        if not np.allclose(np.linalg.norm(data["root_quat_w"], axis=1), 1, atol=1e-3):
            raise ValueError("root_quat_w 必须是单位四元数，顺序 wxyz")
        lim = profile["joint_limits"]
        if np.any(q < np.asarray(lim["lower"]) - 1e-4) or np.any(q > np.asarray(lim["upper"]) + 1e-4):
            raise ValueError("参考动作超出关节限位")
        return {"frames": frames, "fps": fps, "duration_s": frames / fps,
                "joint_count": len(names), "deploy_export_available": frames <= 15_000}
