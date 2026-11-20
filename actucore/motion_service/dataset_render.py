"""Render verified reference-motion frames, without creating a physics session.

The dataset supplies world root poses and named scalar joints, not a MuJoCo
qpos layout. Each cached entry resolves that mapping against the bound model
and owns its kinematic data; FrameRenderer owns a separate graphics copy.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from pathlib import Path
import threading

from .contracts import identifier, integer, load_model_fresh, model_fingerprint, validate_motion
from .store import file_sha256


def _resolve(model, kind, name, count):
    import mujoco

    exact = mujoco.mj_name2id(model, kind, name)
    if exact >= 0:
        return exact
    matches = [i for i in range(count)
               if (mujoco.mj_id2name(model, kind, i) or "").rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"动作预览模型中的名称 {name!r} 不存在或不唯一")
    return matches[0]


def _bound_path(profile):
    import mujoco

    path = Path(profile.get("model_path", ""))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("机器人配置没有可用的动作预览模型")
    if (not profile.get("compiled_model_sha256")
            or profile.get("mujoco_version") != mujoco.__version__
            or file_sha256(path) != profile.get("model_sha256")):
        raise ValueError("机器人模型或 MuJoCo 版本已改变，请重新创建机器人配置")
    return path


class _DatasetScene:
    def __init__(self, profile, path, dataset, model_path):
        import mujoco
        import numpy as np
        from .rendering import FrameRenderer

        self.model = model = load_model_fresh(model_path)
        if model_fingerprint(model) != profile["compiled_model_sha256"]:
            raise ValueError("机器人模型资产已改变，请重新创建机器人配置")
        if profile.get("quaternion_order") != "wxyz":
            raise ValueError("动作预览要求 wxyz 根姿态四元数")
        scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        joint_ids = [_resolve(model, mujoco.mjtObj.mjOBJ_JOINT, name, model.njnt)
                     for name in profile["joint_names"]]
        all_scalar = {i for i in range(model.njnt) if int(model.jnt_type[i]) in scalar}
        if len(set(joint_ids)) != len(joint_ids) or set(joint_ids) != all_scalar:
            raise ValueError("动作关节必须完整且唯一地对应模型中的单自由度关节")
        free = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
        if len(free) != 1 or model.njnt != len(joint_ids) + 1:
            raise ValueError("动作预览要求一个自由根关节和单自由度关节，不支持球关节或多机器人模型")
        root = int(free[0])
        self.root_body = int(model.jnt_bodyid[root])
        if (model.body_parentid[self.root_body] != 0
                or any(model.body_rootid[model.jnt_bodyid[j]] != self.root_body for j in joint_ids)):
            raise ValueError("动作关节不属于同一个自由根机器人")
        root_name = profile.get("root_body_name")
        if root_name and _resolve(model, mujoco.mjtObj.mjOBJ_BODY, root_name, model.nbody) != self.root_body:
            raise ValueError("动作根姿态必须对应模型的自由根刚体")
        self.qadr = model.jnt_qposadr[joint_ids].copy()
        self.vadr = model.jnt_dofadr[joint_ids].copy()
        self.root_qadr = int(model.jnt_qposadr[root])
        self.root_vadr = int(model.jnt_dofadr[root])
        self.geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == self.root_body)
        if not len(self.geoms):
            raise ValueError("动作预览机器人没有可渲染的几何体")
        from .camera_fit import RobotCameraFit
        self.fit = RobotCameraFit(model, self.geoms)

        checked = validate_motion(path, profile)
        if any(dataset.get(key) != checked[key] for key in ("frames", "fps", "joint_count")):
            raise ValueError("动作目录记录与文件中的帧数、采样率或关节数量不一致")
        with np.load(path, allow_pickle=False) as data:
            self.motion = {name: np.array(data[name], dtype=float, copy=True) for name in (
                "joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")}
        for values in self.motion.values():
            values.setflags(write=False)
        self.data = mujoco.MjData(model)
        self.renderer = FrameRenderer(model, max_width=960, max_height=640)

    def frame(self, offset, fps):
        import mujoco
        import numpy as np

        data, model = self.data, self.model
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0
        data.qpos[self.qadr] = self.motion["joint_pos"][offset]
        data.qvel[self.vadr] = self.motion["joint_vel"][offset]
        q, v = self.root_qadr, self.root_vadr
        quat = self.motion["root_quat_w"][offset].copy()
        quat /= np.linalg.norm(quat)
        data.qpos[q:q + 3] = self.motion["root_pos_w"][offset]
        data.qpos[q + 3:q + 7] = quat
        data.qvel[v:v + 3] = self.motion["root_lin_vel_w"][offset]
        # MuJoCo free-joint translation is world-frame; angular qvel is local
        # to the free body. The imported NPZ explicitly stores both in world.
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        data.qvel[v + 3:v + 6] = rotation.reshape(3, 3).T @ self.motion["root_ang_vel_w"][offset]
        data.time = offset / fps
        mujoco.mj_forward(model, data)
        return {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "time": float(data.time)}

    def camera(self, width, height, overrides):
        return self.fit.view(self.data, width, height, overrides)


class DatasetRenderer:
    """A bounded, single-request reference viewer, independent of simulation."""

    def __init__(self, store):
        self.store = store
        self._lock = threading.Lock()
        self._scenes = OrderedDict()
        self._closed = False

    def render(self, dataset_id, offset=0, width=960, height=540, azimuth=None,
               elevation=None, distance=None, lookat=None, jpeg_quality=88):
        from .rendering import FrameRenderer

        dataset_id = identifier(dataset_id)
        offset = integer(offset, 0, 999_999, "offset")
        width = integer(width, 160, 960, "width")
        height = integer(height, 120, 640, "height")
        jpeg_quality = integer(jpeg_quality, 40, 95, "jpeg_quality")
        camera = FrameRenderer._camera_parameters({key: value for key, value in {
            "azimuth": azimuth, "elevation": elevation, "distance": distance, "lookat": lookat,
        }.items() if value is not None})
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("动作预览正在渲染，请稍后重试")
        try:
            if self._closed:
                raise RuntimeError("动作预览已关闭")
            dataset, path = self.store.dataset(dataset_id)
            profile = self.store.profile(dataset["profile_id"])
            if offset >= dataset["frames"]:
                raise ValueError("动作预览帧超出文件范围")
            model_path = _bound_path(profile)
            key = (dataset_id, dataset["sha256"], profile["profile_hash"], str(model_path),
                   dataset["frames"], dataset["fps"], dataset["joint_count"])
            if key not in self._scenes:
                while len(self._scenes) >= 2:
                    oldest = next(iter(self._scenes))
                    self._scenes[oldest].renderer.close()
                    del self._scenes[oldest]
                self._scenes[key] = _DatasetScene(profile, path, dataset, model_path)
            self._scenes.move_to_end(key)
            scene = self._scenes[key]
            snapshot = scene.frame(offset, dataset["fps"])
            camera = scene.camera(width, height, camera)
            rendered = scene.renderer.render(snapshot, camera=camera, width=width, height=height,
                                             timeout=5, jpeg_quality=jpeg_quality)
            q = scene.root_qadr
            return {"available": True, "dataset_id": dataset_id, "profile_id": profile["id"],
                    "frame_index": offset, "frames": dataset["frames"], "fps": dataset["fps"],
                    "time": snapshot["time"], "duration_s": dataset["frames"] / dataset["fps"],
                    "root_pose": {"pos": snapshot["qpos"][q:q + 3].tolist(),
                                  "quat": snapshot["qpos"][q + 3:q + 7].tolist()},
                    "camera": camera,
                    "image": {"mime": rendered["mime_type"], "width": rendered["width"],
                              "height": rendered["height"],
                              "data": base64.b64encode(rendered["data"]).decode("ascii")}}
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            self._closed = True
            failure = None
            for key, scene in list(self._scenes.items()):
                try:
                    scene.renderer.close()
                except Exception as exc:
                    failure = failure or exc
                else:
                    del self._scenes[key]
            if failure is not None:
                raise failure
