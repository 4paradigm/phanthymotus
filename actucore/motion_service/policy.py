"""Native-MuJoCo adapter for SuperMimic's motion/proprio ONNX export.

The layout is taken from upstream runners/exporter.py, tasks/env_cfgs.py and
representations.py (abc124c1): term-major oldest-to-newest proprio history,
then 10 reference frames at stride 5 and their pelvis-relative orientations.
Only this verified export contract is accepted. Future values come exclusively
from the bound reference clip, never from future simulated/robot observations.
This is a native-simulator debugging adapter, not a claim of mjlab parity.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np

from .contracts import profile_hash, validate_motion


ACTOR_TERMS = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions")
GOAL_TERMS = ("command", "motion_anchor_ori_b")
OUTPUTS = ("actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
           "body_lin_vel_w", "body_ang_vel_w")
GOAL_FRAMES = 10
GOAL_STRIDE = 5
METADATA_ATOL = 0.00051  # Upstream attach_metadata_to_onnx serializes 3 decimals.


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid numbers") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result


def _rotation(quaternion):
    q = _array(quaternion, (4,), "wxyz quaternion")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1., atol=1e-3, rtol=0):
        raise ValueError("quaternion must be unit length in wxyz order")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _csv(metadata, key):
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"ONNX metadata missing {key}")
    return [part.strip() for part in value.split(",")]


def _numbers(metadata, key, count):
    try:
        values = [float(v) for v in _csv(metadata, key)]
    except ValueError as exc:
        raise ValueError(f"invalid ONNX metadata {key}") from exc
    return _array(values, (count,), key)


class SuperMimicPolicy:
    """Validated ONNX policy with a single reference clip and explicit history.

    ``step`` accepts SimulationSession's current observation and returns joint
    position targets in profile order. The returned action is transformed once
    by ``raw_action * action_scale + default_joint_pos``; normalization already
    lives inside the upstream graph. ``reset`` clears raw-action/history state.
    """

    def __init__(self, model_path, profile, reference_path, manifest=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is not installed") from exc
        self.profile = deepcopy(profile)
        self.manifest = deepcopy(manifest or {})
        self._validate_binding(reference_path)
        stats = validate_motion(reference_path, profile)
        if stats["frames"] > 15_000:
            raise ValueError("SuperMimic deploy clips are limited to 15000 frames")
        with np.load(reference_path, allow_pickle=False) as data:
            self.reference = {name: np.asarray(data[name], dtype=np.float64).copy()
                              for name in ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w",
                                           "root_lin_vel_w", "root_ang_vel_w")}
        self.n = len(profile["joint_names"])
        self.history_length = profile["history_length"]
        self.reference_frames = stats["frames"]
        self.widths = (3, 3, self.n, self.n, self.n)
        self.hist_size = sum(self.widths) * self.history_length
        self.obs_size = self.hist_size + GOAL_FRAMES * (2*self.n + 6)
        self.default_q = _array(profile["default_joint_pos"], (self.n,), "default_joint_pos")
        self.action_scale = _array(profile["action_scale"], (self.n,), "action_scale")
        if np.any(self.action_scale <= 0):
            raise ValueError("action_scale must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(model_path)), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self._validate_graph()
        self.initial_state = {
            "joint_pos": self.reference["joint_pos"][0].tolist(),
            "joint_vel": self.reference["joint_vel"][0].tolist(),
            "root_pos": self.reference["root_pos_w"][0].tolist(),
            "root_quat": self.reference["root_quat_w"][0].tolist(),
            "root_lin_vel_w": self.reference["root_lin_vel_w"][0].tolist(),
            "root_ang_vel_w": self.reference["root_ang_vel_w"][0].tolist(),
        }
        # These output tables must belong to the supplied clip. Each subsequent
        # frame is checked again on inference; the manifest binds the full NPZ.
        zero = np.zeros((1, self.obs_size), dtype=np.float32)
        for frame in (0, self.reference_frames - 1):
            self._check_reference_outputs(self._infer(zero, frame), frame)
        self.reset()

    def _validate_binding(self, reference_path):
        m, p = self.manifest, self.profile
        if m.get("adapter") != "supermimic-motion-v1":
            raise ValueError("a supermimic-motion-v1 deployment manifest is required")
        if m.get("profile_hash") != profile_hash(p):
            raise ValueError("model deployment profile hash does not match")
        digest = hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()
        if m.get("reference_sha256") != digest:
            raise ValueError("reference clip checksum does not match deployment manifest")
        if p.get("robot_type") != "unitree_g1" or p.get("quaternion_order") != "wxyz":
            raise ValueError("only the Unitree G1 wxyz motion export is supported")
        if p.get("root_body_name", "pelvis") != "pelvis":
            raise ValueError("SuperMimic motion observations require pelvis as robot root")
        if p.get("control_hz") != 50 or m.get("control_hz") != 50:
            raise ValueError("the verified SuperMimic reference contract requires 50 Hz")
        if m.get("goal_frames") != GOAL_FRAMES or m.get("goal_stride") != GOAL_STRIDE:
            raise ValueError("unsupported reference window contract")
        h = p.get("history_length")
        if type(h) is not int or not 1 <= h <= 64:
            raise ValueError("invalid policy history_length")

    def _validate_graph(self):
        inputs = {value.name: value for value in self.session.get_inputs()}
        if set(inputs) != {"obs", "time_step"}:
            raise ValueError("only motion ONNX inputs obs/time_step are supported")
        for name, shape in (("obs", [1, self.obs_size]), ("time_step", [1, 1])):
            if inputs[name].shape != shape or inputs[name].type != "tensor(float)":
                raise ValueError(f"ONNX {name} shape/type does not match motion contract")
        outputs = {value.name: value for value in self.session.get_outputs()}
        if set(outputs) != set(OUTPUTS):
            raise ValueError("ONNX outputs do not match the SuperMimic motion export")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if _csv(metadata, "joint_names") != self.profile["joint_names"]:
            raise ValueError("ONNX joint names/order do not match robot profile")
        if tuple(_csv(metadata, "observation_names")) != ACTOR_TERMS + GOAL_TERMS:
            raise ValueError("unsupported ONNX observation terms/order")
        if _csv(metadata, "command_names") != ["motion"]:
            raise ValueError("only motion command exports are supported")
        if metadata.get("anchor_body_name") != "pelvis":
            raise ValueError("ONNX anchor_body_name must be pelvis")
        self.body_names = _csv(metadata, "body_names")
        if len(set(self.body_names)) != len(self.body_names) or "pelvis" not in self.body_names:
            raise ValueError("invalid tracked body names")
        self.anchor_index = self.body_names.index("pelvis")
        histories = _numbers(metadata, "observation_history_lengths", 7)
        if histories.tolist() != [self.history_length]*5 + [1, 1]:
            raise ValueError("ONNX observation history does not match profile")
        # The exporter preserves base metadata for the five actor terms only.
        if not np.array_equal(_numbers(metadata, "observation_terms_scale", 5), np.ones(5)):
            raise ValueError("scaled proprio terms are not supported by this export adapter")
        # bool is an int in Python: upstream's CSV formatter emits 1.000 for
        # True, not the string "True".
        if not np.array_equal(_numbers(metadata, "observation_terms_flatten_history_dim", 5), np.ones(5)):
            raise ValueError("only flattened term-major histories are supported")
        clips = _csv(metadata, "observation_terms_clip")
        if len(clips) != 5:
            raise ValueError("invalid proprio clipping metadata")
        for clip in clips:
            try:
                lo, hi = (float(v) for v in clip.split(";"))
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid proprio clipping metadata") from exc
            if lo != -math.inf or hi != math.inf:
                raise ValueError("clipped proprio terms are not supported")
        for metadata_name, profile_name in (("default_joint_pos", "default_joint_pos"),
                                            ("action_scale", "action_scale"),
                                            ("joint_stiffness", "kp"), ("joint_damping", "kd")):
            actual = _numbers(metadata, metadata_name, self.n)
            expected = _array(self.profile[profile_name], (self.n,), profile_name)
            if not np.allclose(actual, expected, atol=METADATA_ATOL, rtol=0):
                raise ValueError(f"ONNX {metadata_name} does not match robot profile")
        expected_shapes = {"actions": [1, self.n], "joint_pos": [1, self.n],
                           "joint_vel": [1, self.n], "body_pos_w": [1, len(self.body_names), 3],
                           "body_quat_w": [1, len(self.body_names), 4],
                           "body_lin_vel_w": [1, len(self.body_names), 3],
                           "body_ang_vel_w": [1, len(self.body_names), 3]}
        for name, shape in expected_shapes.items():
            if outputs[name].shape != shape or outputs[name].type != "tensor(float)":
                raise ValueError(f"ONNX {name} output shape/type is invalid")

    def _infer(self, obs, frame):
        values = self.session.run(list(OUTPUTS), {
            "obs": obs, "time_step": np.asarray([[frame]], dtype=np.float32)})
        return dict(zip(OUTPUTS, values))

    def _check_reference_outputs(self, outputs, frame):
        for key in ("joint_pos", "joint_vel"):
            actual = _array(outputs[key], (1, self.n), key)[0]
            if not np.allclose(actual, self.reference[key][frame], atol=1e-5, rtol=1e-5):
                raise ValueError(f"ONNX embedded {key} does not match bound reference clip")
        for key, reference_key, width in (("body_pos_w", "root_pos_w", 3),
                                           ("body_quat_w", "root_quat_w", 4),
                                           ("body_lin_vel_w", "root_lin_vel_w", 3),
                                           ("body_ang_vel_w", "root_ang_vel_w", 3)):
            actual = _array(outputs[key], (1, len(self.body_names), width), key)[0, self.anchor_index]
            expected = self.reference[reference_key][frame]
            if key == "body_quat_w":
                matches = np.allclose(_rotation(actual), _rotation(expected), atol=1e-5, rtol=1e-5)
            else:
                matches = np.allclose(actual, expected, atol=1e-5, rtol=1e-5)
            if not matches:
                raise ValueError(f"ONNX embedded {key} does not match bound reference clip")

    def reset(self):
        self._history = None
        self._last_action = np.zeros(self.n)
        self._next_step = 0
        self.reference_frame = 0
        self.reference_complete = False
        self.last_observation = None

    def _proprio(self, observation):
        root = observation.get("root_pose", {})
        pos = _array(root.get("pos"), (3,), "root position")
        rotation = _rotation(root.get("quat"))
        pelvis = [body for body in observation.get("bodies", []) if body.get("name") == "pelvis"]
        if len(pelvis) != 1:
            raise ValueError("current observation must contain exactly one pelvis body")
        if not np.allclose(pos, _array(pelvis[0].get("pos"), (3,), "pelvis position"), atol=1e-6):
            raise ValueError("observed root is not the pelvis")
        if not np.allclose(rotation, _rotation(pelvis[0].get("quat")), atol=1e-6):
            raise ValueError("observed root orientation differs from pelvis")
        angular = _array(observation.get("root_ang_vel_b"), (3,), "root body angular velocity")
        world_angular = _array(observation.get("root_ang_vel_w"), (3,), "root world angular velocity")
        if not np.allclose(angular, rotation.T @ world_angular, atol=1e-5, rtol=1e-5):
            raise ValueError("root angular velocity frames are inconsistent")
        q = _array(observation.get("joint_pos"), (self.n,), "joint_pos")
        dq = _array(observation.get("joint_vel"), (self.n,), "joint_vel")
        return (rotation.T @ np.array([0., 0., -1.]), angular,
                q - self.default_q, dq, self._last_action.copy()), rotation

    def step(self, observation):
        step = observation.get("step")
        if type(step) is not int or step != self._next_step:
            raise ValueError("policy observations must advance one control step at a time; reset before replay")
        dt = observation.get("dt")
        if isinstance(dt, bool) or not isinstance(dt, (float, int)) or not math.isclose(dt, .02, abs_tol=1e-9):
            raise ValueError("policy observation dt must be 0.02 seconds")
        now = observation.get("time")
        if isinstance(now, bool) or not isinstance(now, (float, int)) or not math.isclose(now, step*.02, abs_tol=1e-5):
            raise ValueError("policy observation time does not match control step")
        terms, root_rotation = self._proprio(observation)
        # Each term is independently chronological; first frame backfills the
        # entire window, matching mjlab CircularBuffer.append.
        history = ([np.repeat(term[None], self.history_length, axis=0) for term in terms]
                   if self._history is None else
                   [np.concatenate((old[1:], term[None]), axis=0)
                    for old, term in zip(self._history, terms)])
        frame = min(step, self.reference_frames - 1)
        indices = np.minimum(frame + np.arange(GOAL_FRAMES)*GOAL_STRIDE, self.reference_frames - 1)
        command = np.concatenate((self.reference["joint_pos"][indices],
                                  self.reference["joint_vel"][indices]), axis=1).reshape(-1)
        orientations = np.concatenate([(root_rotation.T @ _rotation(self.reference["root_quat_w"][i]))[:, :2].reshape(-1)
                                       for i in indices])
        obs = np.concatenate([*(term.reshape(-1) for term in history), command, orientations])
        if obs.shape != (self.obs_size,) or not np.isfinite(obs).all():
            raise ValueError("constructed policy observation is invalid")
        outputs = self._infer(obs.astype(np.float32)[None], frame)
        self._check_reference_outputs(outputs, frame)
        action = _array(outputs["actions"], (1, self.n), "ONNX actions")[0]
        target = action*self.action_scale + self.default_q
        if not np.isfinite(target).all():
            raise ValueError("ONNX action transform produced invalid joint targets")
        self._history = history
        self._last_action = action.copy()
        self._next_step += 1
        self.reference_frame = frame
        self.reference_complete = frame == self.reference_frames - 1
        self.last_observation = obs.copy()
        return target.tolist()
