"""PR152 head_yaw mapper, Apache-2.0.

Copied from 0adb64cb63e358244f4e02c0a7823523824f17a8:
unitree/g1/teleop/adapter.py. See NOTICE.md.
"""
import math
from collections.abc import Mapping
import numpy as np

_OPENXR_TO_ROBOT = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)

_TO_UNITREE_LEFT_ARM = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)

_TO_UNITREE_RIGHT_ARM = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)

class G1ControllerPoseMapper:
    """Map OpenXR head/controller poses into a pelvis-fixed G1 IK frame."""

    def __init__(
        self,
    ):
        # V1 is intentionally fixed to the Apache-2.0 upstream G1 controller
        # calibration.  Any future calibration change requires a new profile.
        self._waist_offset = np.array([0.15, 0.0, 0.45])

    def map_frame(self, frame: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
        head = self._pose_matrix(frame["head"])
        left = self._pose_matrix(frame["left_controller"])
        right = self._pose_matrix(frame["right_controller"])

        head_robot = self._change_basis(head)
        left_robot = self._change_basis(left) @ _TO_UNITREE_LEFT_ARM
        right_robot = self._change_basis(right) @ _TO_UNITREE_RIGHT_ARM
        yaw_inverse = self._head_yaw(head_robot).T
        return (
            self._head_relative(left_robot, head_robot, yaw_inverse),
            self._head_relative(right_robot, head_robot, yaw_inverse),
        )

    def _head_relative(
        self,
        pose: np.ndarray,
        head: np.ndarray,
        yaw_inverse: np.ndarray,
    ) -> np.ndarray:
        # V1 is locked to xr_teleoperate's tested arm_reference_mode=head_yaw:
        # express wrist rotation/translation in headset yaw, ignore pitch and
        # roll, then translate the origin from head to the fixed IK waist.
        result = np.eye(4)
        result[:3, :3] = yaw_inverse @ pose[:3, :3]
        relative = yaw_inverse @ (pose[:3, 3] - head[:3, 3])
        result[:3, 3] = self._waist_offset + relative
        if not np.all(np.isfinite(result)) or np.linalg.norm(relative) > 2.0:
            raise ValueError("controller pose is outside the bounded G1 workspace")
        return result

    @staticmethod
    def _head_yaw(head: np.ndarray) -> np.ndarray:
        x_axis = head[:3, 0].copy()
        x_axis[2] = 0.0
        norm = float(np.linalg.norm(x_axis))
        if not math.isfinite(norm) or norm <= 1e-6:
            return np.eye(3)
        x_axis /= norm
        z_axis = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(z_axis, x_axis)
        y_norm = float(np.linalg.norm(y_axis))
        if not math.isfinite(y_norm) or y_norm <= 1e-6:
            return np.eye(3)
        y_axis /= y_norm
        return np.column_stack((x_axis, y_axis, z_axis))

    @staticmethod
    def _change_basis(pose: np.ndarray) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = _OPENXR_TO_ROBOT @ pose[:3, :3] @ _OPENXR_TO_ROBOT.T
        result[:3, 3] = _OPENXR_TO_ROBOT @ pose[:3, 3]
        return result

    @classmethod
    def _pose_matrix(cls, value: object) -> np.ndarray:
        if not isinstance(value, Mapping):
            raise ValueError("pose must be an object")
        position = np.asarray(value.get("position"), dtype=float)
        quaternion = np.asarray(value.get("orientation"), dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("pose position/quaternion dimensions are invalid")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("pose contains non-finite values")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0.0:
            raise ValueError("pose quaternion is invalid")
        x, y, z, w = quaternion / norm
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )
        result = np.eye(4)
        result[:3, :3] = rotation
        result[:3, 3] = position
        return result


class G1ClutchRelativeMapper(G1ControllerPoseMapper):
    """PR152 basis and wrist transforms, anchored to each clutch's measured palms."""
    def __init__(self):
        super().__init__()
        self._anchor = None

    def reset(self, frame, palms):
        import copy
        anchor = tuple(x.copy() for x in super().map_frame(frame))
        self._head = copy.deepcopy(frame['head'])
        self._anchor = anchor
        self._palms = tuple(x.copy() for x in palms)

    def map_frame(self, frame):
        if self._anchor is None:
            raise ValueError('mapping_baseline_missing')
        # Head motion after acquisition must not move either robot arm.
        current = super().map_frame({**frame, 'head': self._head})
        targets = []
        for pose, anchor, palm in zip(current, self._anchor, self._palms):
            target = palm.copy()
            target[:3,3] += pose[:3,3] - anchor[:3,3]
            target[:3,:3] = pose[:3,:3] @ anchor[:3,:3].T @ palm[:3,:3]
            targets.append(target)
        return tuple(targets)
