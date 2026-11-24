"""ROS-free rigid transforms shared by input admission and saved-map viewing."""
import numpy as np


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError('extrinsic must be a rigid transform')
    return matrix.copy()


def pose_matrix(pose):
    quaternion = np.asarray(pose.quaternion, dtype=float)
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = pose.position
    return rigid(matrix)
