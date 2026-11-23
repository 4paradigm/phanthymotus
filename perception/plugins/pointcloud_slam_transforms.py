"""ROS-free numeric transforms shared by input admission and saved-map viewing."""
import math

import numpy as np


def within_measured_motion_bounds(distance, angle, dt):
    """Check motion between two dense odometry measurements in SI units."""
    return distance <= .1 + 1.5*dt and angle <= math.radians(10) + 2.5*dt


def rigid(value):
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError('extrinsic must be a finite numeric 4x4 transform')
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('extrinsic must be a finite numeric 4x4 transform')
    return matrix.copy()


def pose_matrix(pose):
    quaternion = np.asarray(pose.quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.any(quaternion):
        raise ValueError('pose quaternion must be finite nonzero xyzw')
    quaternion = quaternion / np.max(np.abs(quaternion))
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = pose.position
    return rigid(matrix)
