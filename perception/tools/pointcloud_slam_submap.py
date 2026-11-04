"""Build an offline local submap from validated poses and explicit extrinsics."""
import numpy as np


def _rigid(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('invalid transform')
    rot = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(rot), 1., atol=1e-6)):
        raise ValueError('transform must be rigid')
    return matrix


def _pose_matrix(pose):
    x, y, z, w = pose.quaternion
    if not np.isfinite([x, y, z, w]).all() or abs(x*x + y*y + z*z + w*w - 1.) > .002:
        raise ValueError('invalid pose quaternion')
    length = np.linalg.norm([x, y, z, w])
    x, y, z, w = np.asarray([x, y, z, w]) / length
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = pose.position
    return _rigid(result)


def compose_submap(clouds, poses, child_from_cloud, *, anchor):
    """Express optical clouds in anchor optical frame; never cross segments.

    The caller must verify calibration and clock provenance. This function does
    not infer or validate a physical extrinsic from coordinate labels.
    """
    if len(clouds) != len(poses) or not 0 <= anchor < len(clouds):
        raise ValueError('each cloud needs one pose and a valid anchor')
    reference = poses[anchor]
    identity = (reference.frame_id, reference.child_frame_id, reference.segment_id, reference.clock_id)
    if any((p.frame_id, p.child_frame_id, p.segment_id, p.clock_id) != identity for p in poses):
        raise ValueError('submap cannot cross frame, clock or segment boundaries')
    extrinsic = _rigid(child_from_cloud)
    world_from_anchor = _pose_matrix(reference) @ extrinsic
    anchor_from_world = np.linalg.inv(world_from_anchor)
    output = []
    for cloud, pose in zip(clouds, poses):
        points = np.asarray(cloud, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('cloud must contain finite Nx3 points')
        transform = anchor_from_world @ _pose_matrix(pose) @ extrinsic
        output.append(points @ transform[:3, :3].T + transform[:3, 3])
    return np.concatenate(output, axis=0)
