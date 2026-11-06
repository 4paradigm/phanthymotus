from types import SimpleNamespace

import numpy as np
import pytest

from plugins.pointcloud_slam_keyframes import Keyframe, Segment
from tools.pointcloud_slam_localization_replay import ArchiveLocalizer


def frame(segment, number, position, points):
    pose = SimpleNamespace(segment_id=segment, position=[position, 0, 0],
                           quaternion=[0, 0, 0, 1])
    return Keyframe(pose, number * 100000000, points)


def setup_scene(duplicate=False):
    o3d = pytest.importorskip('open3d')
    o3d.utility.random.seed(20)
    rng = np.random.default_rng(19)
    points = rng.uniform([0.5, -1, -0.3], [5, 2, 1.2], (1800, 3))
    geometry = dict(descriptor_voxel_m=.1, voxel_m=.04, max_distance_m=.15,
                    thresholds=dict(min_correspondences=100, min_fitness=.85,
                                    max_forward_depth_score_m=.025,
                                    max_cycle_translation_m=.03, max_cycle_angle_deg=2))
    frames = [frame('A', 1, 0, points)]
    poses = [np.eye(4)]
    if duplicate:
        frames.append(frame('A', 2, 4, points))
        poses.append(np.eye(4))
        poses[-1][0, 3] = 4
    return points, ArchiveLocalizer(Segment('A', tuple(frames)), poses, geometry)


def test_only_two_moving_holdouts_publish_and_failed_prediction_withdraws_pose():
    pts, loc = setup_scene()
    def observe(i, x, segment='B', shift=0):
        return loc.update(i, frame(segment, i, x+1, pts-[x, shift, 0]))
    assert observe(1, 0)['map_pose'] is None
    assert observe(2, 0)['support'] == []
    assert observe(3, .2)['state'] == 'confirming'
    row = observe(4, .4)
    assert row['support'] == [[3, 1], [4, 1]]
    np.testing.assert_allclose(np.array(row['map_pose'])[:3, 3], [.4, 0, 0], atol=.005)
    assert observe(5, .45, shift=2)['map_pose'] is None
    assert observe(6, .0, segment='C')['state'] == 'confirming'
    assert loc.lost()['map_pose'] is None
    assert observe(7, .2, segment='C')['state'] == 'lost'
    assert observe(8, 0, segment='D')['state'] == 'confirming'
    with pytest.raises(ValueError, match='timestamps'):
        observe(8, .2, segment='D')


def test_conflicting_places_never_seed_tracking():
    pts, loc = setup_scene(duplicate=True)
    row = loc.update(1, frame('B', 1, 0, pts))
    assert row['reason'] == 'conflicting_reference_poses'
    assert row['map_pose'] is None
