"""Real numerical camera/lidar geometry regressions (never substitute ROS stubs)."""
import numpy as np
import pytest

pytest.importorskip('open3d')
from plugins.pointcloud_slam_observation import make_view
from test_pointcloud_slam_observation import camera, pinhole, lidar, scoring


def test_explicit_camera_support_denominators_and_missing_far_band():
    p = np.array([[0., 0., 1.], [.4, .2, 3.], [-.6, .4, 5.]])
    spec = pinhole(); s = scoring()
    target = make_view(p, spec, .01, s)
    source = make_view(np.vstack((p, 2*p)), spec, .01, s)
    result = source.score(target, np.eye(4), .3)
    assert result['coverage_numerator'] == 3 and result['coverage_denominator'] == 6
    assert result['fitness_numerator'] == 3 and result['fitness_denominator'] == 3
    assert result['fitness'] == 1 and result['range_stratified_rmse_m'] == pytest.approx(0.)
    assert result['score_metadata'] == s
    sparse = make_view(p[:2], spec, .01, s).score(target, np.eye(4), .3)
    assert sparse['range_stratified_rmse_m'] is None
    assert sparse['reason'] == 'missing_range_support'


def test_camera_sensor_origin_fov_and_empty_cloud_are_not_fallbacks():
    from plugins.pointcloud_slam_observation import resolve_observation
    tr = np.array([[0, 0, 1, 2], [-1, 0, 0, 3], [0, -1, 0, 0], [0, 0, 0, 1.]])
    spec = resolve_observation(dict(observation_model='pinhole', source_id='lens', frame_id='optical'),
                               camera(), tr)
    optical = np.array([[0., 0, 1.], [.2, 0, 3.], [-.3, 0, 5.]])
    child = optical @ tr[:3, :3].T + tr[:3, 3]
    view = make_view(child, spec, .01, scoring())
    np.testing.assert_allclose(np.sort(view.range_values(child)), [1., 3., 5.])
    assert not view.visible(np.array([[2., 30, 0]]), .3)[0]
    empty = make_view(np.empty((0, 3)), spec, .01, scoring())
    result = empty.score(view, np.eye(4), .3)
    assert result['coverage'] == 0 and result['fitness'] == 0 and result['error_m'] is None


def test_lidar_wraparound_blind_sector_partial_fov_and_range_boundaries():
    # Rays on both sides of the angular seam belong to the same full-circle scan.
    angles = np.array([-np.pi+1e-5, np.pi-1e-5, 0., 1.])
    points = np.column_stack((3*np.cos(angles), 3*np.sin(angles), np.zeros(4)))
    spec = lidar(blind_azimuth_intervals_rad=[[-.1, .1]])
    view = make_view(points, spec, .01, scoring(True))
    assert view.visible(points, .3).tolist() == [True, True, False, True]
    partial = make_view(points, lidar(azimuth_min_rad=0., azimuth_span_rad=np.pi/2), .01, scoring(True))
    assert partial.visible(points, .3).tolist() == [False, False, True, True]
    assert not view.visible(np.array([[.01, 0, 0], [30., 0, 0], [3., 0, 4.]]), .3).any()


def test_radial_bands_are_rotation_invariant_and_occluded_returns_are_excluded():
    points = np.array([[1., 0, 0], [0., 3, 0], [-5., 0, 0]])
    s = scoring(True); spec = lidar()
    target = make_view(points, spec, .01, s)
    source = make_view(np.vstack((points, 2*points)), spec, .01, s)
    result = source.score(target, np.eye(4), .3)
    assert result['counts'] == [1, 1, 1] and result['coverage'] == .5
    assert result['range_stratified_rmse_m'] == 0.
    rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
    rotated = make_view(points @ rotation[:3, :3].T, spec, .01, s)
    result = target.score(rotated, rotation, .3)
    assert result['counts'] == [1, 1, 1] and result['fitness'] == 1.


def test_reference_windows_keep_only_valid_observations_including_empty_views():
    from plugins.pointcloud_slam_visible_overlap import ReferenceWindow
    p = np.array([[0., 0., 1.], [.4, .2, 3.], [-.6, .4, 5.], [1000., 0, 1.]])
    views = [make_view(p, pinhole(), .01, scoring()) for _ in range(2)]
    window = ReferenceWindow([p, p], views, [np.eye(4), np.eye(4)], .01)
    assert len(window.points) == 3
    empty = np.array([[1000., 0, 1.]])
    views = [make_view(empty, pinhole(), .01, scoring()) for _ in range(2)]
    window = ReferenceWindow([empty, empty], views, [np.eye(4), np.eye(4)], .01)
    result = window.score(views[0], np.eye(4), .3)
    assert result['coverage'] == 0 and result['reason'] == 'missing_range_support'


def test_registration_and_fixed_overlap_keep_the_explicit_score_contract():
    from plugins.pointcloud_slam_registration import registration_diagnostics, fixed_overlap
    p = np.array([[1., 0, 0], [0., 3, 0], [-5., 0, 0], [0., -5, 0]])
    s = scoring(True); spec = lidar()
    gates = dict(min_fitness=.5, min_correspondences=1,
                 max_cycle_translation_m=.1, max_cycle_angle_deg=2.,
                 max_range_stratified_rmse_m=.09, score_contract=s)
    d = registration_diagnostics(p, p, np.eye(4), voxel_m=.01, max_distance_m=.3,
                                 observation=spec, scoring=s)
    assert d['forward']['score_metadata'] == s
    assert d['forward']['range_stratified_rmse_m'] == pytest.approx(0., abs=1e-12)
    assert 'forward_depth_score_m' not in d['forward']
    scores = fixed_overlap(p, p, np.eye(4), dict(voxel_m=.01, max_distance_m=.3,
                           thresholds=gates, observation=spec, scoring=s))
    assert all(value['passes'] for value in scores)
    bad = registration_diagnostics(p, p+[0, 8, 0], np.eye(4), voxel_m=.01,
                                   max_distance_m=.3, observation=spec, scoring=s)
    from plugins.pointcloud_slam_geometry import classify_registration
    assert not classify_registration(bad, gates)['passes']


def test_full_circle_revisit_accepts_reverse_heading_and_parameterized_holdouts():
    from types import SimpleNamespace
    from plugins.pointcloud_slam_revisit import RevisitSearch
    rng = np.random.default_rng(273)
    az = rng.uniform(-np.pi, np.pi, 2400)
    elev = rng.uniform(-.25, .25, len(az))
    radius = np.repeat([1., 3., 5.], 800)
    world = np.column_stack((radius*np.cos(elev)*np.cos(az),
                             radius*np.cos(elev)*np.sin(az), radius*np.sin(elev)))
    xs = [0., .12, .24, 2., 0., 0., .12]
    poses, frames = [], []
    for i, x in enumerate(xs):
        pose = np.eye(4)
        if i >= 4:
            pose[:3, :3] = np.diag([-1., -1., 1.])
        raw = (world-[x, 0, 0]) @ pose[:3, :3]
        pose[0, 3] = x + (.02 if i >= 4 else 0)
        poses.append(pose)
        frames.append(SimpleNamespace(points=raw, cloud_stamp_ns=i*5_000_000_000))
    s = scoring(True)
    opts = dict(voxel_m=.04, max_distance_m=.3, min_separation_s=10,
                thresholds=dict(max_range_stratified_rmse_m=.09, score_contract=s,
                    min_fitness=.6, min_correspondences=5, max_cycle_translation_m=.03,
                    max_cycle_angle_deg=2))
    search = RevisitSearch(None, None, opts, observation=lidar(), scoring=s,
                           policy=dict(confirmation_frames=1, reference_window_frames=2))
    assert search.update(frames[:5], poses[:5], [0]) is None
    assert search.pending is not None
    assert search.update(frames[:6], poses[:6], []) is None
    result = search.update(frames, poses, [])
    assert result is not None and result['confirmation_stamps_ns'] == [30_000_000_000]
    assert result['diagnostic']['score_metadata'] == s
    assert search.confirmation_evaluations == 2


def test_explicit_legacy_visible_profile_retains_legacy_denominators():
    from plugins.pointcloud_slam_loop_search import ExperimentalLoopSearch
    from test_pointcloud_slam_strategy import options
    opts = options(); opts['thresholds'].pop('max_rmse_m')
    opts['thresholds']['max_forward_depth_score_m'] = .09
    worker = ExperimentalLoopSearch(**opts, observation=pinhole(), legacy_profile='visible_optical_depth')
    p = np.array([[0., 0., 1.], [.4, .2, 3.], [-.6, .4, 5.]])
    target = worker.revisit.view(type('Frame', (), dict(points=p)))
    source = worker.revisit.view(type('Frame', (), dict(points=np.vstack((p, 2*p)))))
    result = source.score(target, np.eye(4), .3)
    assert result['fitness'] == .5 and result['coverage'] == .5
    assert result['score_metadata']['metric_id'] == 'legacy_visible_optical_depth'
    assert worker.strategy_snapshot()['resolved_scoring']['metric_id'] == 'legacy_visible_optical_depth'


def test_dynamic_head_models_keep_each_capture_origin_in_both_registration_directions():
    from types import SimpleNamespace
    from plugins.pointcloud_slam_observation import observation_for_frame
    from plugins.pointcloud_slam_registration import registration_diagnostics
    from plugins.pointcloud_slam_revisit import RevisitSearch
    # Keep rays away from 4-pixel boundaries; ICP has machine-epsilon pose error.
    p = np.array([[.01, .01, 1.], [.2, .11, 3.], [-.31, -.11, 5.], [.41, -.31, 5.]])
    a = np.eye(4); a[0, 3] = 2.
    b = np.array([[0, 0, 1, -1], [0, 1, 0, 1], [-1, 0, 0, 0], [0, 0, 0, 1.]])
    frames = [SimpleNamespace(points=p @ t[:3, :3].T+t[:3, 3],
                              capture_context={'child_from_cloud': t.tolist()}) for t in (a, b)]
    s = scoring(); observation = pinhole()
    views = [make_view(f.points, observation_for_frame(observation, f), .01, s) for f in frames]
    for view, frame in zip(views, frames):
        np.testing.assert_allclose(view.range_values(frame.points), [1., 3., 5., 5.])
    transform = b @ np.linalg.inv(a)
    d = registration_diagnostics(frames[0].points, frames[1].points, transform,
        voxel_m=.01, max_distance_m=.3, scoring=s,
        observation=observation_for_frame(observation, frames[0]),
        target_observation=observation_for_frame(observation, frames[1]))
    assert d['forward']['coverage'] == 1. and d['reverse']['coverage'] == 1.
    assert d['forward']['range_stratified_rmse_m'] == pytest.approx(0., abs=1e-10)
    search = RevisitSearch(None, None, dict(voxel_m=.01), observation=observation, scoring=s)
    np.testing.assert_allclose(search.view(frames[1]).range_values(frames[1].points), [1., 3., 5., 5.])


def test_legacy_registration_and_fixed_overlap_also_label_their_original_metric():
    from plugins.pointcloud_slam_registration import registration_diagnostics, fixed_overlap
    from test_pointcloud_slam_strategy import options
    p = np.array([[1., 0., 0.], [3., .1, 0], [5., 0, .1], [6., .1, .1]])
    d = registration_diagnostics(p, p, np.eye(4), voxel_m=.01, max_distance_m=.3)
    assert d['forward']['score_metadata']['metric_id'] == 'legacy_inlier_rmse'
    geometry = options(); geometry['thresholds']['min_correspondences'] = 1
    result = fixed_overlap(p, p, np.eye(4), geometry)
    assert all(s['score_metadata']['metric_id'] == 'legacy_inlier_rmse' for s in result)
    assert all(s['compatibility'] for s in result)
