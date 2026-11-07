"""Sensor facts and score contracts must never be guessed from a point cloud."""
from copy import deepcopy

import numpy as np
import pytest


def camera():
    return dict(schema='motus.camera/1', id='lens', width=160, height=120,
                K=[100., 0, 80., 0, 100., 60., 0, 0, 1],
                distortion_model='pinhole', D=[0., 0., 0., 0., 0.], source='calibrated')


def pinhole():
    from plugins.pointcloud_slam_observation import resolve_observation
    return resolve_observation(dict(observation_model='pinhole', source_id='lens',
                                    frame_id='optical'), camera(), np.eye(4))


def lidar(**overrides):
    from plugins.pointcloud_slam_observation import resolve_observation
    facts = dict(axis_convention='x_forward_y_left_z_up', azimuth_min_rad=-np.pi,
                 azimuth_span_rad=2*np.pi, elevation_min_rad=-.5,
                 elevation_max_rad=.5, min_range_m=.1, max_range_m=20.)
    facts.update(overrides)
    return resolve_observation(dict(observation_model='lidar', source_id='laser',
                                    frame_id='laser', observation=facts), {}, np.eye(4))


def scoring(radial=False):
    return dict(schema='motus.slam_score/1', metric_id='range_stratified_rmse',
                version=1, support_set='target_visible_source',
                range_definition='sensor_radial' if radial else 'sensor_optical_z',
                band_edges_m=[2., 4.], required_bands=[0, 1, 2],
                missing_match_policy='clipped_max_distance',
                visibility=(dict(azimuth_bin_rad=.02, elevation_bin_rad=.02)
                            if radial else dict(pixel_bin=4)))


@pytest.mark.parametrize('change', [lambda c: c.update(K=None),
                                  lambda c: c.update(distortion_model='unknown'),
                                  lambda c: c.update(D=[.1, 0, 0, 0]),
                                  lambda c: c.update(roi={'x_offset': 1}),
                                  lambda c: c.update(binning_x=2)])
def test_pinhole_rejects_missing_or_unsupported_calibration(change):
    from plugins.pointcloud_slam_observation import resolve_observation
    c = camera(); change(c)
    with pytest.raises(ValueError):
        resolve_observation(dict(observation_model='pinhole', frame_id='optical', source_id='lens'),
                            c, np.eye(4))


def test_resolved_model_keeps_real_sensor_origin_and_detaches_producer_data():
    from plugins.pointcloud_slam_observation import resolve_observation
    transform = np.eye(4); transform[0, 3] = 2.
    c = camera()
    result = resolve_observation(dict(observation_model='pinhole', frame_id='optical', source_id='lens'),
                                 c, transform)
    assert result['sensor_from_child'][0][3] == -2.
    assert result['kind'] == 'pinhole' and result['parameters']['K'][0] == 100.
    c['K'][0] = 2.
    assert result['parameters']['K'][0] == 100.


@pytest.mark.parametrize('key', ['axis_convention', 'azimuth_min_rad', 'azimuth_span_rad',
                               'elevation_min_rad', 'elevation_max_rad', 'min_range_m', 'max_range_m'])
def test_lidar_never_infers_missing_angular_or_range_facts(key):
    with pytest.raises(ValueError):
        lidar(**{key: None})


def test_score_contract_rejects_axis_mismatch_and_invalid_bands():
    from plugins.pointcloud_slam_observation import resolve_scoring
    assert resolve_scoring(scoring(), pinhole()) == scoring()
    for value in [scoring(True), {**scoring(), 'band_edges_m': [4., 2.]},
                  {**scoring(), 'required_bands': [3]}, {**scoring(), 'version': 2}]:
        with pytest.raises(ValueError):
            resolve_scoring(value, pinhole())


def test_new_threshold_cannot_accept_old_or_different_score_contract():
    from plugins.pointcloud_slam_geometry import classify_registration
    s = scoring()
    gates = dict(min_fitness=.5, min_correspondences=1,
                 max_cycle_translation_m=.1, max_cycle_angle_deg=2.,
                 max_range_stratified_rmse_m=.09, score_contract=s)
    d = dict(cycle_translation_m=0., cycle_angle_deg=0.)
    for direction in ['forward', 'reverse']:
        d[direction] = dict(fitness=1., correspondences=3,
                            range_stratified_rmse_m=0., score_metadata=deepcopy(s))
    assert classify_registration(d, gates)['passes']
    d['reverse']['score_metadata']['range_definition'] = 'sensor_radial'
    assert 'reverse_score_contract' in classify_registration(d, gates)['failed']
    d['reverse'].pop('score_metadata')
    assert 'reverse_score_contract' in classify_registration(d, gates)['failed']
    with pytest.raises(ValueError):
        classify_registration(d, {**gates, 'max_forward_depth_score_m': .09})


def test_revisit_direction_and_support_are_explicit_sensor_policies():
    from plugins.pointcloud_slam_observation import resolve_revisit
    assert resolve_revisit(None, pinhole())['direction_policy'] == 'sensor_forward'
    assert resolve_revisit(None, lidar())['direction_policy'] == 'none'
    policy = resolve_revisit(dict(confirmation_frames=3, reference_window_frames=2,
                                  return_distance_m=.8), pinhole())
    assert policy['confirmation_frames'] == 3 and policy['reference_window_frames'] == 2
    for invalid in [dict(confirmation_frames=0), dict(seed_min_coverage=1.1),
                    dict(min_support_translation_m=0), dict(direction_min_dot=2),
                    dict(reference_window_frames=True)]:
        with pytest.raises(ValueError):
            resolve_revisit(invalid, pinhole())
    with pytest.raises(ValueError):
        resolve_revisit(dict(direction_policy='sensor_forward'), lidar())


def test_legacy_numeric_gates_are_explicitly_diagnosed_and_cannot_take_a_new_metric():
    from plugins.pointcloud_slam_geometry import classify_registration
    gates = dict(min_fitness=.5, min_correspondences=1, max_cycle_translation_m=.1,
                 max_cycle_angle_deg=2., max_forward_depth_score_m=.09)
    d = dict(cycle_translation_m=0., cycle_angle_deg=0.)
    for direction in ['forward', 'reverse']:
        d[direction] = dict(fitness=1., correspondences=3, forward_depth_score_m=0.)
    assert classify_registration(d, gates)['compatibility']
    d['forward']['score_metadata'] = scoring()
    assert 'forward_score_contract' in classify_registration(d, gates)['failed']


def test_capture_transform_overrides_only_this_frames_sensor_origin():
    from types import SimpleNamespace
    from plugins.pointcloud_slam_observation import observation_for_frame
    spec = pinhole(); matrix = np.eye(4); matrix[0, 3] = 2.
    frame = SimpleNamespace(capture_context={'child_from_cloud': matrix.tolist()})
    result = observation_for_frame(spec, frame)
    assert result['sensor_from_child'][0][3] == -2.
    assert spec['sensor_from_child'][0][3] == 0.
    assert result['parameters'] == spec['parameters']
    with pytest.raises(ValueError):
        observation_for_frame(spec, SimpleNamespace(capture_context={'child_from_cloud': None}))


def test_dynamic_model_requires_capture_evidence_without_an_identity_placeholder():
    from types import SimpleNamespace
    from plugins.pointcloud_slam_observation import resolve_observation, observation_for_frame, make_view
    result = resolve_observation(dict(observation_model='pinhole', source_id='lens', frame_id='optical',
                                      extrinsic={'mode': 'dynamic'}), camera(), None)
    assert result['sensor_from_child'] is None and result['transform_mode'] == 'per_capture'
    with pytest.raises(ValueError):
        observation_for_frame(result, SimpleNamespace())
    with pytest.raises(ValueError):
        make_view(np.ones((3, 3)), result, .01, scoring())
