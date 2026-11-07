"""Resolved sensor facts and versioned scoring contracts, without ROS dependencies."""
from copy import deepcopy
import math

import numpy as np

from .pointcloud_slam_stream import _rigid


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def resolve_observation(cloud_metadata, camera_info, child_from_cloud):
    """Resolve declared facts only; the transform maps cloud into pose-child coordinates."""
    if not isinstance(cloud_metadata, dict):
        raise ValueError('cloud observation metadata required')
    kind = cloud_metadata.get('observation_model')
    if kind not in ('pinhole', 'lidar'):
        raise ValueError('supported observation_model required')
    for name in ('source_id', 'frame_id'):
        if not isinstance(cloud_metadata.get(name), str) or not cloud_metadata[name].strip():
            raise ValueError('observation ' + name + ' required')
    dynamic = (cloud_metadata.get('extrinsic') or {}).get('mode') == 'dynamic'
    transform = None if dynamic else np.linalg.inv(_rigid(child_from_cloud)).tolist()
    if kind == 'pinhole':
        c = camera_info if isinstance(camera_info, dict) else {}
        k = c.get('K')
        size = c.get('width'), c.get('height')
        if (c.get('schema') != 'motus.camera/1' or not c.get('id') or
                c.get('distortion_model') != 'pinhole' or
                any(type(v) is not int or v <= 0 for v in size) or
                not isinstance(k, (list, tuple)) or len(k) != 9 or
                not all(_finite(v) for v in k) or min(k[0], k[4]) <= 0 or
                any(k[i] != 0 for i in (1, 3, 6, 7)) or k[8] != 1):
            raise ValueError('supported calibrated pinhole intrinsics required')
        d = c.get('D')
        if not isinstance(d, (list, tuple)) or any(not _finite(v) or abs(v) > 1e-9 for v in d):
            raise ValueError('rectified pinhole distortion coefficients required')
        roi = c.get('roi') or {}
        if (not isinstance(roi, dict) or any(roi.get(k, 0) for k in ('x_offset', 'y_offset')) or
                roi.get('width', 0) not in (0, size[0]) or roi.get('height', 0) not in (0, size[1]) or
                any(c.get(k, 0) not in (0, 1) for k in ('binning_x', 'binning_y'))):
            raise ValueError('unresolved ROI or binning is unsupported')
        parameters = dict(width=size[0], height=size[1], K=list(k), D=list(d),
                          distortion_model='pinhole')
        source = dict(camera=deepcopy(c))
    else:
        p = cloud_metadata.get('observation')
        if not isinstance(p, dict) or p.get('axis_convention') != 'x_forward_y_left_z_up':
            raise ValueError('declared lidar axes and angular/range facts required')
        names = ('azimuth_min_rad', 'azimuth_span_rad', 'elevation_min_rad',
                 'elevation_max_rad', 'min_range_m', 'max_range_m')
        if (any(not _finite(p.get(k)) for k in names) or
                not 0 < p['azimuth_span_rad'] <= 2*math.pi or
                not -math.pi/2 <= p['elevation_min_rad'] < p['elevation_max_rad'] <= math.pi/2 or
                not 0 <= p['min_range_m'] < p['max_range_m']):
            raise ValueError('invalid or missing lidar angular/range facts')
        blind = p.get('blind_azimuth_intervals_rad', [])
        if (not isinstance(blind, (list, tuple)) or any(
                not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                not all(_finite(v) for v in pair) or not 0 < abs(pair[1]-pair[0]) < 2*math.pi
                for pair in blind)):
            raise ValueError('invalid lidar blind intervals')
        parameters = {k: p[k] for k in ('axis_convention', *names)}
        parameters['blind_azimuth_intervals_rad'] = deepcopy(blind)
        source = dict(observation=deepcopy(p))
    return dict(schema='motus.slam_observation/1', kind=kind,
                sensor_id=cloud_metadata['source_id'], frame_id=cloud_metadata['frame_id'],
                sensor_from_child=transform, transform_mode='per_capture' if dynamic else 'static',
                parameters=parameters, source=source)


def observation_for_frame(observation, frame):
    """Use the producer's transform at this capture; never fill dynamic gaps with identity."""
    result = deepcopy(observation)
    context = getattr(frame, 'capture_context', None)
    if isinstance(context, dict) and 'child_from_cloud' in context:
        result['sensor_from_child'] = np.linalg.inv(_rigid(context['child_from_cloud'])).tolist()
    elif result.get('transform_mode') == 'per_capture':
        raise ValueError('per-capture observation requires capture transform evidence')
    return result


def resolve_scoring(scoring, observation=None):
    """Validate a new score contract without interpreting a legacy numeric threshold."""
    s = scoring if isinstance(scoring, dict) else {}
    edges, required = s.get('band_edges_m'), s.get('required_bands')
    if (s.get('schema') != 'motus.slam_score/1' or s.get('metric_id') != 'range_stratified_rmse' or
            type(s.get('version')) is not int or s['version'] != 1 or
            s.get('support_set') != 'target_visible_source' or
            s.get('range_definition') not in ('sensor_optical_z', 'sensor_radial') or
            s.get('missing_match_policy') != 'clipped_max_distance' or
            not isinstance(edges, (list, tuple)) or
            any(not _finite(v) or v <= 0 for v in edges) or
            any(a >= b for a, b in zip(edges, edges[1:])) or
            not isinstance(required, (list, tuple)) or not required or
            any(type(v) is not int or not 0 <= v <= len(edges) for v in required) or
            len(set(required)) != len(required)):
        raise ValueError('invalid versioned range score contract')
    visibility = s.get('visibility')
    if not isinstance(visibility, dict):
        raise ValueError('explicit visibility resolution required')
    radial = s['range_definition'] == 'sensor_radial'
    if radial:
        if any(not _finite(visibility.get(k)) or not 0 < visibility[k] <= math.pi
               for k in ('azimuth_bin_rad', 'elevation_bin_rad')):
            raise ValueError('positive angular visibility bins required')
    elif type(visibility.get('pixel_bin')) is not int or visibility['pixel_bin'] <= 0:
        raise ValueError('positive pixel visibility bin required')
    if observation is not None:
        expected = 'lidar' if radial else 'pinhole'
        if observation.get('schema') != 'motus.slam_observation/1' or observation.get('kind') != expected:
            raise ValueError('observation and score range definition differ')
    return deepcopy(s)


def legacy_scoring(visible=False, *, rmse=False):
    return dict(schema='motus.slam_score/1', version=1,
                metric_id=('legacy_inlier_rmse' if rmse else
                           'legacy_visible_optical_depth' if visible else 'legacy_child_x_depth'),
                support_set='inlier_pairs' if rmse else 'target_visible_source' if visible else 'all_source',
                range_definition='none' if rmse else 'sensor_optical_z' if visible else 'legacy_child_x',
                band_edges_m=[] if rmse else [2., 4.], required_bands=[] if rmse else [0, 1, 2],
                missing_match_policy='inliers_only' if rmse else 'clipped_max_distance')


def resolve_revisit(policy=None, observation=None):
    """Keep confirmation policy separate from sensor facts and candidate budgets."""
    if policy is not None and not isinstance(policy, dict):
        raise ValueError('revisit policy must be an object')
    lidar = observation is not None and observation.get('kind') == 'lidar'
    result = dict(schema='motus.slam_revisit/1', enabled=True,
                  return_distance_m=1.5, departure_distance_m=1.5, reference_radius_m=2.,
                  reference_window_frames=3, direction_policy='none' if lidar else 'sensor_forward',
                  direction_min_dot=.7, seed_min_coverage=.30, confirmation_min_coverage=.25,
                  min_support_translation_m=.1, confirmation_frames=2)
    if policy:
        if set(policy) - set(result):
            raise ValueError('unsupported revisit policy fields')
        result.update(deepcopy(policy))
    if (result['schema'] != 'motus.slam_revisit/1' or type(result['enabled']) is not bool or
            result['direction_policy'] not in ('none', 'sensor_forward') or
            (lidar and result['direction_policy'] != 'none') or
            not _finite(result['direction_min_dot']) or not -1 <= result['direction_min_dot'] <= 1 or
            any(type(result[k]) is not int or result[k] < 1
                for k in ('reference_window_frames', 'confirmation_frames')) or
            any(not _finite(result[k]) or not 0 <= result[k] <= 1
                for k in ('seed_min_coverage', 'confirmation_min_coverage')) or
            any(not _finite(result[k]) or result[k] <= 0 for k in
                ('return_distance_m', 'departure_distance_m', 'reference_radius_m',
                 'min_support_translation_m'))):
        raise ValueError('invalid sensor revisit policy')
    return result


def make_view(points, observation, voxel_m, scoring):
    if not isinstance(observation, dict) or observation.get('sensor_from_child') is None:
        raise ValueError('resolved observation capture transform required')
    scoring = resolve_scoring(scoring, observation)
    from .pointcloud_slam_visible_overlap import VisibilityView, LidarVisibilityView
    if observation['kind'] == 'lidar':
        return LidarVisibilityView(points, observation, voxel_m, scoring)
    p = observation['parameters']; k = p['K']
    return VisibilityView(points, (p['width'], p['height'], k[0], k[4], k[2], k[5]),
                          np.asarray(observation['sensor_from_child']), voxel_m,
                          scoring=scoring)
