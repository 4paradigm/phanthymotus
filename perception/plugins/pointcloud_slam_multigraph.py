"""Bounded provisional multi-segment pose graph; constraints are not verified loops."""
import math
import numpy as np

from .pointcloud_slam_experimental_backend import _sigmas
from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_map import map_from_odom, rebuild_segment
from .pointcloud_slam_stream import _rigid


def optimize_multisegment(segments, graphs, links, *, odom_sigmas, loop_sigmas,
                          huber_k, max_total_frames, max_total_points, segment_loops=None):
    """Optimize connected segments using target-to-source between constraints."""
    import gtsam
    if (not segments or len(segments) != len(graphs) or
            any(not isinstance(s, Segment) or not s.frames for s in segments) or
            len({s.segment_id for s in segments}) != len(segments) or
            any(type(x) is not int or x <= 0 for x in (max_total_frames, max_total_points)) or
            sum(len(s.frames) for s in segments) > max_total_frames or
            sum(len(f.points) for s in segments for f in s.frames) > max_total_points or
            isinstance(huber_k, bool) or not isinstance(huber_k, (int, float)) or
            not math.isfinite(huber_k) or huber_k <= 0):
        raise ValueError('invalid or oversized segments')
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas(odom_sigmas))
    loop_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
                    gtsam.noiseModel.Diagonal.Sigmas(_sigmas(loop_sigmas)))
    by_id, offsets = {}, []
    offset = 0
    for segment, graph in zip(segments, graphs):
        if (not isinstance(graph, dict) or graph.get('segment_id') != segment.segment_id or
                len(graph.get('poses', ())) != len(segment.frames) or
                not isinstance(graph.get('edges'), list) or len(graph['edges']) != len(segment.frames)-1):
            raise ValueError('matching odometry graphs required')
        for i, edge in enumerate(graph['edges'], 1):
            if (edge.get('from'), edge.get('to')) != (i-1, i):
                raise ValueError('consecutive odometry required')
            _rigid(edge.get('transform'))
        for pose in graph['poses']:
            _rigid(pose)
        by_id[segment.segment_id] = len(offsets)
        offsets.append(offset)
        offset += len(segment.frames)
    align = {segments[0].segment_id: np.eye(4)}
    validated = []
    for link in links:
        try:
            target, source = link['target'], link['source']
            a, b = by_id[target], by_id[source]
            i, j = link['target_index'], link['source_index']
            if (target not in align or source in align or type(i) is not int or type(j) is not int or
                    not 0 <= i < len(segments[a].frames) or not 0 <= j < len(segments[b].frames)):
                raise ValueError('links must connect a new segment to the anchored graph')
            transform = _rigid(link['transform'])
            align[source] = align[target] @ _rigid(graphs[a]['poses'][i]) @ transform @ np.linalg.inv(_rigid(graphs[b]['poses'][j]))
            validated.append((offsets[a]+i, offsets[b]+j, transform))
        except (KeyError, TypeError) as error:
            raise ValueError('invalid segment link') from error
    if len(align) != len(segments):
        raise ValueError('disconnected segments')
    factors, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for segment, graph, base in zip(segments, graphs, offsets):
        for i, pose in enumerate(graph['poses']):
            values.insert(base+i, gtsam.Pose3(align[segment.segment_id] @ _rigid(pose)))
        for edge in graph['edges']:
            factors.add(gtsam.BetweenFactorPose3(base+edge['from'], base+edge['to'],
                                                gtsam.Pose3(_rigid(edge['transform'])), odom_noise))
    factors.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(_rigid(graphs[0]['poses'][0])), odom_noise))
    for segment_id, loops in (segment_loops or {}).items():
        if segment_id not in by_id:
            raise ValueError('loop refers to an unknown segment')
        index = by_id[segment_id]
        for loop in loops:
            source, target = loop['source'], loop['target']
            if (type(source) is not int or type(target) is not int or source == target or
                    not 0 <= source < len(segments[index].frames) or
                    not 0 <= target < len(segments[index].frames)):
                raise ValueError('invalid loop endpoints')
            factors.add(gtsam.BetweenFactorPose3(offsets[index]+source, offsets[index]+target,
                        gtsam.Pose3(np.linalg.inv(_rigid(loop['transform']))), loop_noise))
    for target, source, transform in validated:
        factors.add(gtsam.BetweenFactorPose3(target, source, gtsam.Pose3(transform), loop_noise))
    optimized = gtsam.LevenbergMarquardtOptimizer(factors, values).optimize()
    poses = [[optimized.atPose3(base+i).matrix() for i in range(len(segment.frames))]
             for segment, base in zip(segments, offsets)]
    return {'segment_ids': [s.segment_id for s in segments], 'poses': poses,
            'map_points': np.concatenate([rebuild_segment(s, p) for s, p in zip(segments, poses)]),
            'map_from_active_odom': map_from_odom(poses[-1][-1], graphs[-1]['poses'][-1])}
