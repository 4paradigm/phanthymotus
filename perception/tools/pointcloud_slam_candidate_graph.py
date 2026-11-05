"""Offline geometry-gated graph experiment, never a production loop decision."""
from plugins.pointcloud_slam_experimental_backend import optimize_candidates


def experiment_snapshot(segment, graph, result):
    """Compare one segment's history; never publish or merge segment frames."""
    from plugins.pointcloud_slam_map import map_from_odom, rebuild_segment
    if graph.get('segment_id') != segment.segment_id or result.get('loop_accepted') is not False:
        raise ValueError('expected same segment and an unaccepted experimental result')
    before, after = graph['poses'], result['poses']
    return {'before_map': rebuild_segment(segment, before),
            'after_map': rebuild_segment(segment, after),
            'map_from_odom': map_from_odom(after[-1], before[-1]),
            'loop_accepted': False}
