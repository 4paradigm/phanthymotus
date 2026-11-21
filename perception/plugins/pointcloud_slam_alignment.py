"""Interpolate supplied pose numbers at cloud timestamps within known coordinate origins."""
import math

from .pointcloud_slam_odom import PoseSample


def _slerp(a, b, ratio):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b = tuple(-v for v in b)
        dot = -dot
    dot = min(1., max(-1., dot))
    if dot > .9995:
        q = tuple(x + ratio * (y - x) for x, y in zip(a, b))
        length = math.sqrt(sum(v * v for v in q))
        return tuple(v / length for v in q)
    theta = math.acos(dot)
    scale = math.sin(theta)
    left = math.sin((1. - ratio) * theta) / scale
    right = math.sin(ratio * theta) / scale
    return tuple(left * x + right * y for x, y in zip(a, b))


def crosses_origin(a, b):
    if a.external_origin is not None or b.external_origin is not None:
        return a.external_origin != b.external_origin
    first, second = a.declared_origin, b.declared_origin
    return first is not None and second is not None and first != second


def align_pose(samples, cloud_stamp_ns, *, max_gap_ms):
    """Interpolate numeric timestamps without crossing declared origins.

    None remains an absence barrier. Within each available run, samples can
    arrive out of order and the last value for a duplicate timestamp is used.
    """
    if type(cloud_stamp_ns) is not int:
        raise ValueError('invalid cloud timestamp')
    runs, run = [], []
    for sample in samples:
        if sample is None:
            runs.append(run)
            run = []
        elif not isinstance(sample, PoseSample):
            raise ValueError('expected validated pose or loss barrier')
        else:
            run.append(sample)
    runs.append(run)
    for run in reversed(runs):
        ordered = sorted({s.stamp_ns: s for s in run}.values(), key=lambda s: s.stamp_ns)
        previous = None
        for sample in ordered:
            stamp_ns = sample.stamp_ns
            if stamp_ns == cloud_stamp_ns:
                return sample
            if stamp_ns > cloud_stamp_ns:
                if previous is None:
                    break
                if crosses_origin(previous, sample):
                    raise ValueError('cannot interpolate across declared odometry origins')
                prediction = previous.prediction_anchor_ns or sample.prediction_anchor_ns
                if prediction is not None and (stamp_ns-previous.stamp_ns > max_gap_ms*1_000_000 or
                        cloud_stamp_ns <= prediction):
                    raise ValueError('IMU prediction cannot bridge unbounded capture gaps')
                ratio = (cloud_stamp_ns - previous.stamp_ns) / (stamp_ns - previous.stamp_ns)
                position = tuple((1-ratio)*a + ratio*b for a, b in zip(previous.position, sample.position))
                if not all(math.isfinite(v) for v in position):
                    raise ValueError('interpolated pose is not finite')
                continuity = (sample.external_continuity
                              if previous.external_continuity == sample.external_continuity else None)
                return PoseSample(cloud_stamp_ns // 1_000_000, sample.frame_id, sample.child_frame_id,
                                  sample.segment_id, position,
                                  _slerp(previous.quaternion, sample.quaternion, ratio), None, sample.clock_id,
                                  sample.pose_reference, None, cloud_stamp_ns,
                                  tuple(c for c in (previous.observation,sample.observation) if c is not None),
                                  prediction, continuity,
                                  sample.chassis_origin if sample.chassis_origin == previous.chassis_origin else None,
                                  sample.visual_origin if sample.visual_origin == previous.visual_origin else None,
                                  sample.external_origin if sample.external_origin == previous.external_origin else None)
            previous = sample
    raise ValueError('no numeric pose at cloud timestamp')
