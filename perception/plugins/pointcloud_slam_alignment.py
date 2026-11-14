"""Match a cloud capture time to tracking poses without crossing a reset.

The caller establishes a common clock and supplies a maximum interpolation
interval justified for its sensor and motion. None marks an invalid/lost sample;
it is a barrier, never a pose to interpolate through.
"""
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


def align_pose(samples, cloud_stamp_ns, *, max_gap_ms):
    """Interpolate only between consecutive, compatible tracking samples.

    ``samples`` must be in acquisition order and each pose must have already
    passed the relevant producer's quality and geometry validation. Repeated
    identical poses are allowed; conflicting duplicate timestamps are rejected.
    Timestamp units are integer nanoseconds for clouds and milliseconds for
    odometry. No clock mapping or extrapolation is attempted here.
    """
    if type(cloud_stamp_ns) is not int or cloud_stamp_ns <= 0:
        raise ValueError('invalid cloud timestamp')
    if (isinstance(max_gap_ms, bool) or not isinstance(max_gap_ms, (int, float))
            or not math.isfinite(max_gap_ms) or max_gap_ms <= 0):
        raise ValueError('invalid maximum interpolation gap')
    ordered = []
    last = None
    for sample in samples:
        if sample is None:
            ordered.append(None)
            continue
        if not isinstance(sample, PoseSample):
            raise ValueError('expected validated pose or loss barrier')
        if last is not None and sample.stamp_ms < last.stamp_ms:
            raise ValueError('out-of-order pose')
        if last is not None and sample.stamp_ms == last.stamp_ms:
            if sample != last or (ordered and ordered[-1] is None):
                raise ValueError('ambiguous duplicate pose timestamp')
            continue
        ordered.append(sample)
        last = sample
    previous = None
    for sample in ordered:
        if sample is None:
            previous = None
            continue
        stamp_ns = sample.stamp_ms * 1_000_000
        if stamp_ns == cloud_stamp_ns:
            return sample
        if stamp_ns > cloud_stamp_ns:
            if previous is None:
                break
            left_ns = previous.stamp_ms * 1_000_000
            if (sample.stamp_ms - previous.stamp_ms > max_gap_ms or
                    (sample.segment_id, sample.frame_id, sample.child_frame_id, sample.clock_id) !=
                    (previous.segment_id, previous.frame_id, previous.child_frame_id, previous.clock_id)):
                break
            ratio = (cloud_stamp_ns - left_ns) / (stamp_ns - left_ns)
            position = tuple(a + ratio * (b - a) for a, b in zip(previous.position, sample.position))
            return PoseSample(cloud_stamp_ns // 1_000_000, sample.frame_id, sample.child_frame_id,
                              sample.segment_id, position,
                              _slerp(previous.quaternion, sample.quaternion, ratio), None, sample.clock_id)
        previous = sample
    raise ValueError('no compatible pose at cloud timestamp')
