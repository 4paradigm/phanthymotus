"""Validate original bytes before normalizing native XYZ layout."""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import uuid
import numpy as np

def decode_cloud(msg, *, max_points, sample=False):
    """Read finite XYZ float32 points, including padded and organized clouds."""
    try:
        stamp = msg.header.stamp
        sec, nsec = stamp.sec, stamp.nanosec
        frame = msg.header.frame_id
        width, height, step, row = msg.width, msg.height, msg.point_step, msg.row_step
        if (type(sec) is not int or type(nsec) is not int or
                not 0 <= nsec < 1_000_000_000 or not isinstance(frame, str)):
            raise ValueError('cloud needs an integer timestamp and string frame')
        if (any(type(v) is not int or v <= 0 for v in (width, height, step, row)) or
                (not sample and width * height > max_points) or row < width * step or len(msg.data) != row * height or len(msg.data) > 64*1024*1024):
            raise ValueError('invalid cloud dimensions or capacity')
        fields = {}
        for field in msg.fields:
            if field.name in ('x', 'y', 'z'):
                if (field.name in fields or field.datatype != 7 or
                        field.count != 1 or type(field.offset) is not int or
                        field.offset < 0 or field.offset + 4 > step):
                    raise ValueError('unsupported XYZ fields')
                fields[field.name] = field.offset
        offsets = sorted(fields.values())
        if (set(fields) != {'x', 'y', 'z'} or
                any(b - a < 4 for a, b in zip(offsets, offsets[1:]))):
            raise ValueError('missing or overlapping XYZ fields')
        raw = memoryview(msg.data)
        dtype = '>f4' if msg.is_bigendian else '<f4'
        indices = np.linspace(0, width * height - 1, min(max_points, width * height), dtype=int)
        points = np.stack([np.ndarray((height, width), dtype=dtype, buffer=raw,
                                     offset=fields[key], strides=(row, step))[indices // width, indices % width]
                           for key in ('x', 'y', 'z')], axis=1)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError('cloud has no finite points')
        return sec * 1_000_000_000 + nsec, frame, points
    except (AttributeError, TypeError, BufferError) as error:
        raise ValueError('invalid PointCloud2') from error


@dataclass(frozen=True)
class NormalizedCloud:
    message: object
    source_digest: str
    output_digest: str
    source_digest_verified: bool
    context: object


def normalize_cloud(message, source_context, validated_source_digest, *, max_points=1_000_000):
    digest = hashlib.sha256(message.data).hexdigest()
    _, _, points = decode_cloud(message, max_points=max_points)
    out = deepcopy(message)
    xyz = {f.name:f.offset for f in message.fields if f.name in ('x', 'y', 'z')}
    changed = (message.is_bigendian or message.row_step != message.width*message.point_step or
               xyz != dict(x=0,y=4,z=8) or len(points) != message.width*message.height)
    if changed:
        out.fields = [deepcopy(next(f for f in message.fields if f.name == name)) for name in ('x','y','z')]
        for index, field in enumerate(out.fields):field.offset = index*4
        out.width, out.height = len(points), 1
        out.point_step, out.row_step = 12, len(points)*12
        out.is_bigendian, out.is_dense = False, True
        out.data = np.ascontiguousarray(points,dtype='<f4').tobytes()
    context = source_context
    if context is not None and changed:
        context = replace(context, observation_id=uuid.uuid4().hex,
                          parent_observation_ids=(context.observation_id,),
                          transformations=context.transformations+('normalize_xyz',))
    return NormalizedCloud(out, digest, hashlib.sha256(out.data).hexdigest(),
                           validated_source_digest == digest, context)
