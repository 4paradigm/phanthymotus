"""Perception-only v2 adapter; the mirrored strict v1 contract stays unchanged."""
from copy import deepcopy

from .pointcloud_slam_contract import validate_tracking_pose as validate_v1_pose
from .spatial_time import FrameContext


def validate_tracking_pose(sample):
    return validate_v1_pose(sample)


def input_declaration(topic, kind, frame, context, *, metadata_topic=None):
    value = dict(schema='motus.slam_input/2', topic=topic, kind=kind,
        format='state/odom' if kind=='pose' else 'sensor/pointcloud2', length_unit='m',
        source_id=context.source_id, source_epoch=context.source_epoch, geometry_version=context.geometry_version,
        mapping_version=context.mapping_version, mapping_source=context.mapping_source,
        clock_id=context.normalized_clock_id, time_basis=context.time_basis,
        ingress_clock_id=context.ingress_clock_id,
        stamp_source='measurement' if context.time_basis=='source_capture' else context.time_basis,
        observation_schema='motus.observation/1', clock_accuracy_verified=context.mapping_verified)
    if kind=='pose':value.update(child_frame_id=frame)
    else:value.update(frame_id=frame, cloud_semantics='depth_frame', observation_model='pinhole',
                      timing_model='instantaneous',frame_metadata_topic=metadata_topic)
    return value


def frame_packet(frame, context, data):
    import hashlib
    return dict(schema='motus.slam_frame/2',frame_id=frame,stamp_ns=context.observation_stamp_ns,
                clock_id=context.normalized_clock_id,observation=context.to_dict(),
                data_sha256=hashlib.sha256(data).hexdigest())


def validate_frame_packet(packet):
    if not isinstance(packet,dict) or packet.get('schema')!='motus.slam_frame/2':
        raise ValueError('unsupported spatial frame version')
    return FrameContext.from_dict(packet.get('observation'))


def resolve_v2_inputs(metadata, bindings):
    """Resolve numeric input roles; source declarations are descriptive."""
    import numpy as np
    from .pointcloud_slam_transforms import rigid
    if not isinstance(metadata,dict):raise ValueError('input metadata must be topic keyed')
    inputs=deepcopy({t:metadata.get(t) for t in bindings})
    for topic,fmt in bindings.items():
        value=inputs[topic]
        if not isinstance(value,dict) or value.get('format')!=fmt:
            raise ValueError('invalid spatial input format: '+topic)
    pose=next(d for d in inputs.values() if d['format']=='state/odom')
    cloud=next(d for d in inputs.values() if d['format']=='sensor/pointcloud2')
    frame=cloud.get('frame_id') or pose.get('child_frame_id') or 'input'
    child=pose.get('child_frame_id') or frame
    mount=pose.get('extrinsic') or cloud.get('extrinsic')
    if mount is None:
        mount=dict(parent_frame=child,child_frame=frame,matrix=np.eye(4).tolist(),
                   mode='static',source='input_coordinates',version='identity/1')
    from .pointcloud_slam_contract import validate_mount
    validate_mount(mount,parent_frame=child,child_frame=frame)
    mode='dynamic' if mount.get('mode')=='dynamic' else 'static'
    matrix=None if mode=='dynamic' else rigid(mount.get('matrix')).tolist()
    basis=cloud.get('time_basis','source_capture')
    return dict(schema='motus.slam_input/2',inputs=inputs,frame_metadata_topic=cloud.get('frame_metadata_topic'),
                extrinsic=deepcopy(mount),time_basis=basis,association_method='numeric_time',
                ingress_clock_id=cloud.get('ingress_clock_id'),
                settings=dict(cloud_frame=frame,child_frame=child,
                    clock_id=cloud.get('clock_id') or pose.get('clock_id') or 'input',
                    child_from_cloud=matrix,extrinsic_mode=mode))


def normalize_pose_time(sample, context):
    """Make missing pose timing explicit, retaining every original pose value."""
    quality=(sample.get('vendor') or {}).get('motus_slam') if isinstance(sample,dict) else None
    if (context is None or type(sample.get('stamp_ms')) is int or
            (isinstance(quality,dict) and type(quality.get('stamp_ns')) is int) or
            not isinstance(quality,dict)):
        return sample
    result=deepcopy(sample)
    result['stamp_ms']=context.observation_stamp_ns//1_000_000
    result['vendor']['motus_slam'].update(profile='motus_slam/2',stamp_source='receipt_estimated',
        stamp_ns=context.observation_stamp_ns,clock_id=context.normalized_clock_id,observation=context.to_dict())
    if quality.get('status')=='tracking':validate_tracking_pose(result)
    return result


def validate_observation_declaration(context, declaration):
    return None  # Labels do not alter the numeric observation.
