"""Perception-only v2 adapter; the mirrored strict v1 contract stays unchanged."""
from copy import deepcopy

from .pointcloud_slam_contract import validate_tracking_pose as validate_v1_pose
from .spatial_time import FrameContext


def validate_tracking_pose(sample):
    quality = (sample.get('vendor') or {}).get('motus_slam') if isinstance(sample, dict) else None
    if not isinstance(quality, dict) or quality.get('profile', 'motus_slam/1') == 'motus_slam/1':
        return validate_v1_pose(sample)
    if quality.get('profile') != 'motus_slam/2':
        raise ValueError('unsupported SLAM pose profile')
    context = FrameContext.from_dict(quality.get('observation'))
    expected = 'measurement' if context.time_basis == 'source_capture' else 'receipt_estimated'
    if (quality.get('stamp_source') != expected or quality.get('clock_id') != context.normalized_clock_id or
            quality.get('stamp_ns') != context.observation_stamp_ns or
            sample.get('stamp_ms') != context.observation_stamp_ns // 1_000_000):
        raise ValueError('pose and observation time declarations conflict')
    # Reuse v1 geometry/unit validation on a private copy, after validating v2
    # time above. The wire object remains v2; strict old readers still reject it.
    geometry = deepcopy(sample)
    geometry['vendor']['motus_slam']['stamp_source'] = 'measurement'
    validate_v1_pose(geometry)
    return sample


def input_declaration(topic, kind, frame, context, *, metadata_topic=None):
    value = dict(schema='motus.slam_input/2', topic=topic, kind=kind,
        format='state/odom' if kind=='pose' else 'sensor/pointcloud2', length_unit='m',
        source_id=context.source_id, source_epoch=context.source_epoch,
        clock_id=context.normalized_clock_id, time_basis=context.time_basis,
        ingress_clock_id=context.ingress_clock_id,
        stamp_source='measurement' if context.time_basis=='source_capture' else 'receipt_estimated',
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
    context=FrameContext.from_dict(packet.get('observation'))
    digest=packet.get('data_sha256')
    if (packet.get('stamp_ns')!=context.observation_stamp_ns or
            packet.get('clock_id')!=context.normalized_clock_id or
            not isinstance(packet.get('frame_id'),str) or not packet['frame_id'] or
            not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('spatial frame identity/digest mismatch')
    return context


def resolve_v2_inputs(metadata, bindings):
    """Explicit observation-time inputs; never synthesize v1 clock evidence."""
    import numpy as np
    from .pointcloud_slam_transforms import rigid
    inputs=deepcopy({t:metadata.get(t) for t in bindings})
    for topic,fmt in bindings.items():
        d=inputs[topic]
        if (not isinstance(d,dict) or d.get('schema')!='motus.slam_input/2' or
                d.get('topic')!=topic or d.get('format')!=fmt or d.get('length_unit')!='m' or
                d.get('observation_schema')!='motus.observation/1' or
                d.get('time_basis') not in ('source_capture','receipt_estimated') or
                not isinstance(d.get('clock_id'),str) or not d['clock_id']):
            raise ValueError('invalid versioned spatial input: '+topic)
        expected='measurement' if d['time_basis']=='source_capture' else 'receipt_estimated'
        if d.get('stamp_source')!=expected:raise ValueError('input time basis conflict')
    pose=next(d for d in inputs.values() if d['format']=='state/odom')
    cloud=next(d for d in inputs.values() if d['format']=='sensor/pointcloud2')
    child,frame=pose.get('child_frame_id'),cloud.get('frame_id')
    if (pose.get('kind')!='pose' or cloud.get('kind')!='cloud' or
            any(not isinstance(v,str) or not v for v in (child,frame,cloud.get('frame_metadata_topic'))) or
            cloud['frame_metadata_topic'] in bindings):
        raise ValueError('input kind, frames and sidecar required')
    if (pose['clock_id']!=cloud['clock_id'] or pose['time_basis']!=cloud['time_basis'] or
            pose.get('source_epoch')!=cloud.get('source_epoch')):
        raise ValueError('input clock/time basis/epoch conflict')
    if (pose.get('ingress_clock_id')!=cloud.get('ingress_clock_id') or not pose.get('ingress_clock_id')):
        raise ValueError('common ingress clock evidence required')
    if cloud.get('observation_model') not in ('pinhole','lidar') or cloud.get('timing_model') not in ('instantaneous','deskewed'):
        raise ValueError('unsupported cloud observation model')
    mount=pose.get('extrinsic') or cloud.get('extrinsic')
    if mount is None:
        if child!=frame:raise ValueError('actual child_from_cloud transform required')
        mount=dict(parent_frame=child,child_frame=frame,matrix=np.eye(4).tolist(),
                   mode='static',source='identical_frame',version='identity/1')
    from .pointcloud_slam_contract import validate_mount
    validate_mount(mount,parent_frame=child,child_frame=frame)
    if pose.get('extrinsic') and cloud.get('extrinsic') and pose['extrinsic']!=cloud['extrinsic']:
        raise ValueError('conflicting producer transforms')
    mode=mount.get('mode','static')
    matrix=None if mode=='dynamic' else rigid(mount.get('matrix')).tolist()
    return dict(schema='motus.slam_input/2',inputs=inputs,frame_metadata_topic=cloud['frame_metadata_topic'],
                extrinsic=deepcopy(mount),time_basis=cloud['time_basis'],
                ingress_clock_id=cloud['ingress_clock_id'],
                settings=dict(cloud_frame=frame,child_frame=child,clock_id=cloud['clock_id'],
                              child_from_cloud=matrix,extrinsic_mode=mode))


def normalize_pose_time(sample, context):
    """Make missing pose timing explicit, retaining every original pose value."""
    quality=(sample.get('vendor') or {}).get('motus_slam') if isinstance(sample,dict) else None
    if (context is None or context.time_basis!='receipt_estimated' or not isinstance(quality,dict) or
            quality.get('profile')=='motus_slam/2' or quality.get('stamp_source') not in (None,'unknown')):
        return sample
    result=deepcopy(sample)
    result['stamp_ms']=context.observation_stamp_ns//1_000_000
    result['vendor']['motus_slam'].update(profile='motus_slam/2',stamp_source='receipt_estimated',
        stamp_ns=context.observation_stamp_ns,clock_id=context.normalized_clock_id,observation=context.to_dict())
    if quality.get('status')=='tracking':validate_tracking_pose(result)
    return result
