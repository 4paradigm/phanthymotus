"""ROS-free admission of producer-declared spatial inputs and frame evidence."""
from copy import deepcopy
import math
import re


class InputContractError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('; '.join(f"{d['topic']}: {d['field']} {d['reason']}" for d in diagnostics))


def resolve_inputs(metadata, bindings, *, experimental=False):
    """Resolve only the connected pose/cloud pair; never invent missing facts."""
    diagnostics = []
    inputs = {}

    def require(ok, topic, field, reason='is missing or unsupported'):
        if not ok:
            diagnostics.append(dict(topic=topic, field=field, reason=reason))

    if not isinstance(metadata, dict):
        raise InputContractError([dict(topic='', field='input_metadata', reason='must be topic keyed')])
    for topic, fmt in bindings.items():
        d = metadata.get(topic)
        if not isinstance(d, dict):
            require(False, topic, 'input_metadata')
            continue
        inputs[topic] = deepcopy(d)
        require(d.get('schema') == 'motus.slam_input/1', topic, 'schema')
        require(d.get('topic') == topic and d.get('format') == fmt, topic, 'topic/format')
        require(d.get('kind') == ('pose' if fmt == 'state/odom' else 'cloud'), topic, 'kind')
        for key in ('source_id', 'clock_id', 'reference_point'):
            require(isinstance(d.get(key), str) and bool(d[key].strip()) and d[key] != 'unknown', topic, key)
        require(d.get('stamp_source') == 'measurement', topic, 'stamp_source')
        require(d.get('length_unit') == 'm', topic, 'length_unit')
    odom_topic = next(t for t, f in bindings.items() if f == 'state/odom')
    cloud_topic = next(t for t, f in bindings.items() if f == 'sensor/pointcloud2')
    pose, cloud = inputs.get(odom_topic, {}), inputs.get(cloud_topic, {})
    child, frame = pose.get('child_frame_id'), cloud.get('frame_id')
    require(isinstance(child, str) and bool(child.strip()), odom_topic, 'child_frame_id')
    require(isinstance(frame, str) and bool(frame.strip()), cloud_topic, 'frame_id')
    require(pose.get('clock_id') == cloud.get('clock_id'), cloud_topic, 'clock_id', 'differs from pose clock')
    require(cloud.get('cloud_semantics') in ('depth_frame', 'scan'), cloud_topic, 'cloud_semantics')
    require(cloud.get('timing_model') in ('instantaneous', 'deskewed'), cloud_topic, 'timing_model')
    require(cloud.get('observation_model') in ('pinhole', 'lidar'), cloud_topic, 'observation_model')
    sidecar = cloud.get('frame_metadata_topic')
    require(isinstance(sidecar, str) and bool(sidecar.strip()) and sidecar not in bindings,
            cloud_topic, 'frame_metadata_topic')
    clock = cloud.get('clock')
    clock = clock if isinstance(clock, dict) else {}
    require(isinstance(clock.get('mapping'), str) and bool(clock['mapping'].strip()), cloud_topic, 'clock.mapping')
    pose_clock = pose.get('clock') if isinstance(pose.get('clock'), dict) else {}
    independent = pose.get('source_id') != cloud.get('source_id')
    relation_id = clock.get('relation_id') if independent else None
    verified = clock.get('verified') is True and pose_clock.get('verified') is True
    if independent:
        require(verified and isinstance(relation_id, str) and bool(relation_id.strip()) and
                relation_id != 'unknown' and pose_clock.get('relation_id') == relation_id,
                cloud_topic, 'clock.relation_id', 'independent sources require both verified relation proofs')
    cloud_mount, pose_mount = cloud.get('extrinsic'), pose.get('extrinsic')
    extrinsic = pose_mount if isinstance(pose_mount, dict) else cloud_mount
    extrinsic = extrinsic if isinstance(extrinsic, dict) else {}
    if cloud_mount is not None and pose_mount is not None:
        try:
            import numpy as np
            from .pointcloud_slam_transforms import rigid as _rigid
            same = (cloud_mount.get('parent_frame') == pose_mount.get('parent_frame') and
                    cloud_mount.get('child_frame') == pose_mount.get('child_frame') and
                    cloud_mount.get('mode', 'static') == pose_mount.get('mode', 'static') and
                    (cloud_mount == pose_mount if cloud_mount.get('mode') == 'dynamic' else
                     np.allclose(_rigid(cloud_mount.get('matrix')), _rigid(pose_mount.get('matrix')), atol=1e-6)))
        except (AttributeError, TypeError, ValueError):
            same = False
        require(same, cloud_topic, 'extrinsic', 'conflicting producer transforms')
    if extrinsic.get('pose_reference') is not None:
        require(extrinsic['pose_reference'] == pose.get('pose_reference'), odom_topic,
                'extrinsic.pose_reference', 'does not match actual pose reference')
    require(extrinsic.get('parent_frame') == child and extrinsic.get('child_frame') == frame,
            cloud_topic, 'extrinsic', 'must explicitly map cloud frame into pose child frame')
    for key in ('source', 'version'):
        require(isinstance(extrinsic.get(key), str) and bool(extrinsic[key].strip()), cloud_topic, 'extrinsic.' + key)
    mode = extrinsic.get('mode', 'static')
    require(mode in ('static', 'dynamic'), cloud_topic, 'extrinsic.mode')
    matrix = None
    if mode == 'dynamic':
        require(extrinsic.get('source') == 'tf', cloud_topic, 'extrinsic.source', 'dynamic transform must use tf')
        require('matrix' not in extrinsic, cloud_topic, 'extrinsic.matrix', 'dynamic declaration must not contain a static matrix')
    else:
        try:
            from .pointcloud_slam_transforms import rigid as _rigid
            matrix = _rigid(extrinsic.get('matrix')).tolist()
        except (TypeError, ValueError):
            require(False, cloud_topic, 'extrinsic.matrix', 'must be a rigid 4x4 transform')
    if not experimental:
        require(verified, cloud_topic, 'clock', 'must be verified for strict policy')
        require(extrinsic.get('calibrated') is True, cloud_topic, 'extrinsic', 'must be verified by calibration for strict policy')
    if diagnostics:
        raise InputContractError(diagnostics)
    return dict(inputs=inputs, frame_metadata_topic=sidecar, experimental=experimental,
                extrinsic=deepcopy(extrinsic),
                clock_relation_mode='verified_relation' if independent else 'shared_source',
                relation_id=relation_id,
                settings=dict(cloud_frame=frame, child_frame=child, clock_id=cloud['clock_id'],
                              child_from_cloud=matrix, extrinsic_mode=mode, clock_relation_verified=verified,
                              extrinsic_calibrated=extrinsic.get('calibrated') is True))


def frame_identity(packet, *, clock_id, experimental=False, payload=False, relation_id=None):
    """Validate an individual measurement's provenance, returning its epoch key."""
    if not isinstance(packet, dict) or packet.get('schema') != 'motus.slam_frame/1':
        raise ValueError('measurement provenance is required')
    if type(packet.get('stamp_ns')) is not int or packet['stamp_ns'] <= 0:
        raise ValueError('invalid measurement stamp_ns')
    for key in ('frame_id', 'clock_id', 'clock_epoch', 'source_generation', 'source_clock'):
        if not isinstance(packet.get(key), str) or not packet[key].strip() or packet[key] == 'unknown':
            raise ValueError('measurement ' + key + ' is unknown')
    if packet['clock_id'] != clock_id:
        raise ValueError('measurement clock_id mismatch')
    source = packet.get('source_stamp')
    if (isinstance(source, bool) or not isinstance(source, (int, float)) or
            not math.isfinite(source) or source < 0 or packet.get('source_unit') not in ('ms', 'ns')):
        raise ValueError('original measurement timestamp/unit is required')
    mapping = packet.get('mapping')
    if not isinstance(mapping, dict) or any(not isinstance(mapping.get(k), str) or not mapping[k]
                                            for k in ('method', 'version')):
        raise ValueError('measurement clock mapping is required')
    offset = mapping.get('offset_ns')
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
        raise ValueError('measurement mapping offset is required')
    factor = 1_000_000 if packet['source_unit'] == 'ms' else 1
    if int(source * factor) + offset != packet['stamp_ns']:
        raise ValueError('measurement clock mapping does not reproduce stamp_ns')
    error = mapping.get('uncertainty_ns')
    if error is not None and (isinstance(error, bool) or not isinstance(error, (int, float)) or
                              not math.isfinite(error) or error < 0):
        raise ValueError('invalid measurement mapping uncertainty')
    if relation_id is not None and (mapping.get('relation_id') != relation_id or
                                    mapping.get('verified') is not True or error is None):
        raise ValueError('measurement verified clock relation evidence is required')
    if not experimental and (mapping.get('verified') is not True or error is None):
        raise ValueError('measurement mapping is not verified')
    if payload and (not isinstance(packet.get('data_sha256'), str) or
                    not re.fullmatch('[0-9a-f]{64}', packet['data_sha256'])):
        raise ValueError('measurement payload digest is required')
    return packet['clock_id'], packet['clock_epoch'], packet['source_generation']
