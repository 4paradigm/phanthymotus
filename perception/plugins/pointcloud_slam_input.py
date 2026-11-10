"""ROS-free admission of producer-declared spatial inputs and frame evidence."""
from copy import deepcopy

from .pointcloud_slam_contract import (
    INPUT_SCHEMA, FRAME_SCHEMA, POSE_FORMAT, CLOUD_FORMAT,
    SpatialSlamError, validate_input_port, validate_mount,
    validate_frame_identity,
)


class InputContractError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('; '.join(f"{d['topic']}: {d['field']} {d['reason']}" for d in diagnostics))


def resolve_inputs(metadata, bindings):
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
        require(d.get('schema') == INPUT_SCHEMA, topic, 'schema')
        require(d.get('topic') == topic and d.get('format') == fmt, topic, 'topic/format')
        require(d.get('kind') == ('pose' if fmt == POSE_FORMAT else 'cloud'), topic, 'kind')
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
    if diagnostics:
        raise InputContractError(diagnostics)
    # Keep wire-level validation shared with driver authors, while preserving
    # the existing topic/field diagnostics above for canvas error reporting.
    for topic, declaration in inputs.items():
        try:
            validate_input_port(declaration)
        except SpatialSlamError as error:
            raise InputContractError([dict(topic=topic, field='contract', reason=str(error))]) from error
    try:
        validate_mount(extrinsic, parent_frame=child, child_frame=frame,
                       require_calibrated=False)
    except SpatialSlamError as error:
        raise InputContractError([dict(topic=cloud_topic, field='extrinsic',
                                       reason=str(error))]) from error
    # This describes input geometry, not a strict/experimental admission policy.
    # Unknown clock accuracy is diagnostic; timestamp alignment remains mandatory.
    segment = pose.get('segment_id')
    def template(value):
        return ('{segment_id}' + value[len(segment):]
                if isinstance(segment, str) and segment and
                   isinstance(value, str) and value.startswith(segment + '_') else value)
    # Only the resolver's private copy gains a frame-name template. Keep the
    # producer's actual declaration untouched; its segment ID is an explicit
    # source fact, not a substring guessed from an arbitrary frame name.
    if isinstance(pose.get('frame_id'), str):
        inputs[odom_topic]['frame_id_template'] = template(pose['frame_id'])
    return dict(inputs=inputs, frame_metadata_topic=sidecar,
                extrinsic=deepcopy(extrinsic),
                clock_relation_mode='verified_relation' if independent else 'shared_source',
                relation_id=relation_id,
                # No operator-assertion flags travel with the resolved settings:
                # they were a card policy, not producer evidence.
                settings=dict(cloud_frame=frame, child_frame=template(child), clock_id=cloud['clock_id'],
                              child_from_cloud=matrix, extrinsic_mode=mode))


def frame_identity(packet, *, clock_id, payload=False, relation_id=None):
    """Check available frame evidence without inventing clock verification.

    Exact digest, frame, timestamp and mapping arithmetic remain mandatory
    whenever evidence is supplied; unverified mapping is not an admission mode.
    """
    return validate_frame_identity(packet, clock_id=clock_id,
                                   strict=False,
                                   require_digest=payload,
                                   relation_id=relation_id)
