"""ROS-free routing and numeric transforms for connected SLAM inputs."""
from copy import deepcopy

from .pointcloud_slam_contract import (
    SpatialSlamError, validate_input_port, validate_mount,
    validate_frame_identity,
)


class InputContractError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('; '.join(f"{d['topic']}: {d['field']} {d['reason']}" for d in diagnostics))


def resolve_inputs(metadata, bindings):
    """Resolve connected pose/cloud routing and any numeric transform."""
    if isinstance(metadata, dict) and any(isinstance(metadata.get(t),dict) and
            metadata[t].get('schema')=='motus.slam_input/2' for t in bindings):
        from .spatial_contract import resolve_v2_inputs
        return resolve_v2_inputs(metadata,bindings)
    if not isinstance(metadata, dict):
        raise InputContractError([dict(topic='', field='input_metadata', reason='must be topic keyed')])
    inputs = {}
    for topic, fmt in bindings.items():
        declaration = metadata.get(topic)
        try:
            validate_input_port(declaration)
            if declaration['format'] != fmt:
                raise SpatialSlamError('input format differs from binding')
        except SpatialSlamError as error:
            raise InputContractError([dict(topic=topic, field='contract', reason=str(error))]) from error
        inputs[topic] = deepcopy(declaration)
        inputs[topic]['topic']=topic
    pose = next(d for d in inputs.values() if d['format'] == 'state/odom')
    cloud = next(d for d in inputs.values() if d['format'] == 'sensor/pointcloud2')
    extrinsic = pose.get('extrinsic')
    if extrinsic is None:
        extrinsic = cloud.get('extrinsic')
    mode, matrix = 'static', None
    if extrinsic is not None:
        try:
            validate_mount(extrinsic)
        except SpatialSlamError as error:
            raise InputContractError([dict(topic=cloud['topic'], field='extrinsic', reason=str(error))]) from error
        mode = extrinsic.get('mode', 'static')
        matrix = extrinsic.get('matrix')
    if matrix is None and mode != 'dynamic':
        # Arithmetic convention only: incoming cloud coordinates are already
        # in pose-child coordinates. This makes no calibration assertion.
        matrix = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    return dict(inputs=inputs, frame_metadata_topic=cloud.get('frame_metadata_topic'),
                extrinsic=deepcopy(extrinsic or {}), clock_relation_mode='producer_responsibility',
                relation_id=None,
                settings=dict(cloud_frame=cloud.get('frame_id', ''), child_frame=pose.get('child_frame_id', ''),
                              clock_id=pose.get('clock_id'), child_from_cloud=matrix, extrinsic_mode=mode))


def frame_identity(packet, *, clock_id, payload=False, relation_id=None):
    return validate_frame_identity(packet, clock_id=clock_id, strict=False)


def continuous_pose_clock(previous, current):
    """Clock metadata does not govern numeric pose admission."""
    return True
