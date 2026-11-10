"""Read-only SLAM input preflight: validate claims without starting RTAB-Map.

A successful static check is NOT live mapping readiness. Sample-to-sample
alignment, transport QoS and frame provenance are checked by the running stream.
"""
from .pointcloud_slam_contract import (
    CLOUD_FORMAT, POSE_FORMAT, SpatialSlamError, validate_slam_capability,
)
from .pointcloud_slam_input import InputContractError, resolve_inputs
from .pointcloud_slam_config import reconcile_explicit_extrinsic


EXPECTED_TYPES = {
    POSE_FORMAT: 'std_msgs/msg/String',
    CLOUD_FORMAT: 'sensor_msgs/msg/PointCloud2',
}


def evaluate_preflight(args, *, manual_extrinsic=None, topic_types=None):
    """Return field diagnostics or explicit static-only readiness.

    topic_types is an optional mapping read from an existing ROS node's graph;
    this function never creates subscriptions, nodes, or map databases.
    """
    diagnostics = []
    def fail(topic, field, reason):
        diagnostics.append({'topic': topic, 'field': field, 'reason': reason})

    topics = args.get('input_topics') if isinstance(args, dict) else None
    bindings = args.get('input_bindings') if isinstance(args, dict) else None
    if not isinstance(topics, list) or len(topics) != 2 or any(
            not isinstance(t, str) or not t for t in topics) or len(set(topics)) != 2:
        fail('', 'input_topics', 'two distinct connected source topics are required')
    valid_bindings = (isinstance(bindings, dict)
                      and all(isinstance(k, str) and isinstance(v, str)
                              for k, v in bindings.items())
                      and len(bindings) == 2
                      and set(bindings.values()) == {POSE_FORMAT, CLOUD_FORMAT})
    if (not valid_bindings or not isinstance(topics, list) or
            not all(isinstance(t, str) for t in topics) or
            set(bindings) != set(topics)):
        fail('', 'input_bindings', 'must bind exactly one state/odom and one sensor/pointcloud2')
    if diagnostics:
        return {'state': 'not_ready', 'diagnostics': diagnostics,
                'runtime_validation': 'not_checked'}

    odom_topic = next(t for t, fmt in bindings.items() if fmt == POSE_FORMAT)
    interfaces = args.get('odom_interfaces')
    interface = interfaces.get(odom_topic) if isinstance(interfaces, dict) else None
    try:
        validate_slam_capability(interface)
    except SpatialSlamError as error:
        fail(odom_topic, 'odom_interface', str(error))

    metadata = args.get('input_metadata')
    try:
        resolved = resolve_inputs(metadata, bindings)
        reconcile_explicit_extrinsic(manual_extrinsic, resolved['settings']['child_from_cloud'])
    except InputContractError as error:
        diagnostics.extend(error.diagnostics)
    except (ValueError, TypeError, KeyError) as error:
        fail('', 'extrinsic', str(error))

    graph_checked = topic_types is not None
    if graph_checked:
        if not isinstance(topic_types, dict):
            fail('', 'ros_graph', 'unexpected graph topic listing')
        else:
            for topic, fmt in bindings.items():
                observed = topic_types.get(topic) or []
                if EXPECTED_TYPES[fmt] not in observed:
                    fail(topic, 'ros_type', 'expected ' + EXPECTED_TYPES[fmt] +
                         ', observed ' + repr(observed))

    return {'state': 'not_ready' if diagnostics else 'static_ready',
            'diagnostics': diagnostics, 'runtime_validation': (
                'graph_types_checked_frame_pairing_pending' if graph_checked
                else 'ros_graph_and_frame_pairing_pending'),
            'live_frame_pairing_checked': False,
            'message': ('Static SLAM inputs validated; runtime timestamps, source metadata '
                        'and QoS must still be verified before accepting mapping frames.'
                        if not diagnostics else 'SLAM input preflight failed')}
