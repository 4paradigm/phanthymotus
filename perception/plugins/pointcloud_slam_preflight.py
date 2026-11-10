"""Read-only SLAM input diagnostics, without creating RTAB-Map or ROS subscribers.

The single supported mapping path uses real pose/cloud timestamps and frames;
upstream calibration and clock verification flags are provenance, not modes.
Static readiness never implies live frame synchronization or usable mapping.
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

# A producer can be asked for info before camera intrinsics, its first pose,
# or any native cloud has arrived. These unavailable facts are waiting states,
# not proof of an unsafe or contradictory input.
_PENDING_FIELDS = frozenset({
    'input_metadata', 'source_id', 'stamp_source', 'reference_point',
    'child_frame_id', 'frame_id', 'extrinsic', 'extrinsic.source',
    'extrinsic.version', 'extrinsic.matrix', 'clock.mapping',
    'frame_metadata_topic',
})
_FATAL_FIELDS = frozenset({
    'schema', 'topic/format', 'kind', 'cloud_semantics', 'timing_model',
    'observation_model', 'extrinsic.mode', 'extrinsic.pose_reference',
    'clock.relation_id',
})


def evaluate_preflight(args, *, manual_extrinsic=None, topic_types=None):
    """Distinguish contradictory claims from missing startup-time evidence.

    `static_ready` confirms declarations only; real transport, tracking,
    frame pairing, and timestamp monotonicity are validated on live frames.
    """
    diagnostics, pending, notes = [], [], []

    def fail(topic, field, reason):
        diagnostics.append({'topic': topic, 'field': field, 'reason': reason})

    def wait(topic, field, reason):
        pending.append({'topic': topic, 'field': field, 'reason': reason})

    topics = args.get('input_topics') if isinstance(args, dict) else None
    bindings = args.get('input_bindings') if isinstance(args, dict) else None
    if (not isinstance(topics, list) or len(topics) != 2 or
            any(not isinstance(t, str) or not t for t in topics) or
            len(set(topics)) != 2):
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
                'pending': pending, 'notes': notes, 'runtime_validation': 'not_checked',
                'live_frame_pairing_checked': False}

    odom_topic = next(t for t, fmt in bindings.items() if fmt == POSE_FORMAT)
    interfaces = args.get('odom_interfaces')
    interface = interfaces.get(odom_topic) if isinstance(interfaces, dict) else None
    if interface is None:
        wait(odom_topic, 'odom_interface', 'source has not reported a SLAM pose capability yet')
    else:
        try:
            validate_slam_capability(interface)
        except SpatialSlamError as error:
            # An explicit unavailable SLAM capability is a contradiction, not
            # a transient absent declaration (e.g. G1 leg odometry).
            fail(odom_topic, 'odom_interface', str(error))

    metadata = args.get('input_metadata')
    if not isinstance(metadata, dict) or not metadata:
        wait('', 'input_metadata', 'source declarations are not available yet')
    else:
        try:
            resolved = resolve_inputs(metadata, bindings)
            reconcile_explicit_extrinsic(manual_extrinsic,
                                         resolved['settings']['child_from_cloud'])
            pose = resolved['inputs'][odom_topic]
            cloud_topic = next(t for t, fmt in bindings.items() if fmt == CLOUD_FORMAT)
            cloud = resolved['inputs'][cloud_topic]
            if not (pose.get('clock') or {}).get('verified'):
                notes.append('odometry clock relation is not independently verified')
            if not (cloud.get('clock') or {}).get('verified'):
                notes.append('cloud clock relation is not independently verified')
            if not resolved['extrinsic'].get('calibrated'):
                notes.append('mount not marked calibrated; verify the actual transform')
        except InputContractError as error:
            for issue in error.diagnostics:
                field, reason = issue.get('field', ''), issue.get('reason', '')
                # Missing startup evidence is pending; contradictory present
                # claims remain blocking. Never silently accept a mismatch.
                if (field in _FATAL_FIELDS or 'conflicting' in reason or
                        (field == 'clock_id' and 'differs' in reason) or
                        (field == 'extrinsic' and 'must explicitly map' in reason) or
                        (field not in _PENDING_FIELDS)):
                    diagnostics.append(issue)
                else:
                    pending.append(issue)
        except (ValueError, TypeError, KeyError) as error:
            fail('', 'extrinsic', str(error))

    graph_checked = topic_types is not None
    if graph_checked:
        if not isinstance(topic_types, dict):
            wait('', 'ros_graph', 'ROS topic listing is not available')
        else:
            for topic, fmt in bindings.items():
                observed = topic_types.get(topic) or []
                if EXPECTED_TYPES[fmt] not in observed:
                    if observed:
                        fail(topic, 'ros_type', 'expected ' + EXPECTED_TYPES[fmt] +
                             ', observed ' + repr(observed))
                    else:
                        wait(topic, 'ros_type', 'source topic not publishing yet')

    state = ('not_ready' if diagnostics else
             'waiting_for_inputs' if pending else 'static_ready')
    return {
        'state': state, 'diagnostics': diagnostics, 'pending': pending, 'notes': notes,
        'runtime_validation': ('graph_types_checked_frame_pairing_pending' if graph_checked
                               else 'ros_graph_and_frame_pairing_pending'),
        'live_frame_pairing_checked': False,
        'message': ('Contradictory SLAM source declarations or ROS types'
                    if diagnostics else
                    'Waiting for source frames or metadata; project start may continue'
                    if pending else
                    'Static SLAM inputs resolved; live frame and timestamp pairing pending'),
    }
