"""Robot-independent point-cloud SLAM configuration and runtime input discovery.

The canvas on Core main passes input_topics, not spatial metadata. Observe a
fresh tracking motus.odom/1 pose and a native PointCloud2 header to resolve
frame/clock names without guessing a robot model. A structurally valid mount
may be used without calibration; the calibration flag is retained as provenance.
"""
from collections import deque
import json
import threading
import time
import uuid

import numpy as np

from .pointcloud_slam_contract import SpatialSlamError, validate_mount
from .pointcloud_slam_odom import decode_slam_odom
from .pointcloud_slam_transforms import rigid


def _matrix(value):
    if isinstance(value, str):
        value = json.loads(value)
    return rigid(value)


def reconcile_explicit_extrinsic(manual, producer):
    """An explicit, validated producer contract cannot be silently overridden."""
    if manual is None or producer is None:
        return
    if not np.allclose(_matrix(manual), _matrix(producer), rtol=0, atol=1e-6):
        raise ValueError('child_from_cloud conflicts with validated upstream extrinsic')


def infer_geometry(odom_sample, cloud_frame, cloud_stamp_ns, *, manual=None,
                   max_gap_ms=200, camera_info=None):
    """Build session geometry from actual messages, never robot-model presets.

    This legacy-Core route verifies frame names and a contemporaneous timestamp
    pair. It does NOT claim a verified relationship between independent clocks;
    those proofs require the existing strict motus.slam_input/1 declaration path.
    """
    pose = decode_slam_odom(odom_sample)
    if not isinstance(cloud_frame, str) or not cloud_frame.strip():
        raise ValueError('PointCloud2 header.frame_id is missing')
    if type(cloud_stamp_ns) is not int or cloud_stamp_ns <= 0:
        raise ValueError('PointCloud2 capture timestamp is missing')
    if abs(cloud_stamp_ns - pose.stamp_ms * 1_000_000) > max_gap_ms * 1_000_000:
        raise ValueError('pointcloud and odometry acquisition times do not overlap')
    if not pose.clock_id or not pose.segment_id:
        raise ValueError('SLAM odometry clock/segment is missing')

    quality = odom_sample['vendor']['motus_slam']
    mount = quality.get('extrinsic')
    if (mount is None and cloud_frame != pose.child_frame_id and isinstance(camera_info, dict)
            and isinstance(camera_info.get('vendor'), dict) and camera_info['vendor'].get('body_mount') is not None):
        if manual is not None:
            raise ValueError('static card transform cannot replace a declared capture mount')
        from .pointcloud_slam_capture_mount import resolve_capture_mount
        return resolve_capture_mount(odom_sample, camera_info, cloud_frame)
    proposed = None
    if mount is not None:
        # The legacy route cannot consume a per-capture TF mount. Reject it
        # with the stable public error before static-matrix validation.
        if isinstance(mount, dict) and mount.get('mode', 'static') == 'dynamic':
            raise ValueError('dynamic mount requires the verified per-frame input contract')
        try:
            validate_mount(mount, parent_frame=pose.child_frame_id,
                           child_frame=cloud_frame)
        except SpatialSlamError as error:
            raise ValueError('odometry extrinsic frame/geometry invalid: ' + str(error)) from error
        proposed = _matrix(mount.get('matrix'))

    manual_matrix = _matrix(manual) if manual is not None else None
    # The odometry estimator has already used the upstream base->sensor
    # transform when producing its pose. A conflicting manual matrix cannot
    # silently change that reference point, even for an assumed mount.
    if proposed is not None and manual_matrix is not None and not np.allclose(
            proposed, manual_matrix, rtol=0, atol=1e-6):
        raise ValueError('child_from_cloud conflicts with odometry producer extrinsic')
    if mount is not None and mount.get('calibrated') is True:
        chosen = proposed
        provenance = 'producer_calibrated'
    elif manual_matrix is not None:
        chosen = manual_matrix
        provenance = 'manual_card'
    elif mount is not None:
        chosen = proposed
        provenance = 'producer_declared_uncalibrated'
    elif cloud_frame == pose.child_frame_id:
        # Identical frames imply an identity mapping, not an assumed mount.
        chosen = np.eye(4)
        provenance = 'identical_frame'
    else:
        raise ValueError('child_from_cloud is required: upstream mount is absent '
                         'or is an unverified assumption; enter measured 4x4 on the SLAM card')

    child = pose.child_frame_id
    # Only normalize an explicit segment prefix. Replacing a raw substring
    # would corrupt ordinary frames (segment "a", child "base_link").
    if child.startswith(pose.segment_id + '_'):
        child = '{segment_id}' + child[len(pose.segment_id):]
    return ({
        'cloud_frame': cloud_frame,
        'child_frame': child,
        'clock_id': pose.clock_id,
        'child_from_cloud': chosen.tolist(),
        'extrinsic_mode': 'static',
    }, {
        'resolution': 'observed_ros_messages',
        'extrinsic_source': provenance,
        'clock_relation': 'timestamp_overlap_only_not_independent_clock_proof',
        'segment_id': pose.segment_id,
    })


def discover_runtime_geometry(executor, odom_topic, cloud_topic, *,
                              manual=None, max_gap_ms=200, timeout_s=5.0, camera_info=None, input_registry=None):
    """Temporarily subscribe on the already-running executor.

    Do not start a second odometry estimator or use ROS arrival time as a
    substitute for sensor capture time. Every discovery node is disposed.
    """
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String
    from sensor_msgs.msg import PointCloud2
    from utils.ros_lifecycle import dispose_node

    probe = Node('pointcloud_slam_config_probe_' + uuid.uuid4().hex[:8])
    lock, wake = threading.Lock(), threading.Event()
    poses, clouds = deque(maxlen=12), deque(maxlen=12)

    leases=[]

    def on_odom(msg, context=None):
        try:
            from .spatial_contract import normalize_pose_time
            sample = normalize_pose_time(json.loads(msg.data),context)
            decode_slam_odom(sample)
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        with lock:
            poses.append(sample)
        wake.set()

    def on_cloud(msg, context=None):
        try:
            stamp = msg.header.stamp
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            frame = msg.header.frame_id
            if (stamp_ns <= 0 and context is None) or not frame:
                return
        except (AttributeError, ValueError, TypeError):
            return
        with lock:
            clouds.append((frame, context.observation_stamp_ns if context is not None else stamp_ns, context))
        wake.set()

    try:
        if input_registry is None:
            probe.create_subscription(String, odom_topic, on_odom, qos_profile_sensor_data)
            probe.create_subscription(PointCloud2, cloud_topic, on_cloud, qos_profile_sensor_data)
        else:
            leases.append(input_registry.acquire(odom_topic,String,{},on_odom))
            leases.append(input_registry.acquire(cloud_topic,PointCloud2,camera_info or {},on_cloud))
        executor.add_node(probe)
        deadline = time.monotonic() + timeout_s
        last_error = ''
        while time.monotonic() < deadline:
            with lock:
                samples = list(poses)
                headers = list(clouds)
            for frame, stamp, context in reversed(headers):
                for sample in reversed(samples):
                    try:
                        if context is not None and context.normalized_clock_id!=sample['vendor']['motus_slam']['clock_id']:
                            raise ValueError('cloud and pose have no comparable observation clock')
                        settings,provenance=infer_geometry(sample, frame, stamp, manual=manual,
                                              max_gap_ms=max_gap_ms, camera_info=camera_info)
                        if context is not None:
                            settings['time_basis']=context.time_basis
                            provenance['time_basis']=context.time_basis
                        return settings,provenance
                    except ValueError as error:
                        last_error = str(error)
                        # A malformed or uncalibrated mount cannot be fixed by
                        # waiting for a newer sample with the same frames.
                        if 'child_from_cloud' in last_error or 'extrinsic frame' in last_error:
                            raise
            wake.wait(min(0.1, max(0., deadline - time.monotonic())))
            wake.clear()
        raise ValueError('cannot resolve live SLAM inputs: ' +
                         (last_error or 'waiting for tracking odometry and capture-timed PointCloud2'))
    finally:
        for lease in leases:lease.close()
        dispose_node(executor, probe, label='pointcloud_slam_config_probe')
