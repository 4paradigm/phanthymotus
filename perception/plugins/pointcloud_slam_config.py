"""Robot-independent point-cloud SLAM configuration and runtime input discovery.

The canvas on Core main passes input_topics, not spatial metadata. Observe a
numeric motus.odom/1 pose and a native PointCloud2 header to resolve
frame/clock names without guessing a robot model. A structurally valid mount
may be used without calibration; the calibration flag is retained as provenance.
"""
from collections import deque
from copy import deepcopy
import json
import threading
import time
import uuid

import numpy as np

from .camera_geometry import camera_parameter, camera_transform
from .pointcloud_slam_contract import SpatialSlamError, validate_mount
from .pointcloud_slam_odom import decode_slam_odom
from .pointcloud_slam_transforms import rigid


def _matrix(value):
    if isinstance(value, str):
        value = json.loads(value)
    return rigid(value)


def reconcile_explicit_extrinsic(manual, producer):
    """Validate supplied numeric matrices without requiring identity agreement."""
    for value in (manual, producer):
        if value is not None:
            _matrix(value)


def infer_geometry(odom_sample, cloud_frame, cloud_stamp_ns, *, manual=None,
                   max_gap_ms=200, camera_info=None):
    """Resolve numeric transforms; producers own physical/time alignment."""
    pose = decode_slam_odom(odom_sample)
    if not isinstance(cloud_frame, str):
        raise ValueError('PointCloud2 header.frame_id must be text')
    if type(cloud_stamp_ns) is not int:
        raise ValueError('PointCloud2 capture timestamp must be integer nanoseconds')
    vendor = odom_sample.get('vendor')
    quality = vendor.get('motus_slam', {}) if isinstance(vendor, dict) else {}
    quality = quality if isinstance(quality, dict) else {}
    mount = quality.get('extrinsic')
    baseline = camera_parameter(camera_info, 'color_from_depth') if isinstance(camera_info, dict) else None
    if (mount is None and cloud_frame != pose.child_frame_id and isinstance(camera_info, dict)
            and isinstance(camera_info.get('vendor'), dict) and camera_info['vendor'].get('body_mount') is not None):
        if manual is None:
            from .pointcloud_slam_capture_mount import resolve_capture_mount
            return resolve_capture_mount(odom_sample, camera_info, cloud_frame)
    proposed = None
    if mount is not None:
        try:
            validate_mount(mount, parent_frame=pose.child_frame_id,
                           child_frame=cloud_frame)
        except SpatialSlamError as error:
            raise ValueError('odometry extrinsic frame/geometry invalid: ' + str(error)) from error
        if mount.get('matrix') is not None:
            proposed = _matrix(mount['matrix'])
            if (pose.external_origin is not None and baseline is not None and
                    mount.get('child_frame') not in (None, cloud_frame) and cloud_frame != pose.child_frame_id):
                proposed = proposed @ camera_transform(baseline)

    manual_matrix = _matrix(manual) if manual is not None else None
    if manual_matrix is not None:
        chosen, mode, provenance = manual_matrix, 'static', 'manual_card'
    elif proposed is not None:
        chosen, mode, provenance = proposed, 'static', 'producer_declared'
    elif mount is not None and mount.get('mode') == 'dynamic':
        chosen, mode, provenance = None, 'dynamic', 'producer_declared_dynamic'
    elif pose.external_origin is not None and cloud_frame != pose.child_frame_id:
        chosen, mode, provenance = None, 'dynamic', 'fusion_capture_mount'
    else:
        chosen, mode = np.eye(4), 'static'
        provenance = 'identical_frame' if cloud_frame == pose.child_frame_id else 'incoming_coordinates_assumed_aligned'

    child = pose.child_frame_id
    # Only normalize an explicit segment prefix. Replacing a raw substring
    # would corrupt ordinary frames (segment "a", child "base_link").
    if pose.segment_id and isinstance(child, str) and child.startswith(pose.segment_id + '_'):
        child = '{segment_id}' + child[len(pose.segment_id):]
    return ({
        'cloud_frame': cloud_frame,
        'child_frame': child,
        'clock_id': pose.clock_id,
        'child_from_cloud': chosen.tolist() if chosen is not None else None,
        'extrinsic_mode': mode,
        # An inferred identity can later be replaced when the optional
        # external stream starts publishing in body coordinates.
        **({'pose_capture_mount': True} if manual is None and mount is None else {}),
        **({'color_from_depth': deepcopy(baseline)} if baseline is not None else {}),
    }, {
        'resolution': 'observed_ros_messages',
        'extrinsic_source': provenance,
        'clock_relation': 'producer_responsibility',
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
            if not isinstance(frame, str):
                return
        except (AttributeError, ValueError, TypeError):
            return
        with lock:
            clouds.append((frame, context.observation_stamp_ns if context is not None and
                           context.time_basis == 'receipt_estimated' else stamp_ns, context))
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
                        settings,provenance=infer_geometry(sample, frame, stamp, manual=manual,
                                              max_gap_ms=max_gap_ms, camera_info=camera_info)
                        from .pointcloud_slam_capture_mount import validate_cloud_clock
                        validate_cloud_clock(context, settings['clock_id'], stamp, settings.get('capture_mount'))
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
