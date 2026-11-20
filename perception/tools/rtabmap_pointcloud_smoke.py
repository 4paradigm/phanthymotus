#!/usr/bin/env python3
"""Standalone image gate: real ROS, native cloud mapping and graceful DB save."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.pointcloud_slam_rtabmap import mapping_parameters


def native_command(executable, database, namespace, settings):
    namespace = '/' + namespace.strip('/')
    parameters = mapping_parameters(database, namespace, settings)
    # RTAB-Map's algorithm parameters use names containing "/" ("Grid/3D"), which
    # rcl rejects in "-p name:=value" overrides. The production path
    # (plugins/pointcloud_slam_rtabmap.RtabmapProcess.start) writes a params file;
    # do the same so the build gate exercises that same path.
    parameter_path = Path(database).with_suffix('.params.json')
    parameter_path.write_text(json.dumps({'/**': {'ros__parameters': parameters}}))
    return [str(executable), '--ros-args', '--params-file', str(parameter_path),
            '-r', '__ns:=' + namespace, '-r', '__node:=rtabmap',
            '-r', 'odom:=odom', '-r', 'scan_cloud:=cloud',
            '-r', '/tf:=' + namespace + '/tf', '-r', '/tf_static:=' + namespace + '/tf_static']


def synthetic_points(pose_x):
    # Three room surfaces and an asymmetric pillar, expressed in the moving body.
    points = []
    for a in range(21):
        for b in range(21):
            y, z = -2. + a * .2, b * .1
            points.append((3. - pose_x, y, z))
            points.append((-1. + a * .2 - pose_x, 2., z))
            points.append((-1. + a * .2 - pose_x, -2. + b * .2, 0.))
    for a in range(11):
        for b in range(11):
            points.append((1.1 - pose_x, -.7 + a * .03, .2 + b * .1))
    return points


def pack_cloud(points):
    return b''.join(struct.pack('<fff', *point) for point in points)


def tracking_sample(stamp_ns, pose_x):
    if type(stamp_ns) is not int or stamp_ns <= 0 or stamp_ns % 1_000_000:
        raise ValueError('capture time must be positive and millisecond aligned')
    covariance = [.01 if index % 7 == 0 else 0. for index in range(36)]
    return {'schema': 'motus.odom/1', 'stamp_ms': stamp_ns // 1_000_000,
        'units': {'length': 'm'}, 'pose': {'frame_id': 'smoke_odom',
            'child_frame_id': 'smoke_body', 'position': [pose_x, 0., 0.],
            'quaternion': [0., 0., 0., 1.], 'covariance': covariance},
        'vendor': {'motus_slam': {'status': 'tracking', 'stamp_source': 'measurement',
            'clock_id': 'ros', 'segment_id': 'gateway_smoke'}}}


def mapping_packet_summary(packet):
    raw = bytes(packet)
    if len(raw) < 17:
        raise RuntimeError('truncated mapping packet')
    x, y, yaw, flags, count = struct.unpack('<fffBI', raw[:17])
    if ((flags & 3) != 3 or flags & ~63 or (flags & 24) == 24 or
            not 0 < count <= 80000 or len(raw) != 17 + count * 12):
        raise RuntimeError('invalid mapping header or point count')
    points = list(struct.iter_unpack('<fff', raw[17:]))
    if not all(math.isfinite(value) for value in (x, y, yaw)) or not all(
            math.isfinite(value) for point in points for value in point):
        raise RuntimeError('nonfinite mapping pose or geometry')
    return {'pose': (x, y, yaw), 'point_count': count,
            'z_range_m': (min(point[2] for point in points), max(point[2] for point in points))}


def require_3d_map(message):
    fields = [field for field in message.fields if field.name == 'z']
    if len(fields) != 1 or fields[0].datatype != 7 or fields[0].count != 1:
        raise RuntimeError('native cloud_map has no readable three-dimensional Z field')
    unpack = struct.Struct(('>' if message.is_bigendian else '<') + 'f').unpack_from
    heights = [unpack(message.data, row * message.row_step + column * message.point_step
                      + fields[0].offset)[0]
               for row in range(message.height) for column in range(message.width)]
    heights = [height for height in heights if math.isfinite(height)]
    if not heights or max(heights) - min(heights) < .5:
        raise RuntimeError('native cloud_map lost the synthetic scene\'s three-dimensional Z extent')
    return min(heights), max(heights)


def validate_database(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError('native database was not saved')
    try:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            if db.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                raise RuntimeError('native SQLite database integrity check failed')
            count = db.execute('SELECT COUNT(*) FROM Node').fetchone()[0]
    except sqlite3.Error as error:
        raise RuntimeError('invalid native SQLite database: ' + str(error)) from error
    if count <= 0:
        raise RuntimeError('native database has no persisted nodes')
    return count


def stop_native(process, *, timeout_s=30.):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise RuntimeError('native shutdown timed out; database save is unverified') from error
    if code != 0:
        raise RuntimeError('native process exit was ' + str(code))


def run_smoke(timeout_s):
    # These imports intentionally occur only in the standalone native run.
    import rclpy
    from ament_index_python.packages import get_package_prefix
    from nav_msgs.msg import Odometry
    from rclpy.type_support import check_for_type_support
    from rtabmap_msgs.msg import Info, MapGraph
    from rtabmap_msgs.srv import GetMap, LoadDatabase
    from sensor_msgs.msg import PointCloud2, PointField

    for interface in (Info, MapGraph, GetMap, LoadDatabase):
        check_for_type_support(interface)
    executable = Path(get_package_prefix('rtabmap_slam')) / 'lib/rtabmap_slam/rtabmap'
    if not executable.is_file():
        raise RuntimeError('native rtabmap executable is missing')
    os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')
    rclpy.init()
    node = rclpy.create_node('rtabmap_build_gate')
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix='rtabmap-build-') as directory:
            database, logfile = Path(directory) / 'map.db', Path(directory) / 'native.log'
            namespace = 'rtabmap_build_smoke'
            settings = {'max_pending': 10, 'voxel_size_m': .08,
                        'max_working_nodes': 300, 'max_update_ms': 100}
            parameters = mapping_parameters(database, namespace, settings)
            infos, maps = [], []
            odometry = node.create_publisher(Odometry, '/' + namespace + '/odom', 10)
            clouds = node.create_publisher(PointCloud2, '/' + namespace + '/cloud', 10)
            node.create_subscription(Info, '/' + namespace + '/info', infos.append, 10)
            node.create_subscription(PointCloud2, '/' + namespace + '/cloud_map', maps.append, 10)
            client = node.create_client(GetMap, '/' + namespace + '/rtabmap/get_map_data')
            deadline = time.monotonic() + timeout_s
            try:
                with logfile.open('w') as output:
                    process = subprocess.Popen(native_command(executable, database, namespace, settings),
                                               stdout=output, stderr=subprocess.STDOUT)

                    def wait_for(predicate, label, publish=None):
                        next_publish = 0.
                        while not predicate():
                            if process.poll() is not None:
                                raise RuntimeError('native process exited before ' + label)
                            if time.monotonic() >= deadline:
                                raise RuntimeError('native smoke timed out waiting for ' + label)
                            if publish and time.monotonic() >= next_publish:
                                publish()
                                next_publish = time.monotonic() + .25
                            rclpy.spin_once(node, timeout_sec=.05)

                    def publish(pose_x):
                        stamp = node.get_clock().now().to_msg()
                        odom = Odometry()
                        odom.header.stamp, odom.header.frame_id = stamp, namespace + '/odom'
                        odom.child_frame_id = parameters['frame_id']
                        odom.pose.pose.position.x = pose_x
                        odom.pose.pose.orientation.w = 1.
                        for axis in range(6):
                            odom.pose.covariance[axis * 7] = .01
                        points = synthetic_points(pose_x)
                        cloud = PointCloud2()
                        cloud.header.stamp, cloud.header.frame_id = stamp, parameters['frame_id']
                        cloud.height, cloud.width = 1, len(points)
                        cloud.fields = [PointField(name=name, offset=index * 4,
                            datatype=PointField.FLOAT32, count=1) for index, name in enumerate('xyz')]
                        cloud.point_step, cloud.row_step = 12, len(points) * 12
                        cloud.is_dense = True
                        cloud.data = pack_cloud(points)
                        odometry.publish(odom)
                        clouds.publish(cloud)

                    wait_for(lambda: odometry.get_subscription_count() and clouds.get_subscription_count()
                             and client.service_is_ready(), 'native input subscriptions and service')
                    for step in range(6):
                        previous = len(infos)
                        wait_for(lambda: len(infos) > previous, 'real mapping Info',
                                 lambda x=step * .15: publish(x))
                    wait_for(lambda: any(message.width * message.height > 0 and message.data
                                         for message in maps), 'nonempty native cloud_map')
                    mapped_cloud = max(maps, key=lambda message: message.width * message.height)
                    z_range = require_3d_map(mapped_cloud)
                    request = GetMap.Request()
                    request.global_map, request.optimized, request.graph_only = True, True, True
                    future = client.call_async(request)
                    wait_for(future.done, 'native graph service response')
                    graph = future.result().data.graph
                    if len(graph.poses_id) < 3 or any(not math.isfinite(value)
                            for pose in graph.poses for value in
                            (pose.position.x, pose.position.y, pose.position.z,
                             pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)):
                        raise RuntimeError('native optimized graph is empty or nonfinite')
                    stop_native(process)
                    process = None
                count = validate_database(database)
                print(json.dumps({'phase': 'native_engine', 'native_engine': True, 'info_messages': len(infos),
                    'map_points': mapped_cloud.width * mapped_cloud.height, 'map_z_range_m': z_range,
                    'graph_nodes': len(graph.poses_id), 'database_nodes': count}), flush=True)
            except Exception:
                if process is not None and process.poll() is None:
                    try:
                        stop_native(process)
                    except RuntimeError as error:
                        print(str(error), file=sys.stderr)
                if logfile.exists():
                    print(logfile.read_text(errors='replace')[-16000:], file=sys.stderr)
                raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_gateway_smoke(timeout_s=60.):
    """Exercise the production gateway, mapping wire format and native catalog."""
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import String, UInt8MultiArray
    from plugins.pointcloud_slam_ros import RtabmapNode
    from plugins.pointcloud_slam_library import save_rtabmap_revision, load_map_snapshots

    settings = dict(cloud_frame='smoke_cloud', child_frame='smoke_body', clock_id='ros',
        child_from_cloud=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        max_gap_ms=500, max_age_s=2, max_pending=10, max_cloud_points=10000,
        max_measurement_age_ms=1500, max_future_skew_ms=100, voxel_size_m=.08,
        max_working_nodes=300, max_update_ms=100, max_disk_bytes=128 * 1024**2,
        reserve_bytes=0, preview_interval_s=.1, preview_max_points=10000)
    rclpy.init()
    producer = rclpy.create_node('rtabmap_gateway_build_gate')
    try:
        with tempfile.TemporaryDirectory(prefix='rtabmap-gateway-') as directory:
            root = Path(directory)
            gateway = RtabmapNode('/rtabmap_gateway_smoke/pose', '/rtabmap_gateway_smoke/cloud',
                                  settings, session_dir=root / 'session')
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(producer)
            executor.add_node(gateway)
            try:
                qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
                odom_pub = producer.create_publisher(String, '/rtabmap_gateway_smoke/pose', qos)
                cloud_pub = producer.create_publisher(PointCloud2, '/rtabmap_gateway_smoke/cloud', qos)
                output = next(item['topic'] for item in gateway.outputs() if item['format'] == 'sensor/mapping')
                packets = []
                producer.create_subscription(UInt8MultiArray, output,
                    lambda message: packets.append(bytes(message.data)), qos)
                gateway.start()
                deadline, next_publish, sent = time.monotonic() + timeout_s, 0., 0

                def publish():
                    nonlocal sent
                    stamp_ns = producer.get_clock().now().nanoseconds // 1_000_000 * 1_000_000
                    step = sent % 20
                    pose_x = .15 * min(step, 20 - step)
                    sent += 1
                    sample = String(); sample.data = json.dumps(tracking_sample(stamp_ns, pose_x))
                    points = synthetic_points(pose_x)
                    cloud = PointCloud2()
                    cloud.header.stamp.sec, cloud.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
                    cloud.header.frame_id = settings['cloud_frame']
                    cloud.height, cloud.width = 1, len(points)
                    cloud.fields = [PointField(name=name, offset=index * 4,
                        datatype=PointField.FLOAT32, count=1) for index, name in enumerate('xyz')]
                    cloud.point_step, cloud.row_step = 12, len(points) * 12
                    cloud.is_bigendian, cloud.is_dense, cloud.data = False, True, pack_cloud(points)
                    odom_pub.publish(sample)
                    cloud_pub.publish(cloud)

                while True:
                    status = gateway.status()
                    if status['state'] in ('error', 'storage_limit'):
                        raise RuntimeError('production gateway failed: ' + status['last_error'])
                    if packets and status['native_updates'] >= 3:
                        summary = mapping_packet_summary(packets[-1])
                        if (status['backend'] == 'rtabmap' and status['segment_id'] == 'gateway_smoke'
                                and status['frame_id']
                                and summary['z_range_m'][1] - summary['z_range_m'][0] >= .5):
                            break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('production gateway mapping timed out: ' + json.dumps(status))
                    if time.monotonic() >= next_publish and odom_pub.get_subscription_count() and cloud_pub.get_subscription_count():
                        publish()
                        next_publish = time.monotonic() + .25
                    executor.spin_once(timeout_sec=.05)
                if status['state'] != 'tracking' or not summary['point_count']:
                    raise RuntimeError('production mapping packet lost its native frame or pose')
                native_frame = status['frame_id']
                native_updates = status['native_updates']
                gateway.freeze_capture()
                forwarded = gateway.status()['forwarded_clouds']
                publish()
                until = time.monotonic() + .3
                while time.monotonic() < until:
                    executor.spin_once(timeout_sec=.05)
                if gateway.status()['forwarded_clouds'] != forwarded:
                    raise RuntimeError('production gateway resumed capture after freeze')
                gateway.close()
                artifacts = gateway.artifacts()
                if len(artifacts['segments']) != 1 or artifacts['segments'][0]['snapshot'] is None:
                    raise RuntimeError('production gateway did not finalize its native database and snapshot')
                segment = artifacts['segments'][0]
                if segment['frame_id'] != native_frame:
                    raise RuntimeError('native artifact frame differs from the published map frame')
                source_nodes = validate_database(segment['database'])
                map_id, revision, duplicate = save_rtabmap_revision(root / 'library', artifacts,
                    settings, 'Native gateway smoke', None, 'gateway_build_smoke')
                snapshots = load_map_snapshots(root / 'library', map_id, revision, max_points=10000)
                if (duplicate or revision != 1 or len(snapshots) != 1
                        or snapshots[0]['packet'] != segment['snapshot']
                        or snapshots[0]['frame_id'] != segment['frame_id']
                        or snapshots[0]['metadata']['tracking'] is not False):
                    raise RuntimeError('native map revision or cached snapshot did not round-trip')
                cached = mapping_packet_summary(snapshots[0]['packet'])
                metadata_path = root / 'library/maps' / map_id / '1.json'
                metadata = json.loads(metadata_path.read_text())
                database = metadata_path.parent / metadata['segments'][0]['database']
                if validate_database(database) != source_nodes:
                    raise RuntimeError('saved native database lost graph nodes')
                with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
                    data_rows = connection.execute('SELECT COUNT(*) FROM Data').fetchone()[0]
                if not data_rows:
                    raise RuntimeError('saved native database has no sensor Data records')
                print(json.dumps({'phase': 'gateway_archive', 'native_gateway': True,
                    'map_frame': segment['frame_id'], 'mapping_points': cached['point_count'],
                    'map_z_range_m': cached['z_range_m'], 'native_updates': native_updates,
                    'database_nodes': source_nodes, 'database_data': data_rows, 'revision': revision}), flush=True)
            except Exception:
                for logfile in (root / 'session').rglob('*.log'):
                    print(logfile.read_text(errors='replace')[-16000:], file=sys.stderr)
                raise
            finally:
                try:
                    gateway.close()
                finally:
                    executor.remove_node(gateway)
                    executor.remove_node(producer)
                    executor.shutdown(timeout_sec=5.)
                    gateway.destroy_node()
    finally:
        producer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=120.)
    run_smoke(parser.parse_args().timeout)
    run_gateway_smoke()
