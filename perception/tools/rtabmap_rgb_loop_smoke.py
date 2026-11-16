#!/usr/bin/env python3
"""Real ROS/native optional RGB smoke; synthetic inputs only, no motion APIs.

Usage: ROS_DOMAIN_ID=89 python3 perception/tools/rtabmap_rgb_loop_smoke.py --timeout 60
Requires the built RTAB-Map ROS environment and OpenCV. All databases are temporary.
Checks image -> scan fallback -> image recovery, not real-world loop detection.
Synthetic inputs are published every 1.25 seconds, below the native 1 Hz rate.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import uuid


def run(timeout_s):
    domain = os.environ.get('ROS_DOMAIN_ID', '')
    if not domain.isdecimal() or not 1 <= int(domain) <= 232:
        raise RuntimeError('set a private nonzero ROS_DOMAIN_ID (1..232) before running')
    if not 20 <= timeout_s <= 180:
        raise ValueError('--timeout must be between 20 and 180 seconds')
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    import cv2
    import numpy as np
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.type_support import check_for_type_support
    from rtabmap_msgs.msg import Info, SensorData
    from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
    from std_msgs.msg import String
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from plugins.pointcloud_slam_rgb import validate_rgb_camera
    from plugins import pointcloud_slam_rgb as rgb_module
    from plugins.pointcloud_slam_ros import RtabmapNode

    # Fail before creating a native process if the actual generated interface is absent.
    for message_type in (SensorData, Info, PointCloud2, CompressedImage, String):
        check_for_type_support(message_type)
    prefix = '/rtabmap_rgb_smoke_' + uuid.uuid4().hex
    width, height, focal = 160, 120, 130.
    camera = dict(schema='motus.camera/1', topic=prefix+'/rgb', format='image/jpeg',
        width=width, height=height, distortion_model='pinhole',
        K=[focal, 0., width/2., 0., focal, height/2., 0., 0., 1.], D=[0.]*5,
        vendor=dict(frame_id='smoke_optical', clock_id='ros',
                    timestamp_semantics='source_capture_mapped', registration='registered_rectified'))
    settings = dict(cloud_frame='smoke_optical', child_frame='smoke_body', clock_id='ros',
        child_from_cloud=np.eye(4).tolist(), extrinsic_mode='static',
        rgb_input=validate_rgb_camera(camera, topic=camera['topic']),
        max_gap_ms=1000, max_age_s=2., max_pending=3, max_cloud_points=width*height,
        max_measurement_age_ms=1500, max_future_skew_ms=50, voxel_size_m=.04,
        max_working_nodes=300, max_update_ms=1000, max_disk_bytes=128*1024**2,
        reserve_bytes=0, preview_interval_s=.25, preview_max_points=10000)
    # Camera rays hit one fixed, asymmetric corrugated wall. Texture is a fixed
    # world-coordinate pattern; translating the synthetic camera changes its view.
    v, u = np.indices((height, width), dtype=float)
    rays_x, rays_y = (u-width/2.)/focal, (v-height/2.)/focal
    deadline = time.monotonic()+timeout_s-10.
    rclpy.init()
    producer = gateway = executor = None
    with tempfile.TemporaryDirectory(prefix='rtabmap-rgb-smoke-') as temporary:
        root = Path(temporary)
        sent_phases, native_infos, native_inputs, native_sub = {}, {}, {}, None
        timings = dict(rgb_attachment_ms=[], project_depth_ms=[])
        original_project, original_attachment = rgb_module.project_depth, None
        cadence_s = 1.25

        def timing_summary():
            summary = {}
            for name, samples in timings.items():
                values = list(samples)
                summary[name] = dict(count=len(values), median=float(np.median(values)) if values else None,
                    p95=float(np.percentile(values, 95)) if values else None,
                    max=max(values) if values else None)
            return dict(publish_interval_s=cadence_s, timings=summary)

        try:
            producer = rclpy.create_node('rtabmap_rgb_smoke_driver')
            gateway = RtabmapNode(prefix+'/odom', prefix+'/cloud', settings,
                                  session_dir=root/'session')
            # Smoke-only wrappers measure wall time on the actual worker paths.
            # Production deadlines, parameters and admission rules remain active.
            original_attachment = gateway._rgb_attachment

            def timed_attachment(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_attachment(*args, **kwargs)
                finally:
                    timings['rgb_attachment_ms'].append((time.perf_counter()-started)*1000.)

            def timed_project(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_project(*args, **kwargs)
                finally:
                    timings['project_depth_ms'].append((time.perf_counter()-started)*1000.)

            gateway._rgb_attachment = timed_attachment
            rgb_module.project_depth = timed_project
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(producer); executor.add_node(gateway)
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            odometry = producer.create_publisher(String, prefix+'/odom', qos)
            clouds = producer.create_publisher(PointCloud2, prefix+'/cloud', qos)
            images = producer.create_publisher(CompressedImage, camera['topic'], qos)
            pid, sent, phase_results = None, 0, []
            gateway.start()

            def pump(seconds=.02):
                nonlocal native_sub, pid
                executor.spin_once(timeout_sec=seconds)
                status = gateway.status()
                if status['state'] in ('error', 'storage_limit'):
                    raise RuntimeError('gateway failed: '+json.dumps(status))
                if time.monotonic() >= deadline:
                    raise TimeoutError('native RGB smoke deadline: '+json.dumps(status))
                if gateway._active is not None:
                    process = gateway._active['process']
                    child = process._process
                    if child is not None:
                        if pid is None: pid = child.pid
                        if child.pid != pid or len(gateway._segments) != 1:
                            raise RuntimeError('RGB fallback changed the native process or segment')
                        if child.poll() is not None:
                            raise RuntimeError('native process exited during RGB smoke')
                    if native_sub is None:
                        def info(message):
                            stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
                            if message.ref_id > 0:
                                native_infos[stamp] = message.ref_id
                        def sensor_data(message):
                            stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
                            native_inputs[stamp] = (bool(message.left.data), bool(message.right.data))
                        native_sub = [producer.create_subscription(Info, process.namespace+'/info', info, qos),
                            producer.create_subscription(SensorData, process.namespace+'/sensor_data', sensor_data, qos)]
                return status

            while not (odometry.get_subscription_count() and clouds.get_subscription_count()
                       and images.get_subscription_count()):
                pump()

            for phase, expected in (('image_initial', 'image'), ('invalid_rgb', 'pointcloud'),
                                    ('image_recovered', 'image')):
                baseline = gateway.status()['native_updates']
                next_publish = 0.
                while True:
                    status = pump()
                    expected_images = (True, True) if expected == 'image' else (False, False)
                    accepted = {node_id for stamp, node_id in native_infos.copy().items()
                        if sent_phases.get(stamp) == phase and native_inputs.get(stamp) == expected_images}
                    if (len(accepted) >= 3 and status['native_updates'] >= baseline+3
                            and status['loop_closure_mode'] == expected):
                        phase_results.append(dict(phase=phase, mode=expected,
                            native_updates=status['native_updates'],
                            phase_native_nodes=len(accepted), native_pid=pid,
                            accepted_input_images=expected_images[0],
                            accepted_native_ids=sorted(accepted)))
                        print(json.dumps(phase_results[-1]), flush=True)
                        print(json.dumps(dict(phase='worker_timing', **timing_summary())), flush=True)
                        break
                    if time.monotonic() < next_publish:
                        continue
                    next_publish = time.monotonic()+cadence_s
                    # Slow enough for stable real mapping, with .15 m displacement
                    # beyond the native default node-insertion threshold.
                    x = sent*.15
                    sent += 1
                    stamp = producer.get_clock().now().nanoseconds//1_000_000*1_000_000
                    sample = dict(schema='motus.odom/1', stamp_ms=stamp//1_000_000,
                        units=dict(length='m'), pose=dict(frame_id='smoke_odom', child_frame_id='smoke_body',
                            position=[x, 0., 0.], quaternion=[0., 0., 0., 1.],
                            covariance=(np.eye(6)*.01).reshape(-1).tolist()),
                        vendor=dict(motus_slam=dict(status='tracking', stamp_source='measurement',
                            clock_id='ros', stamp_ns=stamp, segment_id='rgb_smoke')))
                    odom = String(); odom.data = json.dumps(sample); odometry.publish(odom)
                    z = np.full((height, width), 3.)
                    for _ in range(8):
                        world_x, world_y = x+rays_x*z, rays_y*z
                        z = 3.+.24*np.sin(world_x*1.7)+.13*np.cos(world_y*3.1)
                    points = np.stack((rays_x*z, rays_y*z, z), axis=-1).reshape(-1, 3)
                    world_x, world_y = x+rays_x*z, rays_y*z
                    cells_x = np.floor((world_x+8.)*50).astype(np.int64)
                    cells_y = np.floor((world_y+8.)*50).astype(np.int64)
                    texture = ((cells_x*73856093) ^ (cells_y*19349663)) % 256
                    bgr = np.repeat(texture.astype(np.uint8)[..., None], 3, axis=2)
                    ok, jpeg = cv2.imencode('.jpg', bgr)
                    if not ok: raise RuntimeError('synthetic JPEG encoding failed')
                    image = CompressedImage()
                    image.header.stamp.sec, image.header.stamp.nanosec = divmod(stamp, 10**9)
                    image.header.frame_id, image.format = settings['cloud_frame'], 'jpeg'
                    image.data = b'invalid JPEG: scan must continue' if expected == 'pointcloud' else jpeg.tobytes()
                    decoded, failed = (gateway._rgb.info()[name] for name in ('decoded', 'decode_failures'))
                    images.publish(image)
                    # Give asynchronous ingress/decoding its own turn, retaining the
                    # original capture stamp. Never wait for RGB inside the gateway.
                    until = time.monotonic()+.15
                    while time.monotonic() < until:
                        pump(.01)
                        rgb = gateway._rgb.info()
                        if rgb['decoded'] > decoded or rgb['decode_failures'] > failed: break
                    cloud = PointCloud2()
                    cloud.header = image.header
                    cloud.height, cloud.width = 1, len(points)
                    cloud.fields = [PointField(name=name, offset=i*4, datatype=7, count=1)
                                    for i, name in enumerate(('x', 'y', 'z'))]
                    cloud.point_step, cloud.row_step = 12, len(points)*12
                    cloud.is_dense, cloud.is_bigendian = True, False
                    cloud.data = np.ascontiguousarray(points, dtype='<f4').tobytes()
                    sent_phases[stamp] = phase
                    clouds.publish(cloud)

            gateway.freeze_capture()
            gateway._worker_stop.set(); gateway._status_stop.set(); gateway._wake.set()
            for worker in (gateway._worker, gateway._status_worker):
                if worker is not None:
                    worker.join(timeout=2.)
                    if worker.is_alive(): raise RuntimeError('gateway worker did not stop')
            process = gateway._segments[0]['process']
            process.close(timeout_s=7.)
            gateway.close()
            database = process.database_path
            with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
                if connection.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                    raise RuntimeError('saved native database integrity failed')
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                nodes = connection.execute('SELECT COUNT(*) FROM Node').fetchone()[0]
                columns = {row[1] for row in connection.execute('PRAGMA table_info(Data)')}
                image_count = (connection.execute('SELECT COUNT(*) FROM Data WHERE length(image)>0').fetchone()[0]
                               if 'image' in columns else 0)
                words = connection.execute('SELECT COUNT(*) FROM Word').fetchone()[0] if 'Word' in tables else 0
                features = connection.execute('SELECT COUNT(*) FROM Feature').fetchone()[0] if 'Feature' in tables else 0
            database_summary = dict(database_nodes=nodes, database_images=image_count,
                database_words=words, database_features=features,
                database_tables=sorted(tables), data_columns=sorted(columns))
            print(json.dumps(dict(phase='database_evidence', **database_summary)), flush=True)
            if nodes <= 0:
                raise RuntimeError('native database has no retained graph nodes: '+json.dumps(database_summary))
            if image_count <= 0:
                raise RuntimeError('native database has no persisted imagery: '+json.dumps(database_summary))
            if words <= 0 and features <= 0:
                raise RuntimeError('native database has no persisted visual features/words: '+json.dumps(database_summary))
            print(json.dumps(dict(result='passed', native_sensor_data=True, native_pid=pid, segments=1,
                phases=phase_results, database_nodes=nodes, database_images=image_count,
                database_words=words, database_features=features,
                loop_closure_detection_verified=False, **timing_summary())), flush=True)
        except Exception:
            print(json.dumps(dict(phase='worker_timing_failure', **timing_summary())), file=sys.stderr, flush=True)
            if gateway is not None:
                phase_inputs = {phase: dict(
                    image=sum(sent_phases.get(stamp) == phase and present == (True, True)
                              for stamp, present in native_inputs.copy().items()),
                    pointcloud=sum(sent_phases.get(stamp) == phase and present == (False, False)
                                   for stamp, present in native_inputs.copy().items()),
                    native_updates=sum(sent_phases.get(stamp) == phase for stamp in native_infos.copy()))
                    for phase in ('image_initial', 'invalid_rgb', 'image_recovered')}
                print(json.dumps(dict(phase='failure_evidence', native_inputs=phase_inputs,
                    gateway=gateway.status())), file=sys.stderr, flush=True)
            for path in (root/'session').rglob('*.log'):
                print(str(path)+'\n'+path.read_text(errors='replace')[-12000:], file=sys.stderr)
            raise
        finally:
            if gateway is not None:
                gateway.freeze_capture()
                gateway._worker_stop.set(); gateway._status_stop.set(); gateway._wake.set()
                for segment in gateway._segments:
                    try: segment['process'].close(timeout_s=3.)
                    except Exception as error: print('native cleanup: '+str(error), file=sys.stderr)
                try: gateway.close()
                except Exception as error: print('gateway cleanup: '+str(error), file=sys.stderr)
                if executor is not None: executor.remove_node(gateway)
                gateway.destroy_node()
            if executor is not None:
                if producer is not None: executor.remove_node(producer)
                executor.shutdown(timeout_sec=2.)
            if producer is not None: producer.destroy_node()
            rgb_module.project_depth = original_project
            if gateway is not None and original_attachment is not None:
                gateway._rgb_attachment = original_attachment
            rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=60., help='total budget in seconds (20..180)')
    arguments = parser.parse_args()
    try:
        run(arguments.timeout)
    except Exception as error:
        print('RGB native smoke failed: '+str(error), file=sys.stderr)
        raise SystemExit(1)
