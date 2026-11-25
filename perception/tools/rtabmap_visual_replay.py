#!/usr/bin/env python3
"""Replay slam_visual_scene through the production gateway and real ROS/native node.

Requires the pinned patched overlay. No robot endpoints or motion interfaces.
Keeps Info, graph, process, DB, projection timing and archive-reopen evidence.
"""
import argparse
from array import array
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import uuid


def run(scene, output, resume_run=None):
    domain = os.environ.get('ROS_DOMAIN_ID', '')
    if not domain.isdecimal() or not 1 <= int(domain) <= 232:
        raise RuntimeError('set an isolated nonzero ROS_DOMAIN_ID')
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    import cv2
    import numpy as np
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
    from std_msgs.msg import String
    from rtabmap_msgs.msg import Info, MapGraph, SensorData
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from plugins.pointcloud_slam_rgb import validate_rgb_camera
    from plugins.pointcloud_slam_ros import RtabmapNode
    from plugins.pointcloud_slam_library import save_rtabmap_revision, load_map_snapshots, copy_native_map

    scene, output = Path(scene), Path(output)
    origin = np.loadtxt(scene/'origin.txt') if (scene/'origin.txt').is_file() else np.zeros(3)
    rows = [line.split() for line in (scene/'frames.txt').read_text().splitlines()]
    if not rows: raise ValueError('empty replay manifest')
    output.mkdir(parents=True, exist_ok=False)
    restored = None
    if resume_run is not None:
        previous = Path(resume_run)
        saved = json.loads((previous/'result.json').read_text())
        restored = copy_native_map(previous/'archive', saved['map_id'], saved['revision'], output/'restored.db')
    prefix = '/offline_visual_' + uuid.uuid4().hex
    camera = dict(schema='motus.camera/1', format='image/jpeg', width=320, height=240,
        K=[260., 0., 160., 0., 260., 120., 0., 0., 1.], D=[])
    settings = dict(cloud_frame='', child_frame='', clock_id='', child_from_cloud=np.eye(4).tolist(),
        max_gap_ms=250, max_age_s=2., max_pending=3, max_cloud_points=320*240,
        max_measurement_age_ms=None, max_future_skew_ms=None, voxel_size_m=.04,
        max_working_nodes=300, max_update_ms=3000, reserve_bytes=0,
        max_disk_bytes=512*1024**2, preview_interval_s=.1, preview_max_points=10000,
        rgb_input=validate_rgb_camera(camera, topic=prefix+'/rgb'))
    rclpy.init()
    producer = rclpy.create_node('offline_visual_producer')
    gateway = RtabmapNode(prefix+'/odom', prefix+'/cloud', settings, session_dir=output/'session', restored_map=restored)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(producer); executor.add_node(gateway)
    spin = threading.Thread(target=executor.spin, daemon=True); spin.start()
    qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
    odometry = producer.create_publisher(String, prefix+'/odom', qos)
    clouds = producer.create_publisher(PointCloud2, prefix+'/cloud', qos)
    images = producer.create_publisher(CompressedImage, prefix+'/rgb', qos)
    infos, inputs, graphs, subscriptions = {}, {}, [], []
    pid = None

    def stamp(msg): return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
    def wait(check, seconds=15.):
        until = time.monotonic()+seconds
        while not check():
            status = gateway.status()
            if status['state'] in ('error', 'storage_limit'): raise RuntimeError(json.dumps(status))
            if time.monotonic() > until: raise TimeoutError(json.dumps(status))
            time.sleep(.005)
    def info(msg):
        infos[stamp(msg)] = dict(ref=msg.ref_id, loop=msg.loop_closure_id,
            proximity=msg.proximity_detection_id, received_monotonic=time.monotonic(),
            stats=dict(zip(msg.stats_keys, msg.stats_values)))
    def graph(msg):
        graphs.append(dict(stamp_ns=stamp(msg), poses={str(i): [p.position.x,p.position.y,p.position.z,
            p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w] for i,p in zip(msg.poses_id,msg.poses)},
            links=[dict(from_id=l.from_id,to_id=l.to_id,type=l.type) for l in msg.links]))
    def sensor(msg): inputs[stamp(msg)] = bool(msg.left.data and msg.right.data)

    records = []
    try:
        gateway.start()
        wait(lambda: odometry.get_subscription_count() and clouds.get_subscription_count() and images.get_subscription_count())
        next_frame = time.monotonic()
        for number, has_pose, external_x, truth_x, phase in rows:
            i, has_pose = int(number), bool(int(has_pose))
            while time.monotonic() < next_frame: time.sleep(.005)
            next_frame = time.monotonic()+1.3
            capture = (i+10)*1_500_000_000
            sent=time.monotonic()
            sample = dict(schema='motus.odom/1', stamp_ms=capture//1_000_000,
                pose=dict(position=(origin+np.array([float(external_x),0.,0.])).tolist()
                    if phase in ('outbound','revisit') else [float(external_x),0.,0.],
                    quaternion=[0.,0.,0.,1.]) if has_pose else None,
                vendor=dict(motus_slam=dict(stamp_ns=capture, status='tracking' if has_pose else 'lost',
                    segment_id='reset' if 'reset' in phase else 'initial')))
            msg = String(); msg.data = json.dumps(sample); odometry.publish(msg)
            rgb = CompressedImage(); rgb.header.stamp.sec,rgb.header.stamp.nanosec=divmod(capture,10**9)
            rgb.format='jpeg'; rgb.data=(scene/f'{i}.jpg').read_bytes()
            if phase == 'external_rgb_gap':
                failed = gateway._rgb.info()['decode_failures']
                rgb.data=b'invalid JPEG'
                images.publish(rgb)
                wait(lambda:gateway._rgb.info()['decode_failures']>failed)
            elif phase != 'rgb_gap':
                # Exercise production asynchronous decoding; do not pre-wait
                # for readiness and conceal image/cloud scheduling races.
                images.publish(rgb)
            depth = cv2.imread(str(scene/f'{i}.png'), cv2.IMREAD_UNCHANGED).astype('float32')/1000.
            v,u=np.indices(depth.shape,dtype=np.float32)
            points=np.stack(((u-160.)*depth/260.,(v-120.)*depth/260.,depth),axis=-1).reshape(-1,3)
            cloud=PointCloud2();cloud.header=deepcopy(rgb.header);cloud.height,cloud.width=1,len(points)
            cloud.fields=[PointField(name=name,offset=j*4,datatype=7,count=1) for j,name in enumerate(('x','y','z'))]
            cloud.point_step,cloud.row_step=12,len(points)*12;cloud.is_dense=True
            cloud.data=array('B',np.asarray(points,dtype='<f4').tobytes())
            # A stopped depth/pointcloud source produces no metric observation.
            # Keep sending RGB/pose loss; preserve the map until geometry returns.
            if phase != 'depth_gap': clouds.publish(cloud)
            wait(lambda: gateway._active is not None and gateway._active['process']._process is not None, 40.)
            if not subscriptions:
                native=gateway._active['process'];pid=native._process.pid
                subscriptions=[producer.create_subscription(Info,native.namespace+'/info',info,qos),
                    producer.create_subscription(MapGraph,native.namespace+'/mapGraph',graph,qos),
                    producer.create_subscription(SensorData,native.namespace+'/sensor_data',sensor,qos)]
            # Native algorithm rejection may emit no Info. Record that absence
            # and keep delivering observations, rather than treating it as tracking.
            info_deadline=time.monotonic()+(40. if i==0 else 4.)
            wait(lambda: capture in infos or time.monotonic()>=info_deadline, 45. if i==0 else 8.)
            if gateway._active['process']._process.pid != pid or len(gateway._segments)!=1:
                raise RuntimeError('native process/segment changed')
            record=dict(frame=i,phase=phase,stamp_ns=capture,truth_x=float(truth_x),external_pose=has_pose,
                sent_monotonic=sent,
                truth_position=(origin+np.array([float(truth_x),0.,0.])).tolist(),
                native=deepcopy(infos.get(capture)),images=inputs.get(capture),status=gateway.status())
            records.append(record)
            with (output/'info.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            print(json.dumps(dict(frame=i,phase=phase,ref=record['native']['ref'] if record['native'] else None,
                recovery=record['status']['native_recovery'])),flush=True)
        gateway.close()
        artifacts=gateway.artifacts()
        identity,revision,_=save_rtabmap_revision(output/'archive',artifacts,settings,'offline visual recovery',None,'replay')
        reopened=load_map_snapshots(output/'archive',identity,revision,max_points=settings['preview_max_points'])
        expected=[s['snapshot'] for s in artifacts['segments'] if s['snapshot'] is not None]
        if [s['packet'] for s in reopened] != expected:
            raise RuntimeError('saved optimized preview changed on reopening')
        with sqlite3.connect(artifacts['segments'][0]['database']) as db:
            links=db.execute('SELECT from_id,to_id,type FROM Link').fetchall()
            nodes=db.execute('SELECT COUNT(*) FROM Node').fetchone()[0]
        (output/'graph.json').write_text(json.dumps(graphs))
        (output/'native-info.json').write_text(json.dumps(infos))
        result=dict(native_pid=pid,segments=len(artifacts['segments']),database_nodes=nodes,links=links,
            map_id=identity,revision=revision,scene_origin=origin.tolist(),
            archive_reopened_identically=True,statistics=artifacts['statistics'])
        (output/'result.json').write_text(json.dumps(result,indent=2))
    finally:
        gateway.close()
        executor.shutdown(timeout_sec=5.);spin.join(timeout=5.)
        gateway.destroy_node();producer.destroy_node();rclpy.shutdown()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene');parser.add_argument('output');parser.add_argument('--resume-run')
    args=parser.parse_args();run(args.scene,args.output,args.resume_run)
