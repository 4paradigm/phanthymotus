"""Isolated synthetic DDS/paging/restore check. Never use the robot ROS domain."""
import json
import struct
import tempfile
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data as qos
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray
import plugins.pointcloud_slam_plugin as module

rclpy.init()
executor = SingleThreadedExecutor()
source = rclpy.create_node('synthetic_paged_source')
executor.add_node(source)
odom_pub = source.create_publisher(String, '/synthetic/odom', qos)
cloud_pub = source.create_publisher(PointCloud2, '/synthetic/cloud', qos)
live, archived, statuses = [], [], []
source.create_subscription(UInt8MultiArray, '/synthetic/cloud/slam_odom_preview',
                           lambda msg: live.append(bytes(msg.data)), qos)
source.create_subscription(String, '/synthetic/cloud/slam_odom_preview_status',
                           lambda msg: statuses.append(json.loads(msg.data)), qos)


def wait_for(condition, seconds=10):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=.02)
    assert condition(), 'timed out waiting for synthetic DDS state'


options = {'descriptor_voxel_m': .1, 'min_separation_s': 10.,
           'candidate_separation_s': 3., 'max_per_frame': 1, 'max_candidates': 20,
           'voxel_m': .08, 'max_distance_m': .3, 'odom_sigmas': [.1]*6,
           'loop_sigmas': [.05]*6, 'huber_k': 1.,
           'thresholds': {'min_fitness': .6, 'max_rmse_m': .05, 'min_correspondences': 30,
                          'max_cycle_translation_m': .03, 'max_cycle_angle_deg': 2.}}
config = dict(cloud_frame='synthetic_camera', child_frame='synthetic_body', clock_id='synthetic_clock',
              child_from_cloud=np.eye(4).tolist(), clock_relation_verified=True, extrinsic_calibrated=True,
              max_gap_ms=200, max_age_s=2, max_pending=2, max_cloud_points=10,
              max_frames=3, max_points=30, max_measurement_age_ms=1000, max_future_skew_ms=20,
              preview_enabled=True, preview_backend='geometry_experimental', preview_geometry=options,
              preview_interval_s=.1, preview_max_points=40, archive_name='synthetic_paged',
              persistent_history={'max_disk_bytes':1000000, 'reserve_bytes':0,
                                  'min_translation_m':.1, 'min_rotation_deg':10.})
args = {'action':'start', 'input_topics':['/synthetic/odom','/synthetic/cloud'],
        'input_bindings':{'/synthetic/odom':'state/odom','/synthetic/cloud':'sensor/pointcloud2'},
        'odom_interfaces':{'/synthetic/odom':{'schema':'motus.odom/1',
                           'slam_pose':{'available':True,'profile':'motus_slam/1'}}}}
plugin = None
with tempfile.TemporaryDirectory(prefix='synthetic-paged-') as directory:
    module._ARCHIVE_DIR = directory
    try:
        plugin = module.PointCloudSlamPlugin(config, executor)
        assert plugin.dispatch('pointcloud_slam', args)['state'] == 'running'
        wait_for(lambda: odom_pub.get_subscription_count() and cloud_pub.get_subscription_count())
        for i in range(12):
            ms = source.get_clock().now().nanoseconds // 1000000
            sample = {'schema':'motus.odom/1', 'units':{'length':'m'}, 'stamp_ms':ms,
                      'pose':{'position':[i*.2,0,0], 'quaternion':[0,0,0,1],
                              'frame_id':'synthetic_odom','child_frame_id':'synthetic_body','covariance':None},
                      'vendor':{'motus_slam':{'status':'tracking','segment_id':'synthetic_a',
                                             'clock_id':'synthetic_clock','stamp_source':'measurement'}}}
            odom_pub.publish(String(data=json.dumps(sample)))
            wait_for(lambda: bool(plugin._node.stream._poses) and plugin._node.stream._poses[-1].stamp_ms == ms)
            cloud = PointCloud2()
            cloud.header.frame_id='synthetic_camera'
            cloud.header.stamp.sec=ms//1000; cloud.header.stamp.nanosec=(ms%1000)*1000000
            cloud.width=1; cloud.height=1; cloud.point_step=12; cloud.row_step=12
            cloud.fields=[PointField(name=n,offset=j*4,datatype=PointField.FLOAT32,count=1) for j,n in enumerate('xyz')]
            cloud.data=np.array([[1,2,3]],dtype='<f4').tobytes()
            cloud_pub.publish(cloud)
            wait_for(lambda: plugin._node.stream.store._frame_count == i+1)
        wait_for(lambda: len(plugin._node._history_snapshots.get('synthetic_a',{}).get('poses',[])) == 12 and live)
        assert any(s.get('submaps',0)>1 for s in statuses)
        assert plugin.dispatch('pointcloud_slam',{'action':'save_map'})['state']=='saved'
        plugin.stop()
        plugin=module.PointCloudSlamPlugin({**config,'restore_archive':True},executor)
        assert plugin.dispatch('pointcloud_slam',args)['state']=='running'
        assert plugin._node.stream.store._frame_count==12 and plugin._node.stream.store.requires_new_segment
        topics=plugin.dispatch('pointcloud_slam',{'action':'info'})['topic_out']
        history=next(t['topic'] for t in topics if '/history/' in t['topic'])
        source.create_subscription(UInt8MultiArray,history,lambda msg:archived.append(bytes(msg.data)),qos)
        wait_for(lambda: bool(archived))
        length=struct.unpack('<I',archived[-1][:4])[0]; meta=json.loads(archived[-1][4:4+length])
        assert meta['segment_id']=='synthetic_a' and meta['state']=='historical_snapshot'
        assert not plugin._node.stream._poses
        plugin.stop()
        assert executor.get_nodes()==[source]
        print(json.dumps({'result':'passed','synthetic':True,'dds_transport_tested':True,
                          'keyframes':12,'window_limit':3,'live_previews':len(live),
                          'restored_history_previews':len(archived),'restored_requires_new_segment':True}))
    finally:
        if plugin: plugin.stop()
        executor.remove_node(source); source.destroy_node(); executor.shutdown(); rclpy.shutdown()
