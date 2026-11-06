"""Synthetic real-rclpy lifecycle check; never connect this to live sensors."""
import json
import tempfile
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
import plugins.pointcloud_slam_plugin as module

rclpy.init()
executor = SingleThreadedExecutor()
source = rclpy.create_node('synthetic_source_sentinel')
executor.add_node(source)
cfg = dict(cloud_frame='synthetic_camera', child_frame='synthetic_body', clock_id='synthetic_clock',
           clock_relation_verified=True, extrinsic_calibrated=True, child_from_cloud=np.eye(4).tolist(),
           max_gap_ms=200, max_age_s=1, max_pending=2, max_cloud_points=10, max_frames=10,
           max_points=100, max_measurement_age_ms=300, max_future_skew_ms=20, archive_name='synthetic')
args = dict(action='start', input_topics=['/synthetic/odom','/synthetic/cloud'],
            input_bindings={'/synthetic/odom':'state/odom','/synthetic/cloud':'sensor/pointcloud2'},
            odom_interfaces={'/synthetic/odom':{'schema':'motus.odom/1','slam_pose':{'available':True,'profile':'motus_slam/1'}}})

def feed(node, ms, segment='a', status='tracking', frame='synthetic_camera'):
    # Synthetic capture clock, explicitly unrelated to robot sensor clocks.
    node._measurement_clock_ns = lambda: ms*1000000
    sample = dict(schema='motus.odom/1', units={'length':'m'}, stamp_ms=ms,
        pose=dict(position=[0,0,0],quaternion=[0,0,0,1],frame_id='synthetic_odom',child_frame_id='synthetic_body',covariance=None),
        vendor={'motus_slam':dict(status=status,segment_id=segment,clock_id='synthetic_clock',stamp_source='measurement')})
    node._odom(String(data=json.dumps(sample)))
    cloud=PointCloud2()
    cloud.header.frame_id=frame
    cloud.header.stamp.sec=ms//1000
    cloud.header.stamp.nanosec=(ms%1000)*1000000
    cloud.height=1;cloud.width=1;cloud.point_step=12;cloud.row_step=12
    cloud.fields=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,name in enumerate(('x','y','z'))]
    cloud.data=np.array([[1,2,3]],dtype='<f4').tobytes()
    node._cloud(cloud)

def count(node): return sum(len(s.frames) for s in node.stream.store.segments)

plugin=None
with tempfile.TemporaryDirectory(prefix='synthetic-slam-') as archive:
    module._ARCHIVE_DIR=archive
    try:
        blocked=module.PointCloudSlamPlugin({**cfg,'extrinsic_calibrated':False},executor)
        assert blocked.dispatch('pointcloud_slam',args)['state']=='error'
        assert executor.get_nodes()==[source]
        plugin=module.PointCloudSlamPlugin(cfg,executor)
        assert plugin.dispatch('pointcloud_slam',args)['state']=='running'
        node=plugin._node
        # Exercise the real ROS clock before synthetic measurement overrides.
        assert node.get_clock().now().nanoseconds > 0
        assert node._measurement_clock_ns() > 0
        feed(node,1000)
        assert count(node)==1
        assert node.stream.store.segments[0].frames[0].cloud_stamp_ns==1000000000
        feed(node,1100,frame='wrong')
        assert count(node)==1
        feed(node,1200,status='lost')
        assert count(node)==1
        feed(node,1300,segment='b')
        assert count(node)==2 and len(node.stream.graph())==2
        from plugins.pointcloud_slam_archive import save_archive
        save_archive(module._archive_path('synthetic'), node.stream.store, [], {}, cfg)
        plugin.stop()
        assert executor.get_nodes()==[source]
        plugin=module.PointCloudSlamPlugin({**cfg,'restore_archive':True},executor)
        assert plugin.dispatch('pointcloud_slam',args)['state']=='running'
        node=plugin._node
        assert count(node)==2 and node.stream.store.requires_new_segment
        feed(node,1400,segment='b')
        assert count(node)==2
        feed(node,1500,segment='c')
        assert count(node)==3 and len(node.stream.graph())==3
        plugin.stop()
        assert executor.get_nodes()==[source]
        print(json.dumps({'result':'passed','synthetic':True,'real_rclpy_nodes':True,'dds_transport_tested':False,'checks':['calibration_gate','start','capture_stamp','wrong_frame','lost','segment_separation','save','stop_preserves_source','restore_requires_new_segment','restart']}))
    finally:
        if plugin: plugin.stop()
        executor.remove_node(source);source.destroy_node();executor.shutdown();rclpy.shutdown()
