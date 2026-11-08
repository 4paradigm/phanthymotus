"""Synthetic depth DDS contract check in a network-isolated container."""
import json
import time
import zlib
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from plugins.pointcloud import PointCloudPerceptionPlugin

rclpy.init()
executor=SingleThreadedExecutor()
source=rclpy.create_node('synthetic_depth_source')
executor.add_node(source)
results=[]
try:
    for kind,typ in [('zlib',CompressedImage),('z16',Image)]:
        topic='/synthetic_validation/'+kind
        fmt='image/depth-zlib' if kind=='zlib' else 'image/depth-z16'
        declaration=dict(schema='motus.camera/1',topic=topic,format=fmt,id='synthetic_optical',
            source='derived-from-K',distortion_model='pinhole',width=640,height=480,
            K=[400.,0.,320.,0.,400.,240.,0.,0.,1.],
            vendor=dict(depth_unit='mm',depth_source='measured',timestamp_semantics='source_capture_mapped'))
        plugin=PointCloudPerceptionPlugin({'metric_cloud':True,'fps':20},'synthetic',executor)
        publisher=source.create_publisher(typ,topic,qos_profile_sensor_data)
        received=[]
        subscription=source.create_subscription(PointCloud2,topic+'/pointcloud_metric',received.append,qos_profile_sensor_data)
        try:
            result=plugin.dispatch('pointcloud',dict(action='start',input_topic=topic,camera_info=[declaration]))
            assert result['state']=='running', result
            stamp=time.time_ns()  # timestamp at synthetic sample generation, not receive time
            msg=typ();msg.header.frame_id='synthetic_optical'
            msg.header.stamp.sec=stamp//1000000000;msg.header.stamp.nanosec=stamp%1000000000
            if kind=='zlib':
                msg.format='16UC1; compressedDepth zlib'
                msg.data=zlib.compress(np.full((480,640),2000,dtype='<u2').tobytes())
            else:
                msg.height=480;msg.width=640;msg.encoding='16UC1';msg.step=640*2
                msg.data=np.full((480,640),2000,dtype='<u2').tobytes()
            deadline=time.monotonic()+8;next_send=0
            while not received and time.monotonic()<deadline:
                if time.monotonic()>=next_send: publisher.publish(msg);next_send=time.monotonic()+.2
                executor.spin_once(timeout_sec=.05)
            assert received, kind+' received no metric cloud'
            cloud=received[0]
            assert cloud.header==msg.header
            xyz=np.frombuffer(cloud.data,dtype='<f4').reshape(-1,3)
            np.testing.assert_allclose(xyz[0],[-1.6,-1.2,2.],atol=1e-6)
            assert np.all(xyz[:,2]==2.)
            info=plugin.dispatch('pointcloud',{'action':'info','instance_id':topic})
            assert info['topic_in']==[{'topic':topic,'format':fmt}]
            results.append(dict(format=fmt,points=len(xyz),capture_header_preserved=True,metric_geometry=True))
        finally:
            plugin.dispatch('pointcloud',{'action':'stop'})
            source.destroy_subscription(subscription);source.destroy_publisher(publisher)
        assert executor.get_nodes()==[source]
    print(json.dumps(dict(result='passed',synthetic=True,dds_transport_tested=True,formats=results)))
finally:
    executor.remove_node(source);source.destroy_node();executor.shutdown();rclpy.shutdown()
