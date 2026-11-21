"""Real DDS adapter recovery probes using decoded natural source frames.

Not a hardware or full Canvas test; tests both actual sensor_msgs message types.
"""
import argparse
import json
from pathlib import Path
import threading
import time
import sys
import os
import secrets
import socket
import hashlib
import importlib.util
import cv2

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'perception'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--video',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extra-config',default='{}')
    parser.add_argument('--localhost-only',type=int,choices=[0,1],default=0,
                        help='FastDDS uses the explicit loopback whitelist; this runtime needs 0 for discovery')
    args=parser.parse_args();cv2.setNumThreads(2)
    if args.output.exists():raise FileExistsError('preserve prior acceptance evidence')
    profile=args.output.with_suffix('.dds.xml')
    with profile.open('xb') as stream:stream.write((ROOT/'agent-core/deploy/dds-local.xml').read_bytes())
    for _ in range(100):
        domain=120+secrets.randbelow(70);sockets=[]
        try:
            for port in range(7400+250*domain,7400+250*domain+40):
                s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',port));sockets.append(s)
            break
        except OSError:continue
        finally:
            for s in sockets:s.close()
    else:raise RuntimeError('no isolated DDS domain available')
    os.environ.update(ROS_DOMAIN_ID=str(domain),ROS_LOCALHOST_ONLY=str(args.localhost_only),
        FASTRTPS_DEFAULT_PROFILES_FILE=str(profile.resolve()),
        ROS_LOG_DIR=str(args.output.resolve().parent/'adapter-ros-logs'))
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import Image,CompressedImage
    from plugins.face_effects_live.plugin import FaceEffectsLivePlugin
    rclpy.init();executor=MultiThreadedExecutor(num_threads=2)
    spin=threading.Thread(target=executor.spin,daemon=True);spin.start()
    node=Node('natural_replay_adapter_probe');cap=cv2.VideoCapture(str(args.video))
    ok,image=cap.read();cap.release();assert ok
    cfg={'detector_model':str(args.models/'yunet.onnx'),'landmark_model':str(args.models/'osf0.onnx'),
         'max_faces':8,**json.loads(args.extra_config)}
    plugin=FaceEffectsLivePlugin(cfg,executor);results={}
    spec=importlib.util.spec_from_file_location('local_dds_audit',ROOT/'agent-core/src/dds_isolation.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    external_binds=audit._external_dds_binds();assert not external_binds,external_binds
    def await_value(check,seconds=5):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            value=check()
            if value:return value
            time.sleep(.01)
        raise RuntimeError('DDS result timeout; local instance status: '+json.dumps(latest()))
    try:
        for kind,message in [('ros_image',Image),('ros_compressed',CompressedImage)]:
            topic='/natural_adapter_'+kind
            pub=node.create_publisher(message,topic,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
            started=plugin.dispatch('start',{'instance_id':kind,'source_kind':kind,'input_topic':topic})
            assert started['ok'] and started['state']=='running',started
            time.sleep(.5)
            count=0
            def publish(pixels,stamp,invalid=False):
                nonlocal count
                msg=message();msg.header.frame_id='natural-adapter:'+str(count);count+=1
                msg.header.stamp.sec=int(stamp);msg.header.stamp.nanosec=int(stamp%1*1e9)
                if kind=='ros_image':
                    msg.height,msg.width=pixels.shape[:2];msg.step=msg.width*3
                    msg.encoding='unsupported' if invalid else 'bgr8';msg.data=pixels.tobytes()
                else:
                    msg.format='compressedDepth' if invalid else 'jpeg'
                    msg.data=cv2.imencode('.jpg',pixels)[1].tobytes()
                pub.publish(msg)
                return msg.header.frame_id
            def latest():return plugin.dispatch('info',{'instance_id':kind})
            for k in range(10):publish(image,time.time());time.sleep(.05)
            initial=await_value(lambda:latest().get('result'))
            assert initial['size']==[1280,720]
            small=cv2.resize(image,(640,360))
            for k in range(5):publish(small,time.time());time.sleep(.05)
            resized=await_value(lambda:(r if (r:=latest().get('result')) and r['size']==[640,360] else None))
            publish(small,time.time(),invalid=True)
            await_value(lambda:latest().get('input_error')=='invalid ROS image')
            marker=publish(small,1.) # deliberately regress the source clock
            repaired=await_value(lambda:(r if (r:=latest().get('result')) and r['source_metadata'].get('ros_frame_id')==marker else None))
            assert repaired['source_metadata']['timestamp_repaired'] and repaired['scene_reset']
            time.sleep(.4)
            assert not plugin.dispatch('snapshot',{'instance_id':kind})['ok']
            for k in range(5):publish(image,time.time());time.sleep(.05)
            recovered=await_value(lambda:plugin.dispatch('snapshot',{'instance_id':kind}).get('ok'))
            final=latest();stopped=plugin.dispatch('stop',{'instance_id':kind})
            assert stopped['ok']
            node.destroy_publisher(pub)
            results[kind]={'initial_size':initial['size'],'resized':resized['size'],
                'timestamp_repair':True,'scene_reset':True,'invalid_format_rejected':True,
                'stale_snapshot_rejected':True,'stream_resumed':bool(recovered),
                'processed':final['processed'],'timestamp_repairs':final['timestamp_repairs'],'stop_ok':True}
    finally:
        plugin.shutdown();node.destroy_node();executor.shutdown(timeout_sec=3);rclpy.shutdown();spin.join(3)
    args.output.write_text(json.dumps({'schema':'face-effects-real-dds-adapters/1','domain':domain,'results':results,
        'dds_policy':{'localhost_only':args.localhost_only,'profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
                      'external_dds_binds':external_binds},
        'limits':['natural source still repeated for recovery probes','not a physical camera or full Canvas chain']},indent=2)+'\n')
    print(json.dumps(results))


if __name__=='__main__':main()
