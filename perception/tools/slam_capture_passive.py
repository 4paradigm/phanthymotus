#!/usr/bin/env python3
"""Bounded passive ROS recording; subscribes only, never starts sensors or motion."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def run(specification, output, seconds=20., limit=100):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.utilities import get_message
    specification=json.loads(Path(specification).read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rclpy.init();node=rclpy.create_node('slam_passive_record')
    counts={topic:0 for topic in specification}
    subscriptions=[]
    with (output/'records.jsonl').open('w') as records:
        def receive(topic,message):
            if counts[topic]>=limit:return
            payload=serialize_message(message)
            if len(payload)>16*1024**2:return
            stamp=getattr(getattr(message,'header',None),'stamp',None)
            filename=str(len(subscriptions))+'_'+str(sum(counts.values()))+'.cdr'
            (output/filename).write_bytes(payload)
            records.write(json.dumps(dict(topic=topic,type=specification[topic],file=filename,
                stamp_ns=None if stamp is None else stamp.sec*10**9+stamp.nanosec,
                receipt_ns=time.time_ns(),sha256=hashlib.sha256(payload).hexdigest()))+'\n')
            counts[topic]+=1
        for topic,kind in specification.items():
            qos=QoSProfile(depth=100,durability=DurabilityPolicy.TRANSIENT_LOCAL) if topic.endswith('/tf_static') else qos_profile_sensor_data
            subscriptions.append(node.create_subscription(get_message(kind),topic,
                lambda msg,t=topic:receive(t,msg),qos))
        started=time.monotonic()
        while time.monotonic()-started<seconds and not all(v>=limit for v in counts.values()):
            rclpy.spin_once(node,timeout_sec=.1)
    (output/'summary.json').write_text(json.dumps(dict(topics=specification,counts=counts,
        elapsed_s=time.monotonic()-started,passive=True),indent=2))
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('specification');p.add_argument('output')
    p.add_argument('--seconds',type=float,default=20.);p.add_argument('--limit',type=int,default=100)
    a=p.parse_args();run(a.specification,a.output,a.seconds,a.limit)
