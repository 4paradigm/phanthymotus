"""Optional real ROS2/MCP lifecycle integration, no robot or host network.

Run inside isolated Perception image with pinned YuNet model:
  TEST_MOSAIC_YUNET=/models/mosaic_smooth/yunet.onnx \
  PYTHONPATH=perception python3 -B -m unittest \
    discover -s perception/tests -p test_mosaic_smooth_ros_integration.py -v

All topics are synthetic; the test never interacts with hardware or Core.
"""
import json
import os
import threading
import time
import unittest

import numpy as np
import cv2

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from utils.qos import CAMERA_QOS
    from plugins.mosaic_smooth import MosaicSmoothDemoPlugin
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, "ROS2 dependencies unavailable")
class TestIsolatedMcpRosLifecycle(unittest.TestCase):
    def test_synthetic_camera_and_stop(self):
        model=os.environ.get("TEST_MOSAIC_YUNET", "")
        if not os.path.isfile(model):
            self.skipTest("Pinned YuNet model not provisioned")
        rclpy.init()
        executor=MultiThreadedExecutor(num_threads=3)
        sender=Node("s_mosaic_pr_test_source")
        executor.add_node(sender)
        plugin=MosaicSmoothDemoPlugin(
            {"model_path":model},executor)
        name="pr_smooth_demo_test"
        source="/s_mosaic_pr_camera"
        topics=plugin.dispatch("mosaic_smooth_demo",{
            "action":"info","instance_id":name,"input_topic":source,
        })["topic_out"]
        events={"images":[],"statuses":[]}
        lock=threading.Lock()
        def on_image(msg):
            with lock:events["images"].append(
                (msg.header.stamp.sec,msg.header.stamp.nanosec,bytes(msg.data))
            )
        def on_status(msg):
            try:obj=json.loads(msg.data)
            except (ValueError,TypeError):return
            with lock:events["statuses"].append(obj)
        img_sub=sender.create_subscription(
            CompressedImage,topics[0]["topic"],on_image,CAMERA_QOS)
        status_sub=sender.create_subscription(
            String,topics[1]["topic"],on_status,CAMERA_QOS)
        publisher=sender.create_publisher(CompressedImage,source,CAMERA_QOS)
        spinning=threading.Thread(target=executor.spin,daemon=True)
        spinning.start()
        try:
            cfg=plugin.dispatch("mosaic_smooth_demo",{
                "action":"config","instance_id":name,
                "config":{"consent_acknowledged":True},
            })
            self.assertEqual(cfg["state"],"configured")
            start=plugin.dispatch("mosaic_smooth_demo",{
                "action":"start","instance_id":name,"input_topic":source,
            })
            self.assertIn(start["state"],("loading","running"))
            rng=np.random.default_rng(11)
            frame=rng.integers(0,255,(360,640,3),dtype=np.uint8)
            ok,buf=cv2.imencode(".jpg",frame)
            self.assertTrue(ok)
            source_bytes=buf.tobytes()
            # Wait for detector worker startup without dropping test frames.
            ready=False
            for _ in range(100):
                if plugin.dispatch("mosaic_smooth_demo",{
                    "action":"info","instance_id":name
                })["state"]=="running":
                    ready=True;break
                time.sleep(.05)
            self.assertTrue(ready,"Card never entered running state")
            # Only a synthetic topic, with distinct monotonic source stamps.
            for _ in range(90):
                now=time.time_ns()
                msg=CompressedImage()
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(now,10**9)
                msg.header.frame_id="synthetic"
                msg.format="jpeg"
                msg.data=source_bytes
                publisher.publish(msg)
                time.sleep(1/30)
            time.sleep(.5)
            info=plugin.dispatch("mosaic_smooth_demo",{
                "action":"info","instance_id":name
            })
            with lock:
                delivered=list(events["images"])
                statuses=list(events["statuses"])
            self.assertGreater(len(delivered),5)
            self.assertGreater(info["metrics"]["published"],5)
            self.assertGreater(info["effective_output_fps"],3)
            self.assertGreater(len(statuses),1)
            self.assertTrue(all(row[2]!=source_bytes for row in delivered))
            self.assertTrue(all(not s["privacy_guaranteed"]
                                for s in statuses))
            self.assertTrue(all(k=="jpeg" for k in ["jpeg"]))
            last_stamp=max((sec*10**9+ns for sec,ns,_ in delivered))
            self.assertGreater(last_stamp,0)
            stopped=plugin.dispatch("mosaic_smooth_demo",{
                "action":"stop","instance_id":name
            })
            self.assertEqual(stopped["state"],"idle")
            with lock:count_at_stop=len(events["images"])
            time.sleep(.4)
            with lock:self.assertEqual(len(events["images"]),count_at_stop)
        finally:
            plugin.shutdown()
            sender.destroy_subscription(img_sub)
            sender.destroy_subscription(status_sub)
            sender.destroy_publisher(publisher)
            executor.remove_node(sender)
            sender.destroy_node()
            executor.shutdown(timeout_sec=3)
            rclpy.shutdown()
            spinning.join(timeout=2)


if __name__=="__main__":
    unittest.main()
