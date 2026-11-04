"""ROS input adapter for the experimental, odometry-only SLAM graph."""
import json
import threading
import time

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                  history=HistoryPolicy.KEEP_LAST, depth=2,
                  durability=DurabilityPolicy.VOLATILE)


def decode_cloud(msg, *, max_points):
    """Read finite XYZ float32 points, including padded and organized clouds."""
    try:
        stamp = msg.header.stamp
        sec, nsec = stamp.sec, stamp.nanosec
        frame = msg.header.frame_id
        width, height, step, row = msg.width, msg.height, msg.point_step, msg.row_step
        if (type(sec) is not int or type(nsec) is not int or sec < 0 or
                not 0 <= nsec < 1_000_000_000 or sec * 1_000_000_000 + nsec <= 0 or
                not isinstance(frame, str) or not frame.strip()):
            raise ValueError('cloud needs capture timestamp and frame')
        if (any(type(v) is not int or v <= 0 for v in (width, height, step, row)) or
                width * height > max_points or row < width * step or len(msg.data) < row * height):
            raise ValueError('invalid cloud dimensions or capacity')
        fields = {}
        for field in msg.fields:
            if field.name in ('x', 'y', 'z'):
                if (field.name in fields or field.datatype != PointField.FLOAT32 or
                        field.count != 1 or type(field.offset) is not int or
                        field.offset < 0 or field.offset + 4 > step):
                    raise ValueError('unsupported XYZ fields')
                fields[field.name] = field.offset
        offsets = sorted(fields.values())
        if (set(fields) != {'x', 'y', 'z'} or
                any(b - a < 4 for a, b in zip(offsets, offsets[1:]))):
            raise ValueError('missing or overlapping XYZ fields')
        raw = memoryview(msg.data)
        dtype = '>f4' if msg.is_bigendian else '<f4'
        points = np.stack([np.ndarray((height, width), dtype=dtype, buffer=raw,
                                     offset=fields[key], strides=(row, step)).reshape(-1)
                           for key in ('x', 'y', 'z')], axis=1)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError('cloud has no finite points')
        return sec * 1_000_000_000 + nsec, frame, points
    except (AttributeError, TypeError, BufferError) as error:
        raise ValueError('invalid PointCloud2') from error


class SlamInputNode(Node):
    """Subscribe only to explicitly resolved topics; does not control producers."""

    def __init__(self, odom_topic, cloud_topic, stream, *, clock=time.monotonic):
        super().__init__('pointcloud_slam_input')
        self.stream = stream
        self._clock = clock
        self._lock = threading.Lock()
        self.create_subscription(String, odom_topic, self._odom, _QOS)
        self.create_subscription(PointCloud2, cloud_topic, self._cloud, _QOS)

    def _odom(self, msg):
        try:
            sample = json.loads(msg.data)
        except (TypeError, ValueError):
            sample = None
        with self._lock:
            self.stream.odom(sample, now=self._clock())

    def _cloud(self, msg):
        try:
            stamp, frame, points = decode_cloud(msg, max_points=self.stream.max_cloud_points)
            with self._lock:
                self.stream.cloud(stamp, frame, points, now=self._clock())
        except ValueError as error:
            with self._lock:
                self.stream.last_error = str(error)

    def optimize_odometry(self, *, sigmas):
        """Experimental explicit invocation; produces no published map or loops."""
        from .pointcloud_slam_backend import optimize_odometry_graph
        with self._lock:
            graphs = self.stream.graph()
        return [{'segment_id': graph['segment_id'],
                 'poses': optimize_odometry_graph(graph, sigmas=sigmas)}
                for graph in graphs]
