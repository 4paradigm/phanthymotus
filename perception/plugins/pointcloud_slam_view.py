"""Publish one archived map segment on the stable SLAM preview topic.

Selecting a saved map changes the contents of the same mapping output.
Different RTAB-Map odometry segments have independent map coordinates; do not
silently combine or replay them sequentially as if they were one map.
"""
from copy import deepcopy
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import UInt8MultiArray

from .pointcloud_slam_preview import PREVIEW_TOPIC, historical_mapping

_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                  history=HistoryPolicy.KEEP_LAST, depth=1,
                  durability=DurabilityPolicy.VOLATILE)


class SavedMapView(Node):
    def __init__(self, map_id, revision, name, snapshots, *, interval_s=1):
        super().__init__('pointcloud_slam_map_view')
        self._lock = threading.Lock()
        self._closed = False
        self._timer = None
        self._interval = interval_s
        self._outputs = []
        self._publisher = None
        self._packet = None
        self.segment_count = len(snapshots)
        self.selected_segment_id = None
        try:
            if snapshots:
                # The latest map segment is selected for display. Independent
                # segments have no verified common global frame.
                item = snapshots[-1]
                self.selected_segment_id = item['segment_id']
                metadata = deepcopy(item['metadata'])
                metadata['map_view'] = {
                    'schema': 'motus.map-view/1', 'scope': 'history',
                    'map_id': map_id, 'name': name, 'revision': revision,
                    'segment_id': item['segment_id'], 'frame_id': item['frame_id'],
                    'segment_count': self.segment_count,
                }
                self._outputs = [{
                    'topic': PREVIEW_TOPIC, 'format': 'sensor/mapping',
                    'desc': 'Live or saved RTAB-Map preview',
                    'metadata': metadata,
                }]
                self._packet = historical_mapping(item['packet'])
                self._publisher = self.create_publisher(UInt8MultiArray, PREVIEW_TOPIC, _QOS)
        except Exception:
            self.destroy_node()
            raise

    def outputs(self):
        with self._lock:
            return deepcopy(self._outputs)

    def rename(self, name):
        with self._lock:
            for output in self._outputs:
                output['metadata']['map_view']['name'] = name

    def start(self):
        self._publish()
        if self._publisher is not None and self._timer is None:
            self._timer = self.create_timer(self._interval, self._publish)

    def _publish(self):
        with self._lock:
            if self._closed or self._publisher is None:
                return
            message = UInt8MultiArray()
            message.data = self._packet
            self._publisher.publish(message)

    def close(self):
        with self._lock:
            self._closed = True
            if self._timer:
                self._timer.cancel()
                self.destroy_timer(self._timer)
                self._timer = None
