"""Publish bounded archived map snapshots without starting a mapping backend."""
from copy import deepcopy
import hashlib
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import UInt8MultiArray

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
        self._packets = []
        try:
            for item in snapshots:
                suffix = hashlib.sha256(item['segment_id'].encode()).hexdigest()[:16]
                if suffix[0].isdigit():
                    suffix = 's' + suffix
                topic = '/pointcloud_slam/maps/m' + map_id + '/history/' + suffix
                metadata = deepcopy(item['metadata'])
                metadata['map_view'] = {'schema': 'motus.map-view/1', 'scope': 'history',
                    'map_id': map_id, 'name': name, 'revision': revision,
                    'segment_id': item['segment_id'], 'frame_id': item['frame_id'],
                    'plane': metadata.get('map_plane'),
                    'actions': {key: key for key in ('list_tags', 'tag_place', 'untag_place')}}
                self._outputs.append({'topic': topic, 'format': 'sensor/mapping',
                    'desc': 'Saved map snapshot; marker is the last captured frame pose',
                    'segment_id': item['segment_id'], 'metadata': metadata})
                self._packets.append((self.create_publisher(UInt8MultiArray, topic, _QOS), item['packet']))
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
        if self._packets and self._timer is None:
            self._timer = self.create_timer(self._interval, self._publish)

    def _publish(self):
        with self._lock:
            if self._closed:
                return
            for publisher, packet in self._packets:
                message = UInt8MultiArray()
                message.data = packet
                publisher.publish(message)

    def close(self):
        with self._lock:
            self._closed = True
            if self._timer:
                self._timer.cancel()
                self.destroy_timer(self._timer)
                self._timer = None
