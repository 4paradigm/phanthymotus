"""Opt-in experimental input card: odometry graph only, no SLAM map output."""
import threading

from .pointcloud_slam_ros import SlamInputNode
from .pointcloud_slam_stream import SlamStream
from utils.ros_lifecycle import dispose_node


TOOLS = [{'name': 'pointcloud_slam', 'type': 'processor', 'multiInstance': False,
          'description': '实验性点云与里程计输入及分段图；尚不输出 SLAM 地图或回环',
          'inputSchema': {'type': 'object', 'properties': {
              'action': {'type': 'string', 'enum': ['start', 'stop', 'info']},
              'input_topics': {'type': 'array', 'items': {'type': 'string'}},
              'input_bindings': {'type': 'object'}, 'odom_interfaces': {'type': 'object'}}, 'required': ['action']},
          'topic_in': [{'format': 'state/odom'}, {'format': 'sensor/pointcloud2'}],
          'topic_out': []}]


class PointCloudSlamPlugin:
    PREFIX = 'pointcloud_slam'

    def __init__(self, config, executor):
        self._config = dict(config or {})
        self._executor = executor
        self._node = None
        self._topics = {}
        self._lock = threading.RLock()

    def get_tools(self):
        return TOOLS

    def start(self):
        pass  # Canvas start supplies the two resolved inputs.

    def stop(self):
        with self._lock:
            node, self._node = self._node, None
            self._topics = {}
            if node is not None:
                dispose_node(self._executor, node, label='pointcloud_slam')

    def dispatch(self, name, args):
        action = args.get('action', name)
        if action == 'stop':
            self.stop()
            return {'state': 'idle'}
        if action == 'info':
            with self._lock:
                node = self._node
                input_status = node.status() if node else {'odometry': 'waiting'}
                return {'state': 'running' if node else 'idle',
                        'desc': '实验输入与里程计图；无回环和地图输出',
                        'topic_in': [{'topic': topic, 'format': fmt}
                                     for topic, fmt in self._topics.items()],
                        'topic_out': [],
                        'input_status': input_status,
                        'last_error': input_status.get('last_error', '')}
        if action != 'start':
            return None
        with self._lock:
            if self._node is not None:
                return {'state': 'error', 'message': 'already running'}
            if self._executor is None:
                return {'state': 'error', 'message': 'ROS executor unavailable'}
            bindings, topics = args.get('input_bindings'), args.get('input_topics')
            if (not isinstance(bindings, dict) or not isinstance(topics, list) or
                    len(topics) != 2 or any(not isinstance(t, str) or not t.strip() for t in topics) or
                    any(not isinstance(f, str) for f in bindings.values()) or
                    len(set(topics)) != 2 or
                    set(bindings) != set(topics) or
                    set(bindings.values()) != {'state/odom', 'sensor/pointcloud2'}):
                return {'state': 'error', 'message': 'two format-resolved odometry and cloud inputs required'}
            odom_topic = next(t for t, f in bindings.items() if f == 'state/odom')
            interfaces = args.get('odom_interfaces')
            interface = interfaces.get(odom_topic) if isinstance(interfaces, dict) else None
            capability = interface.get('slam_pose') if isinstance(interface, dict) else None
            if (not isinstance(interface, dict) or interface.get('schema') != 'motus.odom/1' or
                    not isinstance(capability, dict) or capability.get('available') is not True or
                    capability.get('profile') != 'motus_slam/1'):
                return {'state': 'error', 'message': 'odometry source lacks declared SLAM pose capability'}
            if (self._config.get('clock_relation_verified') is not True or
                    self._config.get('extrinsic_calibrated') is not True):
                return {'state': 'error', 'message': 'verified clock relation and measured extrinsic required'}
            if (self._config.get('max_measurement_age_ms') is None or
                    self._config.get('max_future_skew_ms') is None):
                return {'state': 'error', 'message': 'measurement age and clock skew limits required'}
            keys = ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud',
                    'max_gap_ms', 'max_age_s', 'max_pending', 'max_cloud_points',
                    'max_frames', 'max_points', 'max_measurement_age_ms', 'max_future_skew_ms')
            try:
                stream = SlamStream(**{key: self._config[key] for key in keys})
                cloud_topic = next(t for t, f in bindings.items() if f == 'sensor/pointcloud2')
                node = SlamInputNode(odom_topic, cloud_topic, stream)
                try:
                    self._executor.add_node(node)
                except Exception:
                    node.destroy_node()
                    raise
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                return {'state': 'error', 'message': str(error)}
            self._topics = dict(bindings)
            self._node = node
            return {'state': 'running', 'message': '实验输入已启动；无回环或地图输出'}
