"""ActuCore adapter for a robot-independent, VOP-driven head policy."""
from __future__ import annotations
import copy
import json
import threading
from .policy import Policy, require, topic

DEFAULT_TOPIC = '/actucore/headfollow/cmd'


def qos():
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)


class HeadFollowPlugin:
    PREFIX = 'headfollow'

    def __init__(self, plugin_cfg, executor, namespace=''):
        self._executor = executor
        self._topic = topic((plugin_cfg or {}).get('topic', DEFAULT_TOPIC))
        self._lock = threading.RLock()
        self._lifecycle = threading.RLock()
        self._generation = 0
        self._node = self._publisher = self._timer = None
        self._policy = None
        self._running = False
        self._detections = self._feedback = None
        self._input = ''
        self._descriptor = {}

    def get_tools(self):
        return [{'name': self.PREFIX, 'type': 'processor', 'multiInstance': False,
            'description': '人物视觉跟随：接 VOP 检测和支持头部接口的驱动；执行后跟随人物，确认无人后回正。只控制实际可动头部或云台，不移动底盘或机械臂。',
            'inputSchema': {'type':'object', 'properties': {
                'action': {'type':'string', 'enum':['follow','start','stop','info'], 'default':'follow'},
                'input_topic': {'type':'string'}, 'input_topics': {'type':'array','items':{'type':'string'}},
                'control_interface': {'type':'object'}}, 'required':['action'],
                'x-action-params': {'follow':{'params':[], 'description':'跟随画面中的人物；确认无人后回正'}},
                'x-resource': ['head']},
            'topic_in': [{'format':'data/json', 'desc':'VOP 人物检测结果'}],
            'topic_out': [{'topic':self._topic, 'format':'control/joint', 'desc':'经过协商的 motus.control/1 头部目标'}]}]

    def start(self):
        pass  # Bundle startup never creates a policy session or sends a command.

    def stop(self):
        self._stop()

    def dispatch(self, name, args):
        action = args.get('action') or name
        try:
            if action == 'start':
                return self._start(args)
            if action == 'stop':
                return self._stop()
            if action == 'info':
                return self._info()
            if action == 'follow':
                with self._lock:
                    require(self._running and self._policy is not None, '先连接 VOP 和头部驱动并启动项目')
                    if not self._policy.enabled:
                        self._policy.enable(self._detections, self._feedback)
                return self._info()
            return None
        except (ValueError, TypeError, KeyError) as exc:
            return {'state':'error', 'error':str(exc)[:240], 'message':str(exc)[:240],
                    'motion_sent_by_this_call':False}

    def _start(self, args):
        with self._lifecycle:
            return self._start_owned(args)

    def _start_owned(self, args):
        descriptor = args.get('control_interface') or {}
        policy = Policy(descriptor)  # Reject wrong/non-head outputs before ROS is touched.
        inputs = args.get('input_topics') or []
        require(isinstance(inputs, list) and all(isinstance(v, str) for v in inputs), '检测输入话题无效')
        if args.get('input_topic'):
            inputs = [args['input_topic'], *inputs]
        inputs = list(dict.fromkeys(topic(v) for v in inputs))
        expected = descriptor['head_tracking']['camera_topic'] + '/objects'
        require(inputs == [expected], 'VOP 必须来自该头部相机：' + expected)
        with self._lock:
            if self._running:
                require(self._input == expected and self._descriptor == descriptor,
                        '接线或头部接口已变化；先停用项目再重新连接')
                return self._info()
        import rclpy
        from std_msgs.msg import String
        self._generation += 1
        generation = self._generation
        node = rclpy.create_node('actucore_headfollow')
        try:
            pub = node.create_publisher(String, self._topic, qos())
            node.create_subscription(String, expected, lambda msg:self._receive(msg,'detections',generation), qos())
            node.create_subscription(String, descriptor['head_tracking']['feedback_topic'], lambda msg:self._receive(msg,'feedback',generation), qos())
            timer = node.create_timer(max(.075,1.0/descriptor['rate']['max_hz']), self._tick)
            with self._lock:
                self._node, self._publisher, self._timer = node, pub, timer
                self._policy = policy
                self._descriptor = copy.deepcopy(descriptor)
                self._input = expected
                self._detections = self._feedback = None
                self._running = True
            self._executor.add_node(node)
        except Exception:
            with self._lock:
                self._running = False
                self._node = self._publisher = self._timer = None
            node.destroy_node()
            raise
        return self._info()

    def _stop(self):
        with self._lifecycle:
            return self._stop_owned()

    def _stop_owned(self):
        with self._lock:
            self._generation += 1
            self._running = False
            if self._policy:
                self._policy.stop()
            node, self._node = self._node, None
            self._publisher = self._timer = None
        if node is not None:
            self._executor.remove_node(node)
            node.destroy_node()
        return self._info()

    def _receive(self, message, kind, generation=None):
        try:
            require(isinstance(message.data, str) and len(message.data) <= 256 * 1024, '输入过大')
            data = json.loads(message.data)
        except (ValueError, TypeError, AttributeError):
            data = None
        with self._lock:
            if not self._running or generation is not None and generation != self._generation:
                return
            if kind == 'detections':
                self._detections = data
            else:
                self._feedback = data

    def _on_detections(self, message):
        self._receive(message, 'detections')

    def _on_feedback(self, message):
        self._receive(message, 'feedback')

    def _tick(self):
        with self._lock:
            if not self._running or not self._policy or not self._policy.enabled:
                return
            msg = self._policy.step(self._detections, self._feedback)
            if msg is not None:
                try:
                    from std_msgs.msg import String
                    message = String()
                    message.data = json.dumps(msg, allow_nan=False)
                    self._publisher.publish(message)
                except Exception as exc:
                    # seq/pending already recorded: never re-publish an uncertain command.
                    self._policy.fail('头部目标发布结果不明：' + str(exc)[:160])

    def _info(self):
        with self._lock:
            state = self._policy.info() if self._policy else {'active':False, 'state':'idle'}
            return {**state, 'running':self._running, 'input_topic':self._input,
                    'control_interface':copy.deepcopy(self._descriptor),
                    'topic_out':self.get_tools()[0]['topic_out'],
                    'stop_note':'停用阻止后续目标；已受理的转头仍可能完成，不是硬件急停'}
