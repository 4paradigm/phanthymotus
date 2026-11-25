"""Single-hand camera processor. Only the downstream driver commands hardware."""

import json
import logging
import threading
import time

from .overlay import draw_hand, encode_jpeg
from .retarget import RateLimiter, flexions
from .vision import create_backend

log = logging.getLogger(__name__)
KEYPOINTS = '/actucore/handdrive/keypoints'
VIEW = '/actucore/handdrive/view'


# Fixed skeleton-preview rendering. The canvas hides these knobs (x-show-when
# below) and the values are pinned here, so they cannot drift via config.yaml
# or a config action. The preview is always on for exhibition demos.
VIEW_ALWAYS_ON = True
VIEW_LINE_WIDTH = 5
VIEW_POINT_RADIUS = 8
VIEW_JPEG_QUALITY = 90

# Wire format exposed to the canvas. `control/hand_flex` renders as a
# structured command stream (like navi's control/velocity), not a raw JSON
# blob. Five values are the per-finger curls in FINGER_ORDER below.
FLEX_FORMAT = "control/hand_flex"
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# Fields accepted by the internal `config` action (tests / automation). This
# is deliberately separate from configSchema.properties: the canvas renders
# every declared property in BOTH the tool-level and the per-card dialog, so
# only hand/hand_confidence belong there. Everything else is fixed at deploy
# time but still needs to be switchable in offline tests (backend=mock etc).
CONFIGURABLE = frozenset({
    "hand", "hand_confidence", "backend", "model_dir", "model_path",
    "model_family", "mock_pose", "slew_rate", "rate_hz",
    "keypoint_topic", "view_topic", "draw_view", "line_width",
    "point_radius", "view_jpeg_quality",
    "handedness_source", "min_visibility",
})


class HandLandmarksPlugin:
    PREFIX = 'hand_landmarks'

    def __init__(self, cfg, executor):
        self.cfg = dict(cfg)
        self.executor = executor
        self._lifecycle = threading.RLock()
        self._frame_lock = threading.Lock()
        self._tick_lock = threading.Lock()
        self._frame = None
        # Identity of the last frame actually consumed by _tick(). A camera
        # dropout leaves the last frame in self._frame; without this guard the
        # timer would re-run inference and re-publish it every 1/rate_hz with
        # a FRESH stamp_ms/seq for up to the 2 s stale-frame window, keeping
        # downstream commands artificially alive after the camera went away.
        self._last_processed_frame = None
        self.node = self.backend = None
        self._attached = False
        self.state = 'idle'
        self.error = self.view_error = self.model_status = ''
        self.hand_present = False
        self.published = 0
        self._seq = 0
        self.last_curl = {}
        self.image_topic = ''

    def _outputs(self):
        return [{'topic': self.cfg.get('view_topic', VIEW), 'format': 'image/jpeg'},
                {'topic': self.cfg.get('keypoint_topic', KEYPOINTS), 'format': FLEX_FORMAT}]

    def get_tools(self):
        def field(kind, default, scope='shared', **extra):
            return dict(type=kind, default=default, scope=scope, **extra)

        # Only the two operator-tunable knobs are declared in configSchema.
        # Everything else (backend/model plumbing, topics, rate, preview) is
        # fixed in config.yaml or module constants and must not appear in
        # either the tool-level or the per-card instance config dialog. The
        # canvas renders every property it is given, so omitting them here is
        # the only reliable way to hide them in both dialogs.
        props = {
            'hand': field('string', 'Right', enum=['Right', 'Left', 'either']),
            'hand_confidence': field('number', 0.3),
        }
        return [{'name': self.PREFIX, 'type': 'processor', 'multiInstance': False,
                 'description': '识别单手 21 个关键点，输出五指弯曲量和绿色骨架预览。',
                 'inputSchema': {'type': 'object', 'required': ['action'], 'properties': {
                     'action': {'type': 'string', 'enum': ['info', 'config', 'start', 'stop', 'pause', 'resume']}},
                     'x-hooks': {'on_interrupt_all': {'action': 'pause'}, 'on_interrupt_motion': {'action': 'pause'}}},
                 'configSchema': {'type': 'object', 'properties': props},
                 'topic_in': [{'format': 'image/jpeg'}], 'topic_out': self._outputs()}]

    def dispatch(self, name, args):
        args = args or {}
        action = args.get('action') or name
        if action == 'info':
            return {'state': self.state, 'topic_in': [{'format': 'image/jpeg'}], 'topic_out': self._outputs(),
                    'keypoint_topic': self._outputs()[1]['topic'], 'view_topic': self._outputs()[0]['topic'],
                    'image_topic': self.image_topic, 'hand_detected': self.hand_present,
                    'flexions': self.last_curl, 'published': self.published, 'model_status': self.model_status,
                    'backend': self.cfg.get('backend', 'rtmpose'), 'hand': self.cfg.get('hand', 'Right'),
                    'error': self.error, 'view_error': self.view_error}
        with self._lifecycle:
            if action == 'config':
                if self.state != 'idle':
                    return {'state': 'error', 'error': '请先停止卡片再修改配置'}
                self.cfg.update({k: v for k, v in args.items()
                                 if k in CONFIGURABLE})
                return {'state': self.state, 'config': dict(self.cfg)}
            if action == 'start':
                if self.state == 'idle':
                    try:
                        self._start(args)
                    except Exception as exc:
                        self.shutdown()
                        self.error = str(exc)
                        return {'state': 'error', 'error': self.error}
            elif action == 'stop':
                self.shutdown()
            elif action in ('pause', 'resume'):
                with self._tick_lock:
                    if self.node is not None:
                        self.state = 'paused' if action == 'pause' else 'running'
                        self.limiter.reset()
                        self.prev_time = 0
                        with self._frame_lock:
                            self._frame = None
            else:
                return None
            return {'state': self.state, 'keypoint_topic': self._outputs()[1]['topic']}

    def _start(self, args):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, Image
        from std_msgs.msg import String

        inputs = list(args.get('input_topics') or [])
        if args.get('input_topic'):
            inputs.insert(0, args['input_topic'])
        if not inputs:
            raise ValueError('请连接相机的 image/jpeg 输出')
        rate = float(self.cfg.get('rate_hz', 10))
        if not 0 < rate <= 120:
            raise ValueError('rate_hz must be in (0, 120]')
        self.node = Node('perception_hand_landmarks')
        deadline = time.monotonic() + 3
        while True:
            graph = dict(self.node.get_topic_names_and_types())
            types = (('sensor_msgs/msg/CompressedImage', CompressedImage), ('sensor_msgs/msg/Image', Image))
            selected = next(((t, cls) for t in inputs for ros_type, cls in types
                             if ros_type in graph.get(t, [])), None)
            if selected:
                break
            if time.monotonic() >= deadline:
                raise ValueError(f'输入话题没有发布相机图像：{inputs}')
            time.sleep(0.1)
        self.image_topic, image_type = selected
        self.model_status = '准备手部模型 …'
        self.backend = create_backend(self.cfg.get('backend', 'rtmpose'), self.cfg,
                                      on_status=lambda s: setattr(self, 'model_status', s))
        self.model_status = ''
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.view = self.node.create_publisher(CompressedImage, self._outputs()[0]['topic'], qos)
        self.keypoints = self.node.create_publisher(String, self._outputs()[1]['topic'], qos)
        self.node.create_subscription(image_type, self.image_topic, self._on_image, qos)
        self.node.create_timer(1 / rate, self._tick)
        self.prev_time = self.published = 0
        self.limiter = RateLimiter(float(self.cfg.get('slew_rate', 20.0)))
        self.error = self.view_error = ''
        self.state = 'running'
        self.executor.add_node(self.node)
        self._attached = True

    def shutdown(self):
        with self._lifecycle, self._tick_lock:
            self.state = 'idle'
            if self.node is not None:
                if self._attached:
                    self.executor.remove_node(self.node)
                    self._attached = False
                self.node.destroy_node()
                self.node = None
            if self.backend is not None:
                self.backend.close()
                self.backend = None
            with self._frame_lock:
                self._frame = None
            self.hand_present = False
            self.last_curl = {}
            self.image_topic = self.model_status = ''

    def _on_image(self, msg):
        stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
        ms = stamp.sec * 1000 + stamp.nanosec // 1_000_000 if stamp else 0
        with self._frame_lock:
            self._frame = (msg, ms or int(time.time() * 1000), time.monotonic())

    @staticmethod
    def _decode(msg):
        from utils.cv2_compat import load_cv2
        import numpy as np

        cv2 = load_cv2()
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if hasattr(msg, 'format'):
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('JPEG 解码失败')
            return frame
        encoding = msg.encoding.lower()
        channels = {'mono8': 1, '8uc1': 1, 'rgb8': 3, 'bgr8': 3}[encoding]
        rows = raw.reshape(msg.height, msg.step)[:, :msg.width * channels]
        frame = rows.reshape(msg.height, msg.width, channels)
        if channels == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if encoding == 'rgb8' else frame.copy()

    def _observe(self, frame, obs_ms):
        wanted = self.cfg.get('hand', 'Right').lower()
        detection = next((d for d in self.backend.detect(frame)
                          if wanted == 'either' or d.handedness.lower() == wanted), None)
        if detection is None:
            return None, None
        raw = flexions(detection.landmarks, min_visibility=float(self.cfg.get('min_visibility', 0)))
        now = time.monotonic()
        curl = self.limiter.update(raw, now - self.prev_time if self.prev_time else 0)
        self.prev_time = now
        curls = {k: round(v, 4) for k, v in curl.items()}
        self._seq += 1
        # Payload keeps `schema: motus.hand/1` and the named `flexions` dict so
        # existing downstream drivers need no change; the extra control-style
        # fields (seq/values/ttl_ms/source/control_mode) make the canvas render
        # it as a command stream under the `control/hand_flex` format.
        return detection, {'schema': 'motus.hand/1', 'stamp_ms': int(time.time() * 1000),
                           'obs_stamp_ms': obs_ms, 'ttl_ms': 200,
                           'seq': self._seq, 'source': 'mcp__perception__hand_landmarks',
                           'control_mode': 'hand_flex',
                           'values': [curls.get(k, 0.0) for k in FINGER_ORDER],
                           'handedness': detection.handedness,
                           'handedness_source': detection.handedness_source, 'score': detection.score,
                           'flexions': curls,
                           'landmarks': [[round(float(x), 5) for x in pt] for pt in detection.landmarks]}

    def _tick(self):
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        with self._tick_lock:
            if self.state != 'running':
                return
            with self._frame_lock:
                sample = self._frame
            if not sample or time.monotonic() - sample[2] > 2:
                self.hand_present = False
                return
            # Consume each frame once. sample[2] is the arrival monotonic time
            # assigned in the subscription callback — unique per frame. If the
            # camera stops, self._frame keeps pointing at this same tuple, so
            # skip re-inference/re-publish instead of emitting fresh-stamped
            # copies of an old observation.
            if self._last_processed_frame == sample[2]:
                return
            try:
                frame = self._decode(sample[0])
            except Exception as exc:
                self.hand_present = False
                self.error = str(exc)
                return
            detection = None
            self.hand_present = False
            try:
                detection, payload = self._observe(frame, sample[1])
                if payload is not None:
                    msg = String()
                    msg.data = json.dumps(payload)
                    self.keypoints.publish(msg)
                    self.last_curl = payload['flexions']
                    self.published += 1
                    self.hand_present = True
                self.error = ''
            except Exception as exc:
                self.error = str(exc)
            if VIEW_ALWAYS_ON:
                try:
                    image = frame if detection is None else draw_hand(frame, detection.landmarks,
                        line_width=VIEW_LINE_WIDTH, point_radius=VIEW_POINT_RADIUS)
                    msg = CompressedImage()
                    msg.format = 'jpeg'
                    msg.data = encode_jpeg(image, VIEW_JPEG_QUALITY)
                    self.view.publish(msg)
                    self.view_error = ''
                except Exception as exc:
                    self.view_error = str(exc)
            # Mark consumed only after a full pass (keypoints + view), so a
            # decode/inference exception does not permanently skip this frame.
            self._last_processed_frame = sample[2]
