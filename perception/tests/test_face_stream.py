"""Stream window + register_by_stream / recognize_by_stream / list_visits."""
import ast
import json
import collections
import queue
import threading
import time
import unittest
from pathlib import Path

import cv2

# These contracts feed real pixel content through cv2 encode/decode and, in the
# detector case, cv2.dnn — the "WxH" marker fake cv2 (vision_stubs.__motus_fake__)
# cannot stand in for that. They run where cv2 is real: the perception image,
# whose container-side suite run is the documented acceptance pass (see
# perception/README.md § Running the tests inside a perception image). Elsewhere
# they skip rather than assert against stub pixels.
_NEEDS_REAL_CV2 = not getattr(cv2, "__motus_fake__", False)
import numpy as np

from test_face_candidates import load_face
from utils.log_sampling import SampledLogGate

_FACE_PATH = Path(__file__).resolve().parents[1] / "plugins" / "face_scrfd.py"


def _face_node_methods():
    """Extract _FaceNode methods without importing ROS."""
    tree = ast.parse(_FACE_PATH.read_text())
    methods = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_FaceNode":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in (
                        "_image_cb", "recent_frames", "_inference_worker"):
                    methods[item.name] = item
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {
                    "ENROLL_WINDOW_S", "ENROLL_WINDOW_MAX_FRAMES",
                    "ENROLL_MAX_ANALYZED", "VISIT_GAP_S",
                } for t in node.targets):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    return methods, constants


_METHODS, _CONSTANTS = _face_node_methods()

# The worker's globals, swappable so a test can record what gets logged.
_WORKER_NS = {"collections": collections, "queue": queue, "time": time,
              "Optional": __import__("typing").Optional,
              "SampledLogGate": SampledLogGate,
              "CompressedImage": type("CompressedImage", (), {})}
class _LogShim:
    """Default stand-in for the module logger in compiled worker code."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _compile_methods():
    """Compile the extracted _FaceNode methods into real functions."""
    module = ast.Module(body=list(_METHODS.values()), type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(_FACE_PATH), "exec")
    # _WORKER_NS **is** the exec globals (not a copy): a test swapping its
    # `log` for a recorder must be visible to the already-compiled functions.
    namespace = _WORKER_NS
    # The worker's real dependencies, from the extraction namespace the pixel
    # suites already build — not hand-stubbed copies that could drift.
    ns = load_face()
    namespace.update(np=np, json=json, threading=threading,
                     log=namespace.setdefault("log", _LogShim()),
                     String=type("String", (), {}),
                     _face_result=ns["_face_result"],
                     escape_log_text=namespace.setdefault("escape_log_text",
                                                          ns["escape_log_text"]))
    exec(code, namespace)
    return {name: namespace[name] for name in _METHODS}


_METHOD_FUNCS = _compile_methods()


def _jpeg(mean):
    ok, buf = cv2.imencode('.jpg', np.full((10, 10, 3), mean, np.uint8))
    assert ok
    return buf.tobytes()


class _WorkerNode:
    """Just enough of _FaceNode for _inference_worker, no ROS."""

    def __init__(self, face_db, adapter):
        self._frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._model = adapter
        self._face_db = face_db
        self._similarity_threshold = .4
        self._detect_count = 0
        self._diag_db_sent = False
        self._input_topic = '/cam'
        self._error_gate = _WORKER_NS["SampledLogGate"]()
        self.published = []
        node = self

        class _Pub:
            def publish(self, msg):
                node.published.append(json.loads(msg.data))

        self._pub = _Pub()


def _run_worker_frames(node, frames, until, timeout=5.0):
    """Push frames through the extracted worker until `until()` holds."""
    import types

    worker = types.MethodType(_METHOD_FUNCS['_inference_worker'], node)
    node._stop_event.clear()      # a previous run leaves it set
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        for data in frames:
            node._frame_queue.put(data)
        deadline = time.time() + timeout
        while time.time() < deadline and not until():
            time.sleep(0.02)
        assert until(), "the worker never reached the expected state"
    finally:
        node._stop_event.set()
        thread.join(timeout=3)


@unittest.skipUnless(_NEEDS_REAL_CV2, "needs real cv2")
class StreamContracts(unittest.TestCase):
    def setUp(self):
        self.ns = load_face()
        self.ns.update(collections=collections, queue=queue, threading=threading,
                       ENROLL_WINDOW_S=_CONSTANTS["ENROLL_WINDOW_S"],
                       ENROLL_WINDOW_MAX_FRAMES=_CONSTANTS["ENROLL_WINDOW_MAX_FRAMES"],
                       ENROLL_MAX_ANALYZED=_CONSTANTS["ENROLL_MAX_ANALYZED"],
                       VISIT_GAP_S=_CONSTANTS["VISIT_GAP_S"])
        self.plugin = self.ns['FaceRecognitionPlugin'].__new__(
            self.ns['FaceRecognitionPlugin'])
        self.plugin._nodes = {}
        self.plugin._face_db = self.ns['FaceDatabase']()
        self.plugin._similarity_threshold = .4
        self.plugin._model_lock = threading.Lock()
        self.plugin._model = None
        self.plugin._ensure_model = lambda: None

        class Adapter:
            def detect_and_embed(self, image):
                if not image.any():
                    return []
                embedding = np.array([1., 0.]) if image.mean() < 150 else np.array([0., 1.])
                return [dict(embedding=embedding, bbox=[0, 0, 8, 8], confidence=.9)]
        self.plugin._model = Adapter()

        class Node:
            _input_topic = '/cam'

            def __init__(self, frames):
                self._window = collections.deque(frames)
                self._window_lock = threading.Lock()
                self._window_seconds = _CONSTANTS["ENROLL_WINDOW_S"]

            def recent_frames(self, window_s=None):
                limit = (self._window_seconds if window_s is None
                         else min(float(window_s), self._window_seconds))
                cutoff = time.time() - max(0.0, limit)
                return [item for item in self._window if item[1] >= cutoff]

        self.Node = Node
        self.now = time.time()

    def call(self, action, **args):
        return self.ns['FaceRecognitionPlugin']._stream_action(
            self.plugin, action, args)

    def test_window_records_every_frame_oldest_first(self):
        class Node:
            pass
        for name, func in _METHOD_FUNCS.items():
            setattr(Node, name, func)
        node = Node()
        node._window = collections.deque(maxlen=_CONSTANTS["ENROLL_WINDOW_MAX_FRAMES"])
        node._window_lock = threading.Lock()
        node._window_seconds = _CONSTANTS["ENROLL_WINDOW_S"]
        node._frame_queue = queue.Queue(maxsize=1)
        node._last_inference_time = 0.0
        node._frame_interval = 0.0
        class Msg:
            def __init__(self, data):
                self.data = data
        for i in range(5):
            node._image_cb(Msg(bytes([i])))
        frames = node.recent_frames(999)
        self.assertEqual([f[0][0] for f in frames], [0, 1, 2, 3, 4])
        for _ in range(100):
            node._image_cb(Msg(bytes([9])))
        self.assertEqual(len(node._window), _CONSTANTS["ENROLL_WINDOW_MAX_FRAMES"])

    def test_register_recognize_and_failure_gates(self):
        self.plugin._nodes = {'n0': self.Node([(_jpeg(80), self.now)])}
        result = self.call('register_by_stream', name='Alice')
        self.assertTrue(result['ok'], result)
        self.assertEqual((result['person_id'], result['frames_used']), ('p-1', 1))
        self.assertEqual(result['name'], 'Alice')

        recognized = self.call('recognize_by_stream')
        self.assertTrue(recognized['ok'], recognized)
        self.assertEqual(recognized['count'], 1)
        self.assertEqual(recognized['faces'][0]['person_id'], 'p-1')

        # 2 of 5 frames agree with the anchor → ambiguous_subject
        frames = [(_jpeg(80), self.now), (_jpeg(220), self.now), (_jpeg(220), self.now),
                  (_jpeg(220), self.now), (_jpeg(80), self.now)]
        self.plugin._nodes = {'n0': self.Node(frames)}
        self.plugin._face_db = self.ns['FaceDatabase']()
        self.assertEqual(self.call('register_by_stream')['reason'], 'ambiguous_subject')

        self.plugin._nodes = {'n0': self.Node([(_jpeg(0), self.now)])}
        self.plugin._face_db = self.ns['FaceDatabase']()
        self.assertEqual(self.call('register_by_stream')['reason'], 'no_face')

        self.plugin._nodes = {'n0': self.Node([])}
        self.assertEqual(self.call('register_by_stream')['reason'], 'no_frames')

        self.plugin._nodes = {'n0': self.Node([(b'notajpeg', self.now)])}
        self.plugin._face_db = self.ns['FaceDatabase']()
        self.assertEqual(self.call('register_by_stream')['reason'], 'bad_input')

    def test_multi_frame_enrollment_and_explicit_id(self):
        self.plugin._nodes = {'n0': self.Node([(_jpeg(80), self.now)] * 3)}
        result = self.call('register_by_stream', name='Bob')
        self.assertTrue(result['ok'], result)
        self.assertEqual((result['frames_used'], result['samples']), (3, 3))
        update = self.call('register_by_stream', person_id='p-1')
        self.assertTrue(update['ok'], update)
        self.assertEqual((update['merged'], update['person_id'], update['samples']),
                         (True, 'p-1', 6))

    def test_unidentified_face_reported_not_nobody(self):
        self.plugin._nodes = {'n0': self.Node([(_jpeg(220), self.now)])}
        result = self.call('recognize_by_stream')
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['count'], 1)
        self.assertFalse(result['faces'][0]['known'])

    def test_list_visits_overlap_and_name_resolution(self):
        db = self.plugin._face_db
        db.record_sighting('p-1', 1000.0, '/cam')
        db.record_sighting('p-1', 1010.0, '/cam')
        visits = self.call('list_visits')
        self.assertEqual(visits['total'], 1)
        self.assertEqual(visits['visits'][0]['sightings'], 2)
        self.assertTrue(visits['visits'][0]['open'])
        self.assertEqual(self.call('list_visits', since=1005)['total'], 1)
        self.assertEqual(self.call('list_visits', since=1020)['total'], 0)
        self.assertEqual(self.call('list_visits', until=995)['total'], 0)
        self.assertEqual(db.close_stale_visits(now=1010 + self.ns['VISIT_GAP_S']), 1)
        db.enroll(np.array([1., 0.]), name='Alice')
        self.assertEqual(self.call('list_visits')['visits'][0]['name'], 'Alice')


class _StubNode:
    def __init__(self, topic):
        self._input_topic = topic

    def stop(self):
        return {"state": "stopped", "input": self._input_topic}


class _StubExecutor:
    def add_node(self, node):
        pass

    def remove_node(self, node):
        pass


def test_stop_closes_open_visits_only_with_the_last_instance():
    """A stopped instance can never send the frame that would close a visit
    by absence, so the last stop force-closes what is open. A still-running
    instance keeps its own visits open — a force-close has no per-instance
    filter, so it is only safe once nothing is left watching."""
    if not _NEEDS_REAL_CV2:      # mirrors the class guard: needs the real stack
        import pytest
        pytest.skip("needs real cv2")

    ns = load_face()
    ns.update(threading=threading)
    plugin = ns['FaceRecognitionPlugin'].__new__(ns['FaceRecognitionPlugin'])
    plugin._face_db = ns['FaceDatabase']()
    plugin._executor = _StubExecutor()
    plugin._nodes = {'/cam-a': _StubNode('/cam-a'), '/cam-b': _StubNode('/cam-b')}
    plugin._input_cfg = {}
    now = time.time()
    plugin._face_db.record_sighting('p-1', now, '/cam-a')

    first = ns['FaceRecognitionPlugin'].dispatch(
        plugin, 'face', {'action': 'stop', 'instance_id': '/cam-a'})
    assert first['visits_closed'] == 0, "the other instance is still watching"
    assert ns['FaceRecognitionPlugin']._stream_action(
        plugin, 'list_visits', {})['visits'][0]['open'] is True

    last = ns['FaceRecognitionPlugin'].dispatch(
        plugin, 'face', {'action': 'stop', 'instance_id': '/cam-b'})
    assert last['visits_closed'] == 1
    visit = ns['FaceRecognitionPlugin']._stream_action(plugin, 'list_visits', {})['visits'][0]
    assert visit.get('open') is not True, "no frame can ever close it now"


def test_stop_all_also_closes_visits():
    if not _NEEDS_REAL_CV2:
        import pytest
        pytest.skip("needs real cv2")

    ns = load_face()
    ns.update(threading=threading)
    plugin = ns['FaceRecognitionPlugin'].__new__(ns['FaceRecognitionPlugin'])
    plugin._face_db = ns['FaceDatabase']()
    plugin._executor = _StubExecutor()
    plugin._nodes = {'/cam-a': _StubNode('/cam-a')}
    plugin._input_cfg = {}
    plugin._face_db.record_sighting('p-1', time.time(), '/cam-a')

    result = ns['FaceRecognitionPlugin'].dispatch(plugin, 'face', {'action': 'stop'})

    assert result['stopped_instances'] == ['/cam-a']
    assert result['visits_closed'] == 1


def test_negative_window_s_is_a_bad_input_not_an_empty_window():
    """The schema says minimum 0.1; a negative request used to clamp to a
    zero-second window and answer "no_frames", which reads like the camera's
    fault rather than the caller's."""
    if not _NEEDS_REAL_CV2:
        import pytest
        pytest.skip("needs real cv2")

    ns = load_face()
    ns.update(threading=threading)
    plugin = ns['FaceRecognitionPlugin'].__new__(ns['FaceRecognitionPlugin'])
    plugin._face_db = ns['FaceDatabase']()
    plugin._nodes = {}
    plugin._ensure_model = lambda: None

    for action, args in (('register_by_stream', {'window_s': -5}),
                         ('recognize_by_stream', {'window_s': '-0.1'})):
        result = ns['FaceRecognitionPlugin']._stream_action(plugin, action, args)
        assert result['ok'] is False and result['reason'] == 'bad_input', result
        assert 'at least 0.1' in result['detail']


def test_empty_frames_close_a_stale_visit_without_any_face():
    """Absence closes a visit, and absence arrives as frames with no face:
    the worker checks on every tick, not only under `if faces`. Before that,
    a person who walked away stayed 'present' until somebody else showed up
    or the instance was stopped."""
    if not _NEEDS_REAL_CV2:
        import pytest
        pytest.skip("needs real cv2")

    ns = load_face()
    ns.update(threading=threading)
    db = ns['FaceDatabase']()
    db.record_sighting('p-1', time.time() - (ns['VISIT_GAP_S'] + 5), '/cam')

    class NoFaces:
        def detect_and_embed(self, image):
            return []

    node = _WorkerNode(db, NoFaces())
    _run_worker_frames(node, [_jpeg(0)], until=lambda: len(node.published) >= 1)

    assert node.published[0]['count'] == 0          # the frame indeed had no face
    visit = db.list_visits()['visits'][0]
    assert visit.get('open') is not True, "empty frames must close stale visits"


def test_repeated_inference_errors_are_sampled_not_per_frame():
    """One line per failure run, not one per frame: the transition logs with
    a traceback and escaped detail, repeats stay quiet until the sample
    cadence (100), and a different failure logs as a new transition."""
    if not _NEEDS_REAL_CV2:
        import pytest
        pytest.skip("needs real cv2")

    ns = load_face()
    ns.update(threading=threading)

    class _Recorder(_LogShim):
        def __init__(self):
            self.errors = []

        def error(self, text, exc_info=False):
            self.errors.append((text, bool(exc_info)))

    recorder = _Recorder()
    previous = _WORKER_NS["log"]
    _WORKER_NS["log"] = recorder
    try:
        class Boom:
            def __init__(self, error):
                self.calls = 0
                self.error = error

            def detect_and_embed(self, image):
                self.calls += 1
                raise self.error

        boom = Boom(RuntimeError("decoder exploded\nwith newline"))
        node = _WorkerNode(ns['FaceDatabase'](), boom)
        _run_worker_frames(node, [_jpeg(0)] * 3, until=lambda: boom.calls >= 3)

        assert len(recorder.errors) == 1, recorder.errors
        text, exc_info = recorder.errors[0]
        assert "RuntimeError" in text and "#1" in text
        assert "\n" not in text                    # escaped, cannot forge a line
        assert exc_info is True                     # the traceback rides the transition

        boom.error = ValueError("graph changed")
        _run_worker_frames(node, [_jpeg(0)], until=lambda: boom.calls >= 4)
        assert len(recorder.errors) == 2, recorder.errors
        assert "ValueError" in recorder.errors[1][0]
    finally:
        _WORKER_NS["log"] = previous


if __name__ == '__main__':
    unittest.main()
