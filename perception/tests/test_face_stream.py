"""Stream window + register_by_stream / recognize_by_stream / list_visits."""
import ast
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

_FACE_PATH = Path(__file__).resolve().parents[1] / "plugins" / "face_scrfd.py"


def _face_node_methods():
    """Extract _FaceNode._image_cb/recent_frames without importing ROS."""
    tree = ast.parse(_FACE_PATH.read_text())
    methods = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_FaceNode":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in ("_image_cb", "recent_frames"):
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


def _compile_methods():
    """Compile the extracted _FaceNode methods into real functions."""
    module = ast.Module(body=list(_METHODS.values()), type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(_FACE_PATH), "exec")
    namespace = {"collections": collections, "queue": queue, "time": time,
                 "Optional": __import__("typing").Optional,
                 "CompressedImage": type("CompressedImage", (), {})}
    exec(code, namespace)
    return {name: namespace[name] for name in _METHODS}


_METHOD_FUNCS = _compile_methods()


def _jpeg(mean):
    ok, buf = cv2.imencode('.jpg', np.full((10, 10, 3), mean, np.uint8))
    assert ok
    return buf.tobytes()


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


if __name__ == '__main__':
    unittest.main()
