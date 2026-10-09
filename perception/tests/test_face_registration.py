"""Exercise real dispatch/database with a deterministic embedding adapter."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.face_corpus import corpus_entries, load_image


@unittest.skipUnless(_NEEDS_REAL_CV2, "needs real cv2")
class RegistrationContracts(unittest.TestCase):
    def setUp(self):
        self.ns = load_face()
        self.ns.update(threading=threading, Path=Path, json=json,
                       corpus_entries=corpus_entries, load_image=load_image)
        self.db = self.ns['FaceDatabase']()
        self.plugin = self.ns['FaceRecognitionPlugin'].__new__(self.ns['FaceRecognitionPlugin'])
        self.plugin._face_db = self.db
        self.plugin._model_lock = threading.Lock()
        self.plugin._similarity_threshold = .4
        self.plugin._max_batch = 1000
        self.plugin._input_cfg = {}   # image_input roots default: /models /tmp /work
        class Adapter:
            def detect_and_embed(self, image):
                if not image.any():
                    return []
                embedding = np.array([1., 0.]) if image.mean() < 150 else np.array([0., 1.])
                return [dict(embedding=embedding, bbox=[0, 0, 8, 8], confidence=.9)]
        self.plugin._model = Adapter()

    def call(self, action, **args):
        return self.plugin.dispatch('face', dict(action=action, **args))

    def test_corpus_lifecycle_and_shared_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [('a.png', 80), ('b.png', 80), ('c.png', 220), ('bad.png', 0)]:
                cv2.imwrite(str(root / name), np.full((10, 10, 3), value, np.uint8))
            (root / 'manifest.json').write_text(json.dumps([
                dict(file='a.png', name='Alice', person='dataset-a'),
                dict(file='b.png', name='Alice', person='dataset-a'),
                dict(file='c.png', name='Bob', person='dataset-b')]))
            result = self.call('register_by_corpus', package=directory)
            self.assertTrue(result['ok'], result)
            self.assertEqual((result['total'], result['registered'], result['failed']), (4, 3, 1))
            rows = {row['file']: row for row in result['results']}
            pid = rows['a.png']['person_id']
            self.assertEqual(pid, rows['b.png']['person_id'])
            self.assertNotEqual(pid, rows['c.png']['person_id'])
            self.assertNotEqual(pid, 'dataset-a')
            self.assertEqual(self.db.match(np.array([1., 0.]), .4)[0], pid)
            self.assertEqual(self.call('get_person', person_id=pid)['person']['samples'], 2)
            update = self.call('update_person', person_id=pid, name='Alice Updated', profile={'role': 'test'})
            self.assertEqual(update['person']['name'], 'Alice Updated')
            photo = self.call('recognize_by_photo', image_path=str(root / 'a.png'))
            self.assertEqual(photo['faces'][0]['identity']['person_id'], pid)
            self.assertEqual(photo['faces'][0]['name'], 'Alice Updated')
            count = self.call('list_persons')['total']
            self.call('recognize_by_photo', image_path=str(root / 'a.png'))
            self.assertEqual(self.call('list_persons')['total'], count)
            deletion = self.call('forget', person_id=pid)
            self.assertTrue(deletion['ok'], deletion)
            self.assertEqual(self.db.match(np.array([1., 0.]), .4)[0], 'unknown')

    def test_photo_and_url_bbox_dimensions(self):
        from unittest.mock import patch

        image = np.full((20, 40, 3), 80, np.uint8)
        # image_path goes through the configured-root check before decoding,
        # so the photo lives under one (temp dirs are /tmp); a relative or
        # foreign path is rejected on purpose — covered in
        # test_face_scrfd_plugin.test_photo_and_corpus_paths_stay_inside_configured_roots.
        with tempfile.TemporaryDirectory() as directory:
            sample = str(Path(directory) / 'sample.png')
            for action, args in [('recognize_by_photo', {'image_path': sample}),
                                 ('recognize_by_url', {'url': 'https://example.invalid/face.png'})]:
                with self.subTest(action=action), patch.dict(self.ns, load_image=lambda *a, **k: image):
                    result = self.call(action, **args)
                    self.assertTrue(result['ok'], result)
                    self.assertEqual(result['image_size'], {'width': 40, 'height': 20})
                    self.assertEqual(result['faces'][0]['bbox_relative'], [0., 0., .2, .4])
                    # The grader reads `bbox`; it must carry the same normalized
                    # xywh, with the pixel box demoted to `bbox_px`.
                    self.assertEqual(result['faces'][0]['bbox'], [0., 0., .2, .4])
                    self.assertEqual(result['faces'][0]['bbox_px'], [0., 0., 8., 8.])

    def test_zip_dispatch(self):
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'batch.zip'
            _, image = cv2.imencode('.png', np.full((10, 10, 3), 80, np.uint8))
            with zipfile.ZipFile(package, 'w') as archive:
                archive.writestr('alice.png', image.tobytes())
                archive.writestr('manifest.json', json.dumps([
                    {'file': 'alice.png', 'name': 'Alice', 'person': 'benchmark-id'}]))
            result = self.call('register_by_corpus', package=str(package))
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['registered'], 1)
            pid = result['results'][0]['person_id']
            face = self.ns['_face_result'](self.db, dict(embedding=np.array([1., 0.]),
                                          bbox=[0, 0, 8, 8], confidence=.9), (10, 10, 3), .4)
            self.assertEqual(face['person_id'], pid)
            self.assertEqual(face['identity']['person_id'], pid)
            self.assertTrue(face['known'])
            self.assertEqual(face['bbox_relative'], [0., 0., .8, .8])

    def test_invalid_requests(self):
        for action, args in [('register_by_corpus', {}), ('get_person', {'person_id': 'absent'}),
                             ('register_by_photo', {'image_path': '/does/not/exist'})]:
            self.assertFalse(self.call(action, **args)['ok'])

    def test_concurrent_enrollment(self):
        threads = [threading.Thread(target=self.db.enroll, args=(np.array([1., 0.]),),
                                    kwargs={'name': 'Alice'}) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        roster = self.db.list_persons()
        self.assertEqual(roster['total'], 1)
        self.assertEqual(roster['persons'][0]['samples'], 8)


@unittest.skipUnless(_NEEDS_REAL_CV2, "needs real cv2")
class PersistenceContracts(unittest.TestCase):
    def setUp(self):
        from unittest.mock import Mock

        self.ns = load_face()
        self.ns.update(threading=threading, Path=Path)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state = self.root / '.state' / 'mcp-15720' / 'state.json'
        self.adapter = Mock()
        self.adapter.detect_and_embed.return_value = [
            dict(embedding=np.array([1., 0.]), confidence=.9)]

    def reopen(self, model='test-model', state=None):
        db = self.ns['FaceDatabase']()
        db.initialize_persistence(state or self.state, model, self.root, self.adapter)
        return db

    def legacy_image(self, pid='p-8'):
        person = self.root / pid
        person.mkdir()
        cv2.imwrite(str(person / 'sample.png'), np.full((10, 10, 3), 80, np.uint8))

    def test_recreate_embeddings_metadata_counter_and_update(self):
        self.legacy_image()
        db = self.reopen()
        self.assertEqual(db.get_person('p-8')['name'], 'p-8')
        result = db.enroll([0., 1.], name='名字', profile={'nested': {'a': [1]}, 'old': True})
        self.assertEqual(result['person_id'], 'p-9')
        db.enroll([.1, 1.], person_id='p-9')
        restored = self.reopen()
        self.assertEqual(restored.get_person('p-9'), db.get_person('p-9'))
        np.testing.assert_array_equal(restored._embeddings['p-9'], db._embeddings['p-9'])
        self.assertEqual(restored.match(np.array([0., 1.]), .4)[0], 'p-9')
        restored.update_person('p-9', name='Updated', profile={'role': 'test'}, profile_delete=['old'])
        restored = self.reopen()
        self.assertEqual(restored.get_person('p-9')['profile'], {'nested': {'a': [1]}, 'role': 'test'})
        self.assertEqual(restored.get_person('p-9')['name'], 'Updated')
        restored.update_person('p-9', name='', profile={'only': 1}, merge=False)
        self.assertEqual(self.reopen().get_person('p-9')['profile'], {'only': 1})
        self.assertFalse(self.reopen().get_person('p-9')['named'])
        restored.forget(person_id='p-9')
        self.assertEqual(self.reopen().enroll([0., 1.])['person_id'], 'p-10')
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o600)
        self.assertEqual(set(json.loads(self.state.read_text())),
                         {'version', 'model_id', 'embeddings', 'persons', 'next_person_id'})

    def test_forget_legacy_and_empty_library_never_reimport(self):
        self.legacy_image()
        db = self.reopen()
        db.enroll([0., 1.])
        db.forget(person_id='p-8')
        restored = self.reopen()
        self.assertNotIn('p-8', restored._embeddings)
        self.assertEqual(restored.forget(named='unknown')['forgotten'], 1)
        self.assertEqual(self.reopen().list_persons()['total'], 0)
        self.assertEqual(self.adapter.detect_and_embed.call_count, 1)
        self.assertEqual(self.reopen().enroll([1., 0.])['person_id'], 'p-10')

    def test_model_mismatch_and_corruption_fail_without_reimport(self):
        self.legacy_image()
        self.reopen()
        original = self.state.read_bytes()
        with self.assertRaisesRegex(ValueError, 'model mismatch'):
            self.reopen(model='other-model')
        self.assertEqual(self.state.read_bytes(), original)
        valid = json.loads(original)
        for field, value in [('version', 99), ('embeddings', {'p-8': [[0., 0.]]}),
                             ('persons', {}), ('next_person_id', 8)]:
            with self.subTest(field=field):
                self.state.write_text(json.dumps({**valid, field: value}))
                damaged = self.state.read_bytes()
                with self.assertRaises(ValueError):
                    self.reopen()
                self.assertEqual(self.state.read_bytes(), damaged)
        self.state.write_text('{broken')
        with self.assertRaises(ValueError):
            self.reopen()
        self.assertEqual(self.state.read_text(), '{broken')
        self.assertEqual(self.adapter.detect_and_embed.call_count, 1)

    def test_failed_atomic_save_rolls_back_and_cleans_temporary_files(self):
        from unittest.mock import patch

        db = self.reopen()
        pid = db.enroll([1., 0.], name='Original')['person_id']
        original = self.state.read_bytes()
        operations = [lambda: db.enroll([0., 1.]),
                      lambda: db.update_person(pid, name='Changed'),
                      lambda: db.forget(person_id=pid)]
        for operation in operations:
            with patch('os.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    operation()
            self.assertEqual(self.state.read_bytes(), original)
            self.assertEqual(db.list_persons(), self.reopen().list_persons())
            self.assertEqual(list(self.state.parent.glob('*.tmp')), [])
        self.assertEqual(db.enroll([0., 1.])['person_id'], 'p-2')

    def test_concurrent_persistent_enrollment(self):
        from concurrent.futures import ThreadPoolExecutor

        db = self.reopen()
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: db.enroll([1., 0.]), range(16)))
        self.assertTrue(all(r['ok'] for r in results))
        self.assertEqual(self.reopen().get_person('p-1')['samples'], 16)

    def test_plugin_port_isolation_restore_and_weight_mismatch(self):
        from unittest.mock import patch

        self.legacy_image()
        weights = self.root / 'test-model.onnx'
        weights.write_bytes(b'test weights')
        def plugin(port):
            obj = self.ns['FaceRecognitionPlugin'].__new__(self.ns['FaceRecognitionPlugin'])
            obj._face_db = self.ns['FaceDatabase']()
            obj._model_lock = threading.Lock()
            obj._model = None
            obj._model_name, obj._model_dir = 'test-model', str(self.root)
            obj._device, obj._confidence, obj._detector = 'cpu', .5, 'scrfd'
            obj._detector_device = obj._recognizer_device = None
            obj._face_db_dir = str(self.root)
            with patch.dict(self.ns, EdgeFaceAdapter=lambda *a, **kw: self.adapter):
                with patch.dict('os.environ', {'MCP_PORT': port}):
                    obj._ensure_model()
            return obj
        first = plugin('15720')
        first._face_db.forget(person_id='p-8')
        self.assertEqual(plugin('15720')._face_db.list_persons()['total'], 0)
        second = plugin('15820')
        self.assertEqual(second._face_db.list_persons()['total'], 1)
        second._face_db.update_person('p-8', name='Second instance')
        self.assertEqual(plugin('15720')._face_db.list_persons()['total'], 0)
        self.assertEqual(plugin('15820')._face_db.get_person('p-8')['name'], 'Second instance')
        self.assertEqual(plugin('')._face_db._state_path,
                         self.root / '.state' / 'default' / 'state.json')
        with self.assertRaisesRegex(ValueError, 'MCP_PORT'):
            plugin('../escape')
        external = self.root / 'test-model.onnx.data'
        external.write_bytes(b'external weights')
        with self.assertRaisesRegex(ValueError, 'model mismatch'):
            plugin('15720')
        plugin('15920')
        external.write_bytes(b'changed external weights')
        with self.assertRaisesRegex(ValueError, 'model mismatch'):
            plugin('15920')
        external.unlink()
        weights.write_bytes(b'changed weights under same model name')
        with self.assertRaisesRegex(ValueError, 'model mismatch'):
            plugin('15720')


if __name__ == '__main__':
    unittest.main()
