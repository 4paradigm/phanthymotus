"""Received-segment boundaries, real node ingress, and batched frame order."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from plugins import pose
from plugins.pose_segment import SegmentBuffer, SegmentActionBackend, rgb_frame_indices
from plugins.vision_runtime import VisionEngineSession
from test_pose_plugin import _plugin, _publisher, _standing_row
from test_pose_idle_finalize import _FinalClassifier
from vision_stubs import _FakeCompressedImage, _wait_until, frame_bytes


def test_buffer_separates_a_pause_even_when_worker_has_not_polled():
    buffer = SegmentBuffer(quiet_s=.2)
    for i in range(4):
        buffer.append(bytes([i]), i*.01)
    for i in range(4, 8):
        buffer.append(bytes([i]), 1.+(i-4)*.01)
    first, stats = buffer.pop_ready(1.04)
    assert [x[1] for x in first] == [bytes([i]) for i in range(4)]
    assert stats['received_frames'] == 4
    assert buffer.pop_ready(1.1) is None
    second, _ = buffer.pop_ready(1.3)
    assert [x[1] for x in second] == [bytes([i]) for i in range(4, 8)]
    assert buffer.pop_ready(2.) is None


def test_buffer_reports_truncation_and_reset_discards_queued_segments():
    buffer = SegmentBuffer(quiet_s=.2, max_bytes=6)
    for i in range(5):
        buffer.append(b'ab', i*.01)
    frames, stats = buffer.pop_ready(1.)
    assert len(frames) == 3 and stats['received_frames'] == 5
    assert stats['truncated_frames'] == 2
    buffer.append(b'old', 2.)
    buffer.append(b'new', 3.)
    buffer.reset()
    assert buffer.pop_ready(4.) is None


class _SegmentModel:
    last_error = None

    def __init__(self, wait_for_stop=False):
        self.clips = []
        self.entered = threading.Event()
        self.wait_for_stop = wait_for_stop

    def recognize(self, received, stop_event):
        self.clips.append(received)
        self.entered.set()
        if self.wait_for_stop:
            stop_event.wait(2.)
        row = np.array(_standing_row(), np.float32)
        return {'image': np.zeros((480, 640, 3), np.uint8),
                'boxes': row[None, :4], 'scores': row[None, 4],
                'keypoints': row[6:].reshape(1, 17, 3), 'primary_box': row[:4],
                'activity': {'name': 'brush teeth', 'name_zh': '刷牙',
                             'score': .81, 'action_id': 3},
                'evidence': {'frames_used': len(received)}}


def _start_segment_node(backend):
    plugin, _ = _plugin({'fps': 1, 'activity_mode': 'segment',
                          'activity_idle_flush_s': .12, 'label_hold': 1})
    plugin._segment_backend_for = lambda cfg: backend
    plugin._classifier_for = lambda cfg: _FinalClassifier()
    plugin.dispatch('pose', {'action': 'start', 'input_topic': '/cam/segment'})
    return plugin._nodes['/cam/segment']


def test_real_node_buffers_all_received_images_before_its_fps_gate():
    backend = _SegmentModel()
    node = _start_segment_node(backend)
    try:
        for _ in range(6):
            node.subscriptions[0].callback(_FakeCompressedImage(frame_bytes(640, 480)))
        _wait_until(lambda: node._detect_count == 1)
        lean = _publisher(node, node._output_topic)
        assert all('activity' not in p for msg in lean.messages
                   for p in json.loads(msg)['persons'])
        _wait_until(lambda: node._segment_count == 1)
        _wait_until(lambda: len(lean.messages) == 2)
        assert len(backend.clips) == 1 and len(backend.clips[0]) == 1
        assert backend.clips[0][0][2] is not None
        assert json.loads(lean.messages[-1])['persons'][0]['activity_score'] == .81
        assert node._detect_count == 1
        assert len(_publisher(node, node._skeleton_topic).messages) == 1
        assert node._last_prediction['received_frames'] == 6
        assert node._last_prediction['processed_observations'] == 1
    finally:
        node.stop()


def test_stop_cancels_publication_and_discards_input_from_a_stopped_subscription():
    backend = _SegmentModel(wait_for_stop=True)
    node = _start_segment_node(backend)
    try:
        for _ in range(4):
            node._image_cb(_FakeCompressedImage(frame_bytes(640, 480)))
        assert backend.entered.wait(2.)
        node.stop()
        node._image_cb(_FakeCompressedImage(frame_bytes(640, 480)))
        assert node._segment_buffer.pop_ready(1e30) is None
        assert node._segment_count == 0
        assert len(_publisher(node, node._output_topic).messages) == 1
    finally:
        node.stop()


@pytest.mark.parametrize('fps,pause,expected', [
    # Fractional FPS retains the existing integer configuration conversion.
    (30., 0., .15), (15., 0., .15), (7.15, 0., 2/7), (1., 0., 2.),
    (30., .4, .4), (1., .4, .4), (1., .12, .12),
])
def test_segment_auto_pause_respects_frame_period_and_explicit_config(fps, pause, expected):
    plugin, _ = _plugin({'fps': fps, 'activity_mode': 'segment',
                          'activity_idle_flush_s': pause})
    plugin._segment_backend_for = lambda cfg: _SegmentModel()
    plugin._classifier_for = lambda cfg: _FinalClassifier()
    plugin.dispatch('pose', {'action': 'start', 'input_topic': '/cam/auto_pause'})
    node = plugin._nodes['/cam/auto_pause']
    try:
        assert node._activity_idle_flush_s == pytest.approx(expected)
        assert node._segment_buffer.quiet_s == pytest.approx(pause or max(.4, expected))
    finally:
        node.stop()


def test_auto_segment_retains_frames_across_short_pause_and_publishes_once(monkeypatch):
    now = 10.
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))
    backend = _SegmentModel()
    node = pose._PoseNode('/cam/segment', None, .35, 30., 'quiet',
                          classifier=_FinalClassifier(), segment_backend=backend)
    messages = _publisher(node, node._output_topic).messages
    row = np.array(_standing_row(), np.float32)
    buffer = node._segment_buffer

    def record_frames(start):
        for i in range(4):
            timestamp = start+i*.03
            buffer.append(bytes([i]), timestamp)
            buffer.record_pose(timestamp, row[None, :4], row[None, 4],
                               row[6:].reshape(1, 17, 3), (480, 640, 3))

    record_frames(10.)
    now = 10.26
    node._flush_received_segment()
    assert [len(clip) for clip in backend.clips] == [4]
    assert messages == []

    record_frames(10.34)
    now = 10.60
    node._flush_received_segment()
    assert [len(clip) for clip in backend.clips] == [4, 8]
    assert messages == []

    now = 10.84
    node._flush_received_segment()
    node._flush_received_segment()
    assert [len(clip) for clip in backend.clips] == [4, 8]
    assert len(messages) == 1
    evidence = node._last_prediction
    assert evidence['received_frames'] == evidence['processed_observations'] == 8


def test_resume_during_preview_reclassifies_full_segment_in_real_worker(monkeypatch):
    now = 10.
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))
    release_first = threading.Event()

    class BlockedPreview(_SegmentModel):
        def recognize(self, received, stop_event):
            result = super().recognize(received, stop_event)
            if len(self.clips) == 1:
                if not release_first.wait(2.):
                    raise RuntimeError('Test did not release the first forward')
                result['activity'] = {**result['activity'], 'name': 'drink water'}
            return result

    backend = BlockedPreview()
    plugin, _ = _plugin({'activity_mode': 'segment', 'label_hold': 1})
    plugin._segment_backend_for = lambda cfg: backend
    plugin._classifier_for = lambda cfg: _FinalClassifier()
    topic = '/cam/segment'
    plugin.dispatch('pose', {'action': 'config', 'instance_id': topic, 'fps': 30})
    plugin.dispatch('pose', {'action': 'start', 'input_topic': topic})
    node = plugin._nodes[topic]
    callback = node.subscriptions[0].callback
    lean = _publisher(node, node._output_topic)
    frame = _FakeCompressedImage(frame_bytes(640, 480))

    def feed(when, expected):
        nonlocal now
        now = when
        callback(frame)
        assert _wait_until(lambda: node._detect_count == expected)

    def activities():
        return [person['activity'] for raw in lean.messages
                for person in json.loads(raw)['persons'] if 'activity' in person]

    try:
        for index in range(4):
            feed(10. + index*.05, index+1)
        now = 10.32
        assert backend.entered.wait(2.)
        assert [len(x) for x in backend.clips] == [4]
        assert activities() == []

        # Deliver a frame while the first inference is still in flight.
        now = 10.40
        callback(frame)
        release_first.set()
        assert _wait_until(lambda: node._detect_count == 5)
        for index in range(1, 4):
            feed(10.40 + index*.05, index+5)
        now = 10.73
        assert _wait_until(lambda: len(backend.clips) == 2)
        assert [len(x) for x in backend.clips] == [4, 8]
        assert activities() == []

        now = 11.0
        assert _wait_until(lambda: activities() == ['brush teeth'])
        assert len(backend.clips) == 2  # Publishing reused the complete preview.
        assert node._detect_count == 8
        assert node._last_prediction['processed_observations'] == 8
    finally:
        release_first.set()
        node.stop()


def test_dynamic_pose_batches_preserve_input_order_and_individual_letterboxes():
    class Engine:
        input_dtype = np.float32
        profiles = [((1, 3, 32, 32), (3, 3, 32, 32), (3, 3, 32, 32))]

        def __init__(self):
            self.calls = []
            self.count = 0

        def infer(self, blob):
            self.calls.append(blob.shape)
            n = len(blob)
            values = np.arange(self.count, self.count+n).reshape(n, 1, 1)
            self.count += n
            return [values]

    session = VisionEngineSession.__new__(VisionEngineSession)
    session._engine = Engine()
    session._in_w = session._in_h = 32
    images = [np.zeros((20+i, 30+i, 3), np.uint8) for i in range(5)]
    results = session.infer_batch(images)
    assert session._engine.calls == [(3, 3, 32, 32), (2, 3, 32, 32)]
    assert [int(output[0].item()) for output, _ in results] == list(range(5))
    assert [(meta.orig_h, meta.orig_w) for _, meta in results] == [(20+i, 30+i) for i in range(5)]


def test_pose_finishing_after_a_gap_stays_with_its_received_segment():
    buffer = SegmentBuffer(quiet_s=.2)
    buffer.append(b'old', 0.)
    buffer.append(b'new', 1.)
    boxes = np.array([[1., 2., 3., 4.]])
    buffer.record_pose(0., boxes, np.array([.9]), np.zeros((1, 17, 3)), (480, 640, 3))
    boxes[0, 0] = 999.
    frames, _ = buffer.pop_ready(1.01)
    assert len(frames) == 1 and frames[0][1] == b'old'
    assert frames[0][2][0][0, 0] == 1.
    buffer.reset()
    assert not buffer._by_time


@pytest.mark.parametrize('rgb_frames', [8, 24])
def test_final_classification_reuses_poses_and_never_calls_the_pose_engine(rgb_frames, monkeypatch):
    import cv2

    class PoseMustNotRun:
        def infer(self, image):
            raise AssertionError('pose inference after the input pause')

        def infer_batch(self, images):
            raise AssertionError('pose batch after the input pause')

    class Head:
        def __init__(self, shape):
            self.input_shape = shape

        def infer(self, blob):
            assert blob.shape == self.input_shape
            p = np.zeros((1, 126), np.float32)
            p[0, 2], p[0, 37] = .9, .1
            return [p]

    row = np.array(_standing_row(), np.float32)
    decoded = (row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3), (480, 640))
    received = [(i*.1, frame_bytes(640, 480), decoded) for i in range(5)]
    backend = SegmentActionBackend(PoseMustNotRun(), Head((1, 2, 100, 17, 6)),
                                   Head((1, rgb_frames, 3, 192, 192)))
    original_decode = cv2.imdecode
    calls = []

    def count_decode(data, flags):
        calls.append(1)
        return original_decode(data, flags)

    monkeypatch.setattr(cv2, 'imdecode', count_decode)
    result = backend.recognize(received)
    # The last primary frame is already one of the RGB classifier's inputs.
    assert len(calls) == len(set(rgb_frame_indices(5, rgb_frames//2)))
    assert np.array_equal(result['image'], original_decode(
        np.frombuffer(received[-1][1], np.uint8), cv2.IMREAD_COLOR))
    assert result['activity']['action_id'] == 3
    assert result['activity']['score'] == .9
    assert result['evidence']['poses_computed_during_reception'] is True
    assert result['evidence']['RGB_frames_per_view'] == rgb_frames//2
    assert backend.recognize(received[:3]) is None


def test_model_bundle_spec_survives_creation_of_an_instance_backend(tmp_path, monkeypatch):
    from utils import tensorrt_runtime

    loaded = []

    class Engine:
        def __init__(self, path, **kwargs):
            loaded.append(path.name)
            self.input_shape = ((1, 2, 100, 17, 6) if path.name == 'body-new.engine'
                                else (1, 24, 3, 192, 192))

        def close(self):
            pass

    monkeypatch.setattr(tensorrt_runtime, 'TensorRTEngine', Engine)
    (tmp_path/'models.json').write_text(json.dumps({'body_engine': 'body-new.engine',
        'rgb_engine': 'teacher-new.engine', 'rgb_ntu_weight': .5}))
    backend = SegmentActionBackend.from_directory(object(), tmp_path)
    assert loaded == ['body-new.engine', 'teacher-new.engine']
    plugin, _ = _plugin({'activity_mode': 'segment'})
    plugin._segment_model = backend
    monkeypatch.setenv('ACTION_SEGMENT_MODEL_DIR', str(tmp_path))
    instance = plugin._segment_backend_for(plugin._merged_config('new-instance'))
    assert instance.rgb_n_per_view == 12
    assert instance.ntu_rgb_weight == .5


def test_deployment_environment_keeps_core_instance_overrides(tmp_path, monkeypatch):
    engine = tmp_path/'pose.engine'
    engine.write_bytes(b'path-resolution-only')
    monkeypatch.setenv('POSE_ACTIVITY_MODE', 'segment')
    monkeypatch.setenv('POSE_ENGINE_PATH', str(engine))
    plugin, _ = _plugin()
    assert plugin._merged_config('fresh')['activity_mode'] == 'segment'
    plugin._instance_configs['core'] = {'fps': 30, 'activity_mode': 'rolling'}
    assert plugin._merged_config('core')['activity_mode'] == 'rolling'
    assert plugin._resolve_engine() == str(engine)


def test_lean_segment_frames_preserve_tracking_and_both_head_inputs(monkeypatch):
    from plugins import pose_segment
    from plugins.pose_action import PoseFrame

    class Head:
        def __init__(self, shape):
            self.input_shape = shape
            self.inputs = []

        def infer(self, blob):
            self.inputs.append(blob.copy())
            p = np.zeros((1, 126), np.float32)
            p[0, 2], p[0, 37] = .9, .1
            return [p]

    def run():
        body = Head((1, 2, 100, 17, 6))
        rgb = Head((1, 8, 3, 192, 192))
        backend = SegmentActionBackend(object(), body, rgb)
        result = backend.recognize(received)
        return result, body.inputs, rgb.inputs

    row = np.array(_standing_row(), np.float32)
    received = []
    # Multiple people, reordered detections, invisible joints, and a gap long
    # enough to expire tracks exercise both IoU and centroid association.
    for i in range(12):
        rows = np.stack([row.copy(), row.copy()])
        rows[1, [0, 2]] += 180
        rows[1, 6:].reshape(17, 3)[:, 0] += 180
        if i == 4:
            rows = rows[::-1].copy()
        if i in (5, 6):
            rows[0, 6:].reshape(17, 3)[:5, 2] = 0
        pose = (rows[:, :4], rows[:, 4], rows[:, 6:].reshape(2, 17, 3), (480, 640))
        received.append((i*.1+(3. if i >= 8 else 0.),
                         frame_bytes(640, 480), pose))
    lean, lean_body, lean_rgb = run()
    monkeypatch.setattr(pose_segment, '_SkeletonFrame', PoseFrame)
    reference, reference_body, reference_rgb = run()
    assert lean['evidence'] == reference['evidence']
    assert lean['activity'] == reference['activity']
    np.testing.assert_array_equal(lean['probabilities'], reference['probabilities'])
    np.testing.assert_array_equal(lean_body[0], reference_body[0])
    np.testing.assert_array_equal(lean_rgb[0], reference_rgb[0])


def test_crop_e_router_and_warmup_propagate_from_manifest(tmp_path, monkeypatch):
    from utils import tensorrt_runtime
    from plugins.pose_segment import build_rgb_input, fuse_calibrated_router
    from plugins.pose_stgcn import resample_clip

    warm_calls = []

    class Engine:
        input_dtype = np.float32

        def __init__(self, path, **kwargs):
            self.name = path.name
            self.input_shape = ((1, 2, 100, 17, 6) if path.name == 'body.engine'
                                else (1, 32, 3, 192, 192))

        def infer(self, blob):
            warm_calls.append((self.name, blob.shape, blob.dtype))
            out = np.zeros((1, 126), np.float32)
            out[0, 55 if self.name == 'rgb.engine' else 0] = 6.0
            return [out]

        def close(self):
            pass

    monkeypatch.setattr(tensorrt_runtime, 'TensorRTEngine', Engine)
    theta = [1.3584, 1.876, -1.3663, -1.197, 0.472, -0.7761, -1.506, 1.1615, 0.5428, 0.2666]
    (tmp_path/'models.json').write_text(json.dumps({
        'body_engine': 'body.engine',
        'rgb_engine': 'rgb.engine',
        'crop_mode': 'E',
        'router': {'mode': 'poe_gate', 'theta': theta},
    }))
    backend = SegmentActionBackend.from_directory(object(), tmp_path)
    backend.warmup()
    assert warm_calls == [
        ('body.engine', (1, 2, 100, 17, 6), np.dtype('float32')),
        ('rgb.engine', (1, 32, 3, 192, 192), np.dtype('float32')),
    ]

    plugin, _ = _plugin({'activity_mode': 'segment'})
    plugin._segment_model = backend
    monkeypatch.setenv('ACTION_SEGMENT_MODEL_DIR', str(tmp_path))
    instance = plugin._segment_backend_for(plugin._merged_config('inst-v4'))
    assert instance.crop_mode == 'E'
    assert instance.rgb_n_per_view == 16
    assert instance.router == {'mode': 'poe_gate', 'theta': theta}

    row = np.array(_standing_row(), np.float32)
    decoded = (row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3), (480, 640))
    received = [(i*.08, frame_bytes(640, 480), decoded) for i in range(16)]
    res = instance.recognize(received)
    assert res is not None and res['probabilities'].shape == (126,)
    assert np.isclose(res['probabilities'].sum(), 1.0, atol=1e-5)

    # Verify Crop E supports both close-wrist tabletop B and raised-chest C modes.
    imgs = [np.zeros((480, 640, 3), np.uint8) for _ in range(16)]
    kps = np.zeros((16, 17, 3), np.float32)
    kps[:, :, 2] = .9
    kps[:, 9, :2] = (300., 300.)
    kps[:, 10, :2] = (310., 300.)
    kps[:, 5, :2] = (260., 180.)
    kps[:, 6, :2] = (360., 180.)
    blob_b = build_rgb_input(imgs, kps, crop_mode='E')
    kps[:, 10, :2] = (480., 300.)
    blob_c = build_rgb_input(imgs, kps, crop_mode='E')
    assert blob_b.shape == blob_c.shape == (1, 32, 3, 192, 192)

    # Verify physical-timestamp resampling handles non-uniform frame gaps cleanly.
    clip = np.zeros((3, 17, 6), np.float32)
    clip[1] = 1.0
    clip[2] = 3.0
    ts = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    sampled = resample_clip(clip, 5, timestamps=ts)
    assert sampled.shape == (5, 17, 6)
    assert np.isclose(sampled[1, 0, 0], 1.0, atol=1e-5)
    assert np.isclose(sampled[2, 0, 0], 1.0 + (0.5 - 0.25) / 0.75 * 2.0, atol=1e-5)

    # Verify router low-FPS gate shifts weight toward RGB when FPS drops from 30 -> 12.
    pb = np.full(126, .1 / 125, np.float32)
    pr = np.full(126, .1 / 125, np.float32)
    pb[0] = .9
    pr[55] = .9
    f_hi = fuse_calibrated_router(pb, pr, {'mode': 'poe_gate', 'theta': theta}, observation_fps=30.0)
    f_lo = fuse_calibrated_router(pb, pr, {'mode': 'poe_gate', 'theta': theta}, observation_fps=12.0)
    assert f_lo[55] > f_hi[55]


def test_preview_finishing_after_quiet_boundary_publishes_immediately_and_evicts_unposed_jpegs(monkeypatch):
    now = 20.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))

    class SlowPreview(_SegmentModel):
        def recognize(self, received, stop_event):
            nonlocal now
            result = super().recognize(received, stop_event)
            now = 20.60  # Cross quiet_s=0.40s while segment inference is in flight.
            return result

    backend = SlowPreview()
    node = pose._PoseNode('/cam/segment', None, .35, 10., 'slow_preview',
                          classifier=_FinalClassifier(), segment_backend=backend)
    messages = _publisher(node, node._output_topic).messages
    row = np.array(_standing_row(), np.float32)
    buffer = node._segment_buffer

    # Simulate throttled + queue-evicted frames releasing JPEG bytes while keeping counts.
    for i in range(4):
        now = 20.0 + i * .03
        node._image_cb(_FakeCompressedImage(b'x' * 4096))
    # Drain the single queued posed item so _frame_queue is empty, and attach 4 poses.
    while not node._frame_queue.empty():
        node._frame_queue.get_nowait()
    # Throttled/evicted frames should have released their raw JPEG bytes before posing.
    assert buffer._bytes <= 4096
    for i in range(4):
        ts = 20.0 + i * .03
        buffer._frames[i][1] = b'j' * 128
        buffer.record_pose(ts, row[None, :4], row[None, 4],
                           row[6:].reshape(1, 17, 3), (480, 640, 3))

    now = 20.32  # Past preview_s (20.09 + 0.20 = 20.29) before quiet_s (20.49) -> starts preview.
    node._flush_received_segment()
    # Because SlowPreview crossed quiet_s=0.40s with no new frames, it confirms in the same call!
    assert len(messages) == 1
    assert node._segment_preview is None
    assert node._last_prediction['received_frames'] == 4
    assert node._last_prediction['processed_observations'] == 4


def test_single_view_rgb_head_and_streaming_crop_parity(tmp_path, monkeypatch):
    from utils import tensorrt_runtime
    from plugins.pose_segment import (
        build_rgb_input, build_rgb_views_from_crops,
        crop_rgb_pair_for_frame, prepare_rgb_crop_geometry,
    )

    rng = np.random.default_rng(123)
    imgs = [rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8) for _ in range(16)]
    kps = rng.uniform(20., 200., size=(16, 17, 3)).astype(np.float32)
    kps[:, :, 2] = .9

    for mode in ('A', 'E'):
        ref = build_rgb_input(imgs, kps, crop_mode=mode)
        geom = prepare_rgb_crop_geometry(kps, imgs[0].shape[:2], crop_mode=mode)
        pairs = [crop_rgb_pair_for_frame(imgs[i], i, geom) for i in range(16)]
        vp, vh = build_rgb_views_from_crops([p[0] for p in pairs], [p[1] for p in pairs])
        np.testing.assert_array_equal(np.concatenate([vp, vh], axis=1), ref)

    seen_shapes = []

    class SVEngine:
        input_dtype = np.float32

        def __init__(self, path, **kwargs):
            self.name = path.name
            self.input_shape = ((1, 2, 100, 17, 6) if path.name == 'body.engine'
                                else (1, 16, 3, 192, 192))

        def infer(self, blob):
            seen_shapes.append((self.name, blob.shape))
            if self.name == 'body.engine':
                out = np.zeros((1, 126), np.float32)
                out[0, 0] = 5.0
                return [out]
            return [np.ones((1, 16, 8), np.float32) * 0.25]

        def close(self):
            pass

    monkeypatch.setattr(tensorrt_runtime, 'TensorRTEngine', SVEngine)
    np.savez(
        tmp_path / 'head.npz',
        att_f_w=np.zeros((4, 16), np.float32),
        att_d_w=np.zeros((4, 16), np.float32),
        att_e_w=np.zeros((4, 1), np.float32),
        att_e_b=np.zeros((4,), np.float32),
        att_out_w=np.zeros((1, 4), np.float32),
        att_out_b=np.zeros((1,), np.float32),
        gamma_ev=np.zeros((1, 16), np.float32),
        gamma_dv=np.zeros((1, 16), np.float32),
        fc_w=np.zeros((126, 16), np.float32),
        fc_b=np.linspace(0., 6., 126, dtype=np.float32),
    )
    (tmp_path / 'models.json').write_text(json.dumps({
        'body_engine': 'body.engine',
        'rgb_engine': 'sv.engine',
        'rgb_head': 'head.npz',
        'crop_mode': 'E',
    }))
    backend = SegmentActionBackend.from_directory(object(), tmp_path)
    plugin, _ = _plugin({'activity_mode': 'segment'})
    plugin._segment_model = backend
    monkeypatch.setenv('ACTION_SEGMENT_MODEL_DIR', str(tmp_path))
    instance = plugin._segment_backend_for(plugin._merged_config('sv-inst'))
    assert instance.rgb_single_view is True
    assert instance.rgb_n_per_view == 16

    row = np.array(_standing_row(), np.float32)
    decoded = (row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3), (480, 640))
    received = [(i * .08, frame_bytes(640, 480), decoded) for i in range(16)]
    res = instance.recognize(received)
    assert res is not None and np.isclose(res['probabilities'].sum(), 1.0, atol=1e-5)
    assert seen_shapes.count(('sv.engine', (1, 16, 3, 192, 192))) == 2

    decoded_hd = (row[None, :4] * 2., row[None, 4],
                  (row[6:].reshape(1, 17, 3) * np.array([2., 2., 1.], np.float32)),
                  (960, 1280))
    received_hd = [(i * .08, frame_bytes(1280, 960), decoded_hd) for i in range(16)]
    res_lean = instance.recognize(received_hd, need_image=False)
    assert res_lean is not None
    assert res_lean['image'].shape == (960, 1280, 3)
    assert not isinstance(res_lean['image'], np.ndarray)


def test_person_roi_patch_zero_jpeg_decode_and_stride_thinning(monkeypatch):
    import cv2
    from plugins.pose_segment import make_person_roi_patch

    class MustNotRunPose:
        def infer(self, image):
            raise AssertionError('unexpected pose infer')

    class SVHead:
        input_dtype = np.float32

        def __init__(self, shape, is_body=False):
            self.input_shape = shape
            self.is_body = is_body

        def infer(self, blob):
            assert blob.shape == self.input_shape
            if self.is_body:
                out = np.zeros((1, 126), np.float32)
                out[0, 12] = 4.0
                return [out]
            return [np.ones((1, 16, 8), np.float32) * 0.2]

    head_cfg = {
        'att_f_w': np.zeros((4, 16), np.float32),
        'att_d_w': np.zeros((4, 16), np.float32),
        'att_e_w': np.zeros((4, 1), np.float32),
        'att_e_b': np.zeros((4,), np.float32),
        'att_out_w': np.zeros((1, 4), np.float32),
        'att_out_b': np.zeros((1,), np.float32),
        'gamma_ev': np.zeros((1, 16), np.float32),
        'gamma_dv': np.zeros((1, 16), np.float32),
        'fc_w': np.zeros((126, 16), np.float32),
        'fc_b': np.linspace(0., 5., 126, dtype=np.float32),
    }
    backend = SegmentActionBackend(
        MustNotRunPose(),
        SVHead((1, 2, 100, 17, 6), is_body=True),
        SVHead((1, 16, 3, 192, 192), is_body=False),
        crop_mode='E',
        rgb_head=head_cfg,
    )

    buffer = SegmentBuffer(quiet_s=.4, max_patches=32)
    row = np.array(_standing_row(), np.float32)
    boxes = row[None, :4] * 2.0
    scores = row[None, 4]
    kps = row[6:].reshape(1, 17, 3) * np.array([2., 2., 1.], np.float32)
    bgr = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    bgr[200:800, 300:900] = (40, 150, 220)

    # Stream 110 posed HD frames with ZERO raw JPEG bytes buffered.
    for i in range(110):
        ts = 100.0 + i * 0.05
        buffer.append(b'', ts)
        patch = make_person_roi_patch(bgr, boxes, scores, kps, roi_hw=320)
        assert max(patch[0].shape[:2]) <= 320
        buffer.record_pose(ts, boxes, scores, kps, bgr.shape, rgb_patch=patch)
        live_patches = sum(
            1 for r in buffer._frames
            if r[2] is not None and len(r[2]) > 4 and r[2][4] is not None
        )
        assert live_patches <= 32
        assert buffer._bytes == 0

    records, stats = buffer.pop_ready(200.0)
    assert len(records) == 110 and stats['received_frames'] == 110
    # Every single skeleton observation stays intact despite patch stride-thinning.
    assert all(r[2] is not None and len(r[2]) >= 4 for r in records)
    # Endpoints retain exact patches.
    assert len(records[0][2]) == 5 and len(records[-1][2]) == 5

    monkeypatch.setattr(
        cv2, 'imdecode',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('imdecode must not run')),
    )
    res_lean = backend.recognize(records, need_image=False)
    assert res_lean is not None
    assert res_lean['image'].shape == (1080, 1920, 3)
    assert not isinstance(res_lean['image'], np.ndarray)
    assert np.isclose(res_lean['probabilities'].sum(), 1.0, atol=1e-5)

    res_img = backend.recognize(records, need_image=True)
    assert res_img is not None
    assert isinstance(res_img['image'], np.ndarray)
    assert res_img['image'].shape == (1080, 1920, 3)


def test_low_confidence_followup_bounce_fragment_is_suppressed(monkeypatch):
    now = 50.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))

    class TwoSegModel(_SegmentModel):
        def recognize(self, received, stop_event):
            result = super().recognize(received, stop_event)
            if len(self.clips) == 2:
                result['activity'] = {**result['activity'], 'name': 'phone call', 'score': .19}
            elif len(self.clips) == 3:
                result['activity'] = {**result['activity'], 'name': 'none', 'score': .67}
            elif len(self.clips) == 4:
                result['activity'] = {**result['activity'], 'name': 'hand waving', 'score': .72}
            return result

    backend = TwoSegModel()
    node = pose._PoseNode('/cam/segment', None, .35, 30., 'bounce_guard',
                          classifier=_FinalClassifier(), segment_backend=backend)
    messages = _publisher(node, node._output_topic).messages
    row = np.array(_standing_row(), np.float32)
    buf = node._segment_buffer

    def push_seg(t0):
        for i in range(4):
            ts = t0 + i * .03
            buf.append(b'', ts)
            buf.record_pose(ts, row[None, :4], row[None, 4],
                            row[6:].reshape(1, 17, 3), (480, 640, 3))

    push_seg(50.0)
    now = 50.60
    node._flush_received_segment()
    assert len(messages) == 1

    # Immediate (<2.5s) low-confidence (0.19 < 0.38) tail bounce is suppressed.
    push_seg(50.70)
    now = 51.30
    node._flush_received_segment()
    assert len(messages) == 1

    # Immediate (<2.5s) trailing 'none' fragment after a confirmed action is also suppressed.
    push_seg(51.40)
    now = 52.00
    node._flush_received_segment()
    assert len(messages) == 1

    # Confident follow-up action (0.72 >= 0.38) still publishes immediately.
    push_seg(52.10)
    now = 52.70
    node._flush_received_segment()
    assert len(messages) == 2

    # Action id 66 ('none') belongs to Other group rather than Kitchen.
    from plugins.pose_segment import _KITCHEN_IDS
    assert 66 not in _KITCHEN_IDS and 13 in _KITCHEN_IDS and 28 in _KITCHEN_IDS


def test_dual_roi_patch_shrinks_buffer_ram_and_preserves_hand_detail(monkeypatch):
    import cv2
    from plugins.pose_segment import (
        crop_rgb_pair_from_patch,
        make_dual_roi_patch,
        prepare_rgb_crop_geometry,
    )

    bgr = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    row = np.array(_standing_row(), np.float32)
    boxes = row[None, :4] * 2.5
    scores = row[None, 4]
    kps = row[6:].reshape(1, 17, 3) * np.array([2.5, 2.5, 1.], np.float32)
    # Paint high-contrast stripe near extended right/left wrists (indices 9, 10)
    for wi in (9, 10):
        wx, wy = int(kps[0, wi, 0]), int(kps[0, wi, 1])
        bgr[max(0, wy - 20):wy + 20, max(0, wx - 20):wx + 20:2] = (10, 240, 20)

    dual_patch = make_dual_roi_patch(bgr, boxes, scores, kps, person_hw=224, hand_hw=192)
    p_pair, h_pair = dual_patch
    assert max(p_pair[0].shape[:2]) <= 224
    total_bytes = p_pair[0].nbytes + (h_pair[0].nbytes if h_pair is not None else 0)
    assert total_bytes <= (224 * 224 + 192 * 192) * 3  # 261 KB < 307 KB (PersonROI-320)

    monkeypatch.setattr(
        cv2, 'imdecode',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('imdecode must not run')),
    )
    geom = prepare_rgb_crop_geometry([kps[0], kps[0]], bgr.shape[:2], crop_mode='E')
    p_crop, h_crop = crop_rgb_pair_from_patch(dual_patch, None, 0, geom, bgr.shape[:2])
    assert p_crop.shape == (3, 192, 192) and h_crop.shape == (3, 192, 192)
    # Also test fallback when hand_patch is None
    p_crop2, h_crop2 = crop_rgb_pair_from_patch((p_pair, None), None, 0, geom, bgr.shape[:2])
    assert p_crop2.shape == (3, 192, 192) and h_crop2.shape == (3, 192, 192)


def test_segment_pause_prevents_mid_clip_split_and_late_weak_followup_suppressed(monkeypatch):
    now = 100.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))

    class LateWeakModel(_SegmentModel):
        def recognize(self, received, stop_event):
            result = super().recognize(received, stop_event)
            # Call 1 = preview at 100.52 (4f); Call 2 = confirmed clip #1 at 101.35 (8f).
            if len(self.clips) == 2:
                result['activity'] = {**result['activity'], 'name': 'drink water', 'score': .80}
            elif len(self.clips) == 3:
                result['activity'] = {**result['activity'], 'name': 'eat meal', 'score': .25}
            elif len(self.clips) == 4:
                result['activity'] = {**result['activity'], 'name': 'none', 'score': .70}
            return result

    backend = LateWeakModel()
    node = pose._PoseNode('/cam/segment', None, .35, 30., 'pause_guard',
                          classifier=_FinalClassifier(), segment_backend=backend,
                          activity_idle_flush_s=0.40, activity_segment_pause_s=0.65)
    messages = _publisher(node, node._output_topic).messages
    row = np.array(_standing_row(), np.float32)
    buf = node._segment_buffer

    def push_frames(t_start, n=4):
        for i in range(n):
            ts = t_start + i * .03
            buf.append(b'', ts)
            buf.record_pose(ts, row[None, :4], row[None, 4],
                            row[6:].reshape(1, 17, 3), (480, 640, 3))

    push_frames(100.0, 4)
    # A 0.50s stall triggers preview (0.40s) WITHOUT cutting the segment (< 0.65s).
    now = 100.52
    node._flush_received_segment()
    assert len(messages) == 0
    push_frames(100.59, 4)
    # Complete 8-frame segment confirms at 101.35 without ever emitting two segments.
    now = 101.35
    node._flush_received_segment()
    assert len(messages) == 1
    assert node._last_prediction['received_frames'] == 8

    # Even after >2.5s gap in same episode, weak (<0.35 after >=0.55) or 'none' fragment is suppressed.
    push_frames(106.0, 4)
    now = 106.75
    node._flush_received_segment()
    assert len(messages) == 1

    push_frames(108.0, 4)
    now = 108.75
    node._flush_received_segment()
    assert len(messages) == 1


def test_context_patch_and_res_ctx_head_runs_third_single_view():
    cv2 = pytest.importorskip('cv2')
    from plugins.pose_segment import (
        SegmentActionBackend,
        SegmentBuffer,
        fuse_dual_event_head,
        make_dual_roi_patch,
    )

    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    frame[60:300, 220:420] = (180, 120, 60)
    row = np.array(_standing_row(), np.float32)
    boxes = row[None, :4]
    scores = row[None, 4]
    kps = row[6:].reshape(1, 17, 3)

    patch_no_ctx = make_dual_roi_patch(frame, boxes, scores, kps)
    assert len(patch_no_ctx) == 2

    patch_ctx = make_dual_roi_patch(frame, boxes, scores, kps, ctx_hw=192)
    assert len(patch_ctx) == 3
    assert patch_ctx[2].shape == (192, 192, 3)
    assert patch_ctx[2].dtype == np.uint8

    class BodyStub:
        input_shape = (1, 2, 100, 17, 6)

        def infer(self, blob):
            out = np.zeros((1, 126), dtype=np.float32)
            out[0, 13] = 3.0
            return [out]

    class RgbFeatStub:
        input_shape = (1, 16, 3, 192, 192)

        def __init__(self):
            self.calls = []

        def infer(self, blob):
            self.calls.append(tuple(blob.shape))
            val = float(len(self.calls))
            return [np.full((1, 16, 16), val * 0.1, dtype=np.float32)]

    d_feat = 16
    rank = 8
    raw_head = {
        'head_type': np.array('res_ctx'),
        'ctx_frames': np.array(4, dtype=np.int32),
        'att_f_w': np.zeros((4, 2 * d_feat), dtype=np.float32),
        'att_d_w': np.zeros((4, 2 * d_feat), dtype=np.float32),
        'att_e_w': np.zeros((4, 1), dtype=np.float32),
        'att_e_b': np.zeros((4,), dtype=np.float32),
        'att_out_w': np.zeros((1, 4), dtype=np.float32),
        'att_out_b': np.zeros((1,), dtype=np.float32),
        'gamma_ev': np.zeros((1, 2 * d_feat), dtype=np.float32),
        'gamma_dv': np.zeros((1, 2 * d_feat), dtype=np.float32),
        'fc_w': np.zeros((126, 2 * d_feat), dtype=np.float32),
        'fc_b': np.zeros((126,), dtype=np.float32),
        'ctx_p_w': np.zeros((rank, 2 * d_feat), dtype=np.float32),
        'ctx_c_w': np.full((rank, d_feat), 0.5, dtype=np.float32),
        'ctx_c_b': np.zeros((rank,), dtype=np.float32),
        'ctx_fc_w': np.zeros((126, rank), dtype=np.float32),
        'ctx_fc_b': np.zeros((126,), dtype=np.float32),
    }
    raw_head['ctx_fc_w'][13, :] = 2.0  # boost class 14 when context fires

    rgb_stub = RgbFeatStub()
    backend = SegmentActionBackend(
        None, BodyStub(), rgb_stub, crop_mode='E', rgb_head=raw_head
    )
    assert backend.rgb_ctx_frames == 4
    assert backend.rgb_head['head_type'] == 'res_ctx'

    buf = SegmentBuffer(quiet_s=0.35)
    for i in range(6):
        ts = 10.0 + i * 0.05
        buf.append(b'', ts)
        buf.record_pose(ts, boxes, scores, kps, frame.shape, rgb_patch=patch_ctx)
    records, _ = buf.pop_ready(11.0)
    res = backend.recognize(records, need_image=False)
    assert res is not None
    assert len(rgb_stub.calls) == 3
    assert all(shape == (1, 16, 3, 192, 192) for shape in rgb_stub.calls)
    # Direct head call must be positive on class 14 because context branch fired.
    logits = fuse_dual_event_head(
        np.full((1, 16, d_feat), 0.1, dtype=np.float32),
        np.full((1, 16, d_feat), 0.2, dtype=np.float32),
        np.zeros((1, 16, 3, 192, 192), dtype=np.float32),
        backend.rgb_head,
        fc=np.full((1, 16, d_feat), 0.3, dtype=np.float32),
    )
    assert logits.shape == (1, 126)
    assert float(logits[0, 13]) > 1.0


def test_pose_node_stall_backfill_picks_up_to_three_and_respects_quiet_gap(monkeypatch):
    now = 100.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))

    class BatchModel:
        def __init__(self):
            self.batch_calls = []

        def infer_batch(self, frames):
            self.batch_calls.append(len(frames))
            return []

    node = pose._PoseNode(
        '/cam/bf', BatchModel(), .35, 30., 'bf_test',
        classifier=_FinalClassifier(), segment_backend=_SegmentModel(),
        activity_idle_flush_s=0.24, activity_segment_pause_s=0.35,
        activity_stall_backfill_s=0.20,
    )
    # Feed frames at 30fps over a 0.33s stall (10 frames between t=100.00 and t=100.33).
    node._last_pose_received_at = 100.00
    for i in range(1, 11):
        t = 100.00 + i * 0.033
        node._record_recent_frame(t, f'frame-{i}'.encode())

    picks = node._pick_backfill_frames(100.33)
    assert len(picks) == 3
    assert [t for t, _ in picks] == sorted(t for t, _ in picks)
    assert all(100.00 < t < 100.33 for t, _ in picks)

    # Gap shorter than threshold returns nothing.
    node._last_pose_received_at = 100.20
    assert node._pick_backfill_frames(100.33) == []

    # Quiet gap (>= 0.35s) between two segments must NOT pull frames from previous segment.
    node._last_pose_received_at = 100.00
    node._recent_frames.clear()
    node._recent_frames_bytes = 0
    for t in (100.05, 100.10, 100.55, 100.58):
        node._record_recent_frame(t, b'jpeg')
    picks_seg = node._pick_backfill_frames(100.61)
    assert [t for t, _ in picks_seg] == [100.55, 100.58]


def test_segment_warmup_defer_rgb_and_short_segment_quiet_guard(tmp_path, monkeypatch):
    from utils import tensorrt_runtime
    from plugins.pose_segment import SegmentActionBackend, SegmentBuffer

    warm_calls = []

    class StubEngine:
        input_dtype = np.float32

        def __init__(self, path, **kwargs):
            self.name = path.name
            self.input_shape = ((1, 2, 100, 17, 6) if path.name == 'body.engine'
                                else (1, 16, 3, 192, 192))

        def infer(self, blob):
            warm_calls.append(self.name)
            return [np.zeros((1, 126), dtype=np.float32)]

        def close(self):
            pass

    monkeypatch.setattr(tensorrt_runtime, 'TensorRTEngine', StubEngine)
    (tmp_path / 'models.json').write_text(json.dumps({
        'body_engine': 'body.engine', 'rgb_engine': 'rgb.engine',
    }))
    backend = SegmentActionBackend.from_directory(object(), tmp_path)
    backend.warmup(include_rgb=False)
    assert warm_calls == ['body.engine']
    backend.warmup(include_rgb=True)
    assert warm_calls == ['body.engine', 'body.engine', 'rgb.engine']

    # Short segment (<10 frames) holds across 0.42s hiccup, confirms at 0.50s.
    buf = SegmentBuffer(quiet_s=0.35, short_min_frames=10, short_quiet_s=0.50)
    for i in range(6):
        buf.append(b'', 10.0 + i * 0.04)  # last at 10.20
    assert buf.pop_ready(10.62) is None   # 0.42s < 0.50s
    ready = buf.pop_ready(10.71)          # 0.51s >= 0.50s
    assert ready is not None and ready[1]['received_frames'] == 6
    assert ready[1]['segment_gen'] == 0

    # Normal segment (>=10 frames) still confirms at 0.35s and gets next segment_gen.
    for i in range(12):
        buf.append(b'', 20.0 + i * 0.03)  # last at 20.33
    ready2 = buf.pop_ready(20.69)         # 0.36s >= 0.35s
    assert ready2 is not None and ready2[1]['received_frames'] == 12
    assert ready2[1]['segment_gen'] == 1


def test_backfill_chunks_infer_batch_to_max_batch_two(monkeypatch):
    cv2 = pytest.importorskip('cv2')
    now = 50.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))

    class ChunkTrackingModel:
        def __init__(self):
            self.batch_sizes = []
            self.single_calls = 0

        def infer(self, frame):
            self.single_calls += 1
            return (np.zeros((1, 56, 0), np.float32), (1.0, 0.0, 0.0, 640, 480))

        def infer_batch(self, frames):
            self.batch_sizes.append(len(frames))
            return [
                (np.zeros((1, 56, 0), np.float32), (1.0, 0.0, 0.0, 640, 480))
                for _ in frames
            ]

    model = ChunkTrackingModel()
    node = pose._PoseNode(
        '/cam/bf_chunk', model, .35, 30., 'bf_chunk',
        classifier=_FinalClassifier(), segment_backend=_SegmentModel(),
        activity_idle_flush_s=0.24, activity_segment_pause_s=0.35,
        activity_stall_backfill_s=0.20,
    )
    assert node._backfill_infer_batch_cap == 2
    jpeg = frame_bytes(64, 48)
    node._last_pose_received_at = 50.00
    node._segment_buffer.append(b'', 50.00)
    for i in range(1, 10):
        t = 50.00 + i * 0.033
        node._segment_buffer.append(b'', t)
        node._record_recent_frame(t, jpeg)
    now = 50.33
    node._segment_buffer.append(b'', now)
    node._frame_queue.put_nowait((now, jpeg))

    orig_publish = node._publish
    def stop_after_one(*args, **kwargs):
        orig_publish(*args, **kwargs)
        node._stop_event.set()
    node._publish = stop_after_one
    node._inference_worker()

    # 3 backfilled + 1 current = 4 frames chunked into [2, 2], never bs=4.
    assert model.batch_sizes == [2, 2]
    assert node._backfill_triggers == 1
    assert node._backfill_frames == 3


def test_async_rgb_precompute_reuses_high_coverage_and_refreshes_low_coverage(monkeypatch):
    cv2 = pytest.importorskip('cv2')
    from plugins.pose_segment import SegmentActionBackend, make_dual_roi_patch

    now = 200.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))

    class BodyStub:
        input_shape = (1, 2, 100, 17, 6)

        def infer(self, blob):
            out = np.zeros((1, 126), dtype=np.float32)
            out[0, 5] = 4.0
            return [out]

    class RgbStub:
        input_shape = (1, 32, 3, 192, 192)

        def __init__(self):
            self.calls = 0

        def infer(self, blob):
            self.calls += 1
            out = np.zeros((1, 126), dtype=np.float32)
            out[0, 14] = 5.0
            return [out]

    rgb_stub = RgbStub()
    backend = SegmentActionBackend(None, BodyStub(), rgb_stub, crop_mode='E')
    node = pose._PoseNode(
        '/cam/async_rgb', None, .35, 30., 'async_rgb_test',
        classifier=_FinalClassifier(), segment_backend=backend,
        activity_idle_flush_s=0.24, activity_segment_pause_s=0.35,
    )
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    row = np.array(_standing_row(), np.float32)
    boxes, scores, kps = row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3)
    patch = make_dual_roi_patch(frame, boxes, scores, kps)

    # Segment 0: 8 frames -> compute_rgb_probs at 6 frames (75% >= 70%) -> reused!
    for i in range(8):
        ts = 200.0 + i * 0.04
        node._segment_buffer.append(b'', ts)
        node._segment_buffer.record_pose(ts, boxes, scores, kps, frame.shape, rgb_patch=patch)
    gen0, posed6 = node._segment_buffer.snapshot_posed()
    rgb_probs, obs_cnt = backend.compute_rgb_probs(posed6[:6])
    assert obs_cnt == 6 and rgb_probs.shape == (126,)
    assert rgb_stub.calls == 1
    with node._rgb_lock:
        node._rgb_res = (gen0, obs_cnt, rgb_probs)

    now = 201.0  # confirm segment 0 (8 obs -> 6/8 = 75% >= 70%)
    node._flush_received_segment()
    assert node._last_prediction['rgb_precomputed'] is True
    assert rgb_stub.calls == 1  # RGB engine was NOT re-invoked at flush!

    # Segment 1: 12 frames with early RGB from only 6 frames (50% < 70%) -> refreshes!
    for i in range(12):
        ts = 205.0 + i * 0.04
        node._segment_buffer.append(b'', ts)
        node._segment_buffer.record_pose(ts, boxes, scores, kps, frame.shape, rgb_patch=patch)
    gen1 = node._segment_buffer.current_gen()
    assert gen1 == gen0 + 1
    with node._rgb_lock:
        node._rgb_res = (gen1, 6, rgb_probs)
    now = 206.0  # confirm segment 1 (12 obs -> 6/12 = 50% < 70%)
    node._flush_received_segment()
    assert node._last_prediction['rgb_precomputed'] is False
    assert rgb_stub.calls == 2  # Full-segment RGB refreshed!


def test_segment_action_bundle_pins_and_jetpack_cache_isolation(monkeypatch):
    import re
    from utils import model_downloader as md

    bundles = md.SEGMENT_ACTION_MODEL_BUNDLES
    assert set(bundles) == {'jp511', 'jp61'}
    expected_files = {
        'stgcnpp-action126-v4-ntu200-y26s.engine',
        'mnv2-sv16f-crope-ctx4-v4-feat.engine',
        'mnv2-sv16f-crope-ctx4-v4-head.npz',
        'models.json',
    }
    rev_re = re.compile(r'/resolve/[0-9a-f]{40}/')
    for family, spec in bundles.items():
        assert rev_re.search(spec['base_url']), (family, spec['base_url'])
        assert '/master/' not in spec['base_url']
        assert set(spec['files']) == expected_files
        for fname, meta in spec['files'].items():
            assert isinstance(meta['size'], int) and meta['size'] > 0, (family, fname)
            assert re.fullmatch(r'[0-9a-f]{64}', meta['sha256']), (family, fname)

    # Distinct TensorRT engine plans across JetPack lines, shared manifest/head.
    assert (bundles['jp511']['files']['stgcnpp-action126-v4-ntu200-y26s.engine']['sha256']
            != bundles['jp61']['files']['stgcnpp-action126-v4-ntu200-y26s.engine']['sha256'])
    assert (bundles['jp511']['files']['mnv2-sv16f-crope-ctx4-v4-feat.engine']['sha256']
            != bundles['jp61']['files']['mnv2-sv16f-crope-ctx4-v4-feat.engine']['sha256'])
    assert bundles['jp511']['files']['models.json'] == bundles['jp61']['files']['models.json']
    assert (bundles['jp511']['files']['mnv2-sv16f-crope-ctx4-v4-head.npz']
            == bundles['jp61']['files']['mnv2-sv16f-crope-ctx4-v4-head.npz'])

    seen = []
    def fake_bundle(name, model_dir, base_url, files, progress_cb=None):
        seen.append((name, model_dir, base_url, progress_cb))
        return {k: f'{model_dir}/{k}' for k in files}

    monkeypatch.setattr(md, 'ensure_verified_bundle', fake_bundle)
    monkeypatch.setattr(md, 'require_models_subpath', lambda path: path)
    sentinel = lambda *a: None
    p5 = md.ensure_segment_action_model('/models/action-segment', family='jp511', progress_cb=sentinel)
    p6 = md.ensure_segment_action_model('/models/action-segment', family='jp61', progress_cb=sentinel)
    assert p5['models.json'] == '/models/action-segment/jp511/models.json'
    assert p6['models.json'] == '/models/action-segment/jp61/models.json'
    assert [(x[0], x[1], x[3]) for x in seen] == [
        ('action-segment/jp511', '/models/action-segment/jp511', sentinel),
        ('action-segment/jp61', '/models/action-segment/jp61', sentinel),
    ]


def test_cold_segment_start_downloads_pinned_bundle_and_reports_live_info_progress(tmp_path, monkeypatch):
    from utils import model_downloader as md

    monkeypatch.delenv('ACTION_SEGMENT_MODEL_DIR', raising=False)
    monkeypatch.delenv('POSE_ACTIVITY_MODE', raising=False)

    # Default rolling mode must never trigger segment bundle downloads.
    monkeypatch.setattr(
        md, 'ensure_segment_action_model',
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('rolling mode downloaded segment bundle')),
    )
    rolling_plugin, _ = _plugin({'activity_mode': 'rolling'})
    rolling_plugin._classifier_for = lambda cfg: _FinalClassifier()
    rolling_plugin.dispatch('pose', {'action': 'start', 'input_topic': '/cam/rolling'})
    try:
        assert rolling_plugin._nodes['/cam/rolling']._segment_backend is None
    finally:
        rolling_plugin.dispatch('pose', {'action': 'stop', 'input_topic': '/cam/rolling'})

    # Segment mode lazily downloads once, releases _nodes_lock so concurrent
    # info() calls from another thread never block, and surfaces live progress.
    manifest = tmp_path / 'models.json'
    manifest.write_text('{}')
    download_calls = []
    warmups = []
    observed_infos = []

    def _info_from_other_thread(plugin, topic):
        box = {}
        def _worker():
            box['info'] = plugin.dispatch('pose', {'action': 'info', 'input_topic': topic})
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=1.5)
        assert not t.is_alive(), 'info() blocked on _nodes_lock during segment bundle load'
        return box['info']

    plugin, _ = _plugin({'activity_mode': 'segment'})
    plugin._classifier_for = lambda cfg: _FinalClassifier()

    def fake_ensure(model_dir='/models/action-segment', family=None, progress_cb=None):
        download_calls.append(model_dir)
        assert progress_cb is not None
        progress_cb(40, 7.2, 18.0)
        info = _info_from_other_thread(plugin, '/cam/seg')
        observed_infos.append(info)
        return {'models.json': str(manifest)}

    class _FakeLoadedBackend:
        def __init__(self):
            self.body = SimpleNamespace(
                _engine=SimpleNamespace(input_shape=(1, 2, 100, 17, 6))
            )
            self.rgb_engine = SimpleNamespace(input_shape=(1, 32, 3, 192, 192))
            self.ntu_rgb_weight = 0.35
            self.crop_mode = 'E'
            self.router = None
            self.rgb_head = None

        def warmup(self, include_rgb=False):
            warmups.append(include_rgb)
            observed_infos.append(_info_from_other_thread(plugin, '/cam/seg'))

    monkeypatch.setattr(md, 'ensure_segment_action_model', fake_ensure)
    monkeypatch.setattr(
        SegmentActionBackend, 'from_directory',
        classmethod(lambda cls, pose_model, directory: _FakeLoadedBackend()),
    )

    plugin.dispatch('pose', {'action': 'start', 'input_topic': '/cam/seg'})
    try:
        assert download_calls == ['/models/action-segment']
        assert warmups == [False]
        assert len(observed_infos) == 2
        assert observed_infos[0]['state'] == 'loading'
        assert 'action-segment' in observed_infos[0]['desc'] and '40%' in observed_infos[0]['desc']
        assert observed_infos[1]['state'] == 'loading'
        assert 'Loading action-segment' in observed_infos[1]['desc']
        post = plugin.dispatch('pose', {'action': 'info', 'input_topic': '/cam/seg'})
        assert post['state'] == 'running'
        assert plugin._model_loading is False
        assert plugin._model_load_status is None
    finally:
        plugin.dispatch('pose', {'action': 'stop', 'input_topic': '/cam/seg'})

    # Explicit ACTION_SEGMENT_MODEL_DIR bypasses downloader completely.
    monkeypatch.setenv('ACTION_SEGMENT_MODEL_DIR', str(tmp_path))
    monkeypatch.setattr(
        md, 'ensure_segment_action_model',
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('local override should not download')),
    )
    local_plugin, _ = _plugin({'activity_mode': 'segment'})
    local_plugin._classifier_for = lambda cfg: _FinalClassifier()
    local_plugin.dispatch('pose', {'action': 'start', 'input_topic': '/cam/local'})
    try:
        assert local_plugin._nodes['/cam/local']._segment_backend is not None
    finally:
        local_plugin.dispatch('pose', {'action': 'stop', 'input_topic': '/cam/local'})

    # Cold start with _model=None stays state="loading" across BOTH pose load
    # and segment bundle download without flashing "idle" in between.
    monkeypatch.delenv('ACTION_SEGMENT_MODEL_DIR', raising=False)
    cold_plugin, _ = _plugin({'activity_mode': 'segment'})
    cold_plugin._model = None
    cold_plugin._classifier_for = lambda cfg: _FinalClassifier()
    bg_states = []
    release_seg = threading.Event()
    entered_seg = threading.Event()

    def fake_ensure_pose():
        bg_states.append(cold_plugin.dispatch('pose', {'action': 'info', 'input_topic': '/cam/cold'})['state'])
        cold_plugin._model = _plugin()[0]._model

    def fake_ensure_seg(model_dir='/models/action-segment', family=None, progress_cb=None):
        progress_cb(60, 10.8, 18.0)
        entered_seg.set()
        assert release_seg.wait(2.0)
        return {'models.json': str(manifest)}

    cold_plugin._ensure_model = fake_ensure_pose
    monkeypatch.setattr(md, 'ensure_segment_action_model', fake_ensure_seg)
    try:
        resp = cold_plugin.dispatch('pose', {'action': 'start', 'input_topic': '/cam/cold'})
        assert resp['state'] == 'loading'
        assert entered_seg.wait(2.0)
        mid = cold_plugin.dispatch('pose', {'action': 'info', 'input_topic': '/cam/cold'})
        assert mid['state'] == 'loading'
        assert 'action-segment' in mid['desc'] and '60%' in mid['desc']
        release_seg.set()
        assert _wait_until(
            lambda: cold_plugin.dispatch('pose', {'action': 'info', 'input_topic': '/cam/cold'})['state'] == 'running',
            timeout=3.0,
        )
        assert bg_states == ['loading']
    finally:
        release_seg.set()
        cold_plugin.dispatch('pose', {'action': 'stop', 'input_topic': '/cam/cold'})
