"""Synthetic ROS samples + actual JPEG/FFmpeg + the production plugin handoff.

No synthetic feed endpoint is shipped in production, and no hardware is driven.
"""
import copy
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from test_video_task import plugin, video, ffmpeg, call, completed, imported, recipe, model_server
from plugins.video_task.capture import DemoCapturePlugin
from plugins.video_task import motion, media, skill_recipe
from vision_stubs import _FakeNode


@pytest.fixture
def recorder(plugin, monkeypatch):
    topics = [('/capture/ego', ['sensor_msgs/msg/CompressedImage']),
              ('/capture/exo', ['sensor_msgs/msg/CompressedImage']),
              ('/capture/state', ['std_msgs/msg/String'])]
    monkeypatch.setattr(_FakeNode, 'get_topic_names_and_types', lambda _: topics, raising=False)
    executor = SimpleNamespace(add_node=lambda n: None, remove_node=lambda n: None)
    r = DemoCapturePlugin({}, executor, plugin)
    clock = [100.]
    r.clock = lambda: clock[0]
    r.test_clock = clock
    yield r
    r.shutdown()


def invoke(r, action, **args):
    value = r.dispatch(r.PREFIX, {'action': action, **args})
    assert 'error' not in value, value
    return value


def sample(r, index, *, external=True, state=True):
    r.test_clock[0] = 100 + .12*index
    frame = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(frame, (10+index, 40), (30+index, 70), (30, 80, 220), -1)
    jpeg = cv2.imencode('.jpg', frame)[1].tobytes()
    msg = SimpleNamespace(data=jpeg, header=SimpleNamespace(stamp=SimpleNamespace(sec=40, nanosec=index*120000000)))
    r.receive('/capture/ego', msg)
    if external:
        r.receive('/capture/exo', msg)
    if state:
        payload = {'position': [index*.01, index*.02], 'gripper_width': .08-index*.003,
                   'units': {'position': 'rad', 'gripper_width': 'm'}}
        r.receive('/capture/state', SimpleNamespace(data=json.dumps(payload)))


def connected(r):
    invoke(r, 'start', input_topics=['/capture/ego', '/capture/exo', '/capture/state'])
    invoke(r, 'configure', sources=[{'topic': '/capture/ego', 'role': 'ego', 'name': '第一视角'}])
    sample(r, 0)


def finish(r):
    invoke(r, 'record_stop')
    r.worker.join(20)
    assert not r.worker.is_alive()
    assert r.session['state'] == 'done', r.session
    return r.session['draft_id']


def test_capture_real_encoding_handoff_quality_and_no_motion(recorder, plugin):
    r = recorder
    connected(r)
    assert not r.root.exists()  # observing is not recording
    assert not r.node.publishers  # no actuator or command publication
    assert invoke(r, 'info')['busy'] is False
    invoke(r, 'record_start', name='合成采集验收，非实机')
    for i in range(1, 20):
        sample(r, i, external=i not in (5, 6, 7, 8, 9))
    invoke(r, 'mark', label='人工干预测试')
    original_node = r.node
    assert 'error' in r.dispatch(r.PREFIX, {'action': 'start', 'input_topics': ['/other']})
    assert r.node is original_node
    did = finish(r)
    draft = call(plugin, 'inspect', draft_id=did)['draft']
    assert draft['recipe'] is None and len(draft['streams']) == 2
    assert draft['streams'][0]['role'] == 'ego'
    assert draft['capture']['time_basis'] == 'host_receive_monotonic'
    assert any(q['gaps'] for q in draft['capture']['quality'])
    assert draft['capture']['marks'][0]['source'] == 'human'
    evidence = call(plugin, 'motion', draft_id=did)['motion']
    assert evidence['signals'][0]['stats']
    widths = next(s for s in evidence['signals'][0]['stats'] if s['channel'] == 'gripper_width')
    assert widths['first'] > widths['last']
    stream = draft['streams'][0]
    samples = media.sample(media.asset_dir(plugin.root, did), stream['media'], 0, stream['media']['duration_s'], 3)
    assert len(samples) == 3 and samples[0]['jpeg'].startswith(b'\xff\xd8')
    raw_video = [json.loads(line) for line in (r.root/did/'0'/'samples.jsonl').read_text().splitlines()]
    pts = media.timeline(media.asset_dir(plugin.root, did))['pts_s']
    assert len(pts) == len(raw_video)  # encoding must not fabricate a tail frame
    assert all(abs(t-(row['time_s']-raw_video[0]['time_s'])) <= .002 for t, row in zip(pts, raw_video))
    assert invoke(r, 'retry', capture_id=did) == {'state': 'done', 'draft_id': did}
    assert len(list(plugin.root.glob('*/draft.json'))) == 1
    raw = json.loads((r.root/did/'2'/'samples.jsonl').read_text().splitlines()[0])
    assert raw['payload']['units']['gripper_width'] == 'm'
    assert raw['time_s'] == pytest.approx(.12)
    assert invoke(r, 'info')['busy'] is False
    invoke(r, 'stop')
    assert original_node.destroyed


def test_disconnected_stale_invalid_and_duration_limit(recorder):
    r = recorder
    assert 'error' in r.dispatch(r.PREFIX, {'action': 'record_start'})
    connected(r)
    r.test_clock[0] += 3
    assert '新鲜' in r.dispatch(r.PREFIX, {'action': 'record_start'})['error']
    sample(r, 30)
    invoke(r, 'record_start')
    sample(r, 31)
    sample(r, 32)
    r.test_clock[0] += .2
    r.receive('/capture/ego', SimpleNamespace(data=b'not jpeg'))
    assert r.session['sources']['/capture/ego']['rejected'] == 1
    assert 'error' in r.dispatch(r.PREFIX, {'action': 'configure', 'sources': []})
    r.test_clock[0] += 121
    r.receive('/capture/state', SimpleNamespace(data='{"position":[1]}'))
    assert r.session['state'] == 'processing'
    r.worker.join(20)
    assert r.session['state'] == 'done'
    assert '上限' in r.session['finish_reason']


def test_failed_recording_retry_and_restart_keeps_raw(recorder, plugin):
    r = recorder
    connected(r)
    invoke(r, 'record_start')
    sample(r, 1)
    invoke(r, 'record_stop')
    r.worker.join(10)
    assert r.session['state'] == 'failed'  # only one frame, cannot invent footage
    cid = r.session['id']
    path = r.root/cid/'manifest.json'
    data = json.loads(path.read_text())
    data['state'] = 'recording'
    path.write_text(json.dumps(data))
    restarted = DemoCapturePlugin({}, r.executor, plugin)
    try:
        record = restarted.info()['recent'][0]
        assert record['state'] == 'interrupted'
        assert (r.root/cid/'0'/'000000.jpg').is_file()
        assert restarted.session is None and restarted.node is None
    finally:
        restarted.shutdown()


def test_failed_handoff_can_retry_without_overwriting_published_draft(recorder, plugin, monkeypatch):
    r = recorder
    connected(r)
    invoke(r, 'record_start')
    for i in range(1, 8):
        sample(r, i)
    original = plugin.accept_capture
    def fail(*args):
        raise OSError('test handoff failure')
    monkeypatch.setattr(plugin, 'accept_capture', fail)
    invoke(r, 'record_stop');r.worker.join(15)
    assert r.session['state'] == 'failed'
    cid = r.session['id']
    assert not (plugin.root/cid/'draft.json').exists()
    assert (r.root/cid/'0'/'000000.jpg').exists()
    monkeypatch.setattr(plugin, 'accept_capture', original)
    invoke(r, 'retry', capture_id=cid);r.worker.join(15)
    assert r.session['state'] == 'done'
    draft = call(plugin, 'inspect', draft_id=cid)['draft']
    changed = call(plugin, 'rename', draft_id=cid, expected_revision=draft['revision'], name='用户后来改名')['draft']
    invoke(r, 'retry', capture_id=cid)
    assert call(plugin, 'inspect', draft_id=cid)['draft']['revision'] == changed['revision']
    assert call(plugin, 'inspect', draft_id=cid)['draft']['name'] == '用户后来改名'


def test_storage_failure_does_not_leave_recording_or_processing_stuck(recorder, monkeypatch):
    r = recorder
    connected(r)
    def no_space():
        raise OSError('test disk full')
    with monkeypatch.context() as patch:
        patch.setattr(r, '_persist', no_space)
        assert 'error' in r.dispatch(r.PREFIX, {'action': 'record_start'})
        assert r.session['state'] == 'failed'
    invoke(r, 'record_start')
    for i in range(1, 5):
        sample(r, i)
    with monkeypatch.context() as patch:
        patch.setattr(r, '_persist', no_space)
        invoke(r, 'record_stop');r.worker.join(15)
        assert not r.worker.is_alive()
        assert r.session['state'] == 'failed'
        assert '无法写入' in r.session['error']


@pytest.mark.parametrize('mode', ['frames', 'native'])
def test_motion_and_synced_view_reach_both_model_routes(plugin, video, model_server, mode):
    d = imported(plugin, video)
    call(plugin, 'add_stream', draft_id=d['id'], expected_revision=d['revision'], video_path=str(video), role='ego', name='辅助')
    completed(plugin)
    d = call(plugin, 'inspect', draft_id=d['id'])['draft']
    sid = d['streams'][1]['id']
    d = call(plugin, 'sync', draft_id=d['id'], stream_id=sid, offset_s=0, confirmed=True, expected_revision=d['revision'])['draft']
    values = {'kind': 'trajectory3d', 'units': 'm', 'coordinate_frame': 'test_fixture_only',
              'names': ['robot0'], 'metadata': {'format': 'synthetic test data'},
              'frames': [{'time_s': i*.1, 'points': [[i*.01, 0, 0, 1]], 'widths': [.08-i*.001]} for i in range(35)]}
    with plugin.lock:
        plugin._store_track(d, d['streams'][0], values, 'imported')
        plugin._commit(d, 'import_data')
    plugin.cfg.update(endpoint=model_server['endpoint'], model='contract-test', native_protocol='qwen_video_url')
    call(plugin, 'analyze', draft_id=d['id'], mode=mode, frame_count=3)
    completed(plugin)
    payload = model_server['requests'][-1]
    text = json.dumps(payload, ensure_ascii=False)
    assert 'gripper_width_m' in text and 'test_fixture_only' in text
    assert '同次示范辅助视角' in text
    d = call(plugin, 'inspect', draft_id=d['id'])['draft']
    candidate = d['analyses'][-1]
    assert candidate['evidence']['supporting_frame_count'] == 2
    assert candidate['evidence']['motion']['references']
    d = call(plugin, 'update', draft_id=d['id'], expected_revision=d['revision'], recipe=candidate['recipe'])['draft']
    source = plugin._skill_source(d)
    assert source['motion_evidence'][0]['tracks'][0]['points'][0]['gripper_width_m']
    assert 'samples' not in source['motion_evidence'][0]['tracks'][0]
    # A later edit of auxiliary sync makes the saved evidence stale.
    d = call(plugin, 'sync', draft_id=d['id'], stream_id=sid, offset_s=.1, confirmed=True, expected_revision=d['revision'])['draft']
    result = plugin.dispatch(plugin.PREFIX, dict(action='update', draft_id=d['id'], expected_revision=d['revision'], recipe=candidate['recipe']))
    assert '同步已改变' in result['error']


def test_motion_reference_hash_and_unconfirmed_layers(plugin, video):
    d = imported(plugin, video)
    values = {'kind': 'points2d', 'units': 'normalized', 'coordinate_frame': 'image',
              'names': ['point'], 'frames': [{'time_s': 0, 'points': [None]}, {'time_s': 1, 'points': [[.5, .4, 1]]}]}
    summary = plugin._store_track(d, d['streams'][0], values, 'imported')
    evidence = motion.collect(plugin.root, d, 0, 2)
    assert evidence['tracks'][0]['points'][0]['missing_frames'] == 1
    motion.validate_references(plugin.root, d, evidence['references'])
    path = plugin.root/d['id']/'tracks'/(summary['id']+'.json')
    altered = json.loads(path.read_text());altered['frames'][1]['points'][0][0] = .7
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='动作证据已改变'):
        motion.validate_references(plugin.root, d, evidence['references'])
    d['streams'][0]['alignment_confirmed'] = False
    assert not motion.collect(plugin.root, d, 0, 2)['tracks']
