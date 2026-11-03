"""Real media/numeric-data checks. HTTP model responses are contract doubles."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import zipfile

import numpy as np
import pytest

from test_video_task import ffmpeg, video, plugin, model_server, call, completed, imported, recipe
from plugins.video_task import analysis, importers, media, tracks


def inspect(plugin, draft):
    return call(plugin, 'inspect', draft_id=draft['id'])['draft']


def edit(plugin, draft, action, **args):
    return call(plugin, action, draft_id=draft['id'], expected_revision=draft['revision'], **args)['draft']


def add(plugin, draft, video, role='ego'):
    call(plugin, 'add_stream', draft_id=draft['id'], expected_revision=draft['revision'],
         video_path=str(video), role=role, name=role)
    completed(plugin)
    return inspect(plugin, draft)


def test_data_only_original_pts_and_task_transition(plugin, video):
    d = call(plugin, 'new', name='无任务数据')['draft']
    assert d['recipe'] is None
    d = add(plugin, d, video, 'external')
    source = media.asset_dir(plugin.root, d['id'])
    assert (source/'source.bin').read_bytes() == video.read_bytes()
    pts = media.timeline(source)['pts_s']
    assert len(pts) == 48 and pts[0] == 0
    assert all(a < b for a, b in zip(pts, pts[1:]))
    assert d['streams'][0]['alignment_confirmed']
    assert plugin.media_file(d['id'], 'video')[0] == source/'video.mp4'
    assert 'error' in plugin.dispatch('video_task', dict(action='analyze', draft_id=d['id']))
    d = edit(plugin, d, 'update', recipe=recipe())
    assert d['recipe']['goal'] and len(d['streams']) == 1


def test_multiview_time_evidence_staleness_and_asset_ownership(plugin, video):
    d = add(plugin, imported(plugin, video), video)
    sid = d['streams'][1]['id']
    d = edit(plugin, d, 'sync', stream_id=sid, offset_s=.5, confirmed=True)
    obs = call(plugin, 'observe', draft_id=d['id'], stream_id=sid, start_s=1, end_s=2, frame_count=3)
    evidence = json.loads(obs['_mcp_content'][0]['text'])['observation']
    assert evidence['stream_asset_id'] == sid
    assert evidence['source_start_s'] == .5 and evidence['start_s'] == 1
    assert plugin.media_file(sid, 'video')[0].is_file()
    value = recipe()
    value['steps'][0]['evidence_ids'] = [evidence['id']]
    d = edit(plugin, d, 'update', recipe=value)
    d = edit(plugin, d, 'sync', stream_id=sid, offset_s=.6)
    result = plugin.dispatch('video_task', dict(action='update', draft_id=d['id'], expected_revision=d['revision'], recipe=value))
    assert '同步已改变' in result['error']
    assert inspect(plugin, d)['recipe_revision'] == d['recipe_revision']
    (media.asset_dir(plugin.root, sid)/'owner.json').write_text(json.dumps({'draft_id': '0'*32}))
    with pytest.raises((ValueError, OSError)):
        plugin.media_file(sid, 'video')


def test_drift_requires_real_two_point_alignment(plugin, video):
    d = add(plugin, imported(plugin, video), video)
    args = dict(action='sync', draft_id=d['id'], stream_id=d['streams'][1]['id'],
                expected_revision=d['revision'], offset_s=.1, rate=1.01)
    assert '锚点' in plugin.dispatch('video_task', args)['error']
    args['anchors'] = [{'source_s': 0, 'task_s': .1}, {'source_s': 3, 'task_s': 3.13}]
    assert 'error' not in plugin.dispatch('video_task', args)


def test_real_lk_and_paginated_track(plugin, video):
    d = imported(plugin, video)
    call(plugin, 'track', draft_id=d['id'], expected_revision=d['revision'], start_s=0, end_s=1,
         points=[{'name': 'corner', 'x': 10/320, 'y': 100/240}])
    completed(plugin)
    d = inspect(plugin, d)
    t = d['tracks'][0]
    result = call(plugin, 'tracks', draft_id=d['id'], track_id=t['id'], limit=2)
    assert len(result['track']['frames']) == 2 and result['next_cursor'] == 2
    value = json.loads((media.asset_dir(plugin.root, d['id'])/'tracks'/(t['id']+'.json')).read_text())
    assert value['frames'][0]['points'][0][0] == pytest.approx(10/320, abs=.001)
    assert value['frames'][-1]['points'][0][0] == pytest.approx(70/320, abs=.02)
    assert tracks.validate(value)['frames'] == value['frames']


def test_hand_engine_unavailable_is_explicit(monkeypatch):
    import sys
    for name in ('__main__', 'main'):
        module = sys.modules.get(name)
        if module:
            monkeypatch.setattr(module, '_bundle', None, raising=False)
    with pytest.raises(ValueError, match='pose'):
        tracks.detect_hands(np.zeros((120, 160, 3), np.uint8))


def test_egodex_retains_metric_rotations_confidence_and_intrinsics(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path/'ego.hdf5'
    matrix = np.tile(np.eye(4), (3, 1, 1))
    matrix[:, 0, 3] = [.1, .2, .3]
    with h5py.File(path, 'w') as f:
        f['transforms/camera'] = matrix
        f['transforms/leftHand'] = matrix
        f['confidences/leftHand'] = [1., 0., .8]
        f['camera/intrinsic'] = np.eye(3)
    result = importers.read(path, 'egodex_hdf5', fps=30)
    assert result['coordinate_frame'] == 'ARKit_session' and result['units'] == 'm'
    assert result['frames'][0]['points'][1] == [.1, 0., 0., 1.]
    assert result['frames'][1]['points'][1] is None
    assert len(result['frames'][0]['rotations'][1]) == 9
    assert result['metadata']['intrinsic'] == np.eye(3).tolist()
    assert result['frames'][2]['time_s'] == pytest.approx(2/30)
    with pytest.raises(ValueError, match='帧率'):
        importers.read(path, 'egodex_hdf5')
    with h5py.File(path, 'w') as f:
        f['transforms'] = h5py.ExternalLink('/etc/passwd', '/data')
    with pytest.raises(ValueError, match='链接'):
        importers.read(path, 'egodex_hdf5', fps=30)


@pytest.mark.parametrize('format_name', ['umi_npz', 'umi_zarr'])
def test_umi_episode_pose_widths_and_missing(tmp_path, format_name):
    data = {'robot0_eef_pos': np.array([[0,0,0], [.1,0,0], [.2,0,0], [np.nan,0,0]]),
            'robot0_eef_rot_axis_angle': np.zeros((4,3)), 'robot0_gripper_width': np.array([.05,.06,.07,.08])}
    path = tmp_path/('run.npz' if format_name == 'umi_npz' else 'run.zip')
    if format_name == 'umi_npz':
        np.savez(path, **data, episode_ends=np.array([2,4]))
    else:
        zarr = pytest.importorskip('zarr')
        with zarr.ZipStore(str(path), mode='w') as store:
            root = zarr.group(store)
            for k, v in data.items():
                root['data/'+k] = v
            root['meta/episode_ends'] = np.array([2,4])
    result = importers.read(path, format_name, fps=20, episode=1, coordinate_frame='capture_world')
    assert result['frames'][0]['points'][0] == [.2,0.,0.,1.]
    assert result['frames'][0]['widths'] == [.07]
    assert result['frames'][0]['rotations'] == [[0.,0.,0.]]
    assert result['frames'][1]['points'] == [None]
    assert result['frames'][1]['time_s'] == .05
    assert result['coordinate_frame'] == 'capture_world'


def test_npz_rejects_object_and_forged_shape_before_allocation(tmp_path):
    path = tmp_path/'invalid.npz'
    np.savez(path, robot0_eef_pos=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match='NPZ'):
        importers.read(path, 'umi_npz', fps=30, coordinate_frame='test')
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(payload, {'descr': '<f8', 'fortran_order': False, 'shape': (10**12,)})
    with zipfile.ZipFile(path, 'w') as f:
        f.writestr('robot0_eef_pos.npy', payload.getvalue())
    with pytest.raises(ValueError, match='NPZ'):
        importers.read(path, 'umi_npz', fps=30, coordinate_frame='test')


@pytest.mark.parametrize('mode', ['native', 'frames'])
def test_gemini_wire_contract_with_real_media(plugin, video, mode):
    d = imported(plugin, video)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            requests.append((self.path, dict(self.headers), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            body = json.dumps({'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(recipe())}]}}],
                               'usageMetadata': {'totalTokenCount': 123}}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        call(plugin, 'config', endpoint=f'http://127.0.0.1:{server.server_port}/v1beta', provider='gemini', model='contract-only', api_key='test-only')
        call(plugin, 'analyze', draft_id=d['id'], mode=mode, start_s=1, end_s=3, frame_count=4)
        completed(plugin)
        path, headers, body = requests[0]
        assert path == '/v1beta/models/contract-only:generateContent'
        assert {k.lower():v for k,v in headers.items()}['x-goog-api-key'] == 'test-only'
        media_parts = [p for p in body['contents'][0]['parts'] if 'inlineData' in p]
        assert len(media_parts) == (1 if mode == 'native' else 4)
        assert media_parts[0]['inlineData']['mimeType'] == ('video/mp4' if mode == 'native' else 'image/jpeg')
        if mode == 'native':
            assert media_parts[0]['videoMetadata']['fps'] == 2
        result = inspect(plugin, d)
        assert result['analyses'][0]['evidence']['usage']['total_tokens'] == 123
        assert result['recipe']['steps'] == []
    finally:
        server.shutdown(); server.server_close()


def test_focused_sampling_budget(plugin, video):
    d = imported(plugin, video)
    _, evidence = analysis.build_request({'sampling': 'focused', 'focus_s': 1.2}, media.asset_dir(plugin.root, d['id']),
        d['media'], 'test', 'frames', 0, 4, 8, 2, threading.Event())
    assert evidence['frame_count'] == 8
    assert sum(.6 <= t <= 1.8 for t in evidence['requested_times_s']) >= 4


def test_variable_pts_origin_and_gaps_survive_import(plugin, video, ffmpeg):
    import subprocess
    source = video.with_name('vfr.mp4')
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(video), '-vf',
                    r'setpts=PTS+5/TB+if(gte(N\,12)\,0.5/TB\,0)', '-vsync', 'vfr',
                    '-c:v', 'libx264', str(source)], check=True)
    d = imported(plugin, source)
    directory = media.asset_dir(plugin.root, d['id'])
    index = media.timeline(directory)
    assert index['origin_s'] == pytest.approx(5, abs=.01)
    assert index['pts_s'][12]-index['pts_s'][11] > .5
    samples = media.sample(directory, d['media'], 0, 4, 2, [.9, 1.2])
    assert samples[1]['time_s'] == pytest.approx(index['pts_s'][11], abs=.001)
    assert media.timeline(directory)['pts_s'] == index['pts_s']


def test_improvement_sends_separate_execution_evidence(plugin, video, model_server):
    d = edit(plugin, imported(plugin, video), 'update', recipe=recipe())
    binding={'tool':'test','arguments':{},'stop_tool':'stop','stop_arguments':{},'completion':'test only'}
    d = edit(plugin, d, 'bind', profile='fixture', bindings={'s1':binding})
    d = add(plugin, d, video, 'execution')
    d = call(plugin, 'record', draft_id=d['id'], record={'profile':'fixture','step_id':'s1',
        'outcome':'unknown','evidence':'synthetic test evidence','source':'human',
        'stream_id':d['streams'][-1]['id'],'start_s':0,'end_s':1})['draft']
    call(plugin, 'config', endpoint=model_server['endpoint'], model='contract-only')
    call(plugin, 'improve', draft_id=d['id'], mode='frames', frame_count=4, start_s=1, end_s=3)
    completed(plugin)
    evidence=inspect(plugin,d)['analyses'][-1]['evidence']
    assert evidence['purpose']=='improvement'
    assert evidence['extra_execution_frame_count']==2
    assert evidence['execution_sources'][0]['record_id']==d['records'][0]['id']
    content=model_server['requests'][0]['messages'][1]['content']
    assert sum(c['type']=='image_url' for c in content)==6
    assert '另一次执行' in content[-2]['text']


def test_improvement_requires_evidence_and_preserves_versions(plugin, video):
    d = edit(plugin, imported(plugin, video), 'update', recipe=recipe())
    binding = {'tool': 'move', 'arguments': {'instance_id': 'arm-a', 'speed': .2},
               'stop_tool': 'stop', 'stop_arguments': {'instance_id': 'arm-a'}, 'completion': 'human observation'}
    d = edit(plugin, d, 'bind', profile='arm-a', bindings={'s1': binding})
    check = call(plugin, 'check', draft_id=d['id'], profile='arm-a', tools=[
        {'name': 'move', 'instance_id': 'arm-b'}, {'name': 'stop', 'instance_id': 'arm-a'}])
    assert not check['ready_for_review']
    d = add(plugin, d, video, 'execution')
    assert 'error' in plugin.dispatch('video_task', dict(action='sync', draft_id=d['id'], expected_revision=d['revision'], stream_id=d['streams'][-1]['id']))
    d = call(plugin, 'record', draft_id=d['id'], record={'profile':'arm-a', 'step_id':'s1', 'outcome':'failed',
         'source':'human', 'evidence':'test fixture feedback', 'stream_id':d['streams'][-1]['id'], 'start_s':1, 'end_s':2})['draft']
    revision = d['recipe_revision']
    improved = copy.deepcopy(recipe()); improved['steps'][0]['preconditions'] = ['核对方块位置']
    observation = call(plugin, 'observe', draft_id=d['id'], start_s=1, end_s=2, frame_count=2)
    improved['steps'][0]['evidence_ids'] = [json.loads(observation['_mcp_content'][0]['text'])['observation']['id']]
    d = edit(plugin, d, 'propose', recipe=improved, purpose='improvement')
    candidate = d['analyses'][-1]['id']
    assert '回归' in plugin.dispatch('video_task', dict(action='adopt', draft_id=d['id'], expected_revision=d['revision'], candidate_id=candidate))['error']
    assert 'error' in plugin.dispatch('video_task', dict(action='review_candidate', draft_id=d['id'], expected_revision=d['revision'], candidate_id=candidate, cases=[{'name':'case','status':'passed','evidence':''}]))
    d = edit(plugin, d, 'review_candidate', candidate_id=candidate, cases=[{'name':'fixture contract check','status':'passed','evidence':'human-marked test record; no hardware claim'}])
    d = edit(plugin, d, 'adopt', candidate_id=candidate)
    assert d['recipe']['steps'][0]['preconditions'] == ['核对方块位置']
    assert d['profiles'] == {} and d['records'][0]['recipe_revision'] == revision
    old = call(plugin, 'inspect', draft_id=d['id'], view_revision=revision)['version']
    assert old['recipe']['steps'][0]['preconditions'] == []
