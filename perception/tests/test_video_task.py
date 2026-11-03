"""Real codec + plugin + HTTP boundary; model replies are explicit test doubles.

Run separately from actucore tests (each owns a top-level `plugins` package):
python -m pytest perception/tests/test_video_task.py -q
Requires real OpenCV and ffmpeg (or imageio-ffmpeg on a development host).
"""
import base64
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from plugins.video_task import VideoTaskPlugin
from plugins.video_task import analysis, media


@pytest.fixture(scope='module')
def ffmpeg():
    found = shutil.which('ffmpeg')
    return found or pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()


@pytest.fixture
def video(tmp_path):
    import cv2
    if not hasattr(cv2, 'VideoCapture'):
        pytest.skip('real OpenCV is required for video decoding evidence')
    path = tmp_path / 'uploads' / 'motion.mp4'
    path.parent.mkdir()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12, (320, 240))
    assert writer.isOpened()
    for i in range(48):
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[:, :] = (100, 20, 10)
        cv2.rectangle(frame, (10+i*5, 100), (40+i*5, 130), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def plugin(tmp_path, ffmpeg, video):
    p = VideoTaskPlugin({'data_dir': str(tmp_path / 'drafts'), 'upload_roots': [str(video.parent)], 'ffmpeg': ffmpeg}, None)
    yield p
    p.shutdown()


def call(plugin, action, **args):
    result = plugin.dispatch('video_task', dict(action=action, **args))
    assert 'error' not in result, result
    return result


def completed(plugin):
    plugin.worker.join(timeout=15)
    assert not plugin.worker.is_alive()
    assert plugin.job['state'] == 'done', plugin.job


def imported(plugin, video):
    draft = call(plugin, 'new', goal='观察方块向右移动')['draft']
    call(plugin, 'import', draft_id=draft['id'], video_path=str(video))
    completed(plugin)
    return call(plugin, 'inspect', draft_id=draft['id'])['draft']


def recipe():
    return {'goal': '观察方块向右移动', 'objects': ['红色方块'], 'uncertainties': ['测试服务固定输出，不是模型理解结果'],
            'steps': [{'id': 's1', 'title': '向右移动', 'start_s': 1, 'end_s': 2,
                       'basis': 'observed', 'preconditions': [], 'postconditions': ['方块出现在右侧'],
                       'depends_on': [], 'notes': '仅用于请求格式和版本回写试验'}]}


@pytest.fixture
def model_server():
    state = {'requests': [], 'recipe': recipe(), 'entered': threading.Event(), 'release': threading.Event(), 'wait': False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            state['requests'].append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            state['entered'].set()
            if state['wait']:
                state['release'].wait(10)
            data = json.dumps({'model': 'contract-test-only', 'choices': [{'message': {'content': json.dumps(state['recipe'])}, 'finish_reason': 'stop'}],
                               'usage': {'prompt_tokens': 12, 'completion_tokens': 34, 'total_tokens': 46}}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state['endpoint'] = f'http://127.0.0.1:{server.server_port}/v1'
    yield state
    state['release'].set()
    server.shutdown()
    server.server_close()
    thread.join()


def test_actual_decode_sample_clip_and_mcp_images(plugin, video, ffmpeg):
    import cv2
    draft = imported(plugin, video)
    assert draft['media']['duration_s'] == 4
    directory = media.asset_dir(plugin.root, draft['id'])
    frames = media.sample(directory, draft['media'], 0, 4, 3)
    centers = []
    for frame in frames:
        image = cv2.imdecode(np.frombuffer(frame['jpeg'], np.uint8), cv2.IMREAD_COLOR)
        centers.append(np.nonzero(image[:, :, 2] > 180)[1].mean())
    assert centers[0] < centers[1] < centers[2]
    assert [f['time_s'] for f in frames] == sorted(f['time_s'] for f in frames)
    clip = media.clip(directory, draft['media'], 1, 3, threading.Event(), ffmpeg)
    assert media.probe(clip)['duration_s'] == pytest.approx(2, abs=.1)
    observed = call(plugin, 'observe', draft_id=draft['id'], frame_count=3)['_mcp_content']
    assert [b['type'] for b in observed] == ['text', 'image', 'image', 'image']
    assert base64.b64decode(observed[1]['data']).startswith(b'\xff\xd8')


def test_both_model_encodings_same_window_and_persistence(plugin, video, model_server):
    draft = imported(plugin, video)
    call(plugin, 'config', endpoint=model_server['endpoint'], model='contract-test-only', native_protocol='qwen_video_url', api_key='test-secret')
    for mode in ('frames', 'native'):
        call(plugin, 'analyze', draft_id=draft['id'], mode=mode, start_s=1, end_s=3, frame_count=4, fps=2)
        completed(plugin)
    requests = model_server['requests']
    frames, native = [r['messages'][1]['content'] for r in requests]
    assert len([c for c in frames if c['type'] == 'image_url']) == 4
    assert '1.000' in frames[1]['text']
    assert native[0]['type'] == 'video_url' and native[0]['fps'] == 2
    assert base64.b64decode(native[0]['video_url']['url'].split(',')[1])[4:8] == b'ftyp'
    assert '1.000' in native[1]['text']  # original clip offset is given explicitly
    reload = VideoTaskPlugin(plugin.cfg, None)
    draft = call(reload, 'inspect', draft_id=draft['id'])['draft']
    assert [a['evidence']['mode'] for a in draft['analyses']] == ['frames', 'native']
    assert draft['analyses'][0]['evidence']['usage']['total_tokens'] == 46
    assert draft['recipe']['steps'] == []  # Candidates do not overwrite the author.
    assert draft['analyses'][0]['recipe']['steps'][0]['title'] == recipe()['steps'][0]['title']
    for candidate in draft['analyses']:
        assert candidate['recipe']['steps'][0]['evidence_ids'] == [candidate['evidence']['id']]
        assert candidate['evidence']['media_sha256'] == draft['media']['sha256']
    assert 'test-secret' not in json.dumps(call(plugin, 'info'))
    assert 'test-secret' not in json.dumps(plugin.get_tools())


def test_no_api_no_silent_fallback_failure_preserves_recipe(plugin, video, model_server):
    draft = imported(plugin, video)
    call(plugin, 'analyze', draft_id=draft['id'], mode='frames')
    plugin.worker.join(5)
    assert plugin.job['state'] == 'failed' and '配置' in plugin.job['error']
    call(plugin, 'config', endpoint=model_server['endpoint'], model='contract-test-only')
    call(plugin, 'analyze', draft_id=draft['id'], mode='native')
    plugin.worker.join(5)
    assert '没有切换' in plugin.job['error'] and model_server['requests'] == []
    model_server['recipe']['steps'][0]['end_s'] = 999
    call(plugin, 'analyze', draft_id=draft['id'], mode='frames')
    plugin.worker.join(5)
    assert plugin.job['state'] == 'failed'
    current = call(plugin, 'inspect', draft_id=draft['id'])['draft']
    assert current['revision'] == draft['revision'] and current['recipe'] == draft['recipe']


def test_cancellation_discards_late_provider_result(plugin, video, model_server):
    draft = imported(plugin, video)
    call(plugin, 'config', endpoint=model_server['endpoint'], model='contract-test-only')
    model_server['wait'] = True
    call(plugin, 'analyze', draft_id=draft['id'], mode='frames')
    assert model_server['entered'].wait(5)
    call(plugin, 'stop')
    model_server['release'].set()
    plugin.worker.join(5)
    assert plugin.job['state'] == 'cancelled'
    assert call(plugin, 'inspect', draft_id=draft['id'])['draft']['revision'] == draft['revision']


def test_edit_bind_two_robot_shapes_feedback_save_and_no_hardware(plugin, video):
    draft = imported(plugin, video)
    draft = call(plugin, 'update', draft_id=draft['id'], expected_revision=draft['revision'], recipe=recipe())['draft']
    assert 'error' in plugin.dispatch('', {'action': 'update', 'draft_id': draft['id'], 'expected_revision': 0, 'recipe': recipe()})
    for name, tool, arguments, required in [('arm', 'mcp__arm__reach', {'pose': [1, 2, 3]}, ['pose']),
                                             ('base', 'mcp__base__move', {'distance': .1}, ['distance'])]:
        bindings = {'s1': {'tool': tool, 'arguments': arguments, 'completion': '实测位置', 'stop_tool': tool, 'stop_arguments': arguments}}
        draft = call(plugin, 'bind', draft_id=draft['id'], expected_revision=draft['revision'], profile=name, bindings=bindings)['draft']
        assert not call(plugin, 'check', draft_id=draft['id'], profile=name, tools=[])['ready_for_review']
        assert call(plugin, 'check', draft_id=draft['id'], profile=name,
                    tools=[{'name': tool, 'inputSchema': {'required': required}}])['ready_for_review']
    draft = call(plugin, 'record', draft_id=draft['id'], record=dict(profile='base', step_id='s1', outcome='unknown', evidence='本地测试没有连接机器人', source='test'))['draft']
    assert len(draft['profiles']) == 2
    assert draft['records'][0]['outcome'] == 'unknown'
    record = draft['records'][0]
    assert record['recipe_revision'] == draft['recipe_revision']
    assert record['binding_revision'] == draft['profile_revisions']['base']
    historic = call(plugin, 'inspect', draft_id=draft['id'], view_revision=record['binding_revision'])['version']
    assert historic['profiles']['base'] == draft['profiles']['base']
    skill = call(plugin, 'save', draft_id=draft['id'])['skill']
    assert 'mcp__arm__reach' in skill['instruction'] and 'mcp__base__move' in skill['instruction']
    assert 'error' in plugin.dispatch('', {'action': 'agent', 'draft_id': draft['id'], 'intent': 'execute', 'profile': 'base'})
    assert plugin.node is None  # no robot or ROS dependency for authoring
    assert len(call(plugin, 'ui')['html']) < 250000


def test_candidate_requires_adoption_preserves_edits_bindings_and_history(plugin, video, model_server):
    draft = imported(plugin, video)
    manual = recipe()
    manual['steps'][0]['title'] = '人工核对后的步骤'
    draft = call(plugin, 'update', draft_id=draft['id'], expected_revision=draft['revision'], recipe=manual)['draft']
    recipe_revision = draft['revision']
    bindings = {'s1': {'tool': 'example.move', 'arguments': {}, 'completion': '查看真实反馈',
                       'stop_tool': 'example.stop', 'stop_arguments': {}}}
    draft = call(plugin, 'bind', draft_id=draft['id'], expected_revision=draft['revision'],
                 profile='example', bindings=bindings)['draft']
    binding_revision = draft['revision']
    call(plugin, 'config', endpoint=model_server['endpoint'], model='contract-test-only')
    call(plugin, 'analyze', draft_id=draft['id'], mode='frames', start_s=1, end_s=3)
    completed(plugin)
    proposal = call(plugin, 'inspect', draft_id=draft['id'])['draft']
    assert proposal['recipe'] == manual and proposal['profiles'] == {'example': bindings}
    assert proposal['recipe_revision'] == recipe_revision
    candidate = proposal['analyses'][0]['recipe']
    stale = plugin.dispatch('', dict(action='update', draft_id=draft['id'],
                                    expected_revision=binding_revision, recipe=candidate))
    assert '草稿已更新' in stale['error']
    adopted = call(plugin, 'update', draft_id=draft['id'], expected_revision=proposal['revision'],
                   recipe=candidate)['draft']
    assert adopted['recipe'] == candidate and adopted['profiles'] == {}
    assert adopted['recipe_revision'] == adopted['revision']
    old = call(plugin, 'inspect', draft_id=draft['id'], view_revision=binding_revision)['version']
    assert old['recipe'] == manual and old['profiles']['example'] == bindings
    assert (media.asset_dir(plugin.root, draft['id']) / 'analyses' / (proposal['analyses'][0]['id'] + '.json')).is_file()


def test_observation_refs_have_real_source_and_reject_foreign_or_out_of_window(plugin, video):
    draft = imported(plugin, video)
    blocks = call(plugin, 'observe', draft_id=draft['id'], start_s=1, end_s=3, frame_count=4)['_mcp_content']
    observed = json.loads(blocks[0]['text'])['observation']
    assert observed['media_sha256'] == media.sha256(plugin.media_file(draft['id'], 'video')[0])
    assert observed['requested_times_s'][0] == 1
    assert len(observed['sample_times_s']) == 4
    assert len(observed['timestamp_sources']) == 4
    assert draft['media']['source_sha256'] == media.sha256(video)
    linked = recipe()
    linked['steps'][0]['evidence_ids'] = [observed['id']]
    updated = call(plugin, 'update', draft_id=draft['id'], expected_revision=draft['revision'], recipe=linked)['draft']
    assert updated['observations'][0]['id'] == observed['id']
    outside = copy.deepcopy(linked)
    outside['steps'][0]['start_s'] = 0
    bad = plugin.dispatch('', dict(action='update', draft_id=draft['id'],
                                  expected_revision=updated['revision'], recipe=outside))
    assert '超出' in bad['error']
    other = imported(plugin, video)
    bad = plugin.dispatch('', dict(action='update', draft_id=other['id'],
                                  expected_revision=other['revision'], recipe=linked))
    assert '找不到本素材' in bad['error']
    restored = VideoTaskPlugin(plugin.cfg, None)
    assert call(restored, 'inspect', draft_id=draft['id'], observation_id=observed['id'])['observation'] == observed


def test_agent_plan_submits_candidate_with_observation_refs(plugin, video):
    draft = imported(plugin, video)
    unsupported = plugin.dispatch('', dict(action='propose', draft_id=draft['id'],
                                           expected_revision=draft['revision'], recipe=recipe()))
    assert '观察记录' in unsupported['error']
    content = call(plugin, 'observe', draft_id=draft['id'], frame_count=4)['_mcp_content']
    evidence = json.loads(content[0]['text'])['observation']
    proposed = recipe()
    proposed['steps'][0]['evidence_ids'] = [evidence['id']]
    result = call(plugin, 'propose', draft_id=draft['id'],
                  expected_revision=draft['revision'], recipe=proposed)['draft']
    assert result['recipe'] == draft['recipe']
    assert result['analyses'][0]['recipe'] == proposed
    assert result['analyses'][0]['evidence']['observation_ids'] == [evidence['id']]
    call(plugin, 'start')
    call(plugin, 'agent', draft_id=draft['id'], intent='plan')
    assert '用 propose' in json.loads(plugin.publisher.messages[-1])['text']


def test_legacy_read_is_non_mutating_migration_keeps_original_and_unknown_fields(plugin, video):
    draft = imported(plugin, video)
    directory = media.asset_dir(plugin.root, draft['id'])
    path = directory / 'draft.json'
    legacy = json.loads(path.read_text())
    for key in ('schema_version', 'recipe_revision', 'profile_revisions', 'change'):
        legacy.pop(key)
    legacy['private_extension'] = {'preserve': ['未知的旧字段']}
    original = json.dumps(legacy, ensure_ascii=False).encode()
    path.write_bytes(original)
    loaded = call(plugin, 'inspect', draft_id=draft['id'])['draft']
    assert path.read_bytes() == original
    updated = call(plugin, 'update', draft_id=draft['id'], expected_revision=loaded['revision'], recipe=recipe())['draft']
    assert (directory / 'draft.legacy.json').read_bytes() == original
    assert json.loads(path.read_text())['private_extension'] == legacy['private_extension']
    historic = call(plugin, 'inspect', draft_id=draft['id'], view_revision=loaded['revision'])['version']
    assert historic['recipe'] == legacy['recipe']
    assert updated['schema_version'] == 3
    future = json.loads(path.read_text())
    future['schema_version'] = 99
    path.write_text(json.dumps(future))
    unchanged = path.read_bytes()
    assert '不支持' in plugin.dispatch('', dict(action='inspect', draft_id=draft['id']))['error']
    assert path.read_bytes() == unchanged


def test_failed_write_keeps_committed_recipe_and_version(plugin, video, monkeypatch):
    draft = imported(plugin, video)
    write = media.write_json
    def fail_draft(path, value):
        if Path(path).name == 'draft.json':
            raise OSError('disk full in test')
        return write(path, value)
    monkeypatch.setattr(media, 'write_json', fail_draft)
    assert 'error' in plugin.dispatch('', dict(action='update', draft_id=draft['id'],
                                               expected_revision=draft['revision'], recipe=recipe()))
    current = call(plugin, 'inspect', draft_id=draft['id'])['draft']
    assert current['revision'] == draft['revision'] and current['recipe'] == draft['recipe']
    assert all(v['revision'] <= draft['revision'] for v in current['versions'])


def test_feedback_archive_outlives_recent_list(plugin, video):
    draft = imported(plugin, video)
    draft = call(plugin, 'update', draft_id=draft['id'], expected_revision=draft['revision'], recipe=recipe())['draft']
    binding = {'tool': 'test.move', 'arguments': {}, 'completion': '实际观察', 'stop_tool': 'test.stop', 'stop_arguments': {}}
    draft = call(plugin, 'bind', draft_id=draft['id'], expected_revision=draft['revision'],
                 profile='test', bindings={'s1': binding})['draft']
    first = None
    for index in range(101):
        draft = call(plugin, 'record', draft_id=draft['id'],
                     record={'profile': 'test', 'step_id': 's1', 'outcome': 'unknown',
                             'evidence': f'契约测试记录 {index}；没有连接机器人', 'source': 'test'})['draft']
        if first is None:
            first = draft['records'][0]
    assert len(draft['records']) == 100 and first not in draft['records']
    path = media.asset_dir(plugin.root, draft['id']) / 'records' / (first['id'] + '.json')
    assert json.loads(path.read_text()) == first
    version = call(plugin, 'inspect', draft_id=draft['id'], view_revision=first['binding_revision'])['version']
    assert version['profiles']['test']['s1'] == binding


def test_schema_params_survive_agent_action_splitting(plugin):
    schema = plugin.get_tools()[0]['inputSchema']
    for action, spec in schema['x-action-params'].items():
        assert set(spec['required']) <= set(spec['params']) <= schema['properties'].keys(), action
    assert 'recipe' in schema['x-action-params']['update']['params']


@pytest.mark.parametrize('mutation', [
    lambda r: r['steps'][0].update(end_s=float('nan')),
    lambda r: r['steps'][0].update(depends_on=['s1']),
    lambda r: r['steps'][0].update(basis='certain'),
])
def test_invalid_evidence_rejected(mutation):
    r = recipe()
    mutation(r)
    with pytest.raises(ValueError):
        analysis.validate_recipe(r, 4)


def test_path_and_decode_boundaries(plugin, video, tmp_path):
    for bad in ('../secret', 'bad', 'a'*32+'/etc'):
        with pytest.raises(ValueError):
            media.asset_dir(plugin.root, bad)
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(video.read_bytes())
    link = video.parent / 'link.mp4'
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='上传'):
        media.ingest(tmp_path/'asset', path=str(link), roots=[video.parent], cancel=threading.Event())
    invalid = video.parent / 'not-video.mp4'
    invalid.write_text('<html>watch page</html>')
    with pytest.raises(ValueError, match='仅接受'):
        media.ingest(tmp_path/'asset', path=str(invalid), roots=[video.parent], cancel=threading.Event())


def test_perception_http_registration_images_and_authenticated_media(plugin, video, monkeypatch):
    import main
    from utils import file_intake
    draft = imported(plugin, video)
    bundle = main.PerceptionBundle.__new__(main.PerceptionBundle)
    bundle._plugins = [plugin]
    monkeypatch.setattr(main, '_bundle', bundle)
    monkeypatch.setattr(file_intake, 'access_token', lambda: 'local-test-token')
    server = ThreadingHTTPServer(('127.0.0.1', 0), main.make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        req = urllib.request.Request(url+'/mcp', data=json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
            'params': {'name': 'video_task', 'arguments': {'action': 'observe', 'draft_id': draft['id'], 'frame_count': 2}}}).encode())
        with urllib.request.urlopen(req) as response:
            content = json.load(response)['result']['content']
        assert [c['type'] for c in content] == ['text', 'image', 'image']
        media_url = url+f'/workbench/media?tool=video_task&asset_id={draft["id"]}&kind=video'
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(media_url)
        assert error.value.code == 401
        with urllib.request.urlopen(urllib.request.Request(media_url, headers={'X-Access-Token': 'local-test-token'})) as response:
            assert response.headers['Content-Type'] == 'video/mp4'
            assert response.read() == plugin.media_file(draft['id'], 'video')[0].read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_router_catalog_is_public_bounded_cached_and_preserves_config(plugin, monkeypatch):
    import io
    calls = []
    rows = [
        {'model_name': 'text-only', 'supported_endpoint_types': ['openai']},
        {'model_name': 'PR-G-Flash-3.8', 'supported_endpoint_types': ['gemini', 'openai']},
        {'model_name': 'another-model', 'supported_endpoint_types': ['gemini']},
        {'model_name': '../invalid', 'supported_endpoint_types': ['gemini']},
        {'model_name': 'bad-shape', 'supported_endpoint_types': 'gemini'},
    ]
    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(json.dumps({'success': True, 'data': rows}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', fetch)
    call(plugin, 'config', api_key='fixture-secret', model='existing-model')
    before = copy.deepcopy(plugin.cfg)
    result = call(plugin, 'model_catalog', url='https://ignored.invalid', api_key='ignored-key')
    assert result['models'] == ['PR-G-Flash-3.8', 'another-model']
    assert len(calls) == 1 and calls[0] == ('https://cloud.phanthy.com/phanrouterapi/api/pricing', 6)
    assert 'fixture-secret' not in json.dumps(result)
    result['models'].clear()
    assert len(call(plugin, 'model_catalog')['models']) == 2 and len(calls) == 1
    assert plugin.cfg == before
    assert '未验证' in result['note']


@pytest.mark.parametrize('payload', [b'x' * (1024 * 1024 + 1), b'not json', b'[]',
                                     b'{"success":true,"data":[]}', b'{"success":false,"data":[]}'])
def test_catalog_failure_keeps_manual_model_configuration(plugin, monkeypatch, payload):
    import io
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(payload))
    result = plugin.dispatch('video_task', {'action': 'model_catalog'})
    assert '手动填写模型ID' in result['error']
    assert plugin.catalog_cache is None
    call(plugin, 'config', endpoint='http://localhost/v1', model='manual-model')
    assert call(plugin, 'info')['model'] == 'manual-model'
