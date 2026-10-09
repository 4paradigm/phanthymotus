import types
import numpy as np
import pytest
import vision_stubs
from vision_stubs import _FakeExecutor
from plugins import visual_depth as vd
from plugins.depthart_runtime import camera_matrix, prepare_image
from plugins.vision_runtime import decode_depth


def camera(fx=900.):
    return {'schema':'motus.camera/1','id':'camera-a','width':1600,'height':900,
            'K':[fx,4.,799.5,0.,fx,449.5,0.,0.,1.]}




def test_preprocess_scales_intrinsics_and_normalizes_rgb():
    frame=np.zeros((900,1600,3),np.uint8); frame[:,:,2]=255
    blob,k=prepare_image(frame,camera(),(864,480))
    expected=np.array(camera()['K'],np.float32).reshape(3,3)
    expected[0]*=864/1600; expected[1]*=480/900
    assert blob.shape==(1,3,480,864)
    np.testing.assert_allclose(k[0],expected)
    np.testing.assert_allclose(blob[0,:,0,0],(np.array([1.,0.,0.])-[.485,.456,.406])/[.229,.224,.225],rtol=1e-6)
    assert decode_depth([np.ones((1,480,864))],None).shape==(480,864)


@pytest.mark.parametrize('bad',[{}, {'K':[1.]*9,'width':1,'height':1},dict(camera(),K=[np.nan]*9),dict(camera(),width=0)])
def test_invalid_camera_rejected(bad):
    with pytest.raises(ValueError):camera_matrix(bad)


def test_unsupported_aspect_rejected():
    with pytest.raises(ValueError,match='aspect ratio'):
        prepare_image(np.zeros((480,640,3),np.uint8),camera(),(864,480))


def make_plugin(cfg=None):
    return vd.VideoDepthPerceptionPlugin(dict(backend='depthart',**(cfg or {})),'test',_FakeExecutor())


def test_model_selection_schema_and_manual_fields():
    p = make_plugin()
    props = p.get_tools()[0]['configSchema']['properties']
    assert props['backend']['scope'] == 'shared'
    assert props['backend']['default'] == 'depthart'
    assert props['backend']['enum'] == ['yolo', 'depthart']
    assert 'oneOf' not in props['backend']
    assert props['fps']['title'] == 'Inference FPS'
    for key in ('cal_a', 'cal_b'):
        assert 'x-show-when' not in props[key]


def test_idle_model_switch_is_lazy_and_resets_calibration():
    p = make_plugin()
    calls = []
    p._model = types.SimpleNamespace(close=lambda: calls.append('closed'))
    p._instance_configs['a'] = dict(fps=3, cal_a=2., cal_b=.2, calibration_preset=vd.CAL_MANUAL)
    p._cal_samples = [{'measured_m': 1.}]
    result = p.dispatch('visual_depth', dict(action='config', backend='yolo'))
    assert result['status'] == 'configured' and p._backend == 'yolo'
    assert p._model is None and calls == ['closed']
    assert p._instance_configs['a'] == {'fps': 3}
    assert (p._cal_a, p._cal_b) == (1., 0.) and not p._cal_samples
    p.dispatch('visual_depth', dict(action='config', backend='yolo'))
    assert calls == ['closed']


@pytest.mark.parametrize('busy', ['node', 'loading', 'closing'])
def test_model_switch_rejected_while_busy(busy):
    p = make_plugin()
    if busy == 'node':p._nodes['a'] = object()
    elif busy == 'loading':p._model_loading = True
    else:p._closing = True
    result = p.dispatch('visual_depth', dict(action='config', backend='yolo'))
    assert result['status'] == 'error' and result['adapter_ok'] is False
    assert p._backend == 'depthart'


def test_model_switch_during_loader_lock_is_nonblocking():
    p = make_plugin()
    with p._model_lock:
        result = p.dispatch('visual_depth', dict(action='config', backend='yolo'))
    assert result['status'] == 'error' and p._backend == 'depthart'


def test_instance_cannot_switch_shared_model():
    p = make_plugin()
    result = p.dispatch('visual_depth', dict(action='config', backend='yolo', instance_id='a'))
    assert result['adapter_ok'] is False and p._backend == 'depthart'


def test_depthart_selection_does_not_load_or_download(monkeypatch):
    from utils import model_downloader
    monkeypatch.setattr(model_downloader, 'ensure_depthart_model', lambda *a, **kw: pytest.fail('selection downloaded'))
    p = vd.VideoDepthPerceptionPlugin({}, 'test', _FakeExecutor())
    assert p.dispatch('visual_depth', dict(action='config', backend='depthart'))['status'] == 'configured'
    assert p._model is None


def test_depthart_first_load_downloads_and_reuses_session(monkeypatch):
    from utils import model_downloader
    from plugins import depthart_runtime
    p = make_plugin()
    downloaded, loaded = [], []
    def download(model_dir, progress_cb=None):
        downloaded.append(model_dir)
        return {'depthart.engine': '/models/depthart/jp61/depthart.engine',
                'libdepthart_selective_scan_trt.so': '/models/depthart/jp61/libdepthart_selective_scan_trt.so'}
    monkeypatch.setattr(model_downloader, 'ensure_depthart_model', download)
    monkeypatch.setattr(depthart_runtime, 'DepthARTSession', lambda *paths: loaded.append(paths) or object())
    p._ensure_model(); p._ensure_model()
    assert downloaded == ['/models/depthart'] and len(loaded) == 1
    assert p._model_load_status == 'Loading DepthART Metric-S engine'


def test_info_exposes_real_downloader_progress_while_loading(monkeypatch):
    from utils import model_downloader
    from plugins import depthart_runtime
    p = make_plugin()
    p._model_loading = True
    observed = []
    def download(model_dir, progress_cb=None):
        for percent in (25, 75, 100):
            progress_cb(percent, percent / 5, 20)
            info = p.dispatch('visual_depth', {'action': 'info'})
            assert info['state'] == 'loading'
            assert info['model'] == 'depthart-metric-s'
            assert f'{percent}%' in info['desc']
            observed.append(info['desc'])
        return {'depthart.engine': '/models/depthart/jp61/depthart.engine',
                'libdepthart_selective_scan_trt.so': '/models/depthart/jp61/libdepthart_selective_scan_trt.so'}
    def session(*paths):
        assert p.dispatch('visual_depth', {'action': 'info'})['desc'] == 'Loading DepthART Metric-S engine'
        return object()
    monkeypatch.setattr(model_downloader, 'ensure_depthart_model', download)
    monkeypatch.setattr(depthart_runtime, 'DepthARTSession', session)
    p._ensure_model()
    assert len(observed) == 3


@pytest.mark.parametrize('status', [None, ''])
def test_loading_info_falls_back_without_progress(status):
    p = make_plugin()
    p._model_loading = True
    p._model_load_status = status
    assert p.dispatch('visual_depth', {'action': 'info'})['desc'] == 'Loading depth engine...'


def test_depthart_explicit_paths_bypass_download_and_partial_override_fails(monkeypatch):
    from utils import model_downloader
    from plugins import depthart_runtime
    monkeypatch.setattr(model_downloader, 'ensure_depthart_model', lambda *a, **kw: pytest.fail('override downloaded'))
    loaded = []
    monkeypatch.setattr(depthart_runtime, 'DepthARTSession', lambda *paths: loaded.append(paths) or object())
    p = make_plugin({'depthart_engine_path': '/custom/engine', 'depthart_plugin_path': '/custom/plugin'})
    p._ensure_model()
    assert loaded == [('/custom/engine', '/custom/plugin')]
    with pytest.raises(ValueError, match='both'):
        make_plugin({'depthart_engine_path': '/custom/engine'})._ensure_model()


@pytest.mark.parametrize('target', ['yolo', 'depthart'])
@pytest.mark.parametrize('tag', [None, 'other'])
def test_restored_calibration_rejects_missing_or_previous_model(target, tag):
    # Simulate shared model config followed by a persisted instance row.
    p = vd.VideoDepthPerceptionPlugin({'backend': target}, 'test', _FakeExecutor())
    assert p.dispatch('visual_depth', dict(action='config', backend=target))['status'] == 'configured'
    cfg = dict(action='config', instance_id='a', calibration_preset=vd.CAL_MANUAL, cal_a=1., cal_b=.5)
    if tag:cfg['calibration_backend'] = 'yolo' if target == 'depthart' else 'depthart'
    result = p.dispatch('visual_depth', cfg)
    assert result['status'] == 'error' and result['adapter_ok'] is False
    assert 'a' not in p._instance_configs


@pytest.mark.parametrize('backend', ['yolo', 'depthart'])
def test_matching_calibration_replay_and_no_correction(backend):
    p = vd.VideoDepthPerceptionPlugin({'backend': backend}, 'test', _FakeExecutor())
    cfg = dict(action='config', instance_id='a', calibration_preset=vd.CAL_MANUAL,
               calibration_backend=backend, cal_a=1., cal_b=.5)
    assert p.dispatch('visual_depth', cfg)['status'] == 'configured'
    assert p._instance_configs['a']['cal_b'] == .5
    assert p.dispatch('visual_depth', dict(action='config', instance_id='a',
                      calibration_preset=vd.CAL_NONE))['status'] == 'configured'


def test_nonidentity_calibration_cannot_bypass_binding_with_auto_preset():
    p = make_plugin()
    result = p.dispatch('visual_depth', dict(action='config', instance_id='a',
                                           calibration_preset=vd.CAL_AUTO, cal_b=.5))
    assert result['adapter_ok'] is False


def test_calibration_model_is_persistable_but_never_auto_stamped():
    for backend in ('yolo', 'depthart'):
        p = vd.VideoDepthPerceptionPlugin({'backend': backend}, 'test', _FakeExecutor())
        prop = p.get_tools()[0]['configSchema']['properties']['calibration_backend']
        assert prop['scope'] == 'instance'
        assert 'default' not in prop
        assert 'x-show-when' not in prop


def test_switch_then_restore_old_row_rejected_after_process_restart(tmp_path):
    artifact = tmp_path / 'artifact'
    artifact.write_bytes(b'test')
    p = vd.VideoDepthPerceptionPlugin({'depthart_engine_path': str(artifact),
        'depthart_plugin_path': str(artifact)}, 'test', _FakeExecutor())
    old = dict(action='config', instance_id='a', calibration_preset=vd.CAL_MANUAL,
               calibration_backend='yolo', cal_b=.5)
    assert p.dispatch('visual_depth', old)['status'] == 'configured'
    assert p.dispatch('visual_depth', dict(action='config', backend='depthart'))['status'] == 'configured'
    assert p.dispatch('visual_depth', old)['adapter_ok'] is False
    restarted = make_plugin()
    assert restarted.dispatch('visual_depth', old)['adapter_ok'] is False


def test_laziness_missing_camera_and_presets(monkeypatch):
    p=make_plugin({'calibration_preset':vd.CAL_AUTO})
    monkeypatch.setattr(p,'_ensure_model',lambda:pytest.fail('unexpected model load'))
    assert p.dispatch('visual_depth',{'action':'info'})['model']=='depthart-metric-s'
    assert p.dispatch('visual_depth',{'action':'config','instance_id':'a','calibration_preset':vd.CAL_AUTO})['config']['calibration_preset']==vd.CAL_NONE
    assert p.dispatch('visual_depth',{'action':'start','input_topic':'/a'})['state']=='error'
    preset=next(iter(vd.CALIBRATION_PRESETS))
    assert p.dispatch('visual_depth',{'action':'config','calibration_preset':preset})['status']=='error'
    assert p.dispatch('visual_depth',{'action':'config','backend':'invalid'})['status']=='error'
    assert p._model is None


def test_unbound_on_demand_start_discards_stopped_camera(monkeypatch):
    p = make_plugin()
    p._model = object()
    p._upstream_camera['a'] = camera()
    starts = []
    monkeypatch.setattr(p, '_start_node', lambda key, topic: starts.append((key, topic)))
    p.dispatch('visual_depth', dict(action='start', instance_id='a'))
    assert starts == [('a', None)]
    assert 'a' not in p._upstream_camera


@pytest.mark.parametrize('topic_arg', [{'input_topic': '/a'}, {'input_topics': ['/a']}])
def test_stream_without_k_never_creates_node(monkeypatch, topic_arg):
    p = make_plugin()
    monkeypatch.setattr(p, '_start_node', lambda *a: pytest.fail('invalid stream started'))
    monkeypatch.setattr(p, '_ensure_model', lambda: pytest.fail('invalid stream loaded model'))
    result = p.dispatch('visual_depth', dict(action='start', **topic_arg))
    assert result['state'] == 'error' and not p._nodes


def test_camera_requests_are_scoped_and_explicit():
    p=make_plugin()
    p._upstream_camera={'a':camera(900),'b':camera(600)}
    p._nodes={'a':types.SimpleNamespace(_input_topic='/a'),'b':types.SimpleNamespace(_input_topic='/b')}
    assert p._request_camera({},'a')['K'][0]==900
    assert p._request_camera({},'b')['K'][0]==600
    with pytest.raises(ValueError):p._request_camera({},'')
    with pytest.raises(ValueError):p._request_camera({},'missing')
    assert p._request_camera({'camera_info':camera(800)},'')['K'][0]==800


@pytest.mark.parametrize('change', ['same', 'scaled', 'focal', 'identity', 'topic', 'missing'])
def test_live_stream_start_preserves_camera_binding(change):
    p = make_plugin()
    bound = camera()
    p._upstream_camera['a'] = bound
    p._nodes['a'] = types.SimpleNamespace(_input_topic='/a', start=lambda: {'state': 'running'})
    requested = camera()
    topic = '/a'
    if change == 'scaled':
        requested['width'] *= 2
        requested['height'] *= 2
        requested['K'] = [v * 2 if i < 6 else v for i, v in enumerate(requested['K'])]
    elif change == 'focal':requested = camera(700)
    elif change == 'identity':requested['id'] = 'another-camera'
    elif change == 'topic':topic = '/b'
    args = dict(action='start', instance_id='a', input_topic=topic)
    if change != 'missing':args['camera_info'] = {topic: requested}
    result = p.dispatch('visual_depth', args)
    assert result['state'] == ('running' if change in ('same', 'scaled') else 'error')
    assert p._upstream_camera['a'] is bound
    assert p._nodes['a']._input_topic == '/a'


@pytest.mark.parametrize('bound', [False, True])
def test_repeated_on_demand_start_does_not_rebind(bound):
    p = make_plugin()
    if bound:p._upstream_camera['a'] = camera()
    p._nodes['a'] = types.SimpleNamespace(_input_topic=None, start=lambda: {'state': 'running'})
    assert p.dispatch('visual_depth', dict(action='start', instance_id='a'))['state'] == 'running'
    result = p.dispatch('visual_depth', dict(action='start', instance_id='a', camera_info=camera(700)))
    assert result['state'] == 'error'
    assert p._upstream_camera.get('a') == (camera() if bound else None)


def test_stopped_instance_can_bind_a_new_camera(monkeypatch):
    p = make_plugin()
    class Model:
        def for_camera(self, camera):return self
    p._model = Model()
    first = dict(action='start', instance_id='a', camera_info=camera())
    assert p.dispatch('visual_depth', first)['state'] == 'running'
    p.dispatch('visual_depth', dict(action='stop', instance_id='a'))
    assert p.dispatch('visual_depth', dict(first, camera_info=camera(700)))['state'] == 'running'
    assert p._upstream_camera['a']['K'][0] == 700
    p.dispatch('visual_depth', dict(action='stop', instance_id='a'))


def test_inactive_camera_metadata_is_not_used_for_photo():
    p=make_plugin();p._upstream_camera['failed-start']=camera()
    with pytest.raises(ValueError,match='running camera'):
        p._request_camera({},'failed-start')


def test_bound_stream_camera_reads_latest_declaration():
    from plugins.depthart_runtime import _CameraSession
    p=make_plugin(); p._upstream_camera['a']=camera(900)
    values=[]
    class Session:
        def infer(self,frame,decl):values.append(decl['K'][0]);return [],None
    bound=_CameraSession(Session(),lambda:p._camera_declaration('a'))
    bound.infer(None); p._upstream_camera['a']=camera(700); bound.infer(None)
    assert values==[900,700]


def test_photo_missing_k_does_not_initialize_model(monkeypatch):
    p=make_plugin()
    monkeypatch.setattr(vd,'load_image_bytes',lambda *a,**kw:(vision_stubs.frame_bytes(1600,900),'test'))
    monkeypatch.setattr(p,'_ensure_model',lambda:pytest.fail('unexpected model load'))
    result=p.dispatch('visual_depth',{'action':'recognize_by_photo','image_path':'dummy'})
    assert result['ok'] is False and 'camera_info' in result['detail']


def test_manual_calibration_preserved():
    p=make_plugin({'calibration_preset':vd.CAL_MANUAL,'cal_b':.2,'calibration_backend':'depthart'})
    assert p._cal_b==pytest.approx(.2)


@pytest.mark.parametrize('backend', ['yolo', 'depthart'])
@pytest.mark.parametrize('tag', [None, 'wrong'])
@pytest.mark.parametrize('preset', [None, vd.CAL_MANUAL])
def test_startup_rejects_unbound_or_mismatched_calibration(backend, tag, preset):
    cfg = {'backend': backend, 'cal_b': .5}
    if preset:cfg['calibration_preset'] = preset
    if tag:cfg['calibration_backend'] = 'yolo' if backend == 'depthart' else 'depthart'
    with pytest.raises(ValueError, match='Calibration model'):
        vd.VideoDepthPerceptionPlugin(cfg, 'test', _FakeExecutor())


@pytest.mark.parametrize('backend', ['yolo', 'depthart'])
@pytest.mark.parametrize('previous_fit', [False, True])
def test_calibrate_save_roundtrip_and_reset_provenance(backend, previous_fit, monkeypatch):
    cfg = {'backend': backend}
    if previous_fit:cfg.update(calibration_backend=backend, calibration_preset=vd.CAL_MANUAL, cal_b=.2)
    p = vd.VideoDepthPerceptionPlugin(cfg, 'test', _FakeExecutor())
    monkeypatch.setattr(p, '_raw_depth_for_calibration',
                        lambda *args: ([np.full((480,640), 2., np.float32)], 'test'))
    result = p.dispatch('visual_depth', {'action': 'calibrate', 'distance_m': 3.})
    saved = result['save_to_config']
    assert saved['calibration_backend'] == backend
    restored = vd.VideoDepthPerceptionPlugin(dict(saved, backend=backend), 'test', _FakeExecutor())
    assert restored._cal_b == pytest.approx(np.log(1.5), abs=1e-6)
    assert restored.dispatch('visual_depth', dict(saved, action='config', instance_id='a'))['status'] == 'configured'
    p.dispatch('visual_depth', {'action': 'reset_calibration'})
    assert p._plugin_cfg.get('calibration_backend') == (backend if previous_fit else None)
    assert (p._cal_a, p._cal_b) == (1., .2 if previous_fit else 0.)


@pytest.mark.parametrize('topic', ['/camera', None])
def test_explicit_photo_cannot_relabel_live_camera(topic):
    p=make_plugin();p._upstream_camera['a']=camera(900)
    p._nodes['a']=types.SimpleNamespace(_input_topic=topic)
    with pytest.raises(ValueError,match='differs'):
        p._request_camera({'camera_info':camera(700)},'a')
    assert p._request_camera({'camera_info':camera(900)},'a')['K'][0]==900


def test_missing_plugin_version_symbol_is_clear(tmp_path, monkeypatch):
    from plugins import depthart_runtime
    artifact = tmp_path / 'artifact'
    artifact.write_bytes(b'test')
    monkeypatch.setattr(depthart_runtime.ctypes, 'CDLL', lambda *a, **kw: object())
    with pytest.raises(ValueError, match='version symbol'):
        depthart_runtime.DepthARTSession(artifact, artifact)


def test_unbound_on_demand_photo_does_not_bind_camera(monkeypatch):
    p = make_plugin()
    p._nodes['a'] = types.SimpleNamespace(_input_topic=None, _cal_a=1., _cal_b=0., calibration_label='metric')
    class Model:
        def for_camera(self, declaration):return self
        def infer(self, frame):return [np.ones((1,480,864), np.float32)], None
    p._model = Model()
    monkeypatch.setattr(vd, 'load_image_bytes', lambda *a, **kw: (vision_stubs.frame_bytes(1600,900), 'test'))
    monkeypatch.setattr(p, '_publish_one_shot', lambda *a: None)
    args = dict(action='recognize_by_photo', instance_id='a', image_path='dummy', camera_info=camera())
    assert p.dispatch('visual_depth', args)['ok']
    assert not p._upstream_camera.get('a')
    del args['camera_info']
    assert not p.dispatch('visual_depth', args)['ok']


def test_photo_inherits_live_manual_calibration(monkeypatch):
    p=make_plugin();p._upstream_camera['a']=camera()
    p._nodes['a']=types.SimpleNamespace(_input_topic='/camera',_cal_a=1.,_cal_b=np.log(2),calibration_label='manual')
    class Model:
        def for_camera(self,c):return self
        def infer(self,frame):return [np.ones((1,480,864),np.float32)],None
    p._model=Model()
    monkeypatch.setattr(vd,'load_image_bytes',lambda *a,**kw:(vision_stubs.frame_bytes(1600,900),'test'))
    monkeypatch.setattr(p,'_publish_one_shot',lambda *a:None)
    result=p.dispatch('visual_depth',{'action':'recognize_by_photo','image_path':'dummy','instance_id':'a'})
    assert result['ok'] and result['range'][0]==pytest.approx(2.) and result['calibration']=='manual'


def test_retired_instance_photo_is_not_rerouted():
    p=make_plugin();calls=[]
    p._nodes['b']=types.SimpleNamespace(_publish=lambda *a:calls.append(a),_depth_topic='/b/depth',_summary_topic='/b/summary')
    assert p._publish_one_shot('a',np.ones((480,640),np.float32),{}) is None
    assert not calls


def test_yolo_inference_error_behavior_unchanged(monkeypatch):
    p=vd.VideoDepthPerceptionPlugin({},'test',_FakeExecutor())
    class Model:
        def infer(self,frame):raise ValueError('engine failure')
    p._model=Model()
    monkeypatch.setattr(vd,'load_image_bytes',lambda *a,**kw:(vision_stubs.frame_bytes(1600,900),'test'))
    with pytest.raises(ValueError,match='engine failure'):
        p.dispatch('visual_depth',{'action':'recognize_by_photo','image_path':'dummy'})


def test_depthart_schema_hides_yolo_presets_without_mutating_default():
    p=make_plugin()
    prop=p.get_tools()[0]['configSchema']['properties']['calibration_preset']
    assert prop['enum']==[vd.CAL_NONE,vd.CAL_MANUAL]
    assert prop['default']==vd.CAL_NONE
    yolo=vd.VideoDepthPerceptionPlugin({},'test',_FakeExecutor())
    default=yolo.get_tools()[0]['configSchema']['properties']['calibration_preset']
    assert vd.CAL_AUTO in default['enum'] and next(iter(vd.CALIBRATION_PRESETS)) in default['enum']
    assert p._model is None


def test_process_shutdown_retires_nodes_before_engine_close():
    p=vd.VideoDepthPerceptionPlugin({},'test',_FakeExecutor())
    calls=[]
    class Model:
        def close(self):
            assert p._nodes=={}
            calls.append('closed')
    p._model=Model()
    p.dispatch('visual_depth',{'action':'start','instance_id':'shutdown-probe'})
    node=p._nodes['shutdown-probe']
    p._backend='depthart'
    p.shutdown()
    assert node.destroyed and calls==['closed'] and p._model is None
    p.shutdown();assert calls==['closed']
    assert p.dispatch('visual_depth',{'action':'start'})['state']=='error'
    with pytest.raises(RuntimeError,match='shutting down'):p._ensure_model()


def test_shutdown_waits_for_loading_and_prevents_late_node(monkeypatch):
    import threading
    p=vd.VideoDepthPerceptionPlugin({},'test',_FakeExecutor())
    entered,release=threading.Event(),threading.Event()
    closed=[]
    class Model:
        def close(self):closed.append(True)
    def load():
        with p._model_lock:
            entered.set();assert release.wait(2)
            p._model=Model()
    monkeypatch.setattr(p,'_ensure_model',load)
    loader=threading.Thread(target=lambda:(p._ensure_model(),p._start_node('late',None)))
    loader.start();assert entered.wait(1)
    p._backend='depthart'
    closer=threading.Thread(target=p.shutdown);closer.start()
    for _ in range(100):
        if p._closing:break
        __import__('time').sleep(.001)
    assert p._closing
    release.set();loader.join(2);closer.join(2)
    assert not loader.is_alive() and not closer.is_alive()
    assert closed==[True] and p._nodes=={} and p._model is None


def test_yolo_keeps_original_process_teardown():
    from plugins.visual_depth import VideoDepthPerceptionPlugin
    p=VideoDepthPerceptionPlugin({},'test',_FakeExecutor())
    class Model:
        def close(self):raise AssertionError('YOLO teardown must remain unchanged')
    model=Model();p._model=model
    p.shutdown()
    assert p._model is model and not p._closing


def test_bundle_shutdown_calls_opted_in_hooks_and_continues_after_error():
    from main import PerceptionBundle
    calls=[]
    class Broken:
        PREFIX='broken'
        def shutdown(self):calls.append('broken');raise RuntimeError('close failure')
    class Working:
        PREFIX='working'
        def shutdown(self):calls.append('working')
    bundle=PerceptionBundle.__new__(PerceptionBundle)
    bundle._plugins=[Broken(),types.SimpleNamespace(PREFIX='no-hook'),Working()]
    bundle.shutdown()
    assert calls==['broken','working']
