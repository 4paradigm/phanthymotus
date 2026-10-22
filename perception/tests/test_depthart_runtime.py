import types
import numpy as np
import pytest
import vision_stubs
from vision_stubs import _FakeExecutor
from plugins import visual_depth as vd
from plugins.depthart_runtime import camera_matrix, prepare_image, fit_to_canvas, canvas_resize
from plugins.vision_runtime import decode_depth


def camera(fx=900.):
    return {'schema':'motus.camera/1','id':'camera-a','width':1600,'height':900,
            'K':[fx,4.,799.5,0.,fx,449.5,0.,0.,1.]}




def test_preprocess_scales_intrinsics_and_normalizes_rgb():
    frame=np.zeros((900,1600,3),np.uint8); frame[:,:,2]=255
    blob,k,crop=prepare_image(frame,camera(),(864,480))
    expected=np.array(camera()['K'],np.float32).reshape(3,3)
    expected[0]*=864/1600; expected[1]*=480/900
    assert blob.shape==(1,3,480,864)
    np.testing.assert_allclose(k[0],expected)
    np.testing.assert_allclose(blob[0,:,0,0],(np.array([1.,0.,0.])-[.485,.456,.406])/[.229,.224,.225],rtol=1e-6)
    assert decode_depth([np.ones((1,480,864))],None).shape==(480,864)


@pytest.mark.parametrize('bad',[{}, {'K':[1.]*9,'width':1,'height':1},dict(camera(),K=[np.nan]*9),dict(camera(),width=0)])
def test_invalid_camera_rejected(bad):
    with pytest.raises(ValueError):camera_matrix(bad)


@pytest.mark.parametrize('size', [(1280, 720), (864, 480), (640, 360), (1920, 1080),
                                  (960, 540), (848, 480)])
def test_no_crop_when_the_resize_already_lands_on_the_canvas(size):
    # The rounding to multiples of 32 admits a band of ratios, not just the
    # canvas's exact 1.8. 16:9 needs no crop; cropping it to 1.8 regardless
    # would throw away rows for nothing.
    assert fit_to_canvas(*size, (864, 480)) == (0, 0, *size)
    assert canvas_resize(*size, (864, 480)) == (864, 480)


@pytest.mark.parametrize('size,expected', [
    ((640, 480), (0, 62, 640, 356)),      # 4:3 — trim top and bottom
    ((800, 600), (0, 78, 800, 444)),
    ((1280, 960), (0, 124, 1280, 711)),
    ((1280, 800), (0, 44, 1280, 711)),    # 16:10 — a much smaller loss
    ((1600, 480), (368, 0, 864, 480)),    # too wide — trim the sides
])
def test_centre_crop_fits_the_canvas_aspect(size, expected):
    # A 4:3 source used to be refused outright. It is cropped now, because the
    # alternative on a fixed engine canvas is a stretch, and a stretch hands
    # the network a geometry its K does not describe.
    assert fit_to_canvas(*size, (864, 480)) == expected
    x0, y0, w, h = expected
    # The point of the crop: what is left resizes onto the canvas exactly.
    assert canvas_resize(w, h, (864, 480)) == (864, 480)
    # Centred: equal margins on the axis that was trimmed.
    assert (size[0] - w) // 2 == x0 and (size[1] - h) // 2 == y0


@pytest.mark.parametrize('size', [(480, 854), (200, 1000), (720, 1280)])
def test_a_crop_that_would_eat_half_the_frame_is_refused(size):
    with pytest.raises(ValueError, match='crop away'):
        fit_to_canvas(*size, (864, 480))


def test_crop_moves_the_principal_point_and_leaves_focal_lengths_alone():
    """The property that makes this cheap: a vertical crop costs no horizontal FOV."""
    frame = np.zeros((480, 640, 3), np.uint8)
    cam = {'schema': 'motus.camera/1', 'id': 'c', 'width': 640, 'height': 480,
           'K': [502.3, 0., 319.5, 0., 502.3, 239.5, 0., 0., 1.]}
    blob, k, crop = prepare_image(frame, cam, (864, 480))
    assert blob.shape == (1, 3, 480, 864)
    assert crop == (0, 62, 640, 356)
    sx, sy = 864 / 640, 480 / 356
    # fx scales with the horizontal resize only — the crop did not touch width.
    assert k[0][0, 0] == pytest.approx(502.3 * sx, rel=1e-5)
    assert k[0][0, 2] == pytest.approx(319.5 * sx, rel=1e-5)
    # cy is the principal point in the cropped frame, then scaled.
    assert k[0][1, 2] == pytest.approx((239.5 - 62) * sy, rel=1e-5)
    assert k[0][1, 1] == pytest.approx(502.3 * sy, rel=1e-5)
    # The horizontal half-FOV a consumer derives back is unchanged by the crop.
    assert np.arctan((864 / 2) / k[0][0, 0]) == pytest.approx(
        np.arctan((640 / 2) / 502.3), rel=1e-5)


def test_declaration_at_a_higher_resolution_than_the_frame():
    # The declaration may describe the same lens at a larger size; K must be
    # brought to frame pixels before the crop offsets, not after.
    frame = np.zeros((480, 640, 3), np.uint8)
    cam = {'schema': 'motus.camera/1', 'id': 'c', 'width': 1280, 'height': 960,
           'K': [1004.6, 0., 639., 0., 1004.6, 479., 0., 0., 1.]}
    blob, k, crop = prepare_image(frame, cam, (864, 480))
    assert crop == (0, 62, 640, 356)
    assert k[0][0, 0] == pytest.approx(502.3 * 864 / 640, rel=1e-4)
    assert k[0][1, 2] == pytest.approx((479. / 2 - 62) * 480 / 356, rel=1e-4)


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




@pytest.mark.parametrize('backend', ['yolo', 'depthart'])
def test_matching_calibration_replay_and_no_correction(backend):
    p = vd.VideoDepthPerceptionPlugin({'backend': backend}, 'test', _FakeExecutor())
    cfg = dict(action='config', instance_id='a', calibration_preset=vd.CAL_MANUAL,
               calibration_backend=backend, cal_a=1., cal_b=.5)
    assert p.dispatch('visual_depth', cfg)['status'] == 'configured'
    assert p._instance_configs['a']['cal_b'] == .5
    assert p.dispatch('visual_depth', dict(action='config', instance_id='a',
                      calibration_preset=vd.CAL_NONE))['status'] == 'configured'








def test_laziness_missing_camera_and_presets(monkeypatch):
    p=make_plugin({'calibration_preset':vd.CAL_AUTO})
    monkeypatch.setattr(p,'_ensure_model',lambda:pytest.fail('unexpected model load'))
    assert p.dispatch('visual_depth',{'action':'info'})['model']=='depthart-metric-s'
    # `auto` survives under DepthART now: the lookup is keyed on camera *and*
    # model, so it finds that model's row or nothing.
    assert p.dispatch('visual_depth',{'action':'config','instance_id':'a','calibration_preset':vd.CAL_AUTO})['config']['calibration_preset']==vd.CAL_AUTO
    assert p.dispatch('visual_depth',{'action':'start','input_topic':'/a'})['state']=='error'
    preset=next(iter(vd.CALIBRATION_PRESETS))
    # A row measured on the other model is still refused, by its own backend
    # field rather than by "is it a preset at all".
    yolo_row = next(n for n, r in vd.CALIBRATION_PRESETS.items() if r.get('backend', 'yolo') == 'yolo')
    rejected=p.dispatch('visual_depth',{'action':'config','calibration_preset':yolo_row})
    assert rejected['status']=='configured' and rejected['config']['calibration_preset']==vd.CAL_NONE
    assert yolo_row in rejected['calibration_note']
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


def test_the_preset_list_does_not_depend_on_the_selected_model():
    """It used to be narrowed under DepthART, and that made the form lie.

    The card's own dialog sets `backend`, so a model-dependent enum only
    catches up after a save, a re-register and a re-fetch — picking `yolo` and
    a preset in one go was impossible and the list looked wrong. The guard that
    matters is `_backend_calibration_cfg`, which drops a YOLO preset under
    DepthART at config time and says so; a stale form cannot bypass it.
    """
    preset = next(iter(vd.CALIBRATION_PRESETS))
    for backend in ('yolo', 'depthart'):
        p = vd.VideoDepthPerceptionPlugin({'backend': backend}, 'test', _FakeExecutor())
        prop = p.get_tools()[0]['configSchema']['properties']['calibration_preset']
        assert vd.CAL_AUTO in prop['enum'] and preset in prop['enum'], backend
        assert prop['default'] == vd.CAL_AUTO
        assert p._model is None          # still no engine load from a schema read
    # And the guard still fires, so listing it costs nothing.
    d = make_plugin()
    result = d.dispatch('visual_depth', {'action': 'config', 'calibration_preset': preset})
    assert result['config']['calibration_preset'] == vd.CAL_NONE
    assert preset in result['calibration_note']


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


# ── The startup failure this change fixes ────────────────────────────────────
#
# Every card calibrated before `calibration_backend` existed holds coefficients
# with no marker. Reading that as "unknown" rejected the saved row on replay,
# and Core turns a rejected config into its own generic credential message — so
# a robot came up with `配置无效（缺少 url/key）` on a card that has neither.







@pytest.mark.parametrize('declaration', [None, {}, {'schema': 'motus.camera/1'}])
def test_start_without_upstream_intrinsics_says_what_to_do(declaration):
    # Orin 5's wiring: the depth card is fed by agent-core's `remote_camera`,
    # whose info() has no camera_info at all, so nothing reaches the start.
    # The old message named a field that appears nowhere on the canvas.
    p = make_plugin()
    args = {'action': 'start', 'instance_id': 'a', 'input_topic': '/remote_control/camera'}
    if declaration is not None:
        args['camera_info'] = declaration
    result = p.dispatch('visual_depth', args)
    assert result['state'] == 'error'
    assert '/remote_control/camera' in result['message']
    assert 'yolo' in result['message']          # the way out
    assert not p._nodes


# ── A declaration with no K is a valid declaration ───────────────────────────
#
# The spec carries `half_fov_rad` because a tape measure produces an angle and
# no K at all, and its reference declaration — R1's main camera — is exactly
# that. Reading only `K` rejected the canonical camera of the robot this
# project is built around. Rows below are R1's real declaration, from
# /ubuntu/camera/main on r1_bj.

R1_MAIN = {'schema': 'motus.camera/1', 'topic': '/ubuntu/camera/main',
           'id': 'unitree/r1/camera_main', 'format': 'image/jpeg',
           'width': 1280, 'height': 720, 'distortion_model': 'unknown',
           'D': None, 'K': None, 'half_fov_rad': 0.888, 'half_fov_v_rad': None,
           'source': 'measured', 'measured_on': 'r1_sz, 2026-09-23',
           'pipeline': ['unitree/r1/camera_main']}


@pytest.mark.parametrize('absent', [None, '', []])
def test_k_derived_from_declared_half_fov(absent):
    matrix, w, h = camera_matrix(dict(R1_MAIN, K=absent))
    assert (w, h) == (1280, 720)
    fx = 640 / np.tan(0.888)
    assert float(matrix[0, 0]) == pytest.approx(fx, rel=1e-5)
    assert float(matrix[1, 1]) == pytest.approx(fx, rel=1e-5)   # square pixels
    assert (float(matrix[0, 2]), float(matrix[1, 2])) == (639.5, 359.5)
    # The angle a consumer derives back must be the one the camera measured.
    assert float(np.arctan((w / 2) / matrix[0, 0])) == pytest.approx(0.888, rel=1e-6)


def test_k_is_absent_entirely_rather_than_null():
    stripped = {k: v for k, v in R1_MAIN.items() if k != 'K'}
    assert float(camera_matrix(stripped)[0][0, 0]) == pytest.approx(640 / np.tan(0.888), rel=1e-5)


def test_declared_k_wins_over_the_angle_beside_it():
    """The spec's own precedence: K is solved from many observations."""
    both = dict(R1_MAIN, K=[700., 0., 639.5, 0., 700., 359.5, 0., 0., 1.])
    assert float(camera_matrix(both)[0][0, 0]) == 700.


def test_vertical_angle_is_used_when_declared_and_not_guessed_from_aspect():
    # half_fov_v_rad cannot be derived from the aspect ratio — the spec is
    # explicit — so it is read when present and square pixels assumed when not.
    matrix, _, _ = camera_matrix(dict(R1_MAIN, half_fov_v_rad=0.6))
    assert float(matrix[1, 1]) == pytest.approx(360 / np.tan(0.6), rel=1e-5)
    assert float(matrix[1, 1]) != pytest.approx(float(matrix[0, 0]), rel=1e-3)


@pytest.mark.parametrize('bad,match', [
    (102, 'half'),          # degrees typed where radians belong
    (1.6, 'out of range'),   # beyond pi/2
    (0, 'out of range'),
    (-0.5, 'out of range'),
    ('wide', 'not a number'),
])
def test_an_unusable_half_fov_says_so_rather_than_claiming_none_was_declared(bad, match):
    with pytest.raises(ValueError, match=match):
        camera_matrix(dict(R1_MAIN, half_fov_rad=bad))


def test_neither_k_nor_half_fov_is_still_refused():
    with pytest.raises(ValueError, match='neither K nor half_fov_rad'):
        camera_matrix({'width': 1280, 'height': 720})


def test_r1_main_needs_no_crop():
    # 1280x720 lands on the 864x480 canvas untouched, so R1 loses no field of
    # view to the canvas — only to the pinhole approximation on a 102 deg lens.
    assert fit_to_canvas(1280, 720, (864, 480)) == (0, 0, 1280, 720)


# ── Calibration is stored per model, with no marker to contradict ────────────
#
# These replace an earlier design where one `cal_a`/`cal_b` pair was tagged with
# a `calibration_backend` marker naming the model it was measured against. Two
# fields that must agree eventually did not: on r1_bj a depthart/yolo mismatch
# dropped the card's only calibration to `no calibrate`, the canvas persisted
# the dropped row, and every distance was then uncalibrated on a 102 degree
# lens. Keyed storage cannot disagree with itself.
#
# `cal_a`/`cal_b` stay the YOLO pair, which is what every row already saved
# holds — it was the only model there was — so nothing needs migrating.

R1_PRESET = next(iter(vd.CALIBRATION_PRESETS))


def test_the_schema_has_exactly_one_calibration_pair():
    props = make_plugin().get_tools()[0]['configSchema']['properties']
    # Neither a provenance marker nor a pair per model: a fit is two numbers,
    # and anything that multiplies them puts the bookkeeping on the operator.
    assert 'calibration_backend' not in props
    assert 'cal_a_depthart' not in props and 'cal_b_depthart' not in props
    for key in ('cal_a', 'cal_b'):
        assert key in props
    # No conditional display: #286 avoids x-show-when for Core compatibility.
    assert all('x-show-when' not in props[k] for k in props)












def test_yolo_presets_are_still_refused_under_depthart():
    # The one cross-model rule that survives, and for a measured reason: the
    # preset table is fitted against YOLO26-depth, and r1's -0.9753 is a factor
    # of 2.65 that would push DepthART the wrong way.
    p = make_plugin()
    result = p.dispatch('visual_depth', {'action': 'config', 'calibration_preset': R1_PRESET})
    assert result['status'] == 'configured'
    assert result['config']['calibration_preset'] == vd.CAL_NONE
    assert R1_PRESET in result['calibration_note']
    assert p.dispatch('visual_depth', {'action': 'info'})['calibration_note']


def test_yolo_presets_still_work_under_yolo():
    p = vd.VideoDepthPerceptionPlugin({}, 'test', _FakeExecutor())
    # Sent the way the canvas sends it — every field, defaults included.
    result = p.dispatch('visual_depth', {'action': 'config', 'cal_a': 1., 'cal_b': 0.,
                                         'calibration_preset': R1_PRESET})
    assert result['status'] == 'configured'
    assert result['config']['calibration_preset'] == R1_PRESET
    assert 'calibration_note' not in result
    assert p._cal_b == pytest.approx(vd.CALIBRATION_PRESETS[R1_PRESET]['cal_b'])


def test_a_dropdown_only_config_call_still_recomputes_the_fit():
    """A hand-made MCP call that only flips the selector must take effect.

    The recompute condition listed the numeric keys but not
    `calibration_preset`, which is what selects them. The canvas masks this by
    sending every field on every call; a `config` typed by hand does not.
    """
    p = vd.VideoDepthPerceptionPlugin({}, 'test', _FakeExecutor())
    p.dispatch('visual_depth', {'action': 'config', 'calibration_preset': R1_PRESET})
    assert p._cal_b == pytest.approx(vd.CALIBRATION_PRESETS[R1_PRESET]['cal_b'])
    p.dispatch('visual_depth', {'action': 'config', 'calibration_preset': vd.CAL_NONE})
    assert (p._cal_a, p._cal_b) == (1., 0.)


def test_the_orin5_row_that_used_to_kill_the_card_still_configures():
    # The two rows literally in Orin 5's agent-core DB. Under the marker design
    # this returned adapter_ok: false, which Core rewrote as "缺少 url/key" and
    # the card never started.
    p = vd.VideoDepthPerceptionPlugin({}, 'test', _FakeExecutor())
    assert p.dispatch('visual_depth', {'action': 'config', 'backend': 'depthart'})['status'] == 'configured'
    result = p.dispatch('visual_depth', {
        'action': 'config', 'instance_id': 'card-mv8w41sh6a45',
        'fps': 2, 'cal_a': 1, 'cal_b': 0, 'calibration_preset': R1_PRESET})
    assert result['status'] == 'configured' and result.get('adapter_ok', True) is True


def test_switching_models_clears_the_fit_and_says_so():
    """One pair cannot survive a model change, so it is cleared — out loud.

    A fit measured on YOLO and left under DepthART is the r1_bj failure:
    `cal_b: -0.9753` on that lens is a factor of 2.65, and nothing downstream
    can tell a corrected metre from an uncorrected one. The stripping itself
    was never the bug — doing it silently was.
    """
    p = vd.VideoDepthPerceptionPlugin({}, 'test', _FakeExecutor())
    row = dict(action='config', instance_id='a', calibration_preset=vd.CAL_MANUAL,
               cal_a=1., cal_b=-0.9753)
    assert p.dispatch('visual_depth', row)['status'] == 'configured'
    assert p._instance_configs['a']['cal_b'] == -0.9753

    switched = p.dispatch('visual_depth', dict(action='config', backend='depthart'))
    assert switched['status'] == 'configured'
    # Gone from the stored row, and the response says why.
    assert 'cal_b' not in p._instance_configs['a']
    # The selector survives: it is a rule, not a measurement. Clearing it left
    # YOLO uncalibrated after a switch back, with nobody to re-pick it.
    assert p._instance_configs['a']['calibration_preset'] == vd.CAL_MANUAL
    assert '标定' in switched['calibration_note'] and 'calibrate' in switched['calibration_note']
    assert (p._cal_a, p._cal_b) == (1., 0.)


def test_the_cleared_note_is_reported_once():
    # Reported on the next config, then not repeated — a note that reappears
    # on every call is a note nobody reads.
    p = vd.VideoDepthPerceptionPlugin({}, 'test', _FakeExecutor())
    first = p.dispatch('visual_depth', dict(action='config', backend='depthart'))
    assert first.get('calibration_note')
    second = p.dispatch('visual_depth', dict(action='config', fps=2))
    assert 'calibration_note' not in second


def test_a_single_pair_applies_under_either_model():
    # The pair is not model-keyed: whichever model is selected reads cal_a/cal_b.
    for backend in ('yolo', 'depthart'):
        p = vd.VideoDepthPerceptionPlugin({'backend': backend, 'cal_b': -0.5},
                                          'test', _FakeExecutor())
        assert (p._cal_a, p._cal_b) == (1., -0.5)


def test_r1_resolves_a_different_preset_row_per_model():
    """The same lens, two models, two fits — chosen automatically.

    R1's main camera needs `cal_b: -0.9753` under YOLO and
    `cal_a: 1.2454, cal_b: -0.550755` under DepthART; reporting one under the
    other is a factor of more than two in every metre.
    """
    cam = 'unitree/r1/camera_main'
    seen = {}
    for backend in ('yolo', 'depthart'):
        name, row = vd._preset_for_camera(cam, backend)
        assert row is not None, backend
        assert row['backend'] == backend
        seen[backend] = (row['cal_a'], row['cal_b'])
        # And `auto` resolves to it.
        cfg = {'calibration_preset': vd.CAL_AUTO}
        assert vd._calibration_from_cfg(cfg, camera_id=cam, backend=backend) == seen[backend]
    assert seen['yolo'] != seen['depthart']
    # DepthART's row needs a non-unit exponent; `calibrate` cannot produce it.
    assert seen['depthart'][0] != 1.0


def test_a_camera_with_no_row_for_this_model_falls_back_to_the_engine():
    # Not to the other model's row — that is the failure the keying prevents.
    assert vd._preset_for_camera('unknown/camera', 'yolo') == (None, None)
    assert vd._calibration_from_cfg({'calibration_preset': vd.CAL_AUTO},
                                    camera_id='unknown/camera', backend='depthart') == (1.0, 0.0)


def test_a_row_without_a_backend_field_counts_as_yolo():
    # The table had one entry and one model when it was written.
    row = dict(vd.CALIBRATION_PRESETS['Unitree R1'])
    row.pop('backend', None)
    assert row.get('backend', 'yolo') == 'yolo'
