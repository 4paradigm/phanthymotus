"""Opt-in real models/codecs/HTTP. Inputs are public photos with synthetic motion."""
import json
import os
from pathlib import Path
import time
import threading
import urllib.request
import numpy as np
import cv2
import pytest
from plugins.face_effects_live.contracts import Frame
from plugins.face_effects_live.engine import Engine
from plugins.face_effects_live.backends import OpenSeeFace, YuNet
from plugins.face_effects_live.render import CPU, GL
from plugins.face_effects_live.effects import Effects
from plugins.face_effects_live.sources.capture import FileSource
from plugins.face_effects_live.sources.push import PushSource
from plugins.face_effects_live.service import Service
from plugins.face_effects_live.plugin import FaceEffectsLivePlugin


@pytest.fixture
def evidence():
    root=os.environ.get('FACE_EFFECTS_EVIDENCE')
    if not root: pytest.skip('set FACE_EFFECTS_EVIDENCE for opt-in actual model tests')
    return Path(root)


def config(root, model='osf0'):
    return {'detector_model':str(root/'models/yunet.onnx'),
            'landmark_model':str(root/f'models/{model}.onnx'),'renderer':'cpu','max_age_ms':2000}


@pytest.mark.parametrize('model',[None,'osf0','osf3'])
def test_real_inference_and_visible_effects(evidence,model):
    cfg=config(evidence,model or 'osf0')
    if model is None: cfg.pop('landmark_model')
    engine=Engine(cfg)
    image=cv2.imread(str(evidence/'fixtures/grace_hopper.jpg'))
    try:
        for k in range(3): result=engine.process(Frame(image,k/30,k))
        assert result.metadata['tracked']==1
        assert len(result.metadata['ids'])==1
        assert np.count_nonzero(image!=result.image)>1000
        assert result.metadata['receive_to_render_ms']>0
    finally: engine.close()


def test_real_yunet_without_opencv_wrapper(evidence, monkeypatch):
    image = cv2.imread(str(evidence / "fixtures/grace_hopper.jpg"))
    assert image is not None
    single = cv2.resize(image, (320, 384))
    multiple = np.concatenate([single, single], axis=1)
    blank = np.zeros_like(multiple)
    path = evidence / "models/yunet.onnx"
    expected = None
    if hasattr(cv2, "FaceDetectorYN"):
        native = YuNet(path)
        try:
            faces = native.detect(multiple)
            if native.runtime == "opencv":
                expected = sorted(faces, key=lambda face: face.box[0])
                assert len(expected) == 2
        finally:
            native.close()
    monkeypatch.delattr(cv2, "FaceDetectorYN", raising=False)
    engine = Engine(config(evidence))
    try:
        engine.process(Frame(single, 0., 0))
        first = engine.process(Frame(single, .03, 1))
        engine.process(Frame(multiple, .06, 2))
        second = engine.process(Frame(multiple, .09, 3))
        third = engine.process(Frame(blank, .12, 4))
        assert first.metadata["tracked"] == 1
        assert second.metadata["tracked"] == 2
        assert third.metadata["detector_proposals"] == 0
        assert second.metadata["detector_runtime"] == "onnxruntime-cpu"
        assert np.count_nonzero(multiple != second.image) > 1000
        proposals = sorted(engine.detector.detect(multiple), key=lambda face: face.box[0])
        assert len(proposals) == 2
        if expected is not None:
            for ref, actual in zip(expected, proposals):
                np.testing.assert_allclose(actual.box, ref.box, atol=3)
                np.testing.assert_allclose(actual.points, ref.points, atol=3)
        large = cv2.resize(multiple, (1280, 768))
        scaled = sorted(engine.detector.detect(large), key=lambda face: face.box[0])
        assert len(scaled) == 2
        for ref, actual in zip(proposals, scaled):
            np.testing.assert_allclose(actual.box / 2, ref.box, atol=3)
            np.testing.assert_allclose(actual.points / 2, ref.points, atol=3)
    finally:
        engine.close()
    assert engine.detector.service is None


def test_real_yunet_bad_graph_reports_safe_mcp_status(evidence, tmp_path, monkeypatch):
    from plugins.face_effects_live import plugin as module

    monkeypatch.delattr(cv2, "FaceDetectorYN", raising=False)
    monkeypatch.setattr(module, "create_source", lambda *args: PushSource())
    invalid = tmp_path / "private-model-path.onnx"
    invalid.write_bytes(b"x" * 2048)
    plugin = FaceEffectsLivePlugin({"detector_model": str(invalid)})
    try:
        start = plugin.dispatch("start", {"source_kind": "file", "source_uri": "unused"})
        status = plugin.dispatch("info", {})
        assert not start["ok"] and status["state"] == "error"
        assert "YuNet ONNX Runtime initialization failed" in status["error"]
        assert str(invalid) not in status["error"]
        assert not plugin.instances
    finally:
        plugin.shutdown()


def test_gpu_provider_strict_failure_and_cpu_fallback(evidence):
    import onnxruntime
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers(): pytest.skip('requires absent CUDA')
    with pytest.raises(RuntimeError): OpenSeeFace(evidence/'models/osf0.onnx','cuda',False)
    model=OpenSeeFace(evidence/'models/osf0.onnx','cuda',True)
    assert model.providers==['CPUExecutionProvider'] and model.fallback_reason
    model.close()


def test_actual_ort_host_session_isolation_and_restart(evidence, monkeypatch):
    # The worker belongs to this pytest process, never a shared platform service.
    first = OpenSeeFace(evidence/'models/osf0.onnx')
    second = OpenSeeFace(evidence/'models/osf0.onnx')
    host = first.worker; first_key = first.session_key; second_key = second.session_key
    image = cv2.imread(str(evidence/'fixtures/grace_hopper.jpg'))
    monkeypatch.delattr(cv2, "FaceDetectorYN", raising=False)
    detector = YuNet(evidence/'models/yunet.onnx')
    detector_key = detector.service.key
    try:
        assert host is second.worker and first_key != second_key
        assert host._proc.pid != os.getpid()
        proposals = detector.detect(image)
        before = second.refine(image, proposals)
        assert before
        first.close()
        assert first_key not in host._loaded and second_key in host._loaded and host.alive
        host._proc.terminate(); host._proc.join(timeout=5)
        recovered = detector.detect(image)
        assert len(recovered) == len(proposals)
        np.testing.assert_array_equal(recovered[0].box, proposals[0].box)
        after = second.refine(image, recovered)
        assert len(after) == len(before)
        assert np.array_equal(after[0].landmarks, before[0].landmarks)
    finally:
        detector.close(); first.close(); second.close()
    assert second_key not in host._loaded
    assert detector_key not in host._services and host.alive


@pytest.mark.parametrize('backend',['osf','hybrid','mediapipe'])
def test_real_natural_frame_topology_and_effects(evidence,backend):
    path=evidence/'round2/natural/close_speaker.mkv'
    if not path.exists():pytest.skip('requires round2 natural dataset')
    cfg=config(evidence)
    if backend!='osf':
        pytest.importorskip('mediapipe')
        cfg['face_landmarker_model']=str(evidence/'models/face_landmarker.task')
        if backend=='mediapipe':cfg['detector_backend']='mediapipe'
        else:cfg['landmarker_backend']='hybrid'
    cap=cv2.VideoCapture(str(path));engine=Engine(cfg)
    try:
        for k in range(3):
            ok,image=cap.read();assert ok
            result=engine.process(Frame(image,k/30,k))
        assert result.metadata['tracked']==1
        assert np.count_nonzero(result.image!=image)>100
        face=engine.tracker.tracks[0].face
        if backend=='osf':
            assert face.topology=='osf66' and face.landmarks.shape==(66,2)
            # Canonical OSF66 mouth corners, confirmed against upstream's
            # mouth_corner feature definitions, rather than iBUG68 indices.
            assert np.allclose(np.sort(face.points[3:,0]),np.sort(face.landmarks[[58,62],0]))
        else:
            assert face.topology=='mediapipe478' and face.landmarks.shape==(478,2)
        assert all(p['approximate'] for p in result.metadata['poses'].values())
    finally:cap.release();engine.close()


@pytest.fixture
def video(tmp_path,evidence):
    path=tmp_path/'input.avi'
    image=cv2.imread(str(evidence/'fixtures/grace_hopper.jpg'))
    image=cv2.resize(image,(640,720))
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30,(1280,720))
    assert writer.isOpened()
    for k in range(60):
        frame=np.zeros((720,1280,3),np.uint8); frame[:,320:960]=image
        writer.write(frame)
    writer.release()
    return path


def test_real_video_decode_eof_and_repeat_start_stop(evidence,video):
    cfg=config(evidence)
    plugin=FaceEffectsLivePlugin(cfg)
    try:
        for k in range(3):
            result=plugin.dispatch('start',{'source_kind':'file','source_uri':str(video),'instance_id':'a'})
            assert result['ok'] and result['state']=='running'
            service=plugin.instances['a']
            deadline=time.monotonic()+5
            while service.processed<2 and time.monotonic()<deadline: time.sleep(.02)
            assert service.processed>=2 and service.latest['tracked']==1
            selected=plugin.dispatch('select_effects',{'instance_id':'a','effects':['glasses']})
            assert selected['ok']
            result=plugin.dispatch('snapshot',{'instance_id':'a'})
            assert result['ok'] and result['jpeg_base64']
            assert plugin.dispatch('stop',{'instance_id':'a'})['ok']
            assert not service.processor.is_alive() and not service.capture.is_alive()
            assert service.jpeg is None
    finally: plugin.shutdown()
    source=FileSource(video,False)
    try:
        frames=[]
        while (item:=source.read()) is not None: frames.append(item)
        assert len(frames)==60 and source.eof and frames[-1].timestamp==pytest.approx(59/30)
    finally: source.close(); source.finish()


def test_bounded_queue_under_real_overload(evidence):
    source=PushSource()
    service=Service(config(evidence),source)
    image=cv2.imread(str(evidence/'fixtures/grace_hopper.jpg'))
    service.start()
    try:
        for k in range(400): source.submit(image,k/2000)
        deadline=time.monotonic()+3
        while service.processed<1 and time.monotonic()<deadline: time.sleep(.02)
        assert service.processed>=1
        assert service.overwritten+source.dropped>0
        assert len(service.samples)<=1200
    finally: service.stop()
    assert not service.processor.is_alive() and not service.capture.is_alive()


def test_egl_shader_composition_software_allowed_for_equivalence():
    try: renderer=GL(allow_software=True)
    except Exception as exc: pytest.skip('EGL unavailable: '+type(exc).__name__)
    from plugins.face_effects_live.effects import Sprite
    try:
        image=np.full((128,160,3),80,np.uint8)
        texture=np.ones((32,32,4),np.float32)*.5
        command=Sprite(texture,np.array([[1,0,30],[0,1,20]],float))
        actual=renderer.render(image,[command]); expected=CPU().render(image,[command])
        assert np.max(np.abs(actual[24:46,34:56].astype(int)-expected[24:46,34:56]))<=1
        assert np.array_equal(actual[80:100],expected[80:100])
        assert renderer.render(np.zeros((80,100,3),np.uint8),[]).shape==(80,100,3)
    finally: renderer.close()


def test_real_offline_mcp_http_and_canvas_output(evidence,video):
    from plugins.face_effects_live.server import make_server
    server,plugin=make_server(config(evidence))
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def rpc(method,params={}):
        data=json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode()
        request=urllib.request.Request(base+'/mcp',data,{'Content-Type':'application/json'})
        with opener.open(request,timeout=10) as response: return json.load(response)['result']
    def call(action,**kwargs):
        return json.loads(rpc('tools/call',{'name':'face_effects_live','arguments':{'action':action,**kwargs}})['content'][0]['text'])
    try:
        assert rpc('initialize')['protocolVersion']=='2024-11-05'
        assert rpc('tools/list')['tools'][0]['type']=='processor'
        assert call('list_effects')['effects']==['crown','glasses','orbit']
        assert call('start',source_kind='file',source_uri=str(video))['ok']
        deadline=time.monotonic()+5
        while plugin.instances['default'].processed<2 and time.monotonic()<deadline: time.sleep(.02)
        with opener.open(base+'/frame.jpg',timeout=5) as response:
            assert response.status==200
            assert cv2.imdecode(np.frombuffer(response.read(),np.uint8),1).shape==(720,1280,3)
        with opener.open(base+'/',timeout=5) as response: assert b'<canvas' in response.read()
        assert call('select_effects',effects=[])['ok']
        assert call('stop')['ok']
        with opener.open(base+'/frame.jpg',timeout=5) as response: assert response.status==204
    finally:
        plugin.shutdown(); server.shutdown(); server.server_close(); thread.join(2)
