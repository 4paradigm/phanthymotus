import time
from types import SimpleNamespace
import threading
import numpy as np
import cv2
import pytest
from plugins.face_effects_live.contracts import Frame, Face
from plugins.face_effects_live.filters import OneEuro
from plugins.face_effects_live.tracking import Tracker
from plugins.face_effects_live.engine import Engine
from plugins.face_effects_live.effects import Effects, Sprite
from plugins.face_effects_live.render import CPU, create_renderer
from plugins.face_effects_live.sources.push import PushSource
from plugins.face_effects_live.sources.ros import decode_raw, decode_compressed


def face(x=40, y=40, w=100):
    points = np.array([[x+.25*w, y+.35*w], [x+.75*w, y+.35*w], [x+.5*w, y+.6*w],
                       [x+.32*w, y+.8*w], [x+.68*w, y+.8*w]])
    return Face(np.array([x, y, w, w], float), points, .95)


class Detector:
    def __init__(self, faces=None):
        self.faces = [face()] if faces is None else faces
        self.closed = False
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.faces

    def close(self):
        self.closed = True


def frame(index=0, shape=(240, 320, 3), received=None):
    return Frame(np.full(shape, 80, np.uint8), index/30, index,
                 time.monotonic() if received is None else received)


@pytest.mark.parametrize("color,dtype,shape", [("rgb8",np.uint8,(20,20,3)), ("bgr8",np.float32,(20,20,3)),
                                                    ("bgr8",np.uint8,(20,20,4)), ("bgr8",np.uint8,(1,20,3))])
def test_frame_rejects_formats(color, dtype, shape):
    with pytest.raises(ValueError):
        Frame(np.zeros(shape,dtype),0,0,color=color)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_timestamp(value):
    with pytest.raises(ValueError):
        Frame(np.zeros((20,20,3),np.uint8),value,0)


def test_tracking_confirmation_loss_recovery_and_expiry():
    tracker=Tracker()
    assert not tracker.update([face()],0,(320,240))
    first=tracker.update([face(42)],.033,(320,240))[0].id
    assert tracker.update([],.066,(320,240)) == []
    assert tracker.update([face(45)],.1,(320,240))[0].id == first
    assert tracker.update([],.6,(320,240)) == []
    assert not tracker.update([face()],.7,(320,240))
    assert tracker.update([face()],.733,(320,240))[0].id != first


def test_crossing_ambiguous_faces_hidden_no_duplicate_tracks():
    tracker=Tracker()
    for k in range(8):
        tracker.update([face(40+k*8),face(210-k*8)],k/30,(400,300))
    tracks=tracker.update([face(124),face(126)],8/30,(400,300))
    assert not tracks
    assert len(tracker.tracks) == 2
    assert tracker.ambiguities > 0


def test_independent_ids_stable_when_detection_order_changes():
    tracker=Tracker()
    tracker.update([face(10),face(200)],0,(400,300))
    ids={round(t.face.box[0]):t.id for t in tracker.update([face(200),face(10)],.033,(400,300))}
    assert len(set(ids.values()))==2
    for k in range(2,30):
        actual={round(t.face.box[0]):t.id for t in tracker.update([face(10),face(200)][::(-1 if k%2 else 1)],k/30,(400,300))}
        assert actual==ids


def test_filter_jitter_response_and_timestamp_reset():
    rng=np.random.default_rng(51)
    filt=OneEuro()
    noise=.5+rng.normal(0,.005,(200,2))
    output=np.array([filt(x,k/30) for k,x in enumerate(noise)])
    assert output[20:].std(axis=0).mean()<noise[20:].std(axis=0).mean()*.65
    step=np.array([filt(np.ones(2), (200+k)/30) for k in range(15)])
    assert step[5].mean()>.9
    assert np.array_equal(filt([0,0],0),[0,0])


def test_real_effects_plugins_and_cpu_composition():
    detector=Detector()
    engine=Engine({"renderer":"cpu"},detector=detector)
    try:
        assert engine.process(frame(0)).metadata["tracked"]==0
        output=engine.process(frame(1))
        assert output.metadata["tracked"]==1
        assert np.count_nonzero(output.image != 80)>200
        engine.effects.select([])
        assert np.all(engine.process(frame(2)).image==80)
        with pytest.raises(ValueError): engine.effects.select(["absent"])
    finally: engine.close()
    assert detector.closed
    engine.close()


def test_out_of_order_stale_resolution_and_gap_reset():
    engine=Engine({},detector=Detector())
    try:
        assert engine.process(frame(0))
        first=engine.process(frame(1)).metadata["ids"][0]
        assert engine.process(frame(1)) is None
        assert engine.process(frame(2, received=time.monotonic()-1)) is None
        assert engine.process(frame(3,shape=(480,640,3))).metadata["scene_reset"]
        second=engine.process(frame(4,shape=(480,640,3))).metadata["ids"][0]
        assert first != second
        assert engine.process(frame(40)).metadata["scene_reset"]
    finally: engine.close()


def test_stale_after_inference_is_not_published():
    class Slow(Detector):
        def detect(self,image):
            time.sleep(.025)
            return super().detect(image)
    engine=Engine({"max_age_ms":10},detector=Slow())
    try: assert engine.process(frame()) is None
    finally: engine.close()


def test_backend_failure_does_not_return_stale_result():
    class Broken(Detector):
        def detect(self,image): raise RuntimeError("inference")
    engine=Engine({},detector=Broken())
    try:
        with pytest.raises(RuntimeError): engine.process(frame())
    finally: engine.close()


def test_push_overwrite_copy_and_timestamp_repair():
    source=PushSource(); image=np.ones((20,20,3),np.uint8)
    source.submit(image,10)
    image[:]=2
    source.submit(image,9)
    image[:]=3
    output=source.read()
    assert np.all(output.image==2) and source.dropped==1
    assert source.timestamp_repairs==1 and output.timestamp>10
    source.submit(image,float('nan'))
    assert source.read().metadata['timestamp_repaired']
    source.close(); assert not source.submit(image,11)


@pytest.mark.parametrize("encoding,ch", [("bgr8",3),("rgb8",3),("bgra8",4),("rgba8",4),("mono8",1)])
def test_ros_raw_padded_stride(encoding,ch):
    data=np.zeros((4,6*ch+7),np.uint8)
    data[:,:6*ch]=np.tile(np.arange(ch,dtype=np.uint8),(4,6))
    message=SimpleNamespace(width=6,height=4,step=6*ch+7,encoding=encoding,data=data.tobytes())
    image=decode_raw(message)
    assert image.shape==(4,6,3)
    if encoding=='rgb8': assert list(image[0,0])==[2,1,0]
    message.step-=1
    with pytest.raises(ValueError): decode_raw(message)


def test_ros_compressed_real_codec_and_bad_formats():
    ok,jpeg=cv2.imencode('.jpg',frame().image); assert ok
    message=SimpleNamespace(format='rgb8; jpeg compressed bgr8',data=jpeg.tobytes())
    assert decode_compressed(message).shape==(240,320,3)
    for format,data in [('16UC1; compressedDepth png',jpeg.tobytes()),('jpeg',b'bad'),('unknown',jpeg.tobytes())]:
        message.format,message.data=format,data
        with pytest.raises(ValueError): decode_compressed(message)


@pytest.mark.parametrize("offset", [(-300,-300),(-5,-5),(0,0),(19,19),(1000,1000)])
def test_renderer_clipping_and_alpha(offset):
    rgba=np.ones((16,16,4),np.float32)*.5
    command=Sprite(rgba,np.array([[1,0,offset[0]],[0,1,offset[1]]],float))
    image=np.zeros((20,20,3),np.uint8)
    output=CPU().render(image,[command])
    assert output.shape==image.shape and output.dtype==np.uint8
    if offset==(0,0): assert np.all(output[8,8]==128)
    if offset[0]>100 or offset[0]<-100: assert not output.any()


def test_effect_asset_path_escape_rejected(tmp_path):
    import json
    path=tmp_path/'effects.json'
    path.write_text(json.dumps({'effects':[{'id':'x','plugin':'sticker','asset':'../bad.png'}]}))
    with pytest.raises(ValueError,match='escapes'): Effects(path)


def test_auto_renderer_cpu_fallback(monkeypatch):
    from plugins.face_effects_live import render
    def unavailable(*args,**kwargs): raise RuntimeError('no device')
    monkeypatch.setattr(render,'GL',unavailable)
    renderer,reason=create_renderer('auto')
    assert isinstance(renderer,CPU) and 'no device' in reason
    with pytest.raises(RuntimeError): create_renderer('gl')
