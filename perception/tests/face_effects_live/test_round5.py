"""Regression contracts for compact transport and bounded single-person recovery."""
import numpy as np
import cv2
import pytest

from plugins.face_effects_live.osf_input import prepare_crop


@pytest.mark.parametrize('shape',[(8,11),(80,120),(224,224),(700,900)])
def test_compact_preprocessing_preserves_frozen_float_input(shape):
    rng=np.random.default_rng(9);image=rng.integers(0,256,(*shape,3),dtype=np.uint8)
    # Frozen release contract: float conversion happens BEFORE interpolation.
    rgb=image[:,:,::-1].astype(np.float32)
    old=cv2.resize(rgb,(224,224))/255.
    old=(old-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    expected=np.ascontiguousarray(old.transpose(2,0,1)[None])
    assert np.array_equal(prepare_crop(image),expected)


def test_bounded_rpc_rejects_empty_and_unbounded_work_without_starting_worker(monkeypatch):
    from plugins.ort_worker import OrtWorker
    worker=OrtWorker();started=[]
    monkeypatch.setattr(worker,'_ensure_started',lambda:started.append(True))
    for feeds in ([],[{}]*9,None):
        with pytest.raises(ValueError):worker.run_many('missing',None,feeds)
    assert not started


def test_actual_compact_batch_preserves_landmarks_isolation_and_crash_reload():
    import os
    from pathlib import Path
    root=os.environ.get('FACE_EFFECTS_EVIDENCE')
    if not root:pytest.skip('set FACE_EFFECTS_EVIDENCE for real model regression')
    from plugins.face_effects_live.backends import OpenSeeFace,YuNet
    root=Path(root);image=cv2.imread(str(root/'fixtures/grace_hopper.jpg'))
    image=cv2.resize(image,(256,300))
    detector=YuNet(root/'models/yunet.onnx');compact=OpenSeeFace(root/'models/osf0.onnx')
    legacy=OpenSeeFace(root/'models/osf0.onnx',batch=False)
    try:
        faces=detector.detect(image);assert faces
        # Repeated real crops test ordering/numerical transport, not multi-person GT.
        faces=faces*3
        before=legacy.refine(image,faces);actual=compact.refine(image,faces)
        assert len(before)==len(actual)==3
        assert compact.last_profile['request_array_bytes']<3*3*224*224*4
        assert compact.last_profile['worker_inference_ms']>0
        for old,new in zip(before,actual):
            assert np.array_equal(old.landmarks,new.landmarks)
            assert np.array_equal(old.points,new.points)
        host=compact.worker;key=legacy.session_key
        host._proc.terminate();host._proc.join(timeout=5)
        after=compact.refine(image,faces)
        for old,new in zip(before,after):assert np.array_equal(old.landmarks,new.landmarks)
        compact.close()
        assert key in host._loaded and host.alive
        assert legacy.refine(image,faces)
    finally:detector.close();compact.close();legacy.close()


def test_model_switch_keeps_track_but_resets_old_anchor_filter():
    from plugins.face_effects_live.contracts import Face
    from plugins.face_effects_live.tracking import Tracker
    p=np.array([[60.,70.],[100.,70.],[80.,90.],[65.,110.],[95.,110.]])
    def face(dx,source):return Face(np.array([40.,40.,80.,100.]),p+[dx,0],.9,landmark_source=source)
    tracker=Tracker();tracker.update([face(0,'fast')],0,(200,200))
    old=tracker.update([face(2,'fast')],1/30,(200,200))[0]
    current=face(12,'detail');new=tracker.update([current],2/30,(200,200))[0]
    assert old.id==new.id
    assert np.array_equal(new.points,current.points)


def test_detail_scheduling_keeps_every_multi_face_and_falls_back_on_missing():
    from plugins.face_effects_live.backends import SingleFaceDetail
    from plugins.face_effects_live.contracts import Face
    class Backend:
        providers=['CPUExecutionProvider'];fallback_reason='';last_profile={}
        def __init__(self):self.calls=[];self.missing=False
        def refine(self,image,faces):self.calls.append(len(faces));return [] if self.missing else faces
        def close(self):pass
    fast,detail=Backend(),Backend();adaptive=SingleFaceDetail(fast,detail)
    p=np.array([[70.,80.],[120.,80.],[95.,100.],[75.,130.],[115.,130.]])
    f=Face(np.array([40.,40.,120.,140.]),p,.9);image=np.zeros((240,240,3),np.uint8)
    assert len(adaptive.refine(image,[f]))==1 and fast.calls==[1] and detail.calls==[1]
    assert adaptive.last_profile['quality_tier']=='detail_consistent'
    # Metadata crosses JSON on the real ROS/Core path. numpy.bool_ would make
    # Service fail after a successful inference while file-only tests passed.
    import json
    assert type(adaptive.last_profile['detail_accepted']) is bool
    json.dumps(adaptive.last_profile,allow_nan=False)
    assert len(adaptive.refine(image,[f]*6))==6 and fast.calls==[1,6] and detail.calls==[1]
    detail.missing=True
    assert len(adaptive.refine(image,[f]))==1 and fast.calls==[1,6,1]
    assert adaptive.last_profile['quality_tier']=='fast_after_detail_missing'


def test_detail_guard_keeps_small_faces_and_rejects_inconsistent_points():
    from plugins.face_effects_live.backends import SingleFaceDetail
    from plugins.face_effects_live.contracts import Face
    p=np.array([[70.,80.],[120.,80.],[95.,100.],[75.,130.],[115.,130.]])
    class Backend:
        providers=[];fallback_reason='';last_profile={}
        def __init__(self,shift):self.shift=shift;self.calls=0
        def refine(self,image,faces):
            self.calls+=1
            return [Face(f.box.copy(),f.points+[self.shift,0],.9) for f in faces]
        def close(self):pass
    fast,detail=Backend(0),Backend(4);adaptive=SingleFaceDetail(fast,detail)
    image=np.zeros((240,240,3),np.uint8)
    large=Face(np.array([40.,40.,120.,140.]),p,.9)
    result=adaptive.refine(image,[large])[0]
    assert np.array_equal(result.points,p) and result.landmark_source=='osf0_fast'
    assert adaptive.last_profile['quality_tier']=='fast_after_detail_disagreement'
    small=Face(np.array([40.,40.,60.,80.]),p,.9)
    result=adaptive.refine(image,[small])[0]
    assert detail.calls==1 and np.array_equal(result.points,p)
    assert result.landmark_source=='osf0_fast'
    clipped=Face(np.array([-2.,40.,120.,140.]),p,.9)
    adaptive.refine(image,[clipped]);assert detail.calls==1


def test_pixel_bridge_tracks_current_image_and_stops_after_two_misses():
    from plugins.face_effects_live.contracts import Face,Frame
    from plugins.face_effects_live.engine import Engine
    rng=np.random.default_rng(42)
    image=rng.integers(0,256,(200,220,3),dtype=np.uint8)
    points=np.array([[70.,80.],[100.,80.],[85.,100.],[75.,115.],[95.,115.]])
    class Detector:
        def __init__(self):self.index=0
        def detect(self,image):
            k=self.index;self.index+=1
            if k in (2,3,4):return []
            return [Face(np.array([55.+k,50.,70.,90.]),points+[k,0],.95)]
        def close(self):pass
    engine=Engine({'smooth':False,'scene_cuts':False,'effects':[]},detector=Detector())
    try:
        results=[]
        for k in range(6):
            moved=cv2.warpAffine(image,np.array([[1.,0,k],[0,1.,0]]),(220,200))
            results.append(engine.process(Frame(moved,k/30,k,source='synthetic_translation_oracle')))
        assert results[0].metadata['tracked']==0 and results[1].metadata['tracked']==1
        for k in (2,3):
            assert results[k].metadata['flow_bridge']
            assert results[k].metadata['model_face_count']==0
            anchors=np.array(next(iter(results[k].metadata['anchors'].values())))
            assert np.allclose(anchors,points+[k,0],atol=.2)
        assert not results[4].metadata['flow_bridge'] and results[4].metadata['tracked']==0
        assert not results[5].metadata['flow_bridge'] and results[5].metadata['tracked']==1
        assert results[5].metadata['ids']==results[1].metadata['ids']
    finally:engine.close()


def test_bridge_never_confirms_a_single_detection_or_crosses_source_reset():
    from plugins.face_effects_live.contracts import Face,Frame
    from plugins.face_effects_live.engine import Engine
    rng=np.random.default_rng(18);image=rng.integers(0,256,(160,180,3),dtype=np.uint8)
    points=np.array([[50.,50.],[90.,50.],[70.,75.],[55.,95.],[85.,95.]])
    class Detector:
        def __init__(self):self.count=0
        def detect(self,image):
            self.count+=1
            return [Face(np.array([30.,30.,90.,100.]),points.copy(),.9)] if self.count==1 else []
        def close(self):pass
    engine=Engine({'effects':[],'scene_cuts':False},detector=Detector())
    try:
        engine.process(Frame(image,0,0))
        r=engine.process(Frame(image,1/30,1))
        assert not r.metadata['flow_bridge'] and r.metadata['tracked']==0
        r=engine.process(Frame(image,2/30,2,metadata={'timestamp_repaired':True}))
        assert r.metadata['scene_reset'] and not r.metadata['flow_bridge']
    finally:engine.close()
