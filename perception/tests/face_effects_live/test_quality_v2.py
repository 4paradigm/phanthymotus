import json
import math
import time
import cv2
import numpy as np
import pytest
from plugins.face_effects_live.continuity import SceneContinuity
from plugins.face_effects_live.contracts import Face
from plugins.face_effects_live.filters import SimilarityFilter
from plugins.face_effects_live.effects import Effects
from plugins.face_effects_live.tracking import Tracker, Track
from plugins.face_effects_live.telemetry import tag_jpeg, read_tag
from plugins.face_effects_live.pose import estimate_pose
from plugins.face_effects_live.plugin import FaceEffectsLivePlugin


def face(x, signature):
    f=Face(np.array([x,30,80,100.]),np.array([[x+20,60],[x+60,60],[x+40,90],[x+25,110],[x+55,110.]]),.9)
    f.appearance=np.array(signature,dtype=float)
    return f


def test_scene_cut_and_global_flash_are_distinguished():
    rng=np.random.default_rng(2)
    image=rng.integers(40,180,(360,640,3),np.uint8)
    detector=SceneContinuity()
    assert not detector.update(image)
    assert not detector.update((image+35).astype(np.uint8))
    assert detector.update(rng.integers(0,255,(360,640,3),np.uint8))


def test_transient_appearance_disambiguates_crossing_and_is_not_persisted():
    tracker=Tracker()
    for k in range(5):
        tracker.update([face(50+k*5,[1,0]),face(130-k*5,[0,1])],k/30,(400,300))
    ids={tuple(t.appearance):t.id for t in tracker.tracks}
    tracks=tracker.update([face(90,[0,1]),face(90,[1,0])],5/30,(400,300))
    assert len(tracks)==2
    assert {tuple(t.appearance):t.id for t in tracks}==ids
    tracker.reset()
    assert not tracker.tracks


def test_similarity_smoothing_preserves_eye_pair_and_angle_wrap():
    filt=SimilarityFilter()
    points=np.array([[100.,100.],[160,100],[130,135],[110,155],[150,155]])
    out=None
    for k in range(20):
        angle=math.radians(179 if k<10 else -179)
        rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        observed=(points-points[:2].mean(0))@rotation.T+[200,200]
        out=filt(observed,k/30,(640,480),np.array([100,100,150,200]))
        assert np.linalg.norm(out[1]-out[0])==pytest.approx(60,abs=1e-5)
    assert out[0,0]>out[1,0]


def test_glasses_sprite_eye_anchors_match_stabilized_eyes():
    effect=Effects()
    f=face(100,[1,0])
    track=Track(1,f,0,points=f.points.copy())
    try:
        commands=effect.plugins['glasses'].commands(track,0,{'yaw_proxy':0})
        assert commands
        actual=np.array([[62,44],[178,44]])@commands[0].matrix[:,:2].T+commands[0].matrix[:,2]
        assert np.allclose(actual,track.points[:2])
        assert not effect.plugins['glasses'].commands(track,0,{'yaw_proxy':.7})
    finally:effect.close()


def test_jpeg_provenance_roundtrip_keeps_decoded_pixels_exact():
    image=np.random.default_rng(4).integers(0,255,(32,48,3),np.uint8)
    ok,encoded=cv2.imencode('.jpg',image);assert ok
    original=encoded.tobytes()
    tagged=tag_jpeg(original,{'frame_id':9,'source_timestamp':12.5,'source_metadata':{'ros_frame_id':'recorded:7:123'}})
    assert read_tag(tagged)['source_frame_id']=='recorded:7:123'
    assert np.array_equal(cv2.imdecode(np.frombuffer(tagged,np.uint8),1),cv2.imdecode(encoded,1))
    assert read_tag(original) is None
    assert read_tag(tagged[:8]) is None


def test_generic_pnp_never_claims_metric_pose_with_calibrated_camera():
    face_value=face(100,[1,0])
    result=estimate_pose(face_value,(640,480),np.array([[600,0,320],[0,600,240],[0,0,1]]))
    assert result['approximate'] and result['camera_calibrated']


def test_canvas_named_effect_overrides_initial_list_and_prior_hot_selection():
    plugin=FaceEffectsLivePlugin({'effects':['crown','glasses','orbit']})
    assert plugin.dispatch('config',{'effect':'glasses'})['ok']
    assert 'effects' not in plugin._config('default')
    assert plugin.dispatch('select_effects',{'effects':['orbit']})['ok']
    assert plugin._config('default')['effects']==['orbit']
    assert plugin.dispatch('config',{'effect':'none'})['ok']
    assert 'effects' not in plugin._config('default')


def test_raw_ros_input_is_not_advertised_as_compressed_message():
    plugin=FaceEffectsLivePlugin({'source_kind':'ros_image','input_topic':'/configured/image'})
    result=plugin.dispatch('info',{})
    assert result['topic_in']==[{'topic':'/configured/image','format':'image/raw','desc':'camera'}]


@pytest.mark.parametrize('identity',['default','ros_image','ros_compressed','123','人脸实例'])
def test_all_generated_ros_topic_tokens_start_with_valid_characters(identity):
    import re
    plugin=FaceEffectsLivePlugin({})
    assert all(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',part) for part in plugin._topic(identity).split('/') if part)


@pytest.mark.parametrize('cfg,reason',[
    ({'detector_backend':'mediapipe','provider':'cuda'},'provider=cpu'),
    ({'landmarker_backend':'hybrid'},'OSF fallback'),
    ({'landmarker_backend':'unknown'},'landmarker_backend')])
def test_invalid_backend_combinations_do_not_silently_downgrade(cfg,reason):
    from plugins.face_effects_live.engine import Engine
    with pytest.raises(ValueError,match=reason):Engine(cfg)
