"""Simulated control, source faults, frame geometry and lifecycle contracts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import cv2
import numpy as np
import pytest
from plugins.face_effects_live.plugin import FaceEffectsLivePlugin
from plugins.face_effects_live.sources.capture import USBSource, RTSPSource
from plugins.face_effects_live.tracking import OpticalFlow
from plugins.face_effects_live.service import Service
from plugins.face_effects_live.sources.push import PushSource
from plugins.face_effects_live.filters import OneEuro
from plugins.face_effects_live.contracts import Frame
from plugins.face_effects_live.engine import Engine
from test_engine import Detector, face, frame


@pytest.mark.parametrize('key,value',[('max_faces',0),('max_faces',9),('max_faces',True),
                                     ('detect_interval',0),('detect_interval',6),('detect_interval',1.5),
                                     ('renderer','metal'),('provider','auto'),('source_kind','robot_sdk'),
                                     ('effect','absent'),('source_uri',123),('input_topic',[])])
def test_invalid_config_is_atomic(key,value):
    plugin=FaceEffectsLivePlugin({})
    before=dict(plugin.configs)
    result=plugin.dispatch('config',{key:value})
    assert not result['ok'] and plugin.configs==before


def test_missing_source_has_actionable_error_no_resource_created():
    plugin=FaceEffectsLivePlugin({})
    result=plugin.dispatch('start',{'source_kind':'file','source_uri':'/no/such/video'})
    assert not result['ok'] and not plugin.instances and not plugin.outputs
    assert 'video' in result['error']


def test_canvas_resolves_unique_output_topics_before_start():
    plugin=FaceEffectsLivePlugin({},executor=object())
    one=plugin.dispatch('info',{'instance_id':'a','input_topic':'/source/image'})
    two=plugin.dispatch('info',{'instance_id':'b','input_topic':'/source/image'})
    assert one['topic_in'][0]['topic']=='/source/image'
    assert [p['format'] for p in one['topic_out']]==['image/jpeg','data/json']
    assert one['topic_out'][0]['topic']!=two['topic_out'][0]['topic']
    assert not plugin.instances and not plugin.outputs


def test_ros_context_required_without_importing_robot_modules():
    plugin=FaceEffectsLivePlugin({})
    result=plugin.dispatch('start',{'input_topic':'/test/image'})
    assert not result['ok'] and 'executor' in result['error']


def test_failed_model_start_error_is_visible_until_stop(monkeypatch):
    from plugins.face_effects_live import plugin as module
    monkeypatch.setattr(module,'create_source',lambda *args:PushSource())
    plugin=FaceEffectsLivePlugin({'detector_model':'/missing/model.onnx'})
    result=plugin.dispatch('start',{'source_kind':'file','source_uri':'simulated-source'})
    assert not result['ok'] and 'initialization' in result['error']
    status=plugin.dispatch('info',{})
    assert status['state']=='error' and 'local models' in status['error']
    assert not plugin.instances
    assert plugin.dispatch('stop',{})['ok']
    assert plugin.dispatch('info',{})['state']=='idle'


def test_simulated_usb_disconnect_reconnect_and_explicit_device(monkeypatch):
    opened=[]
    class Capture:
        def __init__(self,device):
            opened.append(device);self.index=len(opened);self.released=False
        def read(self): return (False,None) if self.index==1 else (True,np.ones((12,16,3),np.uint8))
        def set(self,*args): pass
        def release(self): self.released=True
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    source=USBSource('explicit-device')
    assert source.read() is None and source.reconnects==1
    assert source.read().size==(16,12) and source.error==''
    source.close();source.finish();assert source.capture.released
    with pytest.raises(ValueError):USBSource('')


@pytest.mark.parametrize('url',['https://host/video','rtsp://','file:///private.mp4',''])
def test_invalid_rtsp_rejected_without_opening(url):
    with pytest.raises(ValueError):RTSPSource(url)


def test_optical_flow_rejects_blank_and_resolution_change():
    flow=OpticalFlow(); blank=np.zeros((240,320),np.uint8)
    flow.seed(blank,[face()])
    assert not flow.advance(blank)
    flow.seed(blank,[face()])
    assert not flow.advance(np.zeros((480,640),np.uint8))


def test_variable_frame_rate_and_long_sequence_bounded_tracks():
    engine=Engine({},detector=Detector())
    rng=np.random.default_rng(42);timestamp=0.
    try:
        for k in range(300):
            timestamp+=float(rng.uniform(.01,.12))
            result=engine.process(Frame(np.zeros((240,320,3),np.uint8),timestamp,k))
            assert result is not None
            assert len(engine.tracker.tracks)<=4
            assert result.metadata['frame_id']==k
    finally:engine.close()


def test_filter_optimized_step_delay():
    filt=OneEuro()
    for k in range(120):filt([0.],k/30)
    values=[filt([.25],(120+k)/30)[0] for k in range(3)]
    assert values[1]>=.225  # normalized 0.25 step at 30 Hz, <=33.3ms after first sample


def test_failure_strings_do_not_leak_source_credentials():
    service=Service({},PushSource())
    service._fail('input',RuntimeError('rtsp://test:secret@host'))
    assert 'secret' not in str(service.status())


def test_partial_initialization_closes_detector(monkeypatch):
    from plugins.face_effects_live import engine as runtime
    detector=Detector()
    def fail(*args,**kwargs):raise RuntimeError('render init failed')
    monkeypatch.setattr(runtime,'create_renderer',fail)
    with pytest.raises(RuntimeError):Engine({},detector=detector)
    assert detector.closed


def test_bundle_registers_actual_plugin_ros_framework_stubbed():
    # Existing repository stubs: this checks registration, NOT ROS integration.
    import vision_stubs
    import main
    bundle=main.PerceptionBundle({'plugins':{'face_effects_live':{'enabled':True},
                                          'audio_mixer':{'enabled':False},
                                          'echo_cancellation':{'enabled':False}}},None)
    try:
        tool=next(t for t in bundle.get_all_tools() if t['name']=='face_effects_live')
        assert tool['type']=='processor' and tool['multiInstance']
        assert bundle.dispatch('face_effects_live',{'action':'list_effects'})['effects']==['crown','glasses','orbit']
        assert bundle.dispatch('face_effects_live_info',{})['state']=='idle'
        assert bundle.owns('face_effects_live_start')
    finally:bundle.shutdown()
