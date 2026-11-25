from vision_stubs import PERCEPTION_ROOT
from plugins import asr

def fields(estimates, **kw):
    args=dict(text='小范小范，你知道我在哪吗',char_ends=list(range(1,14)),
              end_pos=4,keyword_phones=4,start_ts=1700000000.,end_ts=1700000004.,
              pre_roll_bytes=16000,estimates=estimates,trusted_audio=True)
    args.update(kw)
    return asr._estimated_wake_direction_fields(**args)

def test_estimate_is_labeled_and_does_not_fabricate_capture_windows():
    r=fields([(80,1700000000.8),(85,1700000001.3)])
    assert r['speaker_direction_state']=='estimated'
    assert '估计' in r['speaker_direction']
    assert not any('window' in k for k in r)

def test_previous_speaker_delayed_result_and_untrusted_audio_are_unknown():
    for samples in ([(90,1699999999.)],[(90,1700000003.),(90,1700000003.5)],
                    [(90,1700000000.8)],[(90,1700000000.8),(90,1700000000.8)],
                    [(90,1700000000.8),(-90,1700000001.3)]):
        assert fields(samples)=={'speaker_direction_state':'unknown'}
    assert fields([(80,1700000000.8),(85,1700000001.3)],trusted_audio=False)=={'speaker_direction_state':'unknown'}

def test_prefixed_wake_unknown_and_rear_wrap_supported():
    samples=[(179,1700000000.8),(-179,1700000001.3)]
    assert fields(samples)['speaker_direction_state']=='estimated'
    assert fields(samples,end_pos=5)=={'speaker_direction_state':'unknown'}


def test_callback_is_opt_in_and_deduplicates_heartbeats(monkeypatch):
    import json, threading
    from collections import deque
    from types import SimpleNamespace
    node=object.__new__(asr._ASRNode)
    node._direction_lock=threading.Lock()
    node._direction_windows=deque()
    node._direction_estimates=deque()
    payload=dict(state='fresh',angle_deg=80,angle_convention=asr.DIRECTION_ANGLE_CONVENTION,
                 source_backend='native', timestamp_basis='receipt_time_only',association_state='unknown',
                 timestamp_ms=1700000000800,audio_window_start_us=0,audio_window_end_us=0)
    monkeypatch.setattr(asr.time,'time',lambda:1700000000.9)
    node._receipt_direction_experiment=False
    node._direction_cb(SimpleNamespace(data=json.dumps(payload)))
    assert not node._direction_estimates
    node._receipt_direction_experiment=True
    for _ in range(3):node._direction_cb(SimpleNamespace(data=json.dumps(payload)))
    assert list(node._direction_estimates)==[(80,1700000000.8)]
    assert not node._direction_windows
    payload['timestamp_ms']-=5000
    node._direction_cb(SimpleNamespace(data=json.dumps(payload)))
    assert len(node._direction_estimates)==1


def test_estimated_angle_rounding_keeps_public_rear_and_zero_convention():
    assert fields([(-179.999,1700000000.8),(-179.999,1700000001.3)])['speaker_direction'].startswith('估计 180°')
    assert fields([(-.001,1700000000.8),(-.001,1700000001.3)])['speaker_direction'].startswith('估计 0°')
