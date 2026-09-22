import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).parents[2]/'deploy'
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
observer=module('record_tianyi_chain');report=module('summarize_tianyi_chain')

def test_observer_preserves_vendor_units_errors_and_monotonic_receipt():
    motor=SimpleNamespace(name=11,pos=.25,spd=1.,cur=0.,speed=-.2,error=7)
    cmd=observer.command_row(SimpleNamespace(cmds=[motor]),100)
    status=observer.status_row(SimpleNamespace(status=[motor]),120)
    assert cmd['observed_ns']==100 and cmd['motors'][0]['q_rad']==.25
    assert status['motors'][0]['dq_rad_s']==-.2 and status['motors'][0]['error']==7
    assert cmd['command_sequence_available'] is False

def test_async_observer_flush_and_failure_are_visible():
    out=io.StringIO();writer=observer.AsyncRows(out)
    for i in range(100):writer.put({'sequence':i})
    writer.finish()
    assert [json.loads(x)['sequence'] for x in out.getvalue().splitlines()]==list(range(100))
    assert writer.dropped==0
    class Bad:
        def write(self,text):raise OSError('full')
    failed=observer.AsyncRows(Bad());failed.put({'row':1})
    with pytest.raises(RuntimeError,match='observer_write_failed'):failed.finish()

def command(stamp,q):
    return {'observed_ns':stamp,'motors':[{'id':i,'q_rad':q} for i in list(range(11,18))+list(range(21,28))]}

def test_vendor_matching_does_not_claim_identity_for_repeated_targets():
    event={'event_id':1,'publish_started_ns':1000,'publish_returned_ns':2000,'q_rad':[.1]*14}
    assert report.correlate([event],[command(2100,.1)])[0]['certainty']=='unique_candidate'
    assert report.correlate([event],[command(2100,.1),command(3100,.1)])[0]['certainty']=='ambiguous'
    assert report.correlate([event],[command(2100,.2)])[0]['candidate_count']==0
    assert report.correlate([event],[command(999,.1)])[0]['candidate_count']==0

def test_report_flags_lost_events_and_missing_observers_without_accuracy_gate(tmp_path):
    trace={'trace_id':'x','event_count':3,'events':[{'event_id':3,'event':'receive_batch','monotonic_ns':100,'superseded':4}]}
    (tmp_path/'ros-driver.jsonl').write_text(json.dumps({'trace':trace})+'\n')
    result=report.summarize(tmp_path)
    assert result['driver_missing_event_ids']=={'x':[1,2]}
    assert not result['same_robot_monotonic_clock'] and result['observer_ends']==0
    assert result['tracking_error_policy']=='report_only'

def test_report_scopes_loss_to_started_trace_and_preserves_real_gaps(tmp_path):
    (tmp_path/'trace-start.json').write_text(json.dumps({'trace_id':'current'}))
    def trace(key,count,ids):
        return {'trace':{'trace_id':key,'event_count':count,'events':[
            {'event_id':i,'event':'receive_batch','monotonic_ns':i,'superseded':0} for i in ids]}}
    (tmp_path/'ros-driver.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        trace('old',100,[99,100]),trace('current',3,[1,3])]))
    result=report.summarize(tmp_path)
    assert result['driver_missing_event_ids']=={'current':[2]}
    assert result['other_trace_ids_observed']==['old']
    assert result['driver_trace_id']=='current'

@pytest.mark.parametrize('count',[1,3])
def test_paired_and_legacy_orchestration_reach_combined_only_after_execution(tmp_path,monkeypatch,count):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'plugins'))
    from teleop import acceptance as a
    profile=tmp_path/'profile.json';profile.write_text('{}')
    (tmp_path/'poses.jsonl').write_text('source')
    calls=[]
    link=SimpleNamespace(lease=None,management_request=None,feedback=lambda:{'feedback':{'q':[0.]*14}},close=lambda:None)
    clean=SimpleNamespace(shutdown=lambda:None,join=lambda **kw:None)
    monkeypatch.setattr(a,'TianyiIK',lambda p:object())
    monkeypatch.setattr(a,'idle',lambda:None)
    monkeypatch.setattr(a,'call',lambda *args:{'state':'ready','first_acceptance_prepared':True})
    monkeypatch.setattr(a,'ros_link',lambda cfg:(link,clean,clean,clean))
    monkeypatch.setattr(a,'schedule',lambda *args:[])
    monkeypatch.setattr(a,'journal_link',lambda *args:None)
    monkeypatch.setattr(a,'execute',lambda *args:calls.append('execute'))
    monkeypatch.setattr(a,'report',lambda d:[{'passed':True}])
    monkeypatch.setattr(a,'combined',lambda *args:calls.append('combined'))
    package={'profile_sha256':a.digest(profile),'recording_sha256':a.digest(tmp_path/'poses.jsonl'),'compiled_sources':{}}
    cfg={'calibration_path':str(profile),'capture':{'state_file':str(tmp_path/'state.json')}}
    a.run(cfg,tmp_path,package,rounds=count)
    assert calls==['execute']*count+['combined']
    manifest=json.loads(next(tmp_path.glob('execution-*/run.json')).read_text())
    assert manifest['execution_rounds']==count

def test_lightweight_runtime_reads_do_not_invoke_public_snapshot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'plugins'))
    from teleop.runtime import TeleopRuntime
    from teleop.dispatch import RecordingAdapter
    runtime=TeleopRuntime(mode='shadow',adapter=RecordingAdapter(),auto_watchdog=False)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        monkeypatch.setattr(runtime,'_public_snapshot_locked',lambda *args:(_ for _ in ()).throw(AssertionError('heavy copy')))
        assert runtime.heartbeat(binding,include_status=False)=={}
        assert runtime.control_status()['dispatch']['fault_code'] is None
        with pytest.raises(AssertionError,match='heavy copy'):runtime.status()
    finally:
        monkeypatch.undo();runtime.close()

def test_replay_drops_superseded_inputs_but_never_restamps_an_expired_tail(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'plugins'))
    from teleop import acceptance as a
    now=[0.]
    monkeypatch.setattr(a.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(a.time,'sleep',lambda wait:now.__setitem__(0,now[0]+wait))
    data=[{'received_ns':i*10_000_000,'frame':{'sequence':i}} for i in range(40)]
    iterator=a.paced_latest_inputs(data,0.)
    assert next(iterator)[0]['frame']['sequence']==0
    now[0]=.135
    row,timing=next(iterator)
    assert row['frame']['sequence']==13 and timing['superseded_count']==12
    assert timing['input_lateness_ms']==pytest.approx(5)
    now[0]=.6
    with pytest.raises(ValueError,match='no_fresh_input'):next(iterator)
