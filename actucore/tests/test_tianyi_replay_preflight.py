"""Replay guards use current state; deployment review is a separate operation."""
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def preflight(monkeypatch):
    root=Path(__file__).parents[2]/'deploy'
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location('replay_runner',root/'run_tianyi_chain.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    values={'edit':{'code':200,'editor':None},'project':{'running':False},
        'container':{'id':'current-id','image':'current-image','state':'running','health':None,'network':'host'},
        'driver':{'state':'idle','ownership_held':False,'output_active':False,'hands_enabled':False,
            'monotonic_ns':1_000_000_000,'feedback':{'arm_ns':990_000_000,'power_ns':990_000_000,
            'fixed_ns':990_000_000,'power_on':True,'estop':False,'fault':False,'q':[0.]*14,'dq':[0.]*14}},
        'card':{'state':'idle','mode':'shadow','authority_valid':False,'dispatch':{}},
        'arm':{'state':'idle'},'servo':{'state':'idle'},'config':{'mode':'shadow'}}
    monkeypatch.setattr(runner,'core_status',lambda path:values['edit' if 'edit-status' in path else 'project'])
    monkeypatch.setattr(runner,'runtime_container',lambda name:dict(values['container']))
    def rpc(action,tool='teleop_executor',port=15707):
        assert action=='info'
        return values[{'teleop_executor':'driver','teleop':'card'}.get(tool,tool)]
    monkeypatch.setattr(runner,'rpc',rpc)
    def command(*args):
        assert args[:3]==('docker','exec',runner.AC)  # No sudo/publisher/Compose operations.
        return json.dumps(values['config'])
    monkeypatch.setattr(runner,'command',command)
    return runner,values


def test_current_images_pass_without_old_deployment_baseline(preflight):
    runner,values=preflight
    result=runner.replay_preflight()
    assert result[runner.DRIVER]=={'id':'current-id','image':'current-image'}
    values['container']['image']='new-current-image'
    assert runner.replay_preflight()[runner.AC]['image']=='new-current-image'


@pytest.mark.parametrize('section,key,value,reason',[
    ('edit','editor',{'session':'other'},'canvas_occupied'),
    ('edit','code',500,'canvas_occupied'),
    ('project','running',True,'canvas_running'),
    ('project','running',None,'canvas_running'),
    ('container','state','exited','container_not_ready'),
    ('container','health',{'Status':'unhealthy'},'container_not_ready'),
    ('container','network','bridge','container_not_ready'),
    ('driver','ownership_held',True,'driver_not_idle'),
    ('driver','output_active',True,'driver_not_idle'),
    ('driver','calibration_error','bad','driver_conflict'),
    ('driver','feedback_executor_error','broken','driver_conflict'),
    ('driver','foreign_publishers',1,'driver_conflict'),
    ('driver','hands_enabled',True,'driver_conflict'),
    ('card','authority_valid',True,'card_not_isolated'),
    ('card','mode','live','card_not_isolated'),
    ('card','dispatch',{'mailbox_depth':1},'dispatch_pending'),
    ('arm','state','running','legacy_busy'),
    ('servo','state','running','legacy_busy'),
    ('config','mode','live','startup_config_not_shadow'),
])
def test_current_runtime_hazards_still_reject(preflight,section,key,value,reason):
    runner,values=preflight;values[section][key]=value
    with pytest.raises(ValueError,match=reason):runner.replay_preflight()


@pytest.mark.parametrize('key,value,reason',[
    ('arm_ns',0,'feedback_stale'),('power_ns',800_000_000,'feedback_stale'),
    ('fixed_ns',1_100_000_000,'feedback_stale'),('power_on',False,'robot_not_ready'),
    ('estop',True,'robot_not_ready'),('fault',True,'robot_not_ready'),
    ('q',[],'joint_feedback_invalid'),('dq',[float('nan')]*14,'joint_feedback_invalid'),
    ('dq',[.03]*14,'arms_not_stationary'),
])
def test_feedback_checks_retained(preflight,key,value,reason):
    runner,values=preflight;values['driver']['feedback'][key]=value
    with pytest.raises(ValueError,match=reason):runner.replay_preflight()


def test_failed_occupancy_query_does_not_continue(preflight,monkeypatch):
    runner,_=preflight
    def fail(path):raise OSError('unreachable')
    monkeypatch.setattr(runner,'core_status',fail)
    with pytest.raises(OSError):runner.replay_preflight()
