"""Pure maintenance predicates and orchestration; no Docker, ROS or hardware."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml

path=Path(__file__).parents[2]/'deploy/recover_tianyi_stationary_fault.py'
spec=importlib.util.spec_from_file_location('stationary_recovery',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def status():
    n=20_000_000_000
    return dict(boot_id=m.BOOT,session_id=m.LEASE,state='fault',reason='stop_not_confirmed',
                ownership_held=True,output_active=True,hands_enabled=False,foreign_publishers=[],
                feedback_executor_error=None,monotonic_ns=n,
                last_vendor_command={'monotonic_ns':1,'q_rad':[.0318]*14},
                feedback=dict(arm_ns=n,power_ns=n,fixed_ns=n,q=[0.]*14,dq=[.001]*14,
                              power_on=True,estop=False,fault=False,fixed_body=True))


def test_exact_fault_only_and_no_motion_flag_as_stop_proof():
    d=status();assert m.sample(d,old=True)['q']==[0.]*14
    for key,value in [('boot_id','other'),('session_id','other'),('reason','power_fault'),
                      ('ownership_held',False),('foreign_publishers',['other']),('hands_enabled',True)]:
        bad=copy.deepcopy(d);bad[key]=value
        with pytest.raises(ValueError):m.sample(bad,old=True)


@pytest.mark.parametrize('key,value',[('arm_ns',0),('power_ns',0),('fixed_ns',0),
    ('power_on',False),('estop',True),('fault',True),('fixed_body',False),
    ('q',[float('nan')]*14),('dq',[.021]*14),('q',[.2]*14)])
def test_bad_feedback_rejected(key,value):
    d=status();d['feedback'][key]=value
    with pytest.raises(ValueError):m.sample(d,old=True)


def test_stationarity_requires_continuous_distinct_samples_and_bounded_drift():
    rows=[dict(arm_ns=i*100_000_000,q=[0.]*14) for i in range(21)]
    assert m.stationary(rows)
    assert not m.stationary(rows[:19])
    for mutation in ('duplicate','gap','drift'):
        bad=copy.deepcopy(rows)
        if mutation=='duplicate':bad[5]['arm_ns']=bad[4]['arm_ns']
        if mutation=='gap':bad[5]['arm_ns']+=90_000_000
        if mutation=='drift':bad[5]['q'][0]=.0021
        with pytest.raises(ValueError):m.stationary(bad)


def test_new_driver_requires_new_boot_no_authority_and_no_vendor_write():
    d=status();d.update(boot_id='new',state='idle',ownership_held=False,
                       output_active=False,last_vendor_command=None)
    assert m.sample(d,old=False)
    for key,value in [('boot_id',m.BOOT),('ownership_held',True),('output_active',True),
                      ('last_vendor_command',{})]:
        bad=copy.deepcopy(d);bad[key]=value
        with pytest.raises(ValueError):m.sample(bad,old=False)


def fake(tmp_path,monkeypatch):
    compose=tmp_path/'compose.yml';compose.write_text('services: {}\n')
    config=tmp_path/'config.yml';config.write_text('plugins: {teleop: {mode: shadow}}\n')
    monkeypatch.setattr(m,'COMPOSE_SHA',m.hashlib.sha256(compose.read_bytes()).hexdigest())
    monkeypatch.setattr(m,'EVIDENCE',tmp_path)
    calls=[];generation=[False]
    def shell(*args):
        calls.append(args)
        if args[1:3]==('image','inspect'):
            service=next(k for k,v in m.IMAGES.items() if v==args[-1])
            return json.dumps([{'Id':args[-1],'Config':{'Labels':{'org.phanthy.teleop.service':service}}}])
        if args[1]=='inspect':
            return (m.IMAGES['tianyi2'] if generation[0] else m.OLD_DRIVER) if args[-1]=='driver' else m.IMAGES['actucore-teleop']
        if 'up' in args:generation[0]=True
        return ''
    h=SimpleNamespace(COMPOSE=compose,CONFIG=config,STATE=tmp_path,yaml=yaml,shell=shell,
        SERVICES={'tianyi2':'driver','actucore-teleop':'card'},OTHERS=('core','perception'),
        info=lambda *args:status(),protected=lambda p:None,containers=lambda _: 'unchanged',
        replace_images=lambda before,images:before+'# two image changes\n',
        atomic=lambda p,s:p.write_text(s),guard=lambda:None)
    monkeypatch.setattr(m,'idle_inputs',lambda _:None)
    monkeypatch.setattr(m,'observe',lambda h,old:([{'q':[0.]*14}],status()))
    monkeypatch.setattr(m.os,'geteuid',lambda:0)
    monkeypatch.setattr(m.os,'fchown',lambda *args:None)
    return h,calls


def test_preflight_has_no_mutating_docker_calls(tmp_path,monkeypatch):
    h,calls=fake(tmp_path,monkeypatch);m.run(h)
    assert all(c[1] in ('image','inspect') for c in calls)
    assert not list(tmp_path.glob('maintenance-*.json'))


def test_apply_preserves_failed_stop_evidence_and_scoped_services(tmp_path,monkeypatch):
    h,calls=fake(tmp_path,monkeypatch);m.run(h,True)
    receipt=json.loads(next(tmp_path.glob('maintenance-*.json')).read_text())
    assert receipt['old_stop_passed'] is False
    assert receipt['status']=='maintenance_complete_not_motion_acceptance'
    assert [c[-1] for c in calls if 'up' in c]==['tianyi2','actucore-teleop']
    assert not any(c[-1] in ('core','perception') for c in calls)


def test_changed_hold_command_aborts_before_compose_or_driver_change(tmp_path,monkeypatch):
    h,calls=fake(tmp_path,monkeypatch)
    def changed(*args):
        d=status();d['last_vendor_command']['monotonic_ns']=2;return d
    h.info=changed
    before=h.COMPOSE.read_text()
    with pytest.raises(ValueError,match='command_changed'):m.run(h,True)
    assert h.COMPOSE.read_text()==before and not any('up' in c for c in calls)


def test_legacy_info_only_accepts_exact_ownership_refusal():
    response={'result':{'isError':True,'content':[{'text':json.dumps({'code':'motion_owned_by_teleop','error':'motion_owned_by_teleop'})}]}}
    h=SimpleNamespace(core=lambda p: {'code':200,'editor':None,'running':False},
        info=lambda *a:dict(mode='shadow',state='idle',authority_valid=False,dispatch={}),
        get=lambda *a:response)
    m.idle_inputs(h)
    response['result']['content'][0]['text']=json.dumps({'code':'network_failed','error':'network_failed'})
    with pytest.raises(ValueError,match='unexpected_legacy'):m.idle_inputs(h)
    response['result']['isError']=False
    with pytest.raises(ValueError,match='legacy_exclusion'):m.idle_inputs(h)


@pytest.mark.parametrize('always_gap',[False,True])
def test_observe_restarts_continuous_window_without_extending_total_deadline(monkeypatch,always_gap):
    clock=[0.];counter=[0]
    monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(m.time,'sleep',lambda seconds:None)
    def info(*args):
        counter[0]+=1
        clock[0]+=.2 if always_gap or counter[0]==8 else .1
        d=status();stamp=20_000_000_000+round(clock[0]*1e9)
        d['monotonic_ns']=stamp
        for k in ('arm_ns','power_ns','fixed_ns'):d['feedback'][k]=stamp
        return d
    h=SimpleNamespace(info=info)
    if always_gap:
        with pytest.raises(ValueError,match='stationary_window_missing'):m.observe(h,old=True)
        assert clock[0]<=10.3
    else:
        rows,_=m.observe(h,old=True)
        assert rows[0]['arm_ns']>=20_900_000_000 and m.stationary(rows)
