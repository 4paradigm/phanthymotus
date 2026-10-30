"""Cross-robot policy contracts and plugin wiring, no GPU, ROS or motors."""
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plugins.headfollow.policy import Policy, negotiate, observation
from plugins.headfollow.plugin import HeadFollowPlugin, DEFAULT_TOPIC

FIXTURE=Path(__file__).parent/'fixtures/head_tron2.json'

def descriptor():
    return json.loads(FIXTURE.read_text())

class World:
    def __init__(self,d=None):
        self.d=d or descriptor();self.t=1000.;self.tick=0
        self.q=list(self.d['head_tracking']['neutral']);self.points=[(0.,0.)]
        self.req=None;self.owner=None;self.session='test-driver-1';self.ready=True;self.stationary=True;self.certified=False
        self.p=Policy(self.d,lambda:self.t,lambda:self.t)
    def det(self):
        return {'timestamp':self.t,'count':len(self.points),'objects':[{'name':'person','confidence':.9,'position':list(p)} for p in self.points],'latency_ms':20}
    def state(self):
        return {'schema':'motus.head-state/1','session':self.session,'timestamp':self.t,
            'available':True,'stream_enabled':True,'ready':self.ready,'stationary':self.stationary,
            'sample_stamp':self.tick,'values':self.q[:],'velocities':[0.]*len(self.q),
            'motion_id':self.owner,'request':copy.deepcopy(self.req),'completion_verified':self.certified}
    def enable(self):self.p.enable(self.det(),self.state())
    def step(self,dt=.3):
        self.t+=dt;self.tick+=1
        return self.p.step(self.det(),self.state())
    def run(self,n=8):
        out=[]
        for _ in range(n):
            m=self.step()
            if m:out.append(m)
        return out
    def complete(self,msg):
        self.q=list(msg['values']);self.owner='action-'+str(msg['seq']);self.certified=True
        self.req={'source':msg['source'],'seq':msg['seq'],'outcome':'completed','action_id':self.owner}
        self.ready=self.stationary=True


def test_real_vop_shape_and_optional_bbox_not_custom_gpu_payload():
    f={'timestamp':1000.,'count':1,'objects':[{'name':'person','position':[.2,-.3],'confidence':.93,'bbox':[10,20,80,90]}]}
    assert observation(f,1000.)['people']==[(.2,-.3)]
    assert observation(f,1000.)['time_basis']=='vop_publication'
    f['latency_ms']=50
    assert observation(f,1000.)['obs_ms']==999950

@pytest.mark.parametrize('bad',[{}, {'timestamp':1000,'count':0,'objects':[],'latency_ms':3000},
    {'timestamp':1000,'count':1,'objects':[{'name':'person','position':[2,0],'confidence':.9}]},
    {'timestamp':1000,'count':2,'objects':[]}])
def test_bad_detection_does_not_count_as_absence(bad):
    with pytest.raises(ValueError):observation(bad,1000)

@pytest.mark.parametrize('mutate',[
    lambda d:d.update(control_interface='other'),lambda d:d.update(mode='twist'),
    lambda d:d.pop('head_tracking'),lambda d:d.update(dof=14),
    lambda d:d['units'].update(angle='deg'),lambda d:d['limits'].pop('max_velocity'),
    lambda d:d['head_tracking']['axes']['yaw'].update(index=0),
    lambda d:d['head_tracking']['axes']['yaw'].update(sign=0),
    lambda d:d['head_tracking'].update(neutral=[99,0]),
    lambda d:d['head_tracking'].update(feedback_topic='http://robot'),
])
def test_incompatible_or_undeclared_head_fails_before_any_output(mutate):
    d=descriptor();mutate(d)
    with pytest.raises((ValueError,TypeError)):Policy(d)


def test_discovery_and_lifecycle_do_not_arm_or_publish():
    p=HeadFollowPlugin({},executor=None)
    assert p.get_tools()[0]['name']=='headfollow'
    assert p.get_tools()[0]['topic_out']==[{'topic':DEFAULT_TOPIC,'format':'control/joint','desc':'经过协商的 motus.control/1 头部目标'}]
    assert list(p.get_tools()[0]['inputSchema']['x-action-params'])==['follow']
    p.start();assert not p.dispatch('headfollow',{'action':'info'})['active']
    assert not p.dispatch('headfollow',{'action':'stop'})['active']


def test_centered_person_does_not_move():
    w=World();w.enable();assert w.run()==[];assert w.p.state=='tracking'


def test_tron2_native_axis_direction_full_target_not_four_degree_steps():
    w=World();w.points=[(.85,.6)];w.enable();m=w.run()[0]
    assert m['values'][0]>0 and m['values'][1]<0
    assert abs(m['values'][1])>math.radians(4)
    assert m['duration_s']>=max(abs(q)/v for q,v in zip(m['values'],w.d['limits']['max_velocity']))
    assert m['obs_stamp_ms']<m['stamp_ms']
    assert m['reference']['values']==[0.,0.]
    assert m['head_session']==w.session
    assert len(w.run(20))==0


def test_different_robot_order_sign_neutral_and_limits_no_policy_change():
    d=descriptor();d['joint_names']=['pan','tilt']
    d['head_tracking'].update(axes={'yaw':{'index':0,'sign':-1,'image_axis':0,'gain_rad':.26},'pitch':{'index':1,'sign':1,'image_axis':1,'gain_rad':.22}},neutral=[.1,-.1])
    d['limits']={'lower':[-.4,-.3],'upper':[.4,.3],'max_velocity':[.2,.2]}
    w=World(d);w.points=[(.8,.6)];w.enable();m=w.run()[0]
    assert m['values'][0]>.1 and m['values'][1]<-.1
    assert -.4<=m['values'][0]<=.4 and -.3<=m['values'][1]<=.3
    w.complete(m);w.step();w.points=[]
    home=w.run(10)[0]
    assert home['values']==[.1,-.1]


def test_pan_only_gimbal_supported_no_invented_pitch_axis():
    d=descriptor();d.update(dof=1,joint_names=['pan'])
    d['limits']={'lower':[-.6],'upper':[.6],'max_velocity':[.2]}
    d['head_tracking'].update(axes={'yaw':{'index':0,'sign':1,'image_axis':0,'gain_rad':.26}},neutral=[.1])
    w=World(d);w.points=[(.8,.8)];w.enable();m=w.run()[0]
    assert len(m['values'])==1 and m['values'][0]<.1


def test_initial_no_person_and_lost_person_return_declared_home_once():
    for acquired in (False,True):
        w=World();w.q=[0.,.3];w.points=[(0.,0.)] if acquired else [];w.enable()
        if acquired:w.run(5);w.points=[]
        m=w.run(10)[0];assert m['kind']=='return_home' and m['values']==[0.,0.]
        w.complete(m);w.step();assert w.run(20)==[]


def test_short_occlusion_and_invalid_camera_result_never_home():
    w=World();w.q=[0.,.3];w.enable();w.run(5);w.points=[]
    assert w.run(4)==[]
    for _ in range(20):
        w.t+=.3;w.tick+=1;assert w.p.step({},w.state()) is None
    assert w.p.seq==0 and w.p.enabled


def test_repeated_publication_cannot_confirm_loss():
    w=World();w.q=[0.,.3];w.points=[];w.enable();f=w.det()
    for _ in range(20):
        w.t+=.3;w.tick+=1;w.p.step(f,w.state())
    assert w.p.seq==0


def test_stop_does_not_publish_home_or_next_target():
    w=World();w.points=[(.8,0.)];w.enable();m=w.run()[0];w.p.stop();w.complete(m)
    assert w.run(20)==[] and w.p.seq==1

@pytest.mark.parametrize('outcome',['pending','accepted','completed'])
def test_accepted_or_stopped_without_verified_arrival_is_not_success(outcome):
    w=World();w.points=[(.8,0.)];w.enable();m=w.run()[0]
    w.req={'source':m['source'],'seq':m['seq'],'outcome':outcome}
    if outcome=='completed':w.certified=False
    assert w.run(8)==[]
    assert w.p.seq==1
    if outcome=='completed':assert w.p.state=='fault'

@pytest.mark.parametrize('outcome',['failed','rejected','unknown'])
def test_failed_or_unknown_request_never_retried_or_homed(outcome):
    w=World();w.points=[(.8,0.)];w.enable();m=w.run()[0]
    w.req={'source':m['source'],'seq':m['seq'],'outcome':outcome};w.points=[]
    assert w.run(20)==[] and w.p.state=='fault' and w.p.seq==1
    with pytest.raises(ValueError):w.enable()


def test_real_completion_then_wait_for_later_detection_before_next_command():
    w=World();w.points=[(.8,0.)];w.enable();m=w.run()[0];w.complete(m);w.step()
    stale=w.det();w.tick+=1
    assert w.p.step(stale,w.state()) is None
    m2=w.step();assert m2 is not None and m2['seq']==2

@pytest.mark.parametrize('change',['session','motion','unavailable','busy','stopped_stream'])
def test_other_owner_or_stale_feedback_blocks_follow(change):
    w=World();w.enable();w.run(5);s=w.state()
    if change=='session':s['session']='other'
    elif change=='motion':s['motion_id']='other'
    elif change=='unavailable':s['available']=False
    elif change=='stopped_stream':s['stream_enabled']=False
    else:s['ready']=False
    assert w.p.step(w.det(),s) is None and not w.p.enabled


def test_missing_ack_timeout_does_not_resend_or_auto_home():
    w=World();w.points=[(.8,0.)];w.enable();w.run();w.t+=40;w.tick+=1
    w.p.step(w.det(),w.state());assert w.p.state=='fault' and w.p.seq==1


def test_wrong_vop_camera_is_rejected_without_constructing_ros_node(monkeypatch):
    import rclpy
    make=Mock();monkeypatch.setattr(rclpy,'create_node',make,raising=False)
    p=HeadFollowPlugin({},None)
    r=p.dispatch('headfollow',{'action':'start','input_topic':'/limx_tron2/camera_left/objects','control_interface':descriptor()})
    assert r['state']=='error';make.assert_not_called()

class Node:
    def __init__(self):self.subs=[];self.pub=Mock();self.timer=None
    def create_publisher(self,*args):return self.pub
    def create_subscription(self,typ,topic,callback,profile):self.subs.append((topic,callback,profile));return object()
    def create_timer(self,interval,callback):self.timer=callback;return object()
    def destroy_node(self):pass


def test_real_plugin_wiring_qos_lifecycle_and_old_callback_generation(monkeypatch):
    import rclpy
    nodes=[]
    def make(name):n=Node();nodes.append(n);return n
    monkeypatch.setattr(rclpy,'create_node',make,raising=False)
    p=HeadFollowPlugin({},Mock())
    args={'action':'start','input_topic':'/limx_tron2/camera/objects','input_topics':['/limx_tron2/camera/objects'],'control_interface':descriptor()}
    assert not p.dispatch('headfollow',args)['active']
    assert len(nodes)==1 and len(nodes[0].subs)==2
    assert all(s[2].reliability=='best_effort' and s[2].depth==1 for s in nodes[0].subs)
    old=nodes[0].subs[0][1];assert not p.dispatch('headfollow',args)['active'];assert len(nodes)==1
    p.dispatch('headfollow',{'action':'stop'});p.dispatch('headfollow',args)
    old(SimpleNamespace(data=json.dumps({'timestamp':123,'count':0,'objects':[]})))
    assert p._detections is None
    assert nodes[0].pub.publish.call_count==nodes[1].pub.publish.call_count==0
    p.stop()

def test_vop_older_payload_without_count_is_supported_but_error_is_not_absence():
    f={'timestamp':1000.,'objects':[]}
    assert observation(f,1000.)['people']==[]
    with pytest.raises(ValueError):observation({**f,'error':'camera stopped'},1000.)
    with pytest.raises(ValueError):observation({**f,'available':False},1000.)


def test_bundle_registers_generic_card_without_robot_or_node_side_effects():
    import ast
    source=Path(__file__).parents[1]/'main.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActuCoreBundle')
    ns={'log':Mock()}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
    b=ns['ActuCoreBundle']({'plugins':{'headfollow':{'enabled':True}}},None)
    assert [t['name'] for t in b.get_all_tools()]==['headfollow']
    assert not b.dispatch('headfollow',{'action':'info'})['active']
    assert not b._plugins[0]._running


def test_malformed_ack_or_descriptor_is_rejected_without_timer_exception():
    w=World();w.points=[(.8,0.)];w.enable();w.run();s=w.state();s['request']=['invalid']
    assert w.p.step(w.det(),s) is None and w.p.state=='fault'
    for field,value in [('joint_names',['yaw','pitch','yaw']),('head_tracking',[]),('units',[])]:
        d=descriptor();d[field]=value
        with pytest.raises(ValueError):negotiate(d)

def test_rotated_camera_axes_are_declared_instead_of_assumed():
    d=descriptor();d['head_tracking']['axes']['yaw']['image_axis']=1;d['head_tracking']['axes']['pitch']['image_axis']=0
    w=World(d);w.points=[(0.,.8)];w.enable();m=w.run()[0]
    assert m['values'][0]==0. and m['values'][1]<0

@pytest.mark.parametrize('gain',[0,float('nan')])
def test_missing_or_invalid_camera_gain_is_rejected(gain):
    d=descriptor();d['head_tracking']['axes']['yaw']['gain_rad']=gain
    with pytest.raises(ValueError):Policy(d)
