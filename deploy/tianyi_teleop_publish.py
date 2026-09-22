#!/usr/bin/python3 -I
"""Root-owned fixed two-image switch. Never imports or executes staged code."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import stat
import subprocess
import tempfile
import time
import urllib.request
import yaml

COMPOSE=Path('/opt/phanthy-motus/docker-compose.yml')
CONFIG=Path('/data/hanzebei/pico-tianyi/preparation/20260920T211624/actucore.shadow.yaml')
STATE=Path('/var/lib/phanthy-teleop-publish')
SERVICES={'actucore-teleop':'embodied-actucore-teleop','tianyi2':'embodied-x-humanoid-tianyi2.0'}
OTHERS=('phanthy-motus-agent-core-1','embodied-perception')
ENV={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','HOME':'/root','LANG':'C.UTF-8'}


def shell(*args):return subprocess.check_output(args,text=True,timeout=45,env=ENV).strip()


def get(url, data=None, headers=None):
    request=urllib.request.Request(url,None if data is None else json.dumps(data).encode(),headers or {'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    with opener.open(request,timeout=5) as f:return json.load(f)


def core(path):
    if path not in ('/config/project-running','/canvas/edit-status?session_id=teleop-publish'):raise ValueError('core_read_only')
    key=next(x.split('=',1)[1].strip() for x in Path('/opt/phanthy-motus/.env').read_text().splitlines() if x.startswith('ACCESS_TOKEN='))
    return get('https://127.0.0.1:15678/api'+path,headers={'Authorization':'Bearer '+key})


def info(port,name):
    r=get(f'http://127.0.0.1:{port}/mcp',{'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':name,'arguments':{'action':'info'}}})
    if r.get('error') or r.get('result',{}).get('isError'):raise ValueError('info_failed')
    v=json.loads(r['result']['content'][0]['text'])
    if v.get('error'):raise ValueError('info_failed')
    return v


def guard():
    e=core('/canvas/edit-status?session_id=teleop-publish')
    if e.get('code')!=200 or e.get('editor') is not None or core('/config/project-running').get('running') is not False:raise ValueError('canvas_occupied')
    d=info(15707,'teleop_executor');v=info(15740,'teleop')
    if d.get('state')!='idle' or d.get('ownership_held') is not False or d.get('output_active') is not False:raise ValueError('driver_not_idle')
    if d.get('foreign_publishers') or d.get('hands_enabled') is not False:raise ValueError('driver_conflict')
    if v.get('authority_valid') is not False or v.get('state') not in ('idle','fault'):raise ValueError('card_not_idle')
    dispatch=v.get('dispatch') or {}
    if dispatch.get('io_inflight') or dispatch.get('mailbox_depth') or dispatch.get('stop_queue_depth'):raise ValueError('dispatch_pending')
    f=d['feedback'];now=d['monotonic_ns']
    if any(not 0<=now-f.get(k,0)<=100_000_000 for k in ('arm_ns','power_ns','fixed_ns')):raise ValueError('feedback_stale')
    if f.get('power_on') is not True or f.get('estop') is not False or f.get('fault') is not False:raise ValueError('robot_not_ready')
    dq=f.get('dq',[])
    if len(dq)!=14 or any(not isinstance(x,(int,float)) or not abs(x)<=.02 for x in dq):raise ValueError('arms_not_stationary')
    if info(15707,'arm').get('state') not in ('idle','ready') or info(15707,'servo').get('state')!='idle':raise ValueError('legacy_busy')
    if yaml.safe_load(CONFIG.read_text())['plugins']['teleop']['mode']!='shadow':raise ValueError('startup_config_not_shadow')


def replace_images(text,images):
    parsed=yaml.safe_load(text);expected=json.loads(json.dumps(parsed));root=yaml.compose(text);edits=[]
    for service,digest in images.items():
        if service not in SERVICES or not re.fullmatch(r'sha256:[0-9a-f]{64}',digest):raise ValueError('image_argument')
        node=root
        for key in ('services',service,'image'):
            if not isinstance(node,yaml.MappingNode):raise ValueError('compose_shape')
            found=[v for k,v in node.value if k.value==key]
            if len(found)!=1:raise ValueError('compose_duplicate_or_missing')
            node=found[0]
        if not isinstance(node,yaml.ScalarNode):raise ValueError('compose_image_shape')
        edits.append((node.start_mark.index,node.end_mark.index,json.dumps(digest)))
        expected['services'][service]['image']=digest
    for a,b,x in sorted(edits,reverse=True):text=text[:a]+x+text[b:]
    if yaml.safe_load(text)!=expected:raise ValueError('compose_scope')
    return text


def protected(path):
    for p in (path,*path.parents):
        s=p.lstat()
        if stat.S_ISLNK(s.st_mode) or s.st_uid!=0 or s.st_mode&0o022:raise ValueError('untrusted_root_path:'+str(p))


def atomic(path,text):
    fd,name=tempfile.mkstemp(prefix='.teleop-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.chmod(name,0o644);os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def containers(names):return shell('/usr/bin/docker','inspect','--format','{{.Id}} {{.Image}} {{.State.StartedAt}}',*names)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check','apply','rollback']);p.add_argument('images',nargs='*');a=p.parse_args()
    if os.geteuid()!=0:raise ValueError('installed_sudo_entry_required')
    protected(Path(__file__).resolve());protected(STATE);protected(COMPOSE)
    with (STATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard();before=COMPOSE.read_text();current=yaml.safe_load(before)
        baseline=STATE/'baseline.json'
        spec=json.loads(json.dumps(current))
        for key in SERVICES:spec['services'][key].pop('image',None)
        if not baseline.exists() or json.loads(baseline.read_text())!=spec:raise ValueError('non_image_config_changed_requires_review')
        if a.action=='rollback':
            if a.images:raise ValueError('rollback_takes_no_arguments')
            images=json.loads((STATE/'rollback.json').read_text())
        else:
            if len(a.images)!=2:raise ValueError('two_image_digests_required')
            images=dict(zip(SERVICES,a.images))
        after=replace_images(before,images)
        for service,digest in images.items():
            image=json.loads(shell('/usr/bin/docker','image','inspect',digest))[0]
            if image['Id']!=digest:raise ValueError('digest_changed')
            if a.action!='rollback' and image.get('Config',{}).get('Labels',{}).get('org.phanthy.teleop.service')!=service:raise ValueError('candidate_service_label')
        if a.action=='check':print('PUBLISH CHECK PASS');return
        others=containers(OTHERS)
        previous={key:shell('/usr/bin/docker','inspect','--format','{{.Image}}',name) for key,name in SERVICES.items()}
        stamp=str(time.time_ns());(STATE/('compose-'+stamp+'.yml')).write_text(before)
        (STATE/'rollback.json').write_text(json.dumps(previous))
        guard()
        if COMPOSE.read_text()!=before:raise ValueError('compose_changed')
        shell('/usr/bin/docker','stop',SERVICES['actucore-teleop'])
        d=info(15707,'teleop_executor')
        if d['state']!='idle' or d['ownership_held'] or d['output_active']:raise ValueError('lease_changed')
        if COMPOSE.read_text()!=before:raise ValueError('compose_changed')
        atomic(COMPOSE,after)
        for service in ('tianyi2','actucore-teleop'):
            shell('/usr/bin/docker','compose','-p','phanthy-motus','-f',str(COMPOSE),'up','-d','--no-deps','--pull','never',service)
        deadline=time.monotonic()+45;last=None
        while time.monotonic()<deadline:
            try:
                guard();v=info(15740,'teleop')
                if v['mode']!='shadow' or v['state']!='idle':raise ValueError('shadow_not_idle')
                for service,name in SERVICES.items():
                    if shell('/usr/bin/docker','inspect','--format','{{.Image}}',name)!=images[service]:raise ValueError('running_image')
                if containers(OTHERS)!=others or COMPOSE.read_text()!=after:raise ValueError('other_service_changed')
                (STATE/'rollback.json').write_text(json.dumps(previous))
                (STATE/('receipt-'+stamp+'.json')).write_text(json.dumps({'images':images,'other_services_unchanged':True,'mode':'shadow'}))
                print('SHADOW SWITCH PASS: two images only; Agent Core unchanged');return
            except Exception as exc:last=exc;time.sleep(1)
        raise RuntimeError('switch_unconfirmed_no_automatic_rollback:'+str(last))

if __name__=='__main__':main()
