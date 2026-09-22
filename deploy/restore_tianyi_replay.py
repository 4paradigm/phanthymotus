#!/usr/bin/python3 -I
"""Operator-only recovery after another deployment replaced the teleop Driver."""
import argparse, copy, hashlib, importlib.machinery, importlib.util, json, os, time
from pathlib import Path
import yaml
EXPECTED='e57b2dbaab5e4ef2662a84ff645aceafe7cc8a65e36fda4b5301e9de445a9c9c'
IMAGES={'actucore-teleop':'sha256:1073e12013199cc44687ae1bfcfa7831cbf35cd381b48782ec2ff5b0ccf160e5',
        'tianyi2':'sha256:4fbcd250a39f44c65335bb75fff56b6cc9ee0bd863d7c82bb6077c4dd8d571ee'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
    path='/usr/local/sbin/tianyi-teleop-publish'
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()=='f53d201d25480f53d25ac81444864761adf5aaea7e4ad8b1635aa9ce8bb21e75'
    loader=importlib.machinery.SourceFileLoader('publisher',path)
    spec=importlib.util.spec_from_loader(loader.name,loader);h=importlib.util.module_from_spec(spec);loader.exec_module(h)
    before=h.COMPOSE.read_text();assert hashlib.sha256(before.encode()).hexdigest()==EXPECTED,'Compose changed; inspect first'
    def idle():
        assert h.core('/canvas/edit-status?session_id=teleop-publish').get('editor') is None,'Canvas editor'
        assert h.core('/config/project-running').get('running') is False,'Canvas running'
        for name in ('arm','servo'):assert h.info(15707,name).get('state')=='idle',name+' busy'
        v=h.info(15740,'teleop');assert v.get('mode')=='shadow' and v.get('state')=='idle' and v.get('authority_valid') is False,'card busy'
        tools=h.get('http://127.0.0.1:15707/mcp',{'jsonrpc':'2.0','id':1,'method':'tools/list'})
        assert not tools.get('error') and 'tools' in tools.get('result',{}),'tools unavailable'
        assert not any(t['name']=='teleop_executor' for t in tools['result']['tools']),'executor exists; use normal publisher'
    idle()
    assert yaml.safe_load(h.CONFIG.read_text())['plugins']['teleop']['mode']=='shadow'
    for name,digest in IMAGES.items():
        image=json.loads(h.shell('/usr/bin/docker','image','inspect',digest))[0]
        assert image['Id']==digest and image['Config']['Labels']['org.phanthy.teleop.service']==name
    print('RUNTIME PREFLIGHT PASS: idle, Shadow, candidate images present; root baseline pending',flush=True)
    if not args.apply:return
    assert os.geteuid()==0,'Operator sudo required for baseline review and Compose'
    h.protected(h.COMPOSE);h.protected(h.STATE)
    import fcntl
    with (h.STATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        h.protected(h.STATE/'baseline.json')
        old=json.loads((h.STATE/'baseline.json').read_text());current=yaml.safe_load(before)
        restored=copy.deepcopy(current)
        for name,digest in IMAGES.items():
            restored['services'][name]=copy.deepcopy(old['services'][name])
            restored['services'][name]['image']=digest
        # Restore only the two teleop definitions from the root-owned reviewed baseline.
        assert {k:v for k,v in restored['services'].items() if k not in IMAGES}=={k:v for k,v in current['services'].items() if k not in IMAGES}
        assert restored['services']['actucore-teleop']['container_name']==h.SERVICES['actucore-teleop']
        after=yaml.safe_dump(restored,sort_keys=False,allow_unicode=True)
        new=copy.deepcopy(restored)
        for name in IMAGES:new['services'][name].pop('image')
        others=h.containers(h.OTHERS);idle()
        assert h.COMPOSE.read_text()==before,'Compose changed'
        stamp=str(time.time_ns())
        (h.STATE/('restore-compose-'+stamp+'.yml')).write_text(before)
        (h.STATE/('restore-baseline-'+stamp+'.json')).write_text(json.dumps(old))
        previous={k:h.shell('/usr/bin/docker','inspect','--format','{{.Image}}',n) for k,n in h.SERVICES.items()}
        h.shell('/usr/bin/docker','stop',h.SERVICES['actucore-teleop'])
        assert h.COMPOSE.read_text()==before,'Compose changed'
        h.atomic(h.COMPOSE,after)
        for name in ('tianyi2','actucore-teleop'):
            h.shell('/usr/bin/docker','compose','-p','phanthy-motus','-f',str(h.COMPOSE),'up','-d','--no-deps','--pull','never',name)
        deadline=time.monotonic()+45;last=None
        while time.monotonic()<deadline:
            try:
                h.guard()
                assert h.info(15740,'teleop')['state']=='idle'
                assert h.containers(h.OTHERS)==others,'Other service changed'
                assert h.COMPOSE.read_text()==after
                for k,n in h.SERVICES.items():assert h.shell('/usr/bin/docker','inspect','--format','{{.Image}}',n)==IMAGES[k]
                (h.STATE/'baseline.json').write_text(json.dumps(new,indent=2)+'\n')
                (h.STATE/'rollback.json').write_text(json.dumps(previous))
                print('RESTORE SHADOW PASS: Core preserved; no lease or motion');return
            except Exception as exc:last=exc;time.sleep(1)
        raise RuntimeError('Restore unconfirmed; no automatic rollback: '+str(last))
if __name__=='__main__':main()
