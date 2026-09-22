#!/usr/bin/env python3
"""Offline full-runtime pose replay with synthetic feedback and parallel FK.

Usage: python replay_tianyi_runtime.py PLUGINS_DIR PROFILE RECORDING [CYCLES] [feedback-gap]
No ROS, Driver, SDK, or network; input is sampled poses rescheduled at 72 Hz.
Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 before starting Python, as deployed.
"""
import sys,json,time,copy,threading,hashlib
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
from teleop.tianyi import TianyiIntentAdapter
from teleop.runtime import TeleopRuntime
from teleop.protocol import bind_rtc_frame_v1
from teleop.tianyi_visualization import snapshot
source=Path(sys.argv[3]);samples=[];seen=set()
for line in source.read_text().splitlines():
 row=json.loads(line);f=(row['card'].get('pose') or {}).get('latest')
 if not f or f['sequence'] in seen:continue
 seen.add(f['sequence']);samples.append((f,row['driver']['feedback']['q'],row['driver']['monotonic_ns']-row['driver']['feedback']['arm_ns']))
current=[samples[0][1],samples[0][2]]
inject_gap=len(sys.argv)>5 and sys.argv[5]=="feedback-gap"
link=SimpleNamespace(lease=None,feedback=lambda:{'feedback':{'q':current[0],'arm_ns':time.monotonic_ns()-current[1]}})
a=TianyiIntentAdapter(link,'shadow');a.calibrate(Path(sys.argv[2]))
r=TeleopRuntime(mode='shadow',adapter=a,pose_timeout_ms=100,motion_interval_ms=20)
r.prepare_local_session();binding,_=r.rtc_authority_snapshot()
done=threading.Event();visual=[]
def display():
 while not done.is_set():
  start=time.monotonic();v=snapshot(a);visual.append((start,v['available'],bool(v.get('ik') or v.get('held_ik'))));done.wait(max(0,1/30-(time.monotonic()-start)))
th=threading.Thread(target=display);th.start()
seq=0;clutch=0;held=False;fault=None
try:
 for cycle in range(int(sys.argv[4]) if len(sys.argv)>4 else 12):
  for index,(original,q,age_ns) in enumerate(samples):
   f=copy.deepcopy(original);now=time.monotonic();current[0]=q
   current[1]=200_000_000 if inject_gap and original["deadman"] and 25<=index<36 else age_ns
   for key in ('boot_id','session_id','epoch'):f.pop(key,None)
   if f['deadman'] and not held:clutch+=1
   held=f['deadman'];f.update(sequence=seq,clutch_sequence=clutch,client_monotonic_ns=time.monotonic_ns())
   r.heartbeat(binding)
   r.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='shadow'),source='offline',include_status=False)
   seq+=1
   time.sleep(max(0,1/72-(time.monotonic()-now)))
  if r.status()['state']=='fault':break
except Exception as e:fault=str(e)
finally:
 state=r.status();done.set();th.join();r.close()
print(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'recorded poses resampled 72Hz; recorded feedback age reconstructed; optional 200ms gap injection; no ROS/network/hardware','injected_feedback_gap':inject_gap,'input_frames':seq,'display_frames':len(visual),'blank_display_frames':sum(not x[2] for x in visual),'exception':fault,'state':state['state'],'reason':state['reason'],'counters':state['counters'],'dispatch':{k:state['dispatch'].get(k) for k in ['state','fault_code','last_io_stall','counters']}},indent=2))

if fault is not None or state["state"] == "fault":
    raise SystemExit(1)
