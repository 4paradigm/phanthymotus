"""Paced real-model Service replay with bounded queues, PSS and whole-run tails.

Recorded input is a load source, never a physical camera or independent motion
truth. Renderer instrumentation is identical on both compared versions.
"""
import argparse
from array import array
from collections import deque
import importlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import psutil

from load_acceptance import LoopSource
from natural_benchmark import load_package,stats
from dataset_adapters import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['package','video','models','output']:p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--seconds',type=float,default=600);p.add_argument('--fps',type=float,default=30)
    p.add_argument('--extra-config',default='{}');a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior experiment')
    a.output.parent.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    engine_module,contracts=load_package(a.package.resolve(),'soak_card')
    module=importlib.import_module('soak_card.service')
    measured={'changed':False,'source_black':False,'commands':0}
    actual_factory=engine_module.create_renderer
    class Renderer:
        def __init__(self,real):self.real=real;self.name=real.name
        def render(self,image,commands):
            output=self.real.render(image,commands)
            measured.update(commands=len(commands),source_black=cv2.norm(image,cv2.NORM_INF)<=1,
                            changed=bool(commands) and cv2.norm(image,output,cv2.NORM_INF)>0)
            return output
        def close(self):self.real.close()
    def create_renderer(*args,**kwargs):
        renderer,fallback=actual_factory(*args,**kwargs);return Renderer(renderer),fallback
    engine_module.create_renderer=create_renderer
    config={'detector_model':str(a.models/'yunet.onnx'),'landmark_model':str(a.models/'osf0.onnx'),
            'max_faces':8,'renderer':'cpu','max_age_ms':250,**json.loads(a.extra_config)}
    latencies=array('d');tail=deque(maxlen=1200);diagnostics=[];phase_times={}
    counters={'callbacks':0,'effect_changed_frames':0,'unchanged_frames':0,'black_source_frames':0}
    def result(jpeg,metadata):
        value=metadata['receive_to_encode_ms'];latencies.append(value);tail.append(value)
        counters['callbacks']+=1;counters['black_source_frames']+=measured['source_black']
        counters['effect_changed_frames']+=measured['changed'] and not measured['source_black']
        counters['unchanged_frames']+=not measured['changed']
        for key,value in {**metadata.get('stages',{}),'jpeg_ms':metadata['jpeg_ms'],
                          'processing_ms':metadata['processing_ms']}.items():
            phase_times.setdefault(key,array('d')).append(value)
        if len(diagnostics)<1200 and metadata.get('inference_diagnostics'):
            diagnostics.append(metadata['inference_diagnostics'])
    source=LoopSource(a.video,a.fps,a.seconds,contracts);service=module.Service(config,source,result)
    parent=psutil.Process();cpu_initial=sum(parent.cpu_times()[:2]);last_cpu={};samples=[]
    start=time.monotonic();next_progress=0.;pending_peak=0
    def sample(phase='running'):
        processes=[parent,*parent.children(recursive=True)];rss=pss=0.;pss_available=True;threads=0
        for process in processes:
            try:
                rss+=process.memory_info().rss;threads+=process.num_threads()
                last_cpu[process.pid]=sum(process.cpu_times()[:2])
                try:pss+=process.memory_full_info().pss
                except (AttributeError,psutil.AccessDenied):pss_available=False
            except psutil.NoSuchProcess:pass
        row={'phase':phase,'elapsed_s':time.monotonic()-start,'parent_rss_mib':parent.memory_info().rss/2**20,
             'tree_rss_mib':rss/2**20,'tree_pss_mib':pss/2**20 if pss_available else None,
             'threads':threads,'pending':int(service.pending is not None)}
        samples.append(row);return row
    status={};error=None
    try:
        service.start()
        while service.state in ('starting','running'):
            row=sample();pending_peak=max(pending_peak,row['pending'])
            if row['elapsed_s']>=next_progress:
                print(json.dumps({'progress':row,'input':source.count,'output':service.processed}),flush=True)
                next_progress+=30
            if row['elapsed_s']>a.seconds+25:raise RuntimeError('soak deadline exceeded')
            time.sleep(1)
        status=service.status()
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        stopped=service.stop();source.finish();sample('stopped')
    elapsed=time.monotonic()-start
    settled=[r for r in samples if r['phase']=='running' and r['elapsed_s']>=min(60,a.seconds/3)]
    slopes={}
    for key in ['parent_rss_mib','tree_rss_mib','tree_pss_mib']:
        values=[r for r in settled if r[key] is not None]
        slopes[key]=float(np.polyfit([r['elapsed_s'] for r in values],[r[key] for r in values],1)[0]*60) if len(values)>2 else None
    from plugins.ort_worker import get_worker
    worker=get_worker();remaining={'sessions':len(worker._loaded),'services':len(worker._services)}
    worker.close()
    report={'schema':'face-effects-soak/2','package':str(a.package.resolve()),'config':config,
        'input_sha256':sha256(a.video),'model_sha256':{k:sha256(v) for k,v in config.items() if k.endswith('_model')},
        'source_sha256':{str(f.relative_to(a.package)):sha256(f) for f in a.package.rglob('*.py')},
        'seconds':a.seconds,'input_fps':a.fps,'elapsed_s':elapsed,'input_frames':source.count,
        'output_frames':service.processed,'output_fps':service.processed/a.seconds,**counters,
        'receive_to_jpeg_all_ms':stats(list(latencies)),'receive_to_jpeg_last_1200_ms':stats(list(tail)),
        'output_frame_stages_ms':{key:stats(list(values)) for key,values in phase_times.items()},
        'harness_sha256':sha256(Path(__file__)),
        'cpu_parent_core_percent':(last_cpu.get(parent.pid,cpu_initial)-cpu_initial)/elapsed*100,
        'cpu_parent_and_workers_core_percent':(sum(last_cpu.values())-cpu_initial)/elapsed*100,
        'rss_peak_mib':max(r['tree_rss_mib'] for r in samples),
        'pss_peak_mib':max((r['tree_pss_mib'] for r in samples if r['tree_pss_mib'] is not None),default=None),
        'threads_peak':max(r['threads'] for r in samples),'threads_end_before_worker_close':samples[-1]['threads'],
        'settled_slopes_mib_per_minute':slopes,'memory_samples':samples,'pending_peak':pending_peak,
        'pending_at_end':int(service.pending is not None),'production_history_size':len(service.samples),
        'status':{k:v for k,v in status.items() if k!='result'},'error':error,
        'service_threads_stopped':not any(t and t.is_alive() for t in [service.processor,service.capture]),
        'worker_sessions_remaining':remaining,'own_worker_stopped':not worker.alive,
        'inference_diagnostic_samples':diagnostics,
        'limits':['recorded video replay; no physical camera/exposure/network/display latency',
                  'changed effect pixels are rendering diagnostics, not ground-truth face accuracy',
                  'same renderer norm instrumentation on both versions; small profiling overhead included',
                  'finite 600s-type run, not proof of hours-long leak freedom; RSS includes shared-page duplication']}
    report['passed']=not error and status.get('state')=='eof' and report['service_threads_stopped'] and report['own_worker_stopped'] and not any(remaining.values()) and report['pending_at_end']==0
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['memory_samples','inference_diagnostic_samples','source_sha256']}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':sys.exit(main())
