"""Paced natural-video replay against the actual bounded Service.

No prediction histories or images are retained by the harness. Increasing the
input rate is a load test, not a natural motion/accuracy experiment.
"""
import argparse
from collections import deque
import importlib
import json
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np
import psutil
from natural_benchmark import load_package, stats
from dataset_adapters import sha256


class LoopSource:
    def __init__(self,path,fps,seconds,contracts):
        self.cap=cv2.VideoCapture(str(path));self.fps=fps;self.seconds=seconds
        self.contracts=contracts;self.closed=threading.Event();self.eof=False
        self.count=0;self.start=None

    def read(self):
        if self.start is None:self.start=time.monotonic()
        deadline=self.start+self.count/self.fps
        if deadline>=self.start+self.seconds:self.eof=True;return None
        if self.closed.wait(max(0,deadline-time.monotonic())):self.eof=True;return None
        ok,image=self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,image=self.cap.read()
        if not ok:raise RuntimeError('cannot replay natural clip')
        frame=self.contracts.Frame(image,self.count/self.fps,self.count,metadata={'provenance':'natural repeated load replay'})
        self.count+=1
        return frame

    def close(self):self.closed.set()
    def finish(self):self.cap.release()


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--video',required=True)
    p.add_argument('--models',required=True,type=Path);p.add_argument('--fps',type=float,default=30)
    p.add_argument('--seconds',type=float,default=300);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--extra-config',default='{}')
    args=p.parse_args();cv2.setNumThreads(2)
    _,contracts=load_package(Path(args.package).resolve(),'load_card')
    Service=importlib.import_module('load_card.service').Service
    source=LoopSource(args.video,args.fps,args.seconds,contracts)
    process=psutil.Process();rss=[];tree_rss=[];cpu0=sum(process.cpu_times()[:2]);start=time.monotonic()
    metrics=deque(maxlen=1200);last_ids=deque(maxlen=1200)
    def result(jpeg,metadata):
        metrics.append(metadata['receive_to_encode_ms']);last_ids.append(metadata['frame_id'])
    config={'detector_model':str(args.models/'yunet.onnx'),'landmark_model':str(args.models/'osf0.onnx'),
            'max_faces':8,'renderer':'cpu','max_age_ms':250,**json.loads(args.extra_config)}
    service=Service(config,source,result)
    try:
        service.start()
        while service.state in ('starting','running'):
            rss.append([time.monotonic()-start,process.memory_info().rss/1048576])
            tree_rss.append([time.monotonic()-start,sum(p.memory_info().rss for p in [process,*process.children(recursive=True)])/1048576])
            if time.monotonic()-start>args.seconds+20:raise RuntimeError('deadline exceeded')
            time.sleep(1)
        status=service.status()
    finally:stopped=service.stop()
    elapsed=time.monotonic()-start
    settled=np.array([r for r in rss if r[0]>=min(30,args.seconds/3)])
    slope=float(np.polyfit(settled[:,0],settled[:,1],1)[0]*60) if len(settled)>2 else None
    report={'schema':'face-effects-natural-load/1','package':str(Path(args.package).resolve()),
            'config':config,'input_sha256':sha256(args.video),
            'source_sha256':{str(f.relative_to(args.package)):sha256(f) for f in sorted(Path(args.package).rglob('*.py'))},
            'input_fps':args.fps,'duration_seconds':args.seconds,'elapsed_s':elapsed,
            'input_frames':source.count,'output_frames':status['processed'],
            'output_fps_per_source_duration':status['processed']/args.seconds,
            'status':{k:v for k,v in status.items() if k!='result'},
            'last_1200_receive_to_encode_ms':stats(list(metrics)),
            'last_1200_output_id_gaps':sum(b-a>1 for a,b in zip(last_ids,list(last_ids)[1:])),
            'cpu_core_percent':(sum(process.cpu_times()[:2])-cpu0)/elapsed*100,
            'cpu_parent_and_children_core_percent':(sum(sum(p.cpu_times()[:2]) for p in [process,*process.children(recursive=True)])-cpu0)/elapsed*100,
            'rss_mib_samples':rss,'rss_peak_mib':max(r[1] for r in rss),
            'rss_parent_and_children_mib_samples':tree_rss,'rss_parent_and_children_peak_mib':max(r[1] for r in tree_rss),
            'settled_rss_linear_mib_per_minute':slope,
            'pending_frames_at_end':int(service.pending is not None),
            'latency_history_size':len(service.samples),
            'workers_stopped':not any(t and t.is_alive() for t in (service.processor,service.capture)),
            'final_state':stopped['state'],
            'limits':['same natural clip repeated; not hours of varied live input',
                      'latency starts after file decode; no DDS/browser or exposure',
                      'RSS slope includes allocator/codec caches; not a proof of leak freedom']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rss_mib_samples'}),flush=True)
    return 0 if report['workers_stopped'] and status['state']=='eof' else 1


if __name__=='__main__':sys.exit(main())
