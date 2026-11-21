"""Same-machine, same-input real inference; public still + synthetic motion.

Runs candidates in fresh sequential processes. No natural-video quality claims.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import cv2
import numpy as np
import psutil
from plugins.face_effects_live.contracts import Frame
from plugins.face_effects_live.engine import Engine
from plugins.face_effects_live.filters import OneEuro


SCENES = ('frontal', 'translation_roll', 'two_faces', 'occlusion', 'low_light', 'blur')


def make_image(photo, scene, k, total):
    image=np.full((720,1280,3),36,np.uint8)
    portrait=cv2.resize(photo,(410,480))
    if scene=='two_faces':
        image[120:600,100:510]=portrait
        image[120:600,770:1180]=portrait[:,::-1]
    else:
        image[120:600,435:845]=portrait
    if scene=='translation_roll':
        angle=20*np.sin(k/total*np.pi*4)
        matrix=cv2.getRotationMatrix2D((640,360),angle,1.)
        matrix[:,2]+=[80*np.sin(k/total*np.pi*2),35*np.cos(k/total*np.pi*2)]
        image=cv2.warpAffine(image,matrix,(1280,720))
    if scene=='occlusion' and total*.3<=k<total*.55:
        cv2.rectangle(image,(545,220),(690,395),(36,36,36),-1)
    if scene=='low_light':
        image=np.clip(image.astype(np.float32)*.3,0,255).astype(np.uint8)
    if scene=='blur':
        kernel=np.zeros((19,19),np.float32);kernel[9,:]=1/19
        image=cv2.filter2D(image,-1,kernel)
    return image


def summary(values):
    values=np.asarray(values,dtype=float)
    return {'mean':float(values.mean()),'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95)),'p99':float(np.percentile(values,99)),
            'max':float(values.max())} if len(values) else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    cv2.setNumThreads(2)  # benchmark process only, never change shared Card globals
    root=Path(args.evidence)
    photo=cv2.imread(str(root/'fixtures/grace_hopper.jpg'))
    cfg={'detector_model':str(root/'models/yunet.onnx'),'renderer':'cpu',
         'max_faces':4,'detect_side':640,'detect_interval':args.interval,'max_age_ms':2000}
    if args.worker!='A_yunet': cfg['landmark_model']=str(root/('models/osf3.onnx' if args.worker=='B_osf3' else 'models/osf0.onnx'))
    process=psutil.Process()
    init=time.perf_counter(); engine=Engine(cfg); init_ms=(time.perf_counter()-init)*1000
    records={}; outputs=Path(args.output); outputs.mkdir(parents=True,exist_ok=True)
    try:
        # Warmup not included; no stale timestamps reused in measured frames.
        for k in range(10): engine.process(Frame(make_image(photo,'frontal',k,10),k/30,k))
        frame_id=10
        for scene in SCENES:
            stages={}; times=[]; latency=[]; counts=[]; ids=[]; rss=[]; anchors=[]; jpeg_times=[]
            digest=hashlib.sha256()
            start=time.perf_counter(); cpu_start=sum(process.cpu_times()[:2])
            for k in range(args.frames):
                image=make_image(photo,scene,k,args.frames)
                digest.update(image.tobytes())
                result=engine.process(Frame(image,frame_id/30,frame_id))
                frame_id+=1
                rss.append(process.memory_info().rss/1024**2)
                if result is None: continue
                meta=result.metadata
                times.append(meta['processing_ms']);latency.append(meta['receive_to_render_ms'])
                counts.append(meta['tracked']);ids.append(meta['ids']);anchors.append(meta['anchors'])
                for key,value in meta['stages'].items(): stages.setdefault(key,[]).append(value)
                t=time.perf_counter();cv2.imencode('.jpg',result.image,[cv2.IMWRITE_JPEG_QUALITY,85]);jpeg_times.append((time.perf_counter()-t)*1000)
                if k==args.frames-1:
                    cv2.imwrite(str(outputs/(args.worker+'-'+scene+'.jpg')),result.image)
            elapsed=time.perf_counter()-start; cpu_seconds=sum(process.cpu_times()[:2])-cpu_start
            records[scene]={'input_kind':'public_real_photo_with_synthetic_transform', 'input_sha256':digest.hexdigest(),
                            'input_frames':args.frames,'processed_frames':len(times),'source_fps_assumption':30,
                            'processing_ms':summary(times),'receive_to_render_ms':summary(latency),
                            'jpeg_ms':summary(jpeg_times),'stages_ms':{k:summary(v) for k,v in stages.items()},
                            'loop_throughput_fps_including_fixture_hashing':len(times)/elapsed,
                            'processing_capacity_fps':1000/np.mean(times) if times else None,
                            'cpu_core_percent':cpu_seconds/elapsed*100, 'rss_peak_mib':max(rss),
                            'gpu_percent':None,'gpu_reason':'no hardware GPU measurement; NVML blocked',
                            'frames_with_effect_tracks':sum(c>0 for c in counts), 'tracked_face_counts':counts,
                            'track_ids':ids,'anchor_accuracy_px':None,'true_id_switches':None,
                            'quality_note':'no annotated natural head motion/identity ground truth'}
        return {'candidate':args.worker,'config':cfg,'init_ms':init_ms,'scenes':records,
                'environment':{'platform':platform.platform(),'python':platform.python_version(),'opencv':cv2.__version__,
                               'numpy':np.__version__,'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
                               'opencv_threads':cv2.getNumThreads()},
                'model_sha256':{k:sha(v) for k,v in cfg.items() if k.endswith('_model')},
                'photo_sha256':sha(root/'fixtures/grace_hopper.jpg')}
    finally: engine.close()


def filter_probe():
    rng=np.random.default_rng(72030)
    signal=np.r_[np.zeros(120),np.ones(120)*.25]
    noisy=signal+rng.normal(0,.005,len(signal))
    filt=OneEuro(); smooth=np.array([filt([v],k/30)[0] for k,v in enumerate(noisy)])
    response=np.where(smooth[120:]>=.225)[0]
    return {'input_kind':'synthetic_scalar_landmark_no_model',
            'raw_stationary_std_normalized':float(noisy[20:120].std()),
            'filtered_stationary_std_normalized':float(smooth[20:120].std()),
            'step_90_percent_response_ms':float(response[0]/30*1000) if len(response) else None,
            'note':'synthetic filter tradeoff; not real video jitter or total AR latency'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--frames',type=int,default=60)
    parser.add_argument('--interval',type=int,default=1)
    parser.add_argument('--worker',choices=['A_yunet','B_osf3','C_osf0'])
    args=parser.parse_args()
    if args.worker:
        result=worker(args)
        Path(args.output,args.worker+'.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    else:
        results=[]
        for name in ['A_yunet','B_osf3','C_osf0']:
            command=[sys.executable,__file__,'--evidence',args.evidence,'--output',args.output,
                     '--frames',str(args.frames),'--interval',str(args.interval),'--worker',name]
            subprocess.run(command,check=True)
            results.append(json.loads(Path(args.output,name+'.json').read_text()))
        output={'schema':'face_effects_benchmark/1','candidates':results,'filter_probe':filter_probe(),
                'limits':['720p input, sequential CPU execution, synthetic motion from one public photo',
                          'not natural video, camera, ROS2, browser paint or hardware GPU acceptance',
                          'no annotated face accuracy, real ID switches or occlusion ground truth']}
        Path(args.output,'comparison.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
        for result in results:
            print(result['candidate'],{k:round(v['processing_ms']['p95'],2) for k,v in result['scenes'].items()})


if __name__=='__main__': main()
