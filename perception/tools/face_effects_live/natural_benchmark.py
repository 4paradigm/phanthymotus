"""Run the frozen baseline and current engines on identical natural frames."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import psutil


def load_package(path, name):
    # The generic Perception ORT worker is shared platform infrastructure. Frozen
    # engines can still be loaded by alias; the host is imported from this checkout.
    perception_root = str(Path(__file__).resolve().parents[2])
    if perception_root not in sys.path: sys.path.insert(0, perception_root)
    # The spawn host imports this canonical module by name. Refuse a frozen
    # decoder from a different revision instead of mixing implementations.
    frozen_decoder = Path(path)/'osf_decode.py'
    host_decoder = Path(perception_root)/'plugins/face_effects_live/osf_decode.py'
    if frozen_decoder.exists() and frozen_decoder.read_bytes() != host_decoder.read_bytes():
        raise ValueError('frozen OSF decoder differs from worker checkout; use matching source revisions')
    spec=importlib.util.spec_from_file_location(name,Path(path)/'__init__.py',
                                               submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return importlib.import_module(name+'.engine'),importlib.import_module(name+'.contracts')


def stats(values):
    return dict(zip(('mean','p50','p95','p99'),[float(np.mean(values)),*np.percentile(values,[50,95,99]).tolist()])) if values else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',required=True,help='round2 directory')
    parser.add_argument('--package',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--model',choices=['osf0','osf3','yunet','mediapipe','hybrid'],default='osf0')
    parser.add_argument('--clips',nargs='+')
    parser.add_argument('--detect-side',type=int,default=640)
    parser.add_argument('--detect-upscale',action='store_true')
    parser.add_argument('--extra-config',default='{}',help='explicit algorithm configuration, saved verbatim')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    root=Path(args.evidence).resolve();models=root.parent/'models'
    engine_module,contracts=load_package(Path(args.package).resolve(),'evaluated_face_effects')
    cfg={'detector_model':str(models/'yunet.onnx'),'renderer':'cpu','max_faces':8,
         'detect_side':args.detect_side,'max_age_ms':2000}
    if args.detect_upscale: cfg['detect_upscale']=True
    if args.model in ('osf0','osf3','hybrid'):cfg['landmark_model']=str(models/(('osf0' if args.model == 'hybrid' else args.model)+'.onnx'))
    if args.model=='mediapipe':cfg.update(detector_backend='mediapipe',face_landmarker_model=str(models/'face_landmarker.task'))
    if args.model=='hybrid':cfg.update(landmarker_backend='hybrid',face_landmarker_model=str(models/'face_landmarker.task'))
    cfg.update(json.loads(args.extra_config))
    manifest=json.loads((root/'natural/manifest.json').read_text())
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    process=psutil.Process();summary={'name':args.name,'config':cfg,'clips':{},
        'source_sha256':{str(p.relative_to(args.package)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(Path(args.package).rglob('*.py'))},
        'environment':{'opencv':cv2.__version__,'numpy':np.__version__,'python':sys.version,'opencv_threads':2}}
    for clip in manifest['clips']:
        if args.clips and clip['id'] not in args.clips:
            continue
        engine=engine_module.Engine(cfg)
        observed={}
        actual_detect=engine.detector.detect
        def measured_detect(image):
            faces=actual_detect(image)
            observed['faces']=faces
            return faces
        engine.detector.detect=measured_detect
        if engine.landmarker is not None:
            actual_refine=engine.landmarker.refine
            def measured_refine(image,faces):
                faces=actual_refine(image,faces)
                observed['faces']=faces
                return faces
            engine.landmarker.refine=measured_refine
        cap=cv2.VideoCapture(str(root/'natural'/clip['file']))
        stream=(output/(clip['id']+'.jsonl')).open('x')
        rows=[];timings=[];stage={};rss=[];cpu0=sum(process.cpu_times()[:2]);begin=time.monotonic()
        try:
            for k in range(clip['frames']):
                ok,image=cap.read()
                if not ok:raise RuntimeError('incomplete benchmark clip')
                frame=contracts.Frame(image,k/clip['fps'],k)
                result=engine.process(frame)
                row={'frame_id':k,'timestamp':k/clip['fps'],'missing':result is None}
                row['detections']=[{'box':f.box.tolist(),'points':f.points.tolist(),
                                     'topology':getattr(f,'topology','osf66')} for f in observed.get('faces',[])]
                if result:
                    row.update(result.metadata)
                    row['boxes']={t.id:t.face.box.tolist() for t in engine.tracker.tracks if t.visible}
                    row['raw_anchors']={t.id:t.face.points.tolist() for t in engine.tracker.tracks if t.visible}
                    row['glasses_eyes']={}
                    for track in engine.tracker.tracks:
                        if not track.visible:
                            continue
                        commands=engine.effects.plugins['glasses'].commands(track,frame.timestamp,row['poses'][track.id])
                        if commands:
                            # Original demo PNG eye centers, independent of implementation.
                            row['glasses_eyes'][track.id]=(np.array([[62.,44.],[178.,44.]]) @ commands[0].matrix[:,:2].T + commands[0].matrix[:,2]).tolist()
                    timings.append(result.metadata['processing_ms'])
                    for key,value in result.metadata['stages'].items():stage.setdefault(key,[]).append(value)
                    if k==clip['frames']//2:cv2.imwrite(str(output/(clip['id']+'.jpg')),result.image)
                rows.append(row);rss.append(process.memory_info().rss/1048576)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if k%1000==0:
                    stream.flush();print(args.name,clip['id'],'frames',k+1,'/',clip['frames'],flush=True)
            elapsed=time.monotonic()-begin
            summary['clips'][clip['id']]={'input_sha256':clip['sha256'],'frames':len(rows),
                'processing_ms':stats(timings),'stages_ms':{k:stats(v) for k,v in stage.items()},
                'cpu_core_percent':(sum(process.cpu_times()[:2])-cpu0)/elapsed*100,
                'rss_peak_mib':max(rss),'visible_face_frames':sum(r.get('tracked',0) for r in rows),
                'frames_with_effect_tracks':sum(r.get('tracked',0)>0 for r in rows)}
            print(args.name,clip['id'],summary['clips'][clip['id']]['processing_ms'],flush=True)
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        finally:stream.close();cap.release();engine.close()
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
