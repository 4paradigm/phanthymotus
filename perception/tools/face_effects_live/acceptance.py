"""Paced 720p/30 offline soak, true source decode + inference + JPEG pipeline."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import psutil
from plugins.face_effects_live.sources.capture import FileSource
from plugins.face_effects_live.service import Service
from benchmark import make_image, summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',required=True)
    parser.add_argument('--seconds',type=int,default=60)
    args=parser.parse_args()
    root=Path(args.evidence)
    cv2.setNumThreads(2)
    photo=cv2.imread(str(root/'fixtures/grace_hopper.jpg'))
    video=root/'fixtures'/'synthetic-720p30.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
    if not writer.isOpened(): raise RuntimeError('video encoder unavailable')
    for k in range(args.seconds*30):
        writer.write(make_image(photo,'translation_roll',k%120,120))
    writer.release()
    cfg={'detector_model':str(root/'models/yunet.onnx'),'landmark_model':str(root/'models/osf0.onnx'),
         'renderer':'cpu','detect_interval':1,'max_faces':4,'max_age_ms':250}
    metadata=[]
    service=Service(cfg,FileSource(video,True),lambda jpeg,meta: metadata.append(meta))
    process=psutil.Process(); cpu0=sum(process.cpu_times()[:2]);rss=[]
    begin=time.monotonic()
    try:
        service.start()
        while service.state in ('starting','running'):
            rss.append((time.monotonic()-begin,process.memory_info().rss/1024**2))
            time.sleep(.1)
        before_stop=service.status()
    finally: after_stop=service.stop()
    elapsed=time.monotonic()-begin
    report={'input_kind':'public_photo_synthetic_motion_encoded_mp4', 'dimensions':[1280,720],
            'source_fps':30,'source_frames':args.seconds*30,'nominal_seconds':args.seconds,
            'wall_seconds':elapsed, 'pipeline_output_fps':len(metadata)/args.seconds,
            'cpu_core_percent':(sum(process.cpu_times()[:2])-cpu0)/elapsed*100,
            'rss_start_mib':rss[0][1] if rss else None,'rss_end_mib':rss[-1][1] if rss else None,
            'rss_peak_mib':max(x[1] for x in rss) if rss else None,'rss_samples':rss,
            'receive_to_render_ms':summary([m['receive_to_render_ms'] for m in metadata]),
            'receive_to_encode_ms':summary([m['receive_to_encode_ms'] for m in metadata]),
            'processing_ms':summary([m['processing_ms'] for m in metadata]),
            'stages_ms':{key:summary([m['stages'][key] for m in metadata]) for key in metadata[0]['stages']} if metadata else {},
            'before_stop':before_stop,'after_stop':after_stop,
            'workers_stopped':not service.capture.is_alive() and not service.processor.is_alive(),
            'gpu_percent':None,'sensor_to_display_ms':None,
            'limits':['short 60-second soak, not hours of stability validation',
                      'file playback, not real camera or natural head motion',
                      'timing starts after decoder returns; no browser presentation measurement']}
    (root/'paced-acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'local-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rss_samples','before_stop','after_stop','stages_ms')},indent=2))


if __name__=='__main__':main()
