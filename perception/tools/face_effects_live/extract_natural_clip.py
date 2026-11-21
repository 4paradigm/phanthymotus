"""Lossless contiguous natural video selection; no synthesized motion or labels."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seconds',type=float,default=300);p.add_argument('--stride',type=int,default=2)
    p.add_argument('--source-url',required=True);p.add_argument('--license',required=True)
    a=p.parse_args();cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(str(a.source));fps=cap.get(cv2.CAP_PROP_FPS)
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps<=0 or a.stride<1:raise ValueError('invalid source FPS/stride')
    count=round(a.seconds*fps/a.stride);target=a.output/'briefing.mkv'
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'FFV1'),fps/a.stride,(w,h))
    if not writer.isOpened():raise RuntimeError('FFV1 codec unavailable')
    pixels=hashlib.sha256()
    try:
        for k in range(count):
            for step in range(a.stride):
                ok,image=cap.read()
                if not ok:raise RuntimeError('incomplete source window')
                if step==0:chosen=image
            pixels.update(chosen.tobytes());writer.write(chosen)
            if k%1000==0:print('natural frames',k,'/',count,flush=True)
    finally:writer.release();cap.release()
    def sha(path):
        r=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):r.update(b)
        return r.hexdigest()
    manifest={'schema':'face_effects_natural_dataset/1','provenance':'real official archival video; fixed initial contiguous window; temporal stride only, native dimensions',
        'sources':{'briefing':{'url':a.source_url,'license':a.license,'file_sha256':sha(a.source),'bytes':a.source.stat().st_size}},
        'clips':[{'id':'briefing','source':'briefing','source_start_seconds':0,'source_first_frame':0,
            'source_stride':a.stride,'source_fps':fps,'fps':fps/a.stride,'frames':count,'size':[w,h],
            'file':target.name,'sha256':sha(target),'decoded_pixels_sha256':pixels.hexdigest(),
            'tags':['continuous_unannotated_press_briefing']}],
        'limits':['no independent landmarks/head-pose/video identities','sampled framing audit is not exhaustive face annotation',
                  'timestamps derived from nominal container FPS; no exposure timestamps','complete transfer; only selected window decoded completely']}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
