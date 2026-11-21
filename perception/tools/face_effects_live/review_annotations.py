"""Validate source hashes and draw source-only labels for independent review.

No inference output is read. These overlays do not certify label correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clips',type=Path,help='also verify PNG pixels against the named decoded clip frame')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    labels=json.loads(args.annotations.read_text());count=0
    colors=[(40,220,40),(40,200,250),(250,130,30),(200,40,240),(240,220,50)]
    for clip,frames in labels['clips'].items():
        thumbnails=[]
        cap=cv2.VideoCapture(str(args.clips/(clip+'.mkv'))) if args.clips else None
        for frame in frames:
            path=args.frames/f"{clip}-{frame['frame_id']}.png"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==frame['image_sha256'],path
            image=cv2.imread(str(path));assert list(image.shape[1::-1])==labels['coordinate_space']
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame['frame_id']);ok,decoded=cap.read()
                assert ok and np.array_equal(decoded,image),'source frame mismatch: '+str(path)
            for face in frame['faces']:
                x,y,w,h=map(int,face['box']);cv2.rectangle(image,(x,y),(x+w,y+h),(80,200,80),1)
                cv2.putText(image,face['id'],(x,max(y-5,15)),0,.6,(80,255,80),2)
                for i,point in enumerate(face['points']):
                    if point is None:continue
                    px,py=map(int,point);cv2.drawMarker(image,(px,py),colors[i],cv2.MARKER_CROSS,12,1)
                    cv2.putText(image,str(i),(px+4,py-4),0,.4,colors[i],1)
            cv2.putText(image,f'{clip} frame {frame["frame_id"]} / labels need human review',(10,30),0,.65,(0,255,255),2)
            cv2.imwrite(str(args.output/f'{clip}-{frame["frame_id"]}.jpg'),image)
            thumbnails.append(cv2.resize(image,(640,360)));count+=1
        if len(thumbnails)==4:
            cv2.imwrite(str(args.output/(clip+'-sheet.jpg')),np.vstack([np.hstack(thumbnails[:2]),np.hstack(thumbnails[2:])]))
        if cap is not None:cap.release()
    (args.output/'validation.json').write_text(json.dumps({'source_image_hashes_verified':count,
        'annotations_sha256':hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        'clip_frame_pixels_verified':count if args.clips else None,
        'independent_human_review':False},indent=2)+'\n')
    print('source image hashes verified:',count,'; overlays generated, not expert truth')


if __name__=='__main__':main()
