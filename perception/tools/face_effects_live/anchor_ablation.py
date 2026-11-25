"""Static post-inference geometry hypotheses, never annotation editing or timing evidence.

Reuses frozen model predictions to compare geometry alone. A selected hypothesis
must subsequently run through real inference/rendering; this tool cannot certify
tracking, runtime performance or natural-video accuracy.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from plugins.face_effects_live.contracts import Face
from plugins.face_effects_live.effects import Effects
from plugins.face_effects_live.pose import estimate_pose
from score_dataset import score


def run(labels_path, predictions, output, mode):
    labels=json.loads(Path(labels_path).read_text())
    if any(m.get('temporal') for m in labels['sequence_metadata'].values()):
        raise ValueError('hypothesis projection requires independent static observations')
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    effects=Effects()
    for clip in labels['clips']:
        rows=[]
        for line in (Path(predictions)/(clip+'.jsonl')).read_text().splitlines():
            row=json.loads(line);row['hypothesis_only']=mode
            for face in row['detections']:
                lms=np.asarray(face['landmarks'],float)
                if mode in ('pixel-center','corners-pixel-center'):
                    x,y,w,h=face['box'];sw,sh=row['size']
                    x1,y1=max(0,int(x-.1*w)),max(0,int(y-.125*h))
                    x2,y2=min(sw,int(x+1.1*w)),min(sh,int(y+1.125*h))
                    lms+=.5*(np.array([x2-x1,y2-y1])/224.-1.)
                eye_centers=[lms[36:42].mean(0),lms[42:48].mean(0)]
                if mode in ('corners','corners-pixel-center'):
                    eye_centers=[lms[[36,39]].mean(0),lms[[42,45]].mean(0)]
                p=np.array([*eye_centers,lms[30],lms[58],lms[62]])
                p[:2]=p[:2][np.argsort(p[:2,0])];p[3:]=p[3:][np.argsort(p[3:,0])]
                face['points']=p.tolist();face['landmarks']=lms.tolist()
            for key,box in row['boxes'].items():
                matches=[f for f in row['detections'] if np.array_equal(f['box'],box)]
                if len(matches)!=1:raise ValueError('cannot associate frozen static track to raw face')
                f=matches[0];p=np.asarray(f['points']);lms=np.asarray(f['landmarks'])
                face=Face(np.array(box),p,float(f['confidence']),lms,float(f['quality']),topology=f['topology'])
                track=SimpleNamespace(points=p,face=face)
                row['anchors'][key]=p.tolist();row['raw_anchors'][key]=p.tolist()
                pose=estimate_pose(face,row['size']);row['poses'][key]=pose
                commands=effects.plugins['glasses'].commands(track,0,pose)
                row['glasses_eyes'].pop(key,None)
                if commands:
                    m=commands[0].matrix
                    row['glasses_eyes'][key]=(np.array([[62.,44.],[178.,44.]])@m[:,:2].T+m[:,2]).tolist()
            rows.append(row)
        (output/(clip+'.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    effects.close()
    result=score(output,labels);result['hypothesis_only']='post-inference static geometry, no performance/temporal claim'
    (output/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'mode':mode,'overall':result['overall']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['labels','predictions','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--mode',required=True,choices=['control','corners','pixel-center','corners-pixel-center'])
    a=p.parse_args();run(a.labels,a.predictions,a.output,a.mode)
