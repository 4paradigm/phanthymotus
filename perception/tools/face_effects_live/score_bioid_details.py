"""Separate official BioID .eye and FGnet-20 annotation protocols.

No prediction-based relabeling, threshold selection or coordinate fitting.
The five-anchor score is a proxy; OSF eye-contour means are not pupil centers.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from score_dataset import distribution,load_rows,match_boxes


def score(labels,root):
    anchors={n:[] for n in ['image_left_eye','image_right_eye','nose','image_left_mouth','image_right_mouth']}
    render_errors=[];eye_e=[];misses=[];per_image={};n=0
    for clip,frames in labels['clips'].items():
        rows=load_rows(root/(clip+'.jsonl'))
        if len(frames)!=1 or len(rows)!=1:raise ValueError('expected static BioID source')
        frame=frames[0];row=rows[frame['frame_id']];truth=frame['faces'][0];n+=1
        detections={str(i):f for i,f in enumerate(row['detections'])}
        matched=match_boxes([truth],{k:f['box'] for k,f in detections.items()})
        if not matched:misses.append(clip);per_image[clip]=None;continue
        points=np.array(detections[matched[0]]['points']);gt=np.array(truth['points'])
        error=np.linalg.norm(points-gt,axis=1)
        for name,value in zip(anchors,error):anchors[name].append(float(value))
        per_image[clip]={'mae_px':float(error.mean()),'nose_px':float(error[2])}
        # Keep this independent integer-pupil protocol separate from FGnet points.
        pupils=np.array(truth['bioid_official_eyes'])
        normalized=np.max(np.linalg.norm(points[:2]-pupils,axis=1))/np.linalg.norm(pupils[1]-pupils[0])
        eye_e.append(float(normalized))
        effect=match_boxes([truth],row.get('boxes',{}))
        if effect and effect[0] in row['glasses_eyes']:
            render_errors.extend(np.linalg.norm(np.array(row['glasses_eyes'][effect[0]])-gt[:2],axis=1).tolist())
    return {'source_images':n,'matched_images':n-len(misses),'unmatched_images':misses,
        'raw_fgnet_five_anchor_errors_px':{k:distribution(v) for k,v in anchors.items()},
        'rendered_eyes_vs_fgnet_pupils_px':distribution(render_errors),
        'official_eye_max_error_over_interocular':distribution(eye_e),
        'official_eye_e_le_0_25_count':sum(e<=.25 for e in eye_e),
        'official_eye_e_le_0_25_rate_all_images':sum(e<=.25 for e in eye_e)/n,
        'per_image':per_image,
        'limits':['.eye and FGnet20 are independent annotation files, reported separately',
                  'five-anchor proxy, not standard 68/98 landmark NME',
                  'eye contour means and pupil centers have different semantics',
                  'original coordinates retained; no offset fitted against predictions',
                  'static confirmation repetitions cannot measure true jitter, lag or recovery']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['labels','baseline','candidate','output']:p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();labels=json.loads(a.labels.read_text())
    results={k:score(labels,path) for k,path in [('baseline',a.baseline),('candidate',a.candidate)]}
    common=[k for k,v in results['baseline']['per_image'].items() if v is not None and results['candidate']['per_image'][k] is not None]
    delta=[results['candidate']['per_image'][k]['mae_px']-results['baseline']['per_image'][k]['mae_px'] for k in common]
    results['paired']={'same_input_matched_images':len(common),'improved':sum(d<0 for d in delta),
                       'worsened':sum(d>0 for d in delta),'equal':sum(d==0 for d in delta),
                       'candidate_minus_baseline_mae_px':distribution(delta)}
    with a.output.open('x') as f:json.dump(results,f,indent=2)
    print(json.dumps(results['paired']))


if __name__=='__main__':main()
