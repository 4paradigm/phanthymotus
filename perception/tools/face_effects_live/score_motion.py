"""Exploratory short natural-motion score with explicit annotation uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np


def score(root, annotations):
    burst=annotations['nose_burst'];clip=burst['clip']
    rows={r['frame_id']:r for r in map(json.loads,(root/(clip+'.jsonl')).read_text().splitlines())}
    truth=np.asarray([p if p is not None else [np.nan,np.nan] for p in burst['nose_xy']],float)
    frame_ids=burst.get('frame_ids',list(range(burst.get('start_frame',0),burst.get('start_frame',0)+len(truth))))
    fps=float(burst.get('fps',30.))
    uncertainty=float(burst['uncertainty_px'])
    if (len(frame_ids)!=len(truth) or not np.isfinite(fps) or fps<=0
            or not np.isfinite(uncertainty) or uncertainty<0
            or any(type(k) is not int or k<0 for k in frame_ids)
            or any(b<=a for a,b in zip(frame_ids,frame_ids[1:]))):
        raise ValueError('invalid motion annotations')
    raw=[];smooth=[];valid=[]
    for offset,point in enumerate(truth):
        row=rows.get(frame_ids[offset],{});candidates=list(row.get('anchors',{}).items())
        if not candidates or not np.isfinite(point).all():raw.append([np.nan,np.nan]);smooth.append([np.nan,np.nan]);continue
        identity,points=min(candidates,key=lambda c:np.linalg.norm(np.asarray(c[1][2])-point))
        smooth.append(points[2]);raw.append(row['raw_anchors'][identity][2]);valid.append(offset)
    smooth=np.asarray(smooth);raw=np.asarray(raw)
    pairs=[k for k in range(1,len(truth)) if k in valid and k-1 in valid and frame_ids[k]-frame_ids[k-1]==1]
    def residual_jitter(points):
        residual=points-truth
        return float(np.sqrt(np.mean([np.sum((residual[k]-residual[k-1])**2) for k in pairs]))) if pairs else None
    shifts=list(range(-4,5));interior=[k for k in range(4,len(truth)-4)
                                      if all(np.isfinite(smooth[k+s]).all() for s in shifts)
                                      and frame_ids[k+4]-frame_ids[k-4]==8]
    if len(interior)<3:
        return {'valid':False,'valid_frames':len(valid),'annotated_frames':len(truth),
                'reason':'insufficient contiguous visible annotations/predictions for lag search',
                'best_lag_frames':None,'best_lag_ms':None,'lag_identifiable':False,
                'raw_first_difference_residual_rms_px':residual_jitter(raw),
                'stabilized_first_difference_residual_rms_px':residual_jitter(smooth)}
    def curve(labels):
        result=[]
        for shift in shifts:
            delta=smooth[np.array(interior)+shift]-labels[interior]
            delta-=delta.mean(axis=0) # remove static landmark/annotation bias
            result.append(float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))))
        return result
    errors=curve(truth);best=shifts[int(np.argmin(errors))]
    rng=np.random.default_rng(72030);selected=[]
    for _ in range(500):
        angle=rng.uniform(0,2*np.pi,len(truth));radius=uncertainty*np.sqrt(rng.uniform(0,1,len(truth)))
        labels=truth+radius[:,None]*np.column_stack([np.cos(angle),np.sin(angle)])
        selected.append(shifts[int(np.argmin(curve(labels)))])
    return {'valid':True,'valid_frames':len(valid),'annotated_frames':len(truth),'adjacent_pairs':len(pairs),
            'missing_annotations':int((~np.isfinite(truth).all(axis=1)).sum()),'fps':fps,
            'raw_first_difference_residual_rms_px':residual_jitter(raw),
            'stabilized_first_difference_residual_rms_px':residual_jitter(smooth),
            'lag_search_frames':shifts,'bias_removed_rmse_px':errors,
            'best_lag_frames':best,'best_lag_ms':best*1000/fps,
            'lag_identifiable':best not in (shifts[0],shifts[-1]),
            'annotation_perturbation_selected_lag_frames':{str(k):int(n) for k,n in zip(*np.unique(selected,return_counts=True))},
            'uncertainty_px':uncertainty,'seed':72030,
            'limits':['short natural head-motion excerpt, not a rapid-turn benchmark',
                      'annotation provenance must be assessed separately; missing points are excluded',
                      'residual jitter includes label noise and landmark definition differences',
                      'positive lag means output trajectory occurs later than annotation',
                      'perturbation sensitivity is not a statistical confidence interval',
                      'no sensor/transport latency included; lag resolution is one annotated frame']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=score(args.predictions,json.loads(args.annotations.read_text()))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
