"""Continuity diagnostics for natural video WITHOUT independent temporal truth.

IDs, motion strata and reference anchors below come from the evaluated model.
They cannot measure identity correctness, true jitter, true lag or 3D accuracy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from score_dataset import distribution


def summarize(path):
    rows=[json.loads(line) for line in Path(path).open()]
    if any(b['frame_id']!=a['frame_id']+1 for a,b in zip(rows,rows[1:])):
        raise ValueError('diagnostics require every source frame, including failures')
    visible=[bool(r['ids']) for r in rows]
    gaps=[];start=None
    for i,present in enumerate(visible+[True]):
        if not present and start is None:start=i
        if present and start is not None:
            before=rows[start-1] if start else None;after=rows[i] if i<len(rows) else None
            gaps.append({'first_frame':rows[start]['frame_id'],'frames':i-start,
                         'duration_ms':(rows[i-1]['timestamp']-rows[start]['timestamp']+
                              (rows[1]['timestamp']-rows[0]['timestamp']))*1000,
                         'before_ids':before['ids'] if before else [],
                         'after_ids':after['ids'] if after else [],
                         'bounded_by_visible_frames':bool(before and after),
                         'scene_reset_at_return':after.get('scene_reset') if after else None})
            start=None
    raw_motion=[];filtered_motion=[];filter_residual=[];roll_motion=[];scales=[]
    low_motion=[];events=[];lag={k:[] for k in range(4)};single_ids=set()
    for i,row in enumerate(rows):
        if len(row['ids'])!=1:continue
        tid=str(row['ids'][0]);single_ids.add(tid)
        if tid not in row['raw_anchors'] or tid not in row['anchors']:continue
        raw=np.asarray(row['raw_anchors'][tid]);filtered=np.asarray(row['anchors'][tid])
        filter_residual.extend(np.linalg.norm(raw-filtered,axis=1).tolist())
        if i==0:continue
        prev=rows[i-1]
        if prev['ids']!=row['ids'] or row.get('scene_reset'):continue
        old_raw=np.asarray(prev['raw_anchors'][tid]);old_filtered=np.asarray(prev['anchors'][tid])
        dr=float(np.mean(np.linalg.norm(raw-old_raw,axis=1)))
        df=float(np.mean(np.linalg.norm(filtered-old_filtered,axis=1)))
        raw_motion.append(dr);filtered_motion.append(df)
        if dr<.2:low_motion.append(df)
        events.append({'frame_id':row['frame_id'],'timestamp':row['timestamp'],
                       'raw_anchor_step_px':dr,'filtered_anchor_step_px':df,
                       'bridge':row.get('flow_bridge',False),
                       'source':row.get('landmark_sources',{}).get(tid)})
        old_eyes=np.asarray(prev.get('glasses_eyes',{}).get(tid,[]))
        eyes=np.asarray(row.get('glasses_eyes',{}).get(tid,[]))
        if eyes.shape==(2,2) and old_eyes.shape==(2,2):
            scales.append(abs(np.linalg.norm(eyes[1]-eyes[0])/np.linalg.norm(old_eyes[1]-old_eyes[0])-1))
        if tid in row['poses'] and tid in prev['poses']:
            delta=row['poses'][tid]['roll_deg']-prev['poses'][tid]['roll_deg']
            roll_motion.append(abs((delta+180)%360-180))
        # Exactly the same complete windows at each lag; never cross a miss,
        # identity, reset or model-source change to improve apparent alignment.
        window=rows[max(0,i-3):i+1]
        if len(window)==4 and all(w['ids']==row['ids'] and not w.get('scene_reset')
             and w.get('landmark_sources',{}).get(tid)==row.get('landmark_sources',{}).get(tid)
             for w in window):
            for offset in lag:
                target=np.asarray(rows[i-offset]['raw_anchors'][tid])
                lag[offset].append(float(np.mean(np.square(filtered-target))))
    bridges=[r for r in rows if r.get('flow_bridge')]
    return {'frames':len(rows),'first_frame':rows[0]['frame_id'],'last_frame':rows[-1]['frame_id'],
        'visible_effect_frames':sum(visible),'visible_effect_coverage':sum(visible)/len(rows),
        'zero_effect_runs':gaps,'internal_zero_effect_runs':sum(g['bounded_by_visible_frames'] for g in gaps),
        'distinct_ids_on_single_prediction_frames':len(single_ids),
        'bridge_frames':len(bridges),'bridge_frame_ids':[r['frame_id'] for r in bridges],
        'scene_reset_frames':[r['frame_id'] for r in rows if r.get('scene_reset')],
        'frames_with_multiple_model_faces':sum(len(r['detections'])>1 for r in rows),
        'raw_anchor_adjacent_step_px':distribution(raw_motion),
        'filtered_anchor_adjacent_step_px':distribution(filtered_motion),
        'filter_vs_current_raw_point_distance_px':distribution(filter_residual),
        'filtered_step_in_model_low_motion_frames_px':distribution(low_motion),
        'glasses_scale_adjacent_relative_change':distribution(scales),
        'reported_2d_roll_adjacent_change_deg':distribution(roll_motion),
        'filtered_alignment_to_previous_raw_rmse_px':{
            str(k):float(np.sqrt(np.mean(v))) if v else None for k,v in lag.items()},
        'largest_adjacent_raw_motion_cases':sorted(events,key=lambda r:r['raw_anchor_step_px'],reverse=True)[:30],
        'truth_based_track_breaks':None,'true_recovery_time':None,'incorrect_attachment_count':None,
        'true_jitter_px':None,'true_lag_ms':None,'yaw_pitch_roll_3d_error_deg':None,
        'limits':['all frames counted; zero-effect includes detector misses, unconfirmed tracks and scene changes',
          'track IDs are model outputs, not independently labeled person identities',
          'adjacent steps include real motion and landmark error; they are not ground-truth jitter',
          'alignment uses noisy model raw anchors; it is not physical motion lag or accuracy',
          'low-motion strata and largest motion cases are model-selected diagnostics, not independent fast-turn labels',
          'no independent 3D angle labels; yaw/pitch/roll correctness remains unmeasured']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result={name:summarize(path) for name,path in [('baseline',a.baseline),('candidate',a.candidate)]}
    if result['baseline']['frames']!=result['candidate']['frames']:raise ValueError('unequal input length')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:{n:v[n] for n in ['frames','visible_effect_coverage','internal_zero_effect_runs','bridge_frames']} for k,v in result.items()}))


if __name__=='__main__':main()
