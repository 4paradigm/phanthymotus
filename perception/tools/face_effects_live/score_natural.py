"""Sparse visual annotations are independent of model predictions.

Reports conditional error AND missing faces. No interpolation invents ground truth.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np


def distribution(values):
    if not values:return None
    return {'n':len(values),'mean':float(np.mean(values)),
            'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95))}


def validate_annotations(annotations):
    for clip,frames in annotations['clips'].items():
        seen=set()
        for frame in frames:
            index=frame['frame_id']
            if type(index) is not int or index<0 or index in seen:
                raise ValueError('frame IDs must be unique nonnegative integers: '+clip)
            seen.add(index)
            for face in frame['faces']:
                box=np.asarray(face['box'],float)
                if box.shape!=(4,) or not np.isfinite(box).all() or min(box[2:])<=0:
                    raise ValueError('invalid annotation face box')
                if len(face['points'])!=5:raise ValueError('expected five semantic points')
                for point in face['points']:
                    if point is not None and (np.asarray(point).shape!=(2,) or not np.isfinite(np.asarray(point,float)).all()):
                        raise ValueError('invalid annotation point; use null for invisible points')


def score(prediction_root, annotations):
    validate_annotations(annotations)
    root=Path(prediction_root)
    per_clip={}
    for clip,frames in annotations['clips'].items():
        predictions={r['frame_id']:r for r in map(json.loads,(root/(clip+'.jsonl')).read_text().splitlines())}
        errors=[];raw_errors=[];normalized=[];roll=[];pose_roll=[];glasses=[];missing=total=detector_missing=0
        detector_errors=[];glasses_eligible=glasses_rendered=0
        id_owner={};person_last={};wrong_reuse=changes=0;matches=[]
        for annotation in frames:
            row=predictions.get(annotation['frame_id'],{})
            anchors=row.get('anchors',{});boxes=row.get('boxes',{})
            assigned=set();raw_assigned=set()
            # GT faces are spatially separated in the sparse set; resolve nearest
            # box center, with a face-size gate, independent of keypoint error.
            for face in annotation['faces']:
                total+=1
                if all(p is not None for p in face['points'][:2]):glasses_eligible+=1
                x,y,w,h=face['box'];center=np.array([x+w/2,y+h/2])
                raw_candidates=[]
                for detection_index,detection in enumerate(row.get('detections',[])):
                    if detection_index in raw_assigned:continue
                    box=np.asarray(detection['box'])
                    distance=np.linalg.norm(box[:2]+box[2:]/2-center)/max(w,h)
                    if distance<.7:raw_candidates.append((distance,detection_index,detection))
                if raw_candidates:
                    _,detection_index,detection=min(raw_candidates,key=lambda c:c[0])
                    raw_assigned.add(detection_index)
                    for index,truth in enumerate(face['points']):
                        if truth is not None:detector_errors.append(float(np.linalg.norm(np.asarray(detection['points'][index])-truth)))
                else:detector_missing+=1
                candidates=[]
                for identity,box in boxes.items():
                    if identity in assigned:continue
                    box=np.asarray(box);distance=np.linalg.norm(box[:2]+box[2:]/2-center)/max(w,h)
                    if distance<.7:candidates.append((distance,identity))
                if not candidates:
                    missing+=1;continue
                _,identity=min(candidates);assigned.add(identity)
                owner=id_owner.get(identity)
                if owner is not None and owner!=face['id']:wrong_reuse+=1
                id_owner[identity]=face['id']
                previous=person_last.get(face['id'])
                if previous is not None and previous!=identity:changes+=1
                person_last[face['id']]=identity
                actual=np.asarray(anchors[identity]);raw=np.asarray(row.get('raw_anchors',anchors)[identity])
                point_errors=[]
                for index,truth in enumerate(face['points']):
                    if truth is None:continue
                    error=float(np.linalg.norm(actual[index]-truth));errors.append(error);point_errors.append(error)
                    raw_errors.append(float(np.linalg.norm(raw[index]-truth)));normalized.append(error/w)
                if face['points'][0] is not None and face['points'][1] is not None:
                    truth=np.asarray(face['points'][:2]);a=truth[1]-truth[0];b=actual[1]-actual[0]
                    delta=math.degrees(math.atan2(b[1],b[0])-math.atan2(a[1],a[0]))
                    roll.append(abs((delta+180)%360-180))
                    reported=row.get('poses',{}).get(identity,{}).get('roll_deg')
                    if reported is not None:
                        reported_delta=reported-math.degrees(math.atan2(a[1],a[0]))
                        pose_roll.append(abs((reported_delta+180)%360-180))
                    rendered=row.get('glasses_eyes',{}).get(identity)
                    if rendered is not None:
                        glasses_rendered+=1
                        glasses.extend(np.linalg.norm(np.asarray(rendered)-truth,axis=1).tolist())
                matches.append({'frame_id':annotation['frame_id'],'gt_id':face['id'],'track_id':identity,
                                'point_mae_px':float(np.mean(point_errors)) if point_errors else None})
        per_clip[clip]={'annotated_face_instances':total,'matched':total-missing,'missing':missing,
                        'missing_rate':missing/total if total else None,
                        'detection_missing':detector_missing,'detection_missing_rate':detector_missing/total if total else None,
                        'unfiltered_detection_point_error_px':distribution(detector_errors),
                        'point_error_px':distribution(errors),'raw_point_error_px':distribution(raw_errors),
                        'point_error_face_width_fraction':distribution(normalized),
                        'roll_2d_error_deg':distribution(roll),'glasses_eye_error_px':distribution(glasses),
                        'reported_pose_roll_2d_error_deg':distribution(pose_roll),
                        'glasses_eligible_eye_pairs':glasses_eligible,'glasses_rendered_eye_pairs':glasses_rendered,
                        'wrong_track_id_reuse_observed':wrong_reuse,'sparse_id_changes':changes,'matches':matches}
    return {'schema':'face_effects_sparse_quality/1','clips':per_clip,
            'annotation_status':{'annotator':annotations.get('annotator'),
                                 'independent_human_review':annotations.get('independent_human_review',False) is True,
                                 'uncertainty_px':annotations.get('uncertainty_px')},
            'limits':['annotation provenance is explicit in annotation_status; do not certify truth from scoring code',
                      'sparse identity metrics only at reviewed frames, not full MOT IDSW',
                      '2D eye-line roll is measured; metric 3D yaw/pitch ground truth unavailable',
                      'conditional point error must be interpreted with missing_rate',
                      'face-box normalization avoids degenerate interocular distances on profiles']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--require-reviewed',action='store_true',help='reject draft/partial annotations for a formal run')
    args=parser.parse_args()
    annotations=json.loads(Path(args.annotations).read_text())
    if args.require_reviewed and annotations.get('independent_human_review',False) is not True:
        parser.error('independent human review has not been declared; use this dataset only for exploratory analysis')
    result=score(args.predictions,annotations)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    for key,row in result['clips'].items():print(key,row['point_error_px'],row['missing_rate'],row['wrong_track_id_reuse_observed'])


if __name__=='__main__':main()
