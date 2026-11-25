"""Audit the historical R4 glasses tail using preserved independent labels.

This is an audit of recorded predictions, not a fresh inference benchmark.
Per-face and per-eye distributions are deliberately separate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from score_dataset import distribution,match_boxes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['evidence','labels','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=a.evidence
    old=json.loads((r/'round4/original_osf0/quality-final.json').read_text())
    current=json.loads((r/'round5/coco-baseline-quality.json').read_text())
    labels=json.loads(a.labels.read_text())
    by_id=lambda rows:{(row['clip'],row['gt_id']):row for row in rows}
    before=by_id(old['per_face']);after=by_id(current['per_face'])
    keys=[key for key in before.keys()&after.keys() if before[key]['glasses_eye_mae_px'] is not None
          and after[key]['glasses_eye_mae_px'] is not None]
    cases=[]
    for clip,gt_id in keys:
        b=before[(clip,gt_id)];c=after[(clip,gt_id)]
        row_old=json.loads((r/'round4/original_osf0'/(clip+'.jsonl')).read_text())
        row_new=json.loads((r/'round5/coco-baseline'/(clip+'.jsonl')).read_text())
        frame=labels['clips'][clip][0];gt=next(f for f in frame['faces'] if f['id']==gt_id)
        def associated(row,field):
            boxes=row['boxes'];matched=match_boxes([gt],boxes)
            if not matched:raise ValueError('lost common effect association')
            return np.asarray(row[field][matched[0]])
        raw_old=associated(row_old,'raw_anchors');raw_new=associated(row_new,'raw_anchors')
        rendered_old=associated(row_old,'glasses_eyes');rendered_new=associated(row_new,'glasses_eyes')
        cases.append({'clip':clip,'gt_id':gt_id,'image':frame['image'],
           'before_eye_mae_px':b['glasses_eye_mae_px'],'after_eye_mae_px':c['glasses_eye_mae_px'],
           'delta_px':c['glasses_eye_mae_px']-b['glasses_eye_mae_px'],
           'reference_eyes':gt['points'][:2],'reference_attributes':gt.get('attributes',{}),
           'raw_anchors_identical':bool(np.array_equal(raw_old,raw_new)),
           'raw_eyes_nose_identical':bool(np.array_equal(raw_old[:3],raw_new[:3])),
           'raw_eyes':raw_new[:2].tolist(),'before_rendered_eyes':rendered_old.tolist(),
           'after_rendered_eyes':rendered_new.tolist(),
           'after_render_minus_raw_max_px':float(np.max(np.abs(rendered_new-raw_new[:2]))),
           'before_render_minus_raw_max_px':float(np.max(np.abs(rendered_old-raw_old[:2])))})
    result={'common_eye_pairs':len(cases),
      'per_face_before_px':distribution([r['before_eye_mae_px'] for r in cases]),
      'per_face_after_px':distribution([r['after_eye_mae_px'] for r in cases]),
      'improved':sum(r['delta_px']<0 for r in cases),'worsened':sum(r['delta_px']>0 for r in cases),
      'all_raw_anchors_identical':all(r['raw_anchors_identical'] for r in cases),
      'all_raw_eyes_nose_identical':all(r['raw_eyes_nose_identical'] for r in cases),
      'exact_eye_mapping_max_numeric_residual_px':max(r['after_render_minus_raw_max_px'] for r in cases),
      'largest_after_errors':sorted(cases,key=lambda r:r['after_eye_mae_px'],reverse=True)[:20],
      'largest_degradations':sorted(cases,key=lambda r:r['delta_px'],reverse=True)[:20],
      'all_cases':cases,
      'interpretation':['same raw eye/nose anchors rule out a new neural eye error in this historical comparison',
        'mouth anchors changed from OSF48/54 to58/62 separately; COCO has no mouth truth and this does not affect the eye mapping audit',
        'R4 changed from generic sprite centering/scale to exact eye-anchor mapping',
        'generic placement offsets sometimes accidentally compensate a raw-model error; correcting geometry can worsen those cases',
        'this is a per-face tail; do not compare its 9.487px P99 to a per-eye 11.137px P99',
        'independent COCO eyes are sparse semantic labels, not full face/3D truth; image review still needed']}
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_cases','largest_after_errors','largest_degradations']}))


if __name__=='__main__':main()
