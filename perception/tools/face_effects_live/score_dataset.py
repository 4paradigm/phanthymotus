"""Independent dataset quality scores, with explicit partial-label limitations.

Fixed matching gates: IoU >= .2 for landmark-envelope/proxy coverage; .5 for
fully annotated temporal face MOT. No model-dependent GT repair or remapping.
NME here is a five-anchor/COCO-partial proxy, never advertised as full 68/98 NME.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from score_natural import validate_annotations


def distribution(values):
    if not values: return None
    return {'n': len(values), 'mean': float(np.mean(values)),
            **{key: float(np.percentile(values, q)) for key, q in (('p50', 50), ('p95', 95), ('p99', 99))}}


def iou(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[:2]+a[2:], b[:2]+b[2:])
    intersect = float(np.prod(np.maximum(0, hi-lo)))
    return intersect/(a[2]*a[3]+b[2]*b[3]-intersect)


def match_boxes(faces, boxes, gate=.2, previous=None):
    """Maximum cardinality then maximum IoU; <=8 predictions, arbitrary GT.

    A previous valid MOT assignment is retained before resolving remaining pairs,
    as in CLEAR association. No keypoint accuracy is used to pick a match.
    """
    keys = list(boxes)
    if len(keys) > 8: raise ValueError('Card supports at most eight output faces')
    pairs = {}; used = set()
    if previous:
        for index, face in enumerate(faces):
            key = previous.get(face['id'])
            if key in boxes and key not in used and iou(face['box'], boxes[key]) >= gate:
                pairs[index] = key; used.add(key)
    remaining = [i for i in range(len(faces)) if i not in pairs]
    keys = [key for key in keys if key not in used]
    states = {0: (0, 0., [])}
    for index in remaining:
        new = dict(states)
        for mask, (count, total, assignments) in states.items():
            for j, key in enumerate(keys):
                overlap = iou(faces[index]['box'], boxes[key])
                if mask & (1 << j) or overlap < gate: continue
                candidate = (count+1, total+overlap, assignments+[(index, key)])
                newmask = mask | (1 << j)
                if newmask not in new or candidate[:2] > new[newmask][:2]: new[newmask] = candidate
        states = new
    pairs.update(dict(max(states.values(), key=lambda s: s[:2])[2]))
    return pairs


def wrapped_angle_error(actual, truth):
    a = np.asarray(actual); t = np.asarray(truth)
    angle = math.degrees(math.atan2(*(a[1]-a[0])[::-1])-math.atan2(*(t[1]-t[0])[::-1]))
    return abs((angle+180) % 360-180)


def metric_bucket():
    return {'faces': 0, 'detector_matched': 0, 'refined_matched': 0, 'effect_matched': 0,
            'unmatched_refined_predictions': 0, 'false_positives': 0,
            'raw_point_errors_px': [], 'raw_face_mae_px': [], 'raw_face_nme_interocular': [],
            'effect_point_errors_px': [], 'glasses_eye_errors_px': [],
            'glasses_eligible_pairs': 0, 'glasses_rendered_pairs': 0,
            'reported_roll_errors_deg': [], 'partial_osf48_nme_interocular': [],
            'nme_eligible_faces': 0, 'matched_nme_faces': 0, 'nme_above_0_1': 0}


def finish(bucket, exhaustive):
    result = {}
    for key, value in bucket.items(): result[key] = distribution(value) if isinstance(value, list) else value
    result['detector_proxy_missing'] = bucket['faces']-bucket['detector_matched']
    result['refined_proxy_missing'] = bucket['faces']-bucket['refined_matched']
    result['effect_proxy_missing'] = bucket['faces']-bucket['effect_matched']
    result['false_positives'] = bucket['false_positives'] if exhaustive else None
    n = bucket['nme_eligible_faces']
    result['nme_failure_including_missing_rate'] = (
        (n-bucket['matched_nme_faces']+bucket['nme_above_0_1'])/n if n else None)
    return result


def load_rows(path):
    rows = {}
    for row in map(json.loads, Path(path).read_text().splitlines()):
        fid = row['frame_id']
        if type(fid) is not int or fid < 0 or fid in rows: raise ValueError('invalid/duplicate prediction frame ID')
        for key in ('detector_observations', 'detections'):
            for face in row.get(key, []):
                b = np.asarray(face['box'], float); p = np.asarray(face['points'], float)
                if b.shape != (4,) or min(b[2:]) <= 0 or not np.isfinite(b).all() or p.shape != (5, 2) or not np.isfinite(p).all():
                    raise ValueError('invalid prediction geometry')
        rows[fid] = row
    return rows


def ignored_prediction(box, ignore_boxes):
    a = np.asarray(box, float)
    for b in ignore_boxes:
        b = np.asarray(b, float)
        overlap = np.prod(np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:])-np.maximum(a[:2], b[:2])))
        if overlap/(a[2]*a[3]) >= .5: return True
    return False


def score(root, labels):
    validate_annotations(labels)
    total = metric_bucket(); groups = {}; matches_detail = []; temporal = {}; mot = []
    full = all(m.get('exhaustive_face_boxes') is True for m in labels['sequence_metadata'].values())
    for clip, frames in labels['clips'].items():
        rows = load_rows(Path(root)/(clip+'.jsonl')); meta = labels['sequence_metadata'][clip]
        if set(rows) != {f['frame_id'] for f in frames}: raise ValueError('prediction/source frame IDs do not match')
        last_owner = {}; previous_mot = {}; switches = fn = fp = gt_count = 0
        if meta.get('identity_labels') is True:
            if meta.get('temporal') is not True or meta.get('exhaustive_face_boxes') is not True:
                raise ValueError('MOT requires temporal, exhaustive face identity labels')
            ids = [f['frame_id'] for f in frames]
            if any(b != a+1 for a, b in zip(ids, ids[1:])): raise ValueError('MOT source must include empty frames and gaps')
        for frame in frames:
            row = rows[frame['frame_id']]; faces = frame['faces']
            detections = {str(i): f for i, f in enumerate(row.get('detections', []))}
            detector = {str(i): f['box'] for i, f in enumerate(row.get('detector_observations', []))}
            raw = match_boxes(faces, {k: f['box'] for k, f in detections.items()})
            det = match_boxes(faces, detector)
            boxes = row.get('boxes', {}); effect = match_boxes(faces, boxes)
            total['unmatched_refined_predictions'] += len(detections)-len(raw)
            if full:
                total['false_positives'] += sum(
                    not ignored_prediction(f['box'], frame.get('ignore_boxes', []))
                    for key, f in detections.items() if key not in raw.values())
            for i, face in enumerate(faces):
                truth = face['points']; eyes = all(p is not None for p in truth[:2])
                iod = float(np.linalg.norm(np.subtract(truth[1], truth[0]))) if eyes else None
                if iod is not None and iod <= 0: raise ValueError('degenerate annotated eye distance')
                names = ['all']+[str(k) for k, v in face.get('attributes', {}).items() if v == 1]
                if iod is not None:
                    names.append('interocular_'+('lt12' if iod < 12 else '12_24' if iod < 24 else '24_48' if iod < 48 else 'ge48'))
                buckets = [total if n == 'all' else groups.setdefault(n, metric_bucket()) for n in names]
                detail = {'clip': clip, 'frame_id': frame['frame_id'], 'gt_id': face['id'],
                          'refined_matched': i in raw, 'effect_matched': i in effect,
                          'interocular_px': iod, 'raw_face_mae_px': None, 'raw_nme_interocular': None,
                          'glasses_eye_mae_px': None, 'effect_anchor_mae_px': None}
                for bucket in buckets:
                    bucket['faces'] += 1; bucket['detector_matched'] += i in det
                    bucket['refined_matched'] += i in raw; bucket['effect_matched'] += i in effect
                    bucket['nme_eligible_faces'] += eyes; bucket['glasses_eligible_pairs'] += eyes
                if i in raw:
                    face_prediction = detections[raw[i]]; actual = np.asarray(face_prediction['points'])
                    errors = [float(np.linalg.norm(actual[j]-p)) for j, p in enumerate(truth) if p is not None]
                    mae = float(np.mean(errors)) if errors else None
                    detail['raw_face_mae_px'] = mae; detail['raw_nme_interocular'] = mae/iod if eyes and mae is not None else None
                    for bucket in buckets:
                        bucket['raw_point_errors_px'].extend(errors)
                        if mae is not None: bucket['raw_face_mae_px'].append(mae)
                        if eyes and mae is not None:
                            bucket['raw_face_nme_interocular'].append(mae/iod)
                            bucket['matched_nme_faces'] += 1; bucket['nme_above_0_1'] += mae/iod > .1
                    # Verified shared contour indices 0..47 only; OSF's mouth
                    # reorders 48 onward. Never call this complete 68-point NME.
                    if face.get('topology') == 'ibug68' and face_prediction.get('topology') == 'osf66' and eyes:
                        dense = face_prediction.get('landmarks')
                        if dense is not None:
                            residual = [np.linalg.norm(np.asarray(dense[j])-p) for j, p in enumerate(face['dense_points'][:48]) if p is not None]
                            if residual:
                                for bucket in buckets: bucket['partial_osf48_nme_interocular'].append(float(np.mean(residual))/iod)
                if i in effect:
                    key = effect[i]; actual = np.asarray(row['anchors'][key])
                    errors = [float(np.linalg.norm(actual[j]-p)) for j, p in enumerate(truth) if p is not None]
                    detail['effect_anchor_mae_px'] = float(np.mean(errors)) if errors else None
                    for bucket in buckets: bucket['effect_point_errors_px'].extend(errors)
                    rendered = row.get('glasses_eyes', {}).get(key)
                    if eyes and rendered is not None:
                        detail['glasses_eye_mae_px'] = float(np.mean(np.linalg.norm(np.asarray(rendered)-truth[:2], axis=1)))
                        for bucket in buckets:
                            bucket['glasses_rendered_pairs'] += 1
                            bucket['glasses_eye_errors_px'].extend(np.linalg.norm(np.asarray(rendered)-truth[:2], axis=1).tolist())
                    roll = row.get('poses', {}).get(key, {}).get('roll_deg')
                    if eyes and roll is not None:
                        t = np.subtract(truth[1], truth[0]); angle = math.degrees(math.atan2(t[1], t[0]))
                        for bucket in buckets: bucket['reported_roll_errors_deg'].append(abs((roll-angle+180) % 360-180))
                if meta.get('temporal'):
                    temporal.setdefault(clip+':'+face['id'], []).append({
                        'frame_id': frame['frame_id'], 'timestamp': frame['timestamp'],
                        'truth_nose': truth[2], 'predicted_nose': row['anchors'][effect[i]][2] if i in effect else None})
                matches_detail.append(detail)
            if meta.get('identity_labels') is True:
                current = match_boxes(faces, boxes, .5, previous_mot); gt_count += len(faces)
                fn += len(faces)-len(current)
                fp += sum(not ignored_prediction(box, frame.get('ignore_boxes', []))
                          for key, box in boxes.items() if key not in current.values())
                for i, key in current.items():
                    owner = faces[i]['id']
                    if owner in last_owner and last_owner[owner] != key: switches += 1
                    last_owner[owner] = key
                previous_mot = {faces[i]['id']: key for i, key in current.items()}
        if meta.get('identity_labels') is True:
            mot.append({'clip': clip, 'gt_instances': gt_count, 'false_negatives': fn,
                        'false_positives': fp, 'id_switches': switches,
                        'mota': 1-(fn+fp+switches)/gt_count if gt_count else None,
                        'matching': 'CLEAR-style previous-valid association, IoU .5; not HOTA/IDF1'})
    return {'schema': 'face_effects_dataset_quality/1', 'dataset': labels['dataset'],
            'annotation_provenance': labels['provenance'], 'frames': sum(map(len, labels['clips'].values())),
            'overall': finish(total, full), 'groups': {k: finish(v, False) for k, v in groups.items()},
            'mot': mot if mot else None, 'temporal': temporal_summary(temporal) if temporal else None,
            'pose_3d_error_deg': None, 'per_face': matches_detail,
            'limits': ['partial annotations: unmatched predictions are not false positives unless exhaustive face GT',
                       'COCO eye/nose are human semantic proxies, not identical dense-model anatomical definitions',
                       'five-anchor interocular NME is not the standard full 68/98 landmark score',
                       'glasses reference uses independently annotated eye points; other effect boundaries are not annotated',
                       'pose yaw/pitch and capture-to-display latency require independent calibrated truth',
                       'static data provides no temporal stability or multi-person ID evidence']}


def temporal_summary(sequences):
    result = {}
    for key, records in sequences.items():
        residuals = []; lag_errors = {lag: [] for lag in range(-4, 5)}; motion = []
        by_id = {r['frame_id']: r for r in records}
        for fid, record in by_id.items():
            following = by_id.get(fid+1)
            if following and all(r[k] is not None for r in (record, following) for k in ('truth_nose', 'predicted_nose')):
                residuals.append(np.subtract(following['predicted_nose'], following['truth_nose'])-np.subtract(record['predicted_nose'], record['truth_nose']))
            window = [by_id.get(fid+lag) for lag in range(-4, 5)]
            if record['truth_nose'] is None or any(r is None or r['predicted_nose'] is None for r in window): continue
            for lag in lag_errors:
                lag_errors[lag].append(float(np.sum(np.square(np.subtract(by_id[fid+lag]['predicted_nose'], record['truth_nose'])))))
            if following and following['truth_nose'] is not None:
                motion.append(float(np.linalg.norm(np.subtract(following['truth_nose'], record['truth_nose']))))
        lag_profile = {str(lag): math.sqrt(float(np.mean(errors))) if errors else None for lag, errors in lag_errors.items()}
        identifiable = len(lag_errors[0]) >= 12 and len(motion) >= 12 and float(np.mean(motion)) > .5
        result[key] = {'source_frames': len(records), 'adjacent_pairs': len(residuals),
                       'first_difference_residual_rms_px': math.sqrt(float(np.mean(np.square(residuals)))) if residuals else None,
                       'lag_profile_rmse_px': lag_profile, 'lag_comparable_samples': len(lag_errors[0]),
                       'best_alignment_frames': min(lag_errors, key=lambda lag: np.mean(lag_errors[lag])) if identifiable else None,
                       'lag_identifiable': identifiable,
                       'limits': 'phase-alignment proxy only; residual contains annotation noise; no interpolation across gaps'}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', required=True); p.add_argument('--predictions', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); result = score(a.predictions, json.loads(Path(a.labels).read_text()))
    with Path(a.output).open('x') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result['overall'], indent=2))


if __name__ == '__main__': main()
