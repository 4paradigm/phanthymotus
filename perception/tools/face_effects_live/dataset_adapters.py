"""Lossless annotation import. No model, interpolation, clipping or GT fitting.

The output extends score_natural's clips/frames/five-points format. Dense labels
retain their original topology; a five-anchor score is NOT a 68/98-point NME.
Download/permission checks are deliberately separate from format conversion.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


ATTRIBUTES = ('pose', 'expression', 'illumination', 'makeup', 'occlusion', 'blur')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_path(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('annotation path must stay inside the dataset')
    return p.as_posix()


def finite_point(point):
    p = np.asarray(point, dtype=float)
    return p.tolist() if p.shape == (2,) and np.isfinite(p).all() else None


def mean_points(points, indices):
    selected = [points[i] for i in indices]
    # A missing contour point does not silently change the center definition.
    return None if any(p is None for p in selected) else np.mean(selected, axis=0).tolist()


def semantic_points(points, topology):
    if topology == 'ibug68':
        groups = (range(36, 42), range(42, 48), [30], [48], [54])
    elif topology == 'wflw98':
        groups = (range(60, 68), range(68, 76), [54], [76], [82])
    else:
        raise ValueError('no verified landmark mapping for ' + topology)
    result = [mean_points(points, indices) for indices in groups]
    # Match the Card's image-left/image-right convention, not subject laterality.
    for a, b in ((0, 1), (3, 4)):
        if result[a] is not None and result[b] is not None and result[a][0] > result[b][0]:
            result[a], result[b] = result[b], result[a]
    return result


def envelope(points):
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        raise ValueError('too few points for a landmark envelope')
    p = np.asarray(valid); lo = p.min(0); extent = p.max(0) - lo
    if min(extent) <= 0:
        raise ValueError('degenerate landmark envelope')
    return [*lo.tolist(), *extent.tolist()]


def base(dataset, source, license_note):
    return {'schema': 'face_effects_dataset/1', 'dataset': dataset,
            'annotator': 'upstream dataset annotators; see provenance',
            'independent_human_review': False,
            'annotation_origin': 'external_dataset_not_local_workbench_review',
            'provenance': {'annotation_file': str(Path(source).resolve()),
                           'annotation_sha256': sha256(source), 'license': license_note,
                           'conversion': 'fixed topology mapping; no predictions or GT repair'},
            'clips': {}, 'sequence_metadata': {}}


def read_pts(path, coordinate_origin):
    if coordinate_origin not in ('zero', 'one'):
        raise ValueError('explicit zero/one coordinate origin required')
    text = Path(path).read_text()
    match = re.fullmatch(r'\s*version:\s*1\s+n_points:\s*(\d+)\s*\{([^}]*)\}\s*', text)
    if not match or int(match[1]) != 68:
        raise ValueError('expected version 1, exactly 68 PTS points')
    lines = [line.split() for line in match[2].splitlines() if line.strip()]
    if len(lines) != 68 or any(len(row) != 2 for row in lines):
        raise ValueError('PTS point count/shape mismatch')
    offset = 1 if coordinate_origin == 'one' else 0
    return [finite_point(np.asarray(row, float) - offset) for row in lines]


def convert_300vw(root, sequence, fps, coordinate_origin, license_note):
    root = Path(root); relative_path(sequence)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('positive source FPS required')
    paths = sorted((root / sequence / 'annot').glob('*.pts'), key=lambda p: int(p.stem))
    if not paths:
        raise ValueError('no sequence/annot/*.pts files')
    result = base('300-VW', paths[0], license_note)
    result['provenance']['pts_sha256'] = {p.name: sha256(p) for p in paths}
    result['provenance']['coordinate_origin'] = coordinate_origin
    frames = []; seen = set()
    for path in paths:
        frame_id = int(path.stem)
        if frame_id < 1 or frame_id in seen:
            raise ValueError('300-VW source frame numbers start at 1')
        seen.add(frame_id)
        dense = read_pts(path, coordinate_origin)
        frames.append({'frame_id': frame_id, 'decode_index': frame_id - 1,
                       'timestamp': (frame_id - 1) / fps,
                       'faces': [{'id': sequence + ':subject', 'box': envelope(dense),
                                  'box_kind': 'landmark_envelope_not_detector_box',
                                  'points': semantic_points(dense, 'ibug68'),
                                  'dense_points': dense, 'topology': 'ibug68',
                                  'visibility': None, 'attributes': {}}]})
    result['clips'][sequence] = frames
    result['sequence_metadata'][sequence] = {
        'temporal': True, 'identity_labels': False, 'fps': fps,
        'video': relative_path(sequence + '/vid.avi'), 'exhaustive_face_boxes': False,
        'limits': 'single annotated subject; never evidence for multi-person ID switches'}
    return result


def convert_wflw(annotation, license_note):
    result = base('WFLW', annotation, license_note)
    for index, line in enumerate(Path(annotation).read_text().splitlines()):
        tokens = line.split()
        if len(tokens) != 207:
            raise ValueError('WFLW requires 196 coordinates + 4 bbox + 6 flags + path')
        dense = [finite_point(p) for p in np.asarray(tokens[:196], float).reshape(98, 2)]
        x1, y1, x2, y2 = map(float, tokens[196:200])
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            raise ValueError('invalid WFLW box')
        flags = list(map(int, tokens[200:206]))
        if any(v not in (0, 1) for v in flags):
            raise ValueError('invalid WFLW attribute')
        clip = 'wflw_' + str(index)
        result['clips'][clip] = [{'frame_id': index, 'timestamp': None,
            'image': relative_path(tokens[-1]), 'faces': [{
                'id': str(index), 'box': [x1, y1, x2 - x1, y2 - y1],
                'box_kind': 'dataset_face_box', 'points': semantic_points(dense, 'wflw98'),
                'dense_points': dense, 'topology': 'wflw98', 'visibility': None,
                'attributes': dict(zip(ATTRIBUTES, flags))}]}]
        result['sequence_metadata'][clip] = {
            'temporal': False, 'identity_labels': False, 'exhaustive_face_boxes': False,
            'limits': 'attribute occlusion is per-face, not per-landmark visibility'}
    return result


def coco_face(annotation):
    p = np.asarray(annotation['keypoints'], float).reshape(17, 3)
    if not np.isfinite(p).all() or any(v not in (0, 1, 2) for v in p[:, 2]):
        raise ValueError('invalid COCO keypoint/visibility')
    # COCO indices are subject-left/subject-right. Retain this distinction when
    # only one eye is visible: the Card's slots cannot be inferred from x alone.
    eyes = [finite_point(p[i, :2]) if p[i, 2] == 2 else None for i in (2, 1)]
    if all(e is not None for e in eyes):
        eyes.sort(key=lambda e: e[0])
    else:
        eyes = [None, None]  # preserve labeled eye in dense_points, not a guessed slot
    nose = finite_point(p[0, :2]) if p[0, 2] == 2 else None
    points = eyes + [nose, None, None]
    head = [finite_point(row[:2]) for row in p[:5] if row[2] > 0]
    if len(head) < 2:
        return None
    try:
        tight = envelope(head)
    except ValueError:
        return None
    # Derived matching region only. This is not a human face-box annotation.
    side = max(tight[2:]) * 2
    center = [tight[0] + tight[2]/2, tight[1] + tight[3]/2]
    return {'id': str(annotation['id']),
            'box': [center[0]-side/2, center[1]-side/2, side, side],
            'box_kind': 'derived_head_region_not_detector_truth', 'points': points,
            'dense_points': [finite_point(row[:2]) if row[2] > 0 else None for row in p],
            'visibility': p[:, 2].astype(int).tolist(), 'topology': 'coco17',
            'person_box': annotation['bbox'],
            'attributes': {'both_eyes_visible': int(all(e is not None for e in eyes)),
                           'head_partly_invisible': int(any(row[2] < 2 for row in p[:3]))}}


def convert_coco(annotation, image_ids, license_note):
    data = json.loads(Path(annotation).read_text()); result = base('COCO2017-keypoints', annotation, license_note)
    categories = {c['id']: c for c in data['categories']}
    if categories[1]['keypoints'][:5] != ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']:
        raise ValueError('unrecognized COCO topology')
    groups = {}
    for a in data['annotations']:
        if a['category_id'] == 1 and not a.get('iscrowd'):
            face = coco_face(a)
            if face is not None and any(p is not None for p in face['points']):
                groups.setdefault(a['image_id'], []).append(face)
    images = {i['id']: i for i in data['images']}
    for image_id in image_ids:
        image = images[image_id]; clip = 'coco_' + str(image_id)
        result['clips'][clip] = [{'frame_id': image_id, 'timestamp': None,
            'image': relative_path(image['file_name']), 'faces': groups.get(image_id, [])}]
        result['sequence_metadata'][clip] = {
            'temporal': False, 'identity_labels': False, 'exhaustive_face_boxes': False,
            'image_license': next(l for l in data['licenses'] if l['id'] == image['license']),
            'source_image': image,
            'limits': 'partial semantic eye/nose proxies; no dense face NME, face AP or temporal identity GT'}
    return result


def convert_mot_face(annotation, frame_root, fps, license_note):
    """MOTChallenge-shaped FACE boxes, only with explicit source provenance.

    This is a format adapter, not a claim that MOT17 person boxes are face GT.
    Header JSON next to CSV must declare annotation_target=face, identity_labels
    true and exhaustive_face_boxes true; ordinary pedestrian MOT input is refused.
    """
    metadata = json.loads(Path(str(annotation) + '.provenance.json').read_text())
    if (metadata.get('annotation_target') != 'face' or metadata.get('identity_labels') is not True
            or metadata.get('exhaustive_face_boxes') is not True or not metadata.get('source_url')
            or metadata.get('coordinate_origin') not in ('zero', 'one')):
        raise ValueError('requires trusted temporal FACE-box provenance, not person boxes')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('positive source FPS required')
    result = base(metadata['dataset'], annotation, license_note); groups = {}; ignored = {}
    for row in csv.reader(Path(annotation).read_text().splitlines()):
        if len(row) != 9:
            raise ValueError('MOT face CSV requires nine columns')
        f, identity = int(row[0]), int(row[1]); box = list(map(float, row[2:6]))
        if f < 1 or identity < 0 or not np.isfinite(box).all() or min(box[2:]) <= 0:
            raise ValueError('invalid MOT face annotation')
        offset = 1 if metadata['coordinate_origin'] == 'one' else 0
        box[:2] = [v-offset for v in box[:2]]
        if float(row[6]) == 0:
            ignored.setdefault(f, []).append(box); continue
        if float(row[7]) != 1: raise ValueError('unknown face class; supply explicit normalized face annotations')
        faces = groups.setdefault(f, [])
        if any(face['id'] == str(identity) for face in faces):
            raise ValueError('duplicate identity within frame')
        faces.append({'id': str(identity), 'box': box, 'box_kind': 'dataset_face_box',
                      'points': [None]*5, 'visibility': float(row[8]), 'attributes': {}})
    # All source frames, including genuinely empty frames, must be provided.
    frames = []
    for path in sorted([*Path(frame_root).glob('*.jpg'), *Path(frame_root).glob('*.png')], key=lambda p: int(p.stem)):
        number = int(path.stem)
        if number < 1 or any(f['frame_id'] == number for f in frames): raise ValueError('duplicate/invalid source frame')
        frames.append({'frame_id': number, 'timestamp': (number-1)/fps,
                       'image': path.name, 'faces': groups.pop(number, []), 'ignore_boxes': ignored.pop(number, [])})
    if groups or ignored or not frames: raise ValueError('missing annotated source frames')
    result['clips']['mot_faces'] = frames
    result['sequence_metadata']['mot_faces'] = {**metadata, 'temporal': True, 'fps': fps}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', required=True, choices=['300vw', 'wflw', 'coco', 'mot-face'])
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--license-note', required=True, help='record actual upstream permission; this is not a license grant')
    p.add_argument('--sequence'); p.add_argument('--fps', type=float)
    p.add_argument('--coordinate-origin', choices=['zero', 'one'])
    p.add_argument('--image-ids', help='JSON list of original image IDs')
    p.add_argument('--frame-root')
    a = p.parse_args()
    if a.kind == '300vw':
        if not a.sequence or not a.fps or not a.coordinate_origin: p.error('300vw requires sequence, source FPS and explicit coordinate origin')
        result = convert_300vw(a.input, a.sequence, a.fps, a.coordinate_origin, a.license_note)
    elif a.kind == 'wflw': result = convert_wflw(a.input, a.license_note)
    elif a.kind == 'mot-face': result = convert_mot_face(a.input, a.frame_root, a.fps, a.license_note)
    else: result = convert_coco(a.input, json.loads(Path(a.image_ids).read_text()), a.license_note)
    with Path(a.output).open('x') as f: json.dump(result, f, indent=2, allow_nan=False)


if __name__ == '__main__': main()
