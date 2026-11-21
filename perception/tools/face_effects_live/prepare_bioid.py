"""Import official BioID images/annotations without predictions or label repair.

Permission is purpose-limited: official BioID face-detection comparison research.
Keep raw data outside Git and do not redistribute face images in reports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

from dataset_adapters import base, envelope, sha256


def points20(data):
    text = data.decode('ascii')
    match = re.fullmatch(r'\s*version:\s*1\s+n_points:\s*20\s*\{([^}]*)\}\s*', text)
    if not match:
        raise ValueError('expected 20 original FGnet points')
    points = np.array([float(x) for x in match[1].split()])
    if points.size != 40 or not np.isfinite(points).all():
        raise ValueError('invalid FGnet coordinates')
    return points.reshape(20, 2).tolist()


def eye_points(data):
    rows = data.decode('ascii').splitlines()
    values = [float(x) for line in rows if line and not line.startswith('#') for x in line.split()]
    if len(values) != 4 or not np.isfinite(values).all():
        raise ValueError('expected LX LY RX RY')
    # Official .eye laterality is subject-left/right, hence reversed in the image.
    return [values[2:], values[:2]]


def prepare(root):
    root = Path(root); source = root/'source'
    output = root/'images'; output.mkdir(exist_ok=False)
    eyes_dir = root/'annotations-eye'; eyes_dir.mkdir(exist_ok=False)
    dense_dir = root/'annotations-20'; dense_dir.mkdir(exist_ok=False)
    note = ('BioID official purpose-limited permission: compare face detection algorithms in research; '
            'https://www.bioid.com/face-database/ . Not a general commercial-product/data redistribution license.')
    labels = base('BioID-FGnet20-v1.2', source/'points20.zip', note)
    labels['provenance'].update(
        source_url='https://www.bioid.com/face-database/',
        manual_annotation_statement='official legend: 20 manually placed points; official .eye: manually set eye positions',
        mapping={'image_left_eye_pupil':0, 'image_right_eye_pupil':1, 'nose_tip':14,
                 'image_left_mouth_corner':2, 'image_right_mouth_corner':3},
        coordinate_origin='source coordinates retained verbatim, no offset or fit; origin not explicitly documented on official page',
        limits=['pupils differ semantically from OSF eye-contour means',
                'no visibility/occlusion flags or 3D pose truth supplied',
                'static images, no timestamps/temporal identities',
                'no published per-point annotator uncertainty; coordinate convention may add systematic pixel error'])
    manifest = {'source_urls':{
        'images.zip':'https://www.bioid.com/uploads/BioID-FaceDatabase-V1.2.zip',
        'eyes.zip':'https://www.bioid.com/wp-content/uploads/BioID-FD-Eyepos-V1.2.zip',
        'points20.zip':'https://www.bioid.com/wp-content/uploads/BioID_pts.zip'},
        'license_note':note, 'archives':{}, 'images':[], 'unpaired_annotation_files':[]}
    with zipfile.ZipFile(source/'images.zip') as images, zipfile.ZipFile(source/'eyes.zip') as eyes, zipfile.ZipFile(source/'points20.zip') as dense:
        for name,z in [('images.zip',images),('eyes.zip',eyes),('points20.zip',dense)]:
            if z.testzip() is not None: raise ValueError('ZIP CRC failure')
            manifest['archives'][name] = {'size':(source/name).stat().st_size, 'sha256':sha256(source/name), 'crc':'all entries passed'}
        # Retain every original annotation, including the unmatched official 1140a alternative.
        for member in dense.namelist():
            if member.endswith('.pts'):
                (dense_dir/Path(member).name).write_bytes(dense.read(member))
        expected = {f'points_20/bioid_{i:04}.pts' for i in range(1521)}
        manifest['unpaired_annotation_files'] = sorted(set(dense.namelist())-expected-{'points_20/'})
        (root/'description.txt').write_bytes(images.read('description.txt'))
        for index in range(1521):
            stem = f'BioID_{index:04}'; filename = stem+'.pgm'
            data = images.read(filename)
            image = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape != (286,384,3): raise ValueError('unexpected image dimensions')
            points = points20(dense.read(f'points_20/bioid_{index:04}.pts'))
            eye_data = eyes.read(stem+'.eye')
            if eye_data != images.read(stem+'.eye'): raise ValueError('official eye archives disagree')
            official_eyes = eye_points(eye_data)
            (eyes_dir/(stem+'.eye')).write_bytes(eye_data); (output/filename).write_bytes(data)
            anchors = [points[i] for i in (0,1,14,2,3)]
            for a,b in ((0,1),(3,4)):
                if anchors[a][0] > anchors[b][0]: anchors[a],anchors[b] = anchors[b],anchors[a]
            clip = stem.lower()
            labels['clips'][clip] = [{'frame_id':index,'timestamp':None,'image':filename,'faces':[{
                'id':stem,'box':envelope(points),'box_kind':'landmark_envelope_not_detector_box',
                'points':anchors,'dense_points':points,'topology':'bioid20','visibility':None,
                'attributes':{},'bioid_official_eyes':official_eyes}]}]
            labels['sequence_metadata'][clip] = {'temporal':False,'identity_labels':False,'exhaustive_face_boxes':False,
                'limits':'one labeled frontal face, no exhaustive background boxes; no video/3D GT'}
            manifest['images'].append({'file':filename,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
                'shape':[384,286],'pts_sha256':sha256(dense_dir/f'bioid_{index:04}.pts'),
                'eye_sha256':hashlib.sha256(eye_data).hexdigest()})
    (root/'labels.json').write_text(json.dumps(labels,indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'frames':len(labels['clips']),'manual_points':20,'archives':manifest['archives'],
                      'unpaired_annotations':manifest['unpaired_annotation_files']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    prepare(parser.parse_args().root)
