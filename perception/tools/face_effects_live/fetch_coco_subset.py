"""Fetch deterministic CC BY 2.0 COCO val2017 images, never a mirror/relabel.

Requires the official person_keypoints_val2017.json. Image choice uses ONLY
original annotations/license, before any model runs. Keeps attribution URLs,
licenses, original IDs, image hashes and all selection/exclusion counts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from dataset_adapters import coco_face, convert_coco, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', required=True); p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=240)
    a = p.parse_args(); out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False); (out/'images').mkdir()
    data = json.loads(Path(a.annotations).read_text()); eligible = set()
    images = {i['id']: i for i in data['images']}
    for ann in data['annotations']:
        if ann['category_id'] != 1 or ann.get('iscrowd'): continue
        image = images[ann['image_id']]
        if image['license'] != 4: continue  # original image CC BY 2.0, no NC/ND
        face = coco_face(ann)
        if face is not None and any(v is not None for v in face['points']):
            eligible.add(ann['image_id'])
    selected = sorted(eligible, key=lambda n: hashlib.sha256(('FEL4:'+str(n)).encode()).digest())[:a.limit]
    (out/'image-ids.json').write_text(json.dumps(selected)+'\n')
    labels = convert_coco(a.annotations, selected, 'COCO Consortium annotations CC BY 4.0; selected images CC BY 2.0')
    (out/'labels.json').write_text(json.dumps(labels, indent=2, allow_nan=False)+'\n')
    def fetch(image_id):
        image = images[image_id]
        url = 'https://s3.amazonaws.com/images.cocodataset.org/val2017/'+image['file_name']
        with urllib.request.urlopen(url, timeout=60) as r: content = r.read(20_000_000)
        if not content.startswith(b'\xff\xd8') or len(content) >= 20_000_000:
            raise ValueError('not a bounded original JPEG')
        (out/'images'/image['file_name']).write_bytes(content)
        photo = re.search(r'/(\d+)_', image['flickr_url'])
        return {'image_id': image_id, 'file': image['file_name'], 'url': url,
                'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                'attribution_source': image['flickr_url'],
                'original_photo_page': 'https://www.flickr.com/photo.gne?id='+photo[1] if photo else None,
                'image_license': 'https://creativecommons.org/licenses/by/2.0/'}
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch, selected):
            rows.append(row)
            if len(rows) % 40 == 0: print('downloaded', len(rows), flush=True)
    manifest = {'schema': 'face_effects_coco_subset/1', 'annotations_sha256': sha256(a.annotations),
                'selection': 'CC BY 2.0 images with a nondegenerate labeled head region and visible eye pair or nose; SHA256(FEL4:image_id) order',
                'eligible_images': len(eligible), 'selected_images': len(selected),
                'selection_used_predictions': False, 'images': rows,
                'labels_sha256': sha256(out/'labels.json'),
                'attribution': 'COCO Consortium: Common Objects in Context, Lin et al., ECCV 2014; keypoints human annotations, COCO 2017 validation split',
                'terms': 'https://cocodataset.org/#termsofuse',
                'limits': 'image eye/nose proxies, incomplete face labels; not temporal GT or dense face benchmark'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('complete', len(rows), sum(r['bytes'] for r in rows), flush=True)


if __name__ == '__main__': main()
