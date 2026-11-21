"""Synthetic format/metric oracles only; none of these fixtures are quality GT."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[2]/'tools/face_effects_live'
sys.path.insert(0, str(TOOLS))
import dataset_adapters as adapters
import score_dataset as metrics


def test_pts_numbering_origin_missing_and_original_frame_time(tmp_path):
    folder = tmp_path/'001/annot'; folder.mkdir(parents=True)
    points = '\n'.join(f'{i+1} {2*i+1}' for i in range(68))
    for number in (1, 3):
        (folder/f'{number:06}.pts').write_text('version: 1\nn_points: 68\n{\n'+points+'\n}\n')
    labels = adapters.convert_300vw(tmp_path, '001', 25., 'one', 'SYNTHETIC_UNIT_TEST')
    a, b = labels['clips']['001']
    assert (a['frame_id'], a['decode_index'], a['timestamp']) == (1, 0, 0.)
    assert (b['frame_id'], b['decode_index'], b['timestamp']) == (3, 2, .08)
    assert a['faces'][0]['dense_points'][30] == [30., 60.]
    assert a['faces'][0]['points'][0] == [38.5, 77.]
    assert a['faces'][0]['points'][3] == [48., 96.]  # iBUG, NOT OSF66 58
    assert not labels['sequence_metadata']['001']['identity_labels']
    assert not labels['independent_human_review']


def test_pts_missing_contour_not_replaced_by_partial_mean(tmp_path):
    path = tmp_path/'one.pts'
    points = ['1 2']*68; points[36] = 'nan nan'; points[30] = '-500 4000'
    path.write_text('version: 1\nn_points: 68\n{\n'+'\n'.join(points)+'\n}\n')
    dense = adapters.read_pts(path, 'zero'); semantic = adapters.semantic_points(dense, 'ibug68')
    assert dense[36] is None and semantic[0] is None
    assert semantic[2] == [-500., 4000.]  # never clamp/repair original truth
    with pytest.raises(ValueError): adapters.read_pts(path, None)
    path.write_text(path.read_text().replace('n_points: 68', 'n_points: 67'))
    with pytest.raises(ValueError): adapters.read_pts(path, 'zero')


def test_wflw_semantics_attributes_and_no_invented_visibility(tmp_path):
    points = np.full((98, 2), 999.)
    points[60:68] = [10, 20]; points[68:76] = [50, 20]
    points[54] = [30, 40]; points[76] = [15, 60]; points[82] = [45, 60]
    points[96:98] = [[11, 21], [49, 21]]  # pupils are NOT contour centers
    path = tmp_path/'wflw.txt'
    tokens = [*map(str, points.ravel()), '0', '0', '100', '100', '1', '0', '1', '0', '1', '1', 'folder/image.jpg']
    path.write_text(' '.join(tokens)+'\n')
    labels = adapters.convert_wflw(path, 'SYNTHETIC_UNIT_TEST'); face = labels['clips']['wflw_0'][0]['faces'][0]
    assert face['points'] == [[10, 20], [50, 20], [30, 40], [15, 60], [45, 60]]
    assert face['attributes'] == dict(pose=1, expression=0, illumination=1, makeup=0, occlusion=1, blur=1)
    assert face['visibility'] is None and len(face['dense_points']) == 98
    path.write_text(' '.join(tokens[:-1]+['../private.jpg']))
    with pytest.raises(ValueError): adapters.convert_wflw(path, 'unit')


def test_coco_visibility_and_eye_laterality_are_not_guessed():
    p = np.zeros((17, 3)); p[0] = [20, 30, 2]; p[1] = [30, 20, 2]; p[2] = [10, 20, 2]
    ann = dict(id=7, keypoints=p.ravel().tolist(), bbox=[0, 0, 40, 100])
    face = adapters.coco_face(ann)
    assert face['points'] == [[10, 20], [30, 20], [20, 30], None, None]
    p[2] = [10, 20, 1]; ann['keypoints'] = p.ravel().tolist(); face = adapters.coco_face(ann)
    assert face['points'][:2] == [None, None]
    assert face['dense_points'][2] == [10, 20] and face['visibility'][2] == 1
    p[2] = [123, 456, 0]; ann['keypoints'] = p.ravel().tolist()
    assert adapters.coco_face(ann)['dense_points'][2] is None
    p[2, 2] = 3; ann['keypoints'] = p.ravel().tolist()
    with pytest.raises(ValueError): adapters.coco_face(ann)


@pytest.mark.parametrize('path', ['../x', '/tmp/private', 'a/../../x'])
def test_annotation_paths_cannot_escape(path):
    with pytest.raises(ValueError): adapters.relative_path(path)


def fixture_labels(faces, *, full=False, temporal=False, identity=False):
    return {'dataset': 'SYNTHETIC_UNIT_TEST', 'provenance': {'license': 'test fixture'},
            'clips': {'clip': [{'frame_id': 1, 'timestamp': 0. if temporal else None, 'faces': faces}]},
            'sequence_metadata': {'clip': {'temporal': temporal, 'identity_labels': identity, 'exhaustive_face_boxes': full}}}


def face(identity='a', x=0):
    return dict(id=identity, box=[x, 0, 30, 40], points=[[x+5, 10], [x+25, 10], [x+15, 20], None, None], attributes={})


def row(faces, frame_id=1, tracks=None):
    tracks = tracks or {str(i+1): f for i, f in enumerate(faces)}
    return {'frame_id': frame_id, 'detections': faces, 'detector_observations': faces,
            'boxes': {k: f['box'] for k, f in tracks.items()}, 'anchors': {k: f['points'] for k, f in tracks.items()}}


def write_rows(tmp_path, rows):
    (tmp_path/'clip.jsonl').write_text('\n'.join(map(json.dumps, rows))+'\n')


def prediction(f):
    return {**f, 'points': [p if p is not None else [0, 0] for p in f['points']]}


def test_one_prediction_is_never_used_for_two_faces_and_partial_labels_have_no_fp(tmp_path):
    f = face(); labels = fixture_labels([f, face('b')])
    write_rows(tmp_path, [row([prediction(f)])]); result = metrics.score(tmp_path, labels)
    assert result['overall']['refined_matched'] == 1
    assert result['overall']['refined_proxy_missing'] == 1
    assert result['overall']['false_positives'] is None
    assert result['mot'] is None and result['temporal'] is None


def test_normalization_uses_truth_eye_distance_and_reports_missing(tmp_path):
    f = face(); actual = prediction(f)
    actual['points'] = [[p[0]+2, p[1]] for p in actual['points']]
    write_rows(tmp_path, [row([actual])]); result = metrics.score(tmp_path, fixture_labels([f]))['overall']
    assert result['raw_face_nme_interocular']['mean'] == pytest.approx(.1)
    assert result['raw_face_mae_px']['mean'] == pytest.approx(2.)
    assert result['nme_failure_including_missing_rate'] == 0.
    write_rows(tmp_path, [row([])]); result = metrics.score(tmp_path, fixture_labels([f]))['overall']
    assert result['raw_face_nme_interocular'] is None and result['nme_failure_including_missing_rate'] == 1.


def test_duplicate_predictions_rejected(tmp_path):
    write_rows(tmp_path, [row([]), row([])])
    with pytest.raises(ValueError, match='duplicate'): metrics.score(tmp_path, fixture_labels([]))


def test_temporal_identity_switches_and_false_positive_counts(tmp_path):
    a, b = face(), face('b', 60); labels = fixture_labels([a, b], full=True, temporal=True, identity=True)
    labels['clips']['clip'].append(dict(frame_id=2, timestamp=1/30, faces=[a, b]))
    first = row([prediction(a), prediction(b)])
    second = row([prediction(a), prediction(b)], 2, {'2': prediction(a), '1': prediction(b), '3': prediction(face('extra', 200))})
    write_rows(tmp_path, [first, second]); result = metrics.score(tmp_path, labels)['mot'][0]
    assert result['id_switches'] == 2 and result['false_positives'] == 1 and result['false_negatives'] == 0
    assert result['mota'] == pytest.approx(.25)
    labels['sequence_metadata']['clip']['temporal'] = False
    with pytest.raises(ValueError, match='MOT'): metrics.score(tmp_path, labels)


def test_mot_requires_face_provenance_and_preserves_empty_frames_and_origin(tmp_path):
    path = tmp_path/'gt.csv'; path.write_text('2,17,11,21,30,40,1,1,0.8\n3,99,101,101,20,20,0,1,0\n')
    meta = dict(dataset='SYNTHETIC_UNIT_TEST', annotation_target='person', identity_labels=True,
                exhaustive_face_boxes=True, coordinate_origin='one', source_url='https://example.invalid/unit')
    sidecar = Path(str(path)+'.provenance.json'); sidecar.write_text(json.dumps(meta))
    frames = tmp_path/'frames'; frames.mkdir()
    for n in (1, 2, 3): (frames/f'{n:06}.jpg').write_bytes(b'unit fixture, not decoded')
    with pytest.raises(ValueError): adapters.convert_mot_face(path, frames, 30., 'unit')
    meta['annotation_target'] = 'face'; sidecar.write_text(json.dumps(meta))
    result = adapters.convert_mot_face(path, frames, 30., 'unit')['clips']['mot_faces']
    assert result[0]['faces'] == [] and result[1]['faces'][0]['id'] == '17'
    assert result[1]['faces'][0]['box'] == [10, 20, 30, 40]
    assert result[2]['ignore_boxes'] == [[100, 100, 20, 20]]


def test_ignored_region_does_not_become_false_positive(tmp_path):
    labels = fixture_labels([], full=True, temporal=True, identity=True)
    labels['clips']['clip'][0]['ignore_boxes'] = [[0, 0, 100, 100]]
    write_rows(tmp_path, [row([prediction(face())])])
    assert metrics.score(tmp_path, labels)['mot'][0]['false_positives'] == 0


def test_motion_phase_known_delay_and_missing_gaps():
    trajectory = lambda n: [float(40*np.sin(n*.3)), float(n*n*.1)]
    records = [dict(frame_id=n, timestamp=n/30, truth_nose=trajectory(n), predicted_nose=trajectory(n-2)) for n in range(50)]
    result = metrics.temporal_summary({'unit': records})['unit']
    assert result['best_alignment_frames'] == 2 and result['lag_comparable_samples'] == 42
    assert result['lag_profile_rmse_px']['2'] == 0.
    result = metrics.temporal_summary({'unit': records[::2]})['unit']
    assert result['best_alignment_frames'] is None and result['first_difference_residual_rms_px'] is None


def test_upscale_is_bounded_and_maps_detector_coordinates_back(monkeypatch, tmp_path):
    from plugins.face_effects_live.backends import YuNet
    from plugins.face_effects_live.plugin import validate_config
    model = tmp_path/'unit.onnx'; model.write_bytes(b'x'*2048)
    class Detector:
        def setInputSize(self, size): self.size = size
        def detect(self, image):
            return None, np.array([[64, 96, 128, 160, 80, 120, 160, 120, 120, 160, 90, 200, 150, 200, .9]], float)
    detector = Detector()
    import cv2
    monkeypatch.setattr(cv2.FaceDetectorYN, 'create', lambda *a: detector)
    backend = YuNet(model, max_side=1920, upscale=True)
    actual = backend.detect(np.zeros((240, 320, 3), np.uint8))[0]
    assert detector.size == (640, 480) and actual.box.tolist() == [32, 48, 64, 80]
    assert actual.points[2].tolist() == [60, 80]
    validate_config(dict(detect_side=960, detect_upscale=True))
    for bad in ('yes', 1, None):
        with pytest.raises(ValueError): validate_config({'detect_upscale': bad})
        with pytest.raises(ValueError): YuNet(model, upscale=bad)


def test_compact_heatmap_decode_has_independent_grid_oracle():
    from plugins.face_effects_live.osf_decode import decode_heatmaps
    output = np.zeros((1, 198, 28, 28), np.float32)
    output[:, 66:] = .5  # logit(.5)==0, so the answer is the grid coordinate
    output[0, :66, 7, 13] = .9
    coordinates, quality = decode_heatmaps([output])
    assert np.allclose(coordinates, [13*223/27, 7*223/27])
    assert np.allclose(quality, .9)
    assert coordinates.nbytes+quality.nbytes < 2000
    output[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError): decode_heatmaps([output])


def test_frozen_benchmark_refuses_mismatched_worker_decoder(tmp_path):
    from natural_benchmark import load_package
    (tmp_path/'osf_decode.py').write_text('# intentionally different frozen decoder')
    with pytest.raises(ValueError, match='matching source revisions'):
        load_package(tmp_path, 'must_not_import_mixed_sources')
