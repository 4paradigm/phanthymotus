"""Synthetic metric correctness cases; these are not natural quality results."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def tool(name):
    path=Path(__file__).resolve().parents[2]/'tools/face_effects_live'/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def test_occluded_motion_truth_is_missing_instead_of_fabricated_zero(tmp_path):
    rows=[]
    for k in range(30):
        points=[[float(k),10.]]*5
        rows.append({'frame_id':k,'anchors':{'1':points},'raw_anchors':{'1':points}})
    (tmp_path/'clip.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    labels={'nose_burst':{'clip':'clip','frame_ids':list(range(30)),
                         'nose_xy':[None]*30,'uncertainty_px':2}}
    result=tool('score_motion').score(tmp_path,labels)
    assert not result['valid'] and result['best_lag_ms'] is None
    assert result['stabilized_first_difference_residual_rms_px'] is None
    json.dumps(result,allow_nan=False)


def test_motion_uses_explicit_frame_ids_and_does_not_bridge_unlabeled_time(tmp_path):
    rows=[]
    for k in range(30):
        points=[[float(k),10.]]*5
        rows.append({'frame_id':2*k+100,'anchors':{'1':points},'raw_anchors':{'1':points}})
    (tmp_path/'clip.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    labels={'nose_burst':{'clip':'clip','frame_ids':list(range(100,160,2)),
                         'nose_xy':[[float(k),10.] for k in range(30)],'uncertainty_px':0}}
    result=tool('score_motion').score(tmp_path,labels)
    assert not result['lag_identifiable']
    assert result['raw_first_difference_residual_rms_px'] is None


def test_one_prediction_cannot_match_two_annotated_faces(tmp_path):
    points=[[20.,20.]]*5
    row={'frame_id':0,'detections':[{'box':[0,0,40,40],'points':points}],
         'boxes':{'1':[0,0,40,40]},'anchors':{'1':points}}
    (tmp_path/'clip.jsonl').write_text(json.dumps(row)+'\n')
    labels={'annotator':'SYNTHETIC_UNIT_TEST','independent_human_review':False,
            'clips':{'clip':[{'frame_id':0,'faces':[{'id':name,'box':[0,0,40,40],'points':points} for name in ['a','b']]}]}}
    result=tool('score_natural').score(tmp_path,labels)
    assert result['clips']['clip']['matched']==1 and result['clips']['clip']['detection_missing']==1
    assert not result['annotation_status']['independent_human_review']


@pytest.mark.parametrize('box,index',[([0,0,0,20],0),([0,0,20,20],'0')])
def test_invalid_labels_are_rejected_instead_of_counted_as_model_errors(tmp_path,box,index):
    labels={'clips':{'clip':[{'frame_id':index,'faces':[{'id':'a','box':box,'points':[None]*5}]}]}}
    with pytest.raises(ValueError):tool('score_natural').score(tmp_path,labels)
