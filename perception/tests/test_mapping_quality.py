import numpy as np
from plugins.spatial_quality import MapGate, QualityReport
from plugins.spatial_time import TimePolicy
from test_spatial_quality import sample


def test_bad_cloud_dropped_while_valid_pose_continues():
    s,c=sample();before=s['pose'].copy()
    assert MapGate().evaluate(s,np.array([[np.nan,0,0]]),c,None).action=='drop'
    assert s['pose']==before


def test_numeric_pose_is_not_rejected_by_status_labels_but_missing_pose_waits():
    s,c=sample();s['vendor']['motus_slam']['status']='coasting'
    assert MapGate().evaluate(s,np.ones((2,3)),c,None).action=='accept'
    s['pose']=None
    assert MapGate().evaluate(s,np.ones((2,3)),c,None).action=='pause'


def test_lidar_not_subject_to_image_depth_fraction_and_unknown_quality_allowed():
    s,c=sample()
    for q in (None,QualityReport(depth_fraction=0)):
        assert MapGate(observation_model='lidar').evaluate(s,np.ones((2,3)),c,q).action=='accept'


def test_receipt_metadata_does_not_pause_numeric_mapping():
    s,c=sample(receipt=True)
    assert MapGate(dynamic_mount=True).evaluate(s,np.ones((2,3)),c,None).action=='accept'
