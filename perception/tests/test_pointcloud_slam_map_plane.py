import json

import pytest

from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
from test_pointcloud_slam_library import saved
from vision_stubs import _FakeExecutor


@pytest.mark.parametrize('declared',[False,True])
def test_tag_action_requires_archived_segment_plane(tmp_path,saved,monkeypatch,declared):
    import plugins.pointcloud_slam_plugin as module
    map_id,_,cfg=saved
    monkeypatch.setattr(module,'_ARCHIVE_DIR',str(tmp_path))
    if declared:
        path=tmp_path/'maps'/map_id/'1.json'
        data=json.loads(path.read_text())
        data['resolved']['segment_metadata']['a']['map_plane']='xy'
        path.write_text(json.dumps(data))
    plugin=PointCloudSlamPlugin({'persistent_history':cfg['persistent_history']},_FakeExecutor())
    result=plugin.dispatch('pointcloud_slam',dict(action='tag_place',map_id=map_id,
        revision=1,segment_id='a',tag_name='地点',x=1,y=2))
    assert result['state']==('ok' if declared else 'error'),result
    if not declared:
        assert 'plane' in result['message']
