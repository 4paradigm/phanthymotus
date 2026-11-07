from copy import deepcopy
import struct

from plugins.pointcloud_slam_view import SavedMapView


def test_saved_map_publishes_original_packet_with_historical_identity_and_stops():
    packet = struct.pack('<fffBIfff', -1, 2, .5, 3, 1, 4, -5, 6)
    item = {'segment_id': 'old/run', 'frame_id': 'old_map', 'packet': packet,
            'metadata': {'backend': 'legacy', 'tracking': False, 'map_plane': 'xy'}}
    original = deepcopy(item)
    view = SavedMapView('a' * 32, 2, 'Hall', [item], interval_s=1)
    try:
        view.start()
        output = view.outputs()[0]
        assert output['format'] == 'sensor/mapping'
        context = output['metadata']['map_view']
        assert context['scope'] == 'history' and context['revision'] == 2
        assert context['segment_id'] == 'old/run' and context['frame_id'] == 'old_map'
        assert view.publishers[0].messages[-1] == packet
        view.rename('Renamed')
        assert view.outputs()[0]['metadata']['map_view']['name'] == 'Renamed'
        count = len(view.publishers[0].messages)
        view.close()
        view._publish()
        assert len(view.publishers[0].messages) == count
        assert item == original
    finally:
        view.close()


def test_missing_snapshot_creates_no_fake_origin_output():
    view = SavedMapView('b' * 32, 1, 'No preview', [], interval_s=1)
    try:
        view.start()
        assert view.outputs() == [] and view.publishers == []
    finally:
        view.close()
