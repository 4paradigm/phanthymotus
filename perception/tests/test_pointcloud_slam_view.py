from copy import deepcopy
import struct

from plugins.pointcloud_slam_view import SavedMapView
from plugins.pointcloud_slam_preview import PREVIEW_TOPIC


def test_saved_map_publishes_original_packet_with_historical_identity_and_stops():
    packet = struct.pack('<fffBIfff', -1, 2, .5, 3, 1, 4, -5, 6)
    item = {'segment_id': 'old/run', 'frame_id': 'old_map', 'packet': packet,
            'metadata': {'backend': 'legacy', 'tracking': False, 'map_plane': 'xy'}}
    original = deepcopy(item)
    view = SavedMapView('a' * 32, 2, 'Hall', [item], interval_s=1)
    try:
        view.start()
        output = view.outputs()[0]
        assert output['format'] == 'sensor/mapping' and output['topic'] == PREVIEW_TOPIC
        context = output['metadata']['map_view']
        assert context['scope'] == 'history' and context['revision'] == 2
        assert context['segment_id'] == 'old/run' and context['frame_id'] == 'old_map'
        assert not context.get('actions')
        assert 'plane' not in context
        assert len(view.publishers) == 1 and view.publishers[0].topic == PREVIEW_TOPIC
        assert view.publishers[0].messages[-1] == packet[:12] + bytes([39]) + packet[13:]
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

def test_multiple_disconnected_segments_use_latest_only_and_one_publisher():
    first = struct.pack('<fffBIfff', 1, 0, 0, 3, 1, 1, 2, 3)
    latest = struct.pack('<fffBIfff', 9, 0, 0, 3, 1, 4, 5, 6)
    snapshots = [
        {'segment_id': 'a', 'frame_id': 'a_map', 'packet': first, 'metadata': {}},
        {'segment_id': 'b', 'frame_id': 'b_map', 'packet': latest, 'metadata': {}},
    ]
    view = SavedMapView('c' * 32, 1, 'Segments', snapshots)
    try:
        view.start()
        assert view.segment_count == 2
        assert view.selected_segment_id == 'b'
        assert len(view.publishers) == 1
        assert view.outputs()[0]['topic'] == PREVIEW_TOPIC
        assert view.outputs()[0]['metadata']['map_view']['segment_count'] == 2
        assert view.publishers[0].messages[-1] == latest[:12] + bytes([39]) + latest[13:]
        assert first != latest  # never combine unrelated segment frames
    finally:
        view.close()
