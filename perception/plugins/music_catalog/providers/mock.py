"""Fictional metadata only. No real songs, audio URLs or network requests."""
from copy import deepcopy

from plugins.music_catalog.contracts import FALLBACK, MusicError, SCHEMA

TRACKS = [
    {'id': f'mock-{i:02}', 'title': f'虚构样例 {i:02}',
     'artist': {'id': 'mock-artist', 'name': '示例虚拟歌手'},
     'genre': 'pop' if i % 2 else 'folk', 'language': 'zh' if i % 3 else 'en',
     'vocal': 'female' if i % 2 else 'male', 'tags': ['睡前', '慵懒'],
     'duration_s': 120 + i, 'audio': None, 'lyrics_excerpt': '纯元数据测试样例，无音频',
     'match_reason': 'mock 示例，不代表真实曲库',
     'usage': {'scope': 'personal_playback', 'attribution': '虚构样例'}}
    for i in range(1, 25)
]


class MockProvider:
    def __init__(self, config):
        pass

    def capabilities(self, **context):
        out = deepcopy(FALLBACK)
        out['provider'] = 'mock'
        out['request_id'] = 'mock-capabilities'
        out['filters']['tags'] = {'scene': ['睡前'], 'mood': ['慵懒']}
        return out

    def search(self, request, **context):
        filters = request['filters']
        def matches(t):
            for field in ('genre', 'language'):
                if filters.get(field) and t[field] not in filters[field]:
                    return False
            if filters.get('vocal') and t['vocal'] != filters['vocal']:
                return False
            if filters.get('artist') and filters['artist'] not in t['artist'].values():
                return False
            duration = filters.get('duration_s', {})
            return (duration.get('min', 0) <= t['duration_s'] <= duration.get('max', float('inf'))
                    and t['id'] not in request['exclude_ids'])
        return {'schema': SCHEMA, 'request_id': 'mock-search', 'relaxed': [],
                'mock': True, 'tracks': deepcopy([t for t in TRACKS if matches(t)][:request['top_k']])}

    def track(self, track_id, **context):
        for track in TRACKS:
            if track['id'] == track_id:
                return {'schema': SCHEMA, 'request_id': 'mock-detail', 'track': deepcopy(track), 'mock': True}
        raise MusicError('not_found', '示例歌曲不存在。')


PROVIDER = MockProvider
