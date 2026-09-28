"""Validation shared by the real and metadata-only mock motus.music/1 providers."""
from copy import deepcopy
from urllib.parse import urlsplit

SCHEMA = 'motus.music/1'
# Only codes explicitly specified in the integration contract are fallback values.
# Other genres come from capabilities; do not guess codes from translated labels.
GENRES = ('pop', 'rnb', 'electronic', 'gufeng', 'rock', 'folk', 'rap', 'jazz', 'country')
LANGUAGES = ('zh', 'en', 'ja', 'ko', 'th', 'pt', 'fr')
VOCALS = ('male', 'female', 'mixed', 'instrumental')
FALLBACK = {
    'schema': SCHEMA, 'provider': 'unavailable', 'limits': {'max_top_k': 10},
    'audio': {'url_ttl_s': None}, 'features': [],
    'filters': {name: [{'value': v, 'label': v} for v in values]
                for name, values in (('genre', GENRES), ('language', LANGUAGES), ('vocal', VOCALS))},
}
FALLBACK['filters']['tags'] = {}


class MusicError(Exception):
    def __init__(self, code, message, *, request_id='', retry_after=None, allowed=None):
        super().__init__(message)
        self.code, self.request_id = code, request_id
        self.retry_after, self.allowed = retry_after, allowed

    def result(self):
        error = {'code': self.code, 'message': str(self)}
        if self.retry_after is not None:
            error['retry_after_s'] = self.retry_after
        if self.allowed is not None:
            error['allowed'] = self.allowed
        return {'schema': SCHEMA, 'request_id': self.request_id, 'error': error}


def https_url(value):
    if not isinstance(value, str):
        raise MusicError('invalid_url', 'HTTPS 地址必须是字符串。')
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == 'https' and parsed.hostname and not parsed.username
                 and not parsed.password and not parsed.fragment)
        _ = parsed.port
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise MusicError('invalid_url', '需要不含用户名、密码或 fragment 的 HTTPS 地址。')
    return value


def envelope(data):
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise MusicError('schema_mismatch', '曲库返回的规范版本不是 motus.music/1。',
                         request_id=str(data.get('request_id', ''))[:200] if isinstance(data, dict) else '')
    return data


def capabilities(data):
    envelope(data)
    out = deepcopy(FALLBACK)
    if not isinstance(data.get('filters'), dict):
        raise MusicError('invalid_response', '曲库能力响应缺少 filters。')
    if not isinstance(data.get('limits', {}), dict):
        raise MusicError('invalid_response', '曲库 limits 无效。')
    maximum = data.get('limits', {}).get('max_top_k', 10)
    if type(maximum) is not int or not 1 <= maximum <= 10:
        raise MusicError('invalid_response', '曲库 max_top_k 无效。')
    out.update({k: data[k] for k in ('provider', 'audio', 'features') if k in data})
    out['limits']['max_top_k'] = maximum
    for name in ('genre', 'language', 'vocal'):
        options = data['filters'].get(name, out['filters'][name])
        if (not isinstance(options, list) or not options or len(options) > 100
                or any(not isinstance(x, dict) or not isinstance(x.get('value'), str)
                       or not x['value'] or len(x['value']) > 80 for x in options)):
            raise MusicError('invalid_response', '曲库筛选词表无效。')
        out['filters'][name] = options
    tags = data['filters'].get('tags', {})
    if not isinstance(tags, dict):
        raise MusicError('invalid_response', '曲库标签词表无效。')
    out['filters']['tags'] = {k: [v[:80] for v in values[:100] if isinstance(v, str)]
                             for k, values in tags.items() if isinstance(values, list)}
    return out


def search_request(query='', genre='', language='', vocal='', tags='', artist='',
                   top_k=3, exclude_ids='', duration_min_s=0, duration_max_s=0, caps=None):
    caps = caps or FALLBACK
    for value in (query, genre, language, vocal, tags, artist, exclude_ids):
        if not isinstance(value, str):
            raise MusicError('invalid_argument', '检索文本和逗号分隔列表必须是字符串。')
    if len(query) > 200 or len(artist) > 200 or len(tags) > 2000:
        raise MusicError('invalid_argument', 'query/artist 最长 200 字，tags 最长 2000 字。')
    if type(top_k) is not int or not 1 <= top_k <= caps['limits']['max_top_k']:
        raise MusicError('invalid_argument', f"top_k 必须为 1–{caps['limits']['max_top_k']}。")
    for value in (duration_min_s, duration_max_s):
        if type(value) is not int or value < 0:
            raise MusicError('invalid_argument', '时长必须为非负整数秒，0 表示不限制。')
    if duration_max_s and duration_min_s > duration_max_s:
        raise MusicError('invalid_argument', '最短时长不能大于最长时长。')
    def split(value):
        return list(dict.fromkeys(x.strip() for x in value.split(',') if x.strip()))
    excluded = split(exclude_ids)
    if len(excluded) > 50 or any(len(x) > 200 for x in excluded):
        raise MusicError('invalid_argument', 'exclude_ids 最多 50 个，每个 ID 最长 200 字。')
    filters = {}
    for field, value in (('genre', genre), ('language', language), ('vocal', vocal)):
        values = split(value)
        allowed = [x['value'] for x in caps['filters'][field]]
        if any(x not in allowed for x in values) or (field == 'vocal' and len(values) > 1):
            raise MusicError('invalid_filter', f'{field} 的取值不合法。', allowed=allowed)
        if values:
            filters[field] = values[0] if field == 'vocal' else values
    if tags:
        filters['tags'] = split(tags)  # Unknown tags are intentionally not rejected.
    if artist:
        filters['artist'] = artist.strip()
    duration = {}
    if duration_min_s:
        duration['min'] = duration_min_s
    if duration_max_s:
        duration['max'] = duration_max_s
    if duration:
        filters['duration_s'] = duration
    return {'schema': SCHEMA, 'query': query, 'filters': filters,
            'top_k': top_k, 'exclude_ids': excluded}


def search_result(data, request, *, mock=False):
    envelope(data)
    tracks, relaxed = data.get('tracks'), data.get('relaxed', [])
    if (not isinstance(tracks, list) or len(tracks) > request['top_k']
            or not isinstance(relaxed, list)
            or any(x not in ('query', 'duration_s', 'vocal', 'genre', 'artist') for x in relaxed)):
        raise MusicError('invalid_response', '曲库返回了无效的结果数量或放宽条件。')
    seen = set()
    for track in tracks:
        if not isinstance(track, dict) or not isinstance(track.get('id'), str) or not track['id']:
            raise MusicError('invalid_response', '曲库歌曲缺少 ID。')
        if track['id'] in request['exclude_ids'] or track['id'] in seen:
            raise MusicError('invalid_response', '曲库返回了重复或已排除的歌曲。')
        seen.add(track['id'])
        if not isinstance(track.get('artist'), dict):
            raise MusicError('invalid_response', '曲库歌曲缺少有效 artist。')
        if not mock:
            audio = track.get('audio')
            if not isinstance(audio, dict):
                raise MusicError('invalid_response', '曲库歌曲缺少有效 audio。')
            https_url(audio.get('url', ''))
        language = request['filters'].get('language')
        if language and track.get('language') not in language:
            raise MusicError('invalid_response', '曲库违反了不可放宽的语种条件。')
        filters = request['filters']
        for field in ('genre', 'vocal'):
            wanted = filters.get(field)
            if wanted and field not in relaxed:
                if track.get(field) not in (wanted if isinstance(wanted, list) else [wanted]):
                    raise MusicError('invalid_response', f'曲库违反了未放宽的 {field} 条件。')
        artist = filters.get('artist')
        if artist and 'artist' not in relaxed and artist not in (track.get('artist') or {}).values():
            raise MusicError('invalid_response', '曲库违反了未放宽的 artist 条件。')
        duration = filters.get('duration_s')
        if duration and 'duration_s' not in relaxed:
            value = track.get('duration_s')
            if not isinstance(value, (int, float)) or not duration.get('min', 0) <= value <= duration.get('max', float('inf')):
                raise MusicError('invalid_response', '曲库违反了未放宽的 duration_s 条件。')
    return data
