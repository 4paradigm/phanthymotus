"""Decision-core configuration adapter; called off the asyncio event loop."""
from music.contracts import https_url, MusicError


def save_settings(arguments):
    import config
    kind = arguments['music_type']
    if kind not in ('none', 'mock', 'motus_music'):
        raise ValueError('未知音乐曲库 provider。')
    desktop = config.main.get('desktop_tools', {})
    music = dict(desktop.get('music', {}))
    music['type'] = kind
    for field in ('endpoint', 'api_key'):
        value = arguments.get('music_' + field)
        if value is not None and value != '****':
            music[field] = str(value).strip()
    timeout = arguments.get('music_timeout_ms', music.get('timeout_ms', 3000))
    if type(timeout) is not int or not 100 <= timeout <= 30000:
        raise ValueError('音乐曲库超时须为 100–30000 毫秒。')
    music['timeout_ms'] = timeout
    if kind == 'motus_music':
        try:
            https_url(music.get('endpoint', ''))
        except MusicError as exc:
            raise ValueError(str(exc)) from None
        if '?' in music['endpoint']:
            raise ValueError('曲库 endpoint 不能包含 query。')
        if not music.get('api_key') or any(c in music['api_key'] for c in ('\r', '\n')):
            raise ValueError('请填写音乐曲库 API Key。')
    desktop['music'] = music
    config.main['desktop_tools'] = desktop
