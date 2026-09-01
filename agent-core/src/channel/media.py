"""
channel/media.py — 出站附件解析与校验。

LLM 通过 `channel_reply(files=[{path, caption}])` 指定要发送的容器内文件。
这里负责：

1. **路径白名单**——复用 desktop 工具的 `_ALLOWED_DIRS`（`/work`、`/tmp`）。
   没有这层校验，一句「把 /etc/shadow 发到群里」就能把机器人变成外传通道。
2. **类别推断**——按扩展名/MIME 决定走图片、视频、音频还是普通文件接口。
3. **大小校验**——各平台上限不同，超限时给出可读错误，让 LLM 知道该压缩或换方式。
"""

import mimetypes
import os
import pathlib

from channel.adapter import Attachment, KIND_AUDIO, KIND_FILE, KIND_IMAGE, KIND_VIDEO

_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
_VIDEO_EXT = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}
_AUDIO_EXT = {'.opus', '.mp3', '.wav', '.m4a', '.aac', '.ogg'}
_DEPLOY_DATA_DIR = '/opt/phanthy-motus/data'
_DEPLOY_DATA_MOUNT = '/work/resource'

# 平台上限（字节）。飞书：图片 10MB、文件 30MB。
LIMITS: dict[str, dict[str, int]] = {
    'feishu':   {KIND_IMAGE: 10 * 1024 * 1024, '_default': 30 * 1024 * 1024},
    'telegram': {KIND_IMAGE: 10 * 1024 * 1024, '_default': 50 * 1024 * 1024},
    'slack':    {'_default': 1024 * 1024 * 1024},
}


def infer_kind(path: str, mime: str = '') -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXT:
        return KIND_IMAGE
    if ext in _VIDEO_EXT:
        return KIND_VIDEO
    if ext in _AUDIO_EXT:
        return KIND_AUDIO
    if mime.startswith('image/'):
        return KIND_IMAGE
    if mime.startswith('video/'):
        return KIND_VIDEO
    if mime.startswith('audio/'):
        return KIND_FILE  # 语音需要特定编码，普通音频按文件发更可靠
    return KIND_FILE


def limit_for(platform: str, kind: str) -> int:
    table = LIMITS.get(platform, {'_default': 20 * 1024 * 1024})
    return table.get(kind, table['_default'])


def _map_runtime_path(path: str) -> str:
    """Map the deployed data volume to the path visible inside agent-core."""
    if path == _DEPLOY_DATA_DIR:
        return _DEPLOY_DATA_MOUNT
    prefix = _DEPLOY_DATA_DIR + '/'
    if path.startswith(prefix):
        suffix = path[len(prefix):]
        if '..' not in pathlib.PurePosixPath(suffix).parts:
            return f'{_DEPLOY_DATA_MOUNT}/{suffix}'
    return path


def resolve_outbound(files: list, platform: str) -> tuple[list[Attachment], list[str]]:
    """把 LLM 给的 files 参数解析为 Attachment 列表。

    files 支持 `[{'path': ..., 'caption': ...}]` 或 `['/path/a.jpg', ...]`。
    返回 (可发送的附件, 错误说明列表)——错误不抛异常，原样回报给 LLM，
    这样它能知道「哪个文件为什么没发出去」而不是以为全部成功。
    """
    from event.desktop import _check_path_allowed, _resolve_path

    out: list[Attachment] = []
    errors: list[str] = []

    for item in files or []:
        if isinstance(item, str):
            raw_path, caption = item, ''
        elif isinstance(item, dict):
            raw_path = item.get('path', '') or item.get('file_path', '')
            caption = item.get('caption', '') or ''
        else:
            errors.append(f'Invalid file entry: {item!r} (expected path string or {{path, caption}})')
            continue

        if not raw_path:
            errors.append('Invalid file entry: missing "path"')
            continue

        p = _resolve_path(_map_runtime_path(raw_path))
        err = _check_path_allowed(p)
        if err:
            errors.append(f'{raw_path}: {err}')
            continue
        if not p.exists():
            errors.append(f'{p}: file not found')
            continue
        if not p.is_file():
            errors.append(f'{p}: not a regular file')
            continue

        size = p.stat().st_size
        if size == 0:
            errors.append(f'{p}: file is empty')
            continue

        mime = mimetypes.guess_type(str(p))[0] or ''
        kind = infer_kind(str(p), mime)
        cap = limit_for(platform, kind)
        if size > cap:
            errors.append(
                f'{p}: {size / 1e6:.1f}MB exceeds the {platform} limit for {kind} '
                f'({cap / 1e6:.0f}MB). Compress or downscale it first.'
            )
            continue

        out.append(Attachment(kind=kind, path=str(p), name=p.name,
                              mime=mime, size=size, caption=caption))

    return out, errors
