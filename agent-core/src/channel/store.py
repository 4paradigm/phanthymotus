"""
channel/store.py — 入站媒体文件的持久化存储。

落盘位置：`./resource/channel_files/{channel_id}/{YYYYMMDD}/{ts}_{rand}_{name}`
`./resource` 在部署时挂载为宿主持久卷（deploy/docker-compose.yml:
`/opt/phanthy-motus/data:/work/resource`），因此容器重建后文件仍在。

给出的 `Attachment.path` 是容器内绝对路径，直接随 trigger 事件传给 LLM
（LLM 可用 Read/Bash 访问，`/work` 在 desktop 工具白名单内）。

嵌入式设备磁盘有限，`prune()` 按「保留天数 + 总字节上限」清理，启动时跑一次、
之后每次保存后节流触发。
"""

import os
import pathlib
import re
import threading
import time

from channel.adapter import Attachment

# 相对 cwd（/work）；与 config.py 的 './resource/...' 约定一致
_ROOT = pathlib.Path('./resource/channel_files')

# 保留策略
_MAX_AGE_DAYS = 14
_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB
_PRUNE_MIN_INTERVAL = 300                    # 两次 prune 最短间隔（秒）

_prune_lock = threading.Lock()
_last_prune = 0.0

_UNSAFE_CHARS = re.compile(r'[^\w.\-一-鿿]+')


def _safe_name(name: str, fallback_ext: str = '') -> str:
    """把平台给的文件名压成安全的 basename（去目录分隔符、限长）。"""
    name = os.path.basename(name or '').strip()
    name = _UNSAFE_CHARS.sub('_', name).strip('._') or 'file'
    if len(name) > 80:
        stem, ext = os.path.splitext(name)
        name = stem[:80 - len(ext)] + ext
    if fallback_ext and not os.path.splitext(name)[1]:
        name += fallback_ext
    return name


def dir_for(channel_id: str) -> pathlib.Path:
    """返回某 channel 今天的落盘目录（已创建）。"""
    day = time.strftime('%Y%m%d')
    d = _ROOT / _safe_name(channel_id) / day
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_inbound_media(path: str | pathlib.Path) -> bool:
    """该路径是否为用户从消息平台发来的附件（本模块落盘的文件）。

    这个目录下的文件只有一个来源：channel adapter 下载的入站附件。工具在把文件内容
    交给 LLM 时用它标注来源 —— 「用户上传的图」和「机器人自己截的图」不是一回事。
    """
    try:
        root = str(_ROOT.resolve())
        return str(pathlib.Path(path).resolve()).startswith(root + os.sep)
    except OSError:
        return False


def save_bytes(channel_id: str, data: bytes, *, kind: str, name: str = '',
               mime: str = '', fallback_ext: str = '') -> Attachment:
    """把入站媒体写入持久化目录，返回带绝对路径的 Attachment。"""
    d = dir_for(channel_id)
    fname = f'{int(time.time() * 1000)}_{os.urandom(3).hex()}_{_safe_name(name, fallback_ext)}'
    p = (d / fname).resolve()
    with open(p, 'wb') as f:
        f.write(data)
    maybe_prune()
    return Attachment(
        kind=kind,
        path=str(p),
        name=_safe_name(name, fallback_ext),
        mime=mime,
        size=len(data),
    )


def maybe_prune() -> None:
    """节流版 prune：距上次超过 _PRUNE_MIN_INTERVAL 才真正执行。"""
    global _last_prune
    now = time.time()
    with _prune_lock:
        if now - _last_prune < _PRUNE_MIN_INTERVAL:
            return
        _last_prune = now
    try:
        prune()
    except Exception as e:
        print(f'[channel/store] prune failed: {e}')


def prune(max_age_days: int = _MAX_AGE_DAYS,
          max_total_bytes: int = _MAX_TOTAL_BYTES) -> dict:
    """删除过期文件；若总量仍超限，按最旧优先继续删。返回统计信息。"""
    if not _ROOT.exists():
        return {'deleted': 0, 'freed': 0, 'remaining': 0}

    files = []
    for p in _ROOT.rglob('*'):
        if p.is_file():
            try:
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, p))

    deleted = freed = 0
    cutoff = time.time() - max_age_days * 86400
    kept = []
    for mtime, size, p in files:
        if mtime < cutoff:
            try:
                p.unlink()
                deleted += 1
                freed += size
                continue
            except OSError:
                pass
        kept.append((mtime, size, p))

    total = sum(s for _, s, _ in kept)
    if total > max_total_bytes:
        kept.sort(key=lambda t: t[0])  # 最旧优先
        for mtime, size, p in kept:
            if total <= max_total_bytes:
                break
            try:
                p.unlink()
                deleted += 1
                freed += size
                total -= size
            except OSError:
                pass

    # 清理空目录
    for d in sorted((p for p in _ROOT.rglob('*') if p.is_dir()),
                    key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass

    if deleted:
        print(f'[channel/store] pruned {deleted} files, freed {freed / 1e6:.1f}MB, '
              f'remaining {total / 1e6:.1f}MB')
    return {'deleted': deleted, 'freed': freed, 'remaining': total}
