"""Bounded video assets. The browser and models never receive filesystem paths."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import bisect

MAX_BYTES = 64 * 1024 * 1024
ASSET_ID = re.compile(r'[a-f0-9]{32}')


def uploaded(path, roots):
    resolved = Path(path).resolve()
    if not any(Path(r).resolve() in resolved.parents for r in roots) or not resolved.is_file():
        raise ValueError('文件必须先通过卡片上传，不能读取任意本地路径')
    if resolved.stat().st_size > MAX_BYTES:
        raise ValueError('上传文件超过 64 MiB')
    return resolved


def timestamps(path, cancel, ffmpeg='ffmpeg'):
    """Index presentation timestamps with the already-required FFmpeg binary."""
    import tempfile
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen([ffmpeg, '-nostdin', '-loglevel', 'info',
            '-protocol_whitelist', 'file,pipe', '-copyts', '-i', str(path),
            '-map', '0:v:0', '-vf', 'showinfo', '-frames:v', '90001',
            '-an', '-f', 'null', '-'], stdout=subprocess.DEVNULL, stderr=log)
        deadline = time.monotonic() + 180
        try:
            while proc.poll() is None:
                if cancel.wait(.05):
                    raise ValueError('操作已取消')
                if time.monotonic() > deadline or log.tell() > 32 * 1024 * 1024:
                    raise ValueError('时间戳索引超出预算，请裁剪视频')
            log.seek(0)
            raw = log.read(32 * 1024 * 1024 + 1)
            if proc.returncode or len(raw) > 32 * 1024 * 1024:
                raise ValueError('无法建立视频原始时间戳索引')
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    values = [float(v) for v in re.findall(rb'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([-+\d.eE]+)', raw)]
    if not values or len(values) > 90000 or any(not math.isfinite(v) for v in values):
        raise ValueError('视频时间戳无效或超过 90,000 帧')
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError('视频时间戳不递增，不能可靠同步')
    origin = values[0]
    return {'origin_s': origin, 'pts_s': [round(v-origin, 6) for v in values], 'source': 'ffmpeg_pts'}


def timeline(directory):
    path = Path(directory) / 'timestamps.json'
    return json.loads(path.read_text()) if path.exists() else None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def asset_dir(root, asset_id):
    if not isinstance(asset_id, str) or not ASSET_ID.fullmatch(asset_id):
        raise ValueError('无效的视频素材 ID')
    root = Path(root).resolve()
    path = (root / asset_id).resolve()
    if path.parent != root:
        raise ValueError('素材目录越界')
    return path


def run(command, cancel, timeout=120):
    """Drain bounded diagnostics into a temporary file, and reap on cancel."""
    import tempfile
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors)
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if cancel.wait(.1):
                    raise ValueError('操作已取消')
                if time.monotonic() > deadline:
                    raise ValueError('视频处理超时，请截取较短片段')
            if proc.returncode:
                raise ValueError('视频解码失败，请使用有效的 MP4、MOV 或 WebM 文件')
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def probe(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = count / fps if fps > 0 else 0
        width, height = (int(cap.get(k)) for k in
                         (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not all(math.isfinite(v) and v > 0 for v in
                                        (fps, duration, width, height)):
            raise ValueError('无法读取视频时长或画面')
        if duration > 900:
            raise ValueError('单段视频最多 15 分钟，请先裁剪')
        if width * height > 33554432 or max(width, height) > 8192:
            raise ValueError('视频画面过大，请先缩放到 4K 以内')
        return dict(duration_s=round(duration, 3), fps=fps, width=width, height=height)
    finally:
        cap.release()


def fetch_public_page(url, directory, cancel):
    """Optional, bounded public-video acquisition. No cookies or account access."""
    from urllib.parse import urlsplit
    host = (urlsplit(url).hostname or '').lower()
    allowed = ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be', 'www.bilibili.com', 'bilibili.com')
    if host not in allowed:
        return None
    try:
        import yt_dlp
    except ImportError:
        raise ValueError('此观看网页需要可选yt-dlp；可上传视频或使用视频直链') from None
    deadline = time.monotonic() + 180
    def progress(status):
        if cancel.is_set() or time.monotonic() > deadline:
            raise ValueError('视频获取已取消或超时')
        if status.get('downloaded_bytes', 0) > MAX_BYTES:
            raise ValueError('视频超过64 MiB')
    target = Path(directory) / 'download.mp4'
    options = {'format': 'best[ext=mp4]', 'outtmpl': str(target), 'noplaylist': True,
               'max_filesize': MAX_BYTES, 'socket_timeout': 15, 'retries': 0,
               'quiet': True, 'no_warnings': True, 'progress_hooks': [progress],
               'writeinfojson': False, 'writethumbnail': False}
    try:
        with yt_dlp.YoutubeDL(options) as client:
            info = client.extract_info(url, download=False)
            if not info or info.get('_type') in ('playlist', 'multi_video') or not 0 < info.get('duration', 0) <= 900:
                raise ValueError('只接受15分钟内的单个公开视频')
            client.download([url])
        if not target.is_file() or target.stat().st_size > MAX_BYTES:
            raise ValueError('未取得大小合适的MP4')
        return target
    except Exception:
        target.unlink(missing_ok=True)
        raise ValueError('公开视频获取失败或超出大小/时长限制；可上传已取得的文件') from None


def ingest(directory, *, path='', url='', roots=(), cancel, ffmpeg='ffmpeg'):
    if bool(path) == bool(url):
        raise ValueError('请选择一个本地视频或一个视频直链')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'source.bin'
    try:
        if url:
            from urllib.parse import urlsplit
            parsed = urlsplit(url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username:
                raise ValueError('视频直链必须是 HTTP(S) 地址')
            # Same explicit URL capability as the existing image-input plugins.
            acquired = fetch_public_page(url, directory, cancel)
            reader = acquired.open('rb') if acquired else urllib.request.urlopen(url, timeout=20)
            if not acquired and 'text/html' in (reader.headers.get('Content-Type') or ''):
                reader.close()
                raise ValueError('这是视频观看网页，尚未取得视频文件；请上传文件或提供视频直链')
        else:
            resolved = Path(path).resolve()
            if not any(Path(r).resolve() in resolved.parents for r in roots):
                raise ValueError('视频必须先通过卡片上传，不能读取任意本地路径')
            if not resolved.is_file():
                raise ValueError('找不到已上传的视频')
            reader = resolved.open('rb')
        with reader, source.open('wb') as dest:
            total = 0
            while True:
                if cancel.is_set():
                    raise ValueError('操作已取消')
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError('单个视频文件不能超过 64 MiB')
                dest.write(chunk)
        with source.open('rb') as f:
            magic = f.read(12)
        if not (magic[4:8] == b'ftyp' or magic[:4] == b'\x1aE\xdf\xa3'):
            raise ValueError('仅接受 MP4、MOV、WebM 视频文件')
        original = probe(source)
        index = timestamps(source, cancel, ffmpeg)
        target = directory / 'video.mp4'
        run([ffmpeg, '-nostdin', '-y', '-loglevel', 'error',
             '-protocol_whitelist', 'file,pipe', '-i', str(source),
             '-map', '0:v:0', '-an', '-vf', "scale='min(960,iw)':-2,setpts=PTS-STARTPTS", '-vsync', '0',
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '25',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
             '-fs', str(MAX_BYTES + 1), str(target)], cancel)
        if target.stat().st_size > MAX_BYTES:
            raise ValueError('预览视频超过 64 MiB，请裁剪后重试')
        result = probe(target)
        source_info = ({'kind': 'url', 'url': parsed._replace(query='', fragment='').geturl()}
                       if url else {'kind': 'upload', 'name': Path(path).name})
        result.update(bytes=target.stat().st_size, mime='video/mp4', audio=False,
                      sha256=sha256(target), source_sha256=sha256(source), source=source_info)
        tail = (index['pts_s'][-1] - index['pts_s'][-2]) if len(index['pts_s']) > 1 else 1 / original['fps']
        result.update(duration_s=round(index['pts_s'][-1] + tail, 3), original=original,
                      frame_count=len(index['pts_s']), timestamp_source='ffmpeg_pts',
                      original_retained=True, source_origin_s=index['origin_s'],
                      pixel_transform={'frame': 'decoded_source',
                          'scale_x': result['width']/original['width'],
                          'scale_y': result['height']/original['height']})
        write_json(directory / 'timestamps.json', index)
        return result
    except BaseException:
        source.unlink(missing_ok=True)
        raise
    finally:
        for name in ('download.mp4', 'download.mp4.part', 'download.mp4.ytdl'):
            (directory / name).unlink(missing_ok=True)


def window(metadata, start, end):
    start = float(start or 0)
    end = float(metadata['duration_s'] if end is None else end)
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= metadata['duration_s'] + .01:
        raise ValueError('分析时间范围必须位于视频内，且结束时间晚于开始时间')
    return start, min(end, metadata['duration_s'])


def sample(directory, metadata, start, end, count=12, requested_times=None):
    import cv2
    count = int(count)
    if not 2 <= count <= 32:
        raise ValueError('抽帧数量必须在 2 到 32 之间')
    start, end = window(metadata, start, end)
    last = max(start, end - 1 / metadata['fps'])
    directory = Path(directory)
    index = timeline(directory)
    source = directory / 'source.bin'
    cap = cv2.VideoCapture(str(source if source.exists() else directory / 'video.mp4'))
    frames = []
    try:
        for i in range(count):
            requested = requested_times[i] if requested_times is not None else start + (last - start) * i / (count - 1)
            if not start <= requested <= end:
                raise ValueError('采样时间超出观察区间')
            if index:
                frame_index = max(0, bisect.bisect_right(index['pts_s'], requested) - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, requested * 1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError('无法读取指定时间的画面')
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            # Some OpenCV backends report zero for every seek. Keep the
            # requested time explicit rather than claiming decoded precision.
            stamp = index['pts_s'][frame_index] if index else (actual if actual > 0 or requested == 0 else requested)
            h, w = frame.shape[:2]
            if max(w, h) > 640:
                frame = cv2.resize(frame, (round(w * 640 / max(w, h)), round(h * 640 / max(w, h))))
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise ValueError('画面编码失败')
            frames.append({'time_s': round(stamp, 3), 'requested_s': round(requested, 3),
                           'timestamp_source': 'ffmpeg_pts' if index else ('decoded' if actual > 0 or requested == 0 else 'requested'),
                           'jpeg': jpeg.tobytes()})
    finally:
        cap.release()
    return frames


def clip(directory, metadata, start, end, cancel, ffmpeg='ffmpeg'):
    start, end = window(metadata, start, end)
    target = Path(directory) / 'analysis.mp4'
    run([ffmpeg, '-nostdin', '-y', '-loglevel', 'error', '-ss', str(start),
         '-i', str(Path(directory) / 'video.mp4'), '-t', str(end-start),
         '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '25',
         '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], cancel)
    if target.stat().st_size > 18 * 1024 * 1024:
        raise ValueError('原生视频请求片段超过 18 MiB，请缩短时间范围；不会自动切换输入方式')
    return target


def data_url(data, mime):
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')


def write_json(path, value):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    os.replace(temp, path)
