"""Bounded, source-labelled 2D tracks and metric trajectories for the workbench."""
from __future__ import annotations

import math
from pathlib import Path
import sys

from . import media

MAX_FRAMES = 3000
MAX_POINTS = 80


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + ' 必须为有限数值')
    return float(value)


def validate(value):
    if not isinstance(value, dict) or value.get('kind') not in ('points2d', 'trajectory3d'):
        raise ValueError('轨迹需要 kind=points2d 或 trajectory3d')
    dimension = 2 if value['kind'] == 'points2d' else 3
    units = 'normalized' if dimension == 2 else 'm'
    if value.get('units') != units or not isinstance(value.get('coordinate_frame'), str) or not value['coordinate_frame'].strip():
        raise ValueError('2D轨迹须为归一化坐标；3D须为米，并明确坐标系')
    names, frames = value.get('names'), value.get('frames')
    if not isinstance(names, list) or not 1 <= len(names) <= MAX_POINTS or any(not isinstance(n, str) or not 0 < len(n) <= 100 for n in names) or len(set(names)) != len(names):
        raise ValueError('关键点名称无效、重复或过多')
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError('请导入最多3000帧的片段')
    cleaned, last = [], -1
    for row in frames:
        stamp = number(row.get('time_s'), '轨迹时间')
        if stamp < 0 or stamp <= last:
            raise ValueError('轨迹时间必须非负并严格递增')
        last = stamp
        points = row.get('points')
        if not isinstance(points, list) or len(points) != len(names):
            raise ValueError('每帧必须对应全部关键点，缺失位置填null')
        result = []
        for point in points:
            if point is None:
                result.append(None)
                continue
            if not isinstance(point, list) or len(point) != dimension + 1:
                raise ValueError('关键点须包含坐标和置信度')
            point = [number(v, '关键点') for v in point]
            if not 0 <= point[-1] <= 1 or (dimension == 2 and any(not 0 <= p <= 1 for p in point[:-1])):
                raise ValueError('关键点超出归一化范围')
            result.append(point)
        item = {'time_s': stamp, 'points': result}
        for field, sizes in (('rotations', (3, 9)), ('widths', (1,))):
            if field not in row:
                continue
            values = row[field]
            if not isinstance(values, list) or len(values) != len(names):
                raise ValueError(field + ' 必须对应关键点')
            checked = []
            for v in values:
                if v is None:
                    checked.append(None)
                elif field == 'widths':
                    n = number(v, field)
                    if n < 0 or n > 2:
                        raise ValueError('夹爪开度需要米制数值')
                    checked.append(n)
                elif isinstance(v, list) and len(v) in sizes:
                    checked.append([number(x, field) for x in v])
                else:
                    raise ValueError(field + ' 格式无效')
            item[field] = checked
        cleaned.append(item)
    meta = value.get('metadata', {})
    import json
    if not isinstance(meta, dict) or len(json.dumps(meta, allow_nan=False)) > 16000:
        raise ValueError('轨迹元数据无效或过大')
    return {'kind': value['kind'], 'units': units, 'coordinate_frame': value['coordinate_frame'][:200],
            'names': names, 'frames': cleaned, 'metadata': meta}


def detect_hands(frame):
    """Reuse the running bundle's hand engine; never instantiate a second model."""
    bundle = next((getattr(sys.modules.get(n), '_bundle', None) for n in ('__main__', 'main')
                   if getattr(sys.modules.get(n), '_bundle', None) is not None), None)
    provider = None if bundle is None else bundle._plugin_for('pose')[0]
    if provider is None or not hasattr(provider, '_ensure_hand_model'):
        raise ValueError('现有pose插件未启用；可先手工标注或导入关键点')
    from plugins.hand_runtime import hands_in_frame, HAND_KEYPOINTS
    try:
        hands = hands_in_frame(provider._ensure_hand_model(), frame, confidence=.5)
    except Exception as error:
        raise ValueError('手部推理引擎不可用：' + str(error)[:160]) from None
    h, w = frame.shape[:2]
    seeds = []
    for hand_index, (points, score) in enumerate(hands[:2]):
        for name, (x, y, confidence) in zip(HAND_KEYPOINTS, points):
            if confidence >= .5 and 0 <= x < w and 0 <= y < h:
                seeds.append({'name': f'hand{hand_index}_{name}', 'x': float(x/w), 'y': float(y/h)})
    if not seeds:
        raise ValueError('未检测到可靠手部点；可手工指定，不会伪造关键点')
    return seeds


def follow(directory, metadata, start, end, seeds, cancel, *, detect=False):
    """Forward/backward LK tracking. Lost identities remain missing until reseeded."""
    import cv2
    import numpy as np
    start, end = media.window(metadata, start, end)
    if end - start > 30:
        raise ValueError('每次跟踪最多30秒，请选择有界片段')
    if not detect and (not isinstance(seeds, list) or not 1 <= len(seeds) <= MAX_POINTS):
        raise ValueError('请先指定1–80个关键点')
    directory = Path(directory)
    cap = cv2.VideoCapture(str(directory / 'video.mp4'))
    index = media.timeline(directory)
    index_values = index['pts_s'] if index else None
    import bisect
    first = max(0, bisect.bisect_right(index_values, start)-1) if index_values else round(start * metadata['fps'])
    count = (bisect.bisect_right(index_values, end)-first) if index_values else math.ceil((end-start)*metadata['fps'])
    if count > MAX_FRAMES:
        cap.release()
        raise ValueError('片段超过3000帧，请缩短区间')
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rows, names, previous, points, alive = [], [], None, None, None
    frame_number = first
    try:
        while len(rows) < MAX_FRAMES:
            if cancel.is_set():
                raise ValueError('操作已取消')
            ok, frame = cap.read()
            if not ok:
                break
            stamp = index_values[frame_number] if index_values and frame_number < len(index_values) else frame_number / metadata['fps']
            frame_number += 1
            if stamp > end + .00001:
                break
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is None:
                if detect:
                    seeds = detect_hands(frame)
                names = [s.get('name') for s in seeds]
                coords = [[number(s.get('x'), 'x'), number(s.get('y'), 'y')] for s in seeds]
                if any(not 0 <= c <= 1 for pair in coords for c in pair):
                    raise ValueError('点坐标需在0–1之间')
                points = np.asarray(coords, dtype=np.float32).reshape(-1, 1, 2) * np.array([w, h], np.float32)
                alive = np.ones(len(points), dtype=bool)
            elif alive.any():
                forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
                if forward is None:
                    alive[:] = False
                else:
                    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None)
                    if backward is None:
                        alive[:] = False
                    else:
                        error = np.linalg.norm(backward-points, axis=2).reshape(-1)
                        xy = forward.reshape(-1, 2)
                        alive &= status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool)
                        alive &= np.isfinite(xy).all(axis=1) & (error < 1.5)
                        alive &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
                    points = np.nan_to_num(forward)
            xy = points.reshape(-1, 2)
            rows.append({'time_s': round(stamp, 6), 'points': [
                [round(float(p[0]/w), 6), round(float(p[1]/h), 6), 1.] if visible else None
                for p, visible in zip(xy, alive)]})
            previous = gray
    finally:
        cap.release()
    if not rows:
        raise ValueError('片段没有可解码画面')
    return validate({'kind': 'points2d', 'coordinate_frame': 'decoded_source', 'units': 'normalized',
        'names': names, 'frames': rows, 'metadata': {'algorithm': 'opencv_lk_forward_backward',
        'seed_source': 'pose.hand_runtime' if detect else 'manual',
        'score_semantics': 'tracking validity, not calibrated confidence',
        'identity': 'unassigned hands; no anatomical left/right inference'}})
