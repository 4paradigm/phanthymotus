"""Small, auditable motion evidence shared by playback, video analysis and skills.

This computes measurements, never contact, task success or robot-space poses.
"""
from __future__ import annotations

import json
import math

from . import media


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def numeric_channels(payload):
    """Flatten bounded numeric sensor values; text is not a model instruction."""
    result = {}
    def walk(value, path, depth):
        if len(result) >= 80 or depth > 4:
            return
        if finite(value):
            result[path] = float(value)
        elif isinstance(value, list):
            for i, item in enumerate(value[:80]):
                walk(item, f'{path}[{i}]', depth + 1)
        elif isinstance(value, dict):
            for key, item in list(value.items())[:80]:
                if key in ('timestamp', 'timestamp_ms', 'timestamp_ns', 'stamp', 'header', 'time', 'seq'):
                    continue
                walk(item, (path + '.' if path else '') + str(key)[:80], depth + 1)
    walk(payload, '', 0)
    return result


def _samples(rows, count=8):
    if len(rows) <= count:
        return rows
    return [rows[round(i * (len(rows)-1)/(count-1))] for i in range(count)]


def collect(root, draft, start_s, end_s):
    """Task-clock measurements with immutable file identities and alignment."""
    directory = media.asset_dir(root, draft['id'])
    streams = {s['id']: s for s in draft['streams'] if s['role'] != 'execution'}
    summaries, references = [], []
    # Latest four data layers only. Disclose the bound, never imply full coverage.
    available = [t for t in draft['tracks'] if t['stream_id'] in streams]
    for layer in available[-4:]:
        stream = streams[layer['stream_id']]
        if not stream['alignment_confirmed']:
            continue
        path = directory / 'tracks' / (layer['id'] + '.json')
        data = json.loads(path.read_text())
        rows = [f for f in data['frames'] if start_s <= stream['rate']*f['time_s'] + stream['offset_s'] <= end_s]
        if not rows:
            continue
        ref = {'track_id': layer['id'], 'sha256': media.sha256(path), 'stream_id': stream['id'],
               'alignment_revision': stream['alignment_revision']}
        references.append(ref)
        names = data['names']
        points = []
        for i, name in enumerate(names[:12]):
            valid = [(f, f['points'][i]) for f in rows if f['points'][i] is not None]
            item = {'name': name, 'valid_frames': len(valid), 'missing_frames': len(rows)-len(valid)}
            if valid:
                first, last = valid[0][1][:-1], valid[-1][1][:-1]
                item.update(first=first, last=last,
                            displacement=round(math.dist(first, last), 6))
                widths = [f['widths'][i] for f in rows if f.get('widths') and f['widths'][i] is not None]
                if widths:
                    item['gripper_width_m'] = {'first': widths[0], 'last': widths[-1], 'min': min(widths), 'max': max(widths)}
            points.append(item)
        samples = []
        for f in _samples(rows):
            row = {'task_time_s': round(stream['rate']*f['time_s']+stream['offset_s'], 6), 'points': f['points'][:12]}
            if 'widths' in f:
                row['widths_m'] = f['widths'][:12]
            samples.append(row)
        summaries.append({**ref, 'kind': data['kind'], 'units': data['units'],
                          'coordinate_frame': data['coordinate_frame'], 'source': layer['source'],
                          'points': points, 'samples': samples, 'total_points': len(names),
                          'omitted_points': max(0, len(names)-12)})
    signals = []
    capture = draft.get('capture')
    if capture:
        path = directory / 'capture-signals.json'
        if path.is_file():
            value = json.loads(path.read_text())
            references.append({'capture_signals': True, 'sha256': media.sha256(path)})
            for source in value['sources']:
                rows = [r for r in source['rows'] if start_s <= r['time_s'] <= end_s]
                keys = sorted({k for row in rows for k in row['values']})[:12]
                stats = []
                for key in keys:
                    values = [r['values'][key] for r in rows if key in r['values']]
                    if values:
                        stats.append({'channel': key, 'first': values[0], 'last': values[-1],
                                      'min': min(values), 'max': max(values), 'samples': len(values)})
                signals.append({'topic': source['topic'], 'units': source.get('units', 'source-defined/unknown'),
                                'stats': stats, 'samples': [{'task_time_s': r['time_s'], 'values': {k: r['values'][k] for k in keys if k in r['values']}} for r in _samples(rows)]})
    result = {'time_basis': 'task_seconds', 'interval_s': [start_s, end_s],
              'synchronization': (capture or {}).get('sync_note', '使用已确认的时间偏移；没有跨相机空间标定'),
              'tracks': summaries, 'signals': signals, 'references': references,
              'human_marks': [m for m in (capture or {}).get('marks', []) if start_s <= m['time_s'] <= end_s][:20],
              'limits': '最多4组已确认同步轨迹，每组12点、8个采样；每路状态12通道。位移/开合不证明接触或成功。没有坐标迁移。',
              'omitted_layers': max(0, len(available)-4)}
    # Keep deterministic summaries usable in model requests and host RPC.
    if len(json.dumps(result, ensure_ascii=False)) > 28000:
        for item in summaries + signals:
            item.pop('samples', None)
        result['limits'] += ' 当前数据较多，已省略时序采样，只提供统计。'
    return result


def validate_references(root, draft, refs):
    directory = media.asset_dir(root, draft['id'])
    for ref in refs:
        if ref.get('capture_signals'):
            path = directory / 'capture-signals.json'
        else:
            if not media.ASSET_ID.fullmatch(ref.get('track_id', '')):
                raise ValueError('无效动作证据')
            stream = next((s for s in draft['streams'] if s['id'] == ref['stream_id']), None)
            if not stream or not stream['alignment_confirmed'] or stream['alignment_revision'] != ref['alignment_revision']:
                raise ValueError('动作证据的同步已改变，请重新分析')
            path = directory / 'tracks' / (ref['track_id'] + '.json')
        if not path.is_file() or media.sha256(path) != ref['sha256']:
            raise ValueError('动作证据已改变，请重新分析')
