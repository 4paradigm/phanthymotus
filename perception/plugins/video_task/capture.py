"""Read-only demonstration recorder. Owned by Perception, shares VideoTask storage.

Only ROS topics supplied by Canvas start are subscribed. Recording is explicit;
opening/closing the workbench never starts/stops acquisition or robot motion.
"""
from __future__ import annotations

import base64
import copy
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

from . import media, motion

ROLES = ('external', 'ego', 'wrist', 'other')
MAX_DURATION = 120.
MAX_BYTES = 256 * 1024 * 1024


class DemoCapturePlugin:
    PREFIX = 'demo_capture'

    def __init__(self, cfg, executor, video_task):
        self.cfg, self.executor, self.video_task = dict(cfg), executor, video_task
        self.root = video_task.root / '_captures'
        self.lock = threading.RLock()
        self.node = None
        self.sources = {}
        self.session = None
        self.worker = None
        self.halt = threading.Event()
        self.closed = False
        self.timer = None
        self.clock = time.monotonic
        self.last_error = ''
        # An interrupted process must never resume collection silently.
        for path in self.root.glob('*/manifest.json'):
            try:
                value = json.loads(path.read_text())
                if value['state'] in ('recording', 'processing'):
                    value.update(state='interrupted', error='上次进程中断；原始记录已保留，可重试整理')
                    media.write_json(path, value)
            except (OSError, ValueError, KeyError):
                continue

    def get_tools(self):
        params = {'ui': [], 'info': [], 'start': ['input_topic', 'input_topics'], 'stop': [],
                  'configure': ['sources'], 'record_start': ['name'], 'record_stop': [],
                  'mark': ['label'], 'retry': ['capture_id']}
        props = {'action': {'type': 'string', 'enum': list(params)},
                 'input_topic': {'type': 'string'}, 'input_topics': {'type': 'array', 'items': {'type': 'string'}},
                 'sources': {'type': 'array', 'items': {'type': 'object'}},
                 'name': {'type': 'string'}, 'label': {'type': 'string'}, 'capture_id': {'type': 'string'}}
        return [{'name': self.PREFIX, 'type': 'processor', 'multiInstance': False,
                 'description': '示范采集：只读录制相机与状态话题，结束后自动进入示范学习。不控制机器人。',
                 'inputSchema': {'type': 'object', 'properties': props, 'required': ['action'],
                     'x-action-params': {k: {'params': v} for k, v in params.items()},
                     'x-workbench': {'protocol': 'motus.workbench/1', 'title': '示范采集',
                                     'entry': 'ui', 'actions': list(params), 'card_actions': ['ui'], 'inputs': True}},
                 'topic_in': [{'format': 'image/jpeg', 'desc': '示范相机，可接多路'},
                              {'format': 'state/joint', 'desc': '机器人关节/末端状态'},
                              {'format': 'data/json', 'desc': '夹爪、动作或其他数值状态'}]}]

    def _path(self, capture_id):
        return media.asset_dir(self.root, capture_id)

    def _persist(self):
        if self.session:
            media.write_json(self._path(self.session['id']) / 'manifest.json', self.session)

    def _start(self, args):
        with self.lock:
            if self.session and self.session['state'] in ('recording', 'processing'):
                raise ValueError('录制或整理中，不能改变采集连线')
            inputs = args.get('input_topics') or ([args['input_topic']] if args.get('input_topic') else [])
            if not isinstance(inputs, list) or not inputs or len(inputs) > 7 or any(not isinstance(t, str) or not re.fullmatch(r'/[A-Za-z0-9_/]+', t) for t in inputs):
                raise ValueError('请在画布连接相机和可选状态话题，再开启采集预览；最多7路输入')
            inputs = list(dict.fromkeys(inputs))
            if self.node and set(inputs) == set(self.sources):
                return {'state': 'running'}
            self._stop_node()
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            from sensor_msgs.msg import CompressedImage
            from std_msgs.msg import String
            node = Node('demo_capture')
            try:
                types = dict(node.get_topic_names_and_types())
                planned = {}
                for topic in inputs:
                    kinds = types.get(topic, [])
                    if 'sensor_msgs/msg/CompressedImage' in kinds:
                        kind, msg_type = 'video', CompressedImage
                    elif 'std_msgs/msg/String' in kinds:
                        kind, msg_type = 'state', String
                    elif 'sensor_msgs/msg/JointState' in kinds:
                        from sensor_msgs.msg import JointState
                        kind, msg_type = 'joints', JointState
                    else:
                        raise ValueError('输入尚未发布或类型不支持：' + topic + '；支持JPEG CompressedImage、JSON String、JointState')
                    planned[topic] = {'topic': topic, 'kind': kind, 'role': 'external', 'name': topic.rsplit('/', 1)[-1],
                                      'received': None, 'preview': '', 'values': {}, 'error': '', 'message_type': kinds[0]}
                    qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
                    node.create_subscription(msg_type, topic, lambda m, t=topic, k=kind: self.receive(t, m, k), qos)
                if not 1 <= sum(s['kind'] == 'video' for s in planned.values()) <= 3 or sum(s['kind'] != 'video' for s in planned.values()) > 4:
                    raise ValueError('需要1–3路JPEG相机，其余最多4路数值状态')
                self.sources, self.node = planned, node
                if self.executor is None:
                    raise ValueError('ROS执行器不可用，不能显示为已接入设备')
                self.executor.add_node(node)
            except Exception:
                node.destroy_node()
                self.node, self.sources = None, {}
                raise
            return {'state': 'running'}

    @staticmethod
    def _stamp(message):
        stamp = getattr(getattr(message, 'header', None), 'stamp', None)
        if stamp is not None:
            value = getattr(stamp, 'sec', 0) + getattr(stamp, 'nanosec', 0)/1e9
            return value if value else None
        return None

    def receive(self, topic, message, kind=None):
        """ROS callback; bounded writes, malformed samples remain explicit gaps."""
        with self.lock:
            source = self.sources.get(topic)
            if source is None or self.closed:
                return
            now = self.clock()
            kind = kind or source['kind']
            # Bound video/state retention at 10/30 Hz per input, before decoding.
            interval = .1 if kind == 'video' else 1/30
            if source.get('accepted') is not None and now-source['accepted'] < interval - 1e-6:
                return
            try:
                if kind == 'video':
                    import cv2
                    import numpy as np
                    raw = bytes(message.data)
                    if len(raw) > 4*1024*1024 or not raw.startswith(b'\xff\xd8'):
                        raise ValueError('只接收不超过4MiB的JPEG帧')
                    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None or frame.shape[0]*frame.shape[1] > 16_000_000:
                        raise ValueError('JPEG图像无效或过大')
                    h, w = frame.shape[:2]
                    thumb = cv2.resize(frame, (min(480, w), max(2, round(h*min(480, w)/w))))
                    ok, data = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if not ok:
                        raise ValueError('预览编码失败')
                    source['preview'] = 'data:image/jpeg;base64,' + base64.b64encode(data.tobytes()).decode()
                    payload = None
                else:
                    if kind == 'joints':
                        payload = {'name': list(message.name), 'position': list(message.position),
                                   'velocity': list(message.velocity), 'effort': list(message.effort),
                                   'units': 'position=rad or m (joint type); velocity=rad/s or m/s; effort=Nm or N'}
                    else:
                        if not isinstance(message.data, str) or len(message.data) > 65536:
                            raise ValueError('状态JSON超过64KiB')
                        payload = json.loads(message.data)
                    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                    if not isinstance(payload, dict) or len(raw) > 65536:
                        raise ValueError('需要有界状态JSON对象')
                    source['values'] = motion.numeric_channels(payload)
                source.update(received=now, accepted=now, error='')
                session = self.session
                if not session or session['state'] != 'recording':
                    return
                if now-session['origin'] >= MAX_DURATION or session['bytes'] + len(raw) > MAX_BYTES:
                    self._finish('达到120秒或256MiB上限')
                    return
                row = session['sources'][topic]
                elapsed = now-session['origin']
                if elapsed <= row.get('last_s', -1):
                    raise ValueError('接收时间未递增')
                target = self._path(session['id']) / row['key']
                if kind == 'video':
                    filename = f'{row["count"]:06d}.jpg'
                    entry = {'time_s': elapsed, 'source_time_s': self._stamp(message), 'file': filename}
                else:
                    entry = {'time_s': elapsed, 'source_time_s': self._stamp(message), 'payload': payload,
                             'values': source['values']}
                encoded = json.dumps(entry, ensure_ascii=False, allow_nan=False)+'\n'
                cost = len(encoded.encode()) + (len(raw) if kind == 'video' else 0)
                if session['bytes']+cost > MAX_BYTES:
                    self._finish('达到256MiB存储上限')
                    return
                if kind == 'video':
                    (target / filename).write_bytes(raw)
                with (target / 'samples.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(encoded)
                row['count'] += 1
                if 'first_s' not in row:
                    row['first_s'] = elapsed
                if elapsed-row.get('last_s', elapsed) > .5:
                    row['gaps'] += 1
                row['last_s'] = elapsed
                session['bytes'] += cost
                if now-session.get('checkpoint', 0) > 2:
                    session['checkpoint'] = now
                    self._persist()
            except Exception as error:
                source['error'] = str(error)[:180]
                if self.session and self.session['state'] == 'recording':
                    self.session['sources'][topic]['rejected'] += 1
                    if isinstance(error, OSError):
                        self._finish('磁盘写入失败：'+str(error)[:100])

    def _begin(self, args):
        with self.lock:
            if self.closed or not self.node:
                raise ValueError('请连接相机并启动项目，等待实时画面')
            if self.worker and self.worker.is_alive() or self.session and self.session['state'] == 'recording':
                raise ValueError('当前录制或整理尚未结束')
            now = self.clock()
            stale = [s['name'] for s in self.sources.values() if s['received'] is None or now-s['received'] > 2 or s['error']]
            if stale:
                raise ValueError('这些来源没有新鲜有效数据：'+'、'.join(stale))
            name = str(args.get('name') or time.strftime('示范 %m-%d %H:%M'))[:100]
            capture_id = uuid.uuid4().hex
            directory = self._path(capture_id)
            directory.mkdir(parents=True)
            self.session = {'id': capture_id, 'name': name, 'state': 'recording', 'origin': now,
                            'started_at': time.time(), 'bytes': 0, 'sources': {}, 'marks': [], 'error': ''}
            for i, source in enumerate(self.sources.values()):
                row = {k: source[k] for k in ('topic', 'kind', 'role', 'name', 'message_type')}
                row.update(key=str(i), count=0, gaps=0, rejected=0)
                self.session['sources'][source['topic']] = row
                (directory / str(i)).mkdir()
            try:
                self._persist()
            except OSError as error:
                self.session.update(state='failed', error='无法保存录制状态：'+str(error)[:150])
                raise
            self.timer = threading.Timer(MAX_DURATION, self._finish, args=('达到120秒上限',))
            self.timer.daemon = True
            self.timer.start()
            return {'state': 'recording', 'capture_id': capture_id}

    def _finish(self, reason='用户结束录制'):
        with self.lock:
            if not self.session or self.session['state'] != 'recording':
                return {'state': self.session['state'] if self.session else 'idle'}
            if self.timer:
                self.timer.cancel()
            self.session.update(state='processing', duration_s=self.clock()-self.session['origin'], finish_reason=reason)
            for row in self.session['sources'].values():
                row['tail_gap_s'] = max(0, self.session['duration_s']-row.get('last_s', 0))
            try:
                self._persist()
            except OSError as error:
                self.session['error'] = '无法保存录制状态：'+str(error)[:150]
            self._launch_finalize(self.session)
            return {'state': 'processing'}

    def _launch_finalize(self, session):
        self.halt = threading.Event()
        self.worker = threading.Thread(target=self._finalize, args=(copy.deepcopy(session),), daemon=True, name='demo-capture-finalize')
        self.worker.start()

    def _finalize(self, value):
        directory = self._path(value['id'])
        try:
            videos, signals = [], []
            for row in value['sources'].values():
                source_dir = directory / row['key']
                path = source_dir / 'samples.jsonl'
                rows, units = [], 'source-defined/unknown'
                if path.exists():
                    with path.open() as handle:
                        for line in handle:
                            sample = json.loads(line)
                            if row['kind'] != 'video':
                                units = sample.get('payload', {}).get('units', units)
                                sample = {k: sample.get(k) for k in ('time_s', 'source_time_s', 'values')}
                            rows.append(sample)
                row['count'] = len(rows)
                if rows:
                    row.update(first_s=rows[0]['time_s'], last_s=rows[-1]['time_s'],
                               gaps=sum(b['time_s']-a['time_s'] > .5 for a, b in zip(rows, rows[1:])))
                    value['duration_s'] = max(value.get('duration_s', 0), rows[-1]['time_s'])
                    row['tail_gap_s'] = max(0, value['duration_s']-rows[-1]['time_s'])
                if row['kind'] != 'video':
                    signals.append({'topic': row['topic'], 'rows': rows, 'units': units})
                    continue
                if len(rows) < 2:
                    row['error'] = '不足两帧，未生成视频'
                    continue
                if any(b['time_s'] <= a['time_s'] for a, b in zip(rows, rows[1:])):
                    raise ValueError('录像时间戳不递增')
                content = []
                for i, sample in enumerate(rows):
                    if not re.fullmatch(r'[0-9]{6}\.jpg', sample['file']):
                        raise ValueError('录像帧路径无效')
                    dt = rows[i+1]['time_s']-sample['time_s'] if i+1 < len(rows) else .1
                    # concat's per-file option needs safe=0. Every filename is
                    # locally generated and regex-checked above, never a URL.
                    content.extend([f"file '{sample['file']}'", 'option framerate 1000', f'duration {dt:.6f}'])
                concat = source_dir / 'frames.txt'
                concat.write_text('\n'.join(content)+'\n')
                target = source_dir / 'capture.mp4'
                media.run([self.video_task.cfg.get('ffmpeg', 'ffmpeg'), '-nostdin', '-y', '-loglevel', 'error',
                           '-f', 'concat', '-safe', '0', '-protocol_whitelist', 'file,pipe', '-i', str(concat), '-vsync', 'vfr',
                           '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-an', '-c:v', 'libx264',
                           '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], self.halt)
                videos.append({'path': target, 'source': row, 'first_s': rows[0]['time_s']})
            if not videos:
                raise ValueError('没有可用录像，原始记录已保留')
            draft_id = self.video_task.accept_capture(value, videos, signals, self.halt)
            value.update(state='done', draft_id=draft_id, error='')
        except Exception as error:
            value.update(state='interrupted' if self.halt.is_set() else 'failed', error=str(error)[:300])
        with self.lock:
            self.session = value
            try:
                self._persist()
            except OSError as error:
                self.session.update(state='failed', error='无法写入录制结果状态；原始文件保留：'+str(error)[:150])

    def info(self):
        with self.lock:
            now = self.clock()
            sources = [{k: v for k, v in s.items() if k not in ('accepted', 'received')} |
                       {'fresh': s['received'] is not None and now-s['received'] <= 2,
                        'age_s': round(now-s['received'], 2) if s['received'] is not None else None}
                       for s in self.sources.values()]
            recent = []
            for p in sorted(self.root.glob('*/manifest.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
                try:
                    v = (copy.deepcopy(self.session) if self.session and p.parent.name == self.session['id']
                         else json.loads(p.read_text()))
                    if v['state'] in ('recording', 'processing') and (not self.session or v['id'] != self.session['id']):
                        v.update(state='interrupted', error='上次进程中断；原始记录保留')
                    recent.append({k: v.get(k) for k in ('id', 'name', 'state', 'draft_id', 'duration_s', 'error')})
                except (ValueError, OSError):
                    pass
            session = copy.deepcopy(self.session)
            if session:
                session['elapsed_s'] = round(now-session['origin'], 1) if session['state'] == 'recording' else session.get('duration_s', 0)
                session.pop('origin', None)
            # Busy means "cannot close the iframe" in the existing host. Capture
            # is independent of its viewer and may safely continue after close.
            return {'state': 'running' if self.node else 'idle', 'busy': False, 'sources': sources,
                    'session': session, 'recent': recent, 'max_duration_s': MAX_DURATION,
                    'time_basis': 'host monotonic reception; original header stamps retained; no hardware synchronization'}

    def dispatch(self, name, args):
        try:
            action = args.get('action', name)
            if action == 'ui':
                return {'protocol': 'motus.workbench/1', 'html': Path(__file__).with_name('capture.html').read_text()}
            if action == 'info':
                return self.info()
            if action == 'start':
                return self._start(args)
            if action == 'record_start':
                return self._begin(args)
            if action == 'record_stop':
                return self._finish()
            if action == 'stop':
                result = self._finish('项目停止')
                with self.lock:
                    self._stop_node()
                return result
            with self.lock:
                if action == 'mark':
                    if not self.session or self.session['state'] != 'recording':
                        raise ValueError('仅能标记当前录制')
                    if len(self.session['marks']) >= 100:
                        raise ValueError('最多100条人工标记')
                    self.session['marks'].append({'time_s': self.clock()-self.session['origin'], 'label': str(args.get('label') or '人工干预')[:200], 'source': 'human'})
                    self._persist()
                    return {'state': 'marked'}
                if action == 'configure':
                    if self.session and self.session['state'] in ('recording', 'processing'):
                        raise ValueError('请结束录制与整理后再修改来源')
                    rows = args.get('sources')
                    if not isinstance(rows, list) or len(rows) > 7:
                        raise ValueError('来源设置无效')
                    changes = {}
                    for row in rows:
                        if row.get('topic') not in self.sources or row.get('role') not in ROLES:
                            raise ValueError('只能修改已连接来源的角色')
                        changes[row['topic']] = {'role': row['role'], 'name': str(row.get('name') or row['topic'])[:100]}
                    for topic, values in changes.items():
                        self.sources[topic].update(values)
                    return {'state': 'configured'}
                if action == 'retry':
                    if self.worker and self.worker.is_alive() or self.session and self.session['state'] == 'recording':
                        raise ValueError('当前录制或整理尚未结束')
                    path = self._path(args.get('capture_id')) / 'manifest.json'
                    value = (copy.deepcopy(self.session) if self.session and self.session['id'] == args.get('capture_id')
                             else json.loads(path.read_text()))
                    if value['state'] == 'done':
                        return {'state': 'done', 'draft_id': value['draft_id']}
                    if value['state'] not in ('failed', 'interrupted'):
                        raise ValueError('只有失败或中断的录制可以重试')
                    value.update(state='processing', duration_s=value.get('duration_s', max((s.get('last_s', 0) for s in value['sources'].values()), default=0)))
                    self.session = value
                    self._persist()
                    self._launch_finalize(value)
                    return {'state': 'processing'}
            raise ValueError('未知采集操作')
        except Exception as error:
            return {'error': str(error)[:500]}

    def _stop_node(self):
        if self.node:
            if self.executor:
                self.executor.remove_node(self.node)
            self.node.destroy_node()
            self.node = None
        self.sources = {}

    def shutdown(self):
        self._finish('服务退出')
        with self.lock:
            self.closed = True
            self._stop_node()
        if self.worker:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                self.halt.set()
                self.worker.join(timeout=3)
