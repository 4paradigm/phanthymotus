"""VOP observations + negotiated head feedback -> bounded motus.control/1 targets.

No robot names, SDKs, addresses, ROS or image/model libraries belong here.
"""
from __future__ import annotations
import copy
import math
import time
import uuid
from ..control_stream.message import build


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def topic(value):
    import re
    require(isinstance(value, str) and len(value) <= 200 and
            re.fullmatch(r'/[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*', value), 'Invalid ROS topic')
    return value


def negotiate(raw):
    """A head must explicitly declare axes, directions, neutral and feedback.

    A random two-joint arm or a chassis must never qualify just by its width.
    Supports one-axis pan/tilt units and arbitrary two-axis joint ordering.
    """
    d = copy.deepcopy(raw)
    require(isinstance(d, dict) and d.get('control_interface') == 'motus.control/1' and
            d.get('mode') == 'joint_position', '下游必须声明 motus.control/1 头部位置接口')
    n = d.get('dof')
    require(type(n) is int and n in (1, 2), '头部接口只接受一或二个实际可动轴')
    names = d.get('joint_names')
    require(isinstance(names, list) and len(names) == n and
            all(isinstance(v, str) and v for v in names) and len(set(names)) == n, '头部关节名称不完整')
    units = d.get('units')
    require(isinstance(units,dict) and units.get('angle') == 'rad' and units.get('time') == 's', '头部接口必须明确使用角度 rad、时间 s')
    h = d.get('head_tracking') or {}
    require(isinstance(h,dict), '头部扩展声明无效')
    require(h.get('schema') == 'motus.head-tracking/1', '下游没有声明可动头部，不能用底盘或机械臂替代')
    axes = h.get('axes')
    require(isinstance(axes, dict) and bool(axes) and set(axes) <= {'yaw', 'pitch'} and len(axes) == n,
            '必须声明 yaw/pitch 轴的实际方向')
    indices = []
    for a in axes.values():
        require(isinstance(a, dict) and type(a.get('index')) is int and 0 <= a['index'] < n and
                type(a.get('sign')) is int and a['sign'] in (-1, 1), '轴顺序或正方向无效')
        require(type(a.get('image_axis')) is int and a['image_axis'] in (0, 1) and
                finite(a.get('gain_rad')) and 0 < a['gain_rad'] <= .5,
                '必须声明对应图像轴及保守的视觉控制增益')
        indices.append(a['index'])
    require(len(set(indices)) == n, '头部轴索引重复')
    require(len({a['image_axis'] for a in axes.values()}) == n, '两个头部轴不能假定控制同一图像方向')
    limits = d.get('limits') or {}
    require(isinstance(limits,dict), '头部限位声明无效')
    for key in ('lower', 'upper', 'max_velocity'):
        a = limits.get(key)
        require(isinstance(a, list) and len(a) == n and all(finite(v) for v in a), '头部限位/速度声明不完整')
    require(all(a < b and v > 0 for a, b, v in zip(limits['lower'], limits['upper'], limits['max_velocity'])),
            '头部限位或速度无效')
    neutral = h.get('neutral')
    require(isinstance(neutral, list) and len(neutral) == n and all(finite(v) and lo <= v <= hi
            for v, lo, hi in zip(neutral, limits['lower'], limits['upper'])), '必须声明限位内的回正位置')
    require(finite(h.get('max_duration_s')) and .6 <= h['max_duration_s'] <= 15, '头部动作时间上限无效')
    topic(h.get('feedback_topic')); topic(h.get('camera_topic'))
    rate = d.get('rate') or {}
    require(isinstance(rate,dict), '头部频率声明无效')
    require(finite(rate.get('max_obs_age_ms')) and 0 < rate['max_obs_age_ms'] <= 2000 and
            finite(rate.get('max_hz')) and rate['max_hz'] > 0, '头部接口缺少新鲜度/频率约束')
    require('force_torque' in d and (d['force_torque'] is None or isinstance(d['force_torque'],list) and
            bool(d['force_torque']) and all(finite(v) and v > 0 for v in d['force_torque'])),
            '必须声明实际测力阈值或明确为 null')
    return d


def observation(data, now, max_age=1.5):
    require(isinstance(data, dict) and finite(data.get('timestamp')) and
            0 <= now - data['timestamp'] <= max_age, '等待新鲜 VOP 检测')
    objects = data.get('objects')
    require(isinstance(objects, list) and len(objects) <= 1000 and
            ('count' not in data or type(data['count']) is int and data['count'] == len(objects)), 'VOP 对象列表无效')
    require(data.get('available') is not False and not data.get('error'), 'VOP 当前不可用')
    latency = data.get('latency_ms', 0)
    require(finite(latency) and 0 <= latency <= max_age * 1000, 'VOP 推理结果过期')
    stamp = data['timestamp'] - latency / 1000
    require(0 <= now - stamp <= max_age, 'VOP 处理起始时间已过期')
    people = []
    for item in objects:
        require(isinstance(item, dict), 'VOP 检测项无效')
        if item.get('name') not in ('person', 'human', '人'):
            continue
        p = item.get('position'); confidence = item.get('confidence')
        require(isinstance(p, list) and len(p) == 2 and all(finite(v) and -1 <= v <= 1 for v in p) and
                finite(confidence) and 0 <= confidence <= 1, 'VOP 人物位置无效')
        if confidence >= .55:
            people.append(tuple(p))
    return {'stamp': data['timestamp'], 'obs_ms': int(stamp * 1000), 'people': people,
            'time_basis': 'vop_processing_start' if 'latency_ms' in data else 'vop_publication'}


class Policy:
    def __init__(self, descriptor, clock=time.monotonic, wall=time.time):
        self.d = negotiate(descriptor)
        self.h = self.d['head_tracking']
        self.clock, self.wall = clock, wall
        self.source = 'headfollow-' + uuid.uuid4().hex
        self.seq = 0
        self.enabled = False
        self.pending = None
        self.error = None
        self.state = 'idle'
        self.target = self.candidate = None
        self.candidate_at = None
        self.missing_at = None
        self.misses = 0
        self.last_frame = float('-inf')
        self.after_frame = float('-inf')
        self.stable = []
        self.session = self.owner = None
        self.time_basis = None

    def feedback(self, state):
        n = self.d['dof']
        require(isinstance(state, dict) and state.get('schema') == 'motus.head-state/1' and
                isinstance(state.get('session'), str) and bool(state['session']) and
                state.get('available') is True and finite(state.get('timestamp')) and
                0 <= self.wall() - state['timestamp'] <= 1 and finite(state.get('sample_stamp')),
                '等待新鲜完整的头部反馈')
        require(state.get('stream_enabled') is True, '下游头部执行端未启用，不能发送跟随指令')
        for key in ('values', 'velocities'):
            require(isinstance(state.get(key), list) and len(state[key]) == n and
                    all(finite(v) for v in state[key]), '头部反馈维数或数值无效')
        require(all(lo <= v <= hi for v, lo, hi in zip(state['values'],
                self.d['limits']['lower'], self.d['limits']['upper'])), '当前头部超出跟随工作范围')
        return state

    def enable(self, detections, state):
        require(not self.pending, '上次头部结果未确认，不能重新启用')
        observation(detections, self.wall())
        s = self.feedback(state)
        require(s.get('ready') is True and s.get('stationary') is True, '头部或其它执行器未就绪/未停止')
        self.session, self.owner = s['session'], s.get('motion_id')
        self.enabled = True
        self.error = None
        self.state = 'waiting_person'
        self.after_frame = self.wall()
        self.target = self.candidate = self.missing_at = None
        self.candidate_at = None
        self.last_frame = float('-inf')
        self.misses = 0
        self.stable = []

    def stop(self):
        self.enabled = False
        self.state = 'idle'

    def info(self):
        return {'active': self.enabled, 'state': self.state, 'error': self.error,
                'pending': copy.deepcopy(self.pending), 'head_commands_sent': self.seq,
                'target': self.target, 'observation_time_basis': self.time_basis,
                'measurement_time_verified': False, 'robot_sdk_imports': False}

    def fail(self, message):
        self.enabled = False
        self.state = 'fault'
        self.error = str(message)[:240]

    def step(self, detections, state):
        if not self.enabled:
            return None
        try:
            s = self.feedback(state)
            require(s['session'] == self.session, '头部驱动已重启或身份改变，跟随停止')
            if self.pending:
                ack = s.get('request') or {}
                require(isinstance(ack,dict), '头部请求确认格式无效')
                if ack.get('source') == self.source and ack.get('seq') == self.pending['seq']:
                    require(ack.get('outcome') not in ('failed', 'rejected', 'unknown'),
                            '头部动作失败或结果不明；不重发、不回正')
                    if ack.get('outcome') == 'completed':
                        require(s.get('completion_verified') is True and s.get('ready') is True and
                                s.get('stationary') is True and max(abs(a-b) for a, b in
                                zip(s['values'], self.pending['target'])) <= math.radians(1),
                                '头部停止不能替代到位和完成确认')
                        self.owner = s.get('motion_id')
                        self.pending = None
                        self.after_frame = self.wall()
                        self.stable = []
                        return None
                require(self.clock() <= self.pending['deadline'], '头部请求确认超时；不重发、不回正')
                return None
            require(s.get('ready') is True and s.get('stationary') is True, '执行器忙或反馈不稳定，跟随停止')
            require(s.get('motion_id') == self.owner, '检测到其它头部控制，跟随停止')
            now = self.clock()
            if not self.stable or s['sample_stamp'] > self.stable[-1][1]:
                if self.stable and max(abs(a-b) for a, b in zip(s['values'], self.stable[0][2])) > math.radians(.2):
                    self.stable = []
                self.stable.append((now, s['sample_stamp'], list(s['values'])))
                self.stable = self.stable[-20:]
            certified = s.get('completion_verified') is True and s.get('motion_id') == self.owner
            if not certified and (len(self.stable) < 3 or now - self.stable[0][0] < .5):
                return None
            try:
                frame = observation(detections, self.wall(), min(1.5, self.d['rate']['max_obs_age_ms'] / 1000))
            except ValueError:
                self.misses = 0
                self.missing_at = None
                self.state = 'input_wait'
                return None
            self.time_basis = frame['time_basis']
            require(frame['stamp'] >= self.last_frame, 'VOP 时间戳回退，请重新启用')
            if frame['stamp'] <= self.last_frame or frame['stamp'] <= self.after_frame:
                return None
            self.last_frame = frame['stamp']
            people = frame['people']
            if self.target is None:
                ranked = sorted(people, key=lambda p: math.hypot(*p))
                target = ranked[0] if ranked else None
                if len(ranked) > 1 and abs(math.hypot(*ranked[0]) - math.hypot(*ranked[1])) < .12:
                    self.candidate = None
                    self.state = 'ambiguous'
                    return None
                if target is not None:
                    self.missing_at = None; self.misses = 0
                    if self.candidate is None or math.dist(target, self.candidate) > .3:
                        self.candidate, self.candidate_at = target, now
                        return None
                    if now - self.candidate_at < .25:
                        return None
                    self.target = target
            else:
                ranked = sorted(people, key=lambda p: math.dist(p, self.target))
                target = ranked[0] if ranked and math.dist(ranked[0], self.target) <= .4 else None
                if target is not None and len(ranked) > 1 and abs(math.dist(ranked[0], self.target) - math.dist(ranked[1], self.target)) < .12:
                    target = None
                if target is not None:
                    self.target = target
            if target is None:
                self.candidate = None
                self.misses += 1
                if self.missing_at is None:
                    self.missing_at = now
                self.state = 'waiting_person'
                if self.misses < 3 or now - self.missing_at < 1.5:
                    return None
                return self.command(s, self.h['neutral'], frame, 'return_home')
            self.misses = 0; self.missing_at = None
            self.state = 'tracking'
            values = list(s['values'])
            for name, axis in self.h['axes'].items():
                error = target[axis['image_axis']]
                if abs(error) <= .16:
                    continue
                delta = -error * axis['gain_rad'] * axis['sign']
                i = axis['index']
                values[i] = max(self.d['limits']['lower'][i], min(self.d['limits']['upper'][i], values[i] + delta))
            return self.command(s, values, frame, 'follow')
        except (ValueError, TypeError, KeyError) as exc:
            self.fail(exc)
            return None

    def command(self, state, values, frame, kind):
        delta = [abs(a-b) for a, b in zip(values, state['values'])]
        if max(delta) < math.radians(.4):
            return None
        duration = max(.6, max(d/v for d, v in zip(delta, self.d['limits']['max_velocity'])))
        require(duration <= self.h['max_duration_s'], '目标超过已协商的运动时间，不截断执行')
        self.seq += 1
        msg = build(seq=self.seq, values=values, mode='joint_position', dof=self.d['dof'],
                    source=self.source, stamp_ms=int(self.wall()*1000), obs_stamp_ms=frame['obs_ms'], ttl_ms=500)
        msg.update(duration_s=duration, kind=kind, reference={'sample_stamp':state['sample_stamp'],
                   'values':list(state['values']), 'motion_id':state.get('motion_id')},
                   head_session=self.session)
        # Record uncertainty before the publisher is touched. Publication failure
        # cannot cause a second physical send or an automatic return-home.
        self.pending = {'seq': self.seq, 'target':list(values), 'deadline':self.clock()+duration+25}
        return msg
