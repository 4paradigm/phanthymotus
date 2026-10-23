"""Deterministic, opt-in station demo. No device calls at import time.

The transport owns the exclusive demo output. Motion completion means an
operator-measured window followed by a confirmed playback exit, never an SDK
physical-completion signal. Speech completion means PCM EOF, not audibility.
"""
from __future__ import annotations

import asyncio
from collections import deque
import math
import re
import time

from guide import effective

WELCOME = '您好，我可以提供车厢问路和候车提醒。'
REMINDER = '请在安全区域候车，留意现场广播。请先下后上，注意列车与站台之间的间隙。'
UNCONFIRMED = '当前车厢方向尚未确认，请查看站台显示屏或咨询现场工作人员。'
STOP_WORDS = {'结束站台引导', '停止站台引导', '结束值守'}


class FlowError(RuntimeError):
    pass


class Interrupted(Exception):
    pass


def number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} 必须为 {low}..{high} 的有限数值')
    return float(value)


def normalize(text):
    return re.sub(r'[\W_]+', '', text).lower()


def carriage(text):
    """Only explicit carriage questions, never an arbitrary number in speech."""
    match = re.search(r'(?:第)?([0-9零〇一二两三四五六七八九十百]+)\s*(?:号\s*)?(?:车厢|车)', text)
    if not match:
        return None
    token = match[1]
    if token.isascii() and token.isdigit():
        return int(token) if len(token) <= 3 else 101
    digits = dict(zip('零〇一二两三四五六七八九', (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    if token in digits:
        return digits[token]
    if token == '一百':
        return 100
    if token.count('十') == 1:
        left, right = token.split('十')
        if (not left or left in digits) and (not right or right in digits):
            return (digits[left] if left else 1) * 10 + (digits[right] if right else 0)
    return None


def validate_flow(raw):
    allowed = {'driver_url', 'perception_url', 'vision_topic', 'asr_topic', 'audio_topic',
               'wave', 'point_a', 'person_hold_s', 'absent_rearm_s', 'vision_max_age_s',
               'reminder_gap_s', 'speech_timeout_s', 'audio_tail_s'}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError('流程配置存在未知字段')
    result = dict(raw)
    from urllib.parse import urlparse
    for key in ('driver_url', 'perception_url'):
        url = urlparse(raw.get(key) or '')
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.username or url.password:
            raise ValueError(f'{key} 必须为机器人本机的 HTTP MCP 地址')
    for key in ('vision_topic', 'asr_topic', 'audio_topic'):
        if not isinstance(raw.get(key), str) or not re.fullmatch(r'/[A-Za-z0-9_/]+', raw[key]):
            raise ValueError(f'{key} 必须为现场核实的 ROS 话题')
    if 'background' in raw['asr_topic'] or len({raw[k] for k in ('vision_topic', 'asr_topic', 'audio_topic')}) != 3:
        raise ValueError('请使用 ASR 主输出，三个输入话题不能相同')
    for key, rid in (('wave', 101), ('point_a', 104)):
        item = raw.get(key)
        if not isinstance(item, dict) or set(item) != {'recording_id', 'window_s', 'measured'}:
            raise ValueError(f'{key} 需要 recording_id/window_s/measured')
        if type(item['recording_id']) is not int or item['recording_id'] != rid:
            raise ValueError(f'{key} 只允许当前已确认的录制编号 {rid}')
        if item['measured'] is not True:
            raise ValueError(f'{rid} 尚未确认实测窗口；请现场完整回放并测时')
        number(item['window_s'], f'{rid}.window_s', 0.1, 120)
    for key, default, low, high in (
        ('person_hold_s', 1, 0.2, 10), ('absent_rearm_s', 5, 1, 60),
        ('vision_max_age_s', 2, 0.2, 10), ('reminder_gap_s', 2, 0, 30),
        ('speech_timeout_s', 60, 5, 120), ('audio_tail_s', 0.5, 0.1, 3),
    ):
        result[key] = number(raw.get(key, default), key, low, high)
    return result


class PresenceGate:
    """One welcome per occupied scene. Missing/stale frames never mean empty."""
    def __init__(self, hold_s=1, rearm_s=5, max_age_s=2):
        self.hold_s, self.rearm_s, self.max_age_s = hold_s, rearm_s, max_age_s
        self.last_stamp = 0.0
        self.seen_since = self.empty_since = None
        self.occupied = False

    def update(self, packet, now):
        if not isinstance(packet, dict):
            return False
        stamp = packet.get('timestamp')
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= now - stamp <= self.max_age_s or stamp <= self.last_stamp:
            return False
        objects = packet.get('objects')
        if not isinstance(objects, list):
            return False
        if self.last_stamp and stamp - self.last_stamp > self.max_age_s:
            self.seen_since = self.empty_since = None
        self.last_stamp = stamp
        person = any(isinstance(o, dict) and o.get('name') == 'person'
                     and type(o.get('confidence')) in (int, float)
                     and math.isfinite(o['confidence']) and 0.5 <= o['confidence'] <= 1 for o in objects)
        if person:
            self.empty_since = None
            if self.seen_since is None:
                self.seen_since = stamp
            if not self.occupied and stamp - self.seen_since >= self.hold_s:
                self.occupied = True
                return True
        else:
            self.seen_since = None
            if self.empty_since is None:
                self.empty_since = stamp
            if stamp - self.empty_since >= self.rearm_s:
                self.occupied = False
        return False


class StationFlow:
    def __init__(self, site, settings, io, *, clock=time.time, report=lambda **kw: None):
        self.site = effective(site)
        if self.site['config']['mode'] != 'demo' or self.site['config']['carriage_count'] != 16:
            raise ValueError('本流程仅支持已确认的室内 demo，不支持真实站台自动运行')
        if not self.site['directions_enabled'] or self.site['direction_table'].get('5') != 'A标识方向':
            raise ValueError('需要已确认的模拟车厢配置，且5车指向A标识方向')
        self.settings, self.io, self.clock, self.report = settings, io, clock, report
        self.gate = PresenceGate(settings['person_hold_s'], settings['absent_rearm_s'], settings['vision_max_age_s'])
        self.started = clock()
        self.last_asr = 0.0
        self.state = 'idle'
        self.queue = asyncio.Queue(maxsize=2)
        self.stopped = asyncio.Event()
        self.cut_reminders = asyncio.Event()
        self.spoken = deque(maxlen=12)
        self.broadcasts = 0
        self.error = None
        self.worker = None
        self.site_invalidated = False

    def _state(self, value):
        self.state = value
        self.report(state=value, broadcasts=self.broadcasts)

    def vision(self, packet):
        if self.stopped.is_set():
            return
        if self.gate.update(packet, self.clock()) and self.state == 'idle' and self.queue.empty():
            self.queue.put_nowait(('welcome', None))

    def speech(self, packet):
        if self.stopped.is_set() or not isinstance(packet, dict):
            return
        text, stamp = packet.get('text'), packet.get('audio_end_ts')
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            return
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or not self.started <= stamp <= self.clock() + 0.5 or self.clock() - stamp > 10 or stamp <= self.last_asr:
            return
        self.last_asr = stamp
        clean = normalize(text)
        # Reject our own recently spoken text, including joined/split ASR sentences.
        for said, when in self.spoken:
            if self.clock() - when < 120 and len(clean) >= 5 and (clean in said or said in clean):
                return
        if clean in STOP_WORDS:
            self.stopped.set()
            return
        if re.search(r'更换|换车|换站|编组|标识.*(?:变|改)|方向.*(?:变|改)', text):
            self.site_invalidated = True
        if self.state == 'reminding':
            self.cut_reminders.set()
        if self.queue.full():
            self.report(event='question_queue_full')
            return
        self.queue.put_nowait(('question', text.strip()))

    async def _say(self, text, cancel=None):
        if self.stopped.is_set():
            raise Interrupted()
        self.spoken.append((normalize(text), self.clock()))
        await self.io.speak(text, self.stopped, cancel)

    async def _welcome(self):
        self._state('waving')
        await self.io.motion(self.settings['wave'], self.stopped)
        self._state('welcoming')
        await self._say(WELCOME)

    async def _answer(self, text):
        self.broadcasts = 0
        self._state('answering')
        if self.site_invalidated:
            await self._say(UNCONFIRMED)
            return
        car = carriage(text)
        if car is None:
            await self._say('请问您要前往几号车厢？')
            return
        if not 1 <= car <= self.site['config']['carriage_count']:
            await self._say(f'模拟列车共{self.site["config"]["carriage_count"]}节车厢，您询问的车厢超出配置，请咨询现场工作人员。')
            return
        direction = self.site['direction_table'].get(str(car))
        if not direction:
            await self._say(UNCONFIRMED)
            return
        answer = f'模拟场景，请沿{direction}前往。'
        if direction == 'A标识方向':
            self._state('pointing')
            # Transport starts speech only after the motion start was confirmed.
            await self.io.motion(self.settings['point_a'], self.stopped, lambda: self._say(answer))
        else:
            await self._say(answer)  # Never reuse 104 for B or an unknown direction.
        self._state('reminding')
        self.cut_reminders.clear()
        # A question that arrived during pointing already has priority.
        if not self.queue.empty():
            return
        for index in range(3):
            if self.stopped.is_set() or self.cut_reminders.is_set():
                break
            try:
                await self._say(REMINDER, self.cut_reminders)
            except Interrupted:
                break
            self.broadcasts += 1
            self.report(state='reminding', broadcasts=self.broadcasts)
            if index < 2:
                await self.io.pause(self.settings['reminder_gap_s'], self.stopped, self.cut_reminders)

    async def run(self):
        self.worker = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                try:
                    kind, text = await asyncio.wait_for(self.queue.get(), 0.1)
                except asyncio.TimeoutError:
                    continue
                try:
                    if kind == 'welcome':
                        await self._welcome()
                    else:
                        await self._answer(text)
                except Interrupted:
                    pass
                finally:
                    self.queue.task_done()
                if not self.stopped.is_set():
                    self._state('idle')
        except Exception as exc:
            self.error = str(exc)
            self.stopped.set()
            self._state('error')
        finally:
            self.stopped.set()
            try:
                await self.io.close()
            except Exception as exc:
                self.error = self.error or str(exc)
            self._state('error' if self.error else 'stopped')

    async def stop(self):
        self.stopped.set()
        if self.worker and self.worker is not asyncio.current_task():
            await self.worker
