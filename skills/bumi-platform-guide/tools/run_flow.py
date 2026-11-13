#!/usr/bin/env python3
"""Opt-in station demo inside the existing Core container; no image rebuild."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
import json
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import time
import urllib.request
from uuid import uuid4

from guide import load_json
from station_flow import FlowError, Interrupted, StationFlow, validate_flow

EOF = b'\x01\x00\xff\xff\x01\x00\xff\xff'


def emit(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


class MCP:
    def __init__(self, url):
        self.url = url

    def request(self, method, params=None):
        body = {'jsonrpc': '2.0', 'id': uuid4().hex, 'method': method, 'params': params or {}}
        req = urllib.request.Request(self.url, data=json.dumps(body, allow_nan=False).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=12) as response:
            packet = json.loads(response.read(2_000_000))
        if not isinstance(packet, dict) or packet.get('error'):
            raise FlowError('MCP请求失败；请核对节点状态，不要自动重试动作')
        return packet['result']

    def tools(self):
        self.request('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {}, 'clientInfo': {'name': 'bumi-station-demo', 'version': '1'}})
        return {t['name']: t for t in self.request('tools/list').get('tools', [])}

    def call(self, tool, **args):
        result = self.request('tools/call', {'name': tool, 'arguments': args})
        if result.get('isError'):
            raise FlowError(f'{tool} 返回MCP错误')
        content = result.get('content', [])
        value = json.loads(next(c['text'] for c in content if c.get('type') == 'text'))
        if not isinstance(value, dict) or value.get('error') or value.get('ok') is False or value.get('state') in ('error', 'loading', 'busy'):
            raise FlowError(f'{tool} 未就绪或执行失败；请核对卡片状态')
        return value


def read_db(path, key):
    path = Path(path).expanduser().resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
        row = conn.execute('SELECT value FROM config WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else {}


def discover(db):
    services = read_db(db, 'services').get('mcp', [])
    result = {}
    for kind, tool in (('driver_url', 'action_recording'), ('perception_url', 'tts')):
        urls = {s.get('url') for s in services if any(isinstance(t, dict) and t.get('name') == tool for t in s.get('tools', []))}
        if len(urls) != 1 or not next(iter(urls)):
            raise FlowError(f'无法唯一发现{tool}；请在流程配置填写实际MCP地址')
        result[kind] = next(iter(urls))
    return result


def topics(info, direction):
    return [t['topic'] for t in info.get(direction, []) if t.get('topic')]


def prepare(db, template):
    raw = dict(template)
    raw.update(discover(db))
    per = MCP(raw['perception_url'])
    per.tools()
    for key, tool in (('vision_topic', 'vop'), ('asr_topic', 'asr'), ('audio_topic', 'tts')):
        found = [t for t in topics(per.call(tool, action='info'), 'topic_out') if 'background' not in t]
        if len(found) != 1:
            raise FlowError(f'{tool} 输出不唯一；请只保留演示所需的一个实例后重试')
        raw[key] = found[0]
    # Import only bound, operator-confirmed calibration evidence, never an estimate.
    driver = MCP(raw['driver_url'])
    tools = driver.tools()
    if 'exercise_routine' in tools:
        info = driver.call('exercise_routine', action='info')
        for key in ('wave', 'point_a'):
            item = raw[key] = dict(raw[key])
            matches = [m for m in info.get('motions', []) if m.get('recording_id') == item['recording_id']]
            if len(matches) == 1:
                evidence = matches[0].get('playback_window') or {}
                if (matches[0].get('verified') is True and evidence.get('source') == 'operator_confirmation'
                        and info.get('robot_id') and evidence.get('robot_id') == info['robot_id']
                        and evidence.get('recording_id') == item['recording_id'] and evidence.get('confirmed_at')):
                    item.update(window_s=evidence.get('seconds'), measured=True)
    return raw


def preflight(settings, db):
    # Stop intelligent control, NOT the Core container; keep sensor/TTS nodes on.
    if read_db(db, 'event').get('subscribe_topics'):
        raise FlowError('Core仍订阅输入；请先停止智能控制，避免两个控制器同时回复')
    driver, per = MCP(settings['driver_url']), MCP(settings['perception_url'])
    dt, pt = driver.tools(), per.tools()
    if not {'action_recording', 'speaker'} <= set(dt) or not {'tts', 'asr', 'vop'} <= set(pt):
        raise FlowError('缺少录制播放、扬声器或感知接口')
    enum = dt['action_recording'].get('inputSchema', {}).get('properties', {}).get('action', {}).get('enum', [])
    if not {'play_recording', 'stop_playback'} <= set(enum):
        raise FlowError('录制播放接口不兼容')
    tts = per.call('tts', action='info')
    if tts.get('state') != 'running' or topics(tts, 'topic_out') != [settings['audio_topic']]:
        raise FlowError('需要且只允许一个运行中的TTS输出，且话题必须匹配')
    speaker = driver.call('speaker', action='info')
    if speaker.get('state') not in ('playing', 'running') or speaker.get('last_error') or settings['audio_topic'] not in topics(speaker, 'topic_in'):
        raise FlowError('speaker未运行或没有连接该TTS音频话题')
    for tool, key in (('vop', 'vision_topic'), ('asr', 'asr_topic')):
        info = per.call(tool, action='info')
        if info.get('state') != 'running' or settings[key] not in topics(info, 'topic_out'):
            raise FlowError(f'{tool}未运行或话题不匹配')
    return driver, per


class Hardware:
    def __init__(self, driver, per, settings):
        self.driver, self.per, self.settings = driver, per, settings
        self.speech_slot = None
        self.motion_owned = False
        self.last_audio = 0.0

    async def _call(self, client, tool, **args):
        return await asyncio.to_thread(client.call, tool, **args)

    def audio(self, data, stamp):
        self.last_audio = time.monotonic()
        slot = self.speech_slot
        if slot is None or stamp < slot['since']:
            return
        if data == EOF:
            slot['eof'].set()
        elif data:
            slot['frames'] += 1

    async def pause(self, seconds, stop, cancel=None):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if stop.is_set() or (cancel and cancel.is_set()):
                raise Interrupted()
            await asyncio.sleep(min(0.05, max(0, until - time.monotonic())))

    async def speak(self, text, stop, cancel=None):
        if self.speech_slot is not None:
            raise FlowError('语音重入')
        slot = self.speech_slot = dict(eof=asyncio.Event(), frames=0, since=time.time())
        try:
            result = await self._call(self.per, 'tts', action='speak', text=text)
            if result.get('status') != 'queued' or not result.get('action_id'):
                raise FlowError('TTS没有返回有效排队回执')
            deadline = time.monotonic() + self.settings['speech_timeout_s']
            while not slot['eof'].is_set():
                if time.monotonic() > deadline:
                    raise FlowError('等待TTS音频EOF超时；没有继续排队')
                await self.pause(0.05, stop, cancel)
            if not slot['frames']:
                raise FlowError('TTS只有EOF而没有音频；不计为完成播报')
            await self.pause(self.settings['audio_tail_s'], stop, cancel)
        except BaseException:
            # Exclusive-output mode only. An uncertain network result is not retried.
            await self._call(self.per, 'tts', action='interrupt')
            raise
        finally:
            self.speech_slot = None

    async def motion(self, motion, stop, during=None):
        if stop.is_set():
            raise Interrupted()
        parallel = None
        try:
            result = await self._call(self.driver, 'action_recording', action='play_recording', recording_id=motion['recording_id'])
            self.motion_owned = result.get('confirmed_started') is True or result.get('command_sent') is True
            if result.get('state') != 'running' or result.get('confirmed_started') is not True:
                raise FlowError('动作启动未确认，停止后续流程')
            if during:
                parallel = asyncio.create_task(during())
            await self.pause(motion['window_s'], stop)
            if parallel and parallel.done():
                await parallel
        except BaseException:
            stop.set()
            if parallel:
                await asyncio.gather(parallel, return_exceptions=True)
            raise
        finally:
            try:
                await self._exit_motion()
            except BaseException:
                if parallel and not parallel.done():
                    stop.set()
                if parallel:
                    await asyncio.gather(parallel, return_exceptions=True)
                raise
        if parallel:
            await parallel

    async def _exit_motion(self):
        if not self.motion_owned:
            return
        result = await self._call(self.driver, 'action_recording', action='stop_playback')
        if result.get('confirmed') is not True or result.get('workmode') != 2:
            raise FlowError('动作退出未确认；请检查机器人，未继续播放')
        self.motion_owned = False

    async def close(self):
        await self._exit_motion()


class RosInputs:
    def __init__(self, flow, hardware, settings):
        import rclpy
        from rclpy.context import Context
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from std_msgs.msg import String
        from audio_msgs.msg import AudioChunk
        self.rclpy, self.flow = rclpy, flow
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('bumi_station_demo_' + uuid4().hex[:8], context=self.context)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        loop = asyncio.get_running_loop()
        self.loop = loop

        def packet(callback, msg):
            try:
                data = json.loads(msg.data)
            except (ValueError, TypeError):
                return
            loop.call_soon_threadsafe(callback, data)

        self.node.create_subscription(String, settings['vision_topic'], lambda m: packet(flow.vision, m), qos)
        self.node.create_subscription(String, settings['asr_topic'], lambda m: packet(flow.speech, m), qos)

        def audio(msg):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            loop.call_soon_threadsafe(hardware.audio, bytes(msg.data), stamp)

        self.node.create_subscription(AudioChunk, settings['audio_topic'], audio, qos)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        try:
            self.rclpy.spin(self.node)
        except Exception:
            if not self.flow.stopped.is_set():
                self.flow.error = 'ROS订阅线程退出'
                self.loop.call_soon_threadsafe(self.flow.stopped.set)

    def close(self):
        self.context.try_shutdown()
        self.thread.join(timeout=3)
        self.node.destroy_node()


async def live(raw, site, db, exclusive):
    if not exclusive:
        raise FlowError('运行需 --exclusive-output：确认演示期间只有本流程使用动作与语音')
    settings = validate_flow(raw)
    driver, per = await asyncio.to_thread(preflight, settings, db)
    hardware = Hardware(driver, per, settings)
    flow = StationFlow(site, settings, hardware, report=emit)
    # One process per Core container; an OS lock also releases on a crash.
    import fcntl
    with open('/tmp/bumi-platform-guide-flow.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FlowError('当前容器已经运行一套站台流程')
        inputs = RosInputs(flow, hardware, settings)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, flow.stopped.set)

        async def watch():
            try:
                while not flow.stopped.is_set():
                    await asyncio.sleep(0.25)
                    last = flow.gate.last_stamp or flow.started
                    if time.time() - last > max(5, settings['vision_max_age_s'] * 2):
                        raise FlowError('视觉数据中断或过期；演示流程已停止')
                    if read_db(db, 'event').get('subscribe_topics'):
                        raise FlowError('智能控制重新订阅输入；演示流程已停止')
            except Exception as exc:
                flow.error = str(exc)
                flow.stopped.set()
        watcher = asyncio.create_task(watch())
        try:
            await flow.run()
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            inputs.close()
        if flow.error:
            raise FlowError(flow.error)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'check', 'run'))
    p.add_argument('--db', required=True, type=Path)
    p.add_argument('--config', type=Path, default=Path('flow.local.json'))
    p.add_argument('--site', type=Path, default=Path('site_config.local.json'))
    p.add_argument('--exclusive-output', action='store_true')
    a = p.parse_args(argv)
    try:
        if a.action == 'prepare':
            if a.config.exists():
                raise FlowError('流程配置已存在，未覆盖；请检查或指定新文件名')
            template = load_json(Path(__file__).resolve().parents[1] / 'flow.example.json')
            raw = prepare(a.db, template)
            with a.config.open('x', encoding='utf-8') as output:
                output.write(json.dumps(raw, ensure_ascii=False, indent=2) + '\n')
            emit(ok=True, config=str(a.config), motion=False, missing_windows=[raw[k]['recording_id'] for k in ('wave', 'point_a') if not raw[k]['measured']])
        else:
            raw, site = load_json(a.config), load_json(a.site)
            if a.action == 'check':
                settings = validate_flow(raw)
                StationFlow(site, settings, None)
                preflight(settings, a.db)
                emit(ok=True, motion=False, reminder_count=3, wave=101, point_a=104)
            else:
                asyncio.run(live(raw, site, a.db, a.exclusive_output))
    except (ValueError, OSError, FlowError, StopIteration, KeyError, ImportError, sqlite3.Error) as exc:
        p.exit(1, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
