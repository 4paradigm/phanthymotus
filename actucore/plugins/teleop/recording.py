"""Bounded asynchronous operator recording; never grants motion authority."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import queue
import secrets
import threading
import time

POSE_KEYS = ('schema_version','mode','client_monotonic_ns','sequence','clutch_sequence','epoch','deadman','tracking','head',
             'left_controller','right_controller','controllers')
FEEDBACK_KEYS = ('monotonic_ns','state','reason','applied_sequence','commanded_q',
                 'output_active','ownership_held','stop_confirmed','calibration_sha256','last_vendor_command','command_state')
SOLUTION_KEYS = ('input_sequence', 'clutch_sequence', 'input_received_ns', 'monotonic_ns',
                 'ik_started_ns', 'ik_succeeded', 'ik_target_q', 'ik_reference_q',
                 'target_diagnostics', 'trajectory_diagnostics', 'output_state', 'published', 'failure')

class PoseRecorder:
    def __init__(self, root, capacity=1024):
        self.root=Path(root);self.capacity=capacity;self.lock=threading.RLock()
        self.thread=None;self.done=threading.Event();self.result={'state':'idle'}

    def start(self, metadata, duration=30, wait_for_deadman=False):
        if type(duration) not in (int,float) or not 1<=duration<=60:raise ValueError('record_duration')
        with self.lock:
            if self.thread and self.thread.is_alive():raise ValueError('recording_active')
            self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
            name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+secrets.token_hex(4)
            self.path=self.root/name;self.path.mkdir(mode=0o700)
            self.queue=queue.Queue(self.capacity);self.done.clear()
            self.duration=duration;self.waiting=wait_for_deadman;self.prelude=None
            self.until=time.monotonic()+(120 if wait_for_deadman else duration)
            self.result=dict(state='armed' if wait_for_deadman else 'recording',recording_id=name,
                             frames=0,dropped=0,solutions=0,solutions_dropped=0,error=None,complete=False)
            (self.path/'manifest.json').write_text(json.dumps({'schema':'motus.teleop.recording.v1',
                'duration_s':duration,'started_monotonic_ns':time.monotonic_ns(),**metadata},allow_nan=False,indent=2)+'\n')
            self.thread=threading.Thread(target=self._write,name='teleop-recorder',daemon=True)
            self.thread.start();return self.status()

    def capture(self, frame, received_at, feedback):
        with self.lock:
            if self.result['state'] not in ('armed','recording') or self.done.is_set():return
            if time.monotonic()>=self.until:self.done.set();return
            row={'received_ns':int(received_at*1e9),'observed_ns':time.monotonic_ns(),'frame':{k:frame[k] for k in POSE_KEYS if k in frame},
                 'driver':{k:feedback[k] for k in FEEDBACK_KEYS if k in feedback}}
            raw=feedback.get('feedback') or {}
            row['driver']['feedback']={k:raw[k] for k in ('q','dq','arm_ns','power_ns','fixed_ns','power_on','estop','fault') if k in raw}
            if getattr(self,'waiting',False):
                tracking=frame.get('tracking',{})
                if not frame.get('deadman') or not all(tracking.get(k) is True for k in ('head','left_controller','right_controller')):
                    self.prelude=copy.deepcopy(row);return
                self.waiting=False;self.until=time.monotonic()+self.duration
                self.result.update(state='recording',trigger_received_ns=row['received_ns'])
                if self.prelude and row['received_ns']-self.prelude['received_ns']<=100_000_000:
                    self.queue.put_nowait(self.prelude)
            try:self.queue.put_nowait(copy.deepcopy(row))
            except queue.Full:self.result['dropped']+=1

    def capture_solution(self, event):
        """Separate post-solve stream joined by input sequence, never by proximity.

        Input observation happens before IK and can include frames overwritten by
        latest-only dispatch. Do not attach a previous solve to a newer pose row.
        """
        with self.lock:
            if self.result['state'] != 'recording' or self.done.is_set():return
            if time.monotonic() >= self.until:self.done.set();return
            row = {k:copy.deepcopy(event[k]) for k in SOLUTION_KEYS if k in event}
            row['_kind'] = 'solution'
            try:self.queue.put_nowait(row)
            except queue.Full:self.result['solutions_dropped']+=1

    def _write(self):
        try:
            with (self.path/'poses.jsonl').open('x') as f, (self.path/'solutions.jsonl').open('x') as solutions:
                while not (self.done.is_set() and self.queue.empty()):
                    if time.monotonic()>=self.until:self.done.set()
                    try:row=self.queue.get(timeout=.05)
                    except queue.Empty:continue
                    is_solution = row.pop('_kind', None) == 'solution'
                    (solutions if is_solution else f).write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                    with self.lock:self.result['solutions' if is_solution else 'frames']+=1
                for stream in (f, solutions):stream.flush();os.fsync(stream.fileno())
            with self.lock:self.result.update(state='finished',complete=(self.result['dropped']==0
                and self.result['solutions_dropped']==0 and self.result['frames']>0))
        except Exception as exc:
            with self.lock:self.result.update(state='failed',error=type(exc).__name__,complete=False)
        finally:
            try:(self.path/'result.json').write_text(json.dumps(self.status(),indent=2)+'\n')
            except OSError:pass

    def status(self):
        with self.lock:return dict(self.result)

    def stop(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=2)
        return self.status()
