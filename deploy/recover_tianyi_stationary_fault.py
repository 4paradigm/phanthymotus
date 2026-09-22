#!/usr/bin/python3 -I
"""One-shot operator maintenance; preserves failed stop evidence, never claims motion.

This is deliberately NOT added to sudoers or the normal release API. It only
handles the exact reviewed fault/container/Compose generation below.
"""
import argparse
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import time

COMPOSE_SHA = '87651be3626eb0c6bad224e04ede3feca3497776ea1f7d5f676e41c49744abcf'
BOOT = 'c87c72cf43e0e0c671ef11ffbb40b7f7'
LEASE = 'e35c86ac51fc77effe79ee125092c71d'
OLD_DRIVER = 'sha256:e5263fd4375903049734272b106594a5ff3b6d9cc9a6d130c15ca54a46fedaf4'
IMAGES = {'actucore-teleop': 'sha256:ff59c66673e4eaff552ccdccf7cccd31f25f5d068b92d293c7744d0127245b7e',
          'tianyi2': 'sha256:87fbef18b0b6d3fc5c5fc54641ce59a4f54d9849b78ad0160b6376bc118289bc'}
EVIDENCE = Path('/data/hanzebei/pico-tianyi/updates/replay')


def vector(v):
    if not isinstance(v, list) or len(v) != 14 or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in v):
        raise ValueError('invalid_joint_feedback')
    return v


def sample(d, *, old):
    f = d['feedback']; now = d['monotonic_ns']
    if type(now) is not int:
        raise ValueError('invalid_feedback_clock')
    for key in ('arm_ns', 'power_ns', 'fixed_ns'):
        if type(f.get(key)) is not int or not 0 <= now-f[key] <= 100_000_000:
            raise ValueError('stale_feedback')
    if any(f.get(k) is not value for k, value in
           (('power_on', True), ('estop', False), ('fault', False), ('fixed_body', True))):
        raise ValueError('unsafe_feedback')
    if d.get('foreign_publishers') != [] or d.get('hands_enabled') is not False or d.get('feedback_executor_error'):
        raise ValueError('conflicting_or_failed_executor')
    q = vector(f['q']); dq = vector(f['dq'])
    if max(map(abs, dq)) > .02:
        raise ValueError('arms_moving')
    if old:
        if (d.get('boot_id'), d.get('session_id'), d.get('state'), d.get('reason')) != (BOOT, LEASE, 'fault', 'stop_not_confirmed'):
            raise ValueError('different_fault_generation')
        if d.get('ownership_held') is not True:
            raise ValueError('ownership_changed')
        command = d['last_vendor_command']
        if now-command['monotonic_ns'] < 10_000_000_000:
            raise ValueError('recent_vendor_command')
        if max(abs(a-b) for a,b in zip(q, vector(command['q_rad']))) > .1:
            raise ValueError('unbounded_settling')
    else:
        if d.get('boot_id') == BOOT or d.get('state') != 'idle' or d.get('ownership_held') is not False or d.get('output_active') is not False:
            raise ValueError('new_driver_not_idle')
        if d.get('last_vendor_command') is not None:
            raise ValueError('unexpected_hardware_write')
    return {'arm_ns': f['arm_ns'], 'q': list(q), 'dq': list(dq), 'monotonic_ns': now}


def stationary(rows):
    if len(rows) < 20 or rows[-1]['arm_ns']-rows[0]['arm_ns'] < 2_000_000_000:
        return False
    if any(not 0 < b['arm_ns']-a['arm_ns'] <= 150_000_000 for a,b in zip(rows, rows[1:])):
        raise ValueError('noncontinuous_feedback')
    if max(max(r['q'][j] for r in rows)-min(r['q'][j] for r in rows) for j in range(14)) > .002:
        raise ValueError('position_drift')
    return True


def observe(h, *, old):
    rows = []; deadline = time.monotonic()+10; command_stamp = None
    while time.monotonic() < deadline:
        d = h.info(15707, 'teleop_executor')
        r = sample(d, old=old)
        if old:
            stamp = d['last_vendor_command']['monotonic_ns']
            if command_stamp is not None and stamp != command_stamp:
                raise ValueError('vendor_command_changed')
            command_stamp = stamp
        if rows and r['arm_ns']-rows[-1]['arm_ns'] > 150_000_000:
            rows = []  # Start a NEW continuous window; never bridge the gap.
        if not rows or r['arm_ns'] != rows[-1]['arm_ns']:
            rows.append(r)
        if stationary(rows):
            return rows, d
        time.sleep(.05)
    raise ValueError('stationary_window_missing')


def idle_inputs(h):
    e = h.core('/canvas/edit-status?session_id=teleop-publish')
    if e.get('code') != 200 or e.get('editor') is not None or h.core('/config/project-running').get('running') is not False:
        raise ValueError('canvas_occupied')
    v = h.info(15740, 'teleop'); dispatch = v.get('dispatch') or {}
    if v.get('mode') != 'shadow' or v.get('state') != 'idle' or v.get('authority_valid') is not False:
        raise ValueError('card_not_isolated')
    if any(dispatch.get(k) for k in ('io_inflight', 'mailbox_depth', 'stop_queue_depth')):
        raise ValueError('input_pending')
    for name in ('arm', 'servo'):
        # This deployed bundle guards even info under the same lease. Require
        # its exact refusal, not a generic failed request treated as idle.
        r = h.get('http://127.0.0.1:15707/mcp', {'jsonrpc':'2.0','id':1,
                  'method':'tools/call','params':{'name':name,'arguments':{'action':'info'}}})
        if r.get('error') or r.get('result',{}).get('isError') is not True:
            raise ValueError('legacy_exclusion_not_confirmed')
        v = json.loads(r['result']['content'][0]['text'])
        if v.get('code') != 'motion_owned_by_teleop' or v.get('error') != 'motion_owned_by_teleop':
            raise ValueError('unexpected_legacy_error')


def run(h, apply=False):
    before = h.COMPOSE.read_text()
    if hashlib.sha256(before.encode()).hexdigest() != COMPOSE_SHA:
        raise ValueError('compose_changed')
    if h.yaml.safe_load(h.CONFIG.read_text())['plugins']['teleop']['mode'] != 'shadow':
        raise ValueError('startup_not_shadow')
    for service, digest in IMAGES.items():
        image = json.loads(h.shell('/usr/bin/docker', 'image', 'inspect', digest))[0]
        if image['Id'] != digest or image['Config']['Labels']['org.phanthy.teleop.service'] != service:
            raise ValueError('candidate_mismatch')
    if h.shell('/usr/bin/docker', 'inspect', '--format', '{{.Image}}', h.SERVICES['tianyi2']) != OLD_DRIVER:
        raise ValueError('old_driver_changed')
    idle_inputs(h)
    rows, fault = observe(h, old=True)
    print('MAINTENANCE PREFLIGHT PASS: stationary evidence; old stop remains FAILED', flush=True)
    if not apply:
        return
    if os.geteuid() != 0:
        raise ValueError('operator_sudo_required')
    h.protected(h.COMPOSE); h.protected(h.STATE)
    if EVIDENCE.resolve() != EVIDENCE or not EVIDENCE.is_dir():
        raise ValueError('evidence_path_changed')
    after = h.replace_images(before, IMAGES)
    with (h.STATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle_inputs(h)
        rows, fault = observe(h, old=True)
        others = h.containers(h.OTHERS)
        path = EVIDENCE/('maintenance-'+str(time.time_ns())+'.json')
        receipt = {'kind':'operator_stationary_maintenance', 'old_stop_passed':False,
                   'fault':fault, 'before_samples':rows, 'images':IMAGES,
                   'status':'prepared', 'compose_before':before}
        # Keep one exclusive descriptor: do not reopen a user-owned path as root.
        stream = path.open('x')
        os.fchmod(stream.fileno(), 0o600)
        owner = EVIDENCE.stat(); os.fchown(stream.fileno(), owner.st_uid, owner.st_gid)
        def save():
            stream.seek(0); stream.truncate()
            stream.write(json.dumps(receipt, indent=2)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        save()
        if h.COMPOSE.read_text() != before:
            raise ValueError('compose_changed')
        h.shell('/usr/bin/docker', 'stop', h.SERVICES['actucore-teleop'])
        # Card shutdown must not result in a new arm command or different lease.
        final = h.info(15707, 'teleop_executor'); sample(final, old=True)
        if final['last_vendor_command'] != fault['last_vendor_command']:
            raise ValueError('command_changed_during_isolation')
        if h.COMPOSE.read_text() != before or h.containers(h.OTHERS) != others:
            raise ValueError('deployment_changed')
        h.atomic(h.COMPOSE, after)
        for service in ('tianyi2', 'actucore-teleop'):
            h.shell('/usr/bin/docker','compose','-p','phanthy-motus','-f',str(h.COMPOSE),
                    'up','-d','--no-deps','--pull','never',service)
        deadline = time.monotonic()+45
        while True:
            try:
                h.guard(); break
            except Exception:
                if time.monotonic() >= deadline: raise
                time.sleep(1)
        post, _ = observe(h, old=False)
        if max(abs(a-b) for a,b in zip(post[-1]['q'], rows[-1]['q'])) > .002:
            raise ValueError('pose_changed_during_maintenance')
        for service, digest in IMAGES.items():
            if h.shell('/usr/bin/docker','inspect','--format','{{.Image}}',h.SERVICES[service]) != digest:
                raise ValueError('running_image_mismatch')
        if h.containers(h.OTHERS) != others or h.COMPOSE.read_text() != after:
            raise ValueError('other_service_changed')
        receipt.update(status='maintenance_complete_not_motion_acceptance', after_samples=post)
        save()
        print('MAINTENANCE SHADOW PASS: old stop FAILED preserved; no claim or motion', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    path = '/usr/local/sbin/tianyi-teleop-publish'
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != 'f53d201d25480f53d25ac81444864761adf5aaea7e4ad8b1635aa9ce8bb21e75':
        raise ValueError('publisher_changed')
    loader = importlib.machinery.SourceFileLoader('publisher', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    h = importlib.util.module_from_spec(spec); loader.exec_module(h)
    if os.geteuid() != 0:
        h.ENV = {**h.ENV, 'DOCKER_CONFIG':str(EVIDENCE)}
    run(h, args.apply)


if __name__ == '__main__':
    main()
