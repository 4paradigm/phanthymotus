"""Bounded read-only ROS observer; stdout is the evidence stream, never commands."""
import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path


class AsyncRows:
    def __init__(self, output, capacity=8192):
        self.output = output
        self.queue = queue.Queue(capacity)
        self.done = threading.Event()
        self.dropped = 0
        self.written = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def put(self, row):
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            while not self.done.is_set() or not self.queue.empty():
                try:
                    row = self.queue.get(timeout=.02)
                except queue.Empty:
                    continue
                self.output.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                if row.get('event') == 'observer_ready':self.output.flush()
                self.written += 1
        except Exception as exc:
            self.error = str(exc)

    def finish(self):
        self.done.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError('observer_flush_timeout')
        if self.error:
            raise RuntimeError('observer_write_failed:'+self.error)
        self.output.flush()


def command_row(msg, observed_ns):
    return {'event': 'arm_cmd_pos', 'observed_ns': observed_ns,
            'motors': [{'id': int(x.name), 'q_rad': float(x.pos),
                        'speed_rad_s': float(x.spd), 'current': float(x.cur)} for x in msg.cmds],
            'command_sequence_available': False}


def status_row(msg, observed_ns):
    return {'event': 'arm_status', 'observed_ns': observed_ns,
            'motors': [{'id': int(x.name), 'q_rad': float(x.pos),
                        'dq_rad_s': float(x.speed), 'error': int(x.error)} for x in msg.status]}


def power_row(msg, observed_ns):
    return {'event':'power_status','observed_ns':observed_ns,
            'power_on':bool(msg.is_power_on.data),'estop':bool(msg.is_estop.data),
            'remote_estop':bool(msg.is_remote_estop.data)}


def graph_row(node, topics):
    def endpoint(item):
        return {'node':item.node_namespace.rstrip('/')+'/'+item.node_name,
                'type':item.topic_type,'reliability':str(item.qos_profile.reliability),
                'durability':str(item.qos_profile.durability),'depth':item.qos_profile.depth}
    return {'event':'ros_graph','observed_ns':time.monotonic_ns(),
            'nodes':[{'name':name,'namespace':ns} for name,ns in node.get_node_names_and_namespaces()],
            'topics':{topic:{'publishers':[endpoint(x) for x in node.get_publishers_info_by_topic(topic)],
                             'subscribers':[endpoint(x) for x in node.get_subscriptions_info_by_topic(topic)]}
                      for topic in topics}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', type=int, choices=(0, 42), required=True)
    p.add_argument('--namespace', required=True)
    p.add_argument('--seconds', type=float, default=120.)
    args = p.parse_args()
    if not 0 < args.seconds <= 300:
        raise ValueError('observer_duration')
    # Vendor/DDS native loggers write to fd 1, bypassing Python sys.stdout.
    # Keep evidence on its own descriptor before importing ROS/native libraries.
    evidence_output=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    if not os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE'):
        os.environ.pop('FASTRTPS_DEFAULT_PROFILES_FILE',None)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    rclpy.init(domain_id=args.domain)
    node = Node('tianyi_chain_observer_'+str(args.domain))
    writer = AsyncRows(evidence_output)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    qos = QoSProfile(depth=2048, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    def save(convert, msg):
        stamp = time.monotonic_ns()
        writer.put(convert(msg, stamp))
    if args.domain == 0:
        topics=['/arm/cmd_pos','/arm/status','/power/board/key_status','/head/status','/waist/status','/leg/status']
        from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg, PowerBoardKeyStatus
        node.create_subscription(CmdSetMotorPosition, '/arm/cmd_pos',
                                 lambda m: save(command_row, m), qos)
        node.create_subscription(MotorStatusMsg, '/arm/status',
                                 lambda m: save(status_row, m), qos)
        node.create_subscription(PowerBoardKeyStatus, '/power/board/key_status',
                                 lambda m: save(power_row,m),qos)
    else:
        topics=[f'/{args.namespace}/motion/teleop/'+suffix for suffix in ('command','feedback')]
        from std_msgs.msg import String
        def feedback(msg):
            stamp = time.monotonic_ns()
            data = json.loads(msg.data)
            writer.put({'event': 'driver_feedback', 'observed_ns': stamp,
                        'driver_ns': data.get('monotonic_ns'),
                        'state': data.get('state'), 'reason': data.get('reason'),
                        'session_id': data.get('session_id'),
                        'applied_sequence': data.get('applied_sequence'),
                        'hold_confirmed': data.get('hold_confirmed'),
                        'feedback':data.get('feedback'), 'watchdog_timing':data.get('watchdog_timing'),
                        'trace': data.get('trace'),
                        'last_vendor_command': data.get('last_vendor_command')})
        node.create_subscription(String, f'/{args.namespace}/motion/teleop/feedback', feedback, qos)
    writer.put({'event': 'observer_ready', 'observed_ns': time.monotonic_ns(),
                'domain': args.domain, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'hardware_publishers': 0, 'qos': 'best_effort_keep_last_2048',
                'loss_caveat': 'ROS receipt count is not proof of every source publication'})
    end = time.monotonic()+args.seconds
    graph_at=time.monotonic()+min(2.,args.seconds/2)
    graph_recorded=False
    try:
        while not stop.is_set() and time.monotonic() < end:
            if writer.error:
                raise RuntimeError('observer_writer_failed')
            rclpy.spin_once(node, timeout_sec=.02)
            if not graph_recorded and time.monotonic()>=graph_at:
                writer.put(graph_row(node,topics));graph_recorded=True
    finally:
        writer.put(graph_row(node,topics))
        node.destroy_node()
        rclpy.try_shutdown()
        writer.put({'event': 'observer_end', 'observed_ns': time.monotonic_ns(),
                    'queue_dropped': writer.dropped, 'written_before_flush': writer.written})
        writer.finish()
    if writer.dropped:
        raise RuntimeError('observer_evidence_incomplete')


if __name__ == '__main__':
    main()
