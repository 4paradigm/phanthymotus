"""
ros2_bridge.py — rclpy 订阅桥接层。

在独立线程中运行 rclpy，将 ROS2 topic 数据回调给 asyncio 协程。

用法:
    import ros2_bridge

    # 在 lifespan 启动时
    ros2_bridge.start()

    # 订阅 topic，cb 会在 asyncio 事件循环中被调用
    ros2_bridge.subscribe(mcp_id, topic, fmt, loop, cb)

    # 取消订阅
    ros2_bridge.unsubscribe(mcp_id)

    # 在 lifespan 关闭时
    ros2_bridge.stop()

cb 签名: async def cb(data: bytes, fmt: str) -> None
"""

import asyncio
import json
import math
import struct
import sys
import threading

_lock            = threading.Lock()
_subs: dict      = {}   # mcp_id → {'node': Node, 'sub': Subscription}
_last_seen: dict = {}   # topic → timestamp of last received message
_node_main       = None
_executor        = None
_thread          = None
_running         = False
_loop            = None  # asyncio event loop captured at start()

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False
    print('[ros2_bridge] rclpy not available — bus streaming disabled', file=sys.stderr)


def _low_lat_qos():
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        durability=DurabilityPolicy.VOLATILE,
    )


def start(loop: asyncio.AbstractEventLoop = None) -> None:
    global _thread, _running, _node_main, _loop, _executor
    if not _HAS_RCLPY:
        return
    if _running:
        return
    _running = True
    _loop = loop
    rclpy.init()
    _node_main = rclpy.create_node('phanthy_bus_bridge')
    _executor = rclpy.executors.MultiThreadedExecutor()
    _executor.add_node(_node_main)
    _thread = threading.Thread(target=_spin_loop, daemon=True, name='ros2_bridge')
    _thread.start()
    print('[ros2_bridge] started')


def _spin_loop() -> None:
    while _running:
        try:
            _executor.spin_once(timeout_sec=0.05)
        except Exception as e:
            print(f'[ros2_bridge] spin error: {e}', file=sys.stderr, flush=True)
            import time as _t
            _t.sleep(0.1)  # avoid tight loop on persistent errors


def stop() -> None:
    global _running, _executor
    if not _HAS_RCLPY:
        return
    _running = False
    with _lock:
        for info in _subs.values():
            try:
                _node_main.destroy_subscription(info['sub'])
            except Exception:
                pass
        _subs.clear()
    if _executor:
        _executor.shutdown()
        _executor = None
    if _node_main:
        _node_main.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass
    print('[ros2_bridge] stopped')


def subscribe(mcp_id: str, topic: str, fmt: str, loop: asyncio.AbstractEventLoop, cb) -> None:
    """订阅 topic；每收到一帧就在 loop 中 schedule cb(data: bytes, fmt: str)。"""
    if not _HAS_RCLPY or not _running:
        return

    # Always use the loop captured at start() — the caller's loop may not be running
    # when run_coroutine_threadsafe is called from the rclpy spin thread.
    _cb_loop = _loop or loop

    unsubscribe(mcp_id)  # 先清掉旧的

    import time as _time

    msg_type = _resolve_msg_type(fmt)
    if msg_type is None:
        print(f'[ros2_bridge] unsupported format {fmt!r} for topic {topic}', file=sys.stderr)
        return

    def _on_msg(msg):
        try:
            data = _encode_message(msg, fmt)
            msg_fmt = getattr(msg, 'format', fmt)
        except Exception as e:
            print(f'[ros2_bridge] decode error: {repr(e)}', file=sys.stderr)
            return
        _last_seen[topic] = _time.time()
        asyncio.run_coroutine_threadsafe(cb(data, msg_fmt), _cb_loop)

    sub = _node_main.create_subscription(msg_type, topic, _on_msg, _low_lat_qos())
    with _lock:
        _subs[mcp_id] = {'sub': sub, 'topic': topic, 'fmt': fmt}
    if _executor:
        _executor.wake()
    print(f'[ros2_bridge] subscribed mcp_id={mcp_id} topic={topic}')


def unsubscribe(mcp_id: str) -> None:
    if not _HAS_RCLPY:
        return
    with _lock:
        info = _subs.pop(mcp_id, None)
    if info:
        try:
            _node_main.destroy_subscription(info['sub'])
        except Exception:
            pass
        print(f'[ros2_bridge] unsubscribed mcp_id={mcp_id}')


def get_last_seen(topic: str) -> float:
    """Return timestamp of last received message for topic, or 0.0 if never seen."""
    return _last_seen.get(topic, 0.0)


def get_dds_topics() -> set:
    """Return set of topic names currently advertised in the DDS network."""
    if not _HAS_RCLPY or not _running or not _node_main:
        return set()
    try:
        return {name for name, _ in _node_main.get_topic_names_and_types()}
    except Exception:
        return set()


# ── Publishing ─────────────────────────────────────────────────────────────────

_publishers: dict = {}  # topic → Publisher


def publish(topic: str, data: str) -> None:
    """发布 String 消息到指定 DDS topic。"""
    if not _HAS_RCLPY or not _running or not _node_main:
        print(f'[ros2_bridge] publish skipped: rclpy={_HAS_RCLPY} running={_running} node={_node_main is not None}')
        return
    if topic not in _publishers:
        from std_msgs.msg import String
        _publishers[topic] = _node_main.create_publisher(String, topic, 10)
    from std_msgs.msg import String
    msg = String()
    msg.data = data
    _publishers[topic].publish(msg)


def _stamp_ns(header) -> int:
    stamp = header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _yaw_from_quaternion(orientation) -> float:
    sin_yaw = 2.0 * (
        float(orientation.w) * float(orientation.z)
        + float(orientation.x) * float(orientation.y)
    )
    cos_yaw = 1.0 - 2.0 * (
        float(orientation.y) ** 2 + float(orientation.z) ** 2
    )
    return math.atan2(sin_yaw, cos_yaw)


def _odometry_payload(msg) -> dict:
    pose = msg.pose.pose
    twist = msg.twist.twist
    return {
        'schema': 'phanthy.sensor.odometry.v1',
        'frame_id': str(msg.header.frame_id),
        'child_frame_id': str(msg.child_frame_id),
        'stamp_ns': _stamp_ns(msg.header),
        'position': {
            'x': float(pose.position.x),
            'y': float(pose.position.y),
            'z': float(pose.position.z),
        },
        'yaw': _yaw_from_quaternion(pose.orientation),
        'linear_velocity': {
            'x': float(twist.linear.x),
            'y': float(twist.linear.y),
            'z': float(twist.linear.z),
        },
        'angular_velocity': {
            'x': float(twist.angular.x),
            'y': float(twist.angular.y),
            'z': float(twist.angular.z),
        },
    }


def _path_payload(msg) -> dict:
    poses = []
    for stamped_pose in msg.poses:
        pose = stamped_pose.pose
        poses.append({
            'x': float(pose.position.x),
            'y': float(pose.position.y),
            'z': float(pose.position.z),
            'yaw': _yaw_from_quaternion(pose.orientation),
        })
    return {
        'schema': 'phanthy.navigation.path.v1',
        'frame_id': str(msg.header.frame_id),
        'stamp_ns': _stamp_ns(msg.header),
        'poses': poses,
    }


def _occupancy_grid_payload(msg) -> dict:
    info = msg.info
    origin = info.origin
    return {
        'schema': 'phanthy.navigation.costmap.v1',
        'frame_id': str(msg.header.frame_id),
        'stamp_ns': _stamp_ns(msg.header),
        'resolution': float(info.resolution),
        'width': int(info.width),
        'height': int(info.height),
        'origin': {
            'x': float(origin.position.x),
            'y': float(origin.position.y),
            'yaw': _yaw_from_quaternion(origin.orientation),
        },
        'data': [int(value) for value in msg.data],
    }


def _pointcloud_payload(msg) -> bytes:
    point_step = int(msg.point_step)
    width = int(msg.width)
    height = int(msg.height)
    if bool(msg.is_bigendian):
        raise ValueError('big-endian PointCloud2 is unsupported')
    if not 12 <= point_step < 256 or width < 0 or height < 0:
        raise ValueError('invalid PointCloud2 dimensions')
    layout = {
        str(field.name): (int(field.offset), int(field.datatype), int(field.count))
        for field in msg.fields
        if str(field.name) in {'x', 'y', 'z'}
    }
    expected_layout = {
        'x': (0, 7, 1),
        'y': (4, 7, 1),
        'z': (8, 7, 1),
    }
    if layout != expected_layout:
        raise ValueError('PointCloud2 requires leading float32 x/y/z fields')

    row_bytes = width * point_step
    row_step = int(msg.row_step) or row_bytes
    if row_step < row_bytes:
        raise ValueError('PointCloud2 row_step is too small')
    raw = bytes(msg.data)
    required = (height - 1) * row_step + row_bytes if height else 0
    if len(raw) < required:
        raise ValueError('PointCloud2 data is truncated')
    if row_step == row_bytes:
        packed = raw[:width * height * point_step]
    else:
        packed = b''.join(
            raw[row * row_step:row * row_step + row_bytes]
            for row in range(height)
        )
    return struct.pack('<II', point_step, width * height) + packed


def _encode_message(msg, fmt: str) -> bytes:
    """Convert supported ROS messages to the dashboard wire representation."""
    if fmt == 'sensor/pointcloud':
        return _pointcloud_payload(msg)
    if fmt == 'sensor/odometry':
        return json.dumps(
            _odometry_payload(msg), separators=(',', ':'), allow_nan=False
        ).encode('utf-8')
    if fmt == 'sensor/path':
        return json.dumps(
            _path_payload(msg), separators=(',', ':'), allow_nan=False
        ).encode('utf-8')
    if fmt in ('sensor/costmap', 'sensor/occupancy-grid'):
        return json.dumps(
            _occupancy_grid_payload(msg), separators=(',', ':'), allow_nan=False
        ).encode('utf-8')

    raw = msg.data
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return raw.encode('utf-8')
    return bytes(raw)


def _resolve_msg_type(fmt: str):
    """根据 data format 返回对应的 ROS2 消息类型，未知返回 None。"""
    if fmt.startswith('audio/'):
        try:
            from audio_msgs.msg import AudioChunk
            return AudioChunk
        except ImportError:
            pass
        # Unitree G1: DDS AudioData_ — 不经 ROS2，此处不需要订阅
        return None
    if fmt == 'sensor/pointcloud':
        try:
            from sensor_msgs.msg import PointCloud2
            return PointCloud2
        except ImportError:
            pass
        return None
    if fmt == 'sensor/mapping':
        try:
            from std_msgs.msg import UInt8MultiArray
            return UInt8MultiArray
        except ImportError:
            pass
        return None
    if fmt == 'sensor/odometry':
        try:
            from nav_msgs.msg import Odometry
            return Odometry
        except ImportError:
            pass
        return None
    if fmt == 'sensor/path':
        try:
            from nav_msgs.msg import Path
            return Path
        except ImportError:
            pass
        return None
    if fmt in ('sensor/costmap', 'sensor/occupancy-grid'):
        try:
            from nav_msgs.msg import OccupancyGrid
            return OccupancyGrid
        except ImportError:
            pass
        return None
    if fmt in ('json', 'data/json') or fmt.startswith('sensor/') or fmt.startswith('data/'):
        try:
            from std_msgs.msg import String
            return String
        except ImportError:
            pass
        return None
    # 未来扩展：video/, sensor/ 等
    if fmt == 'image/jpeg':
        try:
            from sensor_msgs.msg import CompressedImage
            return CompressedImage
        except ImportError as e:
            print(f'[ros2_bridge] cannot import CompressedImage: {e}', file=sys.stderr)
            pass
        return None
    if fmt == 'image/depth-z16':
        try:
            from sensor_msgs.msg import Image
            return Image
        except ImportError as e:
            print(f'[ros2_bridge] cannot import Image: {e}', file=sys.stderr)
            pass
        return None
    if fmt == 'image/depth-zlib':
        try:
            from sensor_msgs.msg import CompressedImage
            return CompressedImage
        except ImportError as e:
            print(f'[ros2_bridge] cannot import CompressedImage: {e}', file=sys.stderr)
            pass
        return None
    return None
