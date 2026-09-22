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
            # 上下文没了就收工，不要接着敲。
            #
            # `rcl_shutdown()` 不一定是 `stop()` 调的 —— 别的组件（插件 teardown、
            # 关机路径里的另一个模块）也会让 context 失效，而那时 `_running` 还是
            # True。原来这里只是 sleep(0.1) 再试，于是变成每秒 10 次打印同一行
            # `failed to create timer: the given context is not valid`，而 context
            # 是不会自己变回有效的。天轶实测：每次关机刷 100 行左右，日志里累计 876 行。
            #
            # 判据用 `rclpy.ok()` 而不是匹配错误文本：文本是 rcl 的实现细节，会随
            # 版本变；`ok()` 问的正是「这个 context 还能用吗」。
            try:
                context_dead = not rclpy.ok()
            except Exception:
                context_dead = True
            if context_dead:
                print('[ros2_bridge] rcl context is gone — spin loop exiting',
                      file=sys.stderr, flush=True)
                return
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
            raw = msg.data
            data = raw if isinstance(raw, bytes) else (raw.encode('utf-8') if isinstance(raw, str) else bytes(raw))
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
    if fmt in ('sensor/pointcloud', 'sensor/mapping'):
        try:
            from std_msgs.msg import UInt8MultiArray
            return UInt8MultiArray
        except ImportError:
            pass
        return None
    # `control/*` is the command stream an execution model sends to a driver
    # (motus.control/1); `state/*` is what a driver reports about itself
    # (motus.odom/1 and friends). Both are JSON inside a std_msgs/String, like
    # the perception cards — spec in phanthymotus-driver/README_dev.md.
    #
    # Neither prefix used to be in this table, and the only consequence was one
    # line on stderr: an unresolved format means the subscription is silently
    # abandoned, so the topic stays registered in inspection, the producer keeps
    # publishing at 10 Hz, and the canvas data-flow panel is empty for ever.
    # That was navi on r1_sz — subscribing with rclpy received the commands
    # while the panel showed nothing.
    if (fmt in ('json', 'data/json') or fmt.startswith('sensor/')
            or fmt.startswith('data/') or fmt.startswith('control/')
            or fmt.startswith('state/')):
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
