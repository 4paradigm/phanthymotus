"""
topic_subscriber.py — 直接用 rclpy 订阅配置的 DDS topic，结果注入 event_bus。

配置（SQLite config table, key='event'）：
  {"subscribe_topics": ["/robot/mic/audio/asr_event", ...]}

每个 topic 收到 String 消息后，enqueue 到 event_bus（source='dds:<topic>'，text=msg.data）。
"""

import asyncio
import logging
import threading

log = logging.getLogger(__name__)

_HAS_RCLPY = False
try:
    import rclpy
    _HAS_RCLPY = True
except ImportError:
    pass

# Module-level state for dynamic subscribe/unsubscribe
_node = None
_loop: asyncio.AbstractEventLoop | None = None
_subscriptions: dict = {}  # topic -> subscription object
_lock = threading.Lock()


def start(topics: list[str], loop: asyncio.AbstractEventLoop):
    """启动后台线程，订阅给定的 DDS topic 列表。"""
    global _loop
    _loop = loop

    if not _HAS_RCLPY:
        log.warning('[topic_sub] rclpy not available, DDS subscription disabled')
        return
    if not topics:
        # Still start the thread so the node is ready for dynamic subscriptions
        t = threading.Thread(target=_spin, args=([], loop), daemon=True, name='topic_subscriber')
        t.start()
        return
    t = threading.Thread(target=_spin, args=(topics, loop), daemon=True, name='topic_subscriber')
    t.start()
    log.info('[topic_sub] subscribing to %d topics: %s', len(topics), topics)


def _spin(topics: list[str], loop: asyncio.AbstractEventLoop):
    global _node
    import event_bus
    from std_msgs.msg import String
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

    try:
        rclpy.init()
    except Exception:
        pass  # already initialized

    _node = rclpy.create_node('motus_core_subscriber')

    for topic in topics:
        sub = _node.create_subscription(
            String,
            topic,
            lambda msg, t=topic: asyncio.run_coroutine_threadsafe(
                event_bus.enqueue(source=f'dds:{t}', text=msg.data),
                loop,
            ),
            qos,
        )
        with _lock:
            _subscriptions[topic] = sub
        log.info('[topic_sub] subscribed to %s', topic)

    try:
        rclpy.spin(_node)
    except Exception as e:
        log.warning('[topic_sub] spin exited: %s', e)
    finally:
        try:
            _node.destroy_node()
        except Exception:
            pass


def subscribe(topic: str):
    """动态订阅单个 topic（运行时调用）。"""
    if not _HAS_RCLPY or _node is None:
        log.warning('[topic_sub] cannot subscribe: rclpy not ready')
        return False

    with _lock:
        if topic in _subscriptions:
            log.info('[topic_sub] already subscribed to %s', topic)
            return True

    import event_bus
    from std_msgs.msg import String
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    loop = _loop

    sub = _node.create_subscription(
        String,
        topic,
        lambda msg, t=topic: asyncio.run_coroutine_threadsafe(
            event_bus.enqueue(source=f'dds:{t}', text=msg.data),
            loop,
        ),
        qos,
    )
    with _lock:
        _subscriptions[topic] = sub
    log.info('[topic_sub] dynamically subscribed to %s', topic)
    return True


def unsubscribe(topic: str):
    """动态退订单个 topic（运行时调用）。"""
    with _lock:
        sub = _subscriptions.pop(topic, None)

    if sub is None:
        log.info('[topic_sub] not subscribed to %s, nothing to remove', topic)
        return False

    if _node is not None:
        _node.destroy_subscription(sub)
    log.info('[topic_sub] dynamically unsubscribed from %s', topic)
    return True
