"""
topic_subscriber.py — Agent Core 的 DDS 输入订阅管理。

订阅分为两层：
- desired topics：来自持久化配置 / Canvas 的期望集合；
- active subscriptions：当前 ROS2 Node 已成功创建的真实订阅。

动态增删和启动恢复统一经过 reconcile，避免把「配置存在」误报为「已经订阅」。
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
    rclpy = None

_node = None
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_subscriptions: dict[str, object] = {}
_desired_topics: set[str] = set()
_subscription_errors: dict[str, str] = {}
_startup_error = ''
_lock = threading.RLock()


def _normalize_topics(topics) -> set[str]:
    if not isinstance(topics, (list, tuple, set)):
        return set()
    return {
        topic.strip()
        for topic in topics
        if isinstance(topic, str) and topic.strip().startswith('/')
    }


def _create_subscription_locked(topic: str) -> bool:
    """Create one subscription. Caller must hold _lock."""
    if topic in _subscriptions:
        return True
    if _node is None or _loop is None or _loop.is_closed():
        return False

    try:
        import event_bus
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        def _on_message(msg, subscribed_topic=topic):
            try:
                asyncio.run_coroutine_threadsafe(
                    event_bus.enqueue(
                        source=f'dds:{subscribed_topic}', text=msg.data),
                    _loop,
                )
            except RuntimeError:
                log.warning('[topic_sub] event loop unavailable for %s',
                            subscribed_topic)

        subscription = _node.create_subscription(
            String, topic, _on_message, qos)
    except Exception as exc:  # noqa: BLE001
        _subscription_errors[topic] = type(exc).__name__
        log.warning('[topic_sub] subscribe failed for %s: %s',
                    topic, type(exc).__name__)
        return False

    _subscriptions[topic] = subscription
    _subscription_errors.pop(topic, None)
    log.info('[topic_sub] subscribed to %s', topic)
    return True


def _destroy_subscription_locked(topic: str) -> bool:
    """Destroy one active subscription. Caller must hold _lock."""
    subscription = _subscriptions.get(topic)
    if subscription is None:
        return False
    if _node is None:
        _subscriptions.pop(topic, None)
        return True
    try:
        _node.destroy_subscription(subscription)
    except Exception as exc:  # noqa: BLE001
        _subscription_errors[topic] = type(exc).__name__
        log.warning('[topic_sub] unsubscribe failed for %s: %s',
                    topic, type(exc).__name__)
        return False
    _subscriptions.pop(topic, None)
    _subscription_errors.pop(topic, None)
    log.info('[topic_sub] unsubscribed from %s', topic)
    return True


def _reconcile_locked(activate: bool) -> None:
    for topic in list(_subscriptions):
        if topic not in _desired_topics:
            _destroy_subscription_locked(topic)
    if activate and _node is not None:
        for topic in sorted(_desired_topics):
            _create_subscription_locked(topic)


def start(topics: list[str], loop: asyncio.AbstractEventLoop):
    """启动唯一的 ROS2 订阅线程；重复调用只更新期望订阅集合。"""
    global _loop, _thread, _desired_topics, _startup_error
    with _lock:
        _loop = loop
        _desired_topics = _normalize_topics(topics)
        if not _HAS_RCLPY:
            _startup_error = 'rclpy not available'
            log.warning('[topic_sub] rclpy not available, DDS subscription disabled')
            return
        if _thread is not None and _thread.is_alive():
            _reconcile_locked(activate=True)
            return
        _startup_error = ''
        _thread = threading.Thread(
            target=_spin, args=(loop,), daemon=True, name='topic_subscriber')
        _thread.start()


def _spin(loop: asyncio.AbstractEventLoop):
    global _node, _loop, _thread, _startup_error
    try:
        import rclpy as ros
        try:
            ros.init()
        except Exception:
            pass  # ROS context may already be initialized by ros2_bridge

        node = ros.create_node('motus_core_subscriber')
        with _lock:
            _node = node
            _loop = loop
            _startup_error = ''
            _reconcile_locked(activate=True)
            log.info('[topic_sub] ready with %d/%d subscriptions',
                     len(_subscriptions), len(_desired_topics))

        ros.spin(node)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _startup_error = type(exc).__name__
        log.warning('[topic_sub] spin exited: %s', type(exc).__name__)
    finally:
        with _lock:
            node = _node
            _node = None
            _subscriptions.clear()
            _thread = None
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass


def subscribe(topic: str) -> bool:
    """动态订阅；Node 尚未就绪时保留期望项，返回 False 表示尚未真实订阅。"""
    if not isinstance(topic, str) or not topic.strip().startswith('/'):
        return False
    topic = topic.strip()
    with _lock:
        _desired_topics.add(topic)
        return _create_subscription_locked(topic)


def unsubscribe(topic: str) -> bool:
    """动态退订，同时从期望集合移除该 topic。"""
    if not isinstance(topic, str):
        return False
    with _lock:
        _desired_topics.discard(topic)
        return _destroy_subscription_locked(topic)


def reconcile(topics: list[str], *, activate: bool = True) -> dict:
    """将期望集合与真实订阅对齐；activate=False 只清理多余项，不新增订阅。"""
    global _desired_topics
    with _lock:
        _desired_topics = _normalize_topics(topics)
        _reconcile_locked(activate=activate)
        return _snapshot_locked()


def is_subscribed(topic: str) -> bool:
    with _lock:
        return topic in _subscriptions


def snapshot() -> dict:
    with _lock:
        return _snapshot_locked()


def _snapshot_locked() -> dict:
    active = sorted(_subscriptions)
    desired = sorted(_desired_topics)
    return {
        'ready': _node is not None,
        'active_subscriptions': active,
        'desired_subscriptions': desired,
        'pending_subscriptions': [topic for topic in desired if topic not in _subscriptions],
        'subscription_errors': dict(_subscription_errors),
        'startup_error': _startup_error,
    }
