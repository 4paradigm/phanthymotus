"""
utils/ros_lifecycle.py — Safe teardown of per-instance ROS2 nodes.

Multi-instance plugins create one rclpy Node per input topic and register it
with the bundle's MultiThreadedExecutor. When an instance is retired (topic
change, stop-all, shutdown) the node must be removed from the executor *and*
destroyed, otherwise its subscriptions keep matching camera publishers and
its callbacks keep waking the executor for the rest of the process. The ASR
plugin established this order; OCR and obstacle reuse it here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import threading
from contextlib import contextmanager

log = logging.getLogger(__name__)


@dataclass
class NodeDisposal:
    """Caller-owned progress; retain it with the resource until complete."""

    removed: bool = False
    destroyed: bool = False
    error: str | None = None
    _lock: object = field(default_factory=threading.RLock, repr=False, compare=False)

    @property
    def complete(self) -> bool:
        return self.removed and self.destroyed


def dispose_node(executor, node, *, label: str = "",
                 progress: NodeDisposal | None = None) -> NodeDisposal:
    """Detach before destruction; report failures and support staged retries.

    rclpy Humble remove_node() and destroy_node() return None on success,
    not a truthy acknowledgement. Verify executor membership when its public
    get_nodes() is available. Explicit False from an adapter is a failure.
    Callers must retain incomplete progress together with its node and retry it.
    """
    result = progress if progress is not None else NodeDisposal()
    with result._lock:
        if result.complete:
            return result
        result.error = None
        name = label or getattr(node, "get_name", lambda: type(node).__name__)()
        if not result.removed:
            try:
                get_nodes = getattr(executor, "get_nodes", None)
                if get_nodes is None or node in get_nodes():
                    if executor.remove_node(node) is False:
                        raise RuntimeError("remove_node returned False")
                if get_nodes is not None and node in get_nodes():
                    raise RuntimeError("node remains registered with executor")
                result.removed = True
            except Exception as error:
                result.error = f"remove_node: {error}"
                log.warning("failed to remove ROS node %r: %s", name, error)
                return result  # destroying a still-registered handle is unsafe
        try:
            if node.destroy_node() is False:
                raise RuntimeError("destroy_node returned False")
            result.destroyed = True
        except Exception as error:
            result.error = f"destroy_node: {error}"
            log.warning("failed to destroy ROS node %r: %s", name, error)
        return result


@dataclass(eq=False)
class _NodeRequest:
    generation: int
    topic: str
    cancelled: bool = False
    busy: bool = False
    error: str | None = None


class NodeLifecycle:
    """Explicit owner for cancelled starts and incompletely retired ROS nodes.

    Registry updates share the plugin's short-held RLock. ROS calls and worker
    joins run only under a per-instance operation lock, never the registry lock.
    Stop cancels before trying that lock; the in-flight start then unwinds its
    own candidate. No retry thread, sleep or replacement node is hidden here.
    """

    def __init__(self, executor, nodes, lock, *, label):
        self.executor, self.nodes, self.lock, self.label = executor, nodes, lock, label
        self.draining = {}
        self._resources = {}
        self._requests = {}
        self._generations = {}
        self._operations = {}
        self._config_lock = threading.Lock()
        self._configuring = False
        self._closing = False

    @property
    def closing(self):
        with self.lock:
            return self._closing

    def shutdown(self):
        with self.lock:
            self._closing = True
        return self.stop()

    def _operation(self, key):
        with self.lock:
            return self._operations.setdefault(key, threading.RLock())

    def current(self, key):
        with self.lock:
            request = self._requests.get(key)
            return request if request is not None and not request.cancelled else None

    def pending(self):
        with self.lock:
            return [(key, request) for key, request in self._requests.items()
                    if not request.cancelled and not request.error and not request.busy
                    and key not in self.nodes and not self.draining.get(key)]

    def _valid(self, key, request):
        return (request is not None and self._requests.get(key) is request
                and not request.cancelled and not self._closing)

    def _status(self, key):
        resources = [self._resources[id(node)] for node in self.draining.get(key, ())]
        errors = [progress.error for progress, _ in resources if progress.error]
        request = self._requests.get(key)
        if resources or (request and request.cancelled and (request.busy or key in self.nodes)):
            return {"state": "failed" if errors else "stopping", "reason": "cleanup_pending",
                    "error": "; ".join(errors) or None,
                    "cleanup": [{"removed": p.removed, "destroyed": p.destroyed,
                                 "error": p.error} for p, _ in resources]}
        if request and request.error:
            return {"state": "failed", "error": request.error}
        return None

    def status(self, key=None):
        with self.lock:
            if key is not None:
                return self._status(key)
            states = {k: self._status(k) for k in self._requests.keys() | self.draining.keys()}
            states = {k: v for k, v in states.items() if v}
            if not states:
                return None
            failed = next((v for v in states.values() if v['state'] == 'failed'), None)
            return {**(failed or next(iter(states.values()))), "cleanup_instances": states}

    def annotate(self, result, instance_id="", *, scoped_info=False):
        status = self.status(instance_id or None)
        if status:
            return {**result, **status}
        with self.lock:
            request = self._requests.get(instance_id)
            if scoped_info and instance_id and instance_id not in self.nodes and (request is None or request.cancelled):
                return {**result, "state": "idle"}
        return result

    @contextmanager
    def configuration(self):
        with self._config_lock:
            with self.lock:
                self._configuring = True
            try:
                yield
            finally:
                with self.lock:
                    self._configuring = False

    def begin(self, key, topic, *, configuring=False):
        topic = topic or ""
        with self.lock:
            if self._closing:
                return None, {"state": "failed", "reason": "shutting_down"}
            if self._configuring and not configuring:
                return None, {"state": "stopping", "reason": "configuration_in_progress"}
            status = self._status(key)
            # A failed *start* without retained resources is retryable. Cleanup
            # failures and cancelled constructors must be resolved by stop first.
            if status and (self.draining.get(key) or status['state'] == 'stopping'):
                return None, status
            previous = self._requests.get(key)
            if previous and previous.cancelled and key in self.nodes:
                return None, {"state": "stopping", "reason": "cleanup_pending"}
            node_state = getattr(self.nodes.get(key), 'state', None)
            if (previous and not previous.cancelled and not previous.error
                    and previous.topic == topic and node_state not in ('error', 'failed')):
                return previous, None
        if previous and not previous.cancelled:
            stopped = self.stop(key)
            if stopped['state'] != 'idle':
                return None, stopped
        with self.lock:
            if self._requests.get(key) is not previous:
                return None, {"state": "stopping", "reason": "request_changed"}
            generation = self._generations.get(key, 0) + 1
            self._generations[key] = generation
            request = _NodeRequest(generation, topic)
            self._requests[key] = request
            return request, None

    def _retain(self, key, node, retire):
        # Must precede deleting the active record. Identity, not key alone,
        # controls removal so old requests can never erase a replacement.
        if id(node) not in self._resources:
            progress = getattr(node, '_lifecycle_disposal', None)
            if progress is None:
                progress = node._lifecycle_disposal = NodeDisposal()
            self._resources[id(node)] = (progress, retire)
            self.draining.setdefault(key, []).append(node)
        if self.nodes.get(key) is node:
            del self.nodes[key]

    def retire(self, key, node, retire, *, never_registered=False):
        with self._operation(key):
            with self.lock:
                self._retain(key, node, retire)
                progress, retire = self._resources[id(node)]
                if never_registered:
                    progress.removed = True
            try:
                # Never destroy resources until the plugin has stopped their
                # workers. A timed-out worker remains owned for a later stop.
                result = ({"state": "idle"} if getattr(node, "_lifecycle_quiesced", False)
                          else retire() or {"state": "idle"})
                if result.get('state') == 'stopping' or getattr(node, 'worker_alive', False):
                    progress.error = None
                    return {"state": "stopping", "reason": "worker_pending"}
                if result.get('state') not in ('idle', 'stopped'):
                    progress.error = result.get('error') or result.get('message') or 'worker stop incomplete'
                else:
                    node._lifecycle_quiesced = True
                    dispose_node(self.executor, node, label=f"{self.label}/{key}", progress=progress)
            except Exception as error:
                progress.error = f"stop: {error}"
            with self.lock:
                if progress.complete:
                    self.draining[key].remove(node)
                    if not self.draining[key]:
                        del self.draining[key]
                    del self._resources[id(node)]
                    return {"state": "idle"}
                return self._status(key)

    def start(self, key, request, factory, retire):
        operation = self._operation(key)
        if not operation.acquire(blocking=False):
            return self.status(key) or {"state": "starting"}
        node = None
        attempted = False
        keep = False
        try:
            with self.lock:
                if not self._valid(key, request):
                    return self._status(key) or {"state": "idle", "cancelled": True}
                if self.draining.get(key):
                    return self._status(key)
                request.busy = True
                existing = self.nodes.get(key)
            if existing is not None:
                node = existing
                attempted = True
            else:
                node = factory()
                with self.lock:
                    # Retain a constructed candidate before *any* executor call.
                    self._retain(key, node, lambda: retire(node))
                    if not self._valid(key, request):
                        return {"state": "stopping", "cancelled": True}
                attempted = True  # add_node may mutate membership then raise
                if self.executor.add_node(node) is False:
                    raise RuntimeError('executor refused node registration')
                with self.lock:
                    if not self._valid(key, request):
                        return {"state": "stopping", "cancelled": True}
                    self.nodes[key] = node
            result = node.start()
            with self.lock:
                if not self._valid(key, request):
                    return {"state": "stopping", "cancelled": True}
                keep = result.get('state') == 'running' or (not request.topic and result.get('state') == 'idle')
                if not keep:
                    raise RuntimeError(result.get('error') or 'node start incomplete')
                if id(node) in self._resources:
                    self.draining[key].remove(node)
                    if not self.draining[key]:
                        del self.draining[key]
                    del self._resources[id(node)]
                return result
        except Exception as error:
            with self.lock:
                if self._valid(key, request):
                    request.error = str(error)
            return {"state": "failed", "error": str(error)}
        finally:
            if node is not None and not keep:
                self.retire(key, node, lambda: retire(node), never_registered=not attempted)
            with self.lock:
                if request is not None:
                    request.busy = False
            operation.release()

    def stop(self, key=None):
        with self.lock:
            keys = ([key] if key is not None else
                    list(self._requests.keys() | self.nodes.keys() | self.draining.keys()))
            stopped_requests = {item: self._requests.get(item) for item in keys}
            for item in keys:
                request = stopped_requests[item]
                if request:
                    request.cancelled = True
                    request.error = None
                node = self.nodes.get(item)
                if node is not None:
                    # This latch only requests cancellation; blocking stop/retire
                    # is serialized below and must not hold the registry lock.
                    signal = getattr(node, 'request_retire', None) or getattr(node, 'request_stop', None)
                    if signal:
                        signal()
        for item in keys:
            operation = self._operation(item)
            if not operation.acquire(blocking=False):
                continue  # the cancelled start owns cleanup in its finally
            try:
                with self.lock:
                    if self._requests.get(item) is not stopped_requests[item]:
                        continue  # this stop cannot retire a newer request
                    node = self.nodes.get(item)
                    if node is not None:
                        self._retain(item, node, lambda node=node: self.retire_node(node))
                    resources = list(self.draining.get(item, ()))
                for node in resources:
                    with self.lock:
                        resource = self._resources.get(id(node))
                    if resource:
                        self.retire(item, node, resource[1])
            finally:
                operation.release()
        return self.status(key) or {"state": "idle"}

    @staticmethod
    def retire_node(node):
        retire = getattr(node, 'retire', None)
        if retire:
            return retire()
        node.request_stop()
        return node.stop()


__all__ = ["NodeDisposal", "dispose_node", "NodeLifecycle"]
