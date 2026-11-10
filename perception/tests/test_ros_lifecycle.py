"""The cleanup contract uses rclpy's None returns and retained stage progress."""
from types import SimpleNamespace

import pytest

from utils.ros_lifecycle import NodeDisposal, dispose_node


@pytest.mark.parametrize('failure', ['remove_raise', 'remove_false', 'remove_silent',
                                    'destroy_raise', 'destroy_false', None])
def test_disposal_is_ordered_verifiable_and_idempotent(failure):
    order = []
    node = SimpleNamespace(get_name=lambda: 'owned')
    nodes = [node]
    inject = [True]

    def remove(target):
        order.append('remove')
        if inject[0] and failure == 'remove_raise':
            raise RuntimeError('remove failed')
        if inject[0] and failure == 'remove_false':
            return False
        if inject[0] and failure == 'remove_silent':
            return None
        nodes.remove(target)
        return None  # real rclpy remove_node returns None

    def destroy():
        assert node not in nodes
        order.append('destroy')
        if inject[0] and failure == 'destroy_raise':
            raise RuntimeError('destroy failed')
        if inject[0] and failure == 'destroy_false':
            return False
        return None  # real rclpy destroy_node returns None

    node.destroy_node = destroy
    executor = SimpleNamespace(remove_node=remove, get_nodes=lambda: list(nodes))
    progress = NodeDisposal()
    assert dispose_node(executor, node, progress=progress) is progress
    if failure:
        assert not progress.complete and progress.error
        assert ('destroy' not in order) == failure.startswith('remove')
        assert progress.removed == failure.startswith('destroy')
    else:
        assert progress.complete and progress.error is None
    inject[0] = False
    assert dispose_node(executor, node, progress=progress).complete
    assert progress.error is None
    completed_order = list(order)
    assert dispose_node(executor, node, progress=progress).complete
    assert order == completed_order  # neither stage runs again


def test_already_detached_node_is_not_removed_twice():
    calls = []
    node = SimpleNamespace(destroy_node=lambda: calls.append('destroy'))
    executor = SimpleNamespace(get_nodes=lambda: [],
                               remove_node=lambda _: pytest.fail('already detached'))
    assert dispose_node(executor, node).complete
    assert calls == ['destroy']


def test_unverifiable_membership_never_destroys_node():
    def fail():
        raise RuntimeError('cannot inspect executor')
    node = SimpleNamespace(destroy_node=lambda: pytest.fail('membership unknown'))
    result = dispose_node(SimpleNamespace(get_nodes=fail), node)
    assert not result.complete and 'inspect' in result.error


def test_shared_progress_serializes_concurrent_callers():
    import threading
    entered, release = threading.Event(), threading.Event()
    calls, results = [], []
    node = SimpleNamespace(get_name=lambda: 'concurrent')
    nodes = [node]
    def remove(target):
        calls.append('remove')
        entered.set()
        assert release.wait(3)
        nodes.remove(target)
    def destroy():
        assert not nodes
        calls.append('destroy')
    node.destroy_node = destroy
    executor = SimpleNamespace(remove_node=remove, get_nodes=lambda: list(nodes))
    progress = NodeDisposal()
    threads = [threading.Thread(target=lambda: results.append(
        dispose_node(executor, node, progress=progress))) for _ in range(8)]
    try:
        for thread in threads:
            thread.start()
        assert entered.wait(2)
    finally:
        release.set()
        for thread in threads:
            thread.join(3)
    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 8 and all(result is progress for result in results)
    assert progress.complete and calls == ['remove', 'destroy']


def test_remove_exception_after_detach_recovers_by_membership_without_second_remove():
    calls = []
    node = SimpleNamespace(get_name=lambda: 'partial')
    nodes = [node]
    def remove(target):
        calls.append('remove')
        nodes.remove(target)
        raise RuntimeError('guard wake failed after removal')
    node.destroy_node = lambda: calls.append('destroy')
    executor = SimpleNamespace(remove_node=remove, get_nodes=lambda: list(nodes))
    progress = NodeDisposal()
    assert not dispose_node(executor, node, progress=progress).complete
    assert progress.error and calls == ['remove']
    assert dispose_node(executor, node, progress=progress).complete
    assert calls == ['remove', 'destroy']
